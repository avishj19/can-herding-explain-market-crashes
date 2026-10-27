"""First draft: can herding explain market crashes?

A simplified Lux-Marchesi (1999) agent model: N traders are fundamentalists
(buy below fundamental value, sell above) or chartists (optimists/pessimists
who follow the crowd and the trend, Kirman-style herding). News moves the
fundamental value with plain Gaussian noise. We compare the simulated return
tails to real S&P 500 daily returns and to the inverse cubic law (alpha ~ 3).

Prices are in logs (results do not depend on the price level), every group
keeps a few members (so nobody is trapped), and each run takes its own seed.

Run:  python3 main.py      -> prints tail stats, saves figures/draft.png
Test: python3 -m pytest tests/ -q ; python3 tests/experiments.py
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

rng = np.random.default_rng(1)   # only for the Gaussian comparison in main()

# Parameters after Lux & Marchesi (2000), rewritten in log-price units so the
# model does not depend on the price level (pf0 only sets the starting price).
P = dict(N=500, v1=3.0, v2=2.0, beta=50.0, tc=0.1, gamma=0.1,
         a1=0.6, a2=2.0, a3=0.5, R=0.0004, s=0.75, sigma_news=0.01,
         pf0=10.0, dt=0.02, min_chartists=4, tau=0.2)
U_MAX = 30.0                     # cap on switching exponents (prevents overflow)


def _p(rate):
    """A switching rate per step as a probability in [0, 1]."""
    return min(1.0, max(0.0, rate))


N_STABLE = 250                  # below this the price swings wildly (tested)
MAX_MISPRICING = 1.0            # healthy runs stay below 0.6 (measured); runaways reach 5-16
POSITIVE = ("v1", "v2", "beta", "gamma", "tc", "sigma_news", "tau", "pf0", "dt")
NON_NEGATIVE = ("a1", "a2", "a3", "s", "R")
WHOLE = ("N", "min_chartists")


class ModelRegimeError(RuntimeError):
    """The simulated market left the range where the model is meaningful."""


def check_settings(p):
    """Every setting present, spelled right, a finite number, with a sensible sign."""
    unknown, missing = set(p) - set(P), set(P) - set(p)
    if unknown:
        raise ValueError(f"unknown setting(s) {sorted(unknown)} -- typo? valid: {sorted(P)}")
    if missing:
        raise ValueError(f"missing setting(s) {sorted(missing)}")
    for key, v in p.items():
        if isinstance(v, bool) or not isinstance(v, (int, float, np.number)) or not np.isfinite(v):
            raise ValueError(f"setting {key} must be a finite number, got {v!r}")
        if key in POSITIVE and v <= 0:
            raise ValueError(f"setting {key} must be > 0, got {v}")
        if key in NON_NEGATIVE and v < 0:
            raise ValueError(f"setting {key} must be >= 0, got {v}")
        if key in WHOLE and float(v) != int(v):
            raise ValueError(f"setting {key} must be a whole number, got {v}")


def check_inputs(days, herding, p):
    """Raise ValueError for inputs that would give silently wrong results."""
    check_settings(p)
    if days < 0 or int(days) != days:
        raise ValueError(f"days must be a whole number >= 0, got {days}")
    dt = p["dt"]
    if not dt > 0 or round(1 / dt) < 1 or abs(round(1 / dt) * dt - 1) > 1e-9:
        raise ValueError(f"dt must divide one day exactly (e.g. 0.01, 0.02, 0.05), got {dt}")
    if dt > 1 / (p["beta"] * p["gamma"]) + 1e-12:
        raise ValueError(f"dt={dt} breaks stability: price would overshoot; "
                         f"need dt <= 1/(beta*gamma) = {1 / (p['beta'] * p['gamma']):.3g}")
    m = p["min_chartists"]
    if herding:
        if m < 2 or m % 2:
            raise ValueError(f"min_chartists must be even and >= 2, got {m}")
        if p["N"] < 10 * m:
            raise ValueError(f"N={p['N']} is too small: need N >= {10 * m} "
                             f"so every group can keep its minimum members")
        if p["N"] < N_STABLE:
            warnings.warn(f"N={p['N']} < {N_STABLE}: too few fundamentalists to anchor "
                          f"the price; expect unrealistically wild swings", UserWarning)
    elif p["N"] < 1:
        raise ValueError(f"N must be >= 1, got {p['N']}")


def initial_counts(N, herding, min_chartists):
    """Starting (fundamentalists, optimists, pessimists): about 10% chartists,
    split evenly, never below the floors."""
    if not herding:
        return N, 0, 0
    each = max(N // 20, min_chartists // 2)
    return N - 2 * each, each, each


def simulate(days, herding=True, p=P, seed=None, paths=False,
             max_mispricing=MAX_MISPRICING):
    """Simulate the market for `days` trading days.

    Returns daily log returns and the chartist fraction at each day's close;
    with paths=True also the daily log price and log fundamental value.
    All prices are logs, so only *relative* mispricing matters.
    Raises ModelRegimeError if |log price - log value| ever exceeds
    max_mispricing (pass None to explore unrealistic settings anyway).
    """
    check_inputs(days, herding, p)
    g = np.random.default_rng(seed)
    N, dt = p["N"], p["dt"]
    steps = int(round(1 / dt))
    m2 = p["min_chartists"] // 2 if herding else 0   # floor per chartist type
    mF = p["min_chartists"]                          # floor for fundamentalists
    nf, npl, nmi = initial_counts(N, herding, p["min_chartists"])
    lp = lf = np.log(p["pf0"])                        # log price, log fundamental
    trend = 0.0                                       # d(log price)/dt per day
    out_lp, out_lf, frac = [lp], [lf], []
    for day in range(days):
        for _ in range(steps):
            if herding:
                nc = npl + nmi
                # optimist <-> pessimist: herding (a1 * opinion) + trend (a2)
                U1 = np.clip(p["a1"] * (npl - nmi) / nc + p["a2"] * trend / p["v1"],
                             -U_MAX, U_MAX)
                rate = p["v1"] * nc / N * dt
                up = min(g.binomial(nmi, _p(rate * np.exp(U1))), nmi - m2)
                down = min(g.binomial(npl, _p(rate * np.exp(-U1))), npl - m2)
                npl, nmi = npl + up - down, nmi - up + down
                # chartist <-> fundamentalist: compare expected profits
                mis = abs(lf - lp)                            # relative mispricing
                yld = p["R"] * np.exp(min(lf - lp, U_MAX))                # dividend yield
                U21 = np.clip(p["a3"] * (yld + trend / p["v2"] - p["R"] - p["s"] * mis),
                              -U_MAX, U_MAX)
                U22 = np.clip(p["a3"] * (p["R"] - yld - trend / p["v2"] - p["s"] * mis),
                              -U_MAX, U_MAX)
                r2 = p["v2"] * dt / N
                # floors: someone must remain in every group, or nobody could
                # ever be recruited back into it (Kirman-style switching)
                f2o = min(g.binomial(nf, _p(r2 * npl * np.exp(U21))), nf - mF)
                f2p = min(g.binomial(nf - f2o, _p(r2 * nmi * np.exp(U22))), nf - f2o - mF)
                o2f = min(g.binomial(npl, _p(r2 * nf * np.exp(-U21))), npl - m2)
                p2f = min(g.binomial(nmi, _p(r2 * nf * np.exp(-U22))), nmi - m2)
                npl += f2o - o2f
                nmi += f2p - p2f
                nf += o2f + p2f - f2o - f2p
            # market maker: log price moves with excess demand
            ed = (npl - nmi) * p["tc"] + nf * p["gamma"] * (lf - lp)
            dlp = p["beta"] * ed / N * dt
            lp += dlp
            trend += (dlp / dt - trend) * min(1.0, dt / p["tau"])  # smoothed
            # Gaussian news moves the fundamental value
            lf += p["sigma_news"] * np.sqrt(dt) * g.standard_normal()
        if not np.isfinite(lp):
            raise FloatingPointError("simulation diverged; try a smaller dt")
        if max_mispricing is not None and abs(lp - lf) > max_mispricing:
            raise ModelRegimeError(
                f"day {day}: price is {np.exp(abs(lp - lf)):.1f}x away from its value "
                f"with {(npl + nmi) / N:.0%} chartists. These settings are outside the "
                f"model's realistic range (known causes: N < {N_STABLE}, s < 0.5, v2 < 1). "
                f"Pass max_mispricing=None to run anyway.")
        out_lp.append(lp)
        out_lf.append(lf)
        frac.append((npl + nmi) / N)
    r, frac = np.diff(out_lp), np.array(frac)
    return (r, frac, np.array(out_lp), np.array(out_lf)) if paths else (r, frac)


def _returns(r):
    """Returns as a 1-D float array; reject empty or NaN/inf input."""
    r = np.asarray(r, dtype=float).ravel()
    if r.size == 0:
        raise ValueError("returns are empty")
    if not np.all(np.isfinite(r)):
        raise ValueError("returns contain NaN or infinity")
    return r


def _standardize(r):
    r = _returns(r)
    if r.std() == 0:
        raise ValueError("returns are constant: no spread to measure")
    return (r - r.mean()) / r.std()


def hill_alpha(r, tail=0.05):
    """Hill estimator of the tail exponent of |returns| (top `tail` fraction).

    Zero returns (flat days) are dropped: they carry no tail information.
    """
    if not 0 < tail < 1:
        raise ValueError(f"tail must be between 0 and 1, got {tail}")
    a = np.sort(np.abs(_returns(r)))[::-1]
    a = a[a > 0]
    k = max(int(len(a) * tail), 10)
    if len(a) <= k:
        raise ValueError(f"need more than {k} nonzero returns, got {len(a)}")
    if a[0] == a[k]:
        raise ValueError("returns are constant in the tail: no exponent to measure")
    return k / np.sum(np.log(a[:k] / a[k]))


def kurtosis(r):
    """Excess kurtosis: 0 for a bell curve, > 0 for fat tails."""
    return np.mean(_standardize(r) ** 4) - 3


def ccdf(r):
    """Sorted |standardized returns| x and the fraction of days with |z| >= x."""
    z = np.sort(np.abs(_standardize(r)))
    return z, 1 - np.arange(len(z)) / len(z)


def load_spx(path="data/spx_daily.csv"):
    """Daily S&P 500 closes, checked: sorted, unique dates, positive prices."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download it with: python3 -c \"import yfinance as yf; "
            f"yf.download('^GSPC', start='1950-01-01')[['Close']].to_csv('{path}')\"")
    spx = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    close = spx["Close"]
    if close.isna().any() or not (close > 0).all():
        raise ValueError(f"{path}: every Close price must be positive (found NaN or <= 0)")
    if not spx.index.is_monotonic_increasing or spx.index.duplicated().any():
        raise ValueError(f"{path}: dates must be sorted and unique")
    return spx


def main():
    spx = load_spx("data/spx_daily.csv")
    real = np.diff(np.log(spx["Close"].values))
    if pd.Timestamp("1987-10-19") not in spx.index:
        raise ValueError("data/spx_daily.csv has no 1987-10-19 (Black Monday); re-download it")
    bm = real[spx.index[1:] == "1987-10-19"][0]
    print(f"S&P 500: {len(real)} daily returns, {spx.index[0].year}-{spx.index[-1].year}")
    print(f"Black Monday 1987: {np.expm1(bm):.1%} (log return {bm:.3f}) = "
          f"{abs(bm) / real.std():.1f} standard deviations")

    days, burn = 8000, 200                     # drop the start-up transient
    herd, frac = simulate(days + burn, herding=True, seed=1)
    base, _ = simulate(days + burn, herding=False, seed=2)
    herd, frac, base = herd[burn:], frac[burn:], base[burn:]

    print(f"\n{'series':<28}{'kurtosis':>10}{'tail alpha':>12}")
    for name, r in [("S&P 500", real), ("model, herding", herd),
                    ("model, no chartists", base),
                    ("Gaussian", rng.standard_normal(len(real)))]:
        print(f"{name:<28}{kurtosis(r):>10.2f}{hill_alpha(r):>12.2f}")
    print("Check: herding alpha should be ~3 (inverse cubic law); no-chartist "
          "kurtosis should be ~0.")

    Path("figures").mkdir(exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    n = min(len(real), days)
    for a, r, title in [(ax[0, 0], real[-n:], f"S&P 500 (last {n} days)"),
                        (ax[0, 1], herd, "Model with herding")]:
        a.plot(r / r.std(), lw=0.4, color="#2a6fdb")
        a.set_title(title); a.set_ylabel("return / std"); a.set_ylim(-12, 12)
    a = ax[1, 0]
    for r, lab, c in [(real, "S&P 500", "black"), (herd, "herding", "#d94f30"),
                      (base, "no chartists", "#2a6fdb")]:
        z, cc = ccdf(r)
        a.loglog(z, cc, lw=1.5, label=lab, color=c)
    g = np.linspace(1, 8, 100)
    from scipy.stats import norm
    a.loglog(g, 2 * norm.sf(g), "--", color="gray", label="Gaussian")
    a.loglog(g, 0.5 * g ** -3.0, ":", color="green", label="slope -3 (inverse cubic)")
    a.set_ylim(1e-5, 1); a.set_xlim(0.5, 30)
    a.set_xlabel("|return| / std"); a.set_ylabel("P(|r| > x)")
    a.set_title("Tail of the return distribution"); a.legend(fontsize=8)
    a = ax[1, 1]
    a.plot(frac, lw=0.6, color="#7a3fb0")
    a.set_title("Fraction of chartists (herding model)"); a.set_xlabel("day")
    fig.tight_layout()
    fig.savefig("figures/draft.png", dpi=120)
    print("\nSaved figures/draft.png")


if __name__ == "__main__":
    main()
