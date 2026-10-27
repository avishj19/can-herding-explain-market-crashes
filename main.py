"""First draft: can herding explain market crashes?

A simplified Lux-Marchesi (1999) agent model: N traders are fundamentalists
(buy below fundamental value, sell above) or chartists (optimists/pessimists
who follow the crowd and the trend, Kirman-style herding). News moves the
fundamental value with plain Gaussian noise. We compare the simulated return
tails to real S&P 500 daily returns and to the inverse cubic law (alpha ~ 3).

Run:  python3 main.py      -> prints tail stats, saves figures/draft.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

rng = np.random.default_rng(1)

# Parameters loosely after Lux & Marchesi (2000), parameter set I
P = dict(N=500, v1=3.0, v2=2.0, beta=6.0, tc=0.1, gamma=0.01,
         a1=0.6, a2=0.2, a3=0.5, R=0.0004, r=0.004, s=0.75, sigma_news=0.005,
         pf0=10.0, dt=0.02, min_chartists=4)


def simulate(days, herding=True, p=P):
    """Return daily log returns and the chartist fraction over time."""
    N, dt = p["N"], p["dt"]
    steps_per_day = int(round(1 / dt))
    nf = N // 2 if herding else N           # fundamentalists
    npl = nmi = (N - nf) // 2               # optimist / pessimist chartists
    price = pf = p["pf0"]
    dpdt = 0.0
    log_close, frac_c = [np.log(price)], []
    for day in range(days):
        for _ in range(steps_per_day):
            nc = npl + nmi
            if herding and nc > 0:
                x = (npl - nmi) / nc                      # chartist opinion
                trend = dpdt / p["v1"]
                # optimist <-> pessimist switching (herding + trend)
                U1 = p["a1"] * x + p["a2"] * trend
                rate = p["v1"] * nc / N
                up = rng.binomial(nmi, min(1, rate * np.exp(U1) * dt))
                down = rng.binomial(npl, min(1, rate * np.exp(-U1) * dt))
                npl, nmi = npl + up - down, nmi - up + down
                # chartist <-> fundamentalist switching (compare profits)
                dev = p["s"] * abs((pf - price) / price)
                gain = (p["r"] + dpdt / p["v2"]) / price
                U21 = p["a3"] * (gain - p["R"] - dev)        # optimist vs fund.
                U22 = p["a3"] * (p["R"] - gain - dev)        # pessimist vs fund.
                rate2 = p["v2"] * dt
                f2o = rng.binomial(nf, min(1, rate2 * npl / N * np.exp(U21)))
                f2p = rng.binomial(nf - f2o, min(1, rate2 * nmi / N * np.exp(U22)))
                o2f = rng.binomial(npl, min(1, rate2 * nf / N * np.exp(-U21)))
                p2f = rng.binomial(nmi, min(1, rate2 * nf / N * np.exp(-U22)))
                # keep a few chartists so herding can never die out forever
                if npl - o2f + f2o < p["min_chartists"] // 2: o2f = 0
                if nmi - p2f + f2p < p["min_chartists"] // 2: p2f = 0
                npl += f2o - o2f
                nmi += f2p - p2f
                nf += o2f + p2f - f2o - f2p
            # excess demand -> price change (market maker)
            ed = (npl - nmi) * p["tc"] + nf * p["gamma"] * (pf - price)
            dp = p["beta"] * ed / N * dt * price
            dpdt = 0.9 * dpdt + 0.1 * dp / dt              # smoothed trend
            price = max(price + dp, 1e-3)
            # Gaussian news hits the fundamental value
            pf *= np.exp(p["sigma_news"] * np.sqrt(dt) * rng.standard_normal())
        log_close.append(np.log(price))
        frac_c.append((npl + nmi) / N)
    return np.diff(log_close), np.array(frac_c)


def hill_alpha(r, tail=0.05):
    """Hill estimator of the tail exponent of |returns| (top `tail` fraction)."""
    a = np.sort(np.abs(r))[::-1]
    k = max(int(len(a) * tail), 10)
    return k / np.sum(np.log(a[:k] / a[k]))


def kurtosis(r):
    z = (r - r.mean()) / r.std()
    return np.mean(z ** 4) - 3


def ccdf(r):
    z = np.sort(np.abs(r - r.mean()) / r.std())
    return z, 1 - np.arange(len(z)) / len(z)


def main():
    spx = pd.read_csv("data/spx_daily.csv", parse_dates=["Date"], index_col="Date")
    real = np.diff(np.log(spx["Close"].values))
    bm = real[spx.index[1:] == "1987-10-19"][0]
    print(f"S&P 500: {len(real)} daily returns, {spx.index[0].year}-{spx.index[-1].year}")
    print(f"Black Monday 1987: {bm:.1%} = {abs(bm) / real.std():.1f} standard deviations")

    days = 8000
    herd, frac = simulate(days, herding=True)
    base, _ = simulate(days, herding=False)

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
