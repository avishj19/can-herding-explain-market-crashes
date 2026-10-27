"""Lux-Marchesi (1999) agent-based market model, with guardrails.

N traders are fundamentalists (buy below the fundamental value, sell above) or
chartists: optimists (buy) and pessimists (sell) who follow the crowd and the
trend, Kirman-style. Traders switch groups by comparing profits, and every
switch is a recruitment: the more members a group has, the faster it grows.
News moves the fundamental value with plain Gaussian noise.

Prices are logs, so only *relative* mispricing matters. The inner loop runs
under numba when it is installed (about 20x faster) and as plain Python
otherwise; both use the same random numbers and give identical results.
"""
import os
import warnings

import numpy as np

try:
    import numba
except ImportError:                       # e.g. a web host without numba
    numba = None

# Parameters after Lux & Marchesi (2000), rewritten in log-price units so the
# model does not depend on the price level (pf0 only sets the starting price).
P = dict(N=500, v1=3.0, v2=2.0, beta=50.0, tc=0.1, gamma=0.1,
         a1=0.6, a2=2.0, a3=0.5, R=0.0004, s=0.75, sigma_news=0.01,
         pf0=10.0, dt=0.02, min_chartists=4, tau=0.2)

U_MAX = 30.0           # cap on switching exponents (prevents overflow)
N_STABLE = 250         # below this the price swings wildly (tested)
MAX_MISPRICING = 1.0   # healthy runs stay below 0.6 (measured); runaways reach 5-16
POSITIVE = ("v1", "v2", "beta", "gamma", "tc", "sigma_news", "tau", "pf0", "dt")
NON_NEGATIVE = ("a1", "a2", "a3", "s", "R")
WHOLE = ("N", "min_chartists")


class ModelRegimeError(RuntimeError):
    """The simulated market left the range where the model is meaningful."""


# ------------------------------------------------------------------ guardrails

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
    if round(1 / dt) < 1 or abs(round(1 / dt) * dt - 1) > 1e-9:
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


# ------------------------------------------------------------------ the model

def _prob(rate):
    """A switching rate per step as a probability in [0, 1]."""
    return min(1.0, max(0.0, rate))


def _clip(u):
    return min(U_MAX, max(-U_MAX, u))


def _run_impl(g, days, steps, herding, N, nf, npl, nmi, m2, mF, lp, v1, v2, beta,
              tc, gamma, a1, a2, a3, R, s, sigma, tau, max_mis):
    """The simulation loop. Returns (log price, log value, chartist fraction,
    status, day): status 0 = ok, 1 = diverged (NaN), 2 = left realistic regime."""
    dt = 1.0 / steps
    lf = lp
    trend = 0.0                                # smoothed d(log price)/dt per day
    out_lp = np.empty(days + 1)
    out_lf = np.empty(days + 1)
    frac = np.empty(days)
    out_lp[0] = lp
    out_lf[0] = lf
    for day in range(days):
        for _ in range(steps):
            if herding:
                nc = npl + nmi
                # optimist <-> pessimist: herding (a1 * opinion) + trend (a2)
                U1 = _clip(a1 * (npl - nmi) / nc + a2 * trend / v1)
                rate = v1 * nc / N * dt
                up = min(g.binomial(nmi, _prob(rate * np.exp(U1))), nmi - m2)
                down = min(g.binomial(npl, _prob(rate * np.exp(-U1))), npl - m2)
                npl, nmi = npl + up - down, nmi - up + down
                # chartist <-> fundamentalist: compare expected profits
                mis = abs(lf - lp)                       # relative mispricing
                yld = R * np.exp(min(lf - lp, U_MAX))    # dividend yield
                U21 = _clip(a3 * (yld + trend / v2 - R - s * mis))
                U22 = _clip(a3 * (R - yld - trend / v2 - s * mis))
                r2 = v2 * dt / N
                # floors: someone must remain in every group, or nobody could
                # ever be recruited back into it (Kirman-style switching)
                f2o = min(g.binomial(nf, _prob(r2 * npl * np.exp(U21))), nf - mF)
                f2p = min(g.binomial(nf - f2o, _prob(r2 * nmi * np.exp(U22))), nf - f2o - mF)
                o2f = min(g.binomial(npl, _prob(r2 * nf * np.exp(-U21))), npl - m2)
                p2f = min(g.binomial(nmi, _prob(r2 * nf * np.exp(-U22))), nmi - m2)
                npl += f2o - o2f
                nmi += f2p - p2f
                nf += o2f + p2f - f2o - f2p
            # market maker: log price moves with excess demand
            ed = (npl - nmi) * tc + nf * gamma * (lf - lp)
            dlp = beta * ed / N * dt
            lp += dlp
            trend += (dlp / dt - trend) * min(1.0, dt / tau)   # smoothed
            # Gaussian news moves the fundamental value
            lf += sigma * np.sqrt(dt) * g.standard_normal()
        out_lp[day + 1] = lp
        out_lf[day + 1] = lf
        frac[day] = (npl + nmi) / N
        if not np.isfinite(lp):
            return out_lp, out_lf, frac, 1, day
        if max_mis > 0 and abs(lp - lf) > max_mis:
            return out_lp, out_lf, frac, 2, day
    return out_lp, out_lf, frac, 0, days


USE_JIT = numba is not None and os.environ.get("MODEL_NO_JIT") != "1"
if USE_JIT:
    _prob = numba.njit(cache=True)(_prob)
    _clip = numba.njit(cache=True)(_clip)
    _run = numba.njit(cache=True)(_run_impl)
else:
    _run = _run_impl


def simulate(days, herding=True, p=P, seed=None, paths=False,
             max_mispricing=MAX_MISPRICING):
    """Simulate the market for `days` trading days.

    Returns daily log returns and the chartist fraction at each day's close;
    with paths=True also the daily log price and log fundamental value.
    Raises ModelRegimeError if |log price - log value| ever exceeds
    max_mispricing (pass None to explore unrealistic settings anyway).
    """
    check_inputs(days, herding, p)
    N, steps = int(p["N"]), int(round(1 / p["dt"]))
    m = int(p["min_chartists"])
    nf, npl, nmi = initial_counts(N, herding, m)
    lp, lf, frac, status, day = _run(
        np.random.default_rng(seed), int(days), steps, bool(herding), N, nf, npl, nmi,
        m // 2 if herding else 0, m, float(np.log(p["pf0"])),
        *(float(p[k]) for k in ("v1", "v2", "beta", "tc", "gamma", "a1", "a2", "a3",
                                "R", "s", "sigma_news", "tau")),
        -1.0 if max_mispricing is None else float(max_mispricing))
    if status == 1:
        raise FloatingPointError("simulation diverged; try a smaller dt")
    if status == 2:
        gap = abs(lp[day + 1] - lf[day + 1])
        raise ModelRegimeError(
            f"day {day}: price is {np.exp(gap):.1f}x away from its value "
            f"with {frac[day]:.0%} chartists. These settings are outside the "
            f"model's realistic range (known causes: N < {N_STABLE}, s < 0.5, v2 < 1). "
            f"Pass max_mispricing=None to run anyway.")
    r = np.diff(lp)
    return (r, frac, lp, lf) if paths else (r, frac)
