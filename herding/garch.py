"""A formula with no traders: the GJR-GARCH(1,1) model, the standard way statisticians describe
volatility that comes in bursts (Bollerslev 1986, J. Econometrics 31:307-327; the GJR term for falls:
Glosten, Jagannathan & Runkle 1993, J. Finance 48:1779-1801).

Each day's return is r_t = mu + e_t, with e_t = s_t * z_t and z_t bell-curve noise (standard normal), and
    s_t^2 = omega + (alpha + gamma * [e_(t-1) < 0]) * e_(t-1)^2 + beta * s_(t-1)^2.
So a big move today makes tomorrow's typical move bigger (alpha), more so after a fall (gamma), and that
fades slowly (beta). Like the herding model it is fed only bell-curve news; unlike it there are no
traders: the bursts are written into the formula. The website uses it to ask whether herding explains
more than a formula does.

Fitted by maximum likelihood (the parameters that make the observed returns most probable).
"""
import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter

from .checks import is_number

NAMES = ("mu", "omega", "alpha", "gamma", "beta")


def _returns(r):
    r = np.asarray(r, dtype=float)
    if r.ndim != 1 or r.size < 100 or not np.all(np.isfinite(r)):
        raise ValueError("need a 1-D list of at least 100 finite returns")
    return r


def variances(r, mu, omega, alpha, gamma, beta):
    """s_t^2 for each day of r (the first day starts from the sample variance of r)."""
    e = _returns(r) - mu
    # with the returns known, s2[t] = omega + c[t-1] + beta s2[t-1] is a linear recursion: lfilter runs
    # it in C (the same numbers as a loop, which test_garch.py checks), fast enough to fit in seconds
    c = (alpha + gamma * (e < 0)) * e ** 2
    drive = np.r_[e.var(), omega + c[:-1]]
    return lfilter([1.0], [1.0, -beta], drive)


def neg_loglik(theta, r):
    """Minus the log-likelihood of returns r under bell-curve z (the quantity fit() makes smallest)."""
    mu, omega, alpha, gamma, beta = theta
    s2 = variances(r, mu, omega, alpha, gamma, beta)
    e = r - mu
    return 0.5 * float(np.sum(np.log(2 * np.pi * s2) + e ** 2 / s2))


def _ok(theta):
    mu, omega, alpha, gamma, beta = theta
    return omega > 0 and alpha >= 0 and alpha + gamma >= 0 and beta >= 0 and alpha + gamma / 2 + beta < 1


def fit(r):
    """The GJR-GARCH(1,1) parameters (dict, NAMES) that fit returns r best."""
    r = _returns(r)
    v = r.var()
    start = np.array([r.mean(), 0.05 * v, 0.05, 0.05, 0.88])
    scale = np.array([r.std(), v, 1, 1, 1])        # optimise numbers of similar size

    def f(x):
        theta = x * scale
        return neg_loglik(theta, r) if _ok(theta) else 1e300

    best = minimize(f, start / scale, method="Nelder-Mead",
                    options=dict(maxiter=6000, xatol=1e-7, fatol=1e-7))
    best = minimize(f, best.x, method="Nelder-Mead", options=dict(maxiter=6000, xatol=1e-9, fatol=1e-9))
    theta = best.x * scale
    if not _ok(theta):
        raise ValueError("the fit did not find valid parameters")
    return dict(zip(NAMES, map(float, theta)))


def simulate(params, days, seed=None, burn=500):
    """`days` daily returns from GJR-GARCH(1,1) with these parameters and bell-curve z (after `burn`
    warm-up days, so the start doesn't matter). The same seed always gives the same returns."""
    if set(params) != set(NAMES) or not all(is_number(params[k]) for k in NAMES):
        raise ValueError(f"params must give the numbers {NAMES}, got {params!r}")
    if not _ok([params[k] for k in NAMES]):
        raise ValueError("params must have omega > 0, alpha >= 0, alpha + gamma >= 0, beta >= 0 and "
                         "alpha + gamma / 2 + beta < 1 (otherwise volatility grows without end)")
    if isinstance(days, bool) or not isinstance(days, (int, np.integer)) or days < 1:
        raise ValueError(f"days must be a whole number >= 1, got {days!r}")
    if isinstance(burn, bool) or not isinstance(burn, (int, np.integer)) or burn < 0:
        raise ValueError(f"burn must be a whole number >= 0, got {burn!r}")
    mu, omega, a, g, b = (params[k] for k in NAMES)
    z = np.random.default_rng(seed).standard_normal(int(days) + int(burn))
    s2 = omega / (1 - a - g / 2 - b)          # start at the long-run variance
    out = np.empty(z.size)
    for t in range(z.size):
        e = np.sqrt(s2) * z[t]
        out[t] = mu + e
        s2 = omega + (a + g * (e < 0)) * e * e + b * s2
    return out[int(burn):]
