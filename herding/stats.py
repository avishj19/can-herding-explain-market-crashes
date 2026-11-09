"""Measurements of daily returns: fat tails, clustering, predictability.

Each function rejects empty, constant, NaN or infinite input with a clear error.
"""
import numpy as np


def _returns(r):
    """Returns as a 1-D float array; reject empty, NaN/inf or not 1-D input."""
    r = np.asarray(r, dtype=float)
    if r.ndim != 1:
        raise ValueError(f"returns must be one list of numbers (1-D), got shape {r.shape}")
    if r.size == 0:
        raise ValueError("returns are empty")
    if not np.all(np.isfinite(r)):
        raise ValueError("returns contain NaN or infinity")
    return r


def _varying(r):
    """_returns(r), and reject a constant series. Decided from the values themselves
    (largest == smallest), not from r.std() == 0: for a constant like 0.0123 the
    computed mean is off in the last digit, so std() is about 1e-18, not 0."""
    r = _returns(r)
    if r.max() == r.min():
        raise ValueError("returns are constant: no spread to measure")
    return r


def standardize(r):
    """Returns minus their mean, in units of their standard deviation."""
    r = _varying(r)
    sd = r.std()
    if not np.isfinite(sd):
        raise ValueError("returns are too large to measure their spread (overflow)")
    return (r - r.mean()) / sd


def hill_alpha(r, tail=0.05):
    """Hill estimator of the tail exponent of |returns| (top `tail` fraction).

    Zero returns (flat days) are dropped: they carry no tail information.
    Smaller alpha = fatter tail; real markets ~3 (inverse cubic law).
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
    return np.mean(standardize(r) ** 4) - 3


def ccdf(r):
    """Sorted |standardized returns| x and the fraction of days with |z| >= x."""
    z = np.sort(np.abs(standardize(r)))
    return z, 1 - np.arange(len(z)) / len(z)


def acf(r, lags):
    """Autocorrelation of r at each lag (Pearson correlation of r_t and r_t+lag)."""
    z = standardize(r)
    lags = np.atleast_1d(lags)
    if lags.dtype == bool or lags.size == 0 or not np.all(lags == np.round(lags)):
        raise ValueError(f"lags must be whole numbers, got {lags}")
    lags = lags.astype(int)
    if np.any(lags < 1) or np.any(lags >= len(z) - 1):
        raise ValueError(f"lags must be between 1 and {len(z) - 2}")
    return np.array([np.corrcoef(z[:-k], z[k:])[0, 1] for k in lags])


def aggregate(r, tau):
    """Returns over tau days: sums of tau consecutive daily log returns
    (a leftover partial block at the end is dropped)."""
    r = _returns(r)
    if int(tau) != tau or tau < 1 or tau > len(r):
        raise ValueError(f"tau must be a whole number from 1 to {len(r)}, got {tau}")
    tau = int(tau)
    n = len(r) // tau
    return r[:n * tau].reshape(n, tau).sum(axis=1)


def tail_fit(r, top=0.30):
    """The straight line behind alpha_regression: (alpha, c, x_min, x_max), where the
    share of days with a bigger move is about c * x**-alpha for x_min <= x <= x_max
    (moves in standard deviations; the window is the largest `top` share of days)."""
    if not 0 < top < 1:
        raise ValueError(f"top must be between 0 and 1, got {top}")
    z, cc = ccdf(r)
    k = int(len(z) * (1 - top))
    z, cc = z[k:], cc[k:]
    keep = z > 0
    if keep.sum() < 10 or np.ptp(np.log(z[keep])) == 0:
        raise ValueError("need at least 10 distinct nonzero moves in the fitted window")
    slope, icept = np.polyfit(np.log(z[keep]), np.log(cc[keep]), 1)
    return float(-slope), float(np.exp(icept)), float(z[keep][0]), float(z[-1])


def alpha_regression(r, top=0.30):
    """Tail exponent the way Lux & Marchesi (1999, Fig 2) measure it: the slope of
    log(share of days with a bigger move) against log(move size), fitted over the
    largest `top` share of days. Unlike hill_alpha, this window also covers the
    bell-shaped middle of the distribution, so the two estimates differ."""
    return tail_fit(r, top)[0]


def dfa(x, scales=(16, 32, 64, 128, 256, 512, 1024, 2048)):
    """Detrended fluctuation analysis (Peng et al. 1994), as in Lux & Marchesi (1999,
    Fig 3): integrate x, cut it into windows of n days, remove a straight-line trend
    in each, and measure the leftover spread F(n). Returns (scales, F)."""
    x = _varying(x)
    scales = np.asarray(scales, dtype=int)
    if scales.size == 0:
        raise ValueError("need at least one window size")
    if np.any(scales < 4) or np.any(scales > len(x) // 4):
        raise ValueError(f"scales must be between 4 and {len(x) // 4} (a quarter of the data)")
    y = np.cumsum(x - x.mean())
    F = []
    for n in scales:
        seg = y[:len(y) // n * n].reshape(-1, n)
        t = np.arange(n)
        slope, icept = np.polyfit(t, seg.T, 1)
        F.append(np.sqrt(np.mean((seg - (np.outer(slope, t) + icept[:, None])) ** 2)))
    return scales, np.array(F)


def hurst(x, scales=(16, 32, 64, 128, 256, 512, 1024, 2048)):
    """Self-similarity exponent H: the slope of log F(n) against log n from dfa().
    H = 0.5 means no memory; H > 0.5 means memory (persistence); H < 0.5 means
    reversals. Lux & Marchesi (1999): returns 0.48, absolute returns 0.85."""
    if len(np.unique(scales)) < 2:
        raise ValueError(f"need at least 2 different window sizes to fit a slope, got {scales}")
    s, F = dfa(x, scales)
    return float(np.polyfit(np.log(s), np.log(F), 1)[0])


def dfa_scales(n, smallest=16, count=12):
    """Window sizes for dfa(): `count` sizes spaced evenly on a log scale, from
    `smallest` days up to a quarter of the data (n days)."""
    if n // 4 < smallest:
        raise ValueError(f"need at least {4 * smallest} values for DFA, got {n}")
    return tuple(int(s) for s in np.unique(np.geomspace(smallest, n // 4, count).astype(int)))


FINGERPRINT = {
    "sd": "typical daily move (standard deviation of returns)",
    "kurtosis": "excess kurtosis: 0 for a bell curve, more = fatter tails",
    "alpha_hill": "tail exponent, Hill estimate from the largest 5% of moves (S&P ~3)",
    "alpha_reg": "tail exponent the paper's way: log-log fit to the largest 30% (paper 2.64)",
    "ac_r_1": "correlation of one day's return with the next (about 0 = unpredictable)",
    "ac_abs_1": "correlation of one day's move size with the next (clustering)",
    "ac_abs_50": "the same 50 days apart (long volatility memory)",
    "H": "DFA exponent of returns: 0.5 = no memory (paper 0.48)",
    "H_abs": "DFA exponent of move sizes |r|: above 0.5 = memory (paper 0.85)",
    "kurt_5": "excess kurtosis of 5-day returns",
    "kurt_25": "excess kurtosis of 25-day returns (paper: back to a bell curve)",
}


def fingerprint(r):
    """The numbers that describe a market's returns, as a dict (see FINGERPRINT).
    Needs at least 25 * 20 = 500 days so every measure is defined."""
    r = _returns(r)
    if len(r) < 500:
        raise ValueError(f"need at least 500 days for a fingerprint, got {len(r)}")
    a = np.abs(r)
    scales = dfa_scales(len(r))
    return {
        "sd": float(r.std()),
        "kurtosis": float(kurtosis(r)),
        "alpha_hill": float(hill_alpha(r, 0.05)),
        "alpha_reg": alpha_regression(r),
        "ac_r_1": float(acf(r, 1)[0]),
        "ac_abs_1": float(acf(a, 1)[0]),
        "ac_abs_50": float(acf(a, 50)[0]),
        "H": hurst(r, scales),
        "H_abs": hurst(a, scales),
        "kurt_5": float(kurtosis(aggregate(r, 5))),
        "kurt_25": float(kurtosis(aggregate(r, 25))),
    }
