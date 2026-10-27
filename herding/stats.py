"""Measurements of daily returns: fat tails, clustering, predictability.

Each function rejects empty, constant, NaN or infinite input with a clear error.
"""
import numpy as np


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
    return np.mean(_standardize(r) ** 4) - 3


def ccdf(r):
    """Sorted |standardized returns| x and the fraction of days with |z| >= x."""
    z = np.sort(np.abs(_standardize(r)))
    return z, 1 - np.arange(len(z)) / len(z)


def acf(r, lags):
    """Autocorrelation of r at each lag (Pearson correlation of r_t and r_t+lag)."""
    z = _standardize(r)
    lags = np.atleast_1d(lags)
    if np.any(lags < 1) or np.any(lags >= len(z) - 1):
        raise ValueError(f"lags must be between 1 and {len(z) - 2}")
    return np.array([np.corrcoef(z[:-k], z[k:])[0, 1] for k in lags])


def tail_share(r, k=4.0):
    """Fraction of days that move more than k standard deviations."""
    return float(np.mean(np.abs(_standardize(r)) > k))


def stylized_facts(r):
    """The fingerprint of a market: tails, predictability, clustering.

    These are the numbers the model is calibrated against the S&P 500.
    """
    a = np.abs(_returns(r))
    return {
        "kurtosis": kurtosis(r),
        "alpha_5pct": hill_alpha(r, 0.05),
        "alpha_1pct": hill_alpha(r, 0.01),
        "share_4sd": tail_share(r, 4.0),
        "ac_r_1": acf(r, 1)[0],
        "ac_r_2": acf(r, 2)[0],
        "ac_r_2to10": float(np.mean(acf(r, range(2, 11)))),   # reversals hiding past day 2
        "ac_abs_1": acf(a, 1)[0],
        "ac_abs_20": acf(a, 20)[0],
        "ac_abs_50": acf(a, 50)[0],
        "ac_abs_100": acf(a, 100)[0],
    }
