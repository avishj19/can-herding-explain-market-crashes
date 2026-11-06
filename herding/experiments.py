"""Running the model many times (used by main.py, make_figures.py and scripts/analysis/).

A "history" is one simulated run with the first BURN days dropped, so the
starting state does not matter. The model is random, so every claim about it is
judged over an ensemble of histories, never one run.
"""
from multiprocessing import Pool

import numpy as np

from . import calibrate, model

BURN = calibrate.BURN


def _one(args):
    p, seed, days, chartists, keep = args
    try:
        o = model.simulate(p, days + BURN, seed=seed, chartists=chartists)
    except model.RunawayError:
        return None
    return {k: o[k][BURN:] for k in keep}


def histories(p, seeds, days, chartists=True, keep=("r",), pool=None):
    """One history per seed, in parallel: a list of dicts with the series in `keep`
    (r = daily returns; lp, lf, z, x = daily closes). A history that ran away is
    None, so callers can count runaways instead of losing them."""
    jobs = [(p, int(s), int(days), chartists, tuple(keep)) for s in seeds]
    if pool is not None:
        return pool.map(_one, jobs)
    with Pool() as own:
        return own.map(_one, jobs)


def returns(p, seeds, days, chartists=True, pool=None):
    """Daily returns of one history per seed; raises if any history ran away."""
    out = histories(p, seeds, days, chartists, ("r",), pool)
    bad = [s for s, h in zip(seeds, out) if h is None]
    if bad:
        raise model.RunawayError(f"histories ran away for seeds {bad}", None)
    return [h["r"] for h in out]


def percentile(value, sims):
    """Where a real value sits among simulated ones, from 0 to 100: the share of
    simulations below it, counting ties as half. 50 = typical; near 0 or 100 = unusual."""
    sims = np.asarray(sims, dtype=float)
    if sims.size == 0:
        raise ValueError("need at least one simulated value")
    return float(100 * (np.mean(sims < value) + 0.5 * np.mean(sims == value)))


def worst_fall(r, k):
    """Largest fall over any k consecutive days, as a positive log return."""
    r = np.asarray(r, dtype=float)
    if isinstance(k, bool) or int(k) != k or not 1 <= k <= len(r):
        raise ValueError(f"k must be a whole number between 1 and {len(r)}, got {k}")
    k = int(k)
    c = np.concatenate([[0.0], np.cumsum(r)])
    return float(-(c[k:] - c[:-k]).min())


def typical(fingerprints, target):
    """Index of the most *typical* history: the one whose mismatch to the target is
    the median, not the smallest (showing the best-fitting run would flatter the
    model). Mismatch = mean squared gap over the measures, each gap in units of
    that measure's spread across the histories."""
    if len(fingerprints) == 0:
        raise ValueError("need at least one history")
    keys = list(target)
    v = np.array([[f[k] for k in keys] for f in fingerprints])
    spread = np.where(v.std(axis=0) > 0, v.std(axis=0), 1.0)
    miss = np.mean(((v - np.array([target[k] for k in keys])) / spread) ** 2, axis=1)
    return int(np.argsort(miss)[(len(miss) - 1) // 2])
