"""Shared helpers for running the model many times (used by main.py,
make_figures.py, calibrate.py and the analysis scripts).

A "history" is one simulated run with the start-up transient dropped. The model
has randomness, so results are judged over an ensemble of histories.
"""
from multiprocessing import Pool

import numpy as np

from . import model

DAYS, BURN = 19300, 300          # same length as the S&P 500 sample; drop start-up
US_BREAKERS = (0.07, 0.13, 0.20)  # US market-wide circuit breakers since 2013


def history(p=model.P, seed=None, days=DAYS, herding=True, burn=BURN, **kw):
    """Daily log returns of one simulated history, after the burn-in."""
    return model.simulate(days + burn, herding, p, seed=seed, **kw)[0][burn:]


def _history(args):
    p, seed, days, herding = args
    return history(p, seed, days, herding)


def ensemble(p, seeds, days=DAYS, herding=True, pool=None):
    """One history per seed, run in parallel. Pass an open Pool to reuse it."""
    jobs = [(p, s, days, herding) for s in seeds]
    if pool is not None:
        return pool.map(_history, jobs)
    with Pool() as own:
        return own.map(_history, jobs)


def worst_fall(r, k):
    """Largest fall over any k consecutive days, as a positive log return."""
    c = np.concatenate([[0.0], np.cumsum(r)])
    return float(-(c[k:] - c[:-k]).min())


def scaled_breakers(model_std, market_std, halt_minutes=15):
    """US breaker rules rescaled to the model's volatility, so '7%' means the
    same number of typical days in the model as in the real market."""
    scale = model_std / market_std
    return dict(levels=tuple(x * scale for x in US_BREAKERS), halt_minutes=halt_minutes), scale
