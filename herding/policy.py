"""Policy experiments (an extension, not part of the paper; SPEC.md "Policy experiments").

1. Circuit breakers: do the US rules (pause at -7% and -13%, close at -20%) make
   crashes rarer, or only spread them out? Each seed is run twice, without and with
   breakers, so the two runs share the same randomness until the first halt.
2. A policy surprise (like an unexpected rate hike): the fundamental value drops
   once by 10%. Does the price fall by 10%, or does the crowd make it overshoot?

The breaker levels are rescaled by the model's volatility relative to the
S&P 500's, so "7%" means the same number of typical days in both.
"""
import numpy as np

from . import model
from .experiments import BURN, worst_fall


def scaled_breakers(model_sd, market_sd, rules=model.US_BREAKERS):
    """The breaker rules with every level multiplied by model_sd / market_sd."""
    if not (model_sd > 0 and market_sd > 0):
        raise ValueError(f"both volatilities must be above 0, got {model_sd}, {market_sd}")
    scale = model_sd / market_sd
    levels = tuple(x * scale for x in rules["levels"])
    if not levels[-1] < 1:
        raise ValueError(f"the model is {scale:.1f}x as volatile as the market: "
                         f"the scaled levels {levels} reach 100%")
    return dict(rules, levels=levels), scale       # every other rule is kept as it is


def crash_measures(r, crash_level):
    """How bad the crashes of one history are: the worst 1-, 5- and 20-day falls
    (positive log returns), and the number of days that fell more than crash_level."""
    return dict(worst_1d=worst_fall(r, 1), worst_5d=worst_fall(r, 5),
                worst_20d=worst_fall(r, 20),
                crash_days=int(np.sum(np.asarray(r) <= np.log(1 - crash_level))))


EVENT_DAYS = (1, 5, 20)            # horizons of the first-halt comparison


def _breaker_pair(args):
    p, seed, days, rules = args
    try:
        off = model.simulate(p, days + BURN, seed=seed)
        on = model.simulate(p, days + BURN, seed=seed, breakers=rules)
    except model.RunawayError:
        return None
    level = rules["levels"][0]
    out = dict(off=crash_measures(off["r"][BURN:], level), on=crash_measures(on["r"][BURN:], level))
    out["on"]["halt_days"] = int(np.sum(on["halts"][BURN:] > 0))
    out["on"]["closed_days"] = int(np.sum(on["halts"][BURN:] == 3))
    # The two runs use identical random numbers up to the first halt, so that day is
    # a clean comparison: how far did the price fall from the previous close
    # within 1, 5 and 20 days, with and without the breaker?
    # (The first halt of the whole run, warm-up included: after any halt the runs
    # differ. It must leave max(EVENT_DAYS) days to measure.)
    fired = np.flatnonzero(on["halts"] > 0)
    out["event"] = None
    if fired.size and fired[0] + max(EVENT_DAYS) <= days + BURN:
        d = int(fired[0])
        out["event"] = {side: {k: float(run["lp"][d + k] - run["lp"][d]) for k in EVENT_DAYS}
                        for side, run in (("off", off), ("on", on))}
    return out


def breaker_experiment(pool, p, seeds, days, rules):
    """Paired runs without and with breakers, one pair per seed. Returns
    (whole, events, runaways):
        whole   {"off"|"on": {measure: array}}: crash_measures of every whole history
        events  {"off"|"on": {k: array}}: at each history's first halt, the log change
                from the previous close after k days (EVENT_DAYS), for histories with a halt
        runaways  number of seeds whose history ran away (left out)"""
    rows = pool.map(_breaker_pair, [(p, int(s), int(days), rules) for s in seeds])
    ok = [r for r in rows if r is not None]
    if not ok:
        return None, None, len(rows)
    whole = {side: {k: np.array([r[side][k] for r in ok]) for k in ok[0][side]}
             for side in ("off", "on")}
    ev = [r["event"] for r in ok if r["event"] is not None]
    events = {side: {k: np.array([e[side][k] for e in ev]) for k in EVENT_DAYS}
              for side in ("off", "on")}
    return whole, events, len(rows) - len(ok)


def _surprise_one(args):
    p, seed, size, horizon = args
    day = BURN + int(np.random.default_rng(seed).integers(0, 5000))   # when the surprise hits
    try:
        o = model.simulate(p, day + horizon, seed=seed, shocks={day: size})
        calm = model.simulate(p, day + horizon, seed=seed)    # placebo: no surprise
    except model.RunawayError:
        return None

    def ratio(run):
        # The mispricing (log price - log value) just before the surprise is gap[day].
        # A price that followed the value exactly would keep it. The fall of the
        # price in units of the surprise is 1 minus the change in mispricing since.
        gap = run["lp"] - run["lf"]
        return 1.0 - (gap[day + 1:day + horizon + 1] - gap[day]) / -np.log1p(size)

    hit, placebo = ratio(o), ratio(calm) - 1.0      # placebo: ordinary wobble only
    return dict(chartists=float(o["z"][day]), peak=float(hit.max()), end=float(hit[-1]),
                placebo_peak=float(placebo.max()))


def surprise_experiment(pool, p, seeds, size=-0.10, horizon=10):
    """One surprise per seed, on a random day after the warm-up: the value drops
    once by `size`. For each history: the chartist share the day before, and the
    price's fall in units of the surprise over the next `horizon` days (1 = the
    price fell exactly as much as the value, 1.2 = it overshot by 20%): `peak` = the
    largest, `end` = after `horizon` days. News is still on, so the value keeps
    moving too; the measure follows the price relative to the value.
    `placebo_peak`: the same peak in the same history without the surprise, minus 1
    (how much of `peak - 1` the ordinary wobble of the mispricing gives by itself).
    Returns ({name: array} over the seeds that did not run away, how many did)."""
    if not (-1 < size < 0):
        raise ValueError(f"size must be a fall between -1 and 0, got {size}")
    rows = list(pool.map(_surprise_one, [(p, int(s), size, horizon) for s in seeds]))
    ok = [r for r in rows if r is not None]
    return {k: np.array([r[k] for r in ok]) for k in ("chartists", "peak", "end", "placebo_peak")}, \
        len(rows) - len(ok)


def median_overshoot(peak, placebo_peak, draws=2000, seed=0):
    """The overshoot caused by the surprise, history by history: the median of
    peak - (placebo_peak + 1) (each history minus its own placebo), with a 95%
    bootstrap interval (resampling the histories). Returns (median, low, high).
    (The difference of the two medians, as read off the table, is only a shortcut.)"""
    d = np.asarray(peak, dtype=float) - (np.asarray(placebo_peak, dtype=float) + 1)
    if d.size < 2:
        raise ValueError(f"need at least 2 histories, got {d.size}")
    g = np.random.default_rng(seed)
    meds = np.median(d[g.integers(0, d.size, (draws, d.size))], axis=1)
    return float(np.median(d)), float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))


def paired_difference(off, on, draws=2000, seed=0):
    """Mean of on - off over paired runs, with a 95% bootstrap interval (resampling
    the pairs). Returns (mean, low, high)."""
    d = np.asarray(on, dtype=float) - np.asarray(off, dtype=float)
    if d.size < 2:
        raise ValueError(f"need at least 2 pairs, got {d.size}")
    g = np.random.default_rng(seed)
    means = d[g.integers(0, d.size, (draws, d.size))].mean(axis=1)
    return float(d.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
