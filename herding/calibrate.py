"""Calibrate the Lux-Marchesi model to the S&P 500's stylized facts.

Searches the model's settings for the combination whose return "fingerprint"
(tails, predictability, volatility clustering) best matches the real market.
Kurtosis is reported but not targeted: one day (Black Monday) dominates it.

Run:  python3 -m herding.calibrate            (~4 min with numba; saves calibrated_params.json)
      python3 -m herding.calibrate --refine   (local round only, starting from the saved file)
"""
import json
from multiprocessing import Pool

import numpy as np

from . import market_data
from . import model
from . import stats

OUT = market_data.ROOT / "calibrated_params.json"
TARGETS = ["alpha_5pct", "alpha_1pct", "share_4sd", "ac_r_1", "ac_r_2", "ac_r_2to10",
           "ac_abs_1", "ac_abs_20", "ac_abs_50", "ac_abs_100"]
# How far off counts as "one unit" of mismatch (about the seed-to-seed noise)
SCALE = dict(alpha_5pct=0.25, alpha_1pct=0.5, share_4sd=0.002, ac_r_1=0.03, ac_r_2=0.03,
             ac_r_2to10=0.01, ac_abs_1=0.05, ac_abs_20=0.04, ac_abs_50=0.03, ac_abs_100=0.03)
# Search ranges (low, high) for each setting that shapes the fingerprint
RANGES = dict(a1=(0.0, 1.5), a2=(0.0, 6.0), a3=(0.02, 2.0), v1=(0.5, 6.0), v2=(1.0, 6.0),
              tc=(0.02, 0.3), gamma=(0.03, 0.8), s=(0.5, 2.0), beta=(20.0, 100.0),
              tau=(0.05, 1.0))
DAYS, BURN = 19300, 300          # same length as the S&P 500 sample


def spx_facts():
    return stats.stylized_facts(market_data.spx_returns().values)


def model_facts(p, seeds, days=DAYS):
    """Average fingerprint over seeds, or None if any run leaves the realistic regime."""
    facts = []
    for seed in seeds:
        try:
            r, _ = model.simulate(days + BURN, p=p, seed=seed)
        except (model.ModelRegimeError, ValueError, FloatingPointError):
            return None
        facts.append(stats.stylized_facts(r[BURN:]))
    return {k: float(np.mean([f[k] for f in facts])) for k in facts[0]}


def loss(facts, target):
    """Sum of squared mismatches, each measured in units of SCALE."""
    if facts is None:
        return float("inf")
    return float(sum(((facts[k] - target[k]) / SCALE[k]) ** 2 for k in TARGETS))


def sample(n, seed):
    """n settings spread evenly over RANGES (Latin hypercube)."""
    g = np.random.default_rng(seed)
    cols = {k: lo + (hi - lo) * (g.permutation(n) + g.random(n)) / n
            for k, (lo, hi) in RANGES.items()}
    return [dict(model.P, **{k: float(cols[k][i]) for k in RANGES}) for i in range(n)]


def _score(args):
    p, seeds, target = args
    f = model_facts(p, seeds)
    return loss(f, target), f


def search(n=400, top=10, seeds=(0, 1, 2), refine_seeds=tuple(range(10, 16)), seed=0):
    target = spx_facts()
    cands = [p for p in sample(n, seed)
             if p["dt"] <= 1 / (p["beta"] * p["gamma"])]          # stability limit
    with Pool() as pool:
        rough = pool.map(_score, [(p, seeds, target) for p in cands])
        best = sorted(zip(rough, cands), key=lambda x: x[0][0])[:top]
        # refine: re-score the finalists, plus small nudges of each, on new seeds
        g = np.random.default_rng(seed + 1)
        finals = [p for _, p in best]
        finals += [dict(p, **{k: float(np.clip(p[k] * g.lognormal(0, 0.15), *RANGES[k]))
                              for k in RANGES}) for p in finals for _ in range(3)]
        finals = [p for p in finals if p["dt"] <= 1 / (p["beta"] * p["gamma"])]
        fine = pool.map(_score, [(p, refine_seeds, target) for p in finals])
    (l, f), p = min(zip(fine, finals), key=lambda x: x[0][0])
    return p, l, f, target, len(cands), sum(np.isinf([r[0] for r in rough]))


def refine(center, n=200, spread=0.3, seeds=(3, 4, 5), final_seeds=tuple(range(20, 28)),
           top=8, seed=2):
    """Local search: n random nudges (log-normal, +-spread) around a good setting."""
    target = spx_facts()
    g = np.random.default_rng(seed)
    cands = [center] + [dict(center, **{k: float(np.clip(center[k] * g.lognormal(0, spread),
                                                          *RANGES[k])) for k in RANGES})
                        for _ in range(n)]
    cands = [p for p in cands if p["dt"] <= 1 / (p["beta"] * p["gamma"])]
    with Pool() as pool:
        rough = pool.map(_score, [(p, seeds, target) for p in cands])
        best = [p for _, p in sorted(zip(rough, cands), key=lambda x: x[0][0])[:top]]
        fine = pool.map(_score, [(p, final_seeds, target) for p in best])
    (l, f), p = min(zip(fine, best), key=lambda x: x[0][0])
    return p, l, f, target


def percentiles(target, sims, keys):
    """Where the real value of each fact falls among simulated histories (0-100)."""
    return {k: 100.0 * float(np.mean([s[k] < target[k] for s in sims])) for k in keys}


def representative(sims, target):
    """Index of the history with the median mismatch: typical, not cherry-picked."""
    losses = [loss(s, target) for s in sims]
    return int(np.argsort(losses)[len(losses) // 2])


def load(path=OUT):
    """Calibrated settings saved by search(), validated like any other settings."""
    p = json.loads(path.read_text())["params"]
    model.check_settings(p)
    return p


if __name__ == "__main__":
    import sys
    if "--refine" in sys.argv:                        # only the local round, from saved result
        p, l, f, target = refine(load())
    else:
        p, l, f, target, n, unstable = search()
        print(f"searched {n} settings ({unstable} left the realistic regime)")
        p, l, f, target = refine(p)                  # second, local round
    default = model_facts(model.P, range(20, 28))
    print(f"loss: default {loss(default, target):.1f} -> calibrated {l:.1f}")
    print(f"{'fact':<12}{'S&P 500':>10}{'default':>10}{'calibrated':>12}")
    for k in target:
        print(f"{k:<12}{target[k]:>10.3f}{default[k]:>10.3f}{f[k]:>12.3f}")
    OUT.write_text(json.dumps(dict(params=p, loss=l, facts=f, target=target), indent=2))
    print(f"\nsaved {OUT.name}:", {k: round(p[k], 3) for k in RANGES})
