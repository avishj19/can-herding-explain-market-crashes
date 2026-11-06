"""Lux & Marchesi (1999)'s efficient-market claims, measured on both fitted settings.

The paper (p. 499-500): (1) "a price which on average equals the fundamental value";
(2) "no pathological situations, with persistent deviations from the economy's
fundamentals"; (3) volatility is above average when the share of noise traders is
high (near a critical value).

Run from the project folder:  python3 analysis/efficiency_claims.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

from herding import calibrate, experiments

DAYS = 100_000


def longest_run(mask):
    best = run = 0
    for m in mask:
        run = run + 1 if m else 0
        best = max(best, run)
    return best


def main(days=DAYS, n=4):
    """Measure the three claims on n held-out histories of `days` days per setting."""
    for kind in ("paper", "spx"):
        p = calibrate.load(kind)
        h = experiments.histories(p, calibrate.HELD_OUT_SEEDS[:n], days, keep=("r", "lp", "lf", "z"))
        h = [x for x in h if x is not None]
        gap = np.concatenate([x["lp"] - x["lf"] for x in h])
        r = np.concatenate([x["r"] for x in h])
        z = np.concatenate([x["z"][1:] for x in h])
        vol = np.abs(r) / r.std()
        # the 10% of days with the fewest / most chartists, by rank (the share z takes
        # only the values k/N, so a quantile cut-off would sweep in ties)
        order, tenth = np.argsort(z, kind="stable"), len(z) // 10
        low, high = vol[order[:tenth]].mean(), vol[order[-tenth:]].mean()
        print(f"{kind}: {len(h)} histories x {days:,} days")
        print(f"  (1) mean log(price/value) {gap.mean():+.4f} (sd {gap.std():.4f})")
        print(f"  (2) days more than 10% mispriced {np.mean(np.abs(gap) > np.log(1.1)):.2%}; "
              f"longest such stretch {max(longest_run(np.abs(x['lp'] - x['lf']) > np.log(1.1)) for x in h)} days")
        print(f"  (3) volatility, least crowded 10% of days {low:.2f}x average, most crowded 10% {high:.2f}x")


if __name__ == "__main__":
    main()
