"""Does the time step dt change the results? It should not if dt is small enough:
the switching probabilities are exact, ticks are Poisson and mu is scaled (SPEC.md sec. 5).

Compares the median fingerprint of 12 histories at several dt, and reports each
difference from dt = 0.01 in units of the seed-to-seed spread.
Finding (November 2026): the S&P-calibrated settings barely change; the paper-matched
settings lose most of their fat tails and clustering as dt shrinks (RESULTS.md, Limitations).

Run from the project folder:  python3 scripts/analysis/dt_check.py [spx|paper]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np

from herding import calibrate, experiments, stats


def main(kind="spx", dts=(0.01, 0.005, 0.002), days=19_329, n=12):
    p = calibrate.load(kind)
    out = {dt: [stats.fingerprint(r) for r in
                experiments.returns(dict(p, dt=dt), calibrate.HELD_OUT_SEEDS[:n], days)] for dt in dts}
    print(f"{kind}: median of {n} histories of {days:,} days; (gap from dt={dts[0]}) / spread")
    print(f"{'measure':<11}" + "".join(f"{'dt=' + str(dt):>18}" for dt in dts))
    for k in stats.FINGERPRINT:
        a = np.array([f[k] for f in out[dts[0]]])
        cells = []
        for dt in dts:
            b = np.array([f[k] for f in out[dt]])
            spread = np.sqrt((a.var() + b.var()) / 2)
            cells.append(f"{np.median(b):>10.4f} ({(np.median(b) - np.median(a)) / spread:+5.1f})")
        print(f"{k:<11}" + "".join(f"{c:>18}" for c in cells))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "spx")
