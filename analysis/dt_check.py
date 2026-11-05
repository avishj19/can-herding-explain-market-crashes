"""Does the time step dt change the results? (It should not: the switching
probabilities are exact, ticks are Poisson and mu is scaled; see notes/SPEC.md.)

Compares the median fingerprint of 12 histories at dt = 0.01 and dt = 0.005, and
reports each difference in units of the seed-to-seed spread.

Run from the project folder:  python3 analysis/dt_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

from herding import calibrate, experiments, stats


def main(days=19_329, n=12):
    p = calibrate.load("spx")
    out = {}
    for dt in (0.01, 0.005):
        runs = experiments.returns(dict(p, dt=dt), calibrate.HELD_OUT_SEEDS[:n], days)
        out[dt] = [stats.fingerprint(r) for r in runs]
    print(f"{'measure':<11}{'dt=0.01':>10}{'dt=0.005':>10}{'gap / spread':>14}")
    for k in stats.FINGERPRINT:
        a = np.array([f[k] for f in out[0.01]])
        b = np.array([f[k] for f in out[0.005]])
        spread = np.sqrt((a.var() + b.var()) / 2)
        print(f"{k:<11}{np.median(a):>10.4f}{np.median(b):>10.4f}{(np.median(b) - np.median(a)) / spread:>14.2f}")


if __name__ == "__main__":
    main()
