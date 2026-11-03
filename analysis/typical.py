"""Does the real S&P 500 look like a typical history of the calibrated model?

Percentile of the S&P among 40 simulated histories (seeds 400-439, never used by
the calibration). Older calibrations (analysis/history/) were fitted to the old
switching rule min(1, rate) and cannot be rerun on the current model.

Run from the project folder:  python3 analysis/typical.py
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from herding import calibrate, experiments, stats  # noqa: E402


def job(seed):
    return stats.stylized_facts(experiments.history(calibrate.load(), seed))


if __name__ == "__main__":
    with Pool() as pool:
        facts = pool.map(job, range(400, 440))
    t = calibrate.spx_facts()
    pct = calibrate.percentiles(t, facts, calibrate.TARGETS)
    print("percentile of the S&P 500 among 40 simulated histories "
          "(50 = perfectly typical; <5 or >95 = S&P looks unusual)")
    for k in calibrate.TARGETS:
        print(f"{k:<12}{pct[k]:>6.0f}")
    print("facts where S&P is outside the 5-95% range:",
          sum(not 5 <= v <= 95 for v in pct.values()), "of", len(pct))
