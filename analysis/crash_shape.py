"""Do big moves bounce back (like the S&P) or keep going (slow-motion slides)?

For every day that moves more than 3 standard deviations, correlate its return with
the next day's. The old model (before the rebuild) gave +0.31 vs the S&P's -0.19.

Run from the project folder:  python3 analysis/crash_shape.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

from herding import calibrate, experiments, market_data


def after_big_moves(r, k=3.0):
    """Correlation of big-move days (|r| > k sd) with the day after, and their count."""
    r = np.asarray(r)
    big = np.where(np.abs(r[:-1]) > k * r.std())[0]
    return float(np.corrcoef(r[big], r[big + 1])[0, 1]), len(big)


def main(days=None, n=20):
    """Print the S&P number and the model's over n histories of `days` days
    (default: as long as the S&P record)."""
    spx = market_data.spx_returns().values
    c, n_big = after_big_moves(spx)
    print(f"S&P 500: next-day correlation after {n_big} big days {c:+.2f}")
    p = calibrate.load("spx")
    runs = experiments.returns(p, calibrate.HELD_OUT_SEEDS[:n], days or len(spx))
    vals = [after_big_moves(r) for r in runs]
    cs = np.array([v[0] for v in vals])
    print(f"model ({n} histories): median {np.median(cs):+.2f}, 5-95% range "
          f"{np.percentile(cs, 5):+.2f} to {np.percentile(cs, 95):+.2f}; "
          f"S&P percentile {experiments.percentile(c, cs):.0f}%")


if __name__ == "__main__":
    main()
