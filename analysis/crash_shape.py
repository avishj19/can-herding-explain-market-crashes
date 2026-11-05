"""Why the calibrated model's 5-day returns are fatter-tailed than daily ones: its
crashes are multi-day slides (momentum after big days), unlike the S&P's one-day shocks.

Run from the project folder:  python3 analysis/crash_shape.py
"""
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from herding import calibrate, experiments, market_data  # noqa: E402


def look(r):
    z = (r - r.mean()) / r.std()
    n = len(r) // 5
    w = r[:n * 5].reshape(n, 5).sum(axis=1); zw = (w - w.mean()) / w.std()
    i = np.argmax(np.abs(zw))                      # the most extreme week
    days = z[i * 5:i * 5 + 5]
    sel = np.abs(z[:-1]) > 3                      # momentum right after big days
    c = np.corrcoef(z[:-1][sel], z[1:][sel])[0, 1]
    return np.abs(zw).max(), np.abs(z).max(), np.sum(np.sign(days) == np.sign(w[i])), c


def job(seed):
    return look(experiments.history(calibrate.load(), seed))


if __name__ == "__main__":
    with Pool() as pool:
        res = np.array(pool.map(job, range(600, 620)))
    s = look(market_data.spx_returns().values)
    for k, n in enumerate(["most extreme WEEK (in weekly std)", "most extreme DAY (in daily std)",
                           "days in that week moving the same way (of 5)", "corr(day, next day) after |move|>3 std"]):
        print(f"{n:<46} S&P {s[k]:6.2f}   model median {np.median(res[:, k]):6.2f} (range {res[:, k].min():.2f}-{res[:, k].max():.2f})")
