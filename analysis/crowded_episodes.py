"""How long crowded episodes (over 50% chartists) last, calibrated vs default settings.

Run from the project folder:  python3 analysis/crowded_episodes.py
"""
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from herding import calibrate, model  # noqa: E402
from herding.experiments import BURN, DAYS  # noqa: E402


def episodes(args):
    p, seed = args
    f = model.simulate(DAYS + BURN, p=p, seed=seed)[1][BURN:] > 0.5
    edges = np.diff(np.r_[0, f.astype(int), 0])
    return list(np.where(edges == -1)[0] - np.where(edges == 1)[0]), float(f.mean())


if __name__ == "__main__":
    for name, p in [("calibrated v3", calibrate.load()), ("default settings", model.P)]:
        with Pool() as pool:
            res = pool.map(episodes, [(p, s) for s in range(600, 620)])
        lens = np.concatenate([r[0] for r in res]) if any(r[0] for r in res) else np.array([0])
        print(f"{name:<17} episodes/77 yrs {len(lens) / 20:5.1f}  median length {np.median(lens):5.0f} days  "
              f"90th pct {np.percentile(lens, 90):5.0f}  longest {lens.max():5.0f}  time crowded {np.mean([r[1] for r in res]):.0%}")
