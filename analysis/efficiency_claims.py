"""Lux & Marchesi (1999) efficient-market claims, measured on the model: price equals
value on average, no persistent large mispricing, volatility rises with the chartist share.

Run from the project folder:  python3 analysis/efficiency_claims.py
"""
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from herding import calibrate, model  # noqa: E402
from herding.experiments import BURN, DAYS  # noqa: E402


def run(args):
    p, seed = args
    o = model.simulate_full(DAYS + BURN, p=p, seed=seed)
    r, frac = o["r"][BURN:], o["frac"][BURN:]
    mis = (o["lp"] - o["lf"])[BURN + 1:]
    # longest stretch where |mispricing| > 10%
    big = np.abs(mis) > 0.10
    e = np.diff(np.r_[0, big.astype(int), 0]); runs = np.where(e == -1)[0] - np.where(e == 1)[0]
    # volatility by chartist share (the paper's "critical fraction")
    bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
    vol = [np.std(r[(frac >= lo) & (frac < hi)]) if np.sum((frac >= lo) & (frac < hi)) > 50 else np.nan
           for lo, hi in zip(bins[:-1], bins[1:])]
    return mis.mean(), np.abs(mis).mean(), runs.max() if len(runs) else 0, big.mean(), vol, r.std()


if __name__ == "__main__":
    for name, p in [("calibrated v3", calibrate.load()), ("default settings", model.P)]:
        with Pool() as pool:
            res = pool.map(run, [(p, s) for s in range(600, 620)])
        m = np.array([x[0] for x in res]); a = np.array([x[1] for x in res])
        L = np.array([x[2] for x in res]); share = np.array([x[3] for x in res])
        vol = np.nanmedian(np.array([np.array(x[4]) / x[5] for x in res]), axis=0)
        print(f"\n{name}")
        print(f"  1. price = value on average: mean log(price/value) {m.mean():+.4f} (sd across histories {m.std():.4f})")
        print(f"     typical |mispricing| {a.mean():.3%}")
        print(f"  2. no persistent big deviations: days with |mispricing|>10%: {share.mean():.2%}, "
              f"longest such stretch: median {np.median(L):.0f} days, max {L.max():.0f}")
        print("  3. volatility (relative to overall) by chartist share 0-10,10-30,30-50,50-70,70-90,90-100%:",
              " ".join("  -  " if np.isnan(v) else f"{v:.2f}" for v in vol))
