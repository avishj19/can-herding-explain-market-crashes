"""Reviewer spot-check of A2 (fresh seeds the A2 report did not use)."""
import sys, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from herding import explore as ex, market_data, stats

def one(args):
    mix, a1, seed = args
    p, burn = ex.fitted_settings()
    days = len(market_data.spx_returns())
    paths = {}
    r, err = ex.run_mix(p, mix, a1, days, seed, burn, paths=paths)
    if r is None:
        return (mix, a1, seed, None, None, None)
    s = ex.summary(r)
    z = paths["z"]
    return (mix, a1, seed, s, float(z.mean()), (float(z.min()), float(np.percentile(z, 99)), float(z.max())))

if __name__ == "__main__":
    p, _ = ex.fitted_settings(); a1 = p["a1"]
    print("fitted settings:", p)
    jobs = [("herding", a1, s) for s in range(910000, 910010)]
    jobs += [("fundamentalists", a1, s) for s in range(910000, 910010)]
    jobs += [("herding", a, s) for a in (a1, 1.0, 0.8) for s in range(920000, 920100)]
    with Pool() as pool:
        res = pool.map(one, jobs, chunksize=1)
    h = [r for r in res[:10] if r[3]]
    print("herding fresh 10:", len(h), "survived")
    for k in ("tail", "kurtosis", "memory", "worst"):
        print(k, np.round(np.median([r[3][k] for r in h]), 3), np.round([r[3][k] for r in h], 2))
    print("big/1000", np.median([1000 * r[3]["big_days"] / r[3]["days"] for r in h]))
    print("z mean per history", np.round([r[4] for r in h], 4))
    print("z min/p99/max", [tuple(np.round(r[5], 3)) for r in h[:3]])
    f = [r for r in res[10:20] if r[3]]
    print("fund kurt median", np.median([r[3]["kurtosis"] for r in f]), "tail", np.median([r[3]["tail"] for r in f]))
    for a in (a1, 1.0, 0.8):
        rows = [r for r in res[20:] if r[1] == a]
        surv = [r for r in rows if r[3]]
        print(f"a1={a}: runaways {len(rows)-len(surv)}/{len(rows)}; survivor median tail {np.median([r[3]['tail'] for r in surv]):.3f}, kurt {np.median([r[3]['kurtosis'] for r in surv]):.2f}, z {np.median([r[4] for r in surv]):.4f}")
