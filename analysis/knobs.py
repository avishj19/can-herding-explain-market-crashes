import sys, warnings; sys.path.insert(0, "/Users/avishjain/02-120 Project")
import types
import numpy as np
from herding import model, stats
main = types.SimpleNamespace(**{**vars(model), **vars(stats)})   # these probes predate the split into modules
from multiprocessing import Pool
def job(a):
    key, val, seed, herding = a
    r, f, lp, lf = main.simulate(2200, herding, dict(main.P, **{key: val}), seed=seed, paths=True)
    r, f, mis = r[200:], f[200:], np.abs(lp - lf)[200:]
    x = r - r.mean()
    return key, val, seed, herding, dict(std=r.std(), mis=mis.mean(), frac=f.mean(),
        kurt=main.kurtosis(r), alpha=main.hill_alpha(r), ac1=np.corrcoef(x[:-1], x[1:])[0, 1])
KNOBS = {"a1": (0.2, 0.9), "a2": (0.0, 4.0), "tc": (0.05, 0.2), "beta": (25.0, 100.0),
         "gamma": (0.05, 0.2), "s": (0.25, 2.0)}
if __name__ == "__main__":
    jobs = [(k, v, s, True) for k, vs in KNOBS.items() for v in vs for s in range(4)]
    jobs += [("sigma_news", v, s, False) for v in (0.005, 0.01, 0.02) for s in range(4)]
    jobs += [("beta", v, s, False) for v in (25.0, 100.0) for s in range(4)]
    with Pool() as p: res = p.map(job, jobs)
    import pandas as pd
    df = pd.DataFrame([dict(knob=k, val=v, seed=s, herding=h, **m) for k, v, s, h, m in res])
    pd.set_option("display.width", 200)
    print(df.groupby(["herding", "knob", "val"])[["std", "mis", "frac", "kurt", "alpha", "ac1"]].mean().round(4))
    # paired: in how many seeds does the high setting beat the low one?
    for k, (lo, hi) in KNOBS.items():
        d = df[(df.knob == k) & df.herding].pivot(index="seed", columns="val")
        print(k, {m: int((d[m][hi] > d[m][lo]).sum()) for m in ["std", "mis", "frac", "kurt"]}, "of 4 seeds: high > low")
