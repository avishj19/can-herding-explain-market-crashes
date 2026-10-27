import sys, json; sys.path.insert(0, "/Users/avishjain/02-120 Project")
import types
import numpy as np
from herding import model, stats
main = types.SimpleNamespace(**{**vars(model), **vars(stats)})   # these probes predate the split into modules
from multiprocessing import Pool
def ac(x, k): x = x - x.mean(); return np.corrcoef(x[:-k], x[k:])[0, 1]
def job(a):
    seed, herding, over = a
    try:
        r, f = main.simulate(4000, herding, dict(main.P, **over), seed=seed)
    except Exception as e:
        return dict(err=str(e), **over)
    r = r[200:]; f = f[200:]
    return dict(**over, h=herding, kurt=main.kurtosis(r), alpha=main.hill_alpha(r),
                ac1=ac(r, 1), ac2=ac(r, 2), abs1=ac(abs(r), 1), abs20=ac(abs(r), 20), abs100=ac(abs(r), 100), frac=f.mean(), std=r.std())
if __name__ == "__main__":
    over = json.loads(sys.argv[1]) if len(sys.argv) > 1 else [{}]
    jobs = [(s, h, o) for o in over for h in (True, False) for s in range(4)]
    import pandas as pd
    with Pool() as p: df = pd.DataFrame(p.map(job, jobs))
    pd.set_option("display.width", 200)
    keys = [c for c in df.columns if c not in ("kurt","alpha","ac1","ac2","abs1","abs20","abs100","frac","std","err")]
    print(df.groupby(keys, dropna=False).mean(numeric_only=True).round(3))
    if "err" in df: print(df[df.err.notna()])
