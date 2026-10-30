import sys, warnings; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # project folder
import types
import numpy as np
from herding import model, stats
main = types.SimpleNamespace(**{**vars(model), **vars(stats)})   # these probes predate the split into modules
from multiprocessing import Pool
def job(a):
    over, seed = a
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, f, lp, lf = main.simulate(5000, True, dict(main.P, **over), seed=seed, paths=True,
                                     max_mispricing=None)   # research: unstable settings on purpose
    return str(over), np.abs(lp - lf).max(), f.max(), r.std()
CASES = [{}, {"a1": 1.2}, {"tc": 0.2}, {"gamma": 0.05}, {"a2": 4.0}, {"N": 250}, {"s": 0.5},   # healthy
         {"s": 0.25}, {"s": 0.35}, {"N": 100}, {"v2": 0.5}]                                        # suspect
if __name__ == "__main__":
    with Pool() as p: res = p.map(job, [(c, s) for c in CASES for s in range(200, 210)])
    import pandas as pd
    df = pd.DataFrame(res, columns=["case", "max_mis", "max_frac", "std"])
    print(df.groupby("case", sort=False).agg(max_mis=("max_mis", "max"), med_mis=("max_mis", "median"),
                                             max_frac=("max_frac", "max"), std=("std", "median")).round(3))
