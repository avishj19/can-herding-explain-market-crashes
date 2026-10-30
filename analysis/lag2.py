import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # project folder
import types
import numpy as np
from herding import model, stats
main = types.SimpleNamespace(**{**vars(model), **vars(stats)})   # these probes predate the split into modules
from multiprocessing import Pool
def ac(x, k): x = x - x.mean(); return np.corrcoef(x[:-k], x[k:])[0, 1]
def job(s):
    r, _ = main.simulate(5200, seed=s); r = r[200:]
    c = np.clip(r, -3 * r.std(), 3 * r.std())     # same, with outliers capped
    return [ac(r, k) for k in (1, 2, 3)] + [ac(c, k) for k in (1, 2, 3)]
if __name__ == "__main__":
    with Pool() as p: a = np.array(p.map(job, range(10, 30)))
    print("raw    lag1-3 mean:", a[:, :3].mean(0).round(3), " worst:", a[:, :3].min(0).round(3))
    print("capped lag1-3 mean:", a[:, 3:].mean(0).round(3), " worst:", a[:, 3:].min(0).round(3))
