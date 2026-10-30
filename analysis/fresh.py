import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # project folder
import types
import numpy as np
from herding import model, stats
main = types.SimpleNamespace(**{**vars(model), **vars(stats)})   # these probes predate the split into modules
from multiprocessing import Pool
def ac(x, k): x = x - x.mean(); return np.corrcoef(x[:-k], x[k:])[0, 1]
def job(s):
    r, f = main.simulate(5200, seed=s); r = r[200:]
    return main.kurtosis(r), main.hill_alpha(r), ac(r, 1), ac(abs(r), 1), np.isfinite(r).all()
if __name__ == "__main__":
    with Pool() as p: res = np.array(p.map(job, range(10, 30)))
    for name, col in zip(["kurt", "alpha", "ac1", "abs_ac1", "finite"], res.T):
        print(f"{name:8s} mean={col.mean():.3f} min={col.min():.3f} max={col.max():.3f}")
