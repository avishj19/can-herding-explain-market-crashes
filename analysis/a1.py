import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # project folder
import types
import numpy as np
from herding import model, stats
main = types.SimpleNamespace(**{**vars(model), **vars(stats)})   # these probes predate the split into modules
from multiprocessing import Pool
def job(a):
    a1, seed = a
    r, f, lp, lf = main.simulate(4200, True, dict(main.P, a1=a1), seed=seed, paths=True)
    r, f = r[200:], f[200:]
    big = np.mean(np.abs(r) > 4 * r.std())        # share of >4-sigma days
    return a1, seed, main.kurtosis(r), main.hill_alpha(r), f.mean(), np.abs(lp - lf)[200:].mean(), big, r.std()
if __name__ == "__main__":
    with Pool() as p: res = np.array(p.map(job, [(a, s) for a in (0.0, 0.3, 0.6, 0.9, 1.2) for s in range(100, 110)]))
    print(" a1   kurt(median)  alpha(mean)  chartists  mispricing  >4sd days   std")
    for a in (0.0, 0.3, 0.6, 0.9, 1.2):
        m = res[res[:, 0] == a]
        print(f"{a:4.1f}  {np.median(m[:,2]):10.2f}  {m[:,3].mean():10.2f}  {m[:,4].mean():9.3f}  {m[:,5].mean():9.4f}  {m[:,6].mean()*100:7.2f}%  {m[:,7].mean():.4f}")
