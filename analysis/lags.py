import sys; sys.path.insert(0, "/Users/avishjain/02-120 Project")
import numpy as np
from herding import model, stats, calibrate, market_data
from multiprocessing import Pool
def job(a):
    which, seed = a
    p = calibrate.load() if which == "cal" else model.P
    r, _ = model.simulate(19600, p=p, seed=seed)
    return which, stats.acf(r[300:], [1, 2, 3, 4, 5, 6])
if __name__ == "__main__":
    with Pool() as pool: res = pool.map(job, [(w, s) for w in ("paper", "cal") for s in range(100, 110)])
    print("S&P   ", stats.acf(market_data.spx_returns().values, [1, 2, 3, 4, 5, 6]).round(3))
    for w in ("paper", "cal"):
        print(f"{w:6s}", np.mean([a for ww, a in res if ww == w], axis=0).round(3))
