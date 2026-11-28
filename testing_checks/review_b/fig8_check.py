import sys, numpy as np
sys.path.insert(0, "scripts")
from multiprocessing import Pool
import make_figures as mf
from herding import calibrate, stats
if __name__ == "__main__":
    p = calibrate.load("spx")
    with Pool() as pool:
        rows, keys, names = mf.fig8_out_of_sample(pool, p, path=mf.Path("testing_checks/review_b/fig08_rerun.png"))
    print(keys)
    for n, r in zip(names, rows): print(f"{n:20s}", " ".join(f"{v:5.0f}" for v in r))
    print("unusual", int(np.sum((rows<5)|(rows>95))), "of", rows.size, "index rows", int(np.sum((rows[:2]<5)|(rows[:2]>95))))
