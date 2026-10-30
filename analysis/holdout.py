import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # project folder
import numpy as np
from herding import model, stats, calibrate
from multiprocessing import Pool
def job(a):
    which, seed, herding = a
    p = calibrate.load() if which == "cal" else model.P
    try:
        r, _ = model.simulate(19600, herding, p, seed=seed)
    except model.ModelRegimeError as e:
        return which, seed, herding, None
    return which, seed, herding, stats.stylized_facts(r[300:])
if __name__ == "__main__":
    jobs = [(w, s, h) for w in ("default", "cal") for s in range(100, 120) for h in (True, False)]
    with Pool() as p: res = p.map(job, jobs)
    target = calibrate.spx_facts()
    for w in ("default", "cal"):
        for h in (True, False):
            fs = [f for ww, s, hh, f in res if ww == w and hh == h]
            ok = [f for f in fs if f is not None]
            m = {k: np.mean([f[k] for f in ok]) for k in ok[0]}
            sd = {k: np.std([f[k] for f in ok]) for k in ok[0]}
            print(f"{w:8s} herding={h!s:5s} regime-errors={len(fs) - len(ok)}/20  loss={calibrate.loss(m, target) if h else float('nan'):.1f}")
            print("   " + "  ".join(f"{k}={m[k]:.3f}±{sd[k]:.3f}" for k in ("kurtosis", "alpha_5pct", "alpha_1pct", "ac_r_1", "ac_r_2", "ac_abs_20", "ac_abs_100")))
