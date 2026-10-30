import sys, json; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # project folder
import numpy as np
from herding import model, stats, calibrate
from multiprocessing import Pool
from pathlib import Path
V1 = json.loads((Path(__file__).resolve().parent / "history" / "calibration_8_targets.json").read_text())["params"]
def job(a):
    which, seed = a
    p = {"paper": model.P, "v1": V1, "v2": calibrate.load()}[which]
    try:
        r, _ = model.simulate(19600, p=p, seed=seed)
    except model.ModelRegimeError:
        return which, None
    return which, stats.stylized_facts(r[300:])
if __name__ == "__main__":
    with Pool() as pool: res = pool.map(job, [(w, s) for w in ("paper", "v1", "v2") for s in range(300, 320)])
    t = calibrate.spx_facts()
    print(f"{'fact':<12}{'S&P':>8}" + "".join(f"{w:>16}" for w in ("paper", "v1", "v2")))
    agg = {}
    for w in ("paper", "v1", "v2"):
        ok = [f for ww, f in res if ww == w and f is not None]
        agg[w] = ({k: np.mean([f[k] for f in ok]) for k in ok[0]}, {k: np.std([f[k] for f in ok]) for k in ok[0]}, 20 - len(ok))
    for k in t:
        print(f"{k:<12}{t[k]:>8.3f}" + "".join(f"{agg[w][0][k]:>9.3f}±{agg[w][1][k]:<6.3f}" for w in ("paper", "v1", "v2")))
    for w in ("paper", "v1", "v2"):
        print(w, "held-out loss", round(calibrate.loss(agg[w][0], t), 1), "runaways", agg[w][2])
