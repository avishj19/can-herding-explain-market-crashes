import sys, json; sys.path.insert(0, "/Users/avishjain/02-120 Project")
import numpy as np
from herding import model, stats, calibrate
from multiprocessing import Pool
from pathlib import Path
V1 = json.loads(Path("/Users/avishjain/02-120 Project/analysis/history/calibration_8_targets.json").read_text())["params"]
def job(a):
    which, seed = a
    p = {"v1": V1, "v2": calibrate.load()}[which]
    r, _ = model.simulate(19600, p=p, seed=seed)
    return which, stats.stylized_facts(r[300:])
if __name__ == "__main__":
    with Pool() as pool: res = pool.map(job, [(w, s) for w in ("v1", "v2") for s in range(400, 440)])
    t = calibrate.spx_facts()
    print("percentile of the S&P 500 among 40 simulated histories (50 = perfectly typical; <5 or >95 = S&P looks unusual)")
    print(f"{'fact':<12}{'v1':>6}{'v2':>6}")
    worst = {}
    for k in calibrate.TARGETS:
        row = []
        for w in ("v1", "v2"):
            vals = np.array([f[k] for ww, f in res if ww == w])
            pct = 100 * np.mean(vals < t[k])
            row.append(pct); worst.setdefault(w, []).append(abs(pct - 50))
        print(f"{k:<12}{row[0]:>6.0f}{row[1]:>6.0f}")
    for w in ("v1", "v2"):
        print(w, "facts where S&P is outside the 5-95% range:", sum(x > 45 for x in worst[w]), "of", len(worst[w]))
