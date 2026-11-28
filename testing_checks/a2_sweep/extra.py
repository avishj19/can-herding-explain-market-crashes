"""Extra: how close is the fitted a1 to the runaway edge? 200 fresh seeds (720000-720199) at a1 near the fitted one,
and how many fresh herding histories are as bad as the S&P's worst day / kurtosis.
    env -u PYTHONDONTWRITEBYTECODE python3 testing_checks/a2_sweep/extra.py"""
import json, pickle, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent))
import run as R
from herding import explore as ex, market_data

if __name__ == "__main__":
    p, burn = ex.fitted_settings(); days = len(market_data.spx_returns())
    seeds = list(range(720000, 720200))
    a1s = [0.9, 0.95, 1.0, 1.05, round(p["a1"], 4), 1.10, 1.15, 1.2]
    a1s[4] = p["a1"]
    with Pool() as pool:
        rows = pool.map(R.job, [("herding", a, s, days, False) for a in a1s for s in seeds], chunksize=2)
    out = {}
    for a in a1s:
        g = [r for r in rows if r["a1"] == a]
        n_run = sum(r["ran"] is not None for r in g)
        out[a] = {"runaways": n_run, "n": len(g), "tail_med": float(np.nanmedian(R.col(g, "tail")))}
        print(f"a1={a:.4f} runaways {n_run}/{len(g)} median tail {out[a]['tail_med']:.2f}")
    # S&P comparison from the 40 fresh + 40 saved herding histories
    res = pickle.loads(R.CACHE.read_bytes())
    for name in ("old_mix", "new_mix"):
        h = [r for r in R.pick(res[name], "herding") if r["ran"] is None]
        w = np.array([r["summary"]["worst"] for r in h]); k = np.array([r["summary"]["kurtosis"] for r in h])
        print(name, "worst <= -23.05:", int((w <= -23.05).sum()), "worst <= -14:", int((w <= -14).sum()),
              "kurtosis >= 25.2:", int((k >= 25.19).sum()), "max kurt", round(k.max(), 1), "min worst", round(w.min(), 1))
    (HERE / "extra_summary.json").write_text(json.dumps({str(k): v for k, v in out.items()}, indent=1))
