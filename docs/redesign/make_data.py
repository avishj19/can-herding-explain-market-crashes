"""Real numbers for the redesign mock-up (docs/redesign/index.html): writes data.js next to it."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from herding import explore as ex, kirman

def thin(x, y, n=1200):
    i = ex.thin_indices(np.asarray(y, float), n)
    return [x[j] for j in i], [float(y[j]) for j in i]

close = ex.saved_prices("S&P 500")
r = ex.log_returns(close)
s = ex.summary(r.values)
p, burn = ex.fitted_settings()
nums = ex.web_numbers()
days = len(r)
seed = int(nums["recipe"]["seeds"][0])
out = {"spx": {"tail": s["tail"], "kurt": s["kurtosis"], "memory": s["memory"], "big": s["big_days"],
               "bell_big": s["bell_big_days"], "last": float(close.iloc[-1]),
               "chg": float(close.iloc[-1] / close.iloc[-2] - 1), "date": close.index[-1].strftime("%d %b %Y")}}
dates = [d.strftime("%Y-%m-%d") for d in close.index]
out["price"] = dict(zip(("x", "y"), thin(dates, close.values)))
big = ex.big_day_list(r)
out["big"] = {"x": [d.strftime("%Y-%m-%d") for d in big.index], "y": [float(close.loc[d]) for d in big.index],
              "move": [float(m) for m in big["move"]]}
xs, ys = ex.tail_curve(r.values)
out["tail_spx"] = {"x": list(map(float, xs)), "y": list(map(float, ys))}
grid = np.geomspace(0.1, 30, 120)
out["bell"] = {"x": list(map(float, grid)), "y": list(map(float, ex.bell_tail(grid)))}
zs = ex.moves_in_sd(r.values)
out["bursts_spx"] = dict(zip(("x", "y"), thin(dates[1:], zs)))
runs = {}
for a1 in (0.9, 1.0, round(float(p["a1"]), 2), 1.2, 1.4):
    m = ex.run_mixes(p, burn, a1, p["a2"], p["v1"], days, seed)
    herd, fund = m["herding"][0], m["fundamentalists"][0]
    entry = {"a1": a1, "ran": m["herding"][1]}
    if herd is not None:
        sm = ex.summary(herd)
        tx, ty = ex.tail_curve(herd)
        zh = ex.moves_in_sd(herd)
        bx, by = thin(list(range(len(zh))), zh)
        entry.update(tail=sm["tail"], kurt=sm["kurtosis"], memory=sm["memory"], big=sm["big_days"],
                     tail_x=list(map(float, tx)), tail_y=list(map(float, ty)), burst_x=bx, burst_y=by,
                     score=ex.scorecard(s, sm))
    runs[str(a1)] = entry
fx, fy = ex.tail_curve(m["fundamentalists"][0])
out["tail_fund"] = {"x": list(map(float, fx)), "y": list(map(float, fy))}
out["runs"] = runs
out["fit_a1"] = round(float(p["a1"]), 2)
mm = ex.model_market(p, burn, "herding", float(p["a1"]), p["a2"], p["v1"], days, seed)
a, b = ex.zoom_window(mm["r"], "crash")
out["crash"] = {"first": a, "day": list(range(a, b + 1)), "price": mm["price"][a:b + 1].tolist(),
                "value": mm["value"][a:b + 1].tolist(), "z": mm["z"][a:b + 1].tolist(),
                "crowd": (mm["z"] * mm["x"])[a:b + 1].tolist()}
k = kirman.simulate(100, 0.002, 0.1, 20_000_000, seed=1)
sim, exact, tv = ex.kirman_match(k, 100, 0.002, 0.1)
st, ct = ex.kirman_path(k)
out["kirman"] = {"steps": (np.asarray(st) / 1e6).tolist(), "count": np.asarray(ct).tolist(),
                 "sim": list(map(float, sim)), "exact": list(map(float, exact)), "tv": float(tv)}
cc = ex.crash_check(nums, r.values)
out["crash_years"] = {"model8": cc["row"]["model"], "bm_model": cc["bm_model"], "spx_years": days / 252}
facts = ex.fact_table(nums, r.values)
out["checks"] = {"tail_median": ex.spread(nums["checks"]["herding"]["tail"])["median"],
                 "kurt_median": ex.spread(nums["checks"]["fundamentalists"]["kurtosis"])["median"],
                 "herd_facts": sum(f["herding_pass"] > 20 for f in facts),
                 "garch_facts": sum(f["garch_pass"] > 20 for f in facts)}
# the herding tail on one shared grid for every run, so the chart can morph between them
ok = [v for v in runs.values() if "tail_x" in v]
top = min(max(v["tail_x"]) for v in ok)
g = np.geomspace(0.1, top, 90)
for v in ok:
    v["morph_y"] = np.exp(np.interp(np.log(g), np.log(v["tail_x"]), np.log(v["tail_y"]))).tolist()
out["morph_x"] = g.tolist()
# the Story's extra steps
gone = m["chartists"][1]
if gone is not None and gone >= 1:
    st_ = ex.runaway_story(p, seed, gone, burn)
    i = ex.thin_indices(st_["gap"], 900)
    out["runaway"] = {"day": [int(j) for j in i], "z": [float(st_["z"][j]) for j in i], "gap": [float(st_["gap"][j]) for j in i],
                      "gone": int(gone), "calm": int(st_["calm_days"]), "cap": st_["gap_cap"]}
out["runaway_count"] = ex.runaway_count(nums, "chartists")
out["sweep"] = [ex.sweep_at(nums, a) for a in sorted({r["a1"] for r in nums["sweep"]["rows"]})]
nm = ex.news_and_moves(mm)
out["news"] = nm
fs = ex.summary(m["fundamentalists"][0])
out["fund"] = {"kurt": fs["kurtosis"], "big": fs["big_days"]}
(Path(__file__).parent / "data.js").write_text("window.MOCK = " + json.dumps(out, default=float) + ";\n")
print("ok", {k: v.get("tail") for k, v in runs.items()}, "kirman tv", round(tv, 4), "crash days", a, b)
