"""Extra real numbers for the redesign mock-up (docs/redesign/index.html): writes more.js, and copies the
figures (figs/) and Plotly's chart library (plotly.min.js) next to it. Both copies are git-ignored."""
import json, math, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from herding import explore as ex, kirman

OUT = Path(__file__).parent
def L(a): return [None if (v is None or (isinstance(v, float) and not math.isfinite(v))) else float(v) for v in np.asarray(a, dtype=float)]
def thin(x, y, n=1200):
    i = ex.thin_indices(np.asarray(y, float), n); return [x[j] for j in i], [float(y[j]) for j in i]

close = ex.saved_prices("S&P 500"); r = ex.log_returns(close); s = ex.summary(r.values)
p, burn = ex.fitted_settings(); nums = ex.web_numbers(); days = len(r); seed = int(nums["recipe"]["seeds"][0])
mixes = ex.run_mixes(p, burn, float(p["a1"]), p["a2"], p["v1"], days, seed)
sums = ex.mix_summaries(mixes)
o = {}
# Price view: drawdown, all dates
dd = ex.drawdown_series(close)
dates = [d.strftime("%Y-%m-%d") for d in close.index]
o["dd"] = dict(zip(("x", "y"), thin(dates, dd.values)))
# Histogram view
edges = np.arange(-8, 8.01, 0.4)
centers = (edges[:-1] + edges[1:]) / 2
def H(rr):
    c_, d_ = ex.histogram(rr, empty=np.nan)
    return {"x": L(c_), "y": L(d_)}
o["bell_density"] = {"x": L(centers), "y": L(ex.bell_density(centers))}
o["hist"] = {"spx": H(r.values), "herd": H(mixes["herding"][0]), "fund": H(mixes["fundamentalists"][0])}
# Memory view
o["memory"] = {"lags": list(ex.LAGS), "spx": L(ex.memory_curve(r.values)[1]), "herd": L(ex.memory_curve(mixes["herding"][0])[1]),
               "fund": L(ex.memory_curve(mixes["fundamentalists"][0])[1])}
# Watch a crash: whole history and wildest year for herding; fundamentalists whole history
def mmjson(mm, a=None, b=None):
    d = np.arange(len(mm["price"]))
    if a is None:
        i = ex.thin_indices(np.log(mm["price"]), 1500)
    else:
        i = np.arange(a, b + 1)
    return {"day": [int(k) for k in i], "price": L(mm["price"][i]), "value": L(mm["value"][i]), "z": L(mm["z"][i]),
            "crowd": L((mm["z"] * mm["x"])[i])}
mmh = ex.model_market(p, burn, "herding", float(p["a1"]), p["a2"], p["v1"], days, seed)
mmf = ex.model_market(p, burn, "fundamentalists", float(p["a1"]), p["a2"], p["v1"], days, seed)
a, b = ex.zoom_window(mmh["r"], "burst")
o["mm"] = {"herding_all": mmjson(mmh), "herding_burst": mmjson(mmh, a, b), "fund_all": mmjson(mmf)}
o["news"] = {m: ex.news_and_moves(mm_) for m, mm_ in (("herding", mmh), ("fund", mmf))}
# Model vs market table
t = ex.mix_table("S&P 500", s, r, sums, mixes)
o["table"] = {"cols": [str(c) for c in t.columns], "rows": [[str(i)] + [None if pd.isna(v) else (float(v) if isinstance(v, (int, float, np.number)) else str(v)) for v in t.loc[i]] for i in t.index]}
# Big days with stories
events = ex.day_events()
big = ex.big_day_list(r)
o["bigdays"] = []
for d, row in big.iterrows():
    text, src = ex.day_story(d, row["move"], row["sd"], events)
    o["bigdays"].append({"date": d.strftime("%Y-%m-%d"), "move": float(row["move"]), "sd": float(row["sd"]), "story": text,
                         "sources": [x for x in (src or []) if str(x).startswith("http")]})
# Insight sentence
shown = {m: sums[m] for m in ex.MIXES if sums[m] is not None}
o["insight"] = ex.insight("S&P 500", s, ex.closest_mix(s, shown))
# Every run on one chart: the sweep band
band = ex.sweep_band(nums["sweep"]["rows"])
o["band"] = json.loads(band.to_json(orient="records"))
# Checks details
o["checks"] = {"seeds": nums["recipe"]["seeds"], "kurt": nums["checks"]["fundamentalists"]["kurtosis"],
               "tails": nums["checks"]["herding"]["tail"], "bell": ex.bell_check(nums["checks"]["fundamentalists"]["kurtosis"], nums["recipe"]["days"]),
               "table": json.loads(ex.check_table(nums, s).reset_index().to_json(orient="records")),
               "facts": ex.fact_table(nums, r.values), "crash": ex.crash_table(nums, r.values)}
for f in o["checks"]["facts"]:
    f.pop("key", None)
kir = {}
for eps in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1):
    k = kirman.simulate(100, eps, 0.1, 20_000_000, seed=1)
    sim, exact, tv = ex.kirman_match(k, 100, eps, 0.1)
    st_, ct = ex.kirman_path(k, every=10_000)
    kir[str(eps)] = {"steps": (np.asarray(st_) / 1e6).tolist(), "count": np.asarray(ct).tolist(), "sim": L(sim), "exact": L(exact), "tv": float(tv)}
o["kirman"] = {"eps": kir, "threshold": float(kirman.herding_threshold(100, 0.1))}
# Compare: every saved market
o["markets"] = json.loads(ex.market_table().to_json(orient="records"))
# Findings: the paper comparison, the figures
o["paper"] = ex.paper_comparison(nums)
o["party"] = []          # the party study was removed (28 Nov 2026); the mock-up's party table stays empty
entries = ex.gallery_entries((ROOT / "figures" / "GALLERY.md").read_text())
(OUT / "figs").mkdir(exist_ok=True)
for e in entries:
    src = ROOT / "figures" / e["file"]
    if src.exists():
        shutil.copy(src, OUT / "figs" / e["file"])
o["gallery"] = entries
import plotly
shutil.copy(Path(plotly.__file__).parent / "package_data" / "plotly.min.js", OUT / "plotly.min.js")
(OUT / "more.js").write_text("window.MORE = " + json.dumps(o, default=lambda v: v.isoformat() if hasattr(v, "isoformat") else float(v)) + ";\n")
print("ok", len(o["bigdays"]), len(o["markets"]), len(entries), o["insight"][:80])
