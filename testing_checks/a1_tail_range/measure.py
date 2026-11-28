"""Tail-exponent measurements for the S&P 500 and the model (run from anywhere):
env -u PYTHONDONTWRITEBYTECODE python3 testing_checks/a1_tail_range/measure.py
Writes results.json next to this file. Read-only on the rest of the project."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from herding import explore as ex, stats  # noqa: E402

out = {}
r = ex.log_returns(ex.saved_prices("S&P 500"))
r = r[np.isfinite(r)]
out["n_days"] = len(r); out["first"] = str(r.index[0].date()); out["last"] = str(r.index[-1].date())

def hill_side(x, tail):
    """Hill on one side (x already one-signed magnitudes)."""
    return stats.hill_alpha(x, tail)

def table(rr):
    v = np.asarray(rr, float)
    t = {}
    for q in (0.025, 0.05, 0.10):
        t[f"hill_abs_{q}"] = stats.hill_alpha(v, q)
        # one-sided: Hill on |positive| and |negative| separately, top q of that side
        pos = v[v > 0]; neg = -v[v < 0]
        t[f"hill_pos_{q}"] = stats.hill_alpha(pos, q)
        t[f"hill_neg_{q}"] = stats.hill_alpha(neg, q)
    t["regression_top30"] = stats.alpha_regression(v, 0.30)
    t["regression_top10"] = stats.alpha_regression(v, 0.10)
    t["regression_top5"] = stats.alpha_regression(v, 0.05)
    t["n"] = len(v)
    return t

out["spx_full"] = table(r.values)
out["spx_1950_1986"] = table(r[:"1986-12-31"].values)
out["spx_1988_2026"] = table(r["1988-01-01":].values)
# bootstrap spread for Hill on the full sample (iid resample; ignores clustering -> optimistic)
rng = np.random.default_rng(0); v = r.values
for q in (0.025, 0.05):
    bs = [stats.hill_alpha(rng.choice(v, len(v)), q) for _ in range(500)]
    out[f"spx_bootstrap_iid_hill_{q}"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
# block bootstrap (blocks of 50 days keep volatility clustering)
for q in (0.025, 0.05):
    bs = []
    for _ in range(500):
        idx = stats.block_indices(len(v), 50, rng)
        bs.append(stats.hill_alpha(v[idx], q))
    out[f"spx_blockboot50_hill_{q}"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
# 10 equal chunks of S&P: spread of the estimate across ~7.7-year windows
chunks = np.array_split(v, 10)
out["spx_10chunks_hill_0.05"] = [float(stats.hill_alpha(c, 0.05)) for c in chunks]
out["spx_10chunks_hill_0.025"] = [float(stats.hill_alpha(c, 0.025)) for c in chunks]

# model: 40 held-out histories (recipe from web/numbers.json)
nj = json.load(open(ROOT / "web/numbers.json"))
rec = nj["recipe"]
p, burn = ex.fitted_settings(ROOT / rec["settings"])
runs = []
for s in rec["seeds"]:
    rr, err = ex.run_mix(p, "herding", p["a1"], rec["days"], s, rec["burn"])
    if rr is not None:
        runs.append(np.asarray(rr))
out["model_n_ok"] = len(runs)
def summ(a):
    a = np.array(a); return {"median": float(np.median(a)), "mean": float(a.mean()), "sd": float(a.std(ddof=1)),
            "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90)), "min": float(a.min()), "max": float(a.max())}
for q in (0.025, 0.05, 0.10):
    out[f"model_hill_abs_{q}"] = summ([stats.hill_alpha(x, q) for x in runs])
out["model_hill_pos_0.05"] = summ([stats.hill_alpha(x[x > 0], 0.05) for x in runs])
out["model_hill_neg_0.05"] = summ([stats.hill_alpha(-x[x < 0], 0.05) for x in runs])
out["model_regression_top30"] = summ([stats.alpha_regression(x, 0.30) for x in runs])
out["saved_numbers_json_tail"] = summ(nj["checks"]["herding"]["tail"])
# does a model history's own noise match the S&P's? share of model histories inside S&P block-bootstrap interval
lo, hi = out["spx_blockboot50_hill_0.05"]
out["model_share_inside_spx_block_ci_0.05"] = float(np.mean([lo <= stats.hill_alpha(x, 0.05) <= hi for x in runs]))
out["fraction_model_in_2.5_3.5"] = float(np.mean([2.5 <= stats.hill_alpha(x, 0.05) <= 3.5 for x in runs]))
out["fraction_model_in_2.6_3.0"] = float(np.mean([2.6 <= stats.hill_alpha(x, 0.05) <= 3.0 for x in runs]))
json.dump(out, open(Path(__file__).with_name("results.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
