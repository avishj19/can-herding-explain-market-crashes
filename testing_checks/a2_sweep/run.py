"""Independent re-check of web/numbers.json (testing_checks/a2_sweep).

    env -u PYTHONDONTWRITEBYTECODE python3 testing_checks/a2_sweep/run.py

Part A  rerun the saved seeds (1000-1039, three mixes, 10-seed sweep) and compare with numbers.json.
Part B  fresh seeds 700000-700039 (mixes, crash check) and 710000-710039 (herding-strength sweep):
        seeds that no file in herding/, scripts/, tests/, web/ uses.
Part C  figures (testing_checks/a2_sweep/figures/).
Everything is cached in results.pkl; delete it to recompute (about 2 minutes on 12 cores).
Writes only inside testing_checks/a2_sweep/.
"""
import json
import math
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from herding import calibrate, explore as ex, market_data, model, stats  # noqa: E402

FRESH_SEEDS = list(range(700000, 700040))      # the mixes and the crash check
SWEEP_FRESH = list(range(710000, 710040))      # the herding-strength sweep
SAVED_SEEDS = list(calibrate.HELD_OUT_SEEDS)
SAVED_SWEEP_SEEDS = SAVED_SEEDS[:10]
GRID = [round(0.2 * i, 1) for i in range(11)]
FINE = [round(0.9 + 0.05 * i, 2) for i in range(13)]    # 0.90 .. 1.50, to find the runaway edge
CACHE = HERE / "results.pkl"
FIGS = HERE / "figures"


def job(args):
    """One history: summary, crash counts, big falls, trader mix. (mix, a1, seed, days, keep_r)"""
    mix, a1, seed, days, keep_r = args
    p, burn = ex.fitted_settings()
    paths = {}
    r, err = ex.run_mix(p, mix, a1, days, seed, burn, paths=paths)
    out = {"mix": mix, "a1": a1, "seed": seed}
    if r is None:
        out.update(ran=ex.runaway_day(err, burn), summary=None)
        try:   # the mix up to the day it ran away (model_market rebuilds it with the guard off)
            mm = ex.model_market(p, burn, mix, a1, p["a2"], p["v1"], days, seed)
            out.update(z_run=float(mm["z"].mean()), x_run=float(np.abs(mm["x"]).mean()))
        except ValueError:
            out.update(z_run=None, x_run=None)
        return out
    out.update(ran=None, summary=ex.summary(r), counts=ex.crash_counts(r), big=ex.big_falls(r),
               z=float(paths["z"].mean()), x=float(np.abs(paths["x"]).mean()))
    if keep_r:
        out["r"] = r
    return out


def run_all():
    p, burn = ex.fitted_settings()
    days = len(market_data.spx_returns())
    a1 = p["a1"]
    res = {"days": days, "burn": burn, "a1": a1}
    with Pool() as pool:
        t0 = time.time()
        # Part A: saved seeds, every mix; saved sweep seeds at every grid point + fitted a1
        jobs = [(m, a1, s, days, True) for m in ex.MIXES for s in SAVED_SEEDS]
        res["old_mix"] = pool.map(job, jobs, chunksize=1)
        a1s = sorted(set(GRID + [a1]))
        res["old_sweep"] = pool.map(job, [("herding", a, s, days, False) for a in a1s for s in SAVED_SWEEP_SEEDS],
                                    chunksize=1)
        print(f"part A done {time.time() - t0:.0f} s", flush=True)
        # Part B: fresh seeds
        res["new_mix"] = pool.map(job, [(m, a1, s, days, True) for m in ex.MIXES for s in FRESH_SEEDS], chunksize=1)
        a1b = sorted(set(GRID + [a1] + FINE))
        res["new_sweep"] = pool.map(job, [("herding", a, s, days, False) for a in a1b for s in SWEEP_FRESH],
                                    chunksize=1)
        print(f"part B done {time.time() - t0:.0f} s", flush=True)
    return res


# ---------------------------------------------------------------- helpers
def pick(rows, mix):
    return [r for r in rows if r["mix"] == mix]


def col(rows, key):
    return np.array([r["summary"][key] if r["summary"] else np.nan for r in rows], dtype=float)


def band(v):
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    return (float(np.median(v)), *[float(x) for x in np.percentile(v, [5, 95])]) if v.size else (np.nan,) * 3


def mw(new, old):
    a, b = np.asarray(new, float), np.asarray(old, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan
    return float(sps.mannwhitneyu(a, b, alternative="two-sided").pvalue)


def f(x, n=2):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{n}f}"


def verdict(new_med, old_med, lo, hi, p):
    inside = lo <= new_med <= hi
    if inside and (np.isnan(p) or p >= 0.05):
        return "reproduced"
    if inside:
        return "shifted (inside saved 90% band, MW p<0.05)"
    return "contradicts (outside saved 90% band)"


def sp_numbers():
    s = ex.summary(market_data.spx_returns().values)
    s["big_per_1000"] = 1000 * s["big_days"] / s["days"]
    return s


# ---------------------------------------------------------------- Part A
def part_a(res):
    out = []
    saved = json.loads((ROOT / "web/numbers.json").read_text())
    chk = saved["checks"]
    old = res["old_mix"]
    maxd = {}

    def upd(name, a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        bad = ~((a == b) | (np.isnan(a) & np.isnan(b)))
        d = np.where(bad, np.abs(a - b), 0.0)
        maxd[name] = (float(np.nanmax(d)) if d.size else 0.0, int(np.isnan(d).sum() + bad.sum() * 0))

    f_ = pick(old, "fundamentalists")
    for k in ("kurtosis", "tail", "memory"):
        a = col(f_, k)
        b = np.array([np.nan if v is None else v for v in chk["fundamentalists"][k]], float)
        maxd[f"fundamentalists {k}"] = float(np.nanmax(np.abs(a - b)))
    h = pick(old, "herding")
    for k in ("tail", "kurtosis", "memory", "worst"):
        a = col(h, k)
        b = np.array([np.nan if v is None else v for v in chk["herding"][k]], float)
        maxd[f"herding {k}"] = float(np.nanmax(np.abs(a - b)))
    gone = lambda rows: [r["ran"] for r in rows]
    same_gone = {"herding": gone(h) == chk["herding"]["runaway_day"],
                 "chartists": gone(pick(old, "chartists")) == chk["chartists"]["runaway_day"]}
    surv = [r for r in h if r["ran"] is None]
    fall_eq = [r["counts"] for r in surv] == saved["crashes"]["falls"]
    big_eq = all(np.array_equal(r["big"], b) for r, b in zip(surv, saved["crashes"]["big_falls"]))
    # sweep
    srows = saved["sweep"]["rows"]
    sd = {"tail": 0.0, "memory": 0.0, "kurtosis": 0.0}
    n_cells = n_bad_run = 0
    key = {(round(r["a1"], 9), r["seed"]): r for r in srows}
    for r in res["old_sweep"]:
        s = key[(round(r["a1"], 9), r["seed"])]
        n_cells += 1
        rd = s["runaway_day"]
        if (rd is None) != (r["ran"] is None) or (rd is not None and int(rd) != r["ran"]):
            n_bad_run += 1
            continue
        if r["summary"]:
            for k in sd:
                sd[k] = max(sd[k], abs(r["summary"][k] - s[k]))
    return {"maxdiff": maxd, "runaway_same": same_gone, "crash_falls_equal": fall_eq, "big_falls_equal": big_eq,
            "sweep_cells": n_cells, "sweep_runaway_mismatch": n_bad_run, "sweep_maxdiff": sd,
            "n_survivors": len(surv)}


# ---------------------------------------------------------------- crash tables
def nums_from(rows):
    surv = [r for r in rows if r["ran"] is None]
    return {"crashes": {"sizes": list(ex.CRASH_SIZES), "days": [r["summary"]["days"] for r in surv],
                        "falls": [r["counts"] for r in surv], "big_falls": [r["big"] for r in surv]}}


def boot_years(rows, size, n=2000, seed=0):
    """Middle 90% (bootstrap over histories) of pooled years between falls of `size` sd."""
    surv = [r for r in rows if r["ran"] is None]
    i = ex.CRASH_SIZES.index(size)
    c = np.array([r["counts"][i] for r in surv], float)
    g = np.random.default_rng(seed)
    tot = np.array([c[g.integers(0, len(c), len(c))].sum() for _ in range(n)])
    yrs = np.where(tot > 0, len(c) * surv[0]["summary"]["days"] / ex.YEAR_DAYS / np.maximum(tot, 1), np.inf)
    return [float(x) for x in np.percentile(yrs, [5, 95])]


# ---------------------------------------------------------------- Part B
def part_b(res, spx):
    sp = sp_numbers()
    saved = json.loads((ROOT / "web/numbers.json").read_text())
    old_h = pick(res["old_mix"], "herding")
    new_h = pick(res["new_mix"], "herding")
    tab = {}

    def big1000(rows):
        return np.array([1000 * r["summary"]["big_days"] / r["summary"]["days"] if r["summary"] else np.nan
                         for r in rows])

    for mix in ex.MIXES:
        o, n = pick(res["old_mix"], mix), pick(res["new_mix"], mix)
        rowset = {}
        for k in ("tail", "kurtosis", "memory", "worst"):
            rowset[k] = (col(o, k), col(n, k))
        rowset["big_per_1000"] = (big1000(o), big1000(n))
        tab[mix] = {"measures": {}, "runaways_old": sum(r["ran"] is not None for r in o), "runaways_new": sum(r["ran"] is not None for r in n),
                    "n": len(n)}
        for k, (a, b) in rowset.items():
            if np.all(np.isnan(b)) or np.all(np.isnan(a)):
                continue
            om, ol, oh = band(a)
            nm, nl, nh = band(b)
            p = mw(b, a)
            tab[mix]["measures"][k] = {"old": (om, ol, oh), "new": (nm, nl, nh), "p": p,
                                       "inside": bool(ol <= nm <= oh), "verdict": verdict(nm, om, ol, oh, p),
                                       "n_old": int(np.sum(~np.isnan(a))), "n_new": int(np.sum(~np.isnan(b)))}
    # crash check
    on, nn = nums_from(old_h), nums_from(new_h)
    cc_old = ex.crash_check(on, spx)
    cc_new = ex.crash_check(nn, spx)
    cc_saved = ex.crash_check(saved, spx)
    crash = {"old_rerun": cc_old, "new": cc_new, "saved_json": cc_saved,
             "table_old": ex.crash_table(on, spx), "table_new": ex.crash_table(nn, spx),
             "ci8_old": boot_years(old_h, 8), "ci8_new": boot_years(new_h, 8),
             "bm_count_new": cc_new["bm_count"], "bm_count_old": cc_old["bm_count"],
             "hist_with_8_old": int(sum(1 for r in old_h if r["ran"] is None and r["counts"][5] > 0)),
             "hist_with_8_new": int(sum(1 for r in new_h if r["ran"] is None and r["counts"][5] > 0))}
    # sweep
    sweeps = {}
    for name in ("old_sweep", "new_sweep"):
        rows = res[name]
        d = {}
        for a1 in sorted(set(r["a1"] for r in rows)):
            g = [r for r in rows if r["a1"] == a1]
            surv = [r for r in g if r["ran"] is None]
            d[a1] = {"n": len(g), "runaway": len(g) - len(surv),
                     "tail": band(col(g, "tail")), "kurtosis": band(col(g, "kurtosis")),
                     "memory": band(col(g, "memory")),
                     "z": float(np.mean([r["z"] for r in surv])) if surv else np.nan,
                     "x": float(np.mean([r["x"] for r in surv])) if surv else np.nan,
                     "z_run": float(np.mean([r["z_run"] for r in g if r["ran"] is not None and r.get("z_run") is not None]))
                     if any(r["ran"] is not None and r.get("z_run") is not None for r in g) else np.nan,
                     "x_run": float(np.mean([r["x_run"] for r in g if r["ran"] is not None and r.get("x_run") is not None]))
                     if any(r["ran"] is not None and r.get("x_run") is not None for r in g) else np.nan,
                     "tails": [float(v) for v in col(g, "tail")]}
        sweeps[name] = d
    return {"sp": sp, "tab": tab, "crash": crash, "sweeps": sweeps}


def where_sp_tail(sw, target=2.9172):
    """Linear interpolation of the a1 where the median tail crosses the S&P's (sorted by a1, survivors only)."""
    a = [(k, v["tail"][0]) for k, v in sorted(sw.items()) if not np.isnan(v["tail"][0]) and v["n"] - v["runaway"] >= 5]
    out = []
    for (x0, y0), (x1, y1) in zip(a, a[1:]):
        if (y0 - target) * (y1 - target) <= 0 and y0 != y1:
            out.append(x0 + (target - y0) * (x1 - x0) / (y1 - y0))
    return out, a


# ---------------------------------------------------------------- Part C: figures
def figures(res, B, spx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from herding import plots
    from herding.plots import INK, MUTED, NEWS, REAL, SPX_MODEL, GRID as GRIDC
    FIGS.mkdir(exist_ok=True)
    nf = [r for r in pick(res["new_mix"], "fundamentalists")]
    nh = [r for r in pick(res["new_mix"], "herding") if r["ran"] is None]
    # ---- fig04-like
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    a = ax[0]
    plots.ccdf(a, np.concatenate([r["r"] for r in nf]), "no chartists", NEWS)
    plots.ccdf(a, np.concatenate([r["r"] for r in nh]), "herding (S&P settings)", SPX_MODEL)
    plots.ccdf(a, spx, "S&P 500", REAL)
    plots.gaussian_ccdf(a)
    a.set_xlim(0.05, 40)
    a.set_ylim(1e-6, 1.5)
    a.set_xlabel("|return| / standard deviation")
    a.set_ylabel("share of days with a bigger move")
    a.set_title("a. Tails (fresh seeds)")
    a.legend(loc="lower left")
    groups = [("no chartists", nf, NEWS), ("herding", nh, SPX_MODEL)]
    for a, key, title, ref in [(ax[1], "kurtosis", "b. Excess kurtosis", 0.0), (ax[2], "tail", "c. Tail exponent (Hill, top 5%)", 3.0)]:
        ticks = []
        for i, (name, rows, color) in enumerate(groups):
            v = np.array([r["summary"][key] for r in rows])
            jit = np.random.default_rng(i).uniform(-0.18, 0.18, len(v))
            a.plot(i + jit, v, "o", color=color, ms=4, alpha=0.75)
            a.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color=INK, lw=1.6)
            ticks.append(f"{name}\nmedian {np.median(v):.2f}")
        real = B["sp"][key]
        a.axhline(real, color=REAL, lw=1.2, label=f"S&P 500: {real:.2f}")
        a.axhline(ref, color=MUTED, ls="--", lw=1, label="bell curve: 0" if ref == 0 else "inverse cubic law: 3")
        a.set_xticks(range(2), ticks)
        a.set_xlim(-0.5, 1.5)
        a.set_title(title)
        a.legend(loc="upper right")
    ax[1].set_yscale("symlog", linthresh=1)
    ax[1].set_ylim(-0.5, None)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    plots.save(fig, FIGS / "fig04_fresh_proposal_checks.png", "Fresh-seed redraw of Fig 4 (seeds 700000-700039)",
               "40 fresh histories of 19,329 days per group; bars are medians.")
    # ---- dial
    sw = B["sweeps"]["new_sweep"]
    old = B["sweeps"]["old_sweep"]
    a1s = sorted(sw)
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.4))
    ax = ax.ravel()
    spx_s = B["sp"]

    def line(a, key, idx=0, lo=None, hi=None):
        ys = [sw[x][key][0] if isinstance(sw[x][key], tuple) else sw[x][key] for x in a1s]
        a.plot(a1s, ys, "o-", color=SPX_MODEL, ms=4, label="fresh seeds (40)")
        if isinstance(sw[a1s[0]][key], tuple):
            a.fill_between(a1s, [sw[x][key][1] for x in a1s], [sw[x][key][2] for x in a1s], color=SPX_MODEL, alpha=0.15)
            oa = sorted(old)
            a.plot(oa, [old[x][key][0] for x in oa], "s--", color=MUTED, ms=3, lw=1, label="saved sweep seeds (10)")

    ax[0].plot(a1s, [sw[x]["runaway"] / sw[x]["n"] for x in a1s], "o-", color=SPX_MODEL, ms=4, label="fresh")
    oa = sorted(old)
    ax[0].plot(oa, [old[x]["runaway"] / old[x]["n"] for x in oa], "s--", color=MUTED, ms=3, lw=1, label="saved seeds")
    ax[0].set_title("a. Share of histories that ran away")
    line(ax[1], "kurtosis")
    ax[1].set_title("b. Excess kurtosis (median, 5-95%)")
    ax[1].set_yscale("symlog", linthresh=1)
    ax[1].axhline(spx_s["kurtosis"], color=REAL, lw=1.2, label=f"S&P 500: {spx_s['kurtosis']:.1f}")
    line(ax[2], "tail")
    ax[2].set_title("c. Tail exponent, Hill (median, 5-95%)")
    ax[2].axhline(spx_s["tail"], color=REAL, lw=1.2, label=f"S&P 500: {spx_s['tail']:.2f}")
    ax[2].axhline(3, color=MUTED, ls=":", lw=1)
    line(ax[3], "memory")
    ax[3].set_title("d. Volatility memory (median, 5-95%)")
    ax[3].axhline(spx_s["memory"], color=REAL, lw=1.2, label=f"S&P 500: {spx_s['memory']:.2f}")
    ax[4].plot(a1s, [sw[x]["z"] for x in a1s], "o-", color=SPX_MODEL, ms=4, label="survivors, whole history")
    ax[4].plot(a1s, [sw[x]["z_run"] for x in a1s], "^:", color="#d97706", ms=4, label="runaways, up to the runaway day")
    ax[4].set_title("e. Average share of trend chasers (z)")
    ax[5].plot(a1s, [sw[x]["x"] for x in a1s], "o-", color=SPX_MODEL, ms=4, label="survivors")
    ax[5].plot(a1s, [sw[x]["x_run"] for x in a1s], "^:", color="#d97706", ms=4, label="runaways, to runaway day")
    ax[5].set_title("f. Average |mood| of chartists (|x|)")
    for a in ax:
        a.axvline(res["a1"], color=INK, ls=":", lw=1.2)
        a.set_xlabel("herding strength a1 (dotted: fitted)")
        a.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    plots.save(fig, FIGS / "fig10_fresh_herding_dial.png", "Fresh-seed redraw of Fig 10: the herding dial (seeds 710000-710039)",
               "Fitted S&P settings with only a1 changed; 40 fresh histories per point; band = middle 90% of survivors.")
    # ---- crash chart
    cr = B["crash"]
    fig, a = plt.subplots(figsize=(8.6, 5.2))
    sizes = list(ex.CRASH_SIZES)
    a.plot(sizes, [r["bell"] for r in cr["table_new"]], "k:o", label="bell curve")
    for tab, lab, c, ls in ((cr["table_new"], "model, fresh seeds", SPX_MODEL, "-"),
                            (cr["table_old"], "model, saved seeds (rerun)", MUTED, "--")):
        pts = [(r["size"], r["model"]) for r in tab if not math.isinf(r["model"])]
        a.plot(*zip(*pts), ls, marker="o", color=c, label=lab)
    pts = [(r["size"], r["sp500"]) for r in cr["table_new"] if not math.isinf(r["sp500"])]
    a.plot(*zip(*pts), "-o", color=REAL, label="S&P 500")
    a.axhline(50, color=GRIDC, lw=1)
    a.axhline(1.38e10, color=GRIDC, ls=":", lw=1)
    a.text(3, 1.38e10, "age of the universe", fontsize=8, va="bottom")
    a.text(3, 50, "within 50 years", fontsize=8, va="bottom")
    a.set_yscale("log")
    a.set_ylim(0.1, 1e14)
    a.set_xlabel("daily fall (sd)")
    a.set_ylabel("years between such falls")
    a.legend()
    plots.save(fig, FIGS / "fig_crash_wait_fresh.png", "Crash wait, fresh seeds vs saved seeds",
               "Pooled over survivors. A size with no fall in the pooled histories is left off the line.")


# ---------------------------------------------------------------- report numbers
def dump(res, A, B):
    def conv(o):
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, float) and math.isinf(o):
            return "inf"
        return o
    (HERE / "numbers_summary.json").write_text(json.dumps(conv({"A": A, "B": B}), indent=1))


def main():
    spx = market_data.spx_returns().values
    if CACHE.exists():
        res = pickle.loads(CACHE.read_bytes())
    else:
        res = run_all()
        CACHE.write_bytes(pickle.dumps(res))
    A = part_a(res)
    B = part_b(res, spx)
    dump(res, A, B)
    figures(res, B, spx)
    print(json.dumps(A, indent=1, default=str))
    for m, t in B["tab"].items():
        print(m, "runaways old/new", t["runaways_old"], t["runaways_new"])
        for k, v in t["measures"].items():
            print(f"  {k:13s} old {[round(x, 3) for x in v['old']]} new {[round(x, 3) for x in v['new']]} p={v['p']:.3f} {v['verdict']}")
    print("S&P", B["sp"])
    for name, sw in B["sweeps"].items():
        print(name)
        for a1, v in sorted(sw.items()):
            print(f"  a1={a1:5.2f} run {v['runaway']:2d}/{v['n']} tail {v['tail'][0]:.2f} kurt {v['kurtosis'][0]:.2f} mem {v['memory'][0]:.3f} z {v['z']:.3f} x {v['x']:.3f} z_run {v['z_run']:.3f} x_run {v['x_run']:.3f}")
        print("  crossing S&P tail:", where_sp_tail(sw)[0])
    c = B["crash"]
    for k in ("old_rerun", "new", "saved_json"):
        v = c[k]
        print(k, {a: v[a] for a in ("size", "decades", "black_monday", "passed", "worst_sd", "bm_bell", "bm_model", "bm_count")}, "8sd years", v["row"]["model"], v["row"]["model_count"], v["row"]["histories"], "of", v["row"]["of"])
    print("ci8", c["ci8_old"], c["ci8_new"])


if __name__ == "__main__":
    main()
