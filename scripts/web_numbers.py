"""Make web/numbers.json: the model results the web app shows on its Checks, Story, Test program and Findings pages.

    python3 scripts/web_numbers.py          (about a minute on all cores)

Same recipe as figures/RESULTS.md (scripts/make_figures.py, Fig 4): the S&P 500 fit, the 40 held-out
seeds 1000-1039 that the fit never saw, and histories as long as the S&P 500 record (19,329 days) after
the usual 1,000 burn-in days. So the site's numbers are the figures' numbers, not a second estimate.
The Findings page's comparison with Lux & Marchesi (1999) uses the recipe of Fig 2: the paper-matched
settings, the first 8 held-out seeds, 100,000 days each.
tests/test_explore.py reruns some seeds and must get the saved numbers back exactly.
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from herding import calibrate, experiments, explore as ex, garch, market_data, stats  # noqa: E402

SWEEP_A1 = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
SWEEP_SEEDS = calibrate.HELD_OUT_SEEDS[:10]
PAPER_SEEDS = calibrate.HELD_OUT_SEEDS[:8]      # as Fig 2 (scripts/make_figures.py)
PAPER_DAYS = 100_000


def main():
    t0 = time.perf_counter()
    p, burn = ex.fitted_settings()
    days = len(market_data.spx_returns())
    seeds = list(calibrate.HELD_OUT_SEEDS)
    with Pool() as pool:
        jobs = [(p, m, p["a1"], days, s, burn) for m in ex.MIXES for s in seeds]
        out = dict(zip([(m, s) for m in ex.MIXES for s in seeds], pool.starmap(ex.run_mix, jobs)))
        a1s = sorted(set(SWEEP_A1 + [p["a1"]]))
        sw = ex.sweep(p, a1s, SWEEP_SEEDS, days, burn, pool=pool)
        paper_runs = experiments.returns(calibrate.load("paper"), PAPER_SEEDS, PAPER_DAYS, pool=pool)
        paper_fp = pool.map(stats.fingerprint, paper_runs)

    def col(mix, key):
        return [ex.summary(out[mix, s][0])[key] if out[mix, s][0] is not None else None for s in seeds]

    def gone(mix):
        return [None if out[mix, s][1] is None else ex.runaway_day(out[mix, s][1], burn) for s in seeds]

    herd = [out["herding", s][0] for s in seeds if out["herding", s][0] is not None]   # runaways left out
    # the textbook checklist (explore.FACTS) for the herding model and for a formula with no traders:
    # GJR-GARCH fitted to the S&P 500, simulated on the same seeds and length (herding/garch.py)
    gp = garch.fit(market_data.spx_returns().values)
    gar = [garch.simulate(gp, days, seed=s) for s in seeds]
    hf = [ex.fact_values(r) for r in herd]
    gf = [ex.fact_values(r) for r in gar]
    d = {
        "recipe": {"settings": "settings/sp500_calibrated.json", "seeds": seeds, "days": days, "burn": burn,
                   "made_by": "scripts/web_numbers.py", "made": time.strftime("%Y-%m-%d")},
        "checks": {
            "fundamentalists": {k: col("fundamentalists", k) for k in ("kurtosis", "tail", "memory")},
            "herding": {k: col("herding", k) for k in ("tail", "kurtosis", "memory", "worst")} |
                       {"runaway_day": gone("herding"),
                        # like for like with the S&P 500 without Black Monday (Checks, check 5)
                        "kurtosis_wo_worst": [ex.kurtosis_without_worst(out["herding", s][0])
                                              if out["herding", s][0] is not None else None for s in seeds]},
            "chartists": {"runaway_day": gone("chartists")},
        },
        # the crash check: per held-out herding history, its falls of each size and every fall >= 8 sd
        "crashes": {"sizes": list(ex.CRASH_SIZES), "days": [len(r) for r in herd],
                    "falls": [ex.crash_counts(r) for r in herd], "big_falls": [ex.big_falls(r) for r in herd]},
        "facts": {"garch_params": gp, "garch_seeds": seeds,
                  "herding": {k: [f[k] for f in hf] for k in ex.FACTS},
                  "garch": {k: [f[k] for f in gf] for k in ex.FACTS}},
        "sweep": {"seeds": list(SWEEP_SEEDS), "days": days,
                  "rows": json.loads(sw.to_json(orient="records"))},
        "paper": {"settings": "settings/paper_matched.json", "seeds": list(PAPER_SEEDS), "days": PAPER_DAYS,
                  **{k: [f[k] for f in paper_fp] for k in ("alpha_reg", "H", "H_abs", "kurtosis", "kurt_25")}},
    }
    path = ex.WEB_NUMBERS
    path.write_text(json.dumps(d, indent=1) + "\n")
    h = ex.spread(d["checks"]["herding"]["tail"])
    k = ex.spread(d["checks"]["fundamentalists"]["kurtosis"])
    print(f"saved {path.relative_to(ROOT)} in {time.perf_counter() - t0:.0f} s")
    print(f"herding tail median {h['median']:.2f} over {h['n']} histories (RESULTS.md: 3.10)")
    print(f"no-chartist kurtosis median {k['median']:.2f} (RESULTS.md: 0.01)")
    print("no-herding runaways: {} of {}".format(*ex.runaway_count(d, "chartists")))
    for row in ex.paper_comparison(d):
        print(f"paper-matched {row['key']}: {row['ours']:.2f} +- {row['ours_sd']:.2f} (paper {row['paper']})")
    for row in ex.fact_table(d, market_data.spx_returns().values):
        print(f"{row['name']:<32} S&P {'yes' if row['sp500_pass'] else 'no ':3}  herding {row['herding_pass']:2d}/{row['of']}"
              f"  GARCH {row['garch_pass']:2d}/{row['of']}")
    cc = ex.crash_check(d, market_data.spx_returns().values)
    print(f"model: {cc['size']}-sd falls every {cc['row']['model']:.0f} years; Black Monday size every "
          f"{cc['bm_model']:.0f} years (bell curve {cc['bm_bell']:.1e})")
    print("(RESULTS.md: tail 2.62, H 0.47, H|r| 0.81 +- 0.03)")


if __name__ == "__main__":
    main()
