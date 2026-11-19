"""Make web/numbers.json: the model results the web app shows on its Checks, Story and Workspace pages.

    python3 scripts/web_numbers.py          (about a minute on all cores)

Same recipe as figures/RESULTS.md (scripts/make_figures.py, Fig 4): the S&P 500 fit, the 40 held-out
seeds 1000-1039 that the fit never saw, and histories as long as the S&P 500 record (19,329 days) after
the usual 1,000 burn-in days. So the site's numbers are the figures' numbers, not a second estimate.
tests/test_explore.py reruns some seeds and must get the saved numbers back exactly.
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from herding import calibrate, explore as ex, market_data  # noqa: E402

SWEEP_A1 = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
SWEEP_SEEDS = calibrate.HELD_OUT_SEEDS[:10]


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

    def col(mix, key):
        return [ex.summary(out[mix, s][0])[key] if out[mix, s][0] is not None else None for s in seeds]

    def gone(mix):
        return [None if out[mix, s][1] is None else ex.runaway_day(out[mix, s][1], burn) for s in seeds]

    d = {
        "recipe": {"settings": "settings/sp500_calibrated.json", "seeds": seeds, "days": days, "burn": burn,
                   "made_by": "scripts/web_numbers.py", "made": time.strftime("%Y-%m-%d")},
        "checks": {
            "fundamentalists": {k: col("fundamentalists", k) for k in ("kurtosis", "tail", "memory")},
            "herding": {k: col("herding", k) for k in ("tail", "kurtosis", "memory", "worst")} |
                       {"runaway_day": gone("herding")},
            "chartists": {"runaway_day": gone("chartists")},
        },
        "sweep": {"seeds": list(SWEEP_SEEDS), "days": days,
                  "rows": json.loads(sw.to_json(orient="records"))},
    }
    path = ex.WEB_NUMBERS
    path.write_text(json.dumps(d, indent=1) + "\n")
    h = ex.spread(d["checks"]["herding"]["tail"])
    k = ex.spread(d["checks"]["fundamentalists"]["kurtosis"])
    print(f"saved {path.relative_to(ROOT)} in {time.perf_counter() - t0:.0f} s")
    print(f"herding tail median {h['median']:.2f} over {h['n']} histories (RESULTS.md: 3.10)")
    print(f"no-chartist kurtosis median {k['median']:.2f} (RESULTS.md: 0.01)")
    print("no-herding runaways: {} of {}".format(*ex.runaway_count(d, "chartists")))


if __name__ == "__main__":
    main()
