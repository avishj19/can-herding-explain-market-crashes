"""Can "panic" give the model the S&P 500's skew (falls bigger than rises)?

    python3 scripts/analysis/panic_skew.py          (about a minute on all cores)

The rebuilt model has no skew: its falls and rises are mirror images, while the S&P 500's big falls are
bigger than its big rises. Bouchaud, Matacz & Potters (2001, Phys. Rev. Lett. 87, 228701) explain the
"leverage effect" in indexes with panic: after falls, traders react more strongly. Here that is
model.simulate(panic=...): a falling trend pushes chartists to pessimism harder than a rising one pushes
them to optimism (U1 + panic / v1 * min(trend, 0); 0 = SPEC.md exactly).

For each panic weight, 10 held-out seeds of the S&P fit, S&P-length histories. Two skew measures:
skewness of the standardized returns, and the 1% quantile's size over the 99% quantile's (1.00 = mirror
images; above 1, the biggest falls are bigger than the biggest rises).

Result (Nov 2026): panic 0 -> 1.00, panic 10 (about 77 times the fitted trend weight a2 = 0.13) -> 1.03; the S&P 500
is 1.09 (skewness -0.95 against the model's -0.05 to -0.34). Panic in the chartists' mood alone does not
give the market its skew: it only tilts who is a pessimist, and the fundamentalists pull the price back
just as hard either way. A real fix would need an asymmetry in how the price moves, not only in mood.
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from herding import calibrate, explore as ex, market_data, model, stats  # noqa: E402

PANICS = (0.0, 1.0, 2.0, 4.0, 7.0, 10.0)
SEEDS = calibrate.HELD_OUT_SEEDS[:10]


def skew_measures(r):
    """(skewness of the standardized returns, -1% quantile / 99% quantile)."""
    z = stats.standardize(r)
    lo, hi = np.quantile(z, [0.01, 0.99])
    return float(np.mean(z ** 3)), float(-lo / hi)


def one(job):
    panic, seed, p, burn, days = job
    try:
        out = model.simulate(p, days + burn, seed=seed, panic=panic)
    except model.RunawayError:
        return panic, None
    r = out["r"][burn:]
    return panic, (*skew_measures(r), float(stats.hill_alpha(r, 0.05)))


def main(panics=PANICS, seeds=SEEDS, days=None, n=None):
    """days: history length (default: the S&P 500's); n: how many seeds (default: all of SEEDS)."""
    seeds = seeds[:n] if n else seeds
    p, burn = ex.fitted_settings()
    spx = market_data.spx_returns().values
    days = days or len(spx)
    s, q = skew_measures(spx)
    print(f"S&P 500: skewness {s:+.2f}, 1%/99% quantile ratio {q:.2f}")
    # threads, not processes: the model's loop lets go of Python's lock (nogil), so they run side by side,
    # and nothing has to be pickled (a script loaded by its path, as the tests do, can't be)
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        res = list(pool.map(one, [(pv, sd, p, burn, days) for pv in panics for sd in seeds]))
    rows = {}
    for pv in panics:
        ok = np.array([v for k, v in res if k == pv and v is not None])
        gone = sum(1 for k, v in res if k == pv and v is None)
        rows[pv] = (np.median(ok, axis=0) if len(ok) else None, gone)
        if len(ok):
            m = rows[pv][0]
            print(f"panic {pv:4.1f}: skewness {m[0]:+.2f}, quantile ratio {m[1]:.2f}, tail {m[2]:.2f}, "
                  f"ran away {gone} of {len(seeds)}")
        else:
            print(f"panic {pv:4.1f}: every history ran away")
    return rows


if __name__ == "__main__":
    main()
