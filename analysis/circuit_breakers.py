"""Policy experiment: do circuit breakers make crashes rarer, or just delay them?

Each of 20 seeds is run twice (paired): with no breakers, and with the US rules
(pause 15 min at -7% and -13%, close the day at -20%). The levels are scaled by
the model's daily volatility relative to the S&P 500's, so "7%" means the same
number of typical days in both.

Run from the project folder:  python3 analysis/circuit_breakers.py
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from herding import calibrate, experiments, market_data, model, stats  # noqa: E402
from herding.experiments import BURN, DAYS, worst_fall  # noqa: E402

SEEDS = range(3000, 3020)


def measures(r, crash):
    return dict(worst_1d=worst_fall(r, 1), worst_5d=worst_fall(r, 5), worst_20d=worst_fall(r, 20),
                crash_days=int(np.sum(r <= np.log(1 - crash))), share_4sd=stats.tail_share(r, 4),
                alpha_1pct=stats.hill_alpha(r, 0.01), ac_abs_20=stats.acf(np.abs(r), 20)[0],
                std=float(r.std()))


def run(args):
    seed, breakers, crash = args
    o = model.simulate_full(DAYS + BURN, p=calibrate.load(), seed=seed, breakers=breakers)
    m = measures(o["r"][BURN:], crash)
    m["halt_days"] = int(np.sum(o["halts"][BURN:] > 0))
    m["closed_days"] = int(np.sum(o["halts"][BURN:] == 3))
    return m


def main():
    p = calibrate.load()
    rules, scale = experiments.scaled_breakers(
        np.median([experiments.history(p, s).std() for s in SEEDS[:5]]), market_data.spx_returns().std())
    levels = rules["levels"]
    print(f"model volatility is {scale:.2f}x the S&P's -> breaker levels "
          + ", ".join(f"{x:.1%}" for x in levels))
    with Pool() as pool:
        off = pool.map(run, [(s, None, levels[0]) for s in SEEDS])
        on = pool.map(run, [(s, rules, levels[0]) for s in SEEDS])
    print(f"\n{'measure':<12}{'no breakers':>14}{'breakers':>12}{'change':>10}"
          f"   seeds: lower / same / higher with breakers")
    for k in off[0]:
        a = np.array([o[k] for o in off], float)
        b = np.array([o[k] for o in on], float)
        change = f"{b.mean() / a.mean() - 1:+.1%}" if a.mean() else "n/a"
        print(f"{k:<12}{a.mean():>14.4f}{b.mean():>12.4f}{change:>10}"
              f"   {int(np.sum(b < a)):>2} / {int(np.sum(b == a)):>2} / {int(np.sum(b > a)):>2}")
    print(f"\n'crash day' = a daily fall beyond {levels[0]:.1%} (the scaled 7% level)")


if __name__ == "__main__":
    main()
