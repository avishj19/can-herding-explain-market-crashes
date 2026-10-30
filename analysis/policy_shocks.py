"""Policy experiment: does herding amplify a policy surprise?

A policy surprise (e.g. an unexpected rate hike) moves the fundamental value at
once. Each seed is run twice, with and without the shock on day SHOCK_DAY; the
difference in log price is the shock's causal effect (same random news both
times). Without chartists the price should simply move by the shock. With
herding, chartists may chase the move and overshoot.

Run from the project folder:  python3 analysis/policy_shocks.py  [--regime]
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from herding import calibrate, model  # noqa: E402

DAYS, SHOCK_DAY, SEEDS = 1300, 1000, range(4000, 4040)
SIZES = [-0.10, -0.05, -0.01, 0.01, 0.05, 0.10]


def response(args):
    """Price path difference (shock - no shock) for days SHOCK_DAY .. SHOCK_DAY+60."""
    seed, size, herding = args
    p = calibrate.load()
    a = model.simulate_full(DAYS, herding, p, seed=seed)
    b = model.simulate_full(DAYS, herding, p, seed=seed, shocks={SHOCK_DAY: size})
    d = (b["lp"] - a["lp"])[SHOCK_DAY:SHOCK_DAY + 61]
    dfrac = (b["frac"] - a["frac"])[SHOCK_DAY:SHOCK_DAY + 61]
    return d, dfrac


def main():
    print("Effect of a one-time policy surprise, average over 40 paired seeds")
    print(f"{'shock':>7}{'model':>14}{'worst move':>12}{'overshoot':>11}{'after 60 d':>12}"
          f"{'extra chartists':>17}{'seeds overshoot':>17}")
    with Pool() as pool:
        for size in SIZES:
            for herding in (False, True):
                res = pool.map(response, [(s, size, herding) for s in SEEDS])
                d = np.array([r[0] for r in res])
                df = np.array([r[1] for r in res])
                target = np.log1p(size)
                extreme = d.min(axis=1) if size < 0 else d.max(axis=1)   # furthest move
                over = extreme / target                                   # 1.0 = no overshoot
                label = "herding" if herding else "no chartists"
                print(f"{size:>+7.0%}{label:>14}{np.mean(np.expm1(extreme)):>+12.2%}"
                      f"{np.mean(over):>10.2f}x{np.mean(np.expm1(d[:, -1])):>+12.2%}"
                      f"{np.mean(df.max(axis=1)):>+17.1%}{int(np.sum(over > 1.05)):>11}/40")


if __name__ == "__main__" and "--regime" not in sys.argv:
    main()


def by_regime(size=-0.10, n=200):
    """Does the overshoot depend on how many chartists there are when the shock hits?
    Shock days are spread over many seeds; compare calm vs crowded markets."""
    p = calibrate.load()
    rows = []
    for seed in range(5000, 5000 + n):
        a = model.simulate_full(DAYS, True, p, seed=seed)
        b = model.simulate_full(DAYS, True, p, seed=seed, shocks={SHOCK_DAY: size})
        d = (b["lp"] - a["lp"])[SHOCK_DAY:SHOCK_DAY + 61]
        rows.append((a["frac"][SHOCK_DAY - 1], d.min() / np.log1p(size)))
    rows = np.array(rows)
    print(f"\n{size:+.0%} shock, {n} seeds, by chartist share on the day before the shock:")
    for lo, hi in [(0, 0.1), (0.1, 0.3), (0.3, 1.01)]:
        m = rows[(rows[:, 0] >= lo) & (rows[:, 0] < hi)]
        if len(m):
            print(f"  chartists {lo:.0%}-{min(hi, 1):.0%}: {len(m):>3} seeds, "
                  f"mean overshoot {m[:, 1].mean():.2f}x, max {m[:, 1].max():.2f}x")


if __name__ == "__main__" and "--regime" in sys.argv:
    by_regime()
