"""Can herding explain market crashes? The Lux-Marchesi model vs the S&P 500.

Run:  python3 main.py    -> prints the comparison table, saves figures/draft.png
Uses the S&P-calibrated settings (settings/spx.json) and held-out seeds only.
"""
import numpy as np

from herding import calibrate, experiments, plots, stats
from herding import market_data as data

FIGURES = data.ROOT / "figures"
SEEDS = calibrate.HELD_OUT_SEEDS[:20]        # never used by the calibration


def table(target, sims):
    """One line per measure: S&P value, model median with its 5-95% range, and the
    S&P's percentile among the model histories (flagged if outside 5-95)."""
    lines = [f"{'measure':<11}{'S&P 500':>10}{'model median (5-95% range)':>34}{'S&P pct':>9}"]
    for k in target:
        v = np.array([f[k] for f in sims])
        lo, med, hi = np.percentile(v, [5, 50, 95])
        pct = experiments.percentile(target[k], v)
        flag = "" if 5 <= pct <= 95 else "  <- unusual"
        lines.append(f"{k:<11}{target[k]:>10.4f}{med:>12.4f} ({lo:8.4f} to {hi:8.4f})"
                     f"{pct:>8.0f}%{flag}")
    return "\n".join(lines)


def main(p=None, out=FIGURES / "draft.png", seeds=SEEDS, days=None):
    p = calibrate.load("spx") if p is None else p
    real = data.spx_returns()
    bm = data.black_monday(real)
    real = real.values
    days = len(real) if days is None else days
    print(f"S&P 500: {len(real):,} daily returns, 1950-2026")
    print(f"Black Monday 1987: {np.expm1(bm):.1%} in one day = "
          f"{abs(bm) / real.std():.1f} standard deviations")
    with experiments.Pool() as pool:
        runs = experiments.histories(p, seeds, days, keep=("r", "z"), pool=pool)
        base = experiments.returns(p, seeds[:1], days, chartists=False, pool=pool)[0]
    ok = [h for h in runs if h is not None]
    ok_seeds = [s for s, h in zip(seeds, runs) if h is not None]
    if len(ok) < len(runs):
        print(f"warning: {len(runs) - len(ok)} of {len(runs)} histories ran away")
    target = stats.fingerprint(real)
    sims = [stats.fingerprint(h["r"]) for h in ok]
    print(f"\n{len(ok)} held-out model histories of {days:,} days each:")
    print(table(target, sims))
    print(f"\nno chartists: kurtosis {stats.kurtosis(base):.3f} (bell curve = 0)")
    i = experiments.typical(sims, target)
    plots.comparison(real, ok[i]["r"], ok[i]["z"], base, out)
    print(f"saved {out} (the typical history: seed {ok_seeds[i]})")
    return real, ok[i]["r"], base


if __name__ == "__main__":
    main()
