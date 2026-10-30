"""Can herding explain market crashes? Lux-Marchesi model vs the S&P 500.

Run:  python3 main.py   -> prints the comparison, saves figures/draft.png
Test: python3 -m pytest tests/ -q          (quick: add  -m "not slow")
"""
import numpy as np

from herding import calibrate, experiments, model, plots, stats
from herding import market_data as data
from herding.experiments import BURN, DAYS

FIGURES = data.ROOT / "figures"
HISTORIES = 20                  # the model has randomness: judge it over many histories
SEEDS = range(600, 600 + HISTORIES)   # held out: never used by the calibration


def run_model(p=model.P, herding=True, seed=1, days=DAYS):
    """One history: returns and chartist share, after the burn-in."""
    r, frac = model.simulate(days + BURN, herding=herding, p=p, seed=seed)
    return r[BURN:], frac[BURN:]


def ensemble(p, seeds, days=DAYS, pool=None):
    """Returns (in parallel) and stylized facts of several simulated histories."""
    runs = experiments.ensemble(p, seeds, days, pool=pool)
    return runs, [stats.stylized_facts(r) for r in runs]


def ensemble_table(target, cal, paper):
    """S&P value, calibrated median [5-95% range], S&P percentile, paper median."""
    pct = calibrate.percentiles(target, cal, target)
    lines = [f"{'fact':<12}{'S&P 500':>9}{'calibrated (5-95% range)':>30}"
             f"{'S&P pct':>9}{'paper set':>11}"]
    for k in target:
        v = np.array([f[k] for f in cal])
        lo, med, hi = np.percentile(v, [5, 50, 95])
        flag = "" if 5 <= pct[k] <= 95 else " <- unusual"
        lines.append(f"{k:<12}{target[k]:>9.3f}{med:>12.3f} ({lo:7.3f} to {hi:7.3f})"
                     f"{pct[k]:>8.0f}%{np.median([f[k] for f in paper]):>11.3f}{flag}")
    return "\n".join(lines)


def main(p=None, out=FIGURES / "draft.png", histories=HISTORIES, days=DAYS):
    """Compare the S&P 500 with many histories of the calibrated model and the
    paper's original settings, plus the no-chartist base case; save a figure
    of the most typical calibrated history (median mismatch, not the best)."""
    p = calibrate.load() if p is None else p
    real = data.spx_returns()
    bm = data.black_monday(real)
    real = real.values
    print(f"S&P 500: {len(real)} daily returns")
    print(f"Black Monday 1987: {np.expm1(bm):.1%} (log return {bm:.3f}) = "
          f"{abs(bm) / real.std():.1f} standard deviations")
    seeds = SEEDS[:histories]
    with experiments.Pool() as pool:
        runs, cal = ensemble(p, seeds, days, pool)
        _, paper = ensemble(model.P, seeds, days, pool)
    target = stats.stylized_facts(real)
    print(f"\n{histories} simulated histories of {days} days each:")
    print(ensemble_table(target, cal, paper))
    i = calibrate.representative(cal, target)
    herd, frac = run_model(p, seed=seeds[i], days=days)    # rerun it to get its chartist share
    base, _ = run_model(p, herding=False, seed=0, days=days)
    print(f"no-chartist base case: kurtosis {stats.kurtosis(base):.3f} (bell curve = 0)")
    plots.comparison(real, herd, frac, base, out)
    print(f"\nSaved {out} (typical history: seed {seeds[i]})")
    return real, herd, base


if __name__ == "__main__":
    main()
