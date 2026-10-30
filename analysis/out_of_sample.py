"""Out-of-sample test: does real data the calibration never saw look like a
typical history of the calibrated model?

Markets: S&P 500 since 1928 (Great Depression), Nasdaq-100 since 1985, and the
20 stocks with market beta near 1. For each, simulate 20 histories of the same
length and report where the real value falls (percentile; 5-95% = typical).

Run from the project folder:  python3 analysis/out_of_sample.py
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from herding import calibrate, experiments, market_data, stats  # noqa: E402

HISTORIES = 20


def ensemble(pool, p, days, first_seed):
    runs = experiments.ensemble(p, range(first_seed, first_seed + HISTORIES), days, pool=pool)
    return [stats.stylized_facts(r) for r in runs]


def report(name, real, sims):
    pct = calibrate.percentiles(real, sims, calibrate.TARGETS)
    unusual = [k for k, v in pct.items() if not 5 <= v <= 95]
    print(f"\n{name}: S&P-style fingerprint vs {HISTORIES} model histories")
    for k in calibrate.TARGETS:
        v = np.array([s[k] for s in sims])
        flag = "  <- unusual" if k in unusual else ""
        print(f"  {k:<12}{real[k]:>8.3f}   model {np.median(v):>7.3f} "
              f"({np.percentile(v, 5):7.3f} to {np.percentile(v, 95):7.3f})  pct {pct[k]:>4.0f}%{flag}")
    return unusual


def main():
    p = calibrate.load()
    with Pool() as pool:
        results = {}
        for name in ["spx_1928", "ndx"]:
            r = market_data.returns(name).values
            results[name] = report(name, stats.stylized_facts(r), ensemble(pool, p, len(r), 1000))
        lengths = {t: len(market_data.stock_returns(t)) for t in market_data.stock_tickers()}
        sims = ensemble(pool, p, int(np.median(list(lengths.values()))), 2000)
    print("\nStocks (market beta 0.9-1.1): measures where the stock looks unusual")
    counts = {k: 0 for k in calibrate.TARGETS}
    for t in market_data.stock_tickers():
        real = stats.stylized_facts(market_data.stock_returns(t).values)
        pct = calibrate.percentiles(real, sims, calibrate.TARGETS)
        bad = [k for k, v in pct.items() if not 5 <= v <= 95]
        for k in bad:
            counts[k] += 1
        print(f"  {t:<5} {len(bad)} unusual: {', '.join(bad) if bad else '-'}")
    print("\nHow many of the 20 stocks look unusual on each measure:")
    for k, c in counts.items():
        print(f"  {k:<12}{c:>3}")
    for name, bad in results.items():
        print(f"{name}: unusual on {len(bad)} of {len(calibrate.TARGETS)} measures {bad}")


if __name__ == "__main__":
    main()
