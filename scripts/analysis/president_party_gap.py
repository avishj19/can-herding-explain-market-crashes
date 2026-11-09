"""Is the S&P 500 different under Democratic and Republican presidents, and is the gap
bigger than a model with no politics produces by chance at the same dates?

Run from the project folder:  python3 scripts/analysis/president_party_gap.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np

from herding import calibrate, experiments, politics


def main(days=None, n=40):
    """Both S&P samples (or only their last `days` days) against n model histories."""
    p = calibrate.load("spx")
    with experiments.Pool() as pool:
        for market in ("spx_1928", "spx"):
            r, dem, real, nulls, pct = politics.compare(market, p, calibrate.HELD_OUT_SEEDS[:n], pool, days)
            print(f"\n{market}: {r.index[0].year}-{r.index[-1].year}, {dem.sum():,} Democratic and "
                  f"{(~dem).sum():,} Republican days")
            print(f"  return gap D-R: {real['mean'] * r.std():+.1%} a year (log returns, before inflation; "
                  f"S&P volatility {r.std() * np.sqrt(politics.DAYS_PER_YEAR):.0%} a year)")
            print(f"  luck yardsticks: {len(nulls['model'])} model histories (no politics), "
                  f"{len(nulls['shifted'])} shifted party calendars on the real returns")
            for k, name in politics.MEASURES.items():
                cells = "".join(f"   {w} {np.percentile([s[k] for s in nulls[w]], 5):+6.2f} to "
                                f"{np.percentile([s[k] for s in nulls[w]], 95):+6.2f} (S&P at {pct[w][k]:3.0f}%)"
                                for w in nulls)
                print(f"  {name:<45} S&P {real[k]:+7.2f}{cells}")


if __name__ == "__main__":
    main()
