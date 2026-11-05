"""How often do the fitted settings run away (an endless bubble or crash)?

The fits use runs of 20,000-50,000 days; a runaway could still be rare but real
over longer horizons. Runs 40 held-out histories of 100,000 days for each setting.

Run from the project folder:  python3 analysis/stability.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from herding import calibrate, experiments


def main(days=100_000, n=40):
    """Count runaways among n held-out histories of `days` days per setting."""
    for kind in ("paper", "spx"):
        h = experiments.histories(calibrate.load(kind), calibrate.HELD_OUT_SEEDS[:n], days)
        bad = sum(x is None for x in h)
        print(f"{kind}: {bad} of {len(h)} histories of {days:,} days ran away")


if __name__ == "__main__":
    main()
