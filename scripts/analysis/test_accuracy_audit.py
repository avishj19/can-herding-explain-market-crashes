"""How accurate is each noisy test? For every recipe in tests/helpers.py: run it on 200 seeds
(never the test seeds 0-4), measure the bias (average minus the exact answer) and the sd, and
print the tolerance |bias| + 5 sd that the tests use (tests/helpers.py, TOLERANCE).

Run:  python3 scripts/analysis/test_accuracy_audit.py [seeds]     (about 2 minutes on 8 cores)
With --write, the TOLERANCE table in tests/helpers.py is replaced by the new one.
To see how much of its tolerance every test uses:
    PYTHONPATH=tests python3 -m pytest tests/ -q -p tolerance_recorder
"""
import math
import re
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import numpy as np

import helpers

SIGMAS = 5


def round_up(x, digits=2):
    """x rounded UP to `digits` significant digits (a tolerance must never shrink by rounding)."""
    if x == 0:
        return 0.0
    scale = 10 ** (math.floor(math.log10(abs(x))) - digits + 1)
    return math.ceil(x / scale) * scale


def audit(n_seeds=200):
    """{"recipe.check": tolerance} from n_seeds seeds, with a printed table."""
    jobs = [(name, s) for name in helpers.RECIPES for s in range(1000, 1000 + n_seeds)]
    with Pool() as pool:
        out = pool.map(helpers.run_recipe, jobs, chunksize=2)
    values = {}
    for name, _, checks in out:
        for k, v in checks.items():
            if k in helpers.RECIPES[name][1]:          # only numbers with an exact answer
                values.setdefault((name, k), []).append(v)
    table = {}
    print(f"{'check':<34}{'exact':>11}{'bias':>11}{'sd':>10}{'tolerance':>11}")
    for (name, k), v in values.items():
        v = np.array(v)
        exact = helpers.RECIPES[name][1][k]
        bias, sd = v.mean() - exact, v.std(ddof=1)
        tol = round_up(abs(bias) + SIGMAS * sd)
        table[f"{name}.{k}"] = tol
        print(f"{name + '.' + k:<34}{exact:11.5g}{bias:+11.2e}{sd:10.2e}{tol:11.3g}")
    return table


def write(table):
    p = ROOT / "tests" / "helpers.py"
    body = "TOLERANCE = {\n" + "".join(f"    {k!r}: {v!r},\n" for k, v in table.items()) + "}\n"
    s = re.sub(r"TOLERANCE = \{.*?\}\n", body, p.read_text(), count=1, flags=re.S)
    p.write_text(s)


def main(days=None, n=200, write_table=False):
    """Audit with n seeds. `days` is unused (each recipe fixes its own sample size); it is
    here so every analysis script can be called the same way."""
    t = audit(n)
    if write_table:
        write(t)
        print("wrote TOLERANCE in tests/helpers.py")
    return t


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--write"]
    main(n=int(args[0]) if args else 200, write_table="--write" in sys.argv)
