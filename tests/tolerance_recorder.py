"""pytest plugin: record every pytest.approx comparison in the test suite (test, expected,
actual, error, allowed error), to see how much of its tolerance each test uses.

Run from the project folder:
    PYTHONPATH=tests python3 -m pytest tests/ -q -p tolerance_recorder
Writes scratch/tolerances.csv. It only records; it never changes a result.
"""
import csv
import math
from pathlib import Path

from _pytest import python_api

ROWS, CURRENT = [], {"test": None}
_orig_eq = python_api.ApproxScalar.__eq__


def _eq(self, actual):
    result = _orig_eq(self, actual)
    try:
        exp, act = complex(self.expected), complex(actual)
        if exp.imag == 0 and act.imag == 0 and math.isfinite(exp.real) and math.isfinite(act.real):
            ROWS.append((CURRENT["test"], exp.real, act.real, abs(act.real - exp.real),
                         float(self.tolerance), bool(result)))
    except (TypeError, ValueError, OverflowError):
        pass                                   # strings, None, ... are not numbers
    return result


python_api.ApproxScalar.__eq__ = _eq


def pytest_runtest_call(item):
    CURRENT["test"] = item.nodeid


def pytest_sessionfinish(session):
    out = Path(__file__).resolve().parents[1] / "scratch" / "tolerances.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test", "expected", "actual", "error", "allowed", "passed"])
        w.writerows(ROWS)
