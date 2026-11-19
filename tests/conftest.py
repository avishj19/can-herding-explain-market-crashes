"""Shared test setup: make the project importable, common seeds, slow marker."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))     # main.py and make_figures.py
sys.path.insert(0, str(Path(__file__).resolve().parent))

SEEDS = [0, 1, 2, 3, 4]

# Precision rule (November 2026): numbers compared with pytest.approx and no explicit
# tolerance must agree to 12 significant digits, not pytest's default of 6. Our arithmetic is
# 64-bit (about 16 digits; checked against exact fractions and 50-digit mpmath, which agree to
# 1e-14), so a gap in the 7th-12th digit is a bug, not round-off. Tests that compare noisy
# estimates say their own tolerance, worked out as bias + 5 sd (docs/TEST_ACCURACY.md).
from _pytest.python_api import ApproxScalar

ApproxScalar.DEFAULT_RELATIVE_TOLERANCE = 1e-12


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long simulations (skip with -m 'not slow')")
    # The overnight stress test (tests/stress/overnight.py) reruns the suite with every seed
    # changed; a correct test must pass for any seed. These few are the exception on purpose:
    # each describes one particular documented history (a golden output, the day a known
    # seed runs away, a number quoted in the documents), so it is skipped in those reruns.
    config.addinivalue_line("markers", "pinned_seed: describes one particular random history")


@pytest.fixture(autouse=True)
def _keep_the_real_main_module():
    """Streamlit's AppTest installs the page it runs as sys.modules["__main__"] and never puts the
    real one back. On macOS a multiprocessing Pool starts its workers by re-importing __main__, so
    every Pool after an AppTest test re-ran app.py in its workers and hung the suite (TEST-2).
    Putting __main__ back after each test keeps one test's AppTest from breaking the next."""
    main = sys.modules.get("__main__")
    yield
    sys.modules["__main__"] = main
