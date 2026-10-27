"""Shared test setup: make the project importable, common seeds, slow marker."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

SEEDS = [0, 1, 2, 3, 4]


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long simulations (skip with -m 'not slow')")
