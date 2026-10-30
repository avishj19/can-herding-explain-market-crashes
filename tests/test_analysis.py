"""Checks on the analysis scripts (they are exploration, but must stay portable)."""
import ast
from pathlib import Path

import pytest

from conftest import ROOT

SCRIPTS = sorted((ROOT / "analysis").glob("*.py"))


@pytest.mark.parametrize("path", SCRIPTS, ids=[p.name for p in SCRIPTS])
def test_no_hardcoded_home_folder(path):
    # Bug fix: absolute paths break on other computers and leak the username
    assert "/Users/" not in path.read_text()


@pytest.mark.parametrize("path", SCRIPTS, ids=[p.name for p in SCRIPTS])
def test_scripts_are_valid_python(path):
    ast.parse(path.read_text())


QUICK = ["fresh.py", "lag2.py", "a1.py", "probe.py", "audit.py", "knobs.py", "regime.py",
         "lags.py", "holdout.py", "holdout2.py", "typical.py", "out_of_sample.py",
         "circuit_breakers.py", "policy_shocks.py"]


@pytest.mark.slow
@pytest.mark.parametrize("name", QUICK)
def test_analysis_script_still_runs(name):
    # Old scripts rot when the program improves (e.g. new guardrails); catch it here
    import subprocess, sys
    out = subprocess.run([sys.executable, str(ROOT / "analysis" / name)], capture_output=True,
                         text=True, timeout=300, cwd=ROOT)
    assert out.returncode == 0, out.stderr[-1500:]


def test_every_script_is_in_the_run_list_or_known_long():
    long = {"stress_tests.py"}                      # ~20 s; run by hand
    assert {p.name for p in SCRIPTS} == set(QUICK) | long
