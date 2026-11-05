"""Checks on the analysis scripts: portable, valid, and they still run."""
import ast
import subprocess
import sys

import pytest

from conftest import ROOT

SCRIPTS = sorted((ROOT / "analysis").glob("*.py"))


@pytest.mark.parametrize("path", SCRIPTS, ids=[p.name for p in SCRIPTS])
def test_no_hardcoded_home_folder(path):
    # absolute paths break on other computers and leak the username
    assert "/Users/" not in path.read_text()


@pytest.mark.parametrize("path", SCRIPTS, ids=[p.name for p in SCRIPTS])
def test_scripts_are_valid_python_with_a_docstring(path):
    assert ast.get_docstring(ast.parse(path.read_text()))


@pytest.mark.parametrize("path", SCRIPTS, ids=[p.name for p in SCRIPTS])
def test_every_script_is_listed_in_the_readme(path):
    assert f"`{path.name}`" in (ROOT / "analysis" / "README.md").read_text()


@pytest.mark.slow
@pytest.mark.parametrize("name", [p.name for p in SCRIPTS])
def test_analysis_script_still_runs(name):
    out = subprocess.run([sys.executable, str(ROOT / "analysis" / name)], capture_output=True,
                         text=True, timeout=900, cwd=ROOT)
    assert out.returncode == 0, out.stderr[-1500:]
