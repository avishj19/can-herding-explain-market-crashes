"""Checks on the analysis scripts: portable, valid, and they still run."""
import ast
import importlib.util
import inspect

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


def load_script(name):
    """Import an analysis script as a module (without running its main)."""
    spec = importlib.util.spec_from_file_location(name[:-3], ROOT / "analysis" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", [p.name for p in SCRIPTS])
def test_main_takes_size_arguments(name):
    # so the tests can run each script on tiny sizes instead of the full ones
    params = inspect.signature(load_script(name).main).parameters
    assert "days" in params and "n" in params


@pytest.mark.slow
@pytest.mark.parametrize("name", [p.name for p in SCRIPTS])
def test_analysis_script_still_runs(name):
    # tiny sizes: this checks the code path works, not the numbers it prints
    load_script(name).main(days=3000, n=2)
