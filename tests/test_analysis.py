"""Checks on the analysis scripts: portable, valid, and they still run."""
import ast
import importlib.util
import inspect

import pytest

from conftest import ROOT

SCRIPTS = sorted((ROOT / "scripts" / "analysis").glob("*.py"))


@pytest.mark.parametrize("path", SCRIPTS, ids=[p.name for p in SCRIPTS])
def test_no_hardcoded_home_folder(path):
    # absolute paths break on other computers and leak the username
    assert "/Users/" not in path.read_text()


@pytest.mark.parametrize("path", SCRIPTS, ids=[p.name for p in SCRIPTS])
def test_scripts_are_valid_python_with_a_docstring(path):
    assert ast.get_docstring(ast.parse(path.read_text()))


@pytest.mark.parametrize("path", SCRIPTS, ids=[p.name for p in SCRIPTS])
def test_every_script_is_listed_in_the_readme(path):
    assert f"`{path.name}`" in (ROOT / "scripts" / "analysis" / "README.md").read_text()


def load_script(name):
    """Import an analysis script as a module (without running its main)."""
    spec = importlib.util.spec_from_file_location(name[:-3], ROOT / "scripts" / "analysis" / name)
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


def test_longest_run_agrees_with_a_plain_loop():
    import sys
    import numpy as np
    sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
    from efficient_market_claims import longest_run

    def by_hand(mask):
        best = run = 0
        for m in mask:
            run = run + 1 if m else 0
            best = max(best, run)
        return best
    g = np.random.default_rng(0)
    for n in (0, 1, 7, 1000):
        for p in (0, 0.3, 0.9, 1):
            m = g.random(n) < p
            assert longest_run(m) == by_hand(m)
    assert longest_run([True, True, False, True]) == 2


def test_web_stress_test_reports_a_page_timeout_instead_of_hanging(monkeypatch):
    # TEST-1: a page that timed out inside part D raised (after up to 600 s per run) and took the
    # whole suite down; now each run is bounded and a timeout is one FAIL line, and part E is skipped
    from streamlit.testing.v1 import AppTest
    ws = load_script("web_stress_test.py")
    assert ws.PAGE_TIMEOUT <= 180

    def timed_out(self, *a, **k):
        raise RuntimeError(f"AppTest script run timed out after {ws.PAGE_TIMEOUT}(s)")
    monkeypatch.setattr(AppTest, "run", timed_out)
    ws.failures.clear()
    assert ws.part_d() is None
    assert len(ws.failures) == 1 and "timed out" in ws.failures[0]


def _square(x):
    return x * x


def test_web_stress_test_puts_the_real_main_module_back_so_pools_still_work():
    # TEST-2: Streamlit's AppTest installs the page as sys.modules["__main__"]; on macOS every later
    # multiprocessing Pool re-imported that page in its workers and the full suite hung forever
    import sys
    from multiprocessing import Pool
    from streamlit.testing.v1 import AppTest
    ws = load_script("web_stress_test.py")
    main = sys.modules["__main__"]
    at = AppTest.from_string("import streamlit as st\nst.write('hello')", default_timeout=60)
    ws.failures.clear()
    assert ws._run_page(at, "tiny page") and not ws.failures
    assert at.markdown[0].value == "hello"             # the page really ran
    assert sys.modules["__main__"] is main
    with Pool(2) as pool:                               # bounded wait: a regression fails, not hangs
        assert pool.map_async(_square, [1, 2, 3]).get(timeout=120) == [1, 4, 9]
