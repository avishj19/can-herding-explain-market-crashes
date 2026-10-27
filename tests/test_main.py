"""Tests for main.py: the end-to-end comparison runs and saves its figure."""
import numpy as np
import pytest

from conftest import ROOT
from herding import calibrate
import main
from herding import model
from herding import stats


def test_main_runs_end_to_end(tmp_path, capsys):
    out = tmp_path / "draft.png"
    real, herd, base = main.main(out=out, histories=3, days=3000)
    text = capsys.readouterr().out
    assert out.exists() and out.stat().st_size > 50_000           # a real figure
    assert "-20.5%" in text and "23.0 standard deviations" in text  # Black Monday line
    assert len(herd) == len(base) == 3000 and len(real) > 19_000
    assert "typical history: seed" in text and "no-chartist base case" in text


def test_main_table_has_every_fact_and_columns(tmp_path, capsys):
    main.main(out=tmp_path / "f.png", histories=3, days=3000)
    text = capsys.readouterr().out
    lines = text.split("simulated histories")[1].splitlines()[1:]
    assert all(c in lines[0] for c in ["S&P 500", "calibrated", "S&P pct", "paper set"])
    for fact in stats.stylized_facts(np.random.default_rng(0).standard_normal(500)):
        assert any(l.startswith(fact) for l in lines), fact


def test_plotted_history_is_the_median_mismatch_one(tmp_path):
    # main must plot the typical history, not a cherry-picked best one
    _, herd, _ = main.main(out=tmp_path / "f.png", histories=5, days=3000)
    runs, facts = main.ensemble(calibrate.load(), range(1, 6), days=3000)
    target = stats.stylized_facts(main.data.spx_returns().values)
    i = calibrate.representative(facts, target)
    assert np.array_equal(herd, runs[i][0])


def test_ensemble_table_flags_unusual_facts():
    target = {"x": 10.0}
    sims = [{"x": float(v)} for v in range(20)]          # S&P at the 50th percentile
    assert "unusual" not in main.ensemble_table(target, sims, sims)
    assert "unusual" in main.ensemble_table({"x": 99.0}, sims, sims)


def test_run_model_drops_burn_in():
    r, f = main.run_model(seed=4, days=500)
    full, _ = model.simulate(500 + main.BURN, seed=4)
    assert np.array_equal(r, full[main.BURN:]) and len(f) == 500
