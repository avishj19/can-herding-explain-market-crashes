"""Tests for main.py: the end-to-end comparison runs and its table is right."""
import numpy as np
import pytest

import main
from herding import calibrate, experiments, stats


def test_table_flags_unusual_measures_by_percentile():
    sims = [{"x": float(v)} for v in range(20)]
    assert "unusual" not in main.table({"x": 10.0}, sims)       # 50th percentile
    assert "unusual" in main.table({"x": 99.0}, sims)           # above every history
    assert "unusual" in main.table({"x": -1.0}, sims)           # below every history


def test_table_has_one_line_per_measure_with_the_right_percentile():
    sims = [{"a": float(v), "b": 2.0 * v} for v in range(10)]
    text = main.table({"a": 4.5, "b": 100.0}, sims).splitlines()
    assert len(text) == 3 and text[1].startswith("a") and text[2].startswith("b")
    assert "50%" in text[1] and "100%" in text[2]


def test_main_uses_only_held_out_seeds():
    assert not set(main.SEEDS) & set(calibrate.TRAIN_SEEDS)


@pytest.mark.slow
def test_main_runs_end_to_end(tmp_path, capsys):
    if not calibrate.settings_path("spx").exists():
        pytest.skip("not fitted yet")
    out = tmp_path / "draft.png"
    real, shown, base = main.main(out=out, seeds=main.SEEDS[:3], days=3000)
    text = capsys.readouterr().out
    assert out.exists() and out.stat().st_size > 50_000
    assert "-20.5% in one day = 23.0 standard deviations" in text      # Black Monday
    for k in stats.FINGERPRINT:
        assert any(line.startswith(k) for line in text.splitlines()), k
    assert len(shown) == len(base) == 3000 and len(real) == 19329
    # the plotted history is the typical (median-mismatch) one, not the best
    p = calibrate.load("spx")
    runs = experiments.returns(p, main.SEEDS[:3], 3000)
    target = stats.fingerprint(real)
    i = experiments.typical([stats.fingerprint(r) for r in runs], target)
    assert np.array_equal(shown, runs[i])
