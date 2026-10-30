"""Smoke tests for make_figures.py: each figure function draws and saves a real PNG."""
import numpy as np
import pytest
from matplotlib.image import imread

import make_figures as mf
from herding import calibrate, experiments, market_data, model


@pytest.fixture
def out(tmp_path, monkeypatch):
    monkeypatch.setattr(mf, "FIG", tmp_path)
    return tmp_path


def _png(path):
    assert path.exists() and imread(path).shape[0] > 500     # a real, full-size figure


def test_kirman_figure(out):
    mf.fig_kirman()
    _png(out / "fig5_kirman.png")


def test_ensemble_figures(out):
    p = calibrate.load()
    spx = market_data.spx_returns().values
    cal = experiments.ensemble(p, range(600, 604), days=3000)
    paper = experiments.ensemble(model.P, range(600, 604), days=3000)
    base = experiments.history(p, seed=600, days=3000, herding=False)
    pct = mf.fig_fingerprint(spx, cal, paper)
    assert set(pct) == set(calibrate.TARGETS) and all(0 <= v <= 100 for v in pct.values())
    mf.fig_tails(spx, cal, base)
    mf.fig_memory(spx, cal, paper)
    for name in ["fig1_fingerprint.png", "fig2_tails.png", "fig3_memory.png"]:
        _png(out / name)


def test_mechanism_figure(out):
    mf.fig_mechanism(calibrate.load())
    _png(out / "fig4_mechanism.png")


def test_figures_use_held_out_seeds():
    assert set(mf.HELD_OUT).isdisjoint(calibrate.CALIBRATION_SEEDS)


def test_importing_make_figures_does_not_change_global_style():
    # Regression: make_figures used to set plt.rcParams globally, which cropped
    # figures made elsewhere (e.g. plots.comparison) once it had been imported
    import matplotlib.pyplot as plt
    assert plt.rcParams["savefig.bbox"] != "tight"
