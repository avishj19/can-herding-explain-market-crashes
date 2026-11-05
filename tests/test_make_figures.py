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
    default = experiments.ensemble(model.P, range(600, 604), days=3000)
    base = experiments.history(p, seed=600, days=3000, herding=False)
    pct = mf.fig_fingerprint(spx, cal, default)
    assert set(pct) == set(calibrate.TARGETS) and all(0 <= v <= 100 for v in pct.values())
    mf.fig_tails(spx, cal, base)
    mf.fig_memory(spx, cal, default)
    for name in ["fig1_fingerprint.png", "fig2_tails.png", "fig3_memory.png"]:
        _png(out / name)


def test_mechanism_figure(out):
    mf.fig_mechanism(calibrate.load(), seed=616)
    _png(out / "fig4_mechanism.png")


def test_figures_use_held_out_seeds():
    assert set(mf.HELD_OUT).isdisjoint(calibrate.CALIBRATION_SEEDS)


def test_importing_make_figures_does_not_change_global_style():
    # Regression: make_figures used to set plt.rcParams globally, which cropped
    # figures made elsewhere (e.g. plots.comparison) once it had been imported
    import matplotlib.pyplot as plt
    assert plt.rcParams["savefig.bbox"] != "tight"


class _SerialPool:
    """Stands in for multiprocessing.Pool so monkeypatched settings reach the workers."""
    def map(self, f, jobs):
        return list(map(f, jobs))


@pytest.mark.slow
def test_out_of_sample_top_row_matches_fig1(out):
    # Bug fix: Fig 6's S&P row used different seeds from Fig 1, so the same
    # comparison printed different percentiles in the two figures
    p = calibrate.load()
    spx = market_data.spx_returns().values
    cal = experiments.ensemble(p, range(600, 604), days=3000)
    pct = mf.fig_fingerprint(spx, cal, cal)
    m, n_bad = mf.fig_out_of_sample(_SerialPool(), p, cal)
    assert np.array_equal(m[0], [pct[k] for k in calibrate.TARGETS])
    assert m.shape == (4, len(calibrate.TARGETS)) and ((0 <= m) & (m <= 100)).all()
    assert ((0 <= n_bad) & (n_bad <= len(market_data.stock_tickers()))).all()
    _png(out / "fig6_out_of_sample.png")


@pytest.mark.slow
def test_policy_figure(out, monkeypatch):
    monkeypatch.setattr(mf, "DAYS", 2000)
    p = calibrate.load()
    cal = experiments.ensemble(p, range(600, 602), days=2000)
    res, frac, over = mf.fig_policy(_SerialPool(), p, cal, market_data.spx_returns().values)
    assert res.shape == (20, 2, 3) and (res > 0).all()          # falls are positive sizes
    assert len(frac) == 200 and sum(len(o) for o in over) == 200  # every history lands in one bin
    _png(out / "fig7_policy.png")


def test_black_monday_is_the_largest_spx_move():
    # Fig 2 labels the right end of the S&P curve as Black Monday
    r = market_data.spx_returns()
    assert r.abs().idxmax().strftime("%Y-%m-%d") == "1987-10-19"


@pytest.mark.slow
def test_paper_check_figure(out, monkeypatch):
    monkeypatch.setattr(mf, "LONG", 20_000)
    monkeypatch.setattr(mf, "LONG_SCALES", (16, 32, 64, 128, 256, 512, 1024, 2048, 4096))
    p = calibrate.load()
    spx = market_data.spx_returns().values
    cal = experiments.ensemble(p, range(600, 602), days=3000)
    default = experiments.ensemble(model.P, range(600, 602), days=3000)
    med = mf.fig_paper_check(_SerialPool(), p, spx, cal, default, seeds=range(1100, 1102))
    assert set(med) == {"default", "calibrated", "S&P 500"}
    for name, m in med.items():
        assert 1 < m["alpha"] < 6 and 0 < m["H_r"] < 1 and 0 < m["H_abs"] < 1.2, (name, m)
    assert abs(med["calibrated"]["H_news"] - 0.5) < 0.1     # the news has no memory
    _png(out / "fig8_paper_check.png")


@pytest.mark.slow
def test_robustness_figure(out):
    ok, base, tails, memory = mf.fig_robustness(_SerialPool(), market_data.spx_returns().values,
                                                n=12, n_base=6)
    assert len(base) == 6 and 0 < len(ok) <= 12 and len(tails) == len(memory) == len(ok)
    assert np.all(base[:, 2] < 1.0)                          # no chartists: bell curve
    _png(out / "fig9_robustness.png")
