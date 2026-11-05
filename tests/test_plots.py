"""Tests for herding/plots.py and the package layout."""
import numpy as np
import pytest
from matplotlib.image import imread

from conftest import ROOT
from herding import plots, stats


def series(seed, n=3000):
    g = np.random.default_rng(seed)
    return g.standard_t(3, n) * 0.01, g.uniform(0, 0.3, n + 1), g.standard_normal(n) * 0.01


def test_comparison_saves_a_real_figure(tmp_path):
    model_r, z, base = series(0)
    real = series(1, 5000)[0]
    out = plots.comparison(real, model_r, z, base, tmp_path / "new" / "fig.png")
    assert out == tmp_path / "new" / "fig.png"
    img = imread(out)
    assert img.shape[0] > 800 and img.shape[1] > 1400


def test_ccdf_draws_exactly_the_stats_ccdf(tmp_path):
    import matplotlib.pyplot as plt
    r = series(2)[0]
    fig, ax = plt.subplots()
    plots.ccdf(ax, r, "x", plots.BLUE)
    x, y = ax.lines[0].get_data()
    z, cc = stats.ccdf(r)
    assert np.array_equal(x, z[z > 0]) and np.array_equal(y, cc[z > 0])
    plt.close(fig)


def test_dfa_label_reports_the_same_H_as_stats():
    import matplotlib.pyplot as plt
    x = np.random.default_rng(3).standard_normal(20000)
    s = stats.dfa_scales(len(x))
    fig, ax = plt.subplots()
    H = plots.dfa(ax, x, s, "noise", plots.AQUA)
    assert H == pytest.approx(stats.hurst(x, s))
    assert f"H = {H:.2f}" in ax.lines[0].get_label()
    plt.close(fig)


def test_standardized_has_mean_0_and_sd_1():
    z = plots.standardized(series(4)[0])
    assert abs(z.mean()) < 1e-12 and z.std() == pytest.approx(1)


def test_palette_roles_are_distinct():
    roles = [plots.REAL, plots.SPX_MODEL, plots.PAPER_MODEL, plots.NEWS]
    assert len(set(roles)) == 4


def test_package_has_one_module_per_job():
    names = {p.stem for p in (ROOT / "herding").glob("*.py")}
    assert names == {"__init__", "model", "stats", "market_data", "kirman", "calibrate",
                     "plots", "experiments", "policy"}
