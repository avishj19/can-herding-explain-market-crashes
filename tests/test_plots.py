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
    H, n0, F0 = plots.dfa(ax, x, s, "noise", plots.AQUA, shift=2.0)
    assert H == pytest.approx(stats.hurst(x, s))
    assert f"H = {H:.2f}" in ax.lines[0].get_label()
    # the anchor is the first point of the drawn (shifted) fit line
    assert n0 == s[0] and F0 == pytest.approx(ax.lines[1].get_ydata()[0])
    plt.close(fig)


def test_ccdf_markers_sit_on_real_data_points():
    import matplotlib.pyplot as plt
    r = series(5)[0]
    fig, ax = plt.subplots()
    plots.ccdf(ax, r, "x", plots.BLUE, points=40)
    x, y = ax.lines[0].get_data()
    z, cc = stats.ccdf(r)
    i = np.searchsorted(z, x)
    assert 10 < len(x) <= 41 and np.array_equal(z[i], x) and np.array_equal(cc[i], y)
    assert x[-1] == z[-1] and ax.lines[0].get_linestyle() == "None"
    plt.close(fig)


def test_power_law_has_the_requested_slope():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plots.power_law(ax, 2.0, 0.3, -2.64, 40.0)
    x, y = ax.lines[0].get_data()
    assert y[0] == 0.3 and np.log(y[1] / y[0]) / np.log(x[1] / x[0]) == pytest.approx(-2.64)
    plt.close(fig)


def test_bell_curve_line_is_the_two_sided_normal_tail():
    """Share of a standard normal beyond +-x: 1 at x=0, 0.3173 at 1, 0.0455 at 2 (the
    68-95 rule), checked against scipy, which plots.py no longer imports."""
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    fig, ax = plt.subplots()
    plots.gaussian_ccdf(ax)
    x, y = ax.lines[0].get_data()
    assert np.allclose(y, 2 * norm.sf(x), rtol=1e-12)
    assert np.interp(1.0, x, y) == pytest.approx(0.3173, abs=2e-3)
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
