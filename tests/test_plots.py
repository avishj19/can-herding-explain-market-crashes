"""Tests for herding/plots.py and the package layout.  Run: python3 -m pytest tests/ -q"""
import numpy as np
import pytest
from matplotlib.image import imread

from conftest import ROOT
from herding import market_data, plots


def series(seed, n=3000):
    g = np.random.default_rng(seed)
    return g.standard_t(3, n) * 0.01, g.uniform(0, 0.3, n), g.standard_normal(n) * 0.01


def test_comparison_saves_full_size_figure(tmp_path):
    herd, frac, base = series(0)
    real = series(1, 5000)[0]
    out = plots.comparison(real, herd, frac, base, tmp_path / "fig.png")
    assert out == tmp_path / "fig.png" and out.exists()
    img = imread(out)
    assert img.shape[:2] == (960, 1440)            # 12 x 8 inches at 120 dpi


def test_comparison_creates_missing_folder(tmp_path):
    herd, frac, base = series(2)
    out = plots.comparison(herd, herd, frac, base, tmp_path / "new_folder" / "f.png")
    assert out.exists()


def test_comparison_accepts_string_path(tmp_path):
    herd, frac, base = series(3)
    assert plots.comparison(herd, herd, frac, base, str(tmp_path / "s.png")).exists()


def test_comparison_rejects_empty_or_constant_returns(tmp_path):
    herd, frac, base = series(4)
    with pytest.raises(ValueError):
        plots.comparison(np.array([]), herd, frac, base, tmp_path / "x.png")
    with pytest.raises(ValueError):
        plots.comparison(herd, herd, frac, np.zeros(100), tmp_path / "x.png")


def test_comparison_closes_its_figures(tmp_path):
    # Repeated calls (e.g. a web app) must not pile up open figures in memory
    import matplotlib.pyplot as plt
    herd, frac, base = series(5)
    before = len(plt.get_fignums())
    for i in range(3):
        plots.comparison(herd, herd, frac, base, tmp_path / f"{i}.png")
    assert len(plt.get_fignums()) == before


def test_project_root_points_at_project_folder():
    # After moving the code into herding/, data paths must still resolve
    assert market_data.ROOT == ROOT
    assert market_data.SPX.exists()
    assert (ROOT / "herding" / "__init__.py").exists()


def test_comparison_closes_figure_even_when_drawing_fails(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    herd, frac, base = series(6)
    def boom(*a, **k):
        raise RuntimeError("drawing failed")
    monkeypatch.setattr(plots, "_draw_comparison", boom)
    before = len(plt.get_fignums())
    with pytest.raises(RuntimeError):
        plots.comparison(herd, herd, frac, base, tmp_path / "x.png")
    assert len(plt.get_fignums()) == before


def test_comparison_error_names_the_bad_series(tmp_path):
    herd, frac, base = series(7)
    with pytest.raises(ValueError, match="base returns"):
        plots.comparison(herd, herd, frac, np.zeros(10), tmp_path / "x.png")
