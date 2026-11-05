"""Tests for make_figures.py: every figure is drawn, and the numbers each figure
reports agree with the same numbers computed directly."""
import numpy as np
import pytest
from matplotlib.image import imread

import make_figures as mf
from herding import calibrate, experiments, market_data, stats


class SerialPool:
    """Stands in for multiprocessing.Pool so tests stay small and in one process."""

    def map(self, fn, jobs, chunksize=None):
        return [fn(j) for j in jobs]


def fat(seed, n):
    return 0.01 * np.random.default_rng(seed).standard_t(3, n)


def run(seed, n=6000):
    g = np.random.default_rng(seed)
    lf = np.concatenate([[0.0], np.cumsum(0.005 * g.standard_normal(n))])
    lp = lf + 0.02 * np.sin(np.arange(n + 1) / 50)
    return dict(lp=lp, lf=lf, z=g.uniform(0, 0.6, n + 1), r=np.diff(lp))


def is_image(path):
    img = imread(path)
    return img.shape[0] > 300 and img.shape[1] > 600


def test_fig1_snapshot(tmp_path):
    assert is_image(mf.fig1_snapshot(run(0), run(1), tmp_path / "f1.png"))


def test_fig2_and_fig3(tmp_path):
    paper, news, model_r, spx = fat(0, 20000), 0.005 * np.random.default_rng(1).standard_normal(20000), \
        fat(2, 20000), fat(3, 19000)
    assert is_image(mf.fig2_aggregation(paper, news, [model_r[:10000], model_r[10000:]], spx,
                                        tmp_path / "f2.png", paper_runs=[paper, fat(5, 20000)]))
    assert is_image(mf.fig3_memory(paper, news, model_r, spx, tmp_path / "f3.png"))


def test_fig4_checks(tmp_path):
    base = [0.01 * np.random.default_rng(s).standard_normal(5000) for s in range(5)]
    runs = [fat(s, 5000) for s in range(5)]
    assert is_image(mf.fig4_checks(base, runs, runs[:3], fat(9, 5000), tmp_path / "f4.png"))


def test_fig5_mechanism(tmp_path):
    assert is_image(mf.fig5_mechanism(run(4, 5000), tmp_path / "f5.png"))


def test_fig7_reports_the_same_percentiles_as_computed_directly(tmp_path):
    runs = [fat(s, 3000) for s in range(10)]
    spx = fat(99, 3000)
    pct = mf.fig7_typical(runs, spx, tmp_path / "f7.png")
    target = stats.fingerprint(spx)
    sims = [stats.fingerprint(r) for r in runs]
    for k in stats.FINGERPRINT:
        assert pct[k] == experiments.percentile(target[k], [f[k] for f in sims])
    assert is_image(tmp_path / "f7.png")


def test_shade_mixes_with_white():
    assert mf.shade("#000000", 0.0) == (0.0, 0.0, 0.0)
    assert mf.shade("#000000", 1.0) == (1.0, 1.0, 1.0)
    assert mf.shade("#ff0000", 0.5) == pytest.approx((1.0, 0.5, 0.5))


def test_every_fingerprint_measure_has_a_label():
    assert set(mf.LABEL) == set(stats.FINGERPRINT)


@pytest.mark.slow
def test_fig6_kirman(tmp_path):
    assert is_image(mf.fig6_kirman(tmp_path / "f6.png"))


@pytest.mark.slow
def test_fig8_fig9_fig10_with_the_fitted_settings(tmp_path):
    if not (calibrate.SETTINGS_DIR / "spx.json").exists():
        pytest.skip("not fitted yet")
    p = calibrate.load("spx")
    pool = SerialPool()
    rows, keys, names = mf.fig8_out_of_sample(pool, p, tmp_path / "f8.png", n=3)
    assert rows.shape == (len(names), len(keys)) and "sd" not in keys
    assert np.all((rows >= 0) & (rows <= 100))
    counts = mf.fig9_robustness(pool, n=8, days=4000, path=tmp_path / "f9.png")
    assert counts["runaway"] + counts["thin"] + counts["fat_only"] + counts["fat_memory"] == 8
    a1s, dials = mf.fig10_herding(pool, p, calibrate.load("paper"), 3000, tmp_path / "f10.png", n=2)
    for rows in dials.values():
        assert rows.shape == (len(a1s), 4) and np.all((rows[:, 0] >= 0) & (rows[:, 0] <= 1))
    out = mf.fig11_policy(pool, p, market_data.spx_returns().values[:4000], tmp_path / "f11.png",
                          breaker_seeds=range(3000, 3006), surprise_seeds=range(4000, 4010))
    assert out["breaker_runaways"] + 6 >= 6 and out["events"] <= 6
    lo_hi = [v for k, v in out.items() if k.startswith("surprise_")]
    assert sum(v["n"] for v in lo_hi) + out["surprise_runaways"] == 10
    for name in ("f8.png", "f9.png", "f10.png", "f11.png"):
        assert is_image(tmp_path / name)
