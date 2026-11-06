"""Tests for make_figures.py: every figure is drawn, and the numbers each figure
reports agree with the same numbers computed directly."""
from multiprocessing import Pool

import numpy as np
import pytest
from matplotlib.image import imread

import make_figures as mf
from herding import calibrate, experiments, market_data, stats
from helpers import SerialPool


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
    assert is_image(mf.fig1_snapshot(run(0), run(1), fat(9, 5000), tmp_path / "f1.png"))


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
    with Pool(4) as pool:                        # independent jobs: a real pool is faster
        check_slow_figures(pool, p, tmp_path)


def check_slow_figures(pool, p, tmp_path):
    rows, keys, names = mf.fig8_out_of_sample(pool, p, tmp_path / "f8.png", n=3)
    assert rows.shape == (len(names), len(keys)) and "sd" not in keys
    assert np.all((rows >= 0) & (rows <= 100))
    counts = mf.fig9_robustness(pool, n=8, days=4000, path=tmp_path / "f9.png")
    assert counts["coarse"] + counts["runaway"] + counts["thin"] + counts["fat_only"] \
        + counts["fat_memory"] == 8
    a1s, dials = mf.fig10_herding(pool, p, calibrate.load("paper"), 3000, tmp_path / "f10.png", n=2)
    for rows in dials.values():
        assert rows.shape == (len(a1s), 4) and np.all((rows[:, 0] >= 0) & (rows[:, 0] <= 1))
    out = mf.fig11_policy(pool, p, market_data.spx_returns().values[:4000], tmp_path / "f11.png",
                          breaker_seeds=range(3000, 3006), surprise_seeds=range(4000, 4010))
    assert 0 <= out["breaker_runaways"] <= 6 and 0 <= out["events"] <= 6
    if out["events"] < 2:                              # too short for a pause: no comparison
        assert all(np.isnan(v).all() for v in out["diffs"].values())
    lo_hi = [v for k, v in out.items() if k.startswith("surprise_") and k.endswith("%")]
    assert sum(v["n"] for v in lo_hi) + out["surprise_runaways"] == 10
    for name in ("f8.png", "f9.png", "f10.png", "f11.png"):
        assert is_image(tmp_path / name)


def test_fig8_gives_the_stocks_their_own_seeds(tmp_path, monkeypatch):
    calls = []

    def fake_returns(p, seeds, days, pool=None):
        calls.append((tuple(seeds), days))
        return [fat(s, days) for s in seeds]

    monkeypatch.setattr(mf.experiments, "returns", fake_returns)
    rows, keys, names = mf.fig8_out_of_sample(SerialPool(), None, tmp_path / "f8.png", n=3)
    index_seeds = {s for seeds, days in calls if days != len(market_data.stock_returns(names[2]))
                   for s in seeds}
    stock_seeds = {s for seeds, days in calls if days == len(market_data.stock_returns(names[2]))
                   for s in seeds}
    assert index_seeds == set(mf.HELD_OUT[:3]) and stock_seeds == set(mf.HELD_OUT[3:6])
    assert len(calls) == 3                       # S&P 1928, Nasdaq, and one run shared by all stocks
    assert rows.shape == (22, len(keys))
    with pytest.raises(ValueError, match="n must be"):
        mf.fig8_out_of_sample(SerialPool(), None, tmp_path / "x.png", n=21)


def test_dial_hides_medians_from_too_few_survivors(monkeypatch):
    # 12 histories; at the first a1 only 4 survive (too few), at the second all do.
    def fake_histories(p, seeds, days, pool=None):
        alive = 4 if p["a1"] == 0.1 else len(seeds)
        return [dict(r=fat(s, 3000)) if i < alive else None for i, s in enumerate(seeds)]

    monkeypatch.setattr(mf.experiments, "histories", fake_histories)
    rows = mf._dial(None, dict(a1=0.0), np.array([0.1, 0.5]), 3000, 12)
    assert rows[0, 0] == pytest.approx(4 / 12) and np.all(np.isnan(rows[0, 1:]))
    assert rows[1, 0] == 1 and np.all(np.isfinite(rows[1, 1:]))
