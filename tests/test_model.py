"""Tests for herding/model.py, the rebuilt Lux-Marchesi model (notes/SPEC.md).

Most tests compare the simulation with an answer worked out by hand from the
model's equations, not with the code's own earlier output.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

from herding import model, stats

BASE = dict(model.SET_I)


def settings(**changes):
    p = dict(BASE)
    p.update(changes)
    return p


def lag1(v):
    return np.corrcoef(v[:-1], v[1:])[0, 1]


# ------------------------------------------------------------------ check()

def test_known_settings_pass():
    model.check(BASE)


@pytest.mark.parametrize("change, words", [
    (dict(N=True), "finite number"),
    (dict(a1=float("nan")), "finite number"),
    (dict(a1="0.6"), "finite number"),
    (dict(a2=-0.1), "not be negative"),
    (dict(beta=0.0), "above 0"),
    (dict(v1=0.0), "above 0"),
    (dict(N=500.5), "whole number"),
    (dict(floor=2.5), "whole number"),
    (dict(s=1.0), "below 1"),
    (dict(tick=1.0), "below 1"),
    (dict(dt=0.03), "whole steps"),
    (dict(dt=0.2), "trend window"),
    (dict(dt=0.5), "trend window"),
    (dict(N=15), "too small"),
])
def test_bad_settings_are_rejected(change, words):
    with pytest.raises(ValueError, match=words):
        model.check(settings(**change))


def test_missing_and_unknown_settings_are_named():
    p = settings()
    del p["a1"]
    p["a4"] = 1.0
    with pytest.raises(ValueError, match=r"unknown settings \['a4'\].*missing settings \['a1'\]"):
        model.check(p)


def test_settings_must_be_a_dict():
    with pytest.raises(ValueError, match="dict"):
        model.check([1, 2])


def test_every_setting_is_documented():
    assert set(model.SET_I) == set(model.SETTINGS)


# ------------------------------------------------------------------ _leave()

def test_nobody_leaves_an_empty_group_or_at_rate_zero():
    g = np.random.default_rng(0)
    assert model._leave(g, 0, 5.0, 5.0, 0.01) == (0, 0)
    assert model._leave(g, 100, 0.0, 0.0, 0.01) == (0, 0)


def test_share_leaving_matches_one_minus_exp():
    # 1 - exp(-(2 + 3) * 0.1) = 0.3935: each trader's chance to leave in the step
    g = np.random.default_rng(1)
    n, trials = 1000, 2000
    left = np.array([sum(model._leave(g, n, 2.0, 3.0, 0.1)) for _ in range(trials)])
    expect = 1 - np.exp(-0.5)
    se = np.sqrt(expect * (1 - expect) / n / trials)
    assert abs(left.mean() / n - expect) < 4 * se


def test_leavers_split_in_proportion_to_their_rates():
    g = np.random.default_rng(2)
    a = b = 0
    for _ in range(2000):
        x, y = model._leave(g, 1000, 2.0, 3.0, 0.1)
        a, b = a + x, b + y
    assert abs(a / (a + b) - 0.4) < 0.005             # 2 / (2 + 3)


def test_a_huge_rate_moves_everybody_but_never_more():
    g = np.random.default_rng(3)
    assert model._leave(g, 37, 1e6, 0.0, 0.01) == (37, 0)
    assert model._leave(g, 37, 0.0, 1e6, 0.01) == (0, 37)


def test_cap_limits_exponents_both_ways():
    assert model._cap(1e9) == model.U_CAP
    assert model._cap(-1e9) == -model.U_CAP
    assert model._cap(0.25) == 0.25


# ------------------------------------------------------------------ simulate(): basics

def test_output_shapes_and_start():
    o = model.simulate(BASE, 50, seed=0)
    assert {len(o[k]) for k in ("lp", "lf", "z", "x")} == {51}
    assert len(o["r"]) == 50
    assert o["lp"][0] == 0 and o["lf"][0] == 0
    assert np.allclose(o["r"], np.diff(o["lp"]))
    assert o["z"][0] == pytest.approx(0.1)            # START_CHARTISTS
    assert o["x"][0] == 0


def test_same_seed_same_history_different_seed_different_history():
    a = model.simulate(BASE, 200, seed=5)
    b = model.simulate(BASE, 200, seed=5)
    c = model.simulate(BASE, 200, seed=6)
    assert np.array_equal(a["lp"], b["lp"])
    assert not np.array_equal(a["lp"], c["lp"])


@pytest.mark.parametrize("days", [0, -3, 2.5, True, "10"])
def test_days_must_be_a_positive_whole_number(days):
    with pytest.raises(ValueError, match="days"):
        model.simulate(BASE, days)


@pytest.mark.parametrize("bad", ["yes", 1, None])
def test_chartists_switch_must_be_true_or_false(bad):
    with pytest.raises(ValueError, match="chartists"):
        model.simulate(BASE, 10, chartists=bad)


@pytest.mark.parametrize("bad", [0, -1, True, "1"])
def test_max_gap_must_be_positive_or_none(bad):
    with pytest.raises(ValueError, match="max_gap"):
        model.simulate(BASE, 10, max_gap=bad)


def test_numba_and_plain_python_give_identical_histories():
    code = ("import sys, numpy as np; sys.path.insert(0, '.');"
            "from herding import model;"
            "o = model.simulate(model.SET_I, 30, seed=9);"
            "np.save(sys.argv[1], np.vstack([o['lp'], o['lf'], o['z'], o['x']]))")
    out = {}
    for flag in ("0", "1"):
        path = f"scratch_numba_{flag}.npy"
        env = dict(os.environ, MODEL_NO_NUMBA=flag)
        subprocess.run([sys.executable, "-c", code, path], check=True, env=env,
                       cwd=os.path.dirname(os.path.dirname(__file__)))
        full = os.path.join(os.path.dirname(os.path.dirname(__file__)), path)
        out[flag] = np.load(full)
        os.remove(full)
    assert np.array_equal(out["0"], out["1"])


# ------------------------------------------------------------------ the news (SPEC sec. 4)

def test_news_is_gaussian_with_no_memory():
    lf = model.simulate(settings(sigma_eps=0.01), 20000, seed=1)["lf"]
    eps = np.diff(lf)
    assert abs(eps.mean()) < 4 * 0.01 / np.sqrt(len(eps))
    assert eps.std() == pytest.approx(0.01, rel=0.02)
    assert abs(stats.kurtosis(eps)) < 0.15
    assert abs(lag1(eps)) < 0.03
    assert abs(lag1(np.abs(eps))) < 0.03


# ------------------------------------------------------------------ the price (SPEC sec. 3)

def test_price_never_moves_without_news_noise_or_chartists():
    o = model.simulate(settings(sigma_eps=0.0, sigma_mu=0.0), 100, seed=0, chartists=False)
    assert np.all(o["lp"] == 0)


def test_price_follows_its_value_like_a_spring():
    """With only fundamentalists, eq. 25 says the gap g = ln p - ln p_f obeys
    dg/dt = -kappa g + (news), with kappa = beta N gamma. That is a spring with
    known behaviour: spread sigma_eps / sqrt(2 kappa), one-day correlation e^-kappa.
    (A tiny tick makes the price's own randomness negligible.)"""
    beta, N, gamma, sig = 6.0, 500, 0.001, 0.005
    kappa = beta * N * gamma                                # 3 per day
    o = model.simulate(settings(beta=beta, gamma=gamma, sigma_eps=sig, sigma_mu=0.0,
                                tick=1e-5), 20000, seed=2, chartists=False)
    gap = o["lp"] - o["lf"]
    assert gap.std() == pytest.approx(sig / np.sqrt(2 * kappa), rel=0.05)
    assert lag1(gap) == pytest.approx(np.exp(-kappa), abs=0.03)


def test_mu_noise_per_day_does_not_depend_on_dt():
    def spread(dt):
        o = model.simulate(settings(sigma_eps=0.0, dt=dt), 5000, seed=3, chartists=False)
        return np.std(o["lp"])
    assert spread(0.005) == pytest.approx(spread(0.01), rel=0.1)


def test_no_chartists_gives_bell_curve_returns():
    """The proposal's base case: returns made only of news (and noise) are Gaussian."""
    o = model.simulate(BASE, 30000, seed=4, chartists=False)
    assert np.all(o["z"] == 0)
    assert abs(stats.kurtosis(o["r"])) < 0.15


# ------------------------------------------------------------------ the traders (SPEC sec. 1, 2, 5)

def test_mood_flips_match_the_two_state_formula():
    """No strategy switching (v2 ~ 0, a3 = 0) and U1 = 0 (a1 = a2 = 0): each of the n_c = 50
    chartists flips mood at rate v1 z. Then the opinion x forgets at rate 2 v1 z,
    and its spread is 1/sqrt(n_c)."""
    v1, nc, z = 3.0, 50, 0.1
    # (v2 also divides the trend inside U2, so a3 = 0 is needed to make U2 = 0)
    o = model.simulate(settings(v1=v1, v2=1e-12, a3=0.0, a1=0.0, a2=0.0), 20000, seed=5)
    assert np.all(o["z"] == z)
    x = o["x"]
    assert lag1(x) == pytest.approx(np.exp(-2 * v1 * z), abs=0.03)
    assert x.std() == pytest.approx(1 / np.sqrt(nc), rel=0.05)


def test_strategy_switching_is_a_fair_game_when_profits_are_equal():
    """With a3 = 0 the profit terms vanish and recruitment is symmetric (Kirman):
    a fundamentalist joins a chartist group exactly as often as a chartist leaves
    it. So the average chartist share stays at its start, 0.1."""
    p = settings(a3=0.0)
    z = [model.simulate(p, 3, seed=s)["z"][-1] for s in range(400)]
    assert abs(np.mean(z) - 0.1) < 4 * np.std(z) / np.sqrt(len(z))


def test_herding_above_one_splits_the_crowd():
    """Mean field: x' is proportional to (a1 - 1) x near x = 0, so a1 > 1 pushes
    chartists to one mood and a1 < 1 keeps them mixed."""
    fixed = dict(v2=1e-12, a3=0.0, a2=0.0)
    low = model.simulate(settings(a1=0.0, **fixed), 3000, seed=6)["x"]
    high = model.simulate(settings(a1=1.6, **fixed), 3000, seed=6)["x"]
    assert np.mean(np.abs(low)) < 0.2
    assert np.mean(np.abs(high)) > 0.6


def test_groups_never_fall_below_the_floor():
    for change in (dict(a3=50.0), dict(a3=50.0, s=0.0, gamma=0.0)):
        o = model.simulate(settings(**change), 300, seed=7, max_gap=None)
        nc = np.round(o["z"] * BASE["N"])
        assert nc.min() >= BASE["floor"]
        assert BASE["N"] - nc.max() >= BASE["floor"]
        assert np.all(np.abs(o["x"]) <= 1)


# ------------------------------------------------------------------ runaways

def test_runaway_bubble_is_stopped_and_reported():
    # with set I, seed 1 leaves its value on day 28,511 and never comes back (all
    # chartists, an endless bubble); it is stopped once the 2.7x gap has lasted
    # 250 days or passes e^20, whichever comes first
    with pytest.raises(model.RunawayError, match="runaway bubble") as e:
        model.simulate(BASE, 30000, seed=1)
    assert 28511 <= e.value.day <= 28511 + model.MAX_DAYS
    o = model.simulate(BASE, e.value.day, seed=1, max_gap=None)
    gap = o["lp"] - o["lf"]
    assert np.all(gap[28511:] > 1.0) and gap[28510] <= 1.0


def test_a_bubble_that_bursts_on_its_own_is_kept():
    """S&P-like settings, seed 1038: the price reaches 7x its value for a few days
    and then returns. That is the paper's mechanism, not a runaway."""
    # the S&P-calibrated settings (settings/spx.json), copied so this test stands alone
    p = settings(v1=1.3765469570480833, v2=0.26765417701072036, a1=0.4055088690417696, a2=1.1061772057289063, a3=0.43885072878839954, tc=0.0013005916669113352, gamma=0.004258471901282281, beta=1.1109772899398405, sigma_mu=0.0419025394459388, sigma_eps=0.0051115116615288465)
    o = model.simulate(p, 6000, seed=1038)
    gap = o["lp"] - o["lf"]
    assert 1 < np.sum(np.abs(gap) > 1.0) < model.MAX_DAYS
    assert abs(gap[-1]) < 0.3


@pytest.mark.parametrize("bad", [0, -5, 2.5, True])
def test_max_days_must_be_a_positive_whole_number(bad):
    with pytest.raises(ValueError, match="max_days"):
        model.simulate(BASE, 10, max_days=bad)


def test_runaway_check_can_be_switched_off():
    o = model.simulate(BASE, 30000, seed=1, max_gap=None)
    assert np.abs(o["lp"] - o["lf"]).max() > 1.0


# ------------------------------------------------------------------ regression guard

def test_matches_golden_output():
    """The exact output of a short run, saved when the rebuilt model was finished
    (November 2026). If this fails, the model's behaviour changed: check whether
    that was intended, and only then save a new golden file."""
    from conftest import ROOT
    gold = np.load(ROOT / "tests" / "golden_v3.npz")
    o = model.simulate(BASE, 300, seed=0)
    for k in ("lp", "lf", "z", "x"):
        assert np.array_equal(o[k], gold[k]), k
