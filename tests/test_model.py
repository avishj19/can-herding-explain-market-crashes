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
            "o = model.simulate(model.SET_I, 30, seed=9, shocks={5: -0.1},"
            " breakers=dict(levels=(0.002, 0.004, 0.006), halt_minutes=30));"
            "np.save(sys.argv[1], np.vstack([o['lp'], o['lf'], o['z'], o['x'],"
            " np.append(o['halts'], -1)]))")
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


# ------------------------------------------------------------------ policy options (extension)

TIGHT = dict(levels=(0.003, 0.006, 0.009), halt_minutes=15)   # fires on many days


def test_breakers_off_and_breakers_that_never_fire_change_nothing():
    # Same seed: with breakers that cannot fire (a 97% fall), every random number is
    # used in the same way, so the history must be identical to no breakers at all.
    a = model.simulate(model.SET_I, 300, seed=3)
    b = model.simulate(model.SET_I, 300, seed=3,
                       breakers=dict(levels=(0.97, 0.98, 0.99), halt_minutes=15))
    for k in ("lp", "lf", "z", "x"):
        assert np.array_equal(a[k], b[k])
    assert not a["halts"].any() and not b["halts"].any()


def test_a_day_without_a_halt_never_fell_to_the_first_level():
    o = model.simulate(model.SET_I, 2000, seed=4, breakers=TIGHT)
    h, r = o["halts"], o["r"]
    assert set(np.unique(h)) == {0, 1, 2, 3}           # every level fires at some point
    assert np.all(r[h == 0] > np.log(1 - 0.003))


def test_closing_freezes_the_price_at_the_fall_that_closed_it():
    # A slow price (beta 0.05: about 2.5 ticks per step at most in set I), so one
    # step cannot jump far past the level; after closing, the price must stay there.
    o = model.simulate(dict(model.SET_I, beta=0.05), 2000, seed=4, breakers=TIGHT)
    r, h = o["r"][o["halts"] == 3], o["halts"]
    one_step = 40 * np.log(1 + 0.001)                  # far more ticks than one step makes
    assert np.all(r <= np.log(1 - 0.009))
    assert np.all(r > np.log(1 - 0.009) - one_step)
    assert (h == 3).sum() > 10


def test_a_pause_of_a_whole_day_freezes_the_close_between_the_levels():
    # halt_minutes=390: after the first-level fall the price is frozen for the rest
    # of the day, so the close sits between level 1 and level 2.
    o = model.simulate(model.SET_I, 2000, seed=5, breakers=dict(TIGHT, halt_minutes=390))
    r = o["r"][o["halts"] == 1]
    assert len(r) > 10
    assert np.all((r <= np.log(1 - 0.003)) & (r > np.log(1 - 0.006)))


def test_pause_steps_converts_minutes_of_a_390_minute_day():
    assert model.pause_steps(15, 0.01) == 4               # 15/390 day = 3.85 steps of 0.01
    assert model.pause_steps(39, 0.01) == 10
    assert model.pause_steps(390, 0.005) == 200
    assert model.pause_steps(0.1, 0.01) == 1              # never shorter than one step


def test_a_policy_surprise_moves_the_value_once_by_exactly_its_size():
    o = model.simulate(settings(sigma_eps=0.0), 50, seed=1, shocks={20: -0.1, 30: 0.05})
    lf = o["lf"]
    assert np.all(lf[:21] == 0)
    assert np.allclose(lf[21:31], np.log(0.9))
    assert np.allclose(lf[31:], np.log(0.9) + np.log(1.05))


@pytest.mark.parametrize("bad", [
    (0.07, 0.13), (0.13, 0.07, 0.2), (0.0, 0.1, 0.2), (0.1, 0.2, 1.0), (0.1, True, 0.3),
    (0.1, float("nan"), 0.3)])
def test_breaker_levels_must_be_three_increasing_falls(bad):
    with pytest.raises(ValueError, match="levels"):
        model.simulate(BASE, 10, breakers=dict(levels=bad, halt_minutes=15))


@pytest.mark.parametrize("bad", [dict(levels=(0.1, 0.2, 0.3)), [0.1, 0.2, 0.3],
                                 dict(levels=(0.1, 0.2, 0.3), halt_minutes=15, extra=1)])
def test_breakers_need_exactly_levels_and_minutes(bad):
    with pytest.raises(ValueError, match="breakers"):
        model.simulate(BASE, 10, breakers=bad)


@pytest.mark.parametrize("bad", [0, -5, 391, True, float("inf")])
def test_halt_minutes_must_fit_in_a_day(bad):
    with pytest.raises(ValueError, match="halt_minutes"):
        model.simulate(BASE, 10, breakers=dict(levels=(0.1, 0.2, 0.3), halt_minutes=bad))


@pytest.mark.parametrize("bad", [{10: 0.1}, {-1: 0.1}, {2.0: 0.1}, {True: 0.1}, {2: -1.0},
                                 {2: float("nan")}, {2: "x"}, [(2, 0.1)]])
def test_shocks_must_be_days_in_range_and_fractions_above_minus_one(bad):
    with pytest.raises(ValueError, match="shock"):
        model.simulate(BASE, 10, shocks=bad)
