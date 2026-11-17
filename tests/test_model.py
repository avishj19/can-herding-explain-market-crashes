"""Tests for herding/model.py, the rebuilt Lux-Marchesi model (docs/SPEC.md).

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
    (dict(floor=0), "above 0"),      # bug fix: floor 0 crashed simulate() (division by 0)
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

def test_step_pull_by_hand():
    # SET_I: beta 6 * N 500 * gamma 0.01 * dt 0.01 = 0.3 (the price moves 30% of the way back per step)
    assert model.step_pull(BASE) == pytest.approx(0.3)
    assert model.step_pull(settings(dt=0.005)) == pytest.approx(0.15)     # half the step, half the pull


def test_too_coarse_a_step_blows_up_without_any_chartists():
    # With no chartists there is no herding, so a "runaway" can only come from the time
    # step: past STEP_LIMIT every step overshoots the value by more than the last.
    unstable = settings(beta=20.0, gamma=0.05)                  # pull 5.0
    assert model.step_pull(unstable) >= model.STEP_LIMIT
    with pytest.raises(model.RunawayError):
        model.simulate(unstable, 200, seed=1, chartists=False)
    # the same settings with a step 4x finer (pull 1.25) are fine
    model.simulate(settings(beta=20.0, gamma=0.05, dt=0.0025), 200, seed=1, chartists=False)


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


def test_numba_and_plain_python_give_identical_histories(tmp_path):
    code = ("import sys, numpy as np; sys.path.insert(0, '.');"
            "from herding import model;"
            "o = model.simulate(model.SET_I, 30, seed=9, shocks={5: -0.1},"
            " breakers=dict(levels=(0.002, 0.004, 0.006), halt_minutes=30));"
            "np.save(sys.argv[1], np.vstack([o['lp'], o['lf'], o['z'], o['x'],"
            " np.append(o['halts'], -1)]))")
    out = {}
    for flag in ("0", "1"):
        # the file goes in pytest's own temporary folder: two test runs at once (the
        # overnight stress test) once overwrote each other's file in the project folder
        path = tmp_path / f"numba_{flag}.npy"
        env = dict(os.environ, MODEL_NO_NUMBA=flag)
        subprocess.run([sys.executable, "-c", code, str(path)], check=True, env=env,
                       cwd=os.path.dirname(os.path.dirname(__file__)))
        out[flag] = np.load(path)
    assert np.array_equal(out["0"], out["1"])


# ------------------------------------------------------------------ the news (SPEC sec. 4)

def test_news_is_gaussian_with_no_memory():
    # The news never depends on the traders, so they are switched off: with them on, set I
    # ran away before day 20,000 on 4 of 39 other seeds (overnight stress test, Nov 2026).
    lf = model.simulate(settings(sigma_eps=0.01), 20000, seed=1, chartists=False)["lf"]
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
    (A tiny tick makes the price's own randomness negligible.) 80,000 days: with 20,000,
    the one-day correlation had sd 0.0077 over 39 other seeds, only 3.5 sd of room."""
    beta, N, gamma, sig = 6.0, 500, 0.001, 0.005
    kappa = beta * N * gamma                                # 3 per day
    o = model.simulate(settings(beta=beta, gamma=gamma, sigma_eps=sig, sigma_mu=0.0,
                                tick=1e-5), 80000, seed=2, chartists=False)
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
    o = model.simulate(settings(v1=v1, v2=1e-12, a3=0.0, a1=0.0, a2=0.0), 40000, seed=5)
    assert np.all(o["z"] == z)
    x = o["x"]
    assert lag1(x) == pytest.approx(np.exp(-2 * v1 * z), abs=0.03)
    assert x.std() == pytest.approx(1 / np.sqrt(nc), rel=0.05)


def test_equal_profits_give_a_small_drift_to_chartists_set_by_the_time_step():
    """With a3 = 0 the profit terms vanish and recruitment is symmetric (Kirman): in
    continuous time a fundamentalist joins the chartists exactly as often as a chartist
    leaves them (both flows are v2 n_f n_c / N per day), a fair game.
    The code moves each trader at most once per step (SPEC sec. 5), and a chartist can
    also use that move to change mood. Over one step a group whose members leave at total
    rate T keeps a fraction about 1 - T dt / 2 of its first-order flow, so chartists
    (T = v1 z + v2 (1 - z) = 2.1 a day) lose more of theirs than fundamentalists
    (T = v2 z = 0.2). Net drift towards the chartists, per day:
        (2.1 - 0.2) / 2 * dt * v2 z (1 - z) = 0.95 * 0.01 * 0.18 = 0.00171.
    It shrinks in proportion to dt (measured 0.00161, 0.00085, 0.00020 per day at dt =
    0.01, 0.005, 0.0025, each +- 0.00013, over 40,000 seeds). Found by the overnight stress
    test (Nov 2026): the old test called this a fair game and passed on its 400 seeds by
    luck (the drift was 20 standard errors over 40,000)."""
    p = settings(a3=0.0)
    z = np.array([model.simulate(p, 1, seed=s)["z"][-1] for s in range(20000)])
    v1, v2, z0, dt = p["v1"], p["v2"], 0.1, p["dt"]
    drift = ((v1 * z0 + v2 * (1 - z0)) - v2 * z0) / 2 * dt * v2 * z0 * (1 - z0)
    se = z.std(ddof=1) / np.sqrt(len(z))                # about 0.00018
    assert abs(z.mean() - 0.1 - drift) < 5 * se
    assert z.mean() - 0.1 > 5 * se                      # really there, not chance


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
        # the strong pull empties one group exactly down to the floor, not one above it
        assert min(nc.min(), BASE["N"] - nc.max()) == BASE["floor"]
        assert np.all(np.abs(o["x"]) <= 1)


def test_smallest_floor_keeps_one_trader_in_each_group():
    """floor=1 (the smallest allowed): a strong pull to the more profitable strategy empties
    one group down to exactly 1 trader, and the run does not crash. Which group empties
    depends on the first few moves (seed 7: the chartists; on 26 of 39 other seeds the
    fundamentalists, found by the overnight stress test), so either may be the one."""
    o = model.simulate(settings(a3=50.0, floor=1), 300, seed=7, max_gap=None)
    nc = np.round(o["z"] * BASE["N"])
    assert min(nc.min(), BASE["N"] - nc.max()) == 1
    assert nc.min() >= 1 and BASE["N"] - nc.max() >= 1


# ------------------------------------------------------------------ runaways

# Strong trend-following (a2 = 6 instead of 0.2): every one of 300 seeds ran away before
# day 2,505 (overnight stress test, Nov 2026), so a test needing a runaway works on any seed.
RUNAWAY = settings(a2=6.0)
RUNAWAY_DAYS = 6000


def test_strong_trend_following_always_runs_away():
    with pytest.raises(model.RunawayError, match="runaway bubble"):
        model.simulate(RUNAWAY, RUNAWAY_DAYS, seed=1)


@pytest.mark.pinned_seed
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


@pytest.mark.pinned_seed
def test_a_bubble_that_bursts_on_its_own_is_kept():
    """S&P-like settings, seed 1038: the price reaches 7x its value for a few days
    and then returns. That is the paper's mechanism, not a runaway."""
    # the S&P-calibrated settings (settings/sp500_calibrated.json), copied so this test stands alone
    p = settings(v1=1.3765469570480833, v2=0.26765417701072036, a1=0.4055088690417696, a2=1.1061772057289063, a3=0.43885072878839954, tc=0.0013005916669113352, gamma=0.004258471901282281, beta=1.1109772899398405, sigma_mu=0.0419025394459388, sigma_eps=0.0051115116615288465)
    o = model.simulate(p, 6000, seed=1038)
    gap = o["lp"] - o["lf"]
    assert 1 < np.sum(np.abs(gap) > 1.0) < model.MAX_DAYS
    assert abs(gap[-1]) < 0.3


def first_runaway_day_by_hand(o, max_gap, max_days, hard_gap):
    """The runaway rule (SPEC.md "Runaways") applied by hand to a run made with the check
    off: the first day the gap has been above max_gap for max_days days in a row, or
    above hard_gap at all."""
    gap, run = np.abs(o["lp"] - o["lf"]), 0
    for day in range(1, len(gap)):
        run = run + 1 if gap[day] > max_gap else 0
        if run >= max_days or gap[day] > hard_gap:
            return day
    return None


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_runaway_day_matches_the_rule_applied_by_hand(seed):
    # The run with the check off uses the same random numbers, so it is the same history
    # up to the day the check stops it. (Added after the overnight mutation test, Nov 2026:
    # changing the 250 days, the e^20 limit or the day counter went unnoticed before.)
    with pytest.raises(model.RunawayError) as e:
        model.simulate(RUNAWAY, RUNAWAY_DAYS, seed=seed)
    o = model.simulate(RUNAWAY, e.value.day, seed=seed, max_gap=None)
    assert e.value.day == first_runaway_day_by_hand(o, model.MAX_GAP, model.MAX_DAYS, model.HARD_GAP)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_a_short_custom_runaway_rule_counts_days_in_a_row(seed):
    # an ordinary market with a strict rule (5% away for 3 days in a row) stops early
    with pytest.raises(model.RunawayError) as e:
        model.simulate(BASE, 3000, seed=seed, max_gap=0.05, max_days=3)
    o = model.simulate(BASE, e.value.day, seed=seed, max_gap=None)
    assert e.value.day == first_runaway_day_by_hand(o, 0.05, 3, model.HARD_GAP)
    gap = np.abs(o["lp"] - o["lf"])
    assert np.all(gap[e.value.day - 2:] > 0.05) and gap[e.value.day - 3] <= 0.05


@pytest.mark.parametrize("max_gap", [None, model.MAX_GAP])
def test_an_absurd_number_of_ticks_stops_the_run_even_with_the_check_off(max_gap):
    # The value jumps 1e20-fold at the start of day 3: the fundamentalists' push would ask
    # for about 1e22 ticks in one step, beyond MAX_TICKS, so the run stops that day and is
    # reported as a crash (price far below value), even with the runaway check off. No
    # test reached this stop before the overnight mutation test (Nov 2026).
    import re
    with pytest.raises(model.RunawayError, match="runaway crash") as e:
        model.simulate(BASE, 10, seed=0, shocks={3: 1e20}, max_gap=max_gap)
    assert e.value.day == 4
    m = re.search(r"is ([0-9.e+]+)x away from its value with (\d+)% chartists", str(e.value))
    assert 1e19 < float(m[1]) < 1e21                     # the price barely moved: about 1e20
    assert 0 <= int(m[2]) <= 100


def test_the_runaway_limits_are_the_documented_ones():
    # SPEC.md "Runaways": e (2.7x) for 250 trading days in a row, or ever e^20
    assert (model.MAX_GAP, model.MAX_DAYS, model.HARD_GAP) == (1.0, 250, 20.0)
    assert model.U_CAP == 30.0 and model.STEP_LIMIT == 2.0 and model.MAX_TICKS == 1e15


def test_numba_is_used_when_it_is_installed():
    import importlib.util
    if importlib.util.find_spec("numba") is None:
        pytest.skip("numba not installed")
    assert model.USE_NUMBA == (os.environ.get("MODEL_NO_NUMBA") != "1")


def test_settings_exactly_at_the_limits_are_accepted():
    model.check(settings(N=16, floor=4))                 # N = 4 * floor
    model.check(settings(dt=model.MIN_DT))
    model.check(settings(N=model.MAX_N))
    model.check(settings(tick=model.MIN_TICK))
    o = model.simulate(BASE, 1, seed=0, max_days=1)      # the smallest run
    assert len(o["r"]) == 1
    model.simulate(BASE, 5, seed=0, breakers=dict(levels=(0.1, 0.2, 0.3), halt_minutes=1))


def test_the_starting_chartists_split_evenly_with_an_odd_count():
    # N = 530: 10% is 53 chartists, which cannot split evenly: 26 optimists and 26 pessimists
    o = model.simulate(settings(N=530), 1, seed=0)
    assert o["z"][0] == 52 / 530 and o["x"][0] == 0


def test_without_chartists_the_opinion_is_zero():
    o = model.simulate(BASE, 50, seed=0, chartists=False)
    assert np.all(o["z"] == 0) and np.all(o["x"] == 0)


def test_a_runaway_error_survives_being_sent_between_processes():
    import pickle
    e = pickle.loads(pickle.dumps(model.RunawayError("ran away", 12)))
    assert isinstance(e, model.RunawayError) and e.day == 12 and str(e) == "ran away"


def test_a_runaway_inside_a_process_pool_is_raised_not_hung():
    from multiprocessing import Pool
    with Pool(1) as pool:
        with pytest.raises(model.RunawayError):
            pool.apply_async(model.simulate, (RUNAWAY, RUNAWAY_DAYS), dict(seed=1)).get(timeout=120)


@pytest.mark.parametrize("bad", [0, -5, 2.5, True])
def test_max_days_must_be_a_positive_whole_number(bad):
    with pytest.raises(ValueError, match="max_days"):
        model.simulate(BASE, 10, max_days=bad)


def test_runaway_check_can_be_switched_off():
    o = model.simulate(RUNAWAY, RUNAWAY_DAYS, seed=1, max_gap=None)
    assert np.abs(o["lp"] - o["lf"]).max() > 1.0


# ------------------------------------------------------------------ regression guard

@pytest.mark.pinned_seed
def test_matches_golden_output():
    """The exact output of a short run, saved when the rebuilt model was finished
    (November 2026). If this fails, the model's behaviour changed: check whether
    that was intended, and only then save a new golden file."""
    from conftest import ROOT
    gold = np.load(ROOT / "tests" / "model_reference_output_v3.npz")
    o = model.simulate(BASE, 300, seed=0)
    for k in ("lp", "lf", "z", "x"):
        assert np.array_equal(o[k], gold[k]), k


# ------------------------------------------------------------------ policy options (extension)

TIGHT = dict(levels=(0.003, 0.006, 0.009), halt_minutes=15)   # fires on many days
# The breaker tests check how trading stops, so they use a market that never runs away: no
# herding pull (a1 = a2 = 0) and weak chartists. Set I with these breakers ran away within
# 2,000 days on 26 of 300 seeds, which stopped the tests on other seeds; this market ran away
# on none of 300 and every level still fired in all 300 (overnight stress test, Nov 2026).
CALM = dict(model.SET_I, a1=0.0, a2=0.0, tc=0.002)


def test_breakers_off_and_breakers_that_never_fire_change_nothing():
    # Same seed: with breakers that cannot fire (a 97% fall), every random number is
    # used in the same way, so the history must be identical to no breakers at all.
    a = model.simulate(CALM, 300, seed=3)
    b = model.simulate(CALM, 300, seed=3,
                       breakers=dict(levels=(0.97, 0.98, 0.99), halt_minutes=15))
    for k in ("lp", "lf", "z", "x"):
        assert np.array_equal(a[k], b[k])
    assert not a["halts"].any() and not b["halts"].any()


def test_a_day_without_a_halt_never_fell_to_the_first_level():
    o = model.simulate(CALM, 2000, seed=4, breakers=TIGHT)
    h, r = o["halts"], o["r"]
    assert set(np.unique(h)) == {0, 1, 2, 3}           # every level fires at some point
    assert np.all(r[h == 0] > np.log(1 - 0.003))


def test_closing_freezes_the_price_at_the_fall_that_closed_it():
    # A slow price (beta 0.05: about 2.5 ticks per step at most in set I), so one
    # step cannot jump far past the level; after closing, the price must stay there.
    # (Set I, not CALM: CALM is too quiet to close often. With beta this low set I did
    # not run away on any of 39 other seeds.)
    o = model.simulate(dict(model.SET_I, beta=0.05), 2000, seed=4, breakers=TIGHT)
    r, h = o["r"][o["halts"] == 3], o["halts"]
    one_step = 40 * np.log(1 + 0.001)                  # far more ticks than one step makes
    assert np.all(r <= np.log(1 - 0.009))
    assert np.all(r > np.log(1 - 0.009) - one_step)
    assert (h == 3).sum() > 10


def test_a_pause_of_a_whole_day_freezes_the_close_between_the_levels():
    # halt_minutes=390: after the first-level fall the price is frozen for the rest
    # of the day, so the close sits between level 1 and level 2.
    o = model.simulate(CALM, 2000, seed=5, breakers=dict(TIGHT, halt_minutes=390))
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
    (0.1, 0.1, 0.2), (0.1, 0.2, 0.2),
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


def test_minute_to_step_counts_steps_before_a_minute():
    assert model.minute_to_step(355, 0.01) == 91         # 355/390 of 100 steps = 91.03
    assert model.minute_to_step(390, 0.01) == 100
    assert model.minute_to_step(0, 0.005) == 0
    assert model.minute_to_step(195, 0.005) == 100


def test_the_us_rules_include_the_3_25_pm_cutoff():
    assert model.US_BREAKERS == dict(levels=(0.07, 0.13, 0.20), halt_minutes=15, last_pause_minute=355)


def test_a_cutoff_at_the_end_of_the_day_changes_nothing():
    a = model.simulate(CALM, 500, seed=6, breakers=TIGHT)
    b = model.simulate(CALM, 500, seed=6, breakers=dict(TIGHT, last_pause_minute=390))
    for k in ("lp", "lf", "z", "x", "halts"):
        assert np.array_equal(a[k], b[k])


def test_after_the_cutoff_only_the_last_level_stops_trading():
    # Cutoff at minute 0: levels 1 and 2 can never pause, so a day is either open
    # all day (0) or closed at level 3; days can fall past level 1 without a halt.
    o = model.simulate(CALM, 2000, seed=4, breakers=dict(TIGHT, last_pause_minute=0))
    assert set(np.unique(o["halts"])) <= {0, 3} and (o["halts"] == 3).any()
    assert np.any(o["r"][o["halts"] == 0] <= np.log(1 - 0.003))


def test_a_cutoff_only_removes_pauses_that_would_start_late():
    # Same random numbers: the two runs are identical until the first day on which
    # the full rules paused trading (halts records only each day's highest level,
    # so compare the prices, not halts). That day must have a halt in the full run.
    full = model.simulate(CALM, 300, seed=8, breakers=TIGHT)
    cut = model.simulate(CALM, 300, seed=8, breakers=dict(TIGHT, last_pause_minute=195))
    differ = np.flatnonzero(full["lp"] != cut["lp"])
    assert differ.size                                  # the cutoff mattered somewhere
    day = differ[0] - 1                                 # lp[i] is the close of day i - 1
    assert full["halts"][day] >= 1
    assert np.array_equal(full["halts"][:day], cut["halts"][:day])


@pytest.mark.parametrize("bad", [-1, 391, True, float("nan"), "355"])
def test_last_pause_minute_must_be_a_minute_of_the_day(bad):
    with pytest.raises(ValueError, match="last_pause_minute"):
        model.simulate(BASE, 10, breakers=dict(levels=(0.1, 0.2, 0.3), halt_minutes=15,
                                               last_pause_minute=bad))


# ------------------------------------------------------------------ found by the stress test (Nov 2026)

def spx_settings():
    from herding import calibrate
    if not calibrate.settings_path("spx").exists():
        pytest.skip("not fitted yet")
    return calibrate.load("spx")


@pytest.mark.pinned_seed
def test_a_runaway_on_the_last_day_is_reported():
    # seed 2013 leaves its value on day 6,810 (gap above e^20 that day). Before the fix,
    # a run of exactly 6,810 days returned normally because the runaway was on the last day.
    with pytest.raises(model.RunawayError):
        model.simulate(spx_settings(), 6810, seed=2013)
    model.simulate(spx_settings(), 6800, seed=2013)          # 10 days earlier: fine


@pytest.mark.parametrize("change, words", [
    (dict(dt=1 / 15), "trend window"),      # 0.1 day is 1.5 steps: the trend would be 33% too big
    (dict(dt=1e-5), "dt must be at least"),
    (dict(tick=1e-17), "tick must be at least"),   # log(1 + 1e-17) rounds to 0: price frozen
    (dict(N=10**8), "N must be at most")])
def test_settings_the_computer_cannot_handle_are_rejected(change, words):
    with pytest.raises(ValueError, match=words):
        model.check(settings(**change))


def test_absurd_tick_rates_are_a_runaway_not_a_hang():
    # noise 1e20 asks for ~1e22 ticks in one step; numba's Poisson draw would loop forever
    with pytest.raises(model.RunawayError):
        model.simulate(settings(sigma_mu=1e20), 5, seed=1)


# ------------------------------------------------------------------ audit of November 2026
# Issue 2: numpy counts complex numbers as numbers, so they slipped through every check
# (sigma_mu = 0.05+0.3j ran with the 0.3j silently dropped; a complex dt crashed with TypeError).

@pytest.mark.parametrize("key", ["sigma_mu", "dt", "N", "a1"])
@pytest.mark.parametrize("value", [np.complex128(0.05 + 0.3j), np.complex128(1 + 0j), 1 + 0j])
def test_complex_settings_are_rejected(key, value):
    with pytest.raises(ValueError, match="finite number"):
        model.check(settings(**{key: value}))


@pytest.mark.parametrize("breakers", [
    dict(levels=(0.1, 0.2, np.complex128(0.3)), halt_minutes=15),
    dict(levels=(0.1, 0.2, 0.3), halt_minutes=np.complex128(15)),
    dict(levels=(0.1, 0.2, 0.3), halt_minutes=15, last_pause_minute=1j)])
def test_complex_breaker_rules_are_rejected(breakers):
    with pytest.raises(ValueError):
        model.check_breakers(breakers)


def test_complex_shock_is_rejected():
    with pytest.raises(ValueError, match="shock"):
        model.check_shocks({2: np.complex128(0.1)}, 10)


# Issue 5: levels that are not a list gave "'float' object is not iterable"
@pytest.mark.parametrize("levels", [0.1, None, "0.1 0.2 0.3"])
def test_breaker_levels_that_are_not_a_list_give_a_clear_error(levels):
    with pytest.raises(ValueError, match="levels"):
        model.check_breakers(dict(levels=levels, halt_minutes=15))


# Issue 9: check_breakers and check_shocks on their own, valid input included
def test_check_breakers_and_check_shocks_accept_valid_rules():
    assert model.check_breakers(None) is None
    assert model.check_breakers(model.US_BREAKERS) is None
    assert model.check_breakers(dict(levels=[0.01, 0.02, 0.99], halt_minutes=390, last_pause_minute=0)) is None
    assert model.check_shocks(None, 5) is None
    assert model.check_shocks({0: -0.99, 4: 2.0, np.int64(3): 0}, 5) is None
