"""Tests for model.py (the Lux-Marchesi simulation and its guardrails).  Run: python3 -m pytest tests/ -q

Every test states what "correct" means, and uses inputs whose answer we know
independently (math or published facts), across several random seeds.
"""
import numpy as np
import pandas as pd
import pytest

from conftest import ROOT, SEEDS
from herding import model
from herding import stats



def run(days, seed, herding=True, **overrides):
    """simulate() with a given seed and parameter overrides."""
    return model.simulate(days, herding=herding, p=dict(model.P, **overrides), seed=seed)


# ---------------------------------------------------------------- simulate
# Spec: returns (daily log returns, chartist fraction), each of length `days`;
# fraction in [0, 1]; no chartists -> bell-curve returns; same seed -> same run.

def test_simulate_shapes():
    r, f = run(50, 0)
    assert r.shape == (50,) and f.shape == (50,)
    r, f, lp, lf = model.simulate(50, seed=0, paths=True)
    assert lp.shape == lf.shape == (51,) and np.allclose(np.diff(lp), r)


@pytest.mark.parametrize("seed", SEEDS[:3])
def test_price_tracks_fundamental(seed):
    # Fundamentalists must keep the price anchored: mispricing stays bounded
    _, _, lp, lf = model.simulate(3000, seed=seed, paths=True)
    assert np.max(np.abs(lp - lf)) < 0.5


def test_simulate_zero_days():
    r, f = run(0, 0)
    assert len(r) == 0 and len(f) == 0


@pytest.mark.parametrize("seed", SEEDS)
def test_simulate_outputs_are_valid(seed):
    r, f = run(400, seed)
    assert np.all(np.isfinite(r))
    assert np.all((f >= 0) & (f <= 1))
    # chartist fraction must be a whole number of traders out of N
    assert np.allclose(f * model.P["N"], np.round(f * model.P["N"]))


@pytest.mark.parametrize("seed", SEEDS)
def test_simulate_keeps_minimum_chartists(seed):
    _, f = run(400, seed)
    assert f.min() * model.P["N"] >= model.P["min_chartists"] - 1e-9


def test_simulate_no_herding_has_zero_chartists():
    _, f = run(200, 0, herding=False)
    assert np.all(f == 0)


def test_simulate_same_seed_same_result():
    a, _ = run(200, 42)
    b, _ = run(200, 42)
    c, _ = run(200, 43)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


@pytest.mark.parametrize("seed", SEEDS[:3])
def test_base_case_no_chartists_is_bell_curve(seed):
    # The proposal's base case: Gaussian news + no chartists -> kurtosis ~ 0
    r, _ = run(4000, seed, herding=False)
    assert abs(stats.kurtosis(r)) < 0.3


@pytest.mark.parametrize("seed", SEEDS[:3])
def test_herding_fattens_tails(seed):
    herd, _ = run(4000, seed)
    base, _ = run(4000, seed, herding=False)
    assert stats.kurtosis(herd) > stats.kurtosis(base) + 1


@pytest.mark.parametrize("herding", [True, False])
def test_simulate_does_not_depend_on_price_units(herding):
    # Science: quoting the stock in cents instead of dollars must not change
    # returns. Same seed, pf0 = 1, 10, 100 -> identical returns.
    r1, _ = run(1500, 9, herding=herding, pf0=1.0)
    r2, _ = run(1500, 9, herding=herding, pf0=100.0)
    assert np.allclose(r1, r2)


@pytest.mark.filterwarnings("ignore:N=100:UserWarning")
def test_small_market_does_not_blow_up():
    # Regression: N=100, seed 3 used to overflow into NaN
    for seed in range(6):
        r, f = model.simulate(3000, p=dict(model.P, N=100), seed=seed, max_mispricing=None)
        assert np.all(np.isfinite(r))


@pytest.mark.filterwarnings("ignore:N=100:UserWarning")
@pytest.mark.parametrize("seed", SEEDS)
def test_min_chartists_small_market(seed):
    _, f = model.simulate(1500, p=dict(model.P, N=100), seed=seed, max_mispricing=None)
    assert f.min() * 100 >= model.P["min_chartists"] - 1e-9


def test_results_do_not_depend_on_time_step():
    # Numerics: halving dt must not change the science (tail exponent),
    # averaged over seeds to beat noise.
    a = [stats.hill_alpha(run(3000, s, dt=dt)[0][200:]) for dt in (0.02, 0.01)
         for s in range(3)]
    assert np.mean(a[:3]) == pytest.approx(np.mean(a[3:]), abs=0.35)


@pytest.mark.parametrize("seed", SEEDS[:3])
def test_no_chartists_no_fake_volatility_clustering(seed):
    # Without chartists, |r| today must NOT predict |r| tomorrow
    r, _ = run(4000, seed, herding=False)
    a = np.abs(r)
    assert abs(np.corrcoef(a[:-1], a[1:])[0, 1]) < 0.06


@pytest.mark.parametrize("seed", SEEDS)
def test_returns_are_not_predictable_day_to_day(seed):
    # Stylized fact of real markets: lag-1 autocorrelation of returns ~ 0
    r, _ = run(4000, seed)
    ac = np.corrcoef(r[:-1], r[1:])[0, 1]
    assert abs(ac) < 0.1, f"lag-1 autocorrelation of model returns = {ac:.2f}"


@pytest.mark.parametrize("seed", SEEDS)
def test_volatility_clusters_in_model(seed):
    # Stylized fact: |r| today predicts |r| tomorrow (positive autocorrelation)
    r, _ = run(4000, seed)
    a = np.abs(r)
    assert np.corrcoef(a[:-1], a[1:])[0, 1] > 0.1


@pytest.mark.parametrize("seed", SEEDS[:3])
def test_fundamentalists_never_die_out(seed):
    # Regression: with slow switching back (v2=0.5) the market used to end up
    # with zero fundamentalists forever (nobody left to recruit traders back)
    _, f = model.simulate(1500, p=dict(model.P, v2=0.5), seed=seed, max_mispricing=None)
    assert f.max() * model.P["N"] <= model.P["N"] - model.P["min_chartists"] + 1e-9


@pytest.mark.xfail(reason="KNOWN LIMITATION: chartist bubbles unwind over ~2 days, "
                          "giving lag-2 autocorrelation of about -0.17 (S&P: about -0.02)",
                   strict=True)
def test_no_two_day_reversal():
    r, _ = run(5200, 10)
    r = r[200:] - r[200:].mean()
    assert abs(np.corrcoef(r[:-2], r[2:])[0, 1]) < 0.1


@pytest.mark.xfail(reason="KNOWN LIMITATION: volatility memory fades in ~30 days; "
                          "S&P still about 0.1 at 100 days", strict=True)
def test_long_volatility_memory():
    a = np.abs(run(8200, 10)[0][200:])
    a = a - a.mean()
    assert np.corrcoef(a[:-100], a[100:])[0, 1] > 0.05


# ------------------------------------------------- step 1: input validation
# Spec: bad inputs raise ValueError instead of returning silently wrong results.

@pytest.mark.parametrize("days", [-1, -5])
def test_rejects_negative_days(days):
    with pytest.raises(ValueError, match="days"):
        model.simulate(days, seed=0)


@pytest.mark.parametrize("dt", [0.03, 0.3, 0.7, 2.0, 0.0, -0.02])
def test_rejects_dt_that_does_not_divide_a_day(dt):
    with pytest.raises(ValueError, match="dt"):
        model.simulate(5, p=dict(model.P, dt=dt), seed=0)


def test_rejects_dt_above_stability_limit():
    # 1/(beta*gamma) = 1/(50*0.1) = 0.2 -> dt=0.25 divides a day but overshoots
    with pytest.raises(ValueError, match="stab"):
        model.simulate(5, p=dict(model.P, dt=0.25), seed=0)


@pytest.mark.parametrize("dt", [0.01, 0.02, 0.05, 0.1, 0.2])
def test_accepts_valid_dt(dt):
    r, _ = model.simulate(5, p=dict(model.P, dt=dt), seed=0)
    assert len(r) == 5 and np.all(np.isfinite(r))


@pytest.mark.parametrize("N", [0, 5, 20, 39])
def test_rejects_market_too_small_for_floors(N):
    with pytest.raises(ValueError, match="N"):
        model.simulate(5, p=dict(model.P, N=N), seed=0)


@pytest.mark.parametrize("N", [40, 100, 249])
def test_warns_for_small_unstable_market(N):
    with pytest.warns(UserWarning, match="N"):
        model.simulate(5, p=dict(model.P, N=N), seed=0)


def test_no_warning_for_normal_market():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # any warning fails the test
        model.simulate(5, p=dict(model.P, N=250), seed=0)
        model.simulate(5, seed=0)


def test_no_chartist_run_allows_any_positive_N():
    r, f = model.simulate(5, herding=False, p=dict(model.P, N=10), seed=0)
    assert np.all(f == 0) and np.all(np.isfinite(r))


@pytest.mark.parametrize("m", [3, 0, -2])
def test_rejects_bad_min_chartists(m):
    with pytest.raises(ValueError, match="min_chartists"):
        model.simulate(5, p=dict(model.P, min_chartists=m), seed=0)


@pytest.mark.parametrize("N", [40, 41, 57, 100, 250, 500, 501, 2000])
def test_initial_counts_respect_floors(N):
    m = model.P["min_chartists"]
    nf, npl, nmi = model.initial_counts(N, True, m)
    assert nf + npl + nmi == N                      # nobody lost or invented
    assert npl >= m // 2 and nmi >= m // 2          # each chartist type at floor
    assert nf >= m                                  # fundamentalist floor
    assert npl == nmi                               # start with no opinion bias
    assert abs((npl + nmi) / N - 0.1) <= 2 / N + 1e-12   # about 10% chartists


def test_initial_counts_without_herding():
    assert model.initial_counts(37, False, 4) == (37, 0, 0)


@pytest.mark.parametrize("N", [40, 60])
@pytest.mark.parametrize("seed", SEEDS[:3])
def test_small_market_stays_within_floors(N, seed):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _, f = model.simulate(800, p=dict(model.P, N=N), seed=seed, max_mispricing=None)
    m = model.P["min_chartists"]
    counts = np.round(f * N)
    assert counts.min() >= m and counts.max() <= N - m


def test_no_chartist_run_rejects_empty_market():
    with pytest.raises(ValueError, match="N"):
        model.simulate(5, herding=False, p=dict(model.P, N=0), seed=0)


# ------------------------------------------- step 2: every setting has an effect
# Spec: each parameter changes the model in the direction the science predicts.
# Paired seeds (same seeds for low and high), averaged over 3 seeds.

def run_stats(seed, herding=True, **over):
    r, f, lp, lf = model.simulate(2200, herding, dict(model.P, **over), seed=seed, paths=True)
    r, f = r[200:], f[200:]
    x = r - r.mean()
    return dict(std=r.std(), mis=np.abs(lp - lf)[200:].mean(), frac=f.mean(),
                kurt=stats.kurtosis(r), ac1=np.corrcoef(x[:-1], x[1:])[0, 1])


def mean_stat(key, value, measure, herding=True):
    return np.mean([run_stats(s, herding, **{key: value})[measure] for s in range(3)])


def test_news_size_sets_volatility_exactly():
    # No chartists: the model is linear in the news, so doubling sigma_news must
    # double every return exactly; volatility ~ sigma_news (a bit lower because
    # the price lags the fundamental by a fraction of a day).
    r1, _ = model.simulate(2000, False, dict(model.P, sigma_news=0.01), seed=5)
    r2, _ = model.simulate(2000, False, dict(model.P, sigma_news=0.02), seed=5)
    assert np.allclose(r2, 2 * r1)
    assert r1.std() == pytest.approx(0.01, rel=0.15)


def test_herding_a1_increases_mispricing():
    # Chartists who agree more push the price further from its value
    assert mean_stat("a1", 0.9, "mis") > mean_stat("a1", 0.2, "mis")


def test_trend_following_a2_fattens_tails_and_volatility():
    assert mean_stat("a2", 4.0, "kurt") > mean_stat("a2", 0.0, "kurt")
    assert mean_stat("a2", 4.0, "std") > mean_stat("a2", 0.0, "std")


def test_chartist_trade_size_tc_increases_mispricing_and_volatility():
    assert mean_stat("tc", 0.2, "mis") > mean_stat("tc", 0.05, "mis")
    assert mean_stat("tc", 0.2, "std") > mean_stat("tc", 0.05, "std")


def test_faster_price_adjustment_beta_tracks_fundamental_closer():
    # No chartists: a faster market maker closes the gap to the fundamental
    # sooner, so mispricing and day-to-day predictability both drop
    assert mean_stat("beta", 100.0, "mis", False) < mean_stat("beta", 25.0, "mis", False)
    assert mean_stat("beta", 100.0, "ac1", False) < mean_stat("beta", 25.0, "ac1", False)


def test_stronger_fundamentalists_gamma_reduce_mispricing():
    assert mean_stat("gamma", 0.2, "mis") < mean_stat("gamma", 0.05, "mis")


def test_mispricing_penalty_s_reduces_chartists():
    assert mean_stat("s", 2.0, "frac") < mean_stat("s", 0.75, "frac")


@pytest.mark.parametrize("key", ["a1", "a2", "a3", "v1", "v2", "R", "s", "tc",
                                 "gamma", "beta", "sigma_news", "tau"])
def test_every_setting_is_actually_used(key):
    # Silent-failure trap: a setting the code ignores would leave output unchanged.
    # R (interest rate) is the exception that proves the rule: dividends and
    # interest nearly cancel near equilibrium, as in Lux-Marchesi, so R only
    # matters when made ~100x larger. Tested that way, with that reason.
    factor, days = (100.0, 300) if key == "R" else (0.5, 60)
    r1, _ = model.simulate(days, p=model.P, seed=3)
    r2, _ = model.simulate(days, p=dict(model.P, **{key: model.P[key] * factor}), seed=3)
    assert not np.allclose(r1, r2), f"changing {key} had no effect"



# ------------------------------------------------------------ guardrails
# Spec: mistakes in settings, unstable settings, and bad data fail loudly with a
# message that says what to change -- never a silent wrong answer or NaN.

def test_rejects_misspelled_setting():
    with pytest.raises(ValueError, match="sgma_news"):
        model.simulate(5, p=dict(model.P, sgma_news=0.02), seed=0)


def test_rejects_missing_setting():
    p = dict(model.P)
    del p["tc"]
    with pytest.raises(ValueError, match="tc"):
        model.simulate(5, p=p, seed=0)


@pytest.mark.parametrize("key", ["v1", "v2", "beta", "gamma", "tc", "sigma_news", "tau", "pf0"])
@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_rejects_nonpositive_rates_and_sizes(key, bad):
    with pytest.raises(ValueError, match=key):
        model.simulate(5, p=dict(model.P, **{key: bad}), seed=0)


@pytest.mark.parametrize("key", ["a1", "a2", "a3", "s", "R"])
def test_rejects_negative_sensitivities(key):
    with pytest.raises(ValueError, match=key):
        model.simulate(5, p=dict(model.P, **{key: -0.1}), seed=0)


def test_zero_sensitivities_are_allowed():
    # a1 = a2 = 0 is a meaningful experiment (no herding / no trend chasing)
    r, _ = model.simulate(20, p=dict(model.P, a1=0.0, a2=0.0, R=0.0), seed=0)
    assert np.all(np.isfinite(r))


@pytest.mark.parametrize("bad", [np.nan, np.inf, "0.6", None])
def test_rejects_non_numbers(bad):
    with pytest.raises(ValueError, match="a1"):
        model.simulate(5, p=dict(model.P, a1=bad), seed=0)


@pytest.mark.parametrize("N", [500.5, 2.5e2 + 0.1])
def test_rejects_fractional_trader_count(N):
    with pytest.raises(ValueError, match="N"):
        model.simulate(5, p=dict(model.P, N=N), seed=0)


@pytest.mark.parametrize("over", [dict(s=0.25), dict(s=0.35)])
def test_runtime_guard_stops_runaway_market(over):
    # These settings look normal for thousands of days, then everyone becomes a
    # chartist and the price runs 12-16 log units from its value (seed 201:
    # day 3031 for s=0.25, day 4273 for s=0.35). Healthy settings never exceed
    # 0.6. The guard must stop the run instead of returning nonsense.
    with pytest.raises(model.ModelRegimeError, match="away from its value"):
        model.simulate(5000, p=dict(model.P, **over), seed=201)


@pytest.mark.parametrize("over", [{}, dict(a1=1.2), dict(tc=0.2), dict(gamma=0.05),
                                  dict(a2=4.0), dict(s=0.5)])
@pytest.mark.parametrize("seed", [200, 201])
def test_runtime_guard_never_fires_on_healthy_settings(over, seed):
    r, _ = model.simulate(3000, p=dict(model.P, **over), seed=seed)
    assert np.all(np.isfinite(r))


def test_runtime_guard_can_be_switched_off_for_research():
    r, _ = model.simulate(300, p=dict(model.P, s=0.25), seed=200, max_mispricing=None)
    assert len(r) == 300



# ------------------------------------------------- refactor and speed safety
# Spec: splitting main.py into modules and adding numba changed nothing.

P_V1 = dict(N=500, v1=3.0, v2=2.0, beta=50.0, tc=0.1, gamma=0.1, a1=0.6, a2=2.0,
            a3=0.5, R=0.0004, s=0.75, sigma_news=0.01, pf0=10.0, dt=0.02,
            min_chartists=4, tau=0.2)
GOLDEN_CASES = [(True, {}, 0), (True, {}, 7), (False, {}, 3),
                (True, dict(a1=1.2, tc=0.2), 5), (True, dict(N=250), 11)]


@pytest.mark.filterwarnings("ignore:N=250")
@pytest.mark.parametrize("i", range(len(GOLDEN_CASES)))
def test_matches_golden_output_of_original_program(i):
    # golden_v1.npz was saved from the single-file main.py before the split
    G = np.load(ROOT / "tests/golden_v1.npz")
    herding, over, seed = GOLDEN_CASES[i]
    r, f, lp, lf = model.simulate(600, herding, dict(P_V1, **over), seed=seed, paths=True)
    for got, key in [(r, "r"), (f, "f"), (lp, "lp"), (lf, "lf")]:
        assert np.array_equal(got, G[f"{key}{i}"]), f"{key} differs in case {i}"


@pytest.mark.parametrize("herding", [True, False])
def test_numba_and_plain_python_agree_exactly(herding):
    # Same code, same random numbers: the fast and the portable path must match
    N, m = 500, 4
    nf, npl, nmi = model.initial_counts(N, herding, m)
    # with circuit breakers on, so the halt logic is compared too
    args = (300, 50, herding, N, nf, npl, nmi, m // 2 if herding else 0, m, np.log(10.0),
            3.0, 2.0, 50.0, 0.1, 0.1, 0.6, 2.0, 0.5, 0.0004, 0.75, 0.05, 0.2, -1.0,
            np.log(0.93), np.log(0.87), np.log(0.80), 2, np.where(np.arange(300) == 150, -0.05, 0.0))
    fast = model._run(np.random.default_rng(9), *args)
    plain = _plain_run(np.random.default_rng(9), *args)
    for a, b in zip(fast[:5], plain[:5]):
        assert np.array_equal(a, b)
    assert fast[5:] == plain[5:]


def _plain_run(g, *args):
    """Run the loop as plain Python even when numba is installed."""
    import importlib, os
    os.environ["MODEL_NO_JIT"] = "1"
    try:
        plain = importlib.reload(model)
        return plain._run(g, *args)
    finally:
        del os.environ["MODEL_NO_JIT"]
        importlib.reload(model)


def test_runs_fast_enough_for_calibration():
    # Calibration needs thousands of 30-year runs: one must take well under 1 s
    import time
    if not model.USE_JIT:
        pytest.skip("numba not installed")
    model.simulate(10, seed=0)                          # compile once
    t = time.perf_counter()
    model.simulate(8200, seed=1)
    assert time.perf_counter() - t < 1.0


# ------------------------------------------------------------ circuit breakers
# Spec (US rules since 2013, simplified): intraday fall of 7% or 13% below the
# previous close -> 15-minute pause (price frozen; traders and news continue);
# 20% -> closed for the rest of the day. Each level fires at most once per day.
# With breakers off, nothing changes (bit-identical).

US_RULES = dict(levels=(0.07, 0.13, 0.20), halt_minutes=15)
WILD = dict(model.P, sigma_news=0.05)          # forces big intraday falls (all 3 seeds hit -20%)


def test_breakers_off_is_bit_identical():
    a = model.simulate(800, seed=3)
    b = model.simulate(800, seed=3, breakers=None)
    out = model.simulate_full(800, seed=3)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert np.array_equal(out["r"], a[0]) and np.all(out["halts"] == 0)


def test_breakers_that_never_fire_change_nothing():
    a = model.simulate_full(800, seed=4)
    b = model.simulate_full(800, seed=4, breakers=dict(levels=(0.9, 0.95, 0.99), halt_minutes=15))
    assert np.array_equal(a["r"], b["r"]) and np.all(b["halts"] == 0)


@pytest.mark.parametrize("seed", SEEDS[:3])
def test_no_day_falls_much_beyond_the_20pct_close(seed):
    out = model.simulate_full(3000, p=WILD, seed=seed, breakers=US_RULES, max_mispricing=None)
    assert out["halts"].max() == 3, "test needs a level-3 day to be meaningful"
    # trading stops at the first step past -20%, so the close is at most one step lower
    assert out["r"].min() > np.log(0.80) - out["max_step"]


@pytest.mark.parametrize("seed", SEEDS[:3])
def test_closing_below_a_level_means_it_fired(seed):
    out = model.simulate_full(3000, p=WILD, seed=seed, breakers=US_RULES, max_mispricing=None)
    r, h = out["r"], out["halts"]
    assert np.all(h[r <= np.log(1 - 0.07)] >= 1)
    assert np.all(h[r <= np.log(1 - 0.13)] >= 2)
    assert set(np.unique(h)) <= {0, 1, 2, 3}


def test_days_that_never_fell_7pct_never_halt():
    # a day can fall, pause and recover, but a day whose LOW stayed above -7% cannot halt
    out = model.simulate_full(3000, p=WILD, seed=1, breakers=US_RULES, max_mispricing=None)
    assert np.all(out["halts"][out["low"] > np.log(1 - 0.07)] == 0)
    assert np.all(out["low"] <= np.minimum(out["r"], 0) + 1e-12)   # low is never above close


def test_halt_freezes_the_price():
    # A full-day closure from the first step: price must not move after it fires
    out = model.simulate_full(400, p=WILD, seed=2, breakers=dict(levels=(1e-9, 2e-9, 3e-9),
                              halt_minutes=15), max_mispricing=None)
    # every day's close sits within one step of the level-3 trigger or above it
    assert np.all(out["r"] > -3e-9 - out["max_step"])


@pytest.mark.parametrize("bad", [dict(levels=(0.13, 0.07, 0.20), halt_minutes=15),
                                 dict(levels=(0.07, 0.13, 1.2), halt_minutes=15),
                                 dict(levels=(0.0, 0.13, 0.2), halt_minutes=15),
                                 dict(levels=(0.07, 0.13), halt_minutes=15),
                                 dict(levels=(0.07, 0.13, 0.2), halt_minutes=0),
                                 dict(levels=(0.07, 0.13, 0.2), halt_minutes=15, extra=1),
                                 dict(levels=(0.07, 0.13, 0.2))])
def test_rejects_bad_breaker_rules(bad):
    with pytest.raises(ValueError, match="breakers"):
        model.simulate(10, seed=0, breakers=bad)


@pytest.mark.parametrize("seed", SEEDS[:3])
def test_breakers_change_nothing_before_the_first_pause(seed):
    # Paired runs share every random number, so until a breaker first fires the
    # two histories must be identical; afterwards they may differ
    off = model.simulate_full(3000, p=WILD, seed=seed, max_mispricing=None)
    on = model.simulate_full(3000, p=WILD, seed=seed, breakers=US_RULES, max_mispricing=None)
    first = int(np.argmax(on["halts"] > 0))
    assert on["halts"][first] > 0
    assert np.array_equal(off["r"][:first], on["r"][:first])
    assert not np.array_equal(off["r"][first:], on["r"][first:])


# ------------------------------------------------------------- policy shocks
# Spec: shocks={day: change} moves the fundamental value by that fraction at the
# start of that day (e.g. -0.05 = a policy surprise that cuts value 5%).
# No shocks -> bit-identical. Shocks use no random numbers, so paired runs
# differ only through the shock.

def test_no_shocks_is_bit_identical():
    a = model.simulate_full(500, seed=5)
    b = model.simulate_full(500, seed=5, shocks={})
    assert np.array_equal(a["r"], b["r"]) and np.array_equal(a["lf"], b["lf"])


@pytest.mark.parametrize("seed", SEEDS[:3])
def test_shock_moves_value_by_exact_amount(seed):
    # Same seed -> same news; the value path must differ by exactly log(1 - 0.05)
    # from the shock day on, and not at all before it
    a = model.simulate_full(400, herding=False, seed=seed)
    b = model.simulate_full(400, herding=False, seed=seed, shocks={200: -0.05})
    d = b["lf"] - a["lf"]
    assert np.allclose(d[:201], 0) and np.allclose(d[201:], np.log(0.95))


def test_without_chartists_price_follows_the_shock_fully():
    # Fundamentalists alone: the price ends up exactly as far down as the value
    a = model.simulate_full(400, herding=False, seed=1)
    b = model.simulate_full(400, herding=False, seed=1, shocks={200: -0.05})
    gap = (b["lp"] - a["lp"])[230:]
    assert np.allclose(gap, np.log(0.95), atol=1e-3)


def test_several_shocks_add_up():
    a = model.simulate_full(300, herding=False, seed=2)
    b = model.simulate_full(300, herding=False, seed=2, shocks={50: -0.05, 100: 0.10})
    assert np.allclose((b["lf"] - a["lf"])[101:], np.log(0.95) + np.log(1.10))


@pytest.mark.parametrize("bad", [{-1: -0.05}, {500: -0.05}, {10: -1.0}, {10: -1.5},
                                 {10: float("nan")}, {2.5: -0.05}, {10: "big"}])
def test_rejects_bad_shocks(bad):
    with pytest.raises(ValueError, match="shock"):
        model.simulate(300, seed=0, shocks=bad)
