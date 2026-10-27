"""Unit tests for main.py. Run:  python3 -m pytest tests/ -v

Every test states what "correct" means, and uses inputs whose answer we know
independently (math or published facts), across several random seeds.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import main  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]


def run(days, seed, herding=True, **overrides):
    """simulate() with a given seed and parameter overrides."""
    return main.simulate(days, herding=herding, p=dict(main.P, **overrides), seed=seed)


# ---------------------------------------------------------------- kurtosis
# Spec: excess kurtosis = 0 for a bell curve, -1.2 uniform, +3 Laplace.

@pytest.mark.parametrize("seed", SEEDS)
def test_kurtosis_gaussian_is_zero(seed):
    r = np.random.default_rng(seed).standard_normal(200_000)
    assert abs(main.kurtosis(r)) < 0.05


@pytest.mark.parametrize("seed", SEEDS)
def test_kurtosis_uniform_is_minus_1_2(seed):
    r = np.random.default_rng(seed).uniform(-1, 1, 200_000)
    assert main.kurtosis(r) == pytest.approx(-1.2, abs=0.03)


@pytest.mark.parametrize("seed", SEEDS)
def test_kurtosis_laplace_is_3(seed):
    r = np.random.default_rng(seed).laplace(0, 1, 500_000)
    assert main.kurtosis(r) == pytest.approx(3.0, abs=0.2)


def test_kurtosis_exact_two_points():
    # z = [-1, 1] -> mean(z^4) = 1 -> excess kurtosis = -2, by hand
    assert main.kurtosis(np.array([-5.0, 5.0])) == pytest.approx(-2.0)


def test_kurtosis_ignores_shift_and_scale():
    r = np.random.default_rng(7).standard_t(5, 10_000)
    assert main.kurtosis(3 * r + 100) == pytest.approx(main.kurtosis(r))


def test_kurtosis_constant_input_is_rejected():
    # Edge: zero variance has no defined kurtosis -> clear error, not nan
    with pytest.raises(ValueError, match="constant"):
        main.kurtosis(np.ones(50))


# -------------------------------------------------------------- hill_alpha
# Spec: for |r| with a Pareto tail P(|r|>x) ~ x^-alpha, return about alpha.

@pytest.mark.parametrize("alpha", [1.5, 3.0, 5.0])
@pytest.mark.parametrize("seed", SEEDS)
def test_hill_recovers_known_pareto_exponent(alpha, seed):
    rng = np.random.default_rng(seed)
    r = (rng.pareto(alpha, 100_000) + 1) * rng.choice([-1, 1], 100_000)
    assert main.hill_alpha(r) == pytest.approx(alpha, rel=0.08)


@pytest.mark.parametrize("seed", SEEDS)
def test_hill_student_t3_tail_is_about_3(seed):
    # Student-t with 3 degrees of freedom has tail exponent exactly 3
    r = np.random.default_rng(seed).standard_t(3, 200_000)
    assert main.hill_alpha(r, tail=0.01) == pytest.approx(3.0, rel=0.12)


def test_hill_gaussian_gives_large_alpha():
    # A bell curve has no power-law tail: Hill should report a big number
    r = np.random.default_rng(0).standard_normal(200_000)
    assert main.hill_alpha(r) > 5


def test_hill_ignores_sign_and_scale():
    r = np.random.default_rng(3).standard_t(3, 20_000)
    assert main.hill_alpha(-7 * r) == pytest.approx(main.hill_alpha(r))


def test_hill_smaller_alpha_means_fatter_tail():
    rng = np.random.default_rng(5)
    fat, thin = rng.pareto(2, 50_000) + 1, rng.pareto(4, 50_000) + 1
    assert main.hill_alpha(fat) < main.hill_alpha(thin)


def test_hill_too_short_input_fails_loudly():
    # Edge: fewer than 11 values -> a clear error, not a crash deep inside
    with pytest.raises(ValueError):
        main.hill_alpha(np.arange(1, 6, dtype=float))


def test_hill_with_many_zero_returns():
    # Edge the spec is silent on: real data has flat days (r == 0).
    # If the k-th largest |r| is 0 the estimator divides by zero.
    r = np.zeros(1000)
    r[:20] = np.arange(1, 21)
    with np.errstate(all="ignore"):
        a = main.hill_alpha(r)
    assert np.isfinite(a), "hill_alpha returns non-finite when many returns are 0"


# -------------------------------------------------------------------- ccdf
# Spec: x sorted ascending; P(|z| >= x) runs from 1 down to 1/n.

def test_ccdf_exact_small_case():
    z, p = main.ccdf(np.array([-2.0, 0.0, 2.0]))
    std = np.std([-2, 0, 2])
    assert np.allclose(z, [0, 2 / std, 2 / std])
    assert np.allclose(p, [1, 2 / 3, 1 / 3])


@pytest.mark.parametrize("seed", SEEDS)
def test_ccdf_gaussian_matches_theory(seed):
    z, p = main.ccdf(np.random.default_rng(seed).standard_normal(200_000))
    # P(|Z| > 2) = 0.0455 for a standard normal
    assert np.interp(2.0, z, p) == pytest.approx(0.0455, abs=0.003)


def test_ccdf_is_monotone():
    z, p = main.ccdf(np.random.default_rng(1).standard_t(3, 5000))
    assert np.all(np.diff(z) >= 0) and np.all(np.diff(p) < 0)
    assert p[0] == 1 and p[-1] == pytest.approx(1 / 5000)


# ---------------------------------------------------------------- simulate
# Spec: returns (daily log returns, chartist fraction), each of length `days`;
# fraction in [0, 1]; no chartists -> bell-curve returns; same seed -> same run.

def test_simulate_shapes():
    r, f = run(50, 0)
    assert r.shape == (50,) and f.shape == (50,)
    r, f, lp, lf = main.simulate(50, seed=0, paths=True)
    assert lp.shape == lf.shape == (51,) and np.allclose(np.diff(lp), r)


@pytest.mark.parametrize("seed", SEEDS[:3])
def test_price_tracks_fundamental(seed):
    # Fundamentalists must keep the price anchored: mispricing stays bounded
    _, _, lp, lf = main.simulate(3000, seed=seed, paths=True)
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
    assert np.allclose(f * main.P["N"], np.round(f * main.P["N"]))


@pytest.mark.parametrize("seed", SEEDS)
def test_simulate_keeps_minimum_chartists(seed):
    _, f = run(400, seed)
    assert f.min() * main.P["N"] >= main.P["min_chartists"] - 1e-9


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
    assert abs(main.kurtosis(r)) < 0.3


@pytest.mark.parametrize("seed", SEEDS[:3])
def test_herding_fattens_tails(seed):
    herd, _ = run(4000, seed)
    base, _ = run(4000, seed, herding=False)
    assert main.kurtosis(herd) > main.kurtosis(base) + 1


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
        r, f = main.simulate(3000, p=dict(main.P, N=100), seed=seed, max_mispricing=None)
        assert np.all(np.isfinite(r))


@pytest.mark.filterwarnings("ignore:N=100:UserWarning")
@pytest.mark.parametrize("seed", SEEDS)
def test_min_chartists_small_market(seed):
    _, f = main.simulate(1500, p=dict(main.P, N=100), seed=seed, max_mispricing=None)
    assert f.min() * 100 >= main.P["min_chartists"] - 1e-9


def test_results_do_not_depend_on_time_step():
    # Numerics: halving dt must not change the science (tail exponent),
    # averaged over seeds to beat noise.
    a = [main.hill_alpha(run(3000, s, dt=dt)[0][200:]) for dt in (0.02, 0.01)
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


# --------------------------------------------------------------- real data
# Spec: data files are clean and match known historical values.

@pytest.fixture(scope="module")
def spx():
    return pd.read_csv(ROOT / "data/spx_daily.csv", parse_dates=["Date"], index_col="Date")


def test_spx_clean(spx):
    assert spx.index.is_monotonic_increasing and not spx.index.duplicated().any()
    assert spx["Close"].notna().all() and (spx["Close"] > 0).all()


def test_spx_black_monday_known_value(spx):
    # Published: S&P 500 closed at 282.70 on 1987-10-16 and 224.84 on 1987-10-19 (-20.47%)
    assert spx.loc["1987-10-16", "Close"] == pytest.approx(282.70, abs=0.01)
    assert spx.loc["1987-10-19", "Close"] == pytest.approx(224.84, abs=0.01)


def test_spx_no_large_gaps(spx):
    gaps = spx.index.to_series().diff().dt.days
    assert gaps.max() <= 7, f"largest gap {gaps.max()} days at {gaps.idxmax().date()}"


def test_vix_known_peak():
    vix = pd.read_csv(ROOT / "data/vix_daily.csv", parse_dates=["Date"], index_col="Date")
    # Published record close: 82.69 on 2020-03-16
    assert vix["Close"].idxmax().date().isoformat() == "2020-03-16"
    assert vix["Close"].max() == pytest.approx(82.69, abs=0.01)


def test_recessions_match_nber():
    rec = pd.read_csv(ROOT / "data/us_recessions_monthly.csv", parse_dates=["observation_date"])
    rec = rec.set_index("observation_date")["USREC"]
    for month in ["2008-06-01", "2020-04-01", "1974-06-01"]:
        assert rec[month] == 1
    for month in ["2019-06-01", "1995-06-01"]:
        assert rec[month] == 0


@pytest.mark.parametrize("seed", SEEDS[:3])
def test_fundamentalists_never_die_out(seed):
    # Regression: with slow switching back (v2=0.5) the market used to end up
    # with zero fundamentalists forever (nobody left to recruit traders back)
    _, f = main.simulate(1500, p=dict(main.P, v2=0.5), seed=seed, max_mispricing=None)
    assert f.max() * main.P["N"] <= main.P["N"] - main.P["min_chartists"] + 1e-9


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
        main.simulate(days, seed=0)


@pytest.mark.parametrize("dt", [0.03, 0.3, 0.7, 2.0, 0.0, -0.02])
def test_rejects_dt_that_does_not_divide_a_day(dt):
    with pytest.raises(ValueError, match="dt"):
        main.simulate(5, p=dict(main.P, dt=dt), seed=0)


def test_rejects_dt_above_stability_limit():
    # 1/(beta*gamma) = 1/(50*0.1) = 0.2 -> dt=0.25 divides a day but overshoots
    with pytest.raises(ValueError, match="stab"):
        main.simulate(5, p=dict(main.P, dt=0.25), seed=0)


@pytest.mark.parametrize("dt", [0.01, 0.02, 0.05, 0.1, 0.2])
def test_accepts_valid_dt(dt):
    r, _ = main.simulate(5, p=dict(main.P, dt=dt), seed=0)
    assert len(r) == 5 and np.all(np.isfinite(r))


@pytest.mark.parametrize("N", [0, 5, 20, 39])
def test_rejects_market_too_small_for_floors(N):
    with pytest.raises(ValueError, match="N"):
        main.simulate(5, p=dict(main.P, N=N), seed=0)


@pytest.mark.parametrize("N", [40, 100, 249])
def test_warns_for_small_unstable_market(N):
    with pytest.warns(UserWarning, match="N"):
        main.simulate(5, p=dict(main.P, N=N), seed=0)


def test_no_warning_for_normal_market():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # any warning fails the test
        main.simulate(5, p=dict(main.P, N=250), seed=0)
        main.simulate(5, seed=0)


def test_no_chartist_run_allows_any_positive_N():
    r, f = main.simulate(5, herding=False, p=dict(main.P, N=10), seed=0)
    assert np.all(f == 0) and np.all(np.isfinite(r))


@pytest.mark.parametrize("m", [3, 0, -2])
def test_rejects_bad_min_chartists(m):
    with pytest.raises(ValueError, match="min_chartists"):
        main.simulate(5, p=dict(main.P, min_chartists=m), seed=0)


@pytest.mark.parametrize("N", [40, 41, 57, 100, 250, 500, 501, 2000])
def test_initial_counts_respect_floors(N):
    m = main.P["min_chartists"]
    nf, npl, nmi = main.initial_counts(N, True, m)
    assert nf + npl + nmi == N                      # nobody lost or invented
    assert npl >= m // 2 and nmi >= m // 2          # each chartist type at floor
    assert nf >= m                                  # fundamentalist floor
    assert npl == nmi                               # start with no opinion bias
    assert abs((npl + nmi) / N - 0.1) <= 2 / N + 1e-12   # about 10% chartists


def test_initial_counts_without_herding():
    assert main.initial_counts(37, False, 4) == (37, 0, 0)


@pytest.mark.parametrize("N", [40, 60])
@pytest.mark.parametrize("seed", SEEDS[:3])
def test_small_market_stays_within_floors(N, seed):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _, f = main.simulate(800, p=dict(main.P, N=N), seed=seed, max_mispricing=None)
    m = main.P["min_chartists"]
    counts = np.round(f * N)
    assert counts.min() >= m and counts.max() <= N - m


def test_no_chartist_run_rejects_empty_market():
    with pytest.raises(ValueError, match="N"):
        main.simulate(5, herding=False, p=dict(main.P, N=0), seed=0)


# ------------------------------------------- step 2: every setting has an effect
# Spec: each parameter changes the model in the direction the science predicts.
# Paired seeds (same seeds for low and high), averaged over 3 seeds.

def stats(seed, herding=True, **over):
    r, f, lp, lf = main.simulate(2200, herding, dict(main.P, **over), seed=seed, paths=True)
    r, f = r[200:], f[200:]
    x = r - r.mean()
    return dict(std=r.std(), mis=np.abs(lp - lf)[200:].mean(), frac=f.mean(),
                kurt=main.kurtosis(r), ac1=np.corrcoef(x[:-1], x[1:])[0, 1])


def mean_stat(key, value, measure, herding=True):
    return np.mean([stats(s, herding, **{key: value})[measure] for s in range(3)])


def test_news_size_sets_volatility_exactly():
    # No chartists: the model is linear in the news, so doubling sigma_news must
    # double every return exactly; volatility ~ sigma_news (a bit lower because
    # the price lags the fundamental by a fraction of a day).
    r1, _ = main.simulate(2000, False, dict(main.P, sigma_news=0.01), seed=5)
    r2, _ = main.simulate(2000, False, dict(main.P, sigma_news=0.02), seed=5)
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
    r1, _ = main.simulate(days, p=main.P, seed=3)
    r2, _ = main.simulate(days, p=dict(main.P, **{key: main.P[key] * factor}), seed=3)
    assert not np.allclose(r1, r2), f"changing {key} had no effect"



# ------------------------------------------------------------ guardrails
# Spec: mistakes in settings, unstable settings, and bad data fail loudly with a
# message that says what to change -- never a silent wrong answer or NaN.

def test_rejects_misspelled_setting():
    with pytest.raises(ValueError, match="sgma_news"):
        main.simulate(5, p=dict(main.P, sgma_news=0.02), seed=0)


def test_rejects_missing_setting():
    p = dict(main.P)
    del p["tc"]
    with pytest.raises(ValueError, match="tc"):
        main.simulate(5, p=p, seed=0)


@pytest.mark.parametrize("key", ["v1", "v2", "beta", "gamma", "tc", "sigma_news", "tau", "pf0"])
@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_rejects_nonpositive_rates_and_sizes(key, bad):
    with pytest.raises(ValueError, match=key):
        main.simulate(5, p=dict(main.P, **{key: bad}), seed=0)


@pytest.mark.parametrize("key", ["a1", "a2", "a3", "s", "R"])
def test_rejects_negative_sensitivities(key):
    with pytest.raises(ValueError, match=key):
        main.simulate(5, p=dict(main.P, **{key: -0.1}), seed=0)


def test_zero_sensitivities_are_allowed():
    # a1 = a2 = 0 is a meaningful experiment (no herding / no trend chasing)
    r, _ = main.simulate(20, p=dict(main.P, a1=0.0, a2=0.0, R=0.0), seed=0)
    assert np.all(np.isfinite(r))


@pytest.mark.parametrize("bad", [np.nan, np.inf, "0.6", None])
def test_rejects_non_numbers(bad):
    with pytest.raises(ValueError, match="a1"):
        main.simulate(5, p=dict(main.P, a1=bad), seed=0)


@pytest.mark.parametrize("N", [500.5, 2.5e2 + 0.1])
def test_rejects_fractional_trader_count(N):
    with pytest.raises(ValueError, match="N"):
        main.simulate(5, p=dict(main.P, N=N), seed=0)


@pytest.mark.parametrize("over", [dict(s=0.25), dict(s=0.35)])
def test_runtime_guard_stops_runaway_market(over):
    # These settings look normal for thousands of days, then everyone becomes a
    # chartist and the price runs 12-16 log units from its value (seed 201:
    # day 3031 for s=0.25, day 4273 for s=0.35). Healthy settings never exceed
    # 0.6. The guard must stop the run instead of returning nonsense.
    with pytest.raises(main.ModelRegimeError, match="away from its value"):
        main.simulate(5000, p=dict(main.P, **over), seed=201)


@pytest.mark.parametrize("over", [{}, dict(a1=1.2), dict(tc=0.2), dict(gamma=0.05),
                                  dict(a2=4.0), dict(s=0.5)])
@pytest.mark.parametrize("seed", [200, 201])
def test_runtime_guard_never_fires_on_healthy_settings(over, seed):
    r, _ = main.simulate(3000, p=dict(main.P, **over), seed=seed)
    assert np.all(np.isfinite(r))


def test_runtime_guard_can_be_switched_off_for_research():
    r, _ = main.simulate(300, p=dict(main.P, s=0.25), seed=200, max_mispricing=None)
    assert len(r) == 300


@pytest.mark.parametrize("fn", [main.kurtosis, main.hill_alpha, main.ccdf])
@pytest.mark.parametrize("bad", [[], [1.0, np.nan, 2.0] * 10, [1.0, np.inf, 2.0] * 10])
def test_statistics_reject_empty_or_nonfinite_input(fn, bad):
    with pytest.raises(ValueError):
        fn(np.array(bad, dtype=float))


def test_ccdf_rejects_constant_input():
    with pytest.raises(ValueError, match="constant"):
        main.ccdf(np.ones(20))


@pytest.mark.parametrize("tail", [0.0, -0.1, 1.0, 1.5])
def test_hill_rejects_tail_outside_0_1(tail):
    with pytest.raises(ValueError, match="tail"):
        main.hill_alpha(np.random.default_rng(0).standard_t(3, 1000), tail=tail)


def test_load_spx_missing_file_explains_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="yfinance"):
        main.load_spx(tmp_path / "nope.csv")


def test_load_spx_rejects_bad_prices(tmp_path):
    f = tmp_path / "bad.csv"
    f.write_text("Date,Close\n2020-01-02,100\n2020-01-03,-5\n2020-01-06,101\n")
    with pytest.raises(ValueError, match="positive"):
        main.load_spx(f)


def test_load_spx_real_file_ok():
    spx = main.load_spx(ROOT / "data/spx_daily.csv")
    assert len(spx) > 19000 and spx.index.is_monotonic_increasing


def test_hill_rejects_constant_tail():
    r = np.ones(200)
    r[:5] = 0.5                                      # tail (top 10) all equal to 1
    with pytest.raises(ValueError, match="constant"):
        main.hill_alpha(r)
