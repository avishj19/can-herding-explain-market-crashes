"""Tests for stats.py (tail and clustering measurements).  Run: python3 -m pytest tests/ -q

Every test states what "correct" means, and uses inputs whose answer we know
independently (math or published facts), across several random seeds.
"""
import numpy as np
import pytest

from conftest import SEEDS
from helpers import RECIPES, TOLERANCE, ar1, regime_vol
from herding import stats


def check(name, seed):
    """Run recipe `name` (tests/helpers.py) on `seed` and compare each number with its exact
    answer, allowing |bias| + 5 sd as measured over 200 other seeds
    (scripts/analysis/test_accuracy_audit.py; docs/TEST_ACCURACY.md)."""
    got, exact = RECIPES[name][0](seed), RECIPES[name][1]
    for k, v in exact.items():
        assert got[k] == pytest.approx(v, abs=TOLERANCE[f"{name}.{k}"]), f"{name}.{k}"
    return got



# ---------------------------------------------------------------- kurtosis
# Spec: excess kurtosis = 0 for a bell curve, -1.2 uniform, +3 Laplace.

@pytest.mark.parametrize("seed", SEEDS)
def test_kurtosis_gaussian_is_zero(seed):
    check("kurtosis_normal", seed)              # 800,000 draws: 0 +- 0.029


@pytest.mark.parametrize("seed", SEEDS)
def test_kurtosis_uniform_is_minus_1_2(seed):
    check("kurtosis_uniform", seed)             # 800,000 draws: -1.2 +- 0.0068


@pytest.mark.parametrize("seed", SEEDS)
def test_kurtosis_laplace_is_3(seed):
    check("kurtosis_laplace", seed)             # 2 million draws: 3 +- 0.13


def test_kurtosis_exact_two_points():
    # z = [-1, 1] -> mean(z^4) = 1 -> excess kurtosis = -2, by hand
    assert stats.kurtosis(np.array([-5.0, 5.0])) == pytest.approx(-2.0)


def test_kurtosis_ignores_shift_and_scale():
    r = np.random.default_rng(7).standard_t(5, 10_000)
    assert stats.kurtosis(3 * r + 100) == pytest.approx(stats.kurtosis(r))


def test_kurtosis_constant_input_is_rejected():
    # Edge: zero variance has no defined kurtosis -> clear error, not nan
    with pytest.raises(ValueError, match="constant"):
        stats.kurtosis(np.ones(50))


# A constant like 0.0123 is the hard case: its mean is not exactly 0.0123 in floating
# point, so the computed sd is tiny (about 1e-18) instead of 0. np.ones() above has an
# exact mean and cannot catch that (audit of November 2026: kurtosis gave -2.0, hurst -0.47).
@pytest.mark.parametrize("value", [0.0123, -1.49256087, 1e-7, 3.3])
@pytest.mark.parametrize("n", [501, 1000, 5000])
@pytest.mark.parametrize("name, measure", [
    ("standardize", stats.standardize), ("kurtosis", stats.kurtosis), ("ccdf", stats.ccdf),
    ("acf", lambda r: stats.acf(r, [1])), ("dfa", lambda r: stats.dfa(r, (16, 32, 64))),
    ("hurst", lambda r: stats.hurst(r, (16, 32, 64))), ("fingerprint", stats.fingerprint)])
def test_every_spread_measure_rejects_any_constant(value, n, name, measure):
    with pytest.raises(ValueError, match="constant"):
        measure(np.full(n, value))


def test_tiny_but_varying_returns_are_still_measured():
    # The other side of the boundary: returns of size 1e-100 do vary, so they must
    # NOT be called constant, and kurtosis must not care about the scale.
    r = np.random.default_rng(0).standard_normal(5000)
    assert stats.kurtosis(r * 1e-100) == pytest.approx(stats.kurtosis(r), abs=1e-9)


# -------------------------------------------------------------- hill_alpha
# Spec: for |r| with a Pareto tail P(|r|>x) ~ x^-alpha, return about alpha.

@pytest.mark.parametrize("alpha", [1.5, 3.0, 5.0])
@pytest.mark.parametrize("seed", SEEDS)
def test_hill_recovers_known_pareto_exponent(alpha, seed):
    check(f"hill_pareto_{alpha}", seed)         # 400,000 draws: alpha +- 3.6%


@pytest.mark.parametrize("seed", SEEDS)
def test_hill_student_t3_tail_is_about_3(seed):
    # Student-t with 3 degrees of freedom has tail exponent exactly 3, but only far out: the
    # Hill estimate is biased low nearer the middle (Hall 1982). 2 million draws, top 0.25%:
    # bias -0.049, sd 0.042, so 3 +- 0.27 (at 200,000 draws and the top 1% the bias was -0.12)
    check("hill_t3", seed)


def test_hill_gaussian_gives_large_alpha():
    # A bell curve has no power-law tail: Hill should report a big number
    r = np.random.default_rng(0).standard_normal(200_000)
    assert stats.hill_alpha(r) > 5


def test_hill_ignores_sign_and_scale():
    r = np.random.default_rng(3).standard_t(3, 20_000)
    assert stats.hill_alpha(-7 * r) == pytest.approx(stats.hill_alpha(r))


def test_hill_smaller_alpha_means_fatter_tail():
    rng = np.random.default_rng(5)
    fat, thin = rng.pareto(2, 50_000) + 1, rng.pareto(4, 50_000) + 1
    assert stats.hill_alpha(fat) < stats.hill_alpha(thin)


def test_hill_too_short_input_fails_loudly():
    # Edge: fewer than 11 values -> a clear error, not a crash deep inside
    with pytest.raises(ValueError):
        stats.hill_alpha(np.arange(1, 6, dtype=float))


def test_hill_with_many_zero_returns():
    # Edge the spec is silent on: real data has flat days (r == 0).
    # If the k-th largest |r| is 0 the estimator divides by zero.
    r = np.zeros(1000)
    r[:20] = np.arange(1, 21)
    with np.errstate(all="ignore"):
        a = stats.hill_alpha(r)
    assert np.isfinite(a), "hill_alpha returns non-finite when many returns are 0"


# -------------------------------------------------------------------- ccdf
# Spec: x sorted ascending; P(|z| >= x) runs from 1 down to 1/n.

def test_ccdf_exact_small_case():
    z, p = stats.ccdf(np.array([-2.0, 0.0, 2.0]))
    std = np.std([-2, 0, 2])
    assert np.allclose(z, [0, 2 / std, 2 / std])
    assert np.allclose(p, [1, 2 / 3, 1 / 3])


@pytest.mark.parametrize("seed", SEEDS)
def test_ccdf_gaussian_matches_theory(seed):
    # P(|Z| > 2) = 0.0455002638963584... (50-digit mpmath, not the rounded 0.0455);
    # 800,000 draws: +- 0.00085
    check("ccdf_normal_2", seed)


def test_ccdf_is_monotone():
    z, p = stats.ccdf(np.random.default_rng(1).standard_t(3, 5000))
    assert np.all(np.diff(z) >= 0) and np.all(np.diff(p) < 0)
    assert p[0] == 1 and p[-1] == pytest.approx(1 / 5000)


@pytest.mark.parametrize("fn", [stats.kurtosis, stats.hill_alpha, stats.ccdf])
@pytest.mark.parametrize("bad", [[], [1.0, np.nan, 2.0] * 10, [1.0, np.inf, 2.0] * 10])
def test_statistics_reject_empty_or_nonfinite_input(fn, bad):
    with pytest.raises(ValueError):
        fn(np.array(bad, dtype=float))


def test_ccdf_rejects_constant_input():
    with pytest.raises(ValueError, match="constant"):
        stats.ccdf(np.ones(20))


@pytest.mark.parametrize("tail", [0.0, -0.1, 1.0, 1.5])
def test_hill_rejects_tail_outside_0_1(tail):
    with pytest.raises(ValueError, match="tail"):
        stats.hill_alpha(np.random.default_rng(0).standard_t(3, 1000), tail=tail)


def test_hill_rejects_constant_tail():
    r = np.ones(200)
    r[:5] = 0.5                                      # tail (top 10) all equal to 1
    with pytest.raises(ValueError, match="constant"):
        stats.hill_alpha(r)



# ------------------------------------------------------------------- acf
# Spec: autocorrelation at lag k; AR(1) x_t = phi x_(t-1) + noise has acf phi^k.

@pytest.mark.parametrize("phi", [0.5, -0.3, 0.8])
@pytest.mark.parametrize("seed", SEEDS[:3])
def test_acf_recovers_ar1(phi, seed):
    check(f"acf_ar1_{phi}", seed)               # 800,000 days: phi^k +- 0.003-0.007


def test_ar1_helper_is_the_plain_recursion():
    # helpers.ar1 uses scipy's lfilter (fast); it must equal x_t = phi x_(t-1) + e_t exactly
    e = np.random.default_rng(0).standard_normal(1000)
    x = np.empty_like(e)
    x[0] = e[0]
    for t in range(1, len(e)):
        x[t] = 0.7 * x[t - 1] + e[t]
    assert ar1(0.7, e) == pytest.approx(x)


@pytest.mark.parametrize("seed", SEEDS)
def test_acf_of_independent_noise_is_zero(seed):
    check("acf_noise", seed)                    # 400,000 days: 0 +- 0.009


def test_acf_exact_alternating_series():
    # +1, -1, +1, ... : lag 1 -> -1, lag 2 -> +1, worked out by hand
    x = np.tile([1.0, -1.0], 50)
    assert np.allclose(stats.acf(x, [1, 2]), [-1, 1])


@pytest.mark.parametrize("lags", [0, -1, 99, [1, 200]])
def test_acf_rejects_impossible_lags(lags):
    with pytest.raises(ValueError, match="lags"):
        stats.acf(np.random.default_rng(0).standard_normal(100), lags)


# --------------------------------------------------------- fingerprint
# Spec: fingerprint(r) returns every measure in FINGERPRINT, each computed by the
# functions tested above; on processes with a known answer it must give that answer.

def test_fingerprint_has_exactly_the_documented_measures():
    f = stats.fingerprint(np.random.default_rng(0).standard_normal(5000))
    assert set(f) == set(stats.FINGERPRINT)
    assert all(isinstance(v, float) for v in f.values())


def test_fingerprint_of_a_bell_curve():
    # i.i.d. Gaussian: no fat tails at any horizon, no memory, H = 0.5. The average over 16
    # series of 200,000 days: one series measures H with an sd of 0.013 (its longest DFA
    # windows are few), too loose to test to 0.02.
    f = check("bell_fingerprint", 1)
    assert f["alpha_hill"] > 4.5
    assert f["alpha_reg"] == pytest.approx(3.7, abs=0.3)      # see the alpha_regression test


def test_fingerprint_lag1_on_ar1_by_formula():
    phi = -0.3
    x = ar1(phi, np.random.default_rng(3).standard_normal(800_000))
    assert stats.fingerprint(x)["ac_r_1"] == pytest.approx(phi, abs=TOLERANCE["acf_ar1_-0.3.l1"])


def test_fingerprint_volatility_memory_50_on_known_process():
    # |x| with a slowly switching volatility regime has memory at lag 50;
    # shuffled, the same values have none
    # exact: |x| = V*|Z|. Lag 50 is inside the same 100-day block half the time, so
    # Cov = (2/pi) * 0.5 * Var(V) with Var(V) = 1; Var(|x|) = E[V^2] - (E[V] E|Z|)^2
    # (helpers.REGIME_AC50); 1.6 million days: +- 0.007, shuffled 0 +- 0.004
    check("regime_ac50", 5)
    g = np.random.default_rng(5)
    x = regime_vol(16_000, g)
    assert stats.fingerprint(x)["H_abs"] > 0.6
    # sd 0.011 over 39 other seeds (fingerprint's H hardly shrinks with length, see
    # docs/TEST_ACCURACY.md), so 5 sd is 0.06; it was 0.03 (2.6 sd)
    assert stats.fingerprint(g.permutation(x))["H_abs"] == pytest.approx(0.5, abs=0.06)


def test_fingerprint_of_the_sp500_has_the_published_shape():
    # Cont (2001) "stylized facts": fat tails with alpha ~3, returns hard to
    # predict, move sizes clustered with long memory; Lux & Marchesi's H values
    from herding import market_data as data
    f = stats.fingerprint(data.spx_returns().values)
    assert f["kurtosis"] > 10 and 2.5 < f["alpha_hill"] < 3.5
    assert abs(f["ac_r_1"]) < 0.05 and 0.4 < f["H"] < 0.55
    assert f["ac_abs_1"] > 0.2 and f["ac_abs_50"] > 0.05 and f["H_abs"] > 0.8


def test_fingerprint_rejects_short_input():
    with pytest.raises(ValueError, match="500 days"):
        stats.fingerprint(np.random.default_rng(0).standard_normal(499))


def test_dfa_scales_span_16_to_a_quarter_of_the_data():
    s = stats.dfa_scales(20000)
    assert s[0] == 16 and s[-1] == 5000 and list(s) == sorted(set(s))
    with pytest.raises(ValueError, match="at least 64"):
        stats.dfa_scales(63)


@pytest.mark.parametrize("lag", [2.5, [1, 1.5], []])
def test_acf_rejects_lags_that_are_not_whole_numbers(lag):
    with pytest.raises(ValueError, match="whole numbers"):
        stats.acf(np.random.default_rng(0).standard_normal(100), lag)


def test_acf_accepts_whole_float_lags():
    r = np.random.default_rng(0).standard_normal(500)
    assert np.array_equal(stats.acf(r, 2.0), stats.acf(r, 2))


# ------------------------------------------- Lux & Marchesi (1999) measures

def test_aggregate_sums_blocks_by_hand():
    r = np.array([0.01, 0.02, -0.03, 0.04, 0.05, 0.06, 0.07])
    assert np.allclose(stats.aggregate(r, 3), [0.0, 0.15])      # last partial block dropped
    assert np.array_equal(stats.aggregate(r, 1), r)


@pytest.mark.parametrize("tau", [0, 2.5, 8, -1])
def test_aggregate_rejects_bad_tau(tau):
    with pytest.raises(ValueError):
        stats.aggregate(np.ones(7) * 0.01, tau)


def _symmetric_pareto(alpha, n=200_000, seed=0):
    # exact Pareto quantiles with random signs: P(|x| > s) = s^-alpha for s >= 1
    u = (np.arange(n) + 0.5) / n
    g = np.random.default_rng(seed)
    return g.choice([-1.0, 1.0], n) * (1 - u) ** (-1 / alpha)


@pytest.mark.parametrize("alpha", [2.5, 3.0, 4.0])
def test_alpha_regression_recovers_a_pure_power_law(alpha):
    # a pure power law is a straight line on log-log axes, so any window gives alpha
    # measured over 200 sign draws: bias -0.0013 to -0.0020 (the fit window bends slightly
    # because returns are standardized), sd 1e-4, so +- 0.003
    assert stats.alpha_regression(_symmetric_pareto(alpha)) == pytest.approx(alpha, abs=0.003)


def test_alpha_regression_is_steeper_for_a_bell_curve():
    # Measured: even a bell curve scores about 3.7 with the paper's 30% window,
    # because the window covers the curved middle. So the paper's 2.64 is
    # "fatter than 3.7", not "fatter than a bell curve's infinite exponent".
    g = np.random.default_rng(1).standard_normal(200_000)
    a = stats.alpha_regression(g)
    assert 3.4 < a < 4.0 and a > stats.alpha_regression(_symmetric_pareto(3.0)) + 0.4


def test_alpha_regression_rejects_bad_window():
    with pytest.raises(ValueError):
        stats.alpha_regression(np.random.default_rng(0).standard_normal(100), top=1.5)


def test_tail_fit_line_passes_through_its_window():
    # The fitted window starts where 30% of days are bigger and ends at the largest
    # move; the line c * x^-alpha should hit about 30% at the start of the window.
    r = _symmetric_pareto(3.0)
    alpha, c, lo, hi = stats.tail_fit(r)
    z = np.sort(np.abs(stats.standardize(r)))
    assert alpha == stats.alpha_regression(r)
    assert lo == pytest.approx(np.quantile(z, 0.70), rel=1e-3) and hi == z[-1]
    assert c * lo ** -alpha == pytest.approx(0.30, rel=0.002)    # measured: -0.05% +- 0.02%


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_hurst_of_white_noise_is_one_half(seed):
    # no memory at all -> H = 0.5 (the paper's news series: 0.49-0.51)
    # 400,000 days; DFA has a small upward bias here (+0.0014, measured over 200 seeds)
    check("hurst_white", seed)


def test_hurst_of_a_random_walk_is_three_halves():
    # DFA of an integrated (random-walk) series has slope 1.5
    check("hurst_walk", 3)                      # 400,000 days: 1.5 +- 0.037


def test_hurst_does_not_depend_on_units():
    x = np.random.default_rng(4).standard_normal(20_000)
    assert stats.hurst(100 * x) == pytest.approx(stats.hurst(x), abs=1e-9)


def test_hurst_sees_memory_in_clustered_volatility():
    # sizes of moves that come in long calm/wild regimes have H > 0.5
    g = np.random.default_rng(5)
    regime = np.repeat(g.choice([0.5, 2.0], 200), 500)          # 200 regimes of 500 days
    r = regime * g.standard_normal(regime.size)
    assert stats.hurst(np.abs(r)) > 0.7 and stats.hurst(r) == pytest.approx(0.5, abs=0.06)


def test_dfa_rejects_scales_too_large_for_the_data():
    with pytest.raises(ValueError, match="scales"):
        stats.dfa(np.random.default_rng(0).standard_normal(1000), scales=(16, 512))


# ------------------------------------------------------------------ found by the stress test (Nov 2026)

def test_hurst_needs_two_window_sizes():
    x = np.random.default_rng(0).standard_normal(5000)
    for scales in [(16,), (16, 16)]:
        with pytest.raises(ValueError, match="2 different window sizes"):
            stats.hurst(x, scales)


def test_inputs_that_used_to_give_silent_nonsense_are_rejected():
    x = np.random.default_rng(0).standard_normal(1000)
    with pytest.raises(ValueError, match="1-D"):
        stats.fingerprint(x.reshape(25, 40))        # used to be read as 1,000 days
    with pytest.raises(ValueError, match="overflow"), np.errstate(over="ignore"):
        stats.kurtosis(1e200 * x)                   # used to give -3.0
    with pytest.raises(ValueError, match="whole numbers"):
        stats.acf(x, True)                          # used to be read as lag 1


# Audit of November 2026, issue 6: dfa with no window sizes returned two empty arrays
def test_dfa_needs_at_least_one_window_size():
    with pytest.raises(ValueError, match="window"):
        stats.dfa(np.random.default_rng(0).standard_normal(1000), ())


# --------------------------------------------------------------- skewness, block_indices, moments
# Spec: skewness is 0 for symmetric data, 2 for an exponential distribution, and flips sign
# in a mirror; block_indices gives n positions in runs of consecutive days; moments gives the
# four plain moments with intervals that contain them.

@pytest.mark.parametrize("seed", SEEDS)
def test_skewness_known_values(seed):
    check("skew_exponential", seed)             # exponential: skewness 2 (1.6 million draws, +- 0.034)
    check("skew_normal", seed)                  # bell curve: 0 +- 0.011
    e = np.random.default_rng(seed).exponential(size=1000)
    assert stats.skewness(-e) == pytest.approx(-stats.skewness(e))  # the mirror image
    assert stats.skewness([1.0, 2, 3]) == 0


def test_skewness_rejects_constant_input():
    with pytest.raises(ValueError, match="constant"):
        stats.skewness([1.0, 1.0, 1.0])


@pytest.mark.parametrize("wrap", [True, False])
def test_block_indices_are_runs_of_consecutive_days(wrap):
    g = np.random.default_rng(0)
    i = stats.block_indices(103, 10, g, wrap)
    assert len(i) == 103 and i.min() >= 0 and i.max() < 103
    for run in np.split(i, range(10, 103, 10)):
        steps = np.diff(run) % 103 if wrap else np.diff(run)
        assert (steps == 1).all()


def test_block_indices_without_wrap_never_runs_past_the_end():
    g = np.random.default_rng(3)
    for _ in range(200):
        i = stats.block_indices(50, 49, g, wrap=False)
        assert (np.diff(i[:49]) == 1).all() and i.max() < 50


@pytest.mark.parametrize("bad", [0, 11, 2.5, True])
def test_block_indices_rejects_bad_blocks(bad):
    with pytest.raises(ValueError, match="block"):
        stats.block_indices(10, bad, np.random.default_rng(0))


def test_moments_are_the_plain_moments_with_intervals_around_them():
    r = np.random.default_rng(4).standard_t(5, 5000) * 0.01
    m = stats.moments(r, block=20, draws=300)
    assert m["mean"][0] == pytest.approx(r.mean()) and m["sd"][0] == pytest.approx(r.std())
    assert m["skew"][0] == pytest.approx(stats.skewness(r)) and m["kurtosis"][0] == pytest.approx(stats.kurtosis(r))
    for v, lo, hi in m.values():
        assert lo <= v <= hi


def test_moments_interval_has_the_textbook_width_for_white_noise():
    # For n independent bell-curve days, the mean's 95% interval is about +-1.96 sd / sqrt(n)
    n = 10_000
    r = np.random.default_rng(5).normal(size=n)
    _, lo, hi = stats.moments(r, block=1, draws=8000)["mean"]
    # 8,000 resamples: measured bias -0.1%, sd 1.3% of the width, so within 6.5%
    assert (hi - lo) / 2 == pytest.approx(1.959963984540054 / np.sqrt(n), rel=0.065)


def test_moments_rejects_bad_draws():
    with pytest.raises(ValueError, match="draws"):
        stats.moments(np.arange(10.0), draws=1)


# ------------------------------------------------------------------ found by the overnight mutation test (Nov 2026)
# Small deliberate bugs that no test noticed: a wrong lag, horizon or percentile inside
# fingerprint() and moments(), and limits that were never tried at their exact edge.

def test_fingerprint_entries_are_the_functions_they_name():
    r = np.random.default_rng(11).standard_t(4, 3000) * 0.01
    f, a = stats.fingerprint(r), np.abs(r)
    assert f["ac_r_1"] == stats.acf(r, 1)[0]
    assert f["ac_abs_1"] == stats.acf(a, 1)[0]
    assert f["ac_abs_50"] == stats.acf(a, 50)[0]
    assert f["kurt_5"] == stats.kurtosis(stats.aggregate(r, 5))
    assert f["kurt_25"] == stats.kurtosis(stats.aggregate(r, 25))
    assert f["alpha_hill"] == stats.hill_alpha(r, 0.05)
    assert f["H"] == stats.hurst(r, stats.dfa_scales(len(r)))
    assert f["H_abs"] == stats.hurst(a, stats.dfa_scales(len(r)))
    assert f["sd"] == r.std() and f["kurtosis"] == stats.kurtosis(r)
    stats.fingerprint(r[:500])                                  # exactly the minimum


def test_moments_intervals_are_the_2_5_and_97_5_percentiles_of_the_same_draws():
    r = np.random.default_rng(12).standard_normal(1000)
    out = stats.moments(r, block=50, draws=300, seed=4)
    g = np.random.default_rng(4)
    four = lambda x: (x.mean(), x.std(), stats.skewness(x), stats.kurtosis(x))
    boot = np.array([four(r[stats.block_indices(1000, 50, g)]) for _ in range(300)])
    for j, k in enumerate(stats.MOMENTS):
        assert out[k] == (four(r)[j], np.percentile(boot[:, j], 2.5), np.percentile(boot[:, j], 97.5))
    stats.moments(r, draws=2)                                   # the smallest allowed


def test_block_indices_by_hand():
    g1, g2 = np.random.default_rng(5), np.random.default_rng(5)
    i = stats.block_indices(10, 3, g1)
    starts = g2.integers(0, 10, 4)                             # ceil(10 / 3) = 4 runs
    assert list(i) == [(s + j) % 10 for s in starts for j in range(3)][:10]
    # every start, including day 0, must be possible
    g = np.random.default_rng(0)
    firsts = {stats.block_indices(10, 10, g)[0] for _ in range(400)}
    assert firsts == set(range(10))
    assert len(stats.block_indices(7, 7, g)) == 7               # block = n is allowed


def test_hill_uses_at_least_10_tail_values():
    a = np.abs(np.random.default_rng(13).standard_normal(400))
    s = np.sort(a)[::-1]                                       # 400 * 0.01 = 4 < 10, so k = 10
    assert stats.hill_alpha(a, 0.01) == pytest.approx(1 / np.mean(np.log(s[:10] / s[10])))


def test_limits_at_their_exact_edge():
    r = np.random.default_rng(14).standard_normal(100)
    assert len(stats.aggregate(r, 100)) == 1                   # tau = all the days
    assert len(stats.acf(r, 98)) == 1                          # the longest lag allowed
    with pytest.raises(ValueError):
        stats.acf(r, 99)
    assert stats.dfa_scales(64)[-1] == 16                       # n // 4 = the smallest scale
