"""Tests for stats.py (tail and clustering measurements).  Run: python3 -m pytest tests/ -q

Every test states what "correct" means, and uses inputs whose answer we know
independently (math or published facts), across several random seeds.
"""
import numpy as np
import pytest

from conftest import SEEDS
from herding import stats



# ---------------------------------------------------------------- kurtosis
# Spec: excess kurtosis = 0 for a bell curve, -1.2 uniform, +3 Laplace.

@pytest.mark.parametrize("seed", SEEDS)
def test_kurtosis_gaussian_is_zero(seed):
    r = np.random.default_rng(seed).standard_normal(200_000)
    assert abs(stats.kurtosis(r)) < 0.05


@pytest.mark.parametrize("seed", SEEDS)
def test_kurtosis_uniform_is_minus_1_2(seed):
    r = np.random.default_rng(seed).uniform(-1, 1, 200_000)
    assert stats.kurtosis(r) == pytest.approx(-1.2, abs=0.03)


@pytest.mark.parametrize("seed", SEEDS)
def test_kurtosis_laplace_is_3(seed):
    r = np.random.default_rng(seed).laplace(0, 1, 500_000)
    assert stats.kurtosis(r) == pytest.approx(3.0, abs=0.2)


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


# -------------------------------------------------------------- hill_alpha
# Spec: for |r| with a Pareto tail P(|r|>x) ~ x^-alpha, return about alpha.

@pytest.mark.parametrize("alpha", [1.5, 3.0, 5.0])
@pytest.mark.parametrize("seed", SEEDS)
def test_hill_recovers_known_pareto_exponent(alpha, seed):
    rng = np.random.default_rng(seed)
    r = (rng.pareto(alpha, 100_000) + 1) * rng.choice([-1, 1], 100_000)
    assert stats.hill_alpha(r) == pytest.approx(alpha, rel=0.08)


@pytest.mark.parametrize("seed", SEEDS)
def test_hill_student_t3_tail_is_about_3(seed):
    # Student-t with 3 degrees of freedom has tail exponent exactly 3
    r = np.random.default_rng(seed).standard_t(3, 200_000)
    assert stats.hill_alpha(r, tail=0.01) == pytest.approx(3.0, rel=0.12)


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
    z, p = stats.ccdf(np.random.default_rng(seed).standard_normal(200_000))
    # P(|Z| > 2) = 0.0455 for a standard normal
    assert np.interp(2.0, z, p) == pytest.approx(0.0455, abs=0.003)


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
    e = np.random.default_rng(seed).standard_normal(200_000)
    x = np.empty_like(e)
    x[0] = e[0]
    for t in range(1, len(e)):
        x[t] = phi * x[t - 1] + e[t]
    assert np.allclose(stats.acf(x, [1, 2, 3]), [phi, phi ** 2, phi ** 3], atol=0.01)


@pytest.mark.parametrize("seed", SEEDS)
def test_acf_of_independent_noise_is_zero(seed):
    x = np.random.default_rng(seed).standard_normal(100_000)
    assert np.all(np.abs(stats.acf(x, [1, 2, 10, 100])) < 0.015)


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
    # i.i.d. Gaussian: no fat tails at any horizon, no memory, H = 0.5
    x = 0.01 * np.random.default_rng(1).standard_normal(200_000)
    f = stats.fingerprint(x)
    assert f["sd"] == pytest.approx(0.01, rel=0.01)
    assert abs(f["kurtosis"]) < 0.05 and abs(f["kurt_5"]) < 0.1 and abs(f["kurt_25"]) < 0.2
    assert f["alpha_hill"] > 4.5
    assert f["alpha_reg"] == pytest.approx(3.7, abs=0.3)      # see the alpha_regression test
    assert all(abs(f[k]) < 0.015 for k in ("ac_r_1", "ac_abs_1", "ac_abs_50"))
    assert f["H"] == pytest.approx(0.5, abs=0.03) and f["H_abs"] == pytest.approx(0.5, abs=0.03)


def test_fingerprint_lag1_on_ar1_by_formula():
    phi = -0.3
    e = np.random.default_rng(3).standard_normal(200_000)
    x = np.empty_like(e)
    x[0] = e[0]
    for t in range(1, len(e)):
        x[t] = phi * x[t - 1] + e[t]
    assert stats.fingerprint(x)["ac_r_1"] == pytest.approx(phi, abs=0.01)


def test_fingerprint_volatility_memory_50_on_known_process():
    # |x| with a slowly switching volatility regime has memory at lag 50;
    # shuffled, the same values have none
    g = np.random.default_rng(5)
    vol = np.repeat(g.choice([1.0, 3.0], 4000), 100)     # regimes last 100 days
    x = vol * g.standard_normal(len(vol))
    # exact: |x| = V*|Z|. Lag 50 is inside the same 100-day block half the time, so
    # Cov = (2/pi) * 0.5 * Var(V) with Var(V) = 1; Var(|x|) = E[V^2] - (E[V] E|Z|)^2
    expected = (2 / np.pi) * 0.5 * 1.0 / (5.0 - (2.0 * np.sqrt(2 / np.pi)) ** 2)
    f = stats.fingerprint(x)
    assert f["ac_abs_50"] == pytest.approx(expected, abs=0.01)
    assert f["H_abs"] > 0.6
    shuffled = stats.fingerprint(g.permutation(x))
    assert abs(shuffled["ac_abs_50"]) < 0.01 and shuffled["H_abs"] == pytest.approx(0.5, abs=0.03)


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
    assert stats.alpha_regression(_symmetric_pareto(alpha)) == pytest.approx(alpha, abs=0.15)


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
    z = np.sort(np.abs(stats._standardize(r)))
    assert alpha == stats.alpha_regression(r)
    assert lo == pytest.approx(np.quantile(z, 0.70), rel=1e-3) and hi == z[-1]
    assert c * lo ** -alpha == pytest.approx(0.30, rel=0.05)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_hurst_of_white_noise_is_one_half(seed):
    # no memory at all -> H = 0.5 (the paper's news series: 0.49-0.51)
    x = np.random.default_rng(seed).standard_normal(100_000)
    assert stats.hurst(x) == pytest.approx(0.5, abs=0.04)


def test_hurst_of_a_random_walk_is_three_halves():
    # DFA of an integrated (random-walk) series has slope 1.5
    x = np.cumsum(np.random.default_rng(3).standard_normal(100_000))
    assert stats.hurst(x) == pytest.approx(1.5, abs=0.06)


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
