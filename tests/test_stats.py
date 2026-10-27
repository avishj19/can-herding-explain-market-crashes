"""Tests for stats.py (tail and clustering measurements).  Run: python3 -m pytest tests/ -q

Every test states what "correct" means, and uses inputs whose answer we know
independently (math or published facts), across several random seeds.
"""
import numpy as np
import pandas as pd
import pytest

from conftest import ROOT, SEEDS
from herding import model
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


# ------------------------------------------------------------- tail_share
# Spec: fraction of days beyond k standard deviations.

@pytest.mark.parametrize("seed", SEEDS[:3])
def test_tail_share_gaussian(seed):
    x = np.random.default_rng(seed).standard_normal(2_000_000)
    # P(|Z| > 3) = 0.0027 exactly for a bell curve
    assert stats.tail_share(x, 3) == pytest.approx(0.0027, rel=0.05)


@pytest.mark.parametrize("seed", SEEDS[:3])
def test_tail_share_laplace(seed):
    # Laplace with scale b has std b*sqrt(2); P(|X| > 4 std) = exp(-4*sqrt(2)) = 0.00349
    x = np.random.default_rng(seed).laplace(0, 1, 2_000_000)
    assert stats.tail_share(x, 4) == pytest.approx(np.exp(-4 * np.sqrt(2)), rel=0.05)


def test_tail_share_exact_small_case():
    x = np.array([-10.0] + [0.0] * 98 + [10.0])     # std = sqrt(2), so |z| = 7.07 twice
    assert stats.tail_share(x, 4) == pytest.approx(0.02)


# --------------------------------------------------------- stylized_facts

def test_stylized_facts_gaussian_reference():
    x = np.random.default_rng(1).standard_normal(200_000)
    f = stats.stylized_facts(x)
    assert abs(f["kurtosis"]) < 0.05 and f["alpha_5pct"] > 4.5
    assert f["share_4sd"] < 2e-4
    assert all(abs(f[k]) < 0.015 for k in ["ac_r_1", "ac_r_2", "ac_abs_1", "ac_abs_20", "ac_abs_100"])


def test_stylized_facts_spx_published_shape():
    # Real market fingerprint (Cont 2001 "stylized facts"): fat tails, alpha ~3,
    # returns unpredictable, |returns| clustered with long memory
    from herding import market_data as data
    f = stats.stylized_facts(data.spx_returns().values)
    assert f["kurtosis"] > 10 and 2.5 < f["alpha_5pct"] < 3.5
    assert abs(f["ac_r_1"]) < 0.05 and abs(f["ac_r_2"]) < 0.05
    assert f["ac_abs_1"] > 0.2 and f["ac_abs_100"] > 0.05


# ------------------------------------------ calibration v2: new fingerprint facts
# Spec: ac_r_2to10 = mean return autocorrelation over lags 2..10 (catches a
# reversal that hides past day 2); ac_abs_50 = volatility memory at 50 days.

def test_reversal_measure_on_ar1_by_formula():
    # AR(1) with phi: mean of phi^k for k = 2..10, computed independently
    phi = -0.5
    e = np.random.default_rng(3).standard_normal(300_000)
    x = np.empty_like(e)
    x[0] = e[0]
    for t in range(1, len(e)):
        x[t] = phi * x[t - 1] + e[t]
    expected = np.mean([phi ** k for k in range(2, 11)])
    assert stats.stylized_facts(x)["ac_r_2to10"] == pytest.approx(expected, abs=0.005)


def test_reversal_measure_catches_day_3_reversal():
    # x_t = e_t - 0.6 e_(t-3): correlation only at lag 3, = -0.6/(1+0.36)
    e = np.random.default_rng(4).standard_normal(300_003)
    x = e[3:] - 0.6 * e[:-3]
    f = stats.stylized_facts(x)
    assert abs(f["ac_r_2"]) < 0.01                       # invisible to the old target
    assert f["ac_r_2to10"] == pytest.approx(-0.6 / 1.36 / 9, abs=0.004)


def test_volatility_memory_50_on_known_process():
    # |x| with a slowly switching volatility regime has memory at lag 50;
    # shuffled, the same values have none
    g = np.random.default_rng(5)
    vol = np.repeat(g.choice([1.0, 3.0], 4000), 100)     # regimes last 100 days
    x = vol * g.standard_normal(len(vol))
    # exact: |x| = V*|Z|. Lag 50 is inside the same 100-day block half the time, so
    # Cov = (2/pi) * 0.5 * Var(V) with Var(V) = 1; Var(|x|) = E[V^2] - (E[V] E|Z|)^2
    expected = (2 / np.pi) * 0.5 * 1.0 / (5.0 - (2.0 * np.sqrt(2 / np.pi)) ** 2)
    assert stats.stylized_facts(x)["ac_abs_50"] == pytest.approx(expected, abs=0.01)
    assert abs(stats.stylized_facts(g.permutation(x))["ac_abs_50"]) < 0.01


def test_fingerprint_has_all_calibration_targets():
    from herding import calibrate
    f = stats.stylized_facts(np.random.default_rng(0).standard_normal(5000))
    assert set(calibrate.TARGETS) <= set(f) and set(calibrate.SCALE) == set(calibrate.TARGETS)
