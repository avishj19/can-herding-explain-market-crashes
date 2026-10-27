"""Tests for kirman.py, checked against Kirman's exact math.  Run: python3 -m pytest tests/ -q"""
import numpy as np
import pytest

from conftest import SEEDS
from herding import kirman

HERD = dict(N=100, eps=0.002, delta=0.01)    # eps < (1-delta)/(N-1) = 0.01 -> herding
SPLIT = dict(N=100, eps=0.05, delta=0.01)    # eps > 0.01 -> about 50/50


def test_transition_probabilities_by_hand():
    # N=4, eps=0.1, delta=0.5, k=1: up = 3/4*(0.1 + 0.5*1/3) = 0.2; down = 1/4*(0.1 + 0.5*3/3) = 0.15
    assert kirman.up_prob(1, 4, 0.1, 0.5) == pytest.approx(0.2)
    assert kirman.down_prob(1, 4, 0.1, 0.5) == pytest.approx(0.15)
    # nobody can leave an empty source or join a full one
    assert kirman.down_prob(0, 4, 0.1, 0.5) == 0 and kirman.up_prob(4, 4, 0.1, 0.5) == 0


def test_stationary_tiny_case_by_hand():
    # N=2, eps=0.5, delta=1 (no recruitment): each ant independent -> Binomial(2, 1/2)
    assert np.allclose(kirman.stationary(2, 0.5, 1.0), [0.25, 0.5, 0.25])


@pytest.mark.parametrize("N", [10, 50, 100])
def test_no_recruitment_gives_binomial(N):
    # delta = 1: ants never recruit, so each is at A half the time, independently
    from scipy.stats import binom
    exact = binom.pmf(np.arange(N + 1), N, 0.5)
    assert np.allclose(kirman.stationary(N, 0.3, 1.0), exact)


@pytest.mark.parametrize("params", [HERD, SPLIT])
def test_stationary_is_symmetric_and_normalized(params):
    pi = kirman.stationary(**params)
    assert pi.sum() == pytest.approx(1) and np.allclose(pi, pi[::-1])


def test_kirman_condition_shapes_the_distribution():
    # Kirman's prediction is about SHAPE: herding -> the most likely states are
    # "everyone at one source"; no herding -> the most likely state is 50/50.
    h, s = kirman.stationary(**HERD), kirman.stationary(**SPLIT)
    assert kirman.herds(**HERD) and np.argmax(h) in (0, 100) and h[50] < h[0]
    assert not kirman.herds(**SPLIT) and np.argmax(s) == 50 and s[0] < s[50] / 1000


@pytest.mark.parametrize("N, eps, delta", [(100, 0.05, 0.01), (50, 0.01, 0.2),
                                           (400, 0.002, 0.0), (200, 0.02, 0.1)])
def test_matches_beta_distribution_in_large_N_limit(N, eps, delta):
    # Continuum limit (Kirman 1993): fraction x = k/N ~ Beta(a, a), a = eps*(N-1)/(1-delta)
    # Independent check of stationary() via a different formula. The Beta curve
    # is only an approximation: for strong herding (a << 1) it misplaces weight
    # at k = 0 and N by ~18% (measured), so it is checked only for a >= 0.5.
    from scipy.stats import beta
    a = eps * (N - 1) / (1 - delta)
    x = (np.arange(N + 1) + 0.5) / (N + 1)
    approx = beta.pdf(x, a, a)
    approx /= approx.sum()
    tv = 0.5 * np.abs(approx - kirman.stationary(N, eps, delta)).sum()
    assert tv < 0.06, f"total variation {tv:.3f}"
    assert (a < 1) == kirman.herds(N, eps, delta)      # Beta is U-shaped iff a < 1


@pytest.mark.parametrize("params", [HERD, SPLIT])
@pytest.mark.parametrize("seed", SEEDS[:3])
def test_simulation_matches_exact_distribution(params, seed):
    # Long run: time spent at each k must match the exact pi (total variation < 0.05).
    # Herding ants flip sources only rarely, so the run must be long (40M steps).
    k = kirman.simulate(**params, steps=40_000_000, seed=seed)
    hist = np.bincount(k, minlength=params["N"] + 1) / len(k)
    tv = 0.5 * np.abs(hist - kirman.stationary(**params)).sum()
    assert tv < 0.05, f"total variation distance {tv:.3f}"


def test_herding_ants_switch_en_masse():
    # Herding regime: the crowd flips from mostly-A to mostly-B several times
    k = kirman.simulate(**HERD, steps=5_000_000, seed=1)
    side = np.sign(k[k != 50] - 50)
    flips = np.count_nonzero(np.diff(side[np.abs(k[k != 50] - 50) > 40]))
    assert flips >= 3


def test_simulation_stays_in_bounds_and_moves_one_ant():
    k = kirman.simulate(**HERD, steps=200_000, seed=0)
    assert k.min() >= 0 and k.max() <= 100 and np.all(np.abs(np.diff(k)) <= 1)


def test_same_seed_same_path():
    a = kirman.simulate(**SPLIT, steps=1000, seed=4)
    assert np.array_equal(a, kirman.simulate(**SPLIT, steps=1000, seed=4))


@pytest.mark.parametrize("bad", [dict(N=1, eps=0.1, delta=0.1), dict(N=10.5, eps=0.1, delta=0.1),
                                 dict(N=10, eps=0.0, delta=0.1), dict(N=10, eps=1.5, delta=0.1),
                                 dict(N=10, eps=0.1, delta=-0.1), dict(N=10, eps=0.1, delta=2)])
def test_rejects_bad_parameters(bad):
    with pytest.raises(ValueError):
        kirman.stationary(**bad)
    with pytest.raises(ValueError):
        kirman.simulate(**bad, steps=10)


@pytest.mark.parametrize("kw", [dict(steps=-1), dict(steps=2.5), dict(steps=10, k0=101)])
def test_rejects_bad_run_settings(kw):
    with pytest.raises(ValueError):
        kirman.simulate(**HERD, **kw)


@pytest.mark.parametrize("N, eps, delta", [(100, 0.002, 0.01), (200, 0.001, 0.0)])
def test_strong_herding_is_u_shaped(N, eps, delta):
    # a < 1: probability rises toward both walls (U shape), as Kirman predicts
    pi = kirman.stationary(N, eps, delta)
    half = pi[: N // 2 + 1]
    assert kirman.herds(N, eps, delta) and np.all(np.diff(half) <= 1e-15)
