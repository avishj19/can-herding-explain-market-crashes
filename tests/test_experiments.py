"""Tests for herding/experiments.py (shared helpers for running many histories)."""
import numpy as np
import pytest

from herding import experiments, model


def test_history_is_simulate_minus_burn_in():
    full, _ = model.simulate(500 + 100, seed=7)
    assert np.array_equal(experiments.history(seed=7, days=500, burn=100), full[100:])


def test_history_passes_options_through():
    a = experiments.history(seed=3, days=300, breakers=dict(levels=(0.9, 0.95, 0.99), halt_minutes=15))
    assert np.array_equal(a, experiments.history(seed=3, days=300))   # breakers never fire


def test_ensemble_keeps_seed_order_and_matches_single_runs():
    seeds = [11, 5, 8]
    runs = experiments.ensemble(model.P, seeds, days=400)
    for s, r in zip(seeds, runs):
        assert np.array_equal(r, experiments.history(model.P, s, 400))


def test_ensemble_with_shared_pool_gives_same_result():
    with experiments.Pool(2) as pool:
        a = experiments.ensemble(model.P, [1, 2], days=300, pool=pool)
    b = experiments.ensemble(model.P, [1, 2], days=300)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_ensemble_no_chartists():
    r = experiments.ensemble(model.P, [4], days=300, herding=False)[0]
    assert np.array_equal(r, experiments.history(model.P, 4, 300, herding=False))


def test_worst_fall_by_hand():
    r = np.array([0.1, -0.2, -0.3, 0.4, -0.1])
    # 1 day: -0.3; 2 days: -0.2-0.3 = -0.5; 3 days: -0.2-0.3+0.4 = -0.1 vs 0.1-0.2-0.3 = -0.4
    assert experiments.worst_fall(r, 1) == pytest.approx(0.3)
    assert experiments.worst_fall(r, 2) == pytest.approx(0.5)
    assert experiments.worst_fall(r, 3) == pytest.approx(0.4)


def test_worst_fall_of_a_rising_market_is_negative():
    assert experiments.worst_fall(np.full(10, 0.01), 3) == pytest.approx(-0.03)


def test_scaled_breakers_by_hand():
    rules, scale = experiments.scaled_breakers(0.02, 0.01)
    assert scale == 2 and rules == dict(levels=(0.14, 0.26, 0.40), halt_minutes=15)
    model.check_breakers(rules)                               # valid rules for the model
