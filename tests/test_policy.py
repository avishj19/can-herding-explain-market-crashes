"""Tests for herding/policy.py (circuit breakers and policy surprises)."""
import numpy as np
import pytest

from herding import model, policy

TIGHT = dict(levels=(0.003, 0.006, 0.009), halt_minutes=15)   # fires often in set I
CALM = dict(model.SET_I, a1=0.0, a2=0.0, tc=0.002)            # chartists barely matter
# Chartists that do not trade and no noise: only fundamentalists move the price.
# (CALM is not enough: with a1 = 0 the chartists can still be 60% of traders, and
# their buy-sell imbalance moves the price by 2-3% a day.)
QUIET = dict(CALM, tc=0.0, sigma_mu=0.0, sigma_eps=0.0)


class SerialPool:
    def map(self, fn, jobs, chunksize=None):
        return [fn(j) for j in jobs]


def test_scaled_breakers_multiplies_every_level():
    rules, scale = policy.scaled_breakers(0.02, 0.01)
    assert scale == 2.0
    assert rules["levels"] == pytest.approx((0.14, 0.26, 0.40))
    assert rules["halt_minutes"] == 15
    assert policy.scaled_breakers(0.01, 0.01)[0]["levels"] == model.US_BREAKERS["levels"]


@pytest.mark.parametrize("model_sd, market_sd", [(0.0, 0.01), (0.01, 0.0), (0.06, 0.01)])
def test_scaled_breakers_rejects_zero_volatility_and_levels_past_100pct(model_sd, market_sd):
    with pytest.raises(ValueError):
        policy.scaled_breakers(model_sd, market_sd)


def test_crash_measures_on_a_known_series():
    r = np.array([0.0, -0.1, -0.05, 0.02, -0.2, 0.0] + [0.0] * 20)
    m = policy.crash_measures(r, 0.07)                # a crash day: below log(0.93) = -0.0726
    assert m["worst_1d"] == pytest.approx(0.2)
    assert m["worst_5d"] == pytest.approx(0.1 + 0.05 - 0.02 + 0.2)
    assert m["worst_20d"] == pytest.approx(0.33)
    assert m["crash_days"] == 2


def test_breakers_that_never_fire_leave_the_pair_identical():
    out = policy._breaker_pair((model.SET_I, 1, 2000, dict(levels=(0.97, 0.98, 0.99),
                                                             halt_minutes=15)))
    assert out["event"] is None and out["on"]["halt_days"] == 0
    for k in out["off"]:
        assert out["off"][k] == out["on"][k]


def test_the_first_halt_event_is_measured_from_identical_histories():
    out = policy._breaker_pair((model.SET_I, 2, 2000, TIGHT))
    off = model.simulate(model.SET_I, 2000 + policy.BURN, seed=2)
    on = model.simulate(model.SET_I, 2000 + policy.BURN, seed=2, breakers=TIGHT)
    d = int(np.flatnonzero(on["halts"])[0])
    assert np.array_equal(off["lp"][:d + 1], on["lp"][:d + 1])   # same until the halt
    for k in policy.EVENT_DAYS:
        assert out["event"]["off"][k] == off["lp"][d + k] - off["lp"][d]
        assert out["event"]["on"][k] == on["lp"][d + k] - on["lp"][d]
    # the halt day fell to the first level in the run with breakers
    assert out["event"]["on"][1] <= np.log(1 - 0.003) or on["halts"][d] > 0
    assert out["on"]["halt_days"] == int(np.sum(on["halts"][policy.BURN:] > 0))


def test_breaker_experiment_collects_every_seed():
    whole, events, bad = policy.breaker_experiment(SerialPool(), model.SET_I, (1, 2, 3), 1500, TIGHT)
    assert bad == 0
    assert all(len(v) == 3 for side in whole.values() for v in side.values())
    assert set(events["on"]) == set(policy.EVENT_DAYS) and len(events["on"][1]) == 3


def test_a_surprise_in_a_calm_market_moves_the_price_by_its_size():
    # Fundamentalists alone pull the price to the new value within days: the fall
    # is 1 surprise (to within one 0.1% tick per 10%), no overshoot, and without the
    # surprise nothing moves (placebo 0).
    out, bad = policy.surprise_experiment(SerialPool(), QUIET, range(1, 9), size=-0.1)
    assert bad == 0
    assert np.all(np.abs(out["end"] - 1) < 0.02)
    assert np.all(out["peak"] < 1.02)
    assert np.all(out["placebo_peak"] == 0)
    assert np.all((out["chartists"] > 0) & (out["chartists"] < 1))


def test_the_surprise_ratio_is_1_minus_the_change_in_mispricing():
    seed, size, horizon = 7, -0.2, 10
    row = policy._surprise_one((CALM, seed, size, horizon))
    day = policy.BURN + int(np.random.default_rng(seed).integers(0, 5000))
    o = model.simulate(CALM, day + horizon, seed=seed, shocks={day: size})
    gap = o["lp"] - o["lf"]
    assert row["end"] == pytest.approx(1 - (gap[day + horizon] - gap[day]) / -np.log(0.8))
    ratios = 1 - (gap[day + 1:day + horizon + 1] - gap[day]) / -np.log(0.8)
    assert row["peak"] == pytest.approx(ratios.max())
    assert row["chartists"] == o["z"][day]


@pytest.mark.parametrize("size", [0.0, 0.1, -1.0])
def test_surprise_size_must_be_a_fall(size):
    with pytest.raises(ValueError, match="size"):
        policy.surprise_experiment(SerialPool(), CALM, (1,), size=size)


def test_paired_difference_mean_and_interval():
    off = np.zeros(400)
    on = np.random.default_rng(0).normal(0.5, 1.0, 400)
    mean, lo, hi = policy.paired_difference(off, on)
    assert mean == pytest.approx(on.mean())
    assert lo < mean < hi
    assert hi - lo == pytest.approx(2 * 1.96 / np.sqrt(400), rel=0.15)   # normal-theory width
    assert policy.paired_difference([1, 2, 3], [1, 2, 3]) == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        policy.paired_difference([1.0], [2.0])
