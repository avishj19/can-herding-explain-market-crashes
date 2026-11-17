"""Tests for herding/policy.py (circuit breakers and policy surprises)."""
import numpy as np
import pytest

from herding import experiments, model, policy
from helpers import SerialPool

TIGHT = dict(levels=(0.003, 0.006, 0.009), halt_minutes=15)   # fires often in set I
CALM = dict(model.SET_I, a1=0.0, a2=0.0, tc=0.002)            # chartists barely matter
# Chartists that do not trade and no noise: only fundamentalists move the price.
# (CALM is not enough: with a1 = 0 the chartists can still be 60% of traders, and
# their buy-sell imbalance moves the price by 2-3% a day.)
QUIET = dict(CALM, tc=0.0, sigma_mu=0.0, sigma_eps=0.0)


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
    # CALM, not set I: set I ran away within these 3,000 days on 11 of 39 other seeds
    # (overnight stress test, Nov 2026); this test is about halts, not runaways
    out = policy._breaker_pair((CALM, 2, 2000, TIGHT))
    off = model.simulate(CALM, 2000 + policy.BURN, seed=2)
    on = model.simulate(CALM, 2000 + policy.BURN, seed=2, breakers=TIGHT)
    d = int(np.flatnonzero(on["halts"])[0])
    assert np.array_equal(off["lp"][:d + 1], on["lp"][:d + 1])   # same until the halt
    for k in policy.EVENT_DAYS:
        assert out["event"]["off"][k] == off["lp"][d + k] - off["lp"][d]
        assert out["event"]["on"][k] == on["lp"][d + k] - on["lp"][d]
    # the halt day fell to the first level in the run with breakers
    assert out["event"]["on"][1] <= np.log(1 - 0.003) or on["halts"][d] > 0
    assert out["on"]["halt_days"] == int(np.sum(on["halts"][policy.BURN:] > 0))


def test_breaker_experiment_collects_every_seed():
    whole, events, bad = policy.breaker_experiment(SerialPool(), CALM, (1, 2, 3), 1500, TIGHT)
    assert bad == 0
    assert all(len(v) == 3 for side in whole.values() for v in side.values())
    assert set(events["on"]) == set(policy.EVENT_DAYS) and len(events["on"][1]) == 3


def test_a_surprise_in_a_calm_market_moves_the_price_by_its_size():
    # Fundamentalists alone pull the price to the new value: the fall is 1 surprise and
    # without the surprise nothing moves (placebo 0). One 0.1% tick is 0.0095 surprises.
    # The price can only rest on whole ticks, so before and after it sits up to about a
    # tick from its value: after 30 days the fall is within 2 ticks of 1 (over 4,000
    # seeds: at most 1.73 ticks). The largest overshoot on the way was 2.7 ticks; 4 are
    # allowed. (The old test looked after 10 days and allowed 2 ticks: the price had not
    # always settled, and it failed on 15 of 39 other seeds; overnight stress test.)
    tick = np.log(1.001) / -np.log(0.9)
    out, bad = policy.surprise_experiment(SerialPool(), QUIET, range(1, 9), size=-0.1, horizon=30)
    assert bad == 0
    assert np.all(np.abs(out["end"] - 1) < 2 * tick)
    assert np.all(out["peak"] < 1 + 4 * tick)
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


# ------------------------------------------------------------------ honest reporting
# `end` is the price's fall in units of the surprise, so an `end` of 1.30 after a 10%
# surprise means the price fell 0.30 x 10.5% (about 3 percentage points) more than the
# value, not "30% further down". RESULTS.md must read it that way, and must say that
# `end` has no placebo (part of it is ordinary wobble).

def test_results_reads_the_10_day_surprise_measure_in_the_right_units():
    from pathlib import Path
    results = (Path(__file__).parent.parent / "figures" / "RESULTS.md").read_text()
    row = next(line for line in results.splitlines() if line.startswith("| 10–20% |"))
    end = float(row.strip("|").split("|")[-1])
    extra_points = (end - 1.0) * -np.log(0.9) * 100      # extra fall beyond the value, in %
    assert extra_points == pytest.approx(3.2, abs=0.1)
    text = " ".join(results.split())                       # ignore line breaks
    assert "30% further down" not in text
    assert "about 3 percentage points more than the value" in text
    assert "no placebo for the 10-day number" in text


# The surprise experiment (make_figures.py, Fig 11) runs only one size (-10%), so the
# documents may compare surprises at different crowding, but must not claim that timing
# matters "as much as" size: no run varies the size.

def test_documents_do_not_compare_timing_with_size_of_a_surprise():
    from pathlib import Path
    root = Path(__file__).parent.parent
    for name in ("figures/RESULTS.md", "docs/KEY_POINTS.md"):
        text = " ".join((root / name).read_text().split())   # ignore line breaks
        assert "matters as much as its size" not in text, name


def test_median_overshoot_pairs_each_history_with_its_own_placebo():
    # placebo_peak is stored minus 1, so the overshoot of one history is
    # peak - (placebo_peak + 1). Here every history overshoots by exactly 0.2.
    placebo = np.random.default_rng(1).normal(0.1, 0.05, 50)
    peak = placebo + 1 + 0.2
    med, lo, hi = policy.median_overshoot(peak, placebo)
    assert (med, lo, hi) == pytest.approx((0.2, 0.2, 0.2))
    # Pairing matters: the median of differences is not the difference of medians.
    peak, placebo = np.array([1.0, 2.0, 3.0]), np.array([1.0, -0.5, 0.0])
    med, lo, hi = policy.median_overshoot(peak, placebo)
    assert med == pytest.approx(1.5)                       # differences -1, 1.5, 2
    assert np.median(peak) - np.median(placebo + 1) == pytest.approx(1.0)
    assert lo <= med <= hi
    with pytest.raises(ValueError):
        policy.median_overshoot([1.0], [0.0])


# The overshoot in a crowded market (10-20% chartists) rests on 18 histories, so the
# documents must say so, give the paired median with its bootstrap interval, and not
# present "a quarter" as a precise number.

def test_documents_give_the_crowded_overshoot_with_its_sample_size_and_interval():
    from pathlib import Path
    root = Path(__file__).parent.parent
    results = " ".join((root / "figures" / "RESULTS.md").read_text().split())
    assert "18 histories" in results
    assert "0.29 (95% interval 0.14 to 0.41)" in results
    assert "difference of the two medians" in results     # says the table gap is a shortcut
    key = " ".join((root / "docs/KEY_POINTS.md").read_text().split())
    assert "18 histories" in key
    assert "0.14 to 0.41" in key


def test_scaled_breakers_keeps_the_3_25_pm_cutoff():
    rules, _ = policy.scaled_breakers(0.02, 0.01)
    assert rules["last_pause_minute"] == 355 and rules["halt_minutes"] == 15


# Audit of November 2026, issue 3: NaN used to give (nan, nan, nan) without an error
@pytest.mark.parametrize("f", [policy.paired_difference, policy.median_overshoot])
@pytest.mark.parametrize("a, b", [([1.0, np.nan, 3.0], [1.0, 2.0, 2.0]),
                                  ([1.0, 2.0, 3.0], [1.0, 2.0])])
def test_bootstrap_helpers_reject_nan_and_unequal_lengths(f, a, b):
    with pytest.raises(ValueError):
        f(a, b)


# ------------------------------------------------------------------ found by the overnight mutation test (Nov 2026)

def test_crash_measures_by_hand():
    r = np.array([0.01, -0.03, -0.02, 0.005, -0.25, 0.02] * 5)
    m = policy.crash_measures(r, 0.02)
    # a crash day falls 2% or more: log return <= ln(0.98) = -0.0202. So -0.03 and -0.25 count
    # and -0.02 (a 1.98% fall) does not: 2 a block, 5 blocks
    assert m["crash_days"] == 10
    assert m["worst_1d"] == experiments.worst_fall(r, 1) == pytest.approx(0.25)   # cumulative sums: round-off
    assert m["worst_5d"] == experiments.worst_fall(r, 5)
    assert m["worst_20d"] == experiments.worst_fall(r, 20)
    assert policy.EVENT_DAYS == (1, 5, 20)


def test_the_bootstrap_intervals_resample_every_history():
    d = np.arange(1.0, 11.0)
    mean, lo, hi = policy.paired_difference(np.zeros(10), d, draws=500, seed=3)
    g = np.random.default_rng(3)
    means = d[g.integers(0, 10, (500, 10))].mean(axis=1)
    assert (mean, lo, hi) == (5.5, np.quantile(means, 0.025), np.quantile(means, 0.975))
    med, lo, hi = policy.median_overshoot(d + 1, np.zeros(10), draws=500, seed=3)
    g = np.random.default_rng(3)
    meds = np.median(d[g.integers(0, 10, (500, 10))], axis=1)
    assert (med, lo, hi) == (5.5, np.quantile(meds, 0.025), np.quantile(meds, 0.975))
    policy.paired_difference([0.0, 0.0], [1.0, 2.0])          # 2 pairs: the smallest allowed


def test_closed_days_count_only_level_3():
    out = policy._breaker_pair((CALM, 2, 2000, TIGHT))
    on = model.simulate(CALM, 2000 + policy.BURN, seed=2, breakers=TIGHT)
    assert out["on"]["closed_days"] == int(np.sum(on["halts"][policy.BURN:] == 3))
    assert out["on"]["closed_days"] < out["on"]["halt_days"]
