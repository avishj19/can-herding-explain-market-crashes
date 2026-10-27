"""Tests for calibrate.py: the search is sound and the result holds on new seeds."""
import json

import numpy as np
import pytest

from herding import calibrate
from herding import model
from herding import stats

TARGET = {k: 0.0 for k in calibrate.TARGETS}


def test_loss_exact_by_hand():
    # one fact off by exactly 2 SCALE units, another by 1 -> 2^2 + 1^2 = 5
    f = dict(TARGET)
    f["alpha_5pct"] = 2 * calibrate.SCALE["alpha_5pct"]
    f["ac_r_2"] = -calibrate.SCALE["ac_r_2"]
    assert calibrate.loss(f, TARGET) == pytest.approx(5.0)


def test_loss_ignores_kurtosis_and_rejects_failed_runs():
    f = dict(TARGET, kurtosis=1e6)                  # kurtosis is reported, not targeted
    assert calibrate.loss(f, TARGET) == 0
    assert calibrate.loss(None, TARGET) == float("inf")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sample_covers_each_range_evenly(seed):
    # Latin hypercube: with n samples, each of the n equal slices of every
    # range holds exactly one sample
    n = 50
    ps = calibrate.sample(n, seed)
    for k, (lo, hi) in calibrate.RANGES.items():
        v = np.array([p[k] for p in ps])
        assert np.all((v >= lo) & (v <= hi))
        assert sorted(np.floor((v - lo) / (hi - lo) * n).astype(int)) == list(range(n))
    for p in ps:                                     # every sample is a full, valid setting
        assert set(p) == set(model.P)


def test_model_facts_returns_none_for_runaway_settings():
    assert calibrate.model_facts(dict(model.P, s=0.25), seeds=[201], days=5000) is None


def test_model_facts_averages_over_seeds():
    one = [calibrate.model_facts(model.P, [s], days=2000) for s in (0, 1)]
    both = calibrate.model_facts(model.P, [0, 1], days=2000)
    for k in both:
        assert both[k] == pytest.approx((one[0][k] + one[1][k]) / 2)


def test_saved_calibration_is_valid_and_inside_ranges():
    p = calibrate.load()
    for k, (lo, hi) in calibrate.RANGES.items():
        assert lo <= p[k] <= hi, f"{k} outside its search range"
    assert p["dt"] <= 1 / (p["beta"] * p["gamma"])  # stability limit


def test_load_rejects_corrupted_file(tmp_path):
    bad = dict(calibrate.load(), sgma_news=0.01)
    f = tmp_path / "cal.json"
    f.write_text(json.dumps({"params": bad}))
    with pytest.raises(ValueError, match="sgma_news"):
        calibrate.load(f)


# ------------------------------------------------- held-out check (the real test)
# Spec: on seeds the search never used, the calibrated model matches the S&P 500
# far better than the paper's settings, never runs away, and keeps the base case.

HELD_OUT = range(100, 106)


@pytest.mark.slow
def test_calibration_beats_paper_settings_on_new_seeds():
    target = calibrate.spx_facts()
    cal = calibrate.loss(calibrate.model_facts(calibrate.load(), HELD_OUT), target)
    paper = calibrate.loss(calibrate.model_facts(model.P, HELD_OUT), target)
    assert cal < paper / 3, f"calibrated {cal:.1f} vs paper {paper:.1f}"


@pytest.mark.slow
def test_calibrated_tail_exponent_is_inverse_cubic():
    f = calibrate.model_facts(calibrate.load(), HELD_OUT)
    assert 2.7 < f["alpha_5pct"] < 3.6 and 2.7 < f["alpha_1pct"] < 3.9


@pytest.mark.slow
def test_calibrated_model_has_no_two_day_reversal():
    # The paper settings fail this (-0.17, see test_model.py); calibration fixes it
    f = calibrate.model_facts(calibrate.load(), HELD_OUT)
    assert abs(f["ac_r_2"]) < 0.1 and abs(f["ac_r_1"]) < 0.1


@pytest.mark.slow
def test_calibrated_model_has_volatility_memory():
    f = calibrate.model_facts(calibrate.load(), HELD_OUT)
    assert f["ac_abs_20"] > 0.08 and f["ac_abs_100"] > 0.0


@pytest.mark.slow
def test_calibrated_model_has_long_memory_like_spx():
    # Was a known limitation (0.02 at 100 days) until the 50- and 100-day memory
    # were both targeted; S&P: 0.125 at 50 days, 0.094 at 100 days
    f = calibrate.model_facts(calibrate.load(), HELD_OUT)
    assert f["ac_abs_50"] > 0.08 and f["ac_abs_100"] > 0.05


@pytest.mark.slow
def test_calibrated_reversal_is_small_over_days_2_to_10():
    # The reversal must not hide past day 2 (paper settings: -0.044, S&P: -0.005)
    f = calibrate.model_facts(calibrate.load(), HELD_OUT)
    assert abs(f["ac_r_2to10"]) < 0.03


@pytest.mark.slow
@pytest.mark.parametrize("seed", HELD_OUT)
def test_calibrated_settings_never_run_away(seed):
    r, _ = model.simulate(19600, p=calibrate.load(), seed=seed)
    assert np.all(np.isfinite(r))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_calibrated_base_case_is_still_a_bell_curve(seed):
    r, _ = model.simulate(5000, herding=False, p=calibrate.load(), seed=seed)
    assert abs(stats.kurtosis(r)) < 0.3


# ------------------------------------------------ typicality (model has randomness)
# Spec: a stochastic model is right if the real market looks like a TYPICAL
# history from it: S&P percentile among simulated histories far from 0 and 100.

def test_percentile_exact_by_hand():
    sims = [{"x": v} for v in [1.0, 2.0, 3.0, 4.0]]
    assert calibrate.percentiles({"x": 2.5}, sims, ["x"]) == {"x": 50.0}
    assert calibrate.percentiles({"x": 0.0}, sims, ["x"]) == {"x": 0.0}
    assert calibrate.percentiles({"x": 9.0}, sims, ["x"]) == {"x": 100.0}


def test_representative_is_median_loss_history():
    target = {k: 0.0 for k in calibrate.TARGETS}
    sims = [{k: d * calibrate.SCALE[k] for k in calibrate.TARGETS} for d in (0.1, 0.5, 0.2, 0.9, 0.3)]
    # losses grow with d -> median d is 0.3, at index 4
    assert calibrate.representative(sims, target) == 4


@pytest.mark.slow
def test_spx_is_a_typical_history_of_the_calibrated_model():
    sims = [calibrate.model_facts(calibrate.load(), [s]) for s in range(500, 520)]
    pct = calibrate.percentiles(calibrate.spx_facts(), sims, calibrate.TARGETS)
    unusual = [k for k, v in pct.items() if not 5 <= v <= 95]
    assert len(unusual) <= 1, f"S&P looks unusual on {unusual}"
