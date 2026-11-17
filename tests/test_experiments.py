"""Tests for herding/experiments.py."""
from multiprocessing import Pool

import numpy as np
import pytest

from herding import experiments, model


def test_histories_drop_the_burn_in_and_match_a_direct_run():
    p = dict(model.SET_I)
    with Pool(2) as pool:
        out = experiments.histories(p, [2, 3], 200, keep=("r", "z"), pool=pool)
    direct = model.simulate(p, 200 + experiments.BURN, seed=2)
    assert np.array_equal(out[0]["r"], direct["r"][experiments.BURN:])
    assert np.array_equal(out[0]["z"], direct["z"][experiments.BURN:])
    assert len(out[1]["r"]) == 200


@pytest.mark.pinned_seed          # needs a seed that runs away and one that does not
def test_a_runaway_history_is_reported_not_lost():
    p = dict(model.SET_I)            # seed 1 runs away on day 28,511
    with Pool(2) as pool:            # one real pool for both calls
        out = experiments.histories(p, [1, 2], 29000 - experiments.BURN, pool=pool)
        assert out[0] is None and out[1] is not None
        with pytest.raises(model.RunawayError, match=r"\[1\]"):
            experiments.returns(p, [1, 2], 29000 - experiments.BURN, pool=pool)


def test_percentile_by_hand():
    sims = [1, 2, 3, 4]
    assert experiments.percentile(0, sims) == 0
    assert experiments.percentile(5, sims) == 100
    assert experiments.percentile(2.5, sims) == 50
    assert experiments.percentile(2, sims) == 37.5      # 1 below, 1 tie -> (1 + 0.5) / 4
    with pytest.raises(ValueError):
        experiments.percentile(1, [])


def test_worst_fall_by_hand():
    r = [0.01, -0.02, -0.03, 0.04, -0.01]
    assert experiments.worst_fall(r, 1) == pytest.approx(0.03)
    assert experiments.worst_fall(r, 2) == pytest.approx(0.05)
    assert experiments.worst_fall(r, 5) == pytest.approx(0.01)
    with pytest.raises(ValueError):
        experiments.worst_fall(r, 6)


def test_typical_picks_the_median_mismatch_not_the_best():
    target = {"a": 0.0, "b": 0.0}
    prints = [{"a": d, "b": d} for d in (5.0, 0.1, 3.0, 1.0, 2.0)]   # distances by hand
    assert experiments.typical(prints, target) == 4                  # the 2.0 one (median)
    assert experiments.typical(prints[:1], target) == 0
    with pytest.raises(ValueError):
        experiments.typical([], target)


def test_worst_fall_needs_a_whole_number_of_days():
    r = np.array([-0.01, 0.02, -0.03, 0.01, -0.02, 0.0])
    assert experiments.worst_fall(r, 2.0) == experiments.worst_fall(r, 2)
    with pytest.raises(ValueError, match="whole number"):
        experiments.worst_fall(r, 2.5)


# Audit of November 2026, issue 3: NaN used to come out as a quiet answer
# (percentile(nan, [1, 2, 3]) was 0.0, which reads as "below every history").
@pytest.mark.parametrize("value, sims", [(np.nan, [1, 2, 3]), (1.0, [1, np.nan, 3]),
                                         (np.inf, [1, 2, 3]), (1.0, [[1, 2], [3, 4]])])
def test_percentile_rejects_nan_and_not_1d(value, sims):
    with pytest.raises(ValueError):
        experiments.percentile(value, sims)


@pytest.mark.parametrize("r", [[np.nan, 1.0], [1.0, np.inf], [[1.0, 2.0]]])
def test_worst_fall_rejects_nan_and_not_1d(r):
    with pytest.raises(ValueError):
        experiments.worst_fall(r, 1)


def test_window_sums_by_hand():
    assert list(experiments.window_sums([1, 2, 4, 8], 2)) == [3, 6, 12]
    assert list(experiments.window_sums([1, 2, 4, 8], 4)) == [15]
    with pytest.raises(ValueError, match="n must"):
        experiments.window_sums([1, 2], 3)


# ------------------------------------------------------------------ found by the overnight mutation test (Nov 2026)

def test_typical_picks_the_median_mismatch_by_hand():
    fps = [dict(a=0.0, b=10.0), dict(a=1.0, b=10.0), dict(a=2.0, b=14.0),
           dict(a=3.0, b=10.0), dict(a=9.0, b=30.0)]
    target = dict(a=1.0, b=12.0)
    v = np.array([[f["a"], f["b"]] for f in fps])
    sd = v.std(axis=0)
    miss = (((v - [1.0, 12.0]) / sd) ** 2).mean(axis=1)
    assert experiments.typical(fps, target) == int(np.argsort(miss)[2])
    # a measure that never varies counts in its own units (spread 1), not divided by 0
    flat = [dict(a=x, c=5.0) for x in (0.0, 1.0, 2.0)]
    assert experiments.typical(flat, dict(a=0.0, c=7.0)) == 1


@pytest.mark.parametrize("k", [0, 4, True, 1.5])
def test_worst_fall_rejects_impossible_windows(k):
    with pytest.raises(ValueError, match="k must"):
        experiments.worst_fall([0.1, -0.2, 0.3], k)
