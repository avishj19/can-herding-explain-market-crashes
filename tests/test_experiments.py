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
