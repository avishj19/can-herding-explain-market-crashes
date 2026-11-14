"""Tests for checks.py: the input checks every module shares."""
import numpy as np
import pytest

from herding import checks, experiments, kirman, model, policy, stats


@pytest.mark.parametrize("x", [0, 1, -2.5, 1e300, np.float32(0.5), np.int64(7), np.float64(2.0)])
def test_real_finite_numbers_are_numbers(x):
    assert checks.is_number(x) is True


@pytest.mark.parametrize("x", [True, False, np.bool_(True), float("nan"), float("inf"), -float("inf"),
                               1 + 0j, np.complex128(1), "1", None, [1], np.array([1.0])])
def test_everything_else_is_not_a_number(x):
    assert checks.is_number(x) is False


def test_finite_1d_returns_a_float_array_and_names_the_input():
    out = checks.finite_1d([1, 2, 3], "widgets")
    assert out.dtype == float and out.tolist() == [1.0, 2.0, 3.0]
    assert checks.finite_1d([], "widgets").size == 0            # empty is allowed here
    for bad, words in [([[1, 2], [3, 4]], "widgets must be one list"), ([1.0, np.nan], "widgets contain NaN"),
                       ([np.inf], "widgets contain NaN"), (5.0, "widgets must be one list")]:
        with pytest.raises(ValueError, match=words):
            checks.finite_1d(bad, "widgets")


def test_every_module_uses_the_one_shared_check():
    # one rule, written once: the modules must not carry their own copies again
    assert model._is_number is checks.is_number and kirman.is_number is checks.is_number
    assert experiments.finite_1d is checks.finite_1d and policy.finite_1d is checks.finite_1d
    assert stats.finite_1d is checks.finite_1d
