"""Input checks shared by every module, so each rule is written once.

A bad input should fail loudly with a clear ValueError, never turn into a quiet wrong
answer (NaN, a dropped imaginary part, ...).
"""
import numpy as np


def is_number(x):
    """True for a finite real number. Not True/False, and not complex: numpy counts
    complex numbers as numbers (np.number), so the real types are named one by one."""
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, (bool, np.bool_)) \
        and bool(np.isfinite(x))


def finite_1d(x, name):
    """x as a 1-D float array; reject NaN/inf (they would turn into a quiet wrong answer).
    `name` says what x is in the error message, e.g. "returns"."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"{name} must be one list of numbers (1-D), got shape {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contain NaN or infinity")
    return x
