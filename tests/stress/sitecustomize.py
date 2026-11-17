"""Seed shift for the overnight stress test (tests/stress/overnight.py). Inert unless the
environment variable HERDING_SEED_SHIFT is set.

Every random number in the project comes from np.random.default_rng(seed). With
HERDING_SEED_SHIFT=k, an integer seed s becomes the seed sequence [s, k], a different but
still repeatable stream: same s and k, same numbers; another k, new numbers. The tests are
not edited at all. Python imports a file named sitecustomize at start-up when its folder is
on PYTHONPATH, so this also reaches worker processes that a test starts.
"""
import os

_SHIFT = os.environ.get("HERDING_SEED_SHIFT")

if _SHIFT:
    import numpy as np

    _default_rng = np.random.default_rng
    _TAG = 20261117                    # keeps shifted streams apart from any plain seed list

    def default_rng(seed=None):
        if isinstance(seed, (int, np.integer)) and not isinstance(seed, bool) and seed >= 0:
            seed = np.random.SeedSequence([int(seed), _TAG, int(_SHIFT)])
        return _default_rng(seed)

    np.random.default_rng = default_rng
