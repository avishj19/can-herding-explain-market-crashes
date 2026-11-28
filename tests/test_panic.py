"""Tests of model.simulate's panic option (extension, Nov 2026; scripts/analysis/panic_skew.py):
off it is SPEC.md exactly, on it changes the history, bad values are refused."""
import json

import numpy as np
import pytest

from conftest import ROOT
from herding import explore as ex, model

P, BURN = ex.fitted_settings()


# Spec: panic = 0 (the default) gives bit-identical histories to the model without the option:
# the saved held-out numbers (made before the option existed) come back exactly.
def test_panic_off_is_the_old_model_exactly():
    nums = json.loads((ROOT / "web" / "numbers.json").read_text())
    r, err = ex.run_mix(P, "herding", P["a1"], nums["recipe"]["days"], 1000, BURN)
    assert err is None
    assert ex.summary(r)["tail"] == nums["checks"]["herding"]["tail"][0]
    a = model.simulate(P, 1500, seed=3)
    b = model.simulate(P, 1500, seed=3, panic=0.0)
    for k in ("lp", "lf", "z", "x"):
        assert np.array_equal(a[k], b[k])


# Spec: panic > 0 changes the history (the term is really used), and the same seed repeats exactly.
def test_panic_on_changes_the_history_repeatably():
    a = model.simulate(P, 1500, seed=3)
    b = model.simulate(P, 1500, seed=3, panic=4.0)
    c = model.simulate(P, 1500, seed=3, panic=4.0)
    assert not np.array_equal(a["lp"], b["lp"])
    assert np.array_equal(b["lp"], c["lp"])


# Spec: by the rule, panic only acts on falls: in a market whose trend never falls it can't matter.
# With no chartists the mood is never used, so any panic gives the same history.
def test_panic_needs_chartists():
    a = model.simulate(P, 800, seed=5, chartists=False)
    b = model.simulate(P, 800, seed=5, chartists=False, panic=10.0)
    assert np.array_equal(a["lp"], b["lp"])


@pytest.mark.parametrize("bad", [-0.1, model.PANIC_MAX + 0.01, float("nan"), float("inf"), "1", None, True])
def test_panic_rejects_bad_values(bad):
    with pytest.raises(ValueError, match="panic"):
        model.simulate(P, 10, seed=0, panic=bad)


# Spec: the analysis script's measures by hand: a mirror-image sample has skewness 0 and ratio 1.
def test_panic_skew_measures_by_hand():
    import importlib.util
    spec = importlib.util.spec_from_file_location("panic_skew", ROOT / "scripts" / "analysis" / "panic_skew.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    x = np.random.default_rng(0).standard_t(4, 5000)
    s, q = mod.skew_measures(np.r_[x, -x])
    assert s == pytest.approx(0, abs=1e-12) and q == pytest.approx(1, abs=1e-12)
    s, q = mod.skew_measures(np.r_[x, -x, [-40.0]])
    assert s < 0
