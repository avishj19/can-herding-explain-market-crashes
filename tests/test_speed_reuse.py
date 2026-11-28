"""Tests of the Nov 2026 speed-ups that must not change any number: a model run hands back its
day-by-day market (explore.run_mix / run_mixes paths=, explore.market_paths), the site reuses it for
the Simulated market chart instead of running the same history again (web/common._MARKETS), and the
server warms the Story and Findings caches at start (common._warm_pages)."""
import numpy as np
import pytest

from herding import explore as ex
from herding import model

P, BURN = ex.fitted_settings()
A1, A2, V1 = float(P["a1"]), float(P["a2"]), float(P["v1"])
KEYS = ("price", "value", "z", "x", "r")


# Spec: the paths a run hands back are exactly model_market's, for every mix and with a party drift.
@pytest.mark.parametrize("party", [0.0, 0.02])
def test_run_mixes_paths_equal_model_market(party):
    paths = {}
    out = ex.run_mixes(P, BURN, A1, 0.3, 1.5, 3000, 7, party=party, paths=paths)
    for m in ex.MIXES:
        if out[m][0] is None:
            assert m not in paths            # a run that ran away hands back nothing
            continue
        mm = ex.model_market(P, BURN, m, A1, 0.3, 1.5, 3000, 7, party=party)
        for k in KEYS:
            assert np.array_equal(paths[m][k], mm[k]), (m, k)
        assert paths[m]["runaway_day"] is None
        assert np.array_equal(paths[m]["r"], out[m][0])


# Spec: mixes given in known= are not run, so they hand back no paths; the others do.
def test_run_mixes_paths_only_for_mixes_run_here():
    first = ex.run_mixes(P, BURN, A1, A2, V1, 2000, 3)
    paths = {}
    ex.run_mixes(P, BURN, 1.1, A2, V1, 2000, 3, known={"fundamentalists": first["fundamentalists"]}, paths=paths)
    assert "fundamentalists" not in paths and "herding" in paths


# Spec: a run that ran away fills nothing (the Simulated market rebuilds it with the guard off).
def test_run_mix_paths_empty_after_a_runaway():
    paths = {}
    r, err = ex.run_mix(P, "chartists", 0.0, 19329, 1000, BURN, paths=paths)
    assert r is None and isinstance(err, model.RunawayError)
    assert paths == {}


# Spec: market_paths by hand on a made-up 3-day output: price and value start from the price's day 0.
def test_market_paths_by_hand():
    out = {"lp": np.log([50.0, 100.0, 110.0, 99.0]), "lf": np.log([50.0, 100.0, 100.0, 100.0]),
           "z": np.array([.1, .2, .3, .4]), "x": np.array([0, .5, -.5, 0])}
    mm = ex.market_paths(out, 1, ran=7)
    assert mm["price"] == pytest.approx([100, 110, 99], rel=1e-12)
    assert mm["value"] == pytest.approx([100, 100, 100], rel=1e-12)
    assert mm["r"] == pytest.approx(np.diff(np.log([100, 110, 99])), rel=1e-12)
    assert list(mm["z"]) == [.2, .3, .4] and mm["runaway_day"] == 7


# Spec: on the site, after the mixes run, the Simulated market's history comes from the store (no second
# model run), and equals a fresh explore.model_market exactly.
def test_site_reuses_the_history_without_running_it_again(monkeypatch):
    import streamlit as st
    from web import common as c
    st.cache_data.clear()
    c._MARKETS.clear()
    c._MIX_RESULTS.clear()
    run = dict(a1=1.23, a2=A2, v1=V1, days=3000, seed=4, party=0.0)
    c.run_mixes(**run)
    calls = []
    real = model.simulate
    monkeypatch.setattr(model, "simulate", lambda *a, **k: calls.append(1) or real(*a, **k))
    mm = c.model_market("herding", **run)
    assert calls == []
    fresh = ex.model_market(P, BURN, "herding", 1.23, A2, V1, 3000, 4)
    for k in KEYS:
        assert np.array_equal(mm[k], fresh[k])


# Spec: the store keeps at most MARKETS_MAX histories, dropping the oldest.
def test_site_market_store_is_bounded():
    import streamlit as st
    from web import common as c
    st.cache_data.clear()
    c._MARKETS.clear()
    c._MIX_RESULTS.clear()
    for i in range(c.MARKETS_MAX // 3 + 3):
        c.run_mixes(a1=0.5 + i / 100, a2=A2, v1=V1, days=600, seed=i, party=0.0)
    assert 0 < len(c._MARKETS) <= c.MARKETS_MAX


# Spec: warming fills the Story's default history, so the page's own call then runs nothing.
def test_warm_pages_fills_the_story_cache(monkeypatch):
    import streamlit as st
    from web import common as c
    st.cache_data.clear()
    c._warm_pages()
    calls = []
    real = model.simulate
    monkeypatch.setattr(model, "simulate", lambda *a, **k: calls.append(1) or real(*a, **k))
    c.run_mixes(**c.default_run())
    assert calls == []
