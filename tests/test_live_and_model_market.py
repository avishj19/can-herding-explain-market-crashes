"""Tests of the November 2026 additions behind the Test program page (herding/explore.py):
run_mixes running its three mixes at the same time, model_market (the "Simulated market" chart),
the fresh end of live prices (recent_prices, splice_recent), Yahoo's live quote (live_quote) and
the model-vs-market scorecard. No test uses the internet: Yahoo is always a fake here."""
import numpy as np
import pandas as pd
import pytest

from herding import explore as ex
from herding import model

P, BURN = ex.fitted_settings()
A1, A2, V1 = float(P["a1"]), float(P["a2"]), float(P["v1"])


# ------------------------------------------------------------------ run_mixes in parallel

# Spec: the same histories as running each mix alone, one after another (the threads share nothing).
def test_parallel_run_mixes_equals_one_mix_at_a_time():
    out = ex.run_mixes(P, BURN, A1, 0.3, 1.5, 3000, 11)
    q = dict(P, a2=0.3, v1=1.5)
    for m in ex.MIXES:
        r, err = ex.run_mix(q, m, A1, 3000, 11, BURN)
        assert (out[m][0] is None) == (r is None)
        if r is not None:
            assert np.array_equal(out[m][0], r)
        else:
            assert out[m][1] == ex.runaway_day(err, BURN)


# Spec: a bad input still raises the same ValueError (it is raised inside a thread, then passed on).
def test_parallel_run_mixes_passes_errors_on():
    with pytest.raises(ValueError, match="days"):
        ex.run_mixes(P, BURN, A1, A2, V1, 10, 0)


# Spec: running it twice gives the same numbers (no shared random state between the threads).
def test_parallel_run_mixes_is_repeatable():
    a = ex.run_mixes(P, BURN, A1, A2, V1, 2000, 5)
    b = ex.run_mixes(P, BURN, A1, A2, V1, 2000, 5)
    for m in ex.MIXES:
        assert (a[m][0] is None and b[m][0] is None) or np.array_equal(a[m][0], b[m][0])


# ------------------------------------------------------------------ model_market

# Spec: its returns are exactly run_mixes' returns for that mix, and price/value start where they should.
def test_model_market_is_the_same_history_as_run_mixes():
    mixes = ex.run_mixes(P, BURN, A1, A2, V1, 3000, 3)
    for m in ex.MIXES:
        if mixes[m][0] is None:
            continue
        mm = ex.model_market(P, BURN, m, A1, A2, V1, 3000, 3)
        assert mm["runaway_day"] is None
        assert np.array_equal(mm["r"], mixes[m][0])
        assert len(mm["price"]) == len(mm["value"]) == len(mm["z"]) == len(mm["x"]) == 3001
        assert mm["price"][0] == 100.0
        # the price index follows the returns: 100 * exp(cumulative log return)
        assert np.allclose(mm["price"][1:], 100 * np.exp(np.cumsum(mm["r"])), rtol=1e-12)


# Spec: value traders only have no chartists (share 0) and their price stays near the value.
def test_model_market_fundamentalists_have_no_chartists():
    mm = ex.model_market(P, BURN, "fundamentalists", A1, A2, V1, 2000, 1)
    assert np.all(mm["z"] == 0)
    assert np.max(np.abs(np.log(mm["price"] / mm["value"]))) < 0.2


# Spec: chartist share is a fraction and mood is between -1 and 1.
def test_model_market_shares_are_in_range():
    mm = ex.model_market(P, BURN, "herding", A1, A2, V1, 2000, 2)
    assert np.all((mm["z"] >= 0) & (mm["z"] <= 1))
    assert np.all((mm["x"] >= -1) & (mm["x"] <= 1))


# Spec: a history that runs away is shown up to its runaway day, and that day matches run_mixes.
def test_model_market_runaway_stops_on_the_runaway_day():
    mixes = ex.run_mixes(P, BURN, 0.0, A2, V1, 10000, 0)
    day = mixes["chartists"][1]
    assert day is not None and day > 0               # documented: chartists, seed 0 run away
    mm = ex.model_market(P, BURN, "chartists", 0.0, A2, V1, 10000, 0)
    assert mm["runaway_day"] == day
    assert len(mm["price"]) == day + 1
    # the days before are the same draws as the full run would have made
    q, _ = ex.mix_settings(P, "chartists", 0.0)
    full = model.simulate(q, BURN + day, seed=0, max_gap=None)
    assert np.array_equal(mm["r"], np.diff(full["lp"][BURN:]))


# Spec: a history that ran away in the warm-up has nothing to show: ValueError.
def test_model_market_warmup_runaway_raises():
    mixes = ex.run_mixes(P, BURN, 0.0, 1.0, 5.0, 5000, 1)
    assert mixes["chartists"][1] is not None and mixes["chartists"][1] < 1   # documented case
    with pytest.raises(ValueError, match="warm-up"):
        ex.model_market(P, BURN, "chartists", 0.0, 1.0, 5.0, 5000, 1)


@pytest.mark.parametrize("bad", [dict(mix="traders"), dict(a1=-1.0), dict(seed=-1), dict(days=0)])
def test_model_market_rejects_bad_input(bad):
    kw = dict(mix="herding", a1=A1, a2=A2, v1=V1, days=1000, seed=0)
    kw.update(bad)
    with pytest.raises(ValueError):
        ex.model_market(P, BURN, kw["mix"], kw["a1"], kw["a2"], kw["v1"], kw["days"], kw["seed"])


# ------------------------------------------------------------------ recent_prices / splice_recent

def _closes(dates, values, name="T"):
    return pd.Series(values, index=pd.DatetimeIndex(dates), name=name, dtype=float)


# Spec: asks Yahoo from RECENT_DAYS calendar days before today, and keeps only those dates.
def test_recent_prices_asks_for_the_last_days_only():
    asked = []

    def fake(t, start):
        asked.append((t, start))
        return _closes(["2026-11-01", "2026-11-20", "2026-11-23", "2026-11-24"], [1, 2, 3, 4])
    out = ex.recent_prices("abc", download=fake, today="2026-11-24 15:00")
    assert asked == [("ABC", "2026-11-14")]
    assert list(out.values) == [2, 3, 4] and out.name == "ABC"


# Spec: a bar stamped within the day becomes its date (the last one of a day wins); a time zone is
# dropped without moving the date (BE-18).
def test_recent_prices_dates_and_time_zones():
    idx = pd.DatetimeIndex(["2026-11-23 00:00", "2026-11-24 00:00"]).tz_localize("UTC")
    out = ex.recent_prices("BTC-USD", download=lambda t, s: pd.Series([5.0, 6.0], index=idx),
                           today="2026-11-24")
    assert list(out.index.strftime("%Y-%m-%d")) == ["2026-11-23", "2026-11-24"]


def test_recent_prices_errors():
    with pytest.raises(ConnectionError):
        ex.recent_prices("ABC", download=lambda t, s: 1 / 0, today="2026-11-24")
    with pytest.raises(ValueError, match="no recent prices"):
        ex.recent_prices("ABC", download=lambda t, s: pd.Series(dtype=float), today="2026-11-24")
    with pytest.raises(ValueError):
        ex.recent_prices("not a ticker!", download=lambda t, s: None)


# Spec: newer closes replace the same dates and later dates are added, in order.
def test_splice_recent_updates_and_appends():
    close = _closes(["2026-11-18", "2026-11-19", "2026-11-20"], [10, 11, 12], "S")
    recent = _closes(["2026-11-19", "2026-11-20", "2026-11-23"], [11, 12.5, 13])
    out = ex.splice_recent(close, recent)
    assert list(out.values) == [10, 11, 12.5, 13] and out.name == "S"
    assert out.index.is_monotonic_increasing and not out.index.has_duplicates


# Spec: recent prices that start after the history's last close are not added (the days between
# would be missing); None or empty recent leaves the history as it is.
def test_splice_recent_never_leaves_a_gap():
    close = _closes(["2026-10-29", "2026-10-30"], [10, 11])
    later = _closes(["2026-11-20", "2026-11-23"], [20, 21])
    assert ex.splice_recent(close, later).equals(close)
    assert ex.splice_recent(close, None).equals(close)
    assert ex.splice_recent(close, _closes([], [])).equals(close)


# Spec: dates in recent older than RECENT_DAYS before the history's end can't rewrite old data.
def test_splice_recent_keeps_old_data():
    close = _closes(pd.bdate_range("2026-10-01", "2026-10-30"), np.arange(22.0))
    recent = _closes(["2026-10-02", "2026-10-30"], [999, 50])
    out = ex.splice_recent(close, recent)
    assert out["2026-10-02"] == close["2026-10-02"] and out["2026-10-30"] == 50


def test_splice_recent_needs_dates():
    with pytest.raises(ValueError):
        ex.splice_recent(pd.Series([1.0, 2.0]), None)


# ------------------------------------------------------------------ live_quote

def test_live_quote_change():
    q = ex.live_quote("spy", quote=lambda t: (110.0, 100.0))
    assert q == {"price": 110.0, "prev_close": 100.0, "change": pytest.approx(0.1, abs=1e-12)}


@pytest.mark.parametrize("bad", [(None, 100.0), (0.0, 100.0), (-1.0, 100.0), (np.nan, 100.0),
                                 (100.0, np.inf), (100.0, 0.0), ("x", 1.0)])
def test_live_quote_rejects_nonsense(bad):
    with pytest.raises(ValueError):
        ex.live_quote("SPY", quote=lambda t: bad)


def test_live_quote_network_trouble_is_a_connection_error():
    with pytest.raises(ConnectionError):
        ex.live_quote("SPY", quote=lambda t: 1 / 0)


# Spec: the tests never reach the internet: conftest makes the real quote act as if Yahoo were down.
def test_tests_have_no_live_quotes():
    with pytest.raises(ConnectionError):
        ex.live_quote("SPY")


# ------------------------------------------------------------------ scorecard

def _summ(tail, big, days, memory):
    return {"tail": tail, "big_days": big, "days": days, "memory": memory}


def test_scorecard_verdicts():
    market = _summ(3.0, 10, 10000, 0.20)
    same = ex.scorecard(market, _summ(3.0, 10, 10000, 0.20))
    assert [r["verdict"] for r in same] == ["match"] * 3
    assert [r["what"] for r in same] == ["Fat tails", "Big days (per 1,000 days)", "Bursts of wild days"]
    close = ex.scorecard(market, _summ(3.6, 20, 10000, 0.29))       # 20% off, 2x, 0.09 apart
    assert [r["verdict"] for r in close] == ["close"] * 3
    off = ex.scorecard(market, _summ(6.0, 0, 10000, 0.0))
    assert [r["verdict"] for r in off] == ["off"] * 3


# Spec: crash days are compared per 1,000 days, so histories of different lengths compare fairly.
def test_scorecard_crash_days_per_1000():
    rows = ex.scorecard(_summ(3.0, 10, 10000, 0.2), _summ(3.0, 20, 20000, 0.2))
    assert rows[1]["market"] == rows[1]["model"] == 1.0 and rows[1]["verdict"] == "match"


# Spec: the edges of each rule: exactly on the match limit is a match.
def test_scorecard_edges():
    m = _summ(3.0, 10, 10000, 0.2)
    assert ex.scorecard(m, _summ(3.3, 15, 10000, 0.25))[0]["verdict"] == "match"
    assert ex.scorecard(m, _summ(3.0, 15, 10000, 0.2))[1]["verdict"] == "match"
    assert ex.scorecard(m, _summ(3.0, 0, 10000, 0.2))[1]["verdict"] == "off"
    assert ex.scorecard(_summ(3.0, 0, 10000, 0.2), _summ(3.0, 0, 5000, 0.2))[1]["verdict"] == "match"


def test_scorecard_ran_away():
    assert ex.scorecard(_summ(3.0, 10, 10000, 0.2), None) == []


# ------------------------------------------------------------------ shareable links

DAYS = [5000, 10000, 19329]
MARKETS = ["S&P 500", "Gold", "NVDA"]
FIT_RUN = dict(a1=A1, a2=A2, v1=V1, days=19329, seed=1000)


# Spec: write_link then read_link gives back exactly the same run (the fit's long decimals included).
@pytest.mark.parametrize("run", [FIT_RUN, dict(FIT_RUN, a1=1.6, seed=7, days=5000)])
def test_link_round_trip_is_exact(run):
    got = ex.read_link(ex.write_link("Gold", run), DAYS, MARKETS)
    assert got == {"market": "Gold", **run}


# Spec: settings equal to the defaults are left out of the link.
def test_link_leaves_out_defaults():
    assert ex.write_link("S&P 500", FIT_RUN, "S&P 500", FIT_RUN) == {}
    assert ex.write_link("Gold", dict(FIT_RUN, seed=7), "S&P 500", FIT_RUN) == {"market": "Gold", "seed": "7"}


# Spec: anything out of range, not a number, unknown or hostile is ignored, never an error, and a
# link can never name a market outside the saved list (so it can't make the server fetch anything).
@pytest.mark.parametrize("params,want", [
    ({"a1": "2.5", "a2": "-0.1", "v1": "0.4", "party": "11"}, {}),
    ({"a1": "nan", "a2": "inf", "v1": "1e309", "party": "-inf"}, {}),
    ({"a1": "<script>", "days": "12345", "seed": "10001", "market": "TSLA"}, {}),
    ({"seed": "-1", "days": "5000.0", "market": "../../etc/passwd"}, {}),
    ({"seed": "1" * 500, "a1": "1" * 500, "market": "S&P 500<img src=x>"}, {}),
    ({"a1": "0", "seed": "0", "days": "5000", "market": "S&P 500"},
     {"a1": 0.0, "seed": 0, "days": 5000, "market": "S&P 500"}),
    ({"a1": "2", "a2": "1", "v1": "5", "seed": "10000"},
     {"a1": 2.0, "a2": 1.0, "v1": 5.0, "seed": 10000}),
    ({"party": "3.5", "a1": "1"}, {"a1": 1.0}),      # an old link's party drift (removed Nov 2026) is ignored
])
def test_read_link_checks_everything(params, want):
    assert ex.read_link(params, DAYS, MARKETS) == want


def test_read_link_never_raises_on_odd_containers():
    for params in (None, [], "a1=1", {"a1": None, "seed": None, "market": None}, {"a1": ["1"]}):
        assert ex.read_link(params, DAYS, MARKETS) == {}


# ------------------------------------------------------------------ big days with a known cause

def test_day_events_file():
    e = ex.day_events()
    assert len(e) == 20 and e.index.is_monotonic_increasing
    assert all(u.startswith("https://") for us in e["sources"] for u in us)     # no "nan" for a blank cell
    assert sum(len(us) == 0 for us in e["sources"]) == 2
    assert e.loc["1987-10-19", "move_percent"] == pytest.approx(-22.61)


def test_day_story_with_and_without_a_cause():
    text, src = ex.day_story("1987-10-19 00:00", -0.2047, -20.93)
    assert text.startswith("19 October 1987: fell 20.5%, 20.9 standard deviations. What happened: Black Monday")
    assert src == ["https://www.federalreservehistory.org/essays/stock-market-crash-of-1987"]
    text, src = ex.day_story("2020-06-11", -0.059, -5.2)
    assert "not among the Dow's 20 biggest days" in text and src == []
    assert ex.day_story("2008-10-13", 0.115, 8.0)[0].startswith("13 October 2008: rose 11.5%")


# ------------------------------------------------------------------ FRED, the backup for daily closes

FRED_TEXT = "observation_date,SP500\n2026-11-20,6910.39\n2026-11-21,.\n2026-11-23,7237.17\n2026-11-24,7201.09\n"


def test_fred_closes_reads_the_table_and_skips_holidays():
    out = ex.fred_closes("^GSPC", download=lambda s: FRED_TEXT)
    assert list(out.index.strftime("%Y-%m-%d")) == ["2026-11-20", "2026-11-23", "2026-11-24"]
    assert list(out.values) == [6910.39, 7237.17, 7201.09] and out.name == "^GSPC"


def test_fred_closes_asks_for_the_right_series():
    asked = []
    ex.fred_closes("^DJI", download=lambda s: asked.append(s) or "observation_date,DJIA\n2026-11-20,1\n2026-11-21,2\n")
    assert asked == ["DJIA"]


@pytest.mark.parametrize("text", ["<html>checking your browser</html>", "", "observation_date,OTHER\n2026-11-20,1\n",
                                  "observation_date,SP500\nnot a date,1\n", "observation_date,SP500\n2026-11-20,5\n"])
def test_fred_closes_rejects_odd_replies(text):
    with pytest.raises(ValueError):
        ex.fred_closes("^GSPC", download=lambda s: text)


def test_fred_closes_errors():
    with pytest.raises(ValueError, match="no daily closes"):
        ex.fred_closes("NVDA", download=lambda s: FRED_TEXT)
    with pytest.raises(ConnectionError):
        ex.fred_closes("^GSPC", download=lambda s: 1 / 0)
    with pytest.raises(ConnectionError):         # conftest: no internet in tests
        ex.fred_closes("^GSPC")


# Spec: FRED's closes join the saved history without a gap (they overlap its last days).
def test_fred_closes_extend_the_saved_history():
    saved = ex.saved_prices("S&P 500")
    text = "observation_date,SP500\n" + "".join(f"{d:%Y-%m-%d},{v}\n" for d, v in saved.iloc[-5:].items()) \
        + "2026-11-02,7600.5\n"
    out = ex.splice_recent(saved, ex.fred_closes("^GSPC", download=lambda s: text))
    assert len(out) == len(saved) + 1 and out.iloc[-1] == 7600.5


# ------------------------------------------------------------------ reusing mixes (mix_key)

def test_mix_key_names_what_each_mix_depends_on():
    assert ex.mix_key("fundamentalists", 1.0, 0.2, 3.0) == ()
    assert ex.mix_key("chartists", 1.0, 0.2, 3.0) == (0.2, 3.0)
    assert ex.mix_key("herding", 1.0, 0.2, 3.0) == (1.0, 0.2, 3.0)
    with pytest.raises(ValueError):
        ex.mix_key("crowd", 1.0, 0.2, 3.0)


# Spec: the claim behind mix_key, checked: fundamentalists don't depend on a1, a2, v1 and chartists
# without herding don't depend on a1, so their histories can be reused.
def test_mix_key_claim_holds():
    a = ex.run_mixes(P, BURN, 0.5, 0.1, 1.0, 1500, 4)
    b = ex.run_mixes(P, BURN, 1.7, 0.1, 1.0, 1500, 4)
    c_ = ex.run_mixes(P, BURN, 1.7, 0.9, 4.0, 1500, 4)
    assert np.array_equal(a["chartists"][0], b["chartists"][0])
    assert np.array_equal(a["fundamentalists"][0], c_["fundamentalists"][0])


# Spec: known results are used as given and not run again; the rest are the usual histories.
def test_run_mixes_reuses_known_results():
    marker = (np.array([9.0]), None)
    out = ex.run_mixes(P, BURN, A1, A2, V1, 1500, 4, known={"fundamentalists": marker})
    assert out["fundamentalists"] is marker
    full = ex.run_mixes(P, BURN, A1, A2, V1, 1500, 4)
    assert np.array_equal(out["herding"][0], full["herding"][0])


# Spec: on the site, moving only the herding slider runs one history (trend chasers who herd); the
# other two mixes come from the shared store (web/common.py _MIX_RESULTS).
def test_site_reruns_only_the_mix_that_changed(monkeypatch):
    import streamlit as st
    from web import common as c
    st.cache_data.clear()
    c._MIX_RESULTS.clear()
    ran = []
    real = ex.run_mix
    monkeypatch.setattr(ex, "run_mix", lambda q, m, *a: ran.append(m) or real(q, m, *a))
    c.run_mixes(1.0, A2, V1, 1500, 3)
    assert sorted(ran) == sorted(ex.MIXES)
    ran.clear()
    out = c.run_mixes(1.4, A2, V1, 1500, 3)
    assert ran == ["herding"]
    assert np.array_equal(out["herding"][0], ex.run_mixes(P, BURN, 1.4, A2, V1, 1500, 3)["herding"][0])
    ran.clear()
    c.run_mixes(1.4, 0.5, V1, 1500, 3)           # a2 changes chartists too, not fundamentalists
    assert sorted(ran) == ["chartists", "herding"]
    st.cache_data.clear()
