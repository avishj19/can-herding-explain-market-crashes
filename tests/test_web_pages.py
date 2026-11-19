"""Tests of the web app's pages, run headless with Streamlit's AppTest (no browser, no server).

    python3 -m pytest tests/test_web_pages.py -q          (slow: each test runs the model once)

Yahoo is faked by replacing herding.explore._yahoo / _yahoo_intraday (the documented way; it works
because live_prices and intraday_prices look those functions up when called), and every network
connection is refused, so a test can never reach the internet.
"""
import json
import math
import re
import socket

import numpy as np
import pandas as pd
import pytest

from pathlib import Path

from conftest import ROOT
from herding import explore as ex

pytestmark = pytest.mark.slow


def _daily_until(end, n=1500, seed=0):
    idx = pd.bdate_range(end=end, periods=n)
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01)), index=idx)


def _no_intraday(t, p, i):
    raise OSError("no intraday bars in this test")


def _no_daily(t, s):
    raise OSError("no daily prices in this test")


def _bars(day="2026-11-18", values=(100.0, 101.0, 100.5, 102.0)):
    """Fake 5-minute bars of one day, in UTC like Yahoo's (14:30 UTC = 9:30 New York)."""
    t0 = pd.Timestamp(f"{day} 14:30", tz="UTC")
    return pd.Series(list(values), index=pd.DatetimeIndex([t0 + pd.Timedelta(minutes=5 * i)
                                                           for i in range(len(values))]))


@pytest.fixture
def app(monkeypatch):
    """A fresh Workspace with no network: returns open_workspace(daily=None, intraday=None)."""
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    def refuse(*a, **k):
        raise OSError("network blocked in tests")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    st.cache_data.clear()                 # cached live prices from another test must not leak in

    def open_workspace(daily=None, intraday=None, ticker=None, frame=None):
        # always faked: yfinance downloads through curl, which the socket block above doesn't stop
        monkeypatch.setattr(ex, "_yahoo", (lambda t, s: daily) if daily is not None else _no_daily)
        monkeypatch.setattr(ex, "_yahoo_intraday", intraday or _no_intraday)
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
        at.run()
        at.switch_page("web/page_explore.py").run()
        if ticker:
            at.text_input(key="ticker_box").set_value(ticker).run()
            assert at.session_state["asset"] == ("live", ticker), at.session_state["ticker_error"]
        if frame:
            at.session_state["frame"] = frame
            at.run()
        return at
    yield open_workspace
    st.cache_data.clear()


def _stat(at, label):
    """The text of one number in the stats row under the range buttons."""
    for m in at.markdown:
        hit = re.search(rf"<div class='l'>{re.escape(label)}</div><div class='v'[^>]*>([^<]*)</div>", m.value)
        if hit:
            return hit.group(1)
    raise AssertionError(f"no {label!r} in the stats row")


def test_workspace_insight_says_more_big_days(app):
    # WS-1: the S&P 500 had about 89 times MORE big days than a bell curve, not "fewer"
    at = app()
    assert not at.exception
    text = " ".join(c.value for c in at.caption)
    assert re.search(r"about \d+ times more", text) and "times fewer" not in text


def test_workspace_1d_with_an_infinite_intraday_bar(app):
    # BE-2: an inf bar used to survive and crash range_stats inside the chart
    bars = pd.Series([100.0, np.inf, 101.0, 102.0], index=pd.DatetimeIndex(
        ["2026-11-18 14:30", "2026-11-18 15:00", "2026-11-18 15:30", "2026-11-18 16:00"], tz="UTC"))
    at = app(daily=_daily_until("2026-11-18"), intraday=lambda t, p, i: bars, ticker="ZZZA", frame="1D")
    assert not at.exception, [e.value for e in at.exception]
    assert _stat(at, "High") == "102.00"


def test_workspace_1d_with_time_zoned_daily_prices(app):
    # BE-3: time-zoned daily closes compared with plain intraday times raised TypeError
    daily = _daily_until("2026-11-18").tz_localize("America/New_York")
    bars = pd.Series([100.0, 101.0, 102.0], index=pd.DatetimeIndex(
        ["2026-11-18 14:30", "2026-11-18 15:00", "2026-11-18 15:30"], tz="UTC"))
    at = app(daily=daily, intraday=lambda t, p, i: bars, ticker="ZZZB", frame="1D")
    assert not at.exception, [e.value for e in at.exception]


def test_workspace_7d_volatility_counts_every_day_shown(app):
    # BE-4: the first day of an intraday range was left out of the volatility
    daily = _daily_until("2026-11-18")
    bars = pd.Series(np.linspace(100, 105, 5), index=pd.DatetimeIndex(
        [f"2026-11-{d} 14:30" for d in (12, 13, 14, 17)] + ["2026-11-18 20:55"], tz="UTC"))
    at = app(daily=daily, intraday=lambda t, p, i: bars, ticker="ZZZC", frame="7D")
    assert not at.exception, [e.value for e in at.exception]
    closes = daily["2026-11-12":"2026-11-18"].values        # 12th-18th: five closes, four moves
    want = np.diff(np.log(closes)).std(ddof=1) * np.sqrt(252)
    assert _stat(at, "Volatility (yr)") == f"{want:.1%}"


def test_workspace_ytd_on_the_first_trading_day_of_the_year(app):
    # BE-1 / WS-3: with one close in the new year, YTD had a single price and range_stats raised
    daily = _daily_until("2027-01-01")
    at = app(daily=daily, ticker="ZZZD", frame="YTD")
    assert not at.exception, [e.value for e in at.exception]
    # year to date = from the last close of 2026 (31 Dec) to 1 Jan 2027: two prices
    two = daily.loc["2026-12-31":"2027-01-01"].values
    assert len(two) == 2
    assert _stat(at, "High") == f"{two.max():,.2f}" and _stat(at, "Low") == f"{two.min():,.2f}"
    assert _stat(at, "Change") == f"{'▲' if two[1] >= two[0] else '▼'} {abs(two[1] / two[0] - 1):.2%}"


# ---------------------------------------------------------------------------------- helpers
def _fig(at, key):
    """The plotly figure (as a dict) drawn with this key."""
    for pc in at.get("plotly_chart"):
        if pc.proto.id.endswith(f"-{key}"):
            return json.loads(pc.proto.spec)
    raise AssertionError(f"no chart with key {key!r}")


def _trace(fig, name):
    for t in fig["data"]:
        if t.get("name") == name:
            return t
    raise AssertionError(f"no trace {name!r} in {[t.get('name') for t in fig['data']]}")


def _values(v):
    """A plotly array (a list, or numpy's base64 'bdata' form) as a numpy array."""
    import base64
    if isinstance(v, dict) and "bdata" in v:
        return np.frombuffer(base64.b64decode(v["bdata"]), dtype=np.dtype(v["dtype"])).astype(float)
    return np.asarray(v)


def _run_model(at, **controls):
    """Set the Workspace's model controls (in_a1=..., in_days=...) and press Run model."""
    for k, v in controls.items():
        if k == "in_days":
            at.selectbox(key=k).set_value(v)
        elif k == "in_seed":
            at.number_input(key=k).set_value(v)
        else:
            at.slider(key=k).set_value(v)
    [b for b in at.button if b.label == "Run model"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]


def _model_table(at):
    for d in at.dataframe:
        if "Tail exponent" in list(d.value.index):
            return d.value
    raise AssertionError("no Model vs market table")


def _captions(at):
    return " ".join(c.value for c in at.caption)


def _simulated_runaway_day(a1, a2, v1, days, seed, chartists=True):
    """The day (after burn-in) a history runs away, straight from model.simulate (not explore)."""
    from herding import model
    fit = json.loads((ROOT / "settings" / "sp500_calibrated.json").read_text())
    p, burn = dict(fit["settings"], a1=a1, a2=a2, v1=v1), int(fit["burn"])
    with pytest.raises(model.RunawayError) as e:
        model.simulate(p, days + burn, seed=seed, chartists=chartists)
    return e.value.day - burn


@pytest.fixture
def open_page(monkeypatch):
    """Any page of the app, with no network: open_page('web/page_story.py')."""
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(socket.socket, "connect", lambda *a, **k: (_ for _ in ()).throw(OSError("blocked")))
    monkeypatch.setattr(ex, "_yahoo", _no_daily)
    monkeypatch.setattr(ex, "_yahoo_intraday", _no_intraday)
    st.cache_data.clear()

    def go(path, **query):
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
        for k, v in query.items():
            at.query_params[k] = v
        at.run()
        at.switch_page(path).run()
        assert not at.exception, [e.value for e in at.exception]
        return at
    yield go
    st.cache_data.clear()


FIT = json.loads((ROOT / "settings" / "sp500_calibrated.json").read_text())
NUMBERS = json.loads((ROOT / "web" / "numbers.json").read_text())


# ---------------------------------------------------------------------------------- Workspace
def test_workspace_runaway_during_warmup_is_not_a_negative_day(app):
    # WS-2: a1 0, a2 1, v1 5, seed 1 runs away during the 1,000 warm-up days; the table said "day -745"
    assert _simulated_runaway_day(0.0, 1.0, 5.0, 5000, 1) < 0       # independent: really in the warm-up
    at = app()
    _run_model(at, in_a1=0.0, in_a2=1.0, in_v1=5.0, in_days=5000, in_seed=1)
    days = _model_table(at).loc["Days", "Chartists, no herding"]
    assert days == "ran away during the warm-up, before day 1"


def test_workspace_1d_uses_saved_closes_when_live_is_off(app):
    # WS-4: with Live prices off, 1D fetched Yahoo's bars while the header showed the saved close
    calls = []

    def bars(t, p, i):
        calls.append(t)
        return _bars()
    at = app(intraday=bars, frame="1D")
    assert not at.exception, [e.value for e in at.exception]
    assert calls == []
    last2 = pd.read_csv(ROOT / "data" / "sp500_daily.csv", index_col=0)["Close"].iloc[-2:].values
    assert _stat(at, "High") == f"{last2.max():,.2f}" and _stat(at, "Low") == f"{last2.min():,.2f}"
    assert "need Live prices on" in _captions(at)


def test_workspace_intraday_note_never_credits_yahoo_for_saved_prices(app):
    # WS-5: Live on, Yahoo's daily prices failing (so the page falls back to the saved file) and the
    # intraday bars working: the note said "Yahoo's daily prices to 30 Oct 2026" (the saved file's date)
    at = app(intraday=lambda t, p, i: _bars())
    at.session_state["asset"] = ("saved", "Gold")
    [t for t in at.toggle if t.label == "Live prices"][0].set_value(True)
    at.session_state["frame"] = "1D"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert "saved data" in at.markdown[1].value or any("saved data" in m.value for m in at.markdown)
    assert "Yahoo's daily prices" not in _captions(at)
    assert "need Live prices on" in _captions(at)


def test_workspace_1y_high_and_low_match_the_52_week_numbers(app):
    # WS-6: 1Y and the 52-week numbers now cover the same 52 weeks
    at = app()
    at.session_state["asset"] = ("saved", "ORCL")
    at.session_state["frame"] = "1Y"
    at.run()
    assert _stat(at, "High") == _stat(at, "52-week high")
    assert _stat(at, "Low") == _stat(at, "52-week low")


def test_workspace_1d_change_is_labelled_since_open(app):
    # WS-7: the 1D "Change" was measured from the first 5-minute bar, not yesterday's close
    bars = _bars(values=(100.0, 99.0, 101.0, 101.5))
    at = app(daily=_daily_until("2026-11-17"), intraday=lambda t, p, i: bars, ticker="ZZZE", frame="1D")
    assert not at.exception, [e.value for e in at.exception]
    assert _stat(at, "Since open") == f"▲ {101.5 / 100.0 - 1:.2%}"
    with pytest.raises(AssertionError):
        _stat(at, "Change")


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_workspace_dial_caption_names_no_color_the_dot_is_not(open_page, mode):
    # WS-8: "White dot" while the dot is near-black in light mode
    at = open_page("web/page_explore.py", mode=mode)
    dot = _trace(_fig(at, "dial"), "Shown now")["marker"]["color"]
    assert dot == {"light": "#111827", "dark": "#F2F4F8"}[mode]
    text = _captions(at)
    assert "Solid dot: the run shown now" in text and "White dot" not in text


def test_workspace_herding_runaway_is_explained_in_every_view(app):
    # WS-9: "Weak herding" (a1 0.6, seed 1000) runs away; Bursts showed an empty panel, Tail a missing line
    day = _simulated_runaway_day(0.6, FIT["settings"]["a2"], FIT["settings"]["v1"], 19329, 1000)
    at = app()
    at.radio(key="preset").set_value("Weak herding").run()
    [b for b in at.button if b.label == "Run model"][0].click().run()
    at.session_state["view"] = "Bursts"
    at.run()
    notes = [a["text"] for a in _fig(at, "bursts_chart")["layout"].get("annotations", [])]
    assert any(f"ran away on day {day:,}" in n for n in notes), notes
    at.session_state["view"] = "Tail"
    at.run()
    assert f"\"Chartists who herd\" ran away on day {day:,}, so it has no line." in _captions(at)


def test_workspace_ticker_error_clears_when_another_market_is_picked(app):
    # WS-10: the error for '!!!' stayed next to the market picked afterwards
    at = app()
    at.text_input(key="ticker_box").set_value("!!!").run()
    assert any("not a ticker" in e.value for e in at.error)
    at.session_state["asset"] = ("saved", "Gold")        # what picking a row in the market list does
    at.run()
    assert not at.error


@pytest.mark.parametrize("typed, saved", [("^GSPC", "S&P 500"), ("^dji", "Dow Jones"), ("gc=f", "Gold"),
                                          ("BZ=F", "Brent oil")])
def test_workspace_yahoo_symbols_of_saved_markets_open_the_saved_market(app, typed, saved):
    # WS-11: ^GSPC opened a live "Stock" instead of the saved S&P 500
    at = app()
    at.session_state["asset"] = ("saved", "AAPL")         # start elsewhere, so staying put can't pass
    at.run()
    at.text_input(key="ticker_box").set_value(typed).run()
    assert at.session_state["asset"] == ("saved", saved)
    assert any(f"<h1 style='margin:0'>{saved}</h1>" in m.value for m in at.markdown)


def test_workspace_run_log_has_number_columns(app):
    # WS-12: "ran away" text in a tail column made Arrow fail and turned the numbers into text
    at = app()
    at.radio(key="preset").set_value("Strong herding").run()
    [b for b in at.button if b.label == "Run model"][0].click().run()
    _run_model(at, in_a1=0.0, in_a2=0.0, in_v1=0.5, in_days=5000, in_seed=0)
    log = [d.value for d in at.dataframe if "Ran away" in d.value.columns][0]
    for m in ("Fundamentalists only", "Chartists, no herding", "Chartists who herd"):
        assert log[m].dtype == float
    assert np.isnan(log.loc[1, "Chartists, no herding"])            # Strong herding: chartists ran away
    assert "Chartists, no herding" in log.loc[1, "Ran away"]


def test_workspace_a1_zero_shows_one_chartist_column(app):
    # WS-13: at a1 = 0 "Chartists who herd" is the same history as "Chartists, no herding"
    at = app()
    _run_model(at, in_a1=0.0, in_a2=0.0, in_v1=0.5, in_days=5000, in_seed=0)
    assert "Chartists who herd" not in _model_table(at).columns
    at.session_state["view"] = "Tail"
    at.run()
    names = [t.get("name") for t in _fig(at, "model_chart_Tail")["data"]]
    assert "Chartists, no herding" in names and "Chartists who herd" not in names


def test_workspace_earlier_runs_toggle_and_labels(app):
    # WS-14: the toggle showed with nothing earlier, labels hid a2/v1, and at most 3 lines showed
    at = app()
    _run_model(at)                                          # the default run again: nothing earlier
    at.session_state["view"] = "Tail"
    at.run()
    assert not [t for t in at.toggle if t.label == "Show my earlier runs"]
    for a2 in (0.10, 0.20, 0.30, 0.40, 0.50):
        _run_model(at, in_a1=1.0, in_a2=a2, in_days=5000, in_seed=0)
    at.toggle(key="t_earlier").set_value(True).run()
    names = [t.get("name") for t in _fig(at, "model_chart_Tail")["data"] if "Earlier" in str(t.get("name"))]
    assert len(names) == 4, names                           # the last four runs other than the one shown
    assert all("a2 0." in n for n in names) and len(set(names)) == 4


def test_workspace_same_run_twice_is_logged_once(app):
    # WS-15
    at = app()
    _run_model(at, in_days=5000)
    _run_model(at)
    assert len(at.session_state["log"]) == 1


def test_workspace_preset_follows_the_sliders_and_can_restore_the_fit(app):
    # WS-17: after moving a1 the radio still said "S&P 500 fit" and clicking it did nothing
    at = app()
    assert at.radio(key="preset").value == "S&P 500 fit"
    _run_model(at, in_a1=1.5)
    assert at.radio(key="preset").value is None
    at.radio(key="preset").set_value("S&P 500 fit").run()
    assert at.slider(key="in_a1").value == pytest.approx(FIT["settings"]["a1"])


def test_workspace_bell_column_says_its_length(app):
    # WS-18: the bell column was for the market's 19,329 days next to 5,000-day model columns
    from scipy.stats import norm
    at = app()
    _run_model(at, in_days=5000)
    assert "Bell curve (market length)" in _model_table(at).columns
    assert f"as long as the model run it expects {5000 * 2 * norm.sf(4):.1f} days beyond 4 sd" in _captions(at)


def test_workspace_bursts_thresholds_are_neutral(app):
    # PG-5 (also on the Workspace): the +-4 sd lines used the loss color
    at = app()
    at.session_state["view"] = "Bursts"
    at.run()
    colors = {s["line"]["color"] for s in _fig(at, "bursts_chart")["layout"]["shapes"]}
    assert colors == {"#5C5E6B"}                          # dark GUIDE, not DOWN #FF7A6B


# ---------------------------------------------------------------------------------- shared parts
def test_mode_button_knows_every_page(open_page):
    # PG-1: the light/dark choice was saved only for pages the browser had opened before
    at = open_page("web/page_guide.py")
    script = [h.proto.body for h in at.get("html") if "stActiveTheme" in h.proto.body][0]
    paths = json.loads(re.search(r"const PAGE_PATHS = (\[.*?\]);", script).group(1))
    want = {"/"} | {f"/{p.stem}" for p in (ROOT / "web").glob("page_*.py") if p.stem != "page_guide"}
    assert set(paths) == want


def test_dial_hides_points_where_most_histories_ran_away(open_page):
    # PG-3: a1 0.4 and 0.6 had 1 surviving history each, drawn as a "median of 10"
    at = open_page("web/page_explore.py")
    line = _trace(_fig(at, "dial"), "Model (median)")
    rows = NUMBERS["sweep"]["rows"]
    want = sorted({r["a1"] for r in rows if sum(q["tail"] is not None for q in rows if q["a1"] == r["a1"]) >= 5})
    assert list(_values(line["x"])) == pytest.approx(want)
    for a1, y in zip(_values(line["x"]), _values(line["y"])):
        assert y == pytest.approx(np.median([r["tail"] for r in rows if r["a1"] == a1 and r["tail"] is not None]))
    assert len(NUMBERS["sweep"]["seeds"]) == 10 and "of 10" in _trace(_fig(at, "dial"), line["name"])["hovertemplate"]


def test_dial_with_every_history_ran_away(monkeypatch):
    # PG-4: an all-None sweep made pandas fail ("dtype 'object' does not support operation 'quantile'")
    from web import common as c
    fake = dict(NUMBERS, sweep=dict(NUMBERS["sweep"], rows=[dict(r, tail=None) for r in NUMBERS["sweep"]["rows"]]))
    monkeypatch.setattr(c, "numbers", lambda: fake)
    fig = c.dial_chart(3.0)
    assert not [t for t in fig.data if t.name == "Model (median)"]
    assert any("ran away" in a.text for a in fig.layout.annotations)


def test_css_only_styles_parts_that_exist():
    # PG-12: rules for test ids Streamlit 1.67 no longer has never applied
    import streamlit
    static = Path(streamlit.__file__).parent / "static"
    bundle = "".join(p.read_text(errors="ignore") for p in static.rglob("*.js"))
    ids = set(re.findall(r'data-testid=\\?"(\w+)', (ROOT / "web" / "common.py").read_text()))
    assert ids and all(i in bundle for i in ids), [i for i in ids if i not in bundle]
    assert "stMetric" not in (ROOT / "web" / "common.py").read_text()       # no page uses st.metric


# ---------------------------------------------------------------------------------- Story
def test_story_runaway_chart_counts_days_like_the_text(open_page):
    # PG-2: the chart's days included the 1,000 warm-up days, so the runaway was drawn 1,000 days late
    day = _simulated_runaway_day(0.0, FIT["settings"]["a2"], FIT["settings"]["v1"], 19329, 1000)
    at = open_page("web/page_story.py")
    x = _values(_trace(_fig(at, "s3"), "Share of chartists")["x"])
    assert x[0] == 0 and x[-1] == day
    assert any(f"for about {day - 3:,} days" in m.value for m in at.markdown)


def test_story_colors_and_wording(open_page):
    # PG-5: red (loss) marked an upward bubble and the +-4 sd lines; PG-10: "three index funds ... it never saw"
    at = open_page("web/page_story.py")
    ann = _fig(at, "s3")["layout"]["annotations"]
    off = [a for a in ann if a.get("text") == "off the chart"][0]
    assert off["arrowcolor"] != "#FF7A6B" and off["font"]["color"] != "#FF7A6B"
    assert {s["line"]["color"] for s in _fig(at, "s4b")["layout"]["shapes"]} == {"#5C5E6B"}
    text = " ".join(m.value for m in at.markdown)
    assert "index funds" not in text and "two other indexes (Nasdaq-100, Dow Jones)" in text


def test_story_numbers_match_the_saved_histories(open_page):
    # PG-8/DUP-17: Story's medians now come from explore.spread; checked with plain numpy here
    at = open_page("web/page_story.py")
    text = " ".join(m.value for m in at.markdown)
    k = NUMBERS["checks"]["fundamentalists"]["kurtosis"]
    tails = [t for t in NUMBERS["checks"]["herding"]["tail"] if t is not None]
    gone = sum(d is not None for d in NUMBERS["checks"]["chartists"]["runaway_day"])
    assert f"**{np.median(k):.2f}** (median) over {len(k)} histories" in text
    assert f"**{np.median(tails):.2f}** (median) over {len(tails)} histories" in text
    assert f"In **{gone} of 40** histories" in text


# ---------------------------------------------------------------------------------- Checks
def test_checks_numbers_match_the_saved_histories(open_page):
    # PG-8: the page's own pass rule and medians replaced by explore.tail_check / spread / check_table
    at = open_page("web/page_checks.py")
    text = " ".join(m.value for m in at.markdown)
    tails = [t for t in NUMBERS["checks"]["herding"]["tail"] if t is not None]
    med = np.median(tails)
    assert f"Model median **{med:.2f}** (range {min(tails):.2f} to {max(tails):.2f})" in text
    assert ("passes" if 2.5 <= med <= 3.5 else "does not pass") in text.split("Model median")[1].split("\n")[0]
    table = [d.value for d in at.dataframe][0]
    v = [x for x in NUMBERS["checks"]["herding"]["memory"] if x is not None]
    lo, hi = np.percentile(v, [5, 95])
    assert f"{np.median(v):.2f} ({lo:.2f} to {hi:.2f})" in table.to_string()


def test_checks_verdict_uses_the_median(open_page, monkeypatch):
    # PG-8: median 3.0 (passes) while the mean is 5.95 (would fail): the page must say passes
    from web import common as c
    tails = [3.0] * 21 + [9.0] * 19
    fake = json.loads(json.dumps(NUMBERS))
    fake["checks"]["herding"]["tail"] = tails
    monkeypatch.setattr(c, "numbers", lambda: fake)
    at = open_page("web/page_checks.py")
    line = [m.value for m in at.markdown if "Model median" in m.value][0]
    assert "Model median **3.00**" in line and "class='ok'>passes" in line


@pytest.mark.parametrize("eps, words", [(0.005, "so they herd."), (0.01, "wander across the whole range"),
                                        (0.02, "split about evenly")])
def test_checks_kirman_words_match_the_exact_shape(open_page, eps, words):
    # PG-9: at eps 0.01 the text said "split about evenly" over an almost flat distribution
    from herding import kirman
    pi = kirman.stationary(100, eps, 0.1)
    flat = pi[50] / pi[0]                       # 0.14, 1.45, 44.9: herd, nearly flat, a clear middle
    at = open_page("web/page_checks.py")
    at.select_slider[0].set_value(eps).run()
    text = " ".join(m.value for m in at.markdown)
    assert words in text
    assert (flat < 2) == (eps < 0.02)


# ---------------------------------------------------------------------------------- Compare
def test_compare_stars_match_the_saved_histories(open_page):
    at = open_page("web/page_compare.py")
    star = _trace(_fig(at, "map"), "Model: chartists who herd")
    tails = [t for t in NUMBERS["checks"]["herding"]["tail"] if t is not None]
    assert _values(star["x"])[0] == pytest.approx(np.median(tails))
    assert _values(star["error_x"]["array"])[0] == pytest.approx(np.percentile(tails, 95) - np.median(tails))


def test_compare_empty_picks_say_what_to_do(open_page):
    # PG-7
    at = open_page("web/page_compare.py")
    at.get("button_group")[0].set_value([]).run()
    assert "Pick a group above" in _captions(at)
    at.multiselect[0].set_value([]).run()
    assert "Pick up to 6 markets" in _captions(at)


def test_compare_some_chartist_histories_survive(open_page, monkeypatch):
    # PG-11: "all 35 of their histories ran away" when 5 of 40 survived, and no star for them
    from web import common as c
    fake = json.loads(json.dumps(NUMBERS))
    h = fake["checks"]["herding"]
    keep = set(range(5))
    fake["checks"]["chartists"] = {
        "runaway_day": [None if i in keep else 100 for i in range(40)],
        "tail": [h["tail"][i] if i in keep else None for i in range(40)],
        "memory": [h["memory"][i] if i in keep else None for i in range(40)]}
    monkeypatch.setattr(c, "numbers", lambda: fake)
    at = open_page("web/page_compare.py")
    assert "35 of 40 histories ran away" in _captions(at) and "all 35" not in _captions(at)
    star = _trace(_fig(at, "map"), "Model: chartists, no herding")
    assert _values(star["x"])[0] == pytest.approx(np.median([h["tail"][i] for i in keep]))


# ---------------------------------------------------------------------------------- Findings, Start here
def test_findings_clicking_the_picked_section_keeps_it(open_page):
    # PG-6: unselecting showed section 0 under an empty control
    at = open_page("web/page_findings.py")
    bg = at.get("button_group")[0]
    sections = list(dict.fromkeys(e["section"] for e in ex.gallery_entries(
        (ROOT / "figures" / "GALLERY.md").read_text())))
    bg.set_value(sections[2]).run()
    first = [m.value for m in at.markdown][2:4]
    at.get("button_group")[0].set_value(None).run()
    assert at.session_state["findings_section"] == sections[2]
    assert [m.value for m in at.markdown][2:4] == first


def test_guide_example_chart_shows_the_whole_tail(open_page):
    # PG-13: 20,000 samples reach a share of 1/20,000, below the axis' 0.01%
    at = open_page("web/page_guide.py")
    fig = _fig(at, "guide_example")
    lowest = min(_values(t["y"]).min() for t in fig["data"])
    assert lowest == pytest.approx(1 / 20000, rel=0.5)
    assert fig["layout"]["yaxis"]["range"][0] <= math.log10(lowest)


# ================================================================== breaker round 1 (WS-19..22)

def test_workspace_max_drawdown_never_shows_minus_zero(app):
    # WS-19: a range that never fell showed "Max drawdown -0.0%"
    at = app()
    for name in ("CVX", "Brent oil"):
        at.session_state["asset"] = ("saved", name)
        at.session_state["frame"] = "1D"
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        last2 = ex.saved_prices(name).iloc[-2:].values         # 1D without intraday: the last 2 closes
        assert _stat(at, "Max drawdown") == f"{max(0.0, 1 - last2[1] / last2[0]):.1%}"
        assert not _stat(at, "Max drawdown").startswith("-")


def test_workspace_bursts_ytd_shows_only_this_years_moves(app):
    # WS-20: Bursts drew one move from before the range (31 Dec 2025 in YTD)
    at = app(frame="YTD")
    at.session_state["view"] = "Bursts"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    x = pd.to_datetime(_trace(_fig(at, "bursts_chart"), "S&P 500, real days")["x"])
    c = ex.saved_prices("S&P 500")
    this_year = c[c.index.year == c.index[-1].year]            # by hand: every close of 2026 is one move
    assert len(x) == len(this_year) and x.min() == this_year.index[0] and (x.year == 2026).all()


def _sig4(x):
    """x with 4 significant digits, written out without an exponent (numpy's own formatter)."""
    return np.format_float_positional(x, precision=4, unique=False, fractional=False, trim="k")


def _crypto(monkeypatch_app):
    idx = pd.date_range(end="2026-11-18", periods=1500, freq="D")       # open every day
    daily = pd.Series(2e-5 * np.exp(np.cumsum(np.random.default_rng(0).standard_normal(1500) * 0.03)), index=idx)
    t0 = pd.Timestamp("2026-11-14 00:00", tz="UTC")                       # a Saturday, around the clock
    bars = pd.Series(2e-5 * (1 + 0.01 * np.sin(np.arange(200) / 5)),
                     index=pd.DatetimeIndex([t0 + pd.Timedelta(minutes=30 * k) for k in range(200)]))
    return daily, bars


def test_workspace_crypto_is_labeled_and_its_prices_are_readable(app):
    # WS-21, WS-22: SHIB-USD read "Stock" and every price "0.00"
    daily, bars = _crypto(app)
    at = app(daily=daily, intraday=lambda t, p, i: bars, ticker="SHIB-USD")
    assert not at.exception, [e.value for e in at.exception]
    head = " ".join(m.value for m in at.markdown if "class='price'" in m.value)
    assert "Crypto · live from Yahoo" in head and "Stock" not in head
    want = _sig4(daily.iloc[-1])                       # 4 significant digits, not 0.00
    assert f"<span class='price'>{want}</span>" in head and want.strip("0.") != ""
    year = daily[daily.index > daily.index[-1] - pd.Timedelta(weeks=52)]     # 52 weeks = 364 dates here
    assert len(year) == 364
    assert _stat(at, "52-week high") == _sig4(year.max()) and _stat(at, "52-week low") == _sig4(year.min())


def test_workspace_crypto_7d_hides_no_nights_or_weekends(app):
    # WS-21: the 7D chart hid every weekend and night bar of a market open around the clock
    daily, bars = _crypto(app)
    at = app(daily=daily, intraday=lambda t, p, i: bars, ticker="SHIB-USD", frame="7D")
    assert not at.exception, [e.value for e in at.exception]
    fig = _fig(at, "price_chart")
    assert not fig["layout"].get("xaxis", {}).get("rangebreaks")
    assert ",.8f" in _trace(fig, "SHIB-USD")["hovertemplate"]


def test_workspace_stock_7d_still_hides_nights_and_weekends(app):
    daily = _daily_until("2026-11-18")
    bars = pd.Series(np.linspace(100, 105, 5), index=pd.DatetimeIndex(      # weekdays only (14 Nov is a Saturday)
        [f"2026-11-{d} 14:30" for d in (12, 13, 16, 17)] + ["2026-11-18 20:55"], tz="UTC"))
    at = app(daily=daily, intraday=lambda t, p, i: bars, ticker="ZZZE", frame="7D")
    assert not at.exception, [e.value for e in at.exception]
    breaks = _fig(at, "price_chart")["layout"]["xaxis"]["rangebreaks"]
    assert {"bounds": ["sat", "mon"]} in [{"bounds": b["bounds"]} for b in breaks]
    assert any(b.get("pattern") == "hour" for b in breaks)


def test_workspace_survives_yahoo_failing_after_a_live_ticker_is_open(app, monkeypatch):
    # WS-23: the cached prices of a typed ticker expire (24 h, or Clear cache) while Yahoo is down;
    # _load raised ConnectionError and a red exception replaced the whole page
    import streamlit as st
    at = app(daily=_daily_until("2026-11-18"), ticker="ZZQ")
    assert not at.exception, [e.value for e in at.exception]
    monkeypatch.setattr(ex, "_yahoo", _no_daily)
    st.cache_data.clear()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["asset"] == ("saved", "S&P 500")
    assert "Live prices for ZZQ aren't available" in at.session_state["ticker_error"]
    assert any("ZZQ" in t.value for t in at.toast)
    spx = pd.read_csv(ROOT / "data" / "sp500_daily.csv", index_col=0)["Close"]
    assert f"{spx.iloc[-1]:,.2f}" in " ".join(m.value for m in at.markdown)     # the saved S&P close
    at.run()                                       # next run: the sidebar shows the reason
    assert any("ZZQ" in e.value for e in at.error)


def test_workspace_plain_message_for_a_price_that_barely_moves(app):
    # WS-25: a pegged or halted asset showed "Need more than 10 nonzero returns, got 1"
    idx = pd.bdate_range(end="2026-11-18", periods=1500)
    at = app(daily=pd.Series([1.0] * 1400 + [2.0] * 100, index=idx))
    at.text_input(key="ticker_box").set_value("STEP").run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["asset"] == ("saved", "S&P 500")
    want = "STEP's price barely moves (only 1 day changed), so there are no tails to measure. Try a stock or an index."
    assert [e.value for e in at.error] == [want]


def test_workspace_dial_warns_about_an_earlier_run_off_the_fit(app):
    # WS-24: after a run with a2 = 0.20 and then one back at the fit, the a2 = 0.20 run still sat on
    # the a1-only sweep, and the caption's warning (which looked at the current run only) was gone
    at = app()
    warn_now, warn_earlier = "Your a2 or v1 differs from the fit", "Some earlier runs used an a2 or v1"
    _run_model(at, in_a1=1.0, in_a2=0.2, in_days=5000)
    assert at.session_state["log"][0][2]["herding"] is not None      # plotted on the dial
    assert warn_now in _captions(at)
    at.radio(key="preset").set_value("S&P 500 fit").run()
    _run_model(at)
    log = at.session_state["log"]
    assert len(log) == 2 and log[0][1]["a2"] == FIT["settings"]["a2"] and log[1][1]["a2"] == 0.2
    caps = _captions(at)
    assert warn_now not in caps and warn_earlier in caps
