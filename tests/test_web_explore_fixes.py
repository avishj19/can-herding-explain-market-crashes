"""Tests of the Test program page's fixes (web/page_explore.py) and the explore.py parts behind them:
the market picker, ticker messages, company names, Run model feedback, the run log at a1 = 0,
remembering a failed Yahoo fetch, hover text, log-scale labels and short daily ranges.

    python3 -m pytest tests/test_web_explore_fixes.py -q

The page tests run headless with Streamlit's AppTest and no network (the `app` and `open_page`
fixtures of test_web_pages.py, which fake Yahoo). The small helpers of the page are tested on
their own by loading the page's code without drawing it (_page_helpers).
"""
import math
import time

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT
from herding import explore as ex
from herding import market_data
from test_web_pages import (_captions, _daily_until, _fig, _model_table, _run_model,  # noqa: F401
                            app, open_page)


# ------------------------------------------------------------------ helpers
def _page_helpers():
    """The page's functions without drawing the page: its code minus the last line, page()."""
    src = (ROOT / "web" / "page_explore.py").read_text().rstrip()
    assert src.endswith("\npage()")
    ns = {"__name__": "page_explore_helpers"}
    exec(compile(src[:-len("page()")], "page_explore.py", "exec"), ns)
    return ns


H = _page_helpers()


def _group(at, key):
    for b in at.get("button_group"):
        if b.proto.id.endswith("-" + key):
            return b
    raise AssertionError(f"no button group {key!r}")


def _pills(at):
    """The market pills now on screen (their key changes with the market)."""
    found = [b for b in at.get("button_group") if "-pick_" in b.proto.id]
    assert len(found) == 1, [b.proto.id for b in found]
    return found[0]


def _h1(at):
    import re
    for m in at.markdown:
        hit = re.search(r"<h1[^>]*>([^<]*)</h1>", m.value)
        if hit:
            return hit.group(1)
    raise AssertionError("no title")


def _errors(at):
    return [e.value for e in at.error]


# ================================================================== pure helpers (no page)
def test_company_names_cover_exactly_the_saved_stocks():
    assert sorted(ex.COMPANY) == sorted(market_data.stock_tickers())


@pytest.mark.parametrize("name, want", [("DIS", "DIS · Disney"), ("RTX", "RTX (Raytheon)"),
                                        ("S&P 500", "S&P 500"), ("Gold", "Gold"), ("NVDA", "NVDA")])
def test_display_name(name, want):
    assert ex.display_name(name) == want


@pytest.mark.parametrize("name, want", [("S&P 500", "Indexes"), ("Dow Jones", "Indexes"), ("AAPL", "Stocks"),
                                        ("UNP", "Stocks"), ("Brent oil", "Commodities"), ("NVDA", None),
                                        ("^GSPC", None)])
def test_market_group(name, want):
    assert ex.market_group(name) == want


def _fake(n):
    return lambda t, s: pd.Series(np.linspace(10, 20, n), index=pd.bdate_range("2000-01-03", periods=n))


def test_message_unknown_ticker_says_what_to_do():
    with pytest.raises(ValueError) as e:
        ex.live_prices("xyz", download=lambda t, s: pd.Series(dtype=float))
    assert str(e.value) == ("Yahoo Finance has no prices for XYZ. Check the spelling (e.g. BRK-B), "
                            "or pick a market above.")


def test_message_short_history_uses_thousands_separators():
    with pytest.raises(ValueError) as e:
        ex.live_prices("XYZ", download=_fake(812))
    assert str(e.value) == ("XYZ has only 812 days of prices. The site needs more than 1,000 days (about 4 "
                            "years for a stock) to count rare big days reliably. Try an older company, or "
                            "pick a market above.")
    with pytest.raises(ValueError, match="only 1,000 days"):         # the boundary: 1,000 is not enough
        ex.live_prices("XYZ", download=_fake(1000))


def test_message_bad_characters():
    with pytest.raises(ValueError) as e:
        ex.clean_ticker("$$")
    assert str(e.value) == ("“$$” is not a ticker. Use only letters, digits and . - = ^ (up to 12), "
                            "like NVDA or BTC-USD.")


def test_message_network():
    def broken(t, s):
        raise OSError("blocked")
    with pytest.raises(ConnectionError) as e:
        ex.live_prices("XYZ", download=broken)
    assert str(e.value) == "Could not reach Yahoo Finance. Try again in a minute, or pick a market above."
    with pytest.raises(ConnectionError) as e:
        ex.intraday_prices("XYZ", "1D", download=lambda t, p, i: broken(t, p))
    assert str(e.value) == ex.YAHOO_DOWN


def test_message_not_a_price_names_the_sp500():
    with pytest.raises(ValueError) as e:
        ex.clean_ticker("^vix")
    assert "^GSPC (the S&P 500)" in str(e.value) and str(e.value).endswith("or pick a market above.")


def test_paper_measure_labels():
    assert ex.PAPER_MEASURES == {"alpha_reg": "Tail exponent (paper's method: largest 30% of days)",
                                 "H": "Hurst H of direction (0.5 = no memory)",
                                 "H_abs": "Hurst H of move size (above 0.5 = bursts of wild days)"}


@pytest.mark.parametrize("lo, hi, vals, labels", [
    (16.7, 6800.0, [100, 1000], None),                                      # S&P 500 since 1950
    (3.0, 120_000.0, [10, 100, 1000, 10_000, 100_000], ["10", "100", "1,000", "10,000", "100,000"]),
    (0.5, 30.0, [0.5, 1, 2, 5, 10, 20], ["0.5", "1", "2", "5", "10", "20"]),             # under 3 powers: 2 and 5 too
    (0.0002, 0.03, [0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02],
     ["0.0002", "0.0005", "0.001", "0.002", "0.005", "0.01", "0.02"]),
])
def test_log_ticks(lo, hi, vals, labels):
    got = H["_log_ticks"](lo, hi)
    if labels is None:                     # 100 and 1,000 only: 2 and 5 times them are added
        assert got[0] == [20, 50, 100, 200, 500, 1000, 2000, 5000]
        assert got[1] == ["20", "50", "100", "200", "500", "1,000", "2,000", "5,000"]
        return
    assert got[0] == pytest.approx(vals) and got[1] == labels


def test_log_ticks_inside_one_step_and_bad_input():
    assert H["_log_ticks"](100.0, 102.0) is None          # plotly's own labels are fine there
    for lo, hi in ((0, 10), (10, 1), (-1, 5), (1, math.inf)):
        with pytest.raises(ValueError):
            H["_log_ticks"](lo, hi)


@pytest.mark.parametrize("p, want", [(0.17, "17%"), (0.024, "2.4%"), (0.00031, "0.031%"), (0.1, "10%")])
def test_share_text(p, want):
    assert H["_share_text"](p) == want


def test_histogram_hover_by_hand():
    # density 0.425 per sd over a 0.4-wide bin = 17% of days; an empty bin says so
    out = H["_histogram_hover"]("S&P 500", np.array([-1.0, 0.0, 2.4]), np.array([0.425, 1.0, np.nan]), 0.4)
    assert out == ["S&P 500: moves near −1.0 sd on 17% of days",
                   "S&P 500: moves near 0.0 sd on 40% of days",
                   "S&P 500: no day near +2.4 sd"]


def test_gone_short():
    assert H["_gone_short"](7282) == "ran away (day 7,282)"
    assert H["_gone_short"](-745) == "ran away (warm-up)"
    assert H["_gone_short"](1) == "ran away (day 1)"


def test_key_items_only_name_lines_the_view_draws():
    k = H["_key_items"]
    assert k("Price", 1.5) == ()
    assert k("Bursts", 1.5) == ("real", "herding") and k("Bursts", 0.0) == ("real",)
    assert k("Tail", 1.5) == ("real", "bell", "fundamentalists", "chartists", "herding")
    assert k("Memory", 0.0) == ("real", "bell", "fundamentalists", "chartists")
    # a mix that ran away has no line, so the key leaves it out (it listed "Trend chasers, no herding")
    assert k("Tail", 1.5, gone=["chartists"]) == ("real", "bell", "fundamentalists", "herding")
    assert k("Bursts", 1.5, gone=["herding"]) == ("real",)
    assert k("Price", 1.5, gone=["herding"]) == ()


def test_range_note():
    n = H["_range_note"]
    assert n("1D", "saved", False, "x") == H["NO_INTRADAY_NOTE"]
    assert n("7D", "saved", True, "x") == "Yahoo Finance is not answering, so this shows the saved daily closes."
    assert n("1D", "live", True, "x") == "x"              # live: explore's own note (or none)
    assert n("1Y", "saved", True, "") == ""


# ================================================================== the page
@pytest.mark.slow
def test_retyping_the_same_ticker_opens_it_again(app):
    # the box kept "NVDA" after a market was picked, so typing NVDA again changed nothing
    at = app(daily=_daily_until("2026-11-18"))
    at.text_input(key="ticker_box").set_value("NVDA").run()
    assert at.session_state["asset"] == ("live", "NVDA")
    assert at.text_input(key="ticker_box").value == ""          # emptied after a success
    _pills(at).set_value("Dow Jones").run()
    assert at.session_state["asset"] == ("saved", "Dow Jones")
    at.text_input(key="ticker_box").set_value("NVDA").run()
    assert at.session_state["asset"] == ("live", "NVDA") and _h1(at) == "NVDA"


@pytest.mark.slow
def test_a_bad_ticker_stays_in_the_box_to_be_corrected(app):
    at = app()
    at.text_input(key="ticker_box").set_value("!!!").run()
    assert at.text_input(key="ticker_box").value == "!!!" and _errors(at)
    assert at.text_input(key="ticker_box").proto.max_chars == 12


@pytest.mark.slow
def test_the_tab_follows_the_market_on_screen(app):
    at = app(daily=_daily_until("2026-11-18"))
    assert _group(at, "group").value == "Indexes"
    at.text_input(key="ticker_box").set_value("dis").run()           # a saved stock: Stocks tab, DIS lit
    assert _group(at, "group").value == "Stocks" and _pills(at).value == "DIS"
    at.text_input(key="ticker_box").set_value("^GSPC").run()         # a saved index's symbol
    assert _group(at, "group").value == "Indexes" and _pills(at).value == "S&P 500"
    at.text_input(key="ticker_box").set_value("GC=F").run()
    assert _group(at, "group").value == "Commodities" and _pills(at).value == "Gold"
    at.text_input(key="ticker_box").set_value("NVDA").run()          # live: the tab stays, nothing lit
    assert _group(at, "group").value == "Commodities" and _pills(at).value is None


@pytest.mark.slow
def test_tab_goes_back_to_indexes_after_a_live_ticker_falls_back(app, monkeypatch):
    import streamlit as st
    at = app(daily=_daily_until("2026-11-18"))
    _group(at, "group").set_value("Stocks").run()
    at.text_input(key="ticker_box").set_value("NVDA").run()
    monkeypatch.setattr(ex, "_yahoo", lambda t, s: (_ for _ in ()).throw(OSError("down")))
    st.cache_data.clear()
    at.run()                                                         # falls back to the S&P 500
    assert at.session_state["asset"] == ("saved", "S&P 500")
    at.run()
    assert _group(at, "group").value == "Indexes" and _pills(at).value == "S&P 500"
    assert at.text_input(key="ticker_box").value == ""
    assert any("NVDA" in e for e in _errors(at))


@pytest.mark.slow
def test_clicking_the_lit_market_clears_an_old_error(app):
    at = app()
    at.text_input(key="ticker_box").set_value("^VIX").run()
    assert _errors(at)
    _pills(at).set_value(None).run()                    # clicking the lit S&P 500 again unselects it...
    assert not _errors(at) and at.session_state["asset"] == ("saved", "S&P 500")
    assert _pills(at).value == "S&P 500"                # ...and it is lit again
    assert at.text_input(key="ticker_box").value == ""


@pytest.mark.slow
def test_no_widget_gets_both_a_default_and_a_session_state_value(app, monkeypatch):
    # Streamlit only checks this with a real server running; pretend one is, and catch its warning
    from streamlit.elements.lib import policies
    warned = []
    from types import SimpleNamespace          # only this check sees a "server"; caching still doesn't
    monkeypatch.setattr(policies, "runtime", SimpleNamespace(exists=lambda: True))
    monkeypatch.setattr(policies, "_shown_default_value_warning", False)
    monkeypatch.setattr(policies._LOGGER, "warning", lambda msg, *a, **k: warned.append(msg % a))
    at = app()
    at.text_input(key="ticker_box").set_value("aapl").run()          # sets the tab and the pills
    _pills(at).set_value(None).run()                                 # puts the lit pill back
    _pills(at).set_value("MSFT").run()
    assert not at.exception
    assert not [w for w in warned if "group" in w or "pick_" in w], warned


@pytest.mark.slow
def test_company_names_in_title_and_pills(app):
    at = app()
    at.text_input(key="ticker_box").set_value("DIS").run()
    assert _h1(at) == "DIS · Disney"
    labels = [o.content for o in _pills(at).proto.options]
    assert labels == [ex.display_name(t) for t in market_data.stock_tickers()]


@pytest.mark.slow
def test_futures_symbol_of_a_spot_series_says_so(app):
    at = app()
    at.text_input(key="ticker_box").set_value("BZ=F").run()
    assert at.session_state["asset"] == ("saved", "Brent oil")
    assert any("showing the saved spot price" in m.value for m in at.markdown)
    _pills(at).set_value("Gold").run()
    assert not any("spot price" in m.value for m in at.markdown)


@pytest.mark.slow
def test_run_model_says_what_ran_and_when_settings_changed(app):
    import streamlit as st
    at = app()
    assert not at.success                                   # nothing run yet
    _run_model(at, in_a1=1.52, in_days=5000)
    p, burn = ex.fitted_settings()
    mixes = ex.run_mixes(p, burn, 1.52, float(p["a2"]), float(p["v1"]), 5000, int(at.session_state["run"]["seed"]))
    tail = ex.summary(mixes["herding"][0])["tail"]
    spx = ex.summary(ex.log_returns(ex.saved_prices("S&P 500")).values)["tail"]
    assert [s.value for s in at.success] == [f"Ran at herding strength 1.52: tail exponent {tail:.2f} "
                                             f"(S&P 500: {spx:.2f})"]
    at.toggle(key="auto_run").set_value(False).run()      # with Run automatically on, it would just run
    at.slider(key="in_a1").set_value(1.1).run()
    assert not at.success
    assert any("Settings changed: press Run model to see them." in m.value for m in at.markdown)
    del st


@pytest.mark.slow
def test_labels_in_plain_words(app):
    at = app()
    labels = [s.label for s in at.slider]
    # the wording guide's names (docs/redesign/WORDING.md): a2 and v1 are said in plain words
    assert labels == ["Herding strength (a1)", "Reaction to price trends (a2)",
                      "How often trend chasers change mood (v1)",
                      "Party effect on true value (extra study; % a year, Democrats − Republicans)"]
    assert at.slider(key="in_party").value == 0.0          # off by default: the paper's model has no politics
    days = at.selectbox(key="in_days")
    # short, so it fits the narrow panel ("19,329 (as long as the S&P 500 record)" was cut off);
    # the full explanation is in the help
    assert days.format_func(19329) == "19,329 (S&P 500 length)" and days.format_func(5000) == "5,000"
    assert "as long as the S&P 500's record since 1950" in days.help
    assert any(m.value == "**Tail exponent vs herding strength**" for m in at.markdown)


@pytest.mark.slow
def test_run_log_at_a1_zero_has_no_separate_herding_number(app):
    at = app()
    _run_model(at, in_a1=0.0, in_a2=0.0, in_v1=0.5, in_days=5000, in_seed=0)
    log = [d.value for d in at.dataframe if "Ran away" in d.value.columns][0]
    assert np.isnan(log.loc[0, "Trend chasers who herd"])
    assert "Trend chasers who herd" not in log.loc[0, "Ran away"]
    # the numbers underneath stay numbers (NaN shown as a dash)
    assert log["Trend chasers, no herding"].dtype == float


@pytest.mark.slow
def test_model_table_cells_are_short(app):
    at = app()
    _run_model(at, in_a1=0.0, in_a2=1.0, in_v1=5.0, in_days=5000, in_seed=1)   # chartists run away in warm-up
    t = _model_table(at)
    assert "Bell curve" in t.columns
    assert t.loc["Days", "Trend chasers, no herding"] == "ran away (warm-up)"


@pytest.mark.slow
def test_a_failed_live_fetch_is_remembered_for_five_minutes(app, monkeypatch):
    calls = []

    def down(t, s):
        calls.append(t)
        raise OSError("down")
    at = app()
    monkeypatch.setattr(ex, "_yahoo", down)
    at.session_state["asset"] = ("saved", "Gold")
    [t for t in at.toggle if t.label == "Live prices"][0].set_value(True)
    at.session_state["frame"] = "1D"
    at.run()
    assert calls == ["GC=F"]
    assert "Yahoo Finance is not answering, so this shows the saved daily closes." in _captions(at)
    assert "need Live prices on" not in _captions(at)
    at.run()
    at.run()
    assert calls == ["GC=F"]                                # not asked again
    failed = dict(at.session_state["live_failed"])
    failed[("daily", "GC=F")] = time.time() - 301           # five minutes later
    at.session_state["live_failed"] = failed
    at.run()
    assert calls == ["GC=F", "GC=F"]


@pytest.mark.slow
def test_short_daily_range_has_dates_not_hours(app):
    at = app(frame="1D")
    fig = _fig(at, "price_chart")
    assert fig["layout"]["xaxis"]["tickformat"] == "%b %d"
    assert fig["data"][0]["mode"] == "lines+markers"
    lo, hi = fig["layout"]["yaxis2"]["range"]
    assert -0.2 < lo < 0 and 0 <= hi < 0.01                   # not -100% to +100%


@pytest.mark.slow
def test_price_view_colors_and_log_labels(app):
    from web import common as c
    at = app()
    fig = _fig(at, "price_chart")
    assert fig["data"][0]["line"]["color"] == c.pal()["REAL"]          # not up/down without an arrow
    assert fig["layout"]["hovermode"] == "x unified"
    assert fig["layout"]["yaxis"]["ticktext"][:4] == ["20", "50", "100", "200"]
    assert not any("class='key'" in m.value for m in at.markdown)       # the market only: no key
    import re
    dd = [m.value for m in at.markdown if "Max drawdown" in m.value][0]
    assert not re.search(r"Max drawdown</div><div class='v' style", dd)


@pytest.mark.slow
@pytest.mark.parametrize("view, word", [("Histogram", "moves near"), ("Memory", "days apart: correlation"),
                                        ("Tail", "sd or more on")])
def test_shape_views_read_one_line_in_plain_words(app, view, word):
    at = app()
    tab = "Tail" if view in ("Tail", "Histogram") else "Bursts"    # the chart tab the view is on
    at.session_state["view"] = tab
    at.session_state[H["SUBVIEWS"][tab][0]] = view
    at.run()
    fig = _fig(at, f"model_chart_{view}")
    assert fig["layout"]["hovermode"] == "closest"
    hover = " ".join(str(t.get("hovertemplate", "")) + " ".join(map(str, t.get("text") or [])) for t in fig["data"])
    assert word in hover
    assert any("class='key'" in m.value for m in at.markdown)


@pytest.mark.slow
def test_model_panel_comes_before_the_tables(app):
    # on a phone Streamlit stacks the page in order: the model (its herding slider) must come before
    # the tabs under the chart; and with Run automatically on there is no Run model button at all
    at = app()
    order = []

    def walk(node):
        typ = getattr(node, "type", "")
        if typ == "slider" and getattr(node, "key", "") == "in_a1":
            order.append("model")
        if typ == "tab":
            order.append("tab")
        for ch in getattr(node, "children", {}).values():
            walk(ch)
    walk(at.main)
    assert "model" in order and "tab" in order and order.index("model") < order.index("tab")
    assert not [b for b in at.button if b.label == "Run model"]
    at.toggle(key="auto_run").set_value(False).run()        # Run automatically off: the button appears
    assert [b for b in at.main.button if b.label == "Run model"]


@pytest.mark.slow
def test_dial_caption_explains_the_marks(open_page):
    at = open_page("web/page_explore.py")
    text = _captions(at)
    assert "Solid dot: the history shown now; hollow circles: your earlier histories" in text
    assert "flat line marked \"this market\" is S&P 500's own tail exponent" in text
    assert "Fig 10" in text


# ------------------------------------------------------------------ the president's party (extension)
def test_party_switch_shades_the_chart_and_shows_the_luck_test(app):
    at = app()                                              # the S&P 500, saved data
    at.toggle(key="t_party").set_value(True).run()
    assert not at.exception, [e.value for e in at.exception]
    v = ex.party_view(ex.log_returns(ex.saved_prices("S&P 500")))
    page_text = " ".join(m.value for m in at.markdown)
    assert f"{v['gap_year']:+.1%}" in page_text and f"beats {v['luck']['mean']:.0f}% of {v['calendars']}" in page_text
    shapes = _fig(at, "price_chart")["layout"]["shapes"]
    dem_terms = [t for t in ex.party_terms(ex.saved_prices("S&P 500").index) if t["party"] == "D"]
    assert len([s for s in shapes if s.get("fillcolor") == "rgba(128,128,128,0.18)"]) >= len(dem_terms)
    assert "it has no politics, so that gap is pure luck" in _captions(at)
    at.toggle(key="t_party").set_value(False).run()
    assert not any(s.get("fillcolor") == "rgba(128,128,128,0.18)"
                   for s in _fig(at, "price_chart")["layout"].get("shapes", []))


def test_party_drift_runs_the_model_with_it_and_logs_it(app):
    at = app()
    _run_model(at, in_party=5.0, in_days=5000)
    assert at.session_state["run"]["party"] == 5.0
    assert any("with a party effect of +5.0% a year" in s.value for s in at.success)
    log = [d.value for d in at.dataframe if "Ran away" in d.value.columns][0]
    assert log.loc[0, "Party effect (% a year)"] == 5.0
    at.toggle(key="t_party").set_value(True).run()
    assert "with your party effect of +5.0%" in _captions(at)


def test_presets_switch_the_party_drift_off(app):
    at = app()
    at.slider(key="in_party").set_value(3.0).run()
    [b for b in at.get("button_group") if b.proto.id.endswith("-preset")][0].set_value("Best fit").run()
    assert at.slider(key="in_party").value == 0.0


def test_findings_party_section_matches_the_backend(open_page):
    at = open_page("web/page_checks.py")
    assert not at.exception, [e.value for e in at.exception]
    text = " ".join(m.value for m in at.markdown)
    dow = ex.log_returns(ex.saved_prices("Dow Jones"))
    v = ex.party_view(dow)
    assert f"The Dow did {v['gap_year']:+.1%} a year better under Democrats since 1896" in text
    assert f"the Dow's gap is {ex.party_gap_without(dow, 'Hoover'):+.1%} a year" in text
    html = " ".join(h.proto.body for h in at.get("html"))
    assert "Nasdaq-100" in html and f"{v['luck']['mean']:.0f}% of {v['calendars']} fake calendars" in html


# Spec: with Run automatically on (the default), moving a slider runs the model at once: the run
# shown and the top of the run log are the new settings, and no "Settings changed" note appears.
@pytest.mark.slow
def test_run_automatically_runs_when_a_slider_moves(app):
    at = app()
    assert at.toggle(key="auto_run").value is True
    at.slider(key="in_a1").set_value(1.3).run()
    assert at.session_state["run"]["a1"] == 1.3
    assert at.session_state["log"][0][1]["a1"] == 1.3
    assert not any("Settings changed" in m.value for m in at.markdown)
    assert any(s.value.startswith("Ran at herding strength 1.30") for s in at.success)


# Spec: under the price chart, picking a big day tells what happened; for one of the Dow's 20 biggest
# days the documented cause and its source come with it.
@pytest.mark.slow
def test_big_day_reader_tells_the_cause(app):
    at = app()
    at.session_state["asset"] = ("saved", "Dow Jones")
    at.run()
    box = [s for s in at.selectbox if s.label == "Read about a big day"][0]
    assert box.value is None and not at.info
    assert any(o.startswith("19 Oct 1987  ▼ 22.6%") for o in box.options)     # the label shown
    box.select(pd.Timestamp("1987-10-19")).run()
    info = " ".join(i.value for i in at.info)
    assert info.startswith("19 October 1987: fell") and "Black Monday" in info
    assert "(https://www.federalreservehistory.org/essays/stock-market-crash-of-1987)" in info
    assert not at.exception


# ------------------------------------------------------------------ the redesigned layout (Nov 2026)
# Spec (docs/redesign/README.md item 5): four chart tabs, four tabs below, presets in the model panel,
# every other setting in the sidebar, no Run model button while runs are automatic.
def test_four_chart_tabs_and_their_switches(app):
    at = app()
    view = _group(at, "view")
    assert list(view.options) == ["Price", "Watch a crash", "How rare big days are", "Bursts of wild days"]
    for tab, (key, subs) in H["SUBVIEWS"].items():
        view.set_value(tab).run()
        assert not at.exception, [e.value for e in at.exception]
        assert list(_group(at, key).options) == subs
        for sub in subs:
            _group(at, key).set_value(sub).run()
            assert not at.exception, [e.value for e in at.exception]
            assert any(pc.proto.id.endswith(f"-model_chart_{sub}") or pc.proto.id.endswith("-bursts_chart")
                       for pc in at.get("plotly_chart"))
    view.set_value("Model market").run()
    assert not at.exception and _group(at, "mm_mix") and _group(at, "mm_zoom")


def test_four_tabs_below_the_chart(app):
    at = app()
    assert [t.label for t in at.tabs] == ["Model and S&P 500", "Big days", "History log",
                                         "Every history on one chart"]
    big = at.tabs[1]
    assert [s for s in big.selectbox if s.label == "Read about a big day"]      # the reader is in Big days
    assert any(pc.proto.id.endswith("-dial") for pc in at.tabs[3].get("plotly_chart"))


@pytest.mark.parametrize("preset", ["Wilder", "Best fit", "Calmer", "No herding"])
def test_each_preset_sets_the_backend_values_and_renders(app, preset):
    at = app()
    _group(at, "preset").set_value(preset).run()
    assert not at.exception, [e.value for e in at.exception]
    p, _ = ex.fitted_settings()
    a1 = H["PRESETS"][preset]
    want = float(p["a1"]) if a1 is None else a1
    assert at.slider(key="in_a1").value == pytest.approx(want, abs=1e-12)
    assert at.slider(key="in_a2").value == pytest.approx(float(p["a2"]), abs=1e-12)
    assert at.slider(key="in_v1").value == pytest.approx(float(p["v1"]), abs=1e-12)
    assert at.slider(key="in_party").value == 0.0
    assert at.session_state["run"]["a1"] == pytest.approx(want, abs=1e-12)     # ran automatically
    assert _group(at, "preset").value == preset


def test_settings_live_in_the_sidebar(app):
    at = app()
    side = at.sidebar
    assert {s.key for s in side.slider} == {"in_a2", "in_v1", "in_party"}
    assert [s.key for s in side.selectbox] == ["in_days"]
    assert [n.key for n in side.number_input] == ["in_seed"]
    assert {t.key for t in side.toggle} >= {"live", "t_big", "t_dd", "t_log", "t_party", "auto_run"}
    assert [t.key for t in side.text_input] == ["ticker_box"]
    assert any(b.proto.id.endswith("-group") for b in side.get("button_group"))
    assert [b.label for b in side.button] == ["Reset everything"]
    assert side.code                                        # the share link
    assert [s.key for s in at.main.slider] == ["in_a1"]     # only herding strength stays on the page
    assert not [b for b in at.button if b.label == "Run model"]


def test_share_link_names_the_settings():
    import streamlit as st
    ss = st.session_state
    ss.clear()
    run = dict(H["c"].default_run(), a1=1.5)
    ss.asset, ss.run = ("saved", "Gold"), run
    assert H["_share_link"]("Gold") == "?" + __import__("urllib.parse").parse.urlencode(
        ex.write_link("Gold", run, "S&P 500", H["c"].default_run()))
    ss.asset = ("live", "NVDA")
    assert H["_share_link"]("NVDA") == ""                  # a typed ticker never goes in a link
    ss.clear()
