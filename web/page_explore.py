"""Test program (was "Workspace"): one market next to the model (redesign of Nov 2026, docs/redesign/).

top           the three tasks (a slim card with a progress bar), then the market's quote bar: name, last
              price, day change and where the prices come from
left column   four chart tabs (a fragment: changing a tab, a switch or the range reruns only the chart):
              Price (range buttons, drawdown), Watch a crash (trader-mix and zoom pickers), How rare big
              days are (Tail / Histogram), Bursts of wild days (Bursts / Memory); a "Try this" line and a
              one-line insight under each
right column  the model: presets (Wilder / Best fit / Calmer / No herding), the herding-strength slider,
              what ran, and the scorecard. Runs are automatic (Run model appears only when Run
              automatically is off, in Settings). On a phone the columns stack, so the model comes after
              the chart
below         four tabs: Model and the market, Big days (with the reader), History log, Every history on
              one chart
sidebar       Settings: other markets, live prices, chart switches, more model settings (a2, v1, days,
              seed), run options, the share link and Reset everything
"""
import hashlib
import time
from datetime import datetime
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from herding import explore as ex
from herding.market_data import ROOT
from web import common as c

# the four chart tabs; their keys are the old view names, so links and tests that set "view" still work
VIEWS = ["Price", "Model market", "Tail", "Bursts"]
VIEW_LABELS = {"Price": "Price", "Model market": "Watch a crash", "Tail": "How rare big days are",
               "Bursts": "Bursts of wild days"}
# a tab with two charts: (the switch's key, its two charts)
SUBVIEWS = {"Tail": ("rare_view", ["Tail", "Histogram"]), "Bursts": ("bursts_view", ["Bursts", "Memory"])}
LEAVES = ["Price", "Model market", "Tail", "Histogram", "Bursts", "Memory"]     # every chart the page draws
# one sentence under each chart for first-time visitors: what it shows and what to look for
HINTS = {
    "Price": "The price over the range you picked. Triangles mark big days: days that moved more than 4 "
             "standard deviations (sd; the usual size of a day's move). A bell curve says these should almost "
             "never happen. The lower panel shows how far the price has fallen from its last high.",
    "Model market": "The simulated market itself, day by day. Top: its price and the true value it is pulled "
                    "back to (triangles: big days, beyond 4 sd). Middle: the share of traders who are trend "
                    "chasers. Bottom: optimists minus pessimists, as a share of all traders "
                    "(above 0, the crowd is buying). Look for crashes after the trend chasers' share climbs and "
                    "the crowd leans hard one way. Biggest crash and Wildest year jump to them. You can also "
                    "drag across the chart to zoom in, and double-click to zoom out.",
    "Tail": "How often big days happen. Read left to right: the higher a line stays, the more often "
            "big days come. The dotted bell curve drops away fast. Look for the line that follows this market best.",
    "Histogram": "How common each size of daily move is. Compared with the dotted bell curve, markets with "
                 "fat tails have more quiet days in the middle and more big days at the edges.",
    "Bursts": "Each day's move, in sd. Notice that wild days come in bursts, both in "
              "the market (top) and in the model with herding (bottom, when its history did not run away).",
    "Memory": "Does a big move today mean big moves later? Above 0 means yes. Real markets stay above 0 for "
              "months. A bell curve drops to 0 at once.",
}
# one thing to do on each chart, then what to look for; {name} is the market on screen
TRY = {
    "Price": "Click **All** and count the triangles. Each one is a day that moved more than 4 sd, "
             "which a bell curve says should almost never happen.",
    "Model market": "Press **Biggest crash**. Look just before the fall: trend chasers pile in, most turn "
                    "optimist, then they flip together.",
    "Tail": "Find 5 on the bottom axis. The dotted bell curve has already dropped off the chart; {name} "
            "is still up there. That gap is what this project is about.",
    "Histogram": "Look past ±4 at the edges. The bell curve is flat at zero there; {name}'s bars are not.",
    "Bursts": "Look at the tall spikes. Do they come alone, or in bursts? Then check the model below.",
    "Memory": "Read both lines at 20 days, a month apart. Above 0, a wild day still predicts wild days a month "
              "later. Where is the bell curve?",
}
# In the chart caches' keys: Streamlit keys a cache by the cached function's own code, so after an edit
# to the code that draws a chart (but not to the cached function) it kept showing the old chart until
# the server restarted (Nov 2026). This file's own text, hashed, changes with any edit to the page.
PAGE_VERSION = hashlib.sha256((ROOT / "web" / "page_explore.py").read_bytes()).hexdigest()[:12]
# the presets, in the order shown: herding strength a1 (None: the S&P 500 fit). The values are the
# saved sweep's strengths (numbers.json), so each preset's note reads straight from it
PRESETS = {"Wilder": 0.6, "Best fit": None, "Calmer": 1.6, "No herding": 0.0}
SAVED_BY_SYMBOL = {v: k for k, v in ex.YAHOO.items()}      # ^GSPC -> S&P 500, GC=F -> Gold, ...
NO_INTRADAY_NOTE = "Prices within the day need Live prices on, so this shows the last daily closes."
YAHOO_DOWN_NOTE = "Yahoo Finance is not answering, so this shows the saved daily closes."
FAIL_MEMORY = 300        # seconds a failed Yahoo fetch is remembered, so every rerun doesn't ask again


def _init():
    ss = st.session_state
    ss.setdefault("asset", ("saved", "S&P 500"))
    ss.setdefault("ticker_error", "")
    ss.setdefault("ticker_error_for", None)        # the market on screen when the error was made
    ss.setdefault("run", c.default_run())
    _read_link()
    ss.setdefault("log", [])
    ss.setdefault("auto_run", True)                 # run the model as soon as a control changes
    # the tab follows the market on screen: a saved market's own tab; a live ticker keeps the tab open
    ss.setdefault("group", ex.market_group(ss.asset[1]) or "Indexes")
    # _load fell back to the S&P 500 after a run had drawn the ticker box: clear it now, before the
    # box is drawn (a widget's value can't be changed after it is drawn in the same run)
    if ss.pop("_fallback_done", False):
        ss.ticker_box = ""
        ss.group = "Indexes"
    if "party" in ss.run:     # a session opened before the party drift was removed (Nov 2026)
        ss.run = {k: v for k, v in ss.run.items() if k != "party"}
        ss.log = [(t, {k: v for k, v in prm.items() if k != "party"}, tl) for t, prm, tl in ss.log]
    for k in ("a1", "a2", "v1", "days", "seed"):
        ss.setdefault(f"in_{k}", ss.run[k])
    # the preset picked is the one the controls match now, or none: a radio that kept showing
    # "S&P 500 fit" after a slider moved could not be clicked again to restore the fit
    ss.preset = _matching_preset()


def _saved_markets():
    return [n for names in ex.catalog().values() for n in names]


def _day_choices():
    return [5000, 10000, c.default_run()["days"]]


def _read_link():
    """Once per visit: the market and model settings in the page's address (a shared link), checked
    by explore.read_link (anything out of range or unknown is ignored, never an error)."""
    ss = st.session_state
    if ss.get("_link_read"):
        return
    ss._link_read = True
    got = ex.read_link(st.query_params, _day_choices(), _saved_markets())
    if "market" in got:
        ss.asset = ("saved", got.pop("market"))
        ss.group = ex.market_group(ss.asset[1])
    if got:
        ss.run = dict(ss.run, **got)
        for k, v in got.items():
            ss[f"in_{k}"] = v


def _write_link(name):
    """Keep the page's address in step with what it shows, so copying it shares this exact run (only
    for saved markets: a link never makes the server fetch a ticker)."""
    ss = st.session_state
    want = ex.write_link(name, ss.run, "S&P 500", c.default_run()) if ss.asset == ("saved", name) else {}
    have = {k: st.query_params.get(k) for k in ex.LINK_KEYS if k in st.query_params}
    if have != want:
        for k in ex.LINK_KEYS:
            if k in st.query_params and k not in want:
                del st.query_params[k]
        st.query_params.update(want)


def _matching_preset():
    ss = st.session_state
    p, _ = c.fitted()
    for name, a1 in PRESETS.items():
        want = {"in_a1": float(p["a1"]) if a1 is None else a1, "in_a2": float(p["a2"]), "in_v1": float(p["v1"])}
        if all(abs(float(ss[k]) - v) < 1e-9 for k, v in want.items()):
            return name
    return None


def _pick_market(key):
    """A market button was clicked. Clicking the picked one again would unpick it: keep it picked.
    Either way the visitor chose a market, so an old ticker error and the typed text are cleared
    (left in the box, typing the same ticker again changed nothing, so nothing happened)."""
    ss = st.session_state
    v = ss[key]
    if v is None:
        ss[key] = ss.asset[1]
    else:
        ss.asset = ("saved", v)
    ss.ticker_error = ""
    ss.ticker_box = ""
    ss.typed_symbol = None


def _fetch():
    """The ticker box: a saved market's ticker or Yahoo symbol opens the saved market (and its tab);
    anything else is fetched live from Yahoo and keeps the tab open. On success the box is emptied,
    so typing the same ticker again later works; on an error the text stays, to be corrected."""
    ss = st.session_state
    if not ss.ticker_box.strip():                   # emptying the box is not an error: keep the market
        ss.ticker_error = ""
        return
    if not _typing_allowed():
        ss.ticker_error = TOO_FAST
        ss.ticker_error_for = ss.asset
        return
    try:
        t = ex.clean_ticker(ss.ticker_box)
        if t in ex.market_data.stock_tickers():
            ss.asset = ("saved", t)
        elif t in SAVED_BY_SYMBOL:                  # ^GSPC, GC=F, ...: the saved market with that symbol
            ss.asset = ("saved", SAVED_BY_SYMBOL[t])
        else:
            c.live_asset(t)
            ss.asset = ("live", t)
        ss.ticker_error = ""
        ss.ticker_box = ""
        ss.typed_symbol = t
        if ss.asset[0] == "saved":
            ss.group = ex.market_group(ss.asset[1])
        _forget_failure(("daily", t))
    except ConnectionError as e:
        ss.ticker_error = c.BUSY if str(e) == c.BUSY else ex.YAHOO_DOWN
    except ValueError as e:
        msg = str(e)                                # (str.capitalize would lower-case the ticker)
        # the message can repeat what was typed: shown as plain text, never as markdown (a typed
        # [link](...) or ![image](...) must not turn into one)
        ss.ticker_error = _plain(msg[:1].upper() + msg[1:])
    ss.ticker_error_for = ss.asset


TYPED_PER_MINUTE = 8      # tickers one visitor may look up per minute (each can mean a download)
TOO_FAST = "That is a lot of stock symbols in a minute. Wait a moment, then try again."


def _typing_allowed(now=None):
    """True (and counted) if this visitor has looked up fewer than TYPED_PER_MINUTE tickers in the
    last minute: a script typing tickers into the box can't make the server download without end."""
    now = time.time() if now is None else now
    ss = st.session_state
    recent = [t for t in ss.get("_typed_at", []) if now - t < 60]
    if len(recent) >= TYPED_PER_MINUTE:
        ss._typed_at = recent
        return False
    ss._typed_at = recent + [now]
    return True


def _plain(text):
    """Text that Streamlit's markdown shows as written: the characters that could make a link, an
    image, HTML or formatting ([ ] ! < > * _ ~ ` and \\) are escaped; everything else, like the
    usual message's parentheses and full stops, stays as it is."""
    return "".join("\\" + ch if ch in "\\`*_[]!<>~" else ch for ch in str(text))


def _failed_recently(key, now=None):
    """True if fetching `key` from Yahoo failed less than FAIL_MEMORY seconds ago (this visitor)."""
    t = st.session_state.get("live_failed", {}).get(key)
    return t is not None and (time.time() if now is None else now) - t < FAIL_MEMORY


def _remember_failure(key):
    st.session_state.setdefault("live_failed", {})[key] = time.time()


def _forget_failure(key):
    st.session_state.get("live_failed", {}).pop(key, None)


def _preset():
    """A preset button was clicked: fill in the model controls (the run follows, as for a slider).
    Clicking the lit preset again unselects it; nothing changes then (_init lights it again)."""
    if st.session_state.preset not in PRESETS:
        return
    p, _ = c.fitted()
    a1 = PRESETS[st.session_state.preset]
    st.session_state.in_a1 = float(p["a1"]) if a1 is None else a1
    st.session_state.in_a2, st.session_state.in_v1 = float(p["a2"]), float(p["v1"])


GROUPS = {"Indexes": "A whole market in one number: the S&P 500 is 500 large US companies "
                      "(an index fund buys all of them).",
          "Stocks": "Single companies: 5 large US stocks from five different industries that rise and fall "
                    "about 1% when the market moves 1% (market beta 0.9 to 1.1).",
          "Commodities": "Raw materials traded worldwide: oil, gold and copper. (Brent oil's saved and live "
                         "prices come from two sources, spot and futures, and can differ: usually by 1 to 2%, "
                         "but by up to 25% in 2026.)"}


def _picker():
    """The market picker (in Settings): a tab per kind of market, a button per market, and a box
    for any ticker."""
    ss = st.session_state
    _, name = ss.asset
    with st.container(key="picker"):
        # no default: the tab is seeded in session state (_init) and set by _fetch, and a widget
        # given both a default and a session-state value makes Streamlit warn
        group = c.pick_one("Pick a market", list(GROUPS), None, "group",
                           format_func=lambda g: f"{g} ({len(ex.catalog()[g])})")
        st.markdown(f"<p class='muted' style='margin:-.3rem 0 .4rem'>{GROUPS[group]}</p>",
                    unsafe_allow_html=True)
        names = ex.catalog()[group]
        key = f"pick_{group}_{name}"            # a new key when the market changes: the right button lit
        if key not in ss:                       # seeded, not a default: same reason as the tab
            ss[key] = name if name in names else None
        st.pills("Markets", names, key=key, format_func=ex.display_name,
                 on_change=_pick_market, args=(key,), label_visibility="collapsed")
        st.text_input("Or type any stock symbol", key="ticker_box", placeholder="e.g. NVDA, then Enter",
                      max_chars=12, on_change=_fetch, help="Any symbol Yahoo Finance knows: a stock, a fund like SPY, "
                                             "or crypto like BTC-USD. Needs the internet.\n\n"
                                             "Will not work: companies listed for 4 years or less; prices "
                                             "that barely move (stablecoins, pegged currencies); "
                                             "measurements rather than prices (volatility indexes like "
                                             "^VIX, interest rates like ^TNX); unknown or delisted "
                                             "stock symbols.")
        # shown only while the same market is on screen: picking another one makes the error stale
        if ss.ticker_error and ss.ticker_error_for == ss.asset:
            st.error(ss.ticker_error)


def _preset_note(name):
    """What a preset will show, in a few words, from the saved herding-strength sweep (numbers.json):
    with too little herding the price runs away, more herding calms it, and the S&P 500's tails are at the edge."""
    p, _ = c.fitted()
    a1 = PRESETS[name]
    w = ex.sweep_at(c.numbers(), float(p["a1"]) if a1 is None else a1)
    if w["tail"] is None:
        return f"herding strength {w['a1']:.2f}: runs away in all {w['n']} saved histories"
    if w["ran_away"] > w["n"] / 2:
        return f"herding strength {w['a1']:.2f}: runs away in {w['ran_away']} of {w['n']}; wild when it does not"
    calm = " (calmer: thin tails)" if w["tail"] > 4 else " (like the S&P 500)" if a1 is None else ""
    return f"herding strength {w['a1']:.2f}: tail exponent about {w['tail']:.1f}{calm}"


def _reset_all():
    """Back to where the page starts: the S&P 500, the fitted settings, the first chart (PhET: a big,
    safe reset invites trying the extremes). The history log and the tasks done stay."""
    ss = st.session_state
    run = c.default_run()
    for k in ("a1", "a2", "v1", "days", "seed"):
        ss[f"in_{k}"] = run[k]
    ss.asset = ("saved", "S&P 500")
    ss.group = "Indexes"
    ss.ticker_box = ""
    for k in ("view", "mm_zoom", "mm_mix", "rare_view", "bursts_view"):
        ss.pop(k, None)


def _settings_head():
    """The top of the sidebar (Settings): the market and where its prices come from. Drawn before the
    prices are loaded, since Live prices decides which prices those are. Returns Live prices' value."""
    with st.sidebar:
        st.markdown("<div class='set-title'>⚙ Settings</div>", unsafe_allow_html=True)
        st.caption("Every option of the Test program that is not on the page itself.")
        st.markdown("### Market")
        _picker()
        return st.toggle("Live prices", value=False, key="live",
                         help="Live prices, with 1-day and 7-day charts. On: today's prices from Yahoo "
                              "Finance. Off: the saved files, which end 30 October 2026.")


def _share_link(name):
    """The address of this exact view (market and model settings), for the Share box in Settings.
    Only saved markets go in a link: a link never makes the server fetch a ticker."""
    ss = st.session_state
    want = ex.write_link(name, ss.run, "S&P 500", c.default_run()) if ss.asset == ("saved", name) else {}
    try:
        base = (st.context.url or "").split("?")[0]
    except Exception:                          # no browser behind the page (tests): a relative link
        base = ""
    return base + ("?" + urlencode(want) if want else "")


def _settings_rest(name):
    """The rest of Settings: chart switches, more model settings, run options, the share link and
    Reset everything. Drawn before the chart, whose fragment reads the switches' values."""
    sp_days = c.default_run()["days"]
    with st.sidebar:
        st.markdown("### Chart")
        st.toggle("Mark big days (beyond 4 sd)", value=True, help=c.HELP["big"] + " (Daily ranges only.)",
                  key="t_big")
        st.toggle("Show the fall from the last peak (drawdown)", value=True, help=c.HELP["dd"], key="t_dd")
        # fixed key and default: a default that changed with the range made Streamlit rebuild
        # the switch (and the chart after it) on every new range
        st.toggle("Log price scale (equal heights = equal % changes)", value=True, key="t_log")
        st.markdown("### More model settings")
        st.slider("Reaction to price trends (a2)", 0.0, 1.0, key="in_a2", step=0.01, format="%.2f",
                  help="How strongly trend chasers react to recent price moves.")
        st.slider("How often trend chasers change mood (v1)", 0.5, 5.0, key="in_v1", step=0.01, format="%.2f",
                  help="How often trend chasers reconsider their mood.")
        # full width with a short label: "19,329 (as long as the S&P 500 record)" was cut off
        st.selectbox("Simulated trading days", [5000, 10000, sp_days], key="in_days",
                     format_func=lambda d: f"{d:,}" + (" (S&P 500 length)" if d == sp_days else ""),
                     help=f"How many trading days to simulate. {sp_days:,} is as long as the S&P 500's "
                          f"record since 1950.")
        st.number_input("Seed (picks the simulated history)", 0, 10_000, key="in_seed",
                        help="The number that picks one simulated history.")
        st.toggle("Run automatically", key="auto_run",
                  help="On: the model runs as soon as you let go of a slider (about half a second, instantly "
                       "for settings you ran before). Off: change several settings, then press Run model "
                       "(it then appears under the herding slider).")
        st.markdown("### More")
        link = _share_link(name)
        st.markdown("<p class='muted' style='margin:0 0 .2rem'>Copy a link to this exact view "
                    "(the copy button is at the right of the box):</p>", unsafe_allow_html=True)
        st.code(link or "(this page's address)", language=None, wrap_lines=True)
        if st.session_state.asset == ("live", name):
            st.caption("A typed stock symbol is not put in links, so this link opens the page at its start.")
        st.button("Reset everything", on_click=_reset_all, icon=":material/restart_alt:",
                  help="Back to the S&P 500 and the fitted settings. Your history log stays.")


def _load(live):
    ss = st.session_state
    source, name = ss.asset
    if source == "live":
        # the cached prices of a typed ticker last 24 hours (or until Clear cache); fetching them
        # again can fail while the visitor still has that ticker open, which used to replace the
        # whole page with a red ConnectionError (WS-23). Fall back to the S&P 500 and say why.
        try:
            return (*c.live_asset(name), "live", name)
        except (ConnectionError, ValueError):
            ss.asset = ("saved", "S&P 500")
            ss.ticker_error = (f"Live prices for {name} are not available right now. Showing the S&P 500. "
                               f"Type {name} again in a minute, or pick a market above.")
            ss.ticker_error_for = ss.asset
            ss.typed_symbol = None
            ss._fallback_done = True              # next run: empty the ticker box, Indexes tab
            st.toast(ss.ticker_error)
            return (*c.saved_asset("S&P 500"), "saved", "S&P 500")   # not live again: Yahoo just failed
    if live:
        symbol = ex.yahoo_symbol(name)
        if not _failed_recently(("daily", symbol)):     # Yahoo just failed: do not ask on every click
            try:
                return (*c.live_asset(symbol), "live", name)
            except (ConnectionError, ValueError):
                _remember_failure(("daily", symbol))
                if symbol not in ex.FRED_SERIES:
                    st.toast(f"Live prices for {name} are not available right now, so this shows the saved data.")
        if symbol in ex.FRED_SERIES:            # Yahoo did not answer: today's daily closes from FRED
            got = c.fred_asset(name, symbol)
            if got is not None:
                return (*got, "fred", name)
            st.toast(f"Live prices for {name} are not available right now, so this shows the saved data.")
    return (*c.saved_asset(name), "saved", name)


def _log_ticks(lo, hi):
    """(values, labels) for a log price axis from lo to hi: powers of ten read 1, 10, 100, 1,000
    (plotly's own labels put small "2" and "5" between them). When fewer than 3 powers of ten fall
    in the range, 2 and 5 times them are added, written out in full; None when the range is inside
    one step (plotly's own labels are fine there)."""
    lo, hi = float(lo), float(hi)
    if not (0 < lo <= hi and np.isfinite(hi)):
        raise ValueError(f"need 0 < lo <= hi, got {lo}, {hi}")
    ks = range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1)
    vals = [10.0 ** k for k in ks if lo <= 10.0 ** k <= hi]
    if len(vals) < 3:
        vals = [m * 10.0 ** k for k in ks for m in (1, 2, 5) if lo <= m * 10.0 ** k <= hi]
    if len(vals) < 2:
        return None
    labels = [f"{v:,.0f}" if v >= 1 else np.format_float_positional(v, precision=12, trim="-") for v in vals]
    return vals, labels


def _price_chart(name, cl, r, daily_range, show_big, show_dd, log, intraday, hours=None):
    """The price (in the market's own color: up/down colors only go with an arrow), big days as
    triangles, and the drawdown below. A short range of daily closes (1D or 7D without prices
    within the day) is drawn with a dot at each close and one date per day, not hours."""
    rows = 2 if show_dd else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[0.74, 0.26] if show_dd else [1])
    x, y = c.thin(cl.index, cl.values)
    fmt = "%b %d %H:%M" if intraday else "%b %d, %Y"
    few_days = not intraday and len(cl) <= 10
    fig.add_scatter(x=x, y=y, mode="lines+markers" if few_days else "lines", name=name,
                    line=dict(color=c.REAL, width=1.8),
                    hovertemplate=f"%{{x|{fmt}}}<br>%{{y:,.{ex.price_decimals(cl.min())}f}}<extra></extra>",
                    row=1, col=1)
    if show_big and not intraday:
        big = ex.big_day_list(r, within=ex.moves_in_range(r, daily_range))   # not the move into the first day
        for up, sym, col in ((True, "triangle-up", c.UP), (False, "triangle-down", c.DOWN)):
            b = big[(big["move"] > 0) == up]
            fig.add_scatter(x=b.index, y=daily_range[b.index].values, mode="markers",
                            name=f"Big {'rise' if up else 'fall'} (> 4 sd)",
                            marker=dict(symbol=sym, size=10, color=col, line=dict(color=c.BG, width=1)),
                            customdata=np.c_[b["move"], b["sd"]],
                            hovertemplate="%{x|%b %d, %Y}<br>%{customdata[0]:+.1%} (%{customdata[1]:+.1f} sd)<extra></extra>",
                            row=1, col=1)
    if show_dd:
        dd = ex.drawdown_series(cl)
        x, y = c.thin(dd.index, dd.values)
        fig.add_scatter(x=x, y=y, mode="lines", name="▼ Drawdown", fill="tozeroy",   # loss color with its arrow
                        line=dict(color=c.DOWN, width=1), fillcolor=c.DOWN_FILL,
                        hovertemplate=f"%{{x|{fmt}}}<br>%{{y:.1%}} below the high<extra></extra>", row=2, col=1)
    c.style(fig, height=470)
    fig.update_yaxes(type="log" if log else "-", title="Price", row=1, col=1)
    ticks = _log_ticks(cl.min(), cl.max()) if log else None
    if ticks:
        fig.update_yaxes(tickvals=ticks[0], ticktext=ticks[1], row=1, col=1)
    if show_dd:
        # a range that never fell is all 0: plotly then spread the axis from -100% to +100%
        low = min(float(dd.min()), -0.01) * 1.05
        fig.update_yaxes(tickformat=".0%", title="Drawdown", range=[low, -low * 0.04], row=2, col=1)
    if few_days:                # one tick per day: plotly labelled the hours between two closes
        fig.update_xaxes(tickformat="%b %d", dtick=86_400_000)
    if intraday:            # hide weekends and nights, but only for markets closed then (crypto never is)
        hours = hours or {"nights": False, "weekends": False}
        breaks = ([] if hours["weekends"] else [dict(bounds=["sat", "mon"])]) + \
                 ([] if hours["nights"] else [dict(bounds=[16, 9.5], pattern="hour")])
        if breaks:
            fig.update_xaxes(rangebreaks=breaks)
    return fig


@st.cache_data(show_spinner=False, max_entries=40)
def _price_fig(version, name, cl, r, daily_range, show_big, show_dd, log, intraday, hours):
    """_price_chart, kept: the price chart doesn't change when the model runs, and building it took
    about 0.13 s on every slider move (most of it plotly checking every axis setting). version
    (PAGE_VERSION) is in the key, so an edit to this page starts afresh."""
    return _price_chart(name, cl, r, daily_range, show_big, show_dd, log, intraday, hours=hours).to_dict()


def _day_reader(r, daily_range):
    """Under the price chart: a menu of the big days in the range shown (biggest first) and what
    happened on the one picked, with its documented cause when it is one of the Dow's 20 biggest days.
    A menu, not a click on the chart's triangles: those are small to hit (on a phone above all), and
    selecting on click needs a hover box that reads one line, not every line at the date."""
    big = ex.big_day_list(r, within=ex.moves_in_range(r, daily_range))
    if big.empty:
        return
    labels = {d: f"{d:%d %b %Y}  {'▲' if m > 0 else '▼'} {abs(m):.1%}" for d, m in big["move"].items()}
    day = st.selectbox("Read about a big day", list(labels), index=None, format_func=labels.get,
                       placeholder=f"Pick one of the {len(big)} big days shown",
                       # a new menu when the list changes (another market or range): a day picked
                       # before can't linger in a list that no longer has it
                       key=f"big_day_{len(big)}_{big.index.min():%Y%m%d}_{big.index.max():%Y%m%d}")
    if day is None:
        return
    text, sources = ex.day_story(day, float(big.loc[day, "move"]), float(big.loc[day, "sd"]))
    links = " ".join(f"[source {i}]({u})" for i, u in enumerate(sources, 1) if u.startswith("https://"))
    st.info(text + (f" {links}" if links else ""), icon=":material/event:")


def _share_text(p):
    """A share of days for a hover box: 17%, 2.4%, 0.031%."""
    p = 100 * float(p)
    return f"{p:.0f}%" if p >= 10 else f"{p:.1f}%" if p >= 1 else f"{p:.2g}%"


def _sd_text(x):
    """A move in standard deviations with a true minus sign: −1.0, +2.4, 0.0."""
    return f"{x:+.1f}".replace("-", "\u2212") if round(x, 1) != 0 else "0.0"


def _histogram_hover(label, centers, density, width):
    """Hover text for each bin of the histogram: "S&P 500: moves near −1.0 sd on 17% of days"
    (density times the bin width is the share of days in the bin)."""
    return [f"{label}: moves near {_sd_text(x)} sd on {_share_text(d * width)} of days"
            if np.isfinite(d) else f"{label}: no day near {_sd_text(x)} sd" for x, d in zip(centers, density)]


def _model_chart(view, name, r, mixes, earlier, a1):
    """The Tail, Histogram or Memory view: the market, the model's trader mixes and (dotted) your
    earlier runs. These compare shapes, so the hover box reads the one line under the mouse."""
    series = [(name, r.values, dict(color=c.REAL, width=3))] + \
             [(c.MIX[m][0], mixes[m][0], c.mix_line(m)) for m in _shown_mixes(a1) if mixes[m][0] is not None] + \
             [(label, rr, dict(color=c.MIX["herding"][2], width=1, dash="dot")) for label, rr in earlier]
    if view == "Tail":
        fig = c.tail_chart(series, height=470)
        fig.update_layout(hovermode="closest")     # the same as style(..., hover="closest")
        return fig
    fig = go.Figure()
    if view == "Histogram":
        width = 0.4                                  # explore.histogram's bin width
        for label, rr, line in series:
            cx, d = ex.histogram(rr, empty=np.nan)        # NaN: an empty bin can't be drawn on a log scale
            fig.add_scatter(x=cx, y=d, name=label, mode="lines", line=line,
                            text=_histogram_hover(label, cx, d, width), hovertemplate="%{text}<extra></extra>")
        g = np.linspace(-8, 8, 161)
        fig.add_scatter(x=g, y=ex.bell_density(g), name="Bell curve", mode="lines",
                        line=dict(color=c.BELL, dash="dot"),
                        text=_histogram_hover("Bell curve", g, ex.bell_density(g), width),
                        hovertemplate="%{text}<extra></extra>")
        fig.update_yaxes(range=[-4.5, 0])
        return c.style(fig, 470, logy=True, xtitle="Daily move (sd)", ytitle="How common (log scale)",
                       hover="closest", plain_log=True)
    for label, rr, line in series:
        lx, cc = ex.memory_curve(rr)
        fig.add_scatter(x=lx, y=cc, name=label, mode="lines+markers", line=line, marker=dict(size=5),
                        hovertemplate=f"{label}<br>%{{x:,}} days apart: correlation %{{y:.2f}}<extra></extra>")
    fig.add_hline(y=0, line=dict(color=c.BELL, dash="dot"))
    return c.style(fig, 470, logx=True, xtitle="Days apart", ytitle="Correlation of move sizes", hover="closest")


def _model_market_chart(mm, mix, zoom=None):
    """The simulated market of one trader mix (explore.model_market): price and true value (log), big
    days, the share of trend chasers and their mood, on one shared day axis, so zooming one panel
    zooms all three."""
    line = c.mix_line(mix)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.58, 0.21, 0.21])
    day = np.arange(len(mm["price"]))
    keep = slice(None) if zoom is None else slice(zoom[0], zoom[1] + 1)   # zoomed: every day of the window
    day = day[keep]
    for key, label, style in (("value", "True value", dict(color=c.GUIDE, width=1.2, dash="dot")),
                              ("price", f"Price ({c.MIX[mix][0].lower()})", dict(line, width=1.6))):
        x, y = c.thin(day, np.log(mm[key][keep]))
        fig.add_scatter(x=x, y=np.exp(y), mode="lines", name=label, line=style,
                        hovertemplate=f"day %{{x:,}}<br>{label}: %{{y:,.1f}}<extra></extra>", row=1, col=1)
    big = ex.big_day_list(mm["r"])          # indexed by move number: move i ends on day i + 1
    for up, sym, col in ((True, "triangle-up", c.UP), (False, "triangle-down", c.DOWN)):
        b = big[(big["move"] > 0) == up]
        d = b.index.to_numpy() + 1
        if zoom is not None:
            inside = (d >= zoom[0]) & (d <= zoom[1])
            b, d = b[inside], d[inside]
        fig.add_scatter(x=d, y=mm["price"][d], mode="markers",
                        name=f"Big {'rise' if up else 'fall'} (> 4 sd)",
                        marker=dict(symbol=sym, size=9, color=col, line=dict(color=c.BG, width=1)),
                        customdata=np.c_[b["move"], b["sd"]],
                        hovertemplate="day %{x:,}<br>%{customdata[0]:+.1%} (%{customdata[1]:+.1f} sd)<extra></extra>",
                        row=1, col=1)
    if mix != "fundamentalists":
        x, y = c.thin(day, mm["z"][keep])
        fig.add_scatter(x=x, y=y, mode="lines", name="Share of trend chasers", fill="tozeroy",
                        line=dict(color=line["color"], width=1), fillcolor=c.HERD_FILL,
                        hovertemplate="day %{x:,}<br>%{y:.0%} of traders are trend chasers<extra></extra>", row=2, col=1)
        # optimists minus pessimists as a share of ALL traders: what pushes the price (excess demand,
        # SPEC). The chartists' own mood (x) swung between -1 and +1 whenever only a few traders were
        # chartists, so its panel was a solid band that said little
        crowd = (mm["z"] * mm["x"])[keep]
        x, y = c.thin(day, crowd)
        fig.add_scatter(x=x, y=y, mode="lines", name="Mood: optimists − pessimists",
                        line=dict(color=c.SHADE_MID, width=1),
                        hovertemplate="day %{x:,}<br>optimists − pessimists: %{y:+.1%} of all traders"
                                      "<extra></extra>", row=3, col=1)
        fig.add_hline(y=0, line=dict(color=c.GUIDE, dash="dot", width=1), row=3, col=1)
    else:
        for row in (2, 3):
            fig.add_annotation(text="No trend chasers in this mix", showarrow=False, x=0.5, y=0.5,
                               xref=f"x{row} domain", yref=f"y{row} domain", font=dict(color=c.MUTED, size=12))
    if mm["runaway_day"] is not None:
        fig.add_vline(x=mm["runaway_day"], line=dict(color=c.DOWN, dash="dash", width=1))
        fig.add_annotation(x=mm["runaway_day"], y=0.98, xref="x", yref="y domain", text="ran away here",
                           showarrow=False, xanchor="right", font=dict(color=c.DOWN, size=11))
    c.style(fig, height=540, hover="closest")
    fig.update_yaxes(type="log", title="Price (day 0 = 100)", row=1, col=1)
    shown = np.r_[mm["price"], mm["value"]] if zoom is None else \
        np.r_[mm["price"][zoom[0]:zoom[1] + 1], mm["value"][zoom[0]:zoom[1] + 1]]
    ticks = _log_ticks(shown.min(), shown.max())
    if ticks:
        fig.update_yaxes(tickvals=ticks[0], ticktext=ticks[1], row=1, col=1)
    fig.update_yaxes(title="Trend chasers", tickformat=".0%", range=[0, 1], row=2, col=1)
    fig.update_yaxes(title="Mood", tickformat="+.0%", row=3, col=1)
    fig.update_xaxes(title="Simulated trading day", row=3, col=1)
    if zoom is not None:           # (first, last) day: zoom every panel there, and fit the price axis to it
        first, last = zoom
        fig.update_xaxes(range=[first - 0.5, last + 0.5])
        lo, hi = np.log10(shown.min()), np.log10(shown.max())
        pad = max(0.02, 0.06 * (hi - lo))
        fig.update_yaxes(range=[lo - pad, hi + pad], row=1, col=1)
    return fig


ZOOMS = {"all": "Whole history", "crash": "Biggest crash", "burst": "Wildest year"}


@st.cache_data(show_spinner=False, max_entries=30)
def _model_market_fig(version, mix, run, zoom="all"):
    """_model_market_chart for one run (run: its settings as sorted pairs), kept like _price_fig."""
    mm = c.model_market(mix, **dict(run))
    window = None if zoom == "all" else ex.zoom_window(mm["r"], zoom)
    return _model_market_chart(mm, mix, window).to_dict()


def _model_market_panel(a1):
    """The Model market view: pick a trader mix, then its simulated market (_model_market_chart)."""
    ss = st.session_state
    run = ss.run
    shown = _shown_mixes(a1)
    mix = c.pick_one("Traders", shown, "herding" if "herding" in shown else "chartists", "mm_mix",
                     format_func=lambda m: c.MIX[m][0], label_visibility="collapsed")
    zoom = c.pick_one("Zoom to", list(ZOOMS), "all", "mm_zoom", format_func=ZOOMS.get,
                      label_visibility="collapsed")
    mm = c.model_market(mix, **run)
    if mm is None:
        st.caption(f"\"{c.MIX[mix][0]}\" ran away in the warm-up, before day 0, so there is nothing to show.")
        return
    note = ""
    if mm["runaway_day"] is not None:
        note = (f"This history ran away on day {mm['runaway_day']:,}: the price left its true value for good, so the "
                f"other charts leave it out. Here it is up to that day.")
    nm = ex.news_and_moves(mm)
    st.caption((note + " " if note else "") +
               (f"On average {nm['chartists_mean']:.0%} of traders are trend chasers, at most "
                f"{nm['chartists_peak']:.0%}. " if mix != "fundamentalists" else "") +
               f"{len(ex.big_day_list(mm['r'])):,} days moved more than 4 sd "
               f"(a bell curve expects {ex.bell_big_days(len(mm['r'])):.1f}).")
    st.markdown(f"**News in, moves out.** The news (daily changes of the true value) has excess kurtosis "
                f"**{nm['news_kurtosis']:+.2f}**: a bell curve. The price's daily moves have "
                f"**{nm['price_kurtosis']:+.2f}**" +
                (": the extra big days come from the traders, not the news." if nm["price_kurtosis"] > 1
                 else ", close to a bell curve too."))
    if len(mm["r"]) > ex.ZOOM_DAYS and zoom != "all":
        first, last = ex.zoom_window(mm["r"], zoom)
        st.caption(f"Showing days {first:,} to {last:,}. "
                   + ("Watch the share of trend chasers and the crowd's mood just before the fall."
                      if zoom == "crash" else "The year with the most movement: wild days come in bursts."))
    c.show(_model_market_fig(PAGE_VERSION, mix, tuple(sorted(run.items())), zoom), key="model_market_chart")
    c.try_this(TRY["Model market"])
    c.hint(HINTS["Model market"])


def _bursts_chart(name, rr, r, herd, herd_label, herd_note):
    """Daily moves in sd: the market's (of all its days) on top, the model's with herding
    below, or herd_note in the empty panel when its history ran away."""
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.1)
    z = ex.moves_in_sd(rr, ref=r)
    x, y = c.thin(z.index, z.values)
    fig.add_scatter(x=x, y=y, mode="lines", line=dict(color=c.REAL, width=1), name=f"{name}, real days",
                    hovertemplate="%{x|%b %d, %Y}: %{y:+.1f} sd<extra></extra>", row=1, col=1)
    if herd is not None:
        hz = ex.moves_in_sd(herd)
        x, y = c.thin(np.arange(len(hz)), hz)
        fig.add_scatter(x=x, y=y, mode="lines", line=dict(color=c.MIX["herding"][2], width=1),
                        name=f"{herd_label}, simulated days",
                        hovertemplate="day %{x}: %{y:+.1f} sd<extra></extra>", row=2, col=1)
    else:
        fig.add_annotation(text=herd_note, showarrow=False, x=0.5, y=0.5, xref="x2 domain", yref="y2 domain",
                           font=dict(color=c.MUTED, size=12))
    for i in (1, 2):
        for lim in (-4, 4):         # a threshold, not a loss: neutral guide color (UP/DOWN mean gains/losses)
            fig.add_hline(y=lim, line=dict(color=c.GUIDE, dash="dot", width=1), row=i, col=1)
    c.style(fig, 470)
    fig.update_yaxes(title="Move (sd)")
    return fig


def _shown_mixes(a1):
    """The trader mixes worth drawing: with a1 = 0 "trend chasers who herd" do not herd, so their history
    is the same as "trend chasers, no herding" and a second, identical line would only confuse."""
    return [m for m in ex.MIXES if not (m == "herding" and a1 == 0)]


def _missing_note(mixes, a1):
    """One sentence for each trader mix the charts leave out, so the color key never names a line
    that silently isn't there."""
    notes = []
    for m in ex.MIXES:
        if m not in _shown_mixes(a1):
            notes.append(f"With herding strength 0 trend chasers do not herd, so \"{c.MIX[m][0]}\" is the same history as "
                         f"\"{c.MIX['chartists'][0]}\" and is drawn once.")
        elif mixes[m][0] is None:
            notes.append(f"\"{c.MIX[m][0]}\" {c.gone_text(mixes[m][1])}, so it has no line.")
    return " ".join(notes)


def _earlier_runs():
    """Up to four of your earlier histories that differ from the one shown now, newest first."""
    ss = st.session_state
    return [prm for _, prm, _ in ss.log if prm != ss.run][:4]


def _earlier_label(prm, now):
    label = f"Earlier history: herding strength {prm['a1']:.2f}, seed {prm['seed']}"
    extra = [f"{k} {prm[k]:.2f}" for k in ("a2", "v1") if prm[k] != now[k]]
    if prm["days"] != now["days"]:
        extra.append(f"{prm['days']:,} days")
    return label + (", " + ", ".join(extra) if extra else "")


def _key_items(view, a1, gone=()):
    """What the color key above the chart lists: only lines the chosen view draws. Price draws the
    market only (no key); Bursts the market and the herding model; the other views every mix shown.
    gone: the mixes whose history ran away, which have no line, so the key leaves them out too."""
    if view in ("Price", "Model market"):      # their own legend names every line
        return ()
    if view == "Bursts":
        return ("real", "herding") if a1 != 0 and "herding" not in gone else ("real",)
    return ("real", "bell", *[m for m in _shown_mixes(a1) if m not in gone])


def _range_note(frame, source, live, note):
    """The note for a 1D or 7D range drawn from daily closes: why there are no prices within the day."""
    if frame not in ex.DAILY_FALLBACK or source == "live":
        return note
    return YAHOO_DOWN_NOTE if live else NO_INTRADAY_NOTE


def _chart_tabs():
    """The four chart tabs and, on a tab with two charts, its switch. Returns the chart to draw (one
    of LEAVES). A chart named directly in "view" (Histogram or Memory, as before the tabs) opens its
    tab with the switch on it."""
    ss = st.session_state
    for tab, (key, subs) in SUBVIEWS.items():
        if ss.get("view") in subs[1:]:
            ss[key], ss["view"] = ss["view"], tab
    with st.container(key="chart_tabs"):
        view = c.pick_one("Chart", VIEWS, "Price", "view", label_visibility="collapsed",
                          format_func=lambda v: VIEW_LABELS.get(v, str(v)))
    if view in SUBVIEWS:
        key, subs = SUBVIEWS[view]
        with st.container(key="chart_sub"):
            return c.pick_one("Show", subs, subs[0], key, label_visibility="collapsed")
    return view


def _chart_panel(name, symbol, close, r, mixes, source, live=False, fetch=None):
    """Everything that only changes the chart. A fragment, so these clicks rerun just this part.
    live: whether Live prices is on (with source "saved", Yahoo did not answer). fetch: the Yahoo
    symbol of a market shown with live prices; the chart then reads its latest prices on every
    rerun (cached for c.LIVE_REFRESH seconds), so it keeps up with the header's price."""
    ss = st.session_state
    if fetch is not None:
        try:
            close, r, _ = c.live_asset(fetch)
        except (ConnectionError, ValueError):        # keep the prices the page was drawn with
            pass
    a1 = ss.run["a1"]
    view = _chart_tabs()
    frame = c.pick_one("Range", ex.FRAMES, "All", "frame", label_visibility="collapsed",
                       disabled=view not in ("Price", "Bursts"),
                       help="1D and 7D need Live prices (in Settings), or a typed stock symbol.")
    items = _key_items(view, a1, gone=[m for m in ex.MIXES if mixes[m][0] is None])
    if items:
        c.color_key(*items)
    missing = _missing_note(mixes, a1)
    if view in ("Price", "Bursts"):
        bursts_note = ""
        if view == "Bursts" and frame in ex.DAILY_FALLBACK:  # bursts are daily moves: a day or a week is too short
            frame, bursts_note = "1Y", "Bursts use daily moves, so 1D and 7D show the past year."
        bars = None
        # prices within the day come only from Yahoo, so they are fetched only when the page shows
        # Yahoo's prices: with saved data, the header (saved close) and the chart (today) disagreed.
        # A failed fetch is remembered for 5 minutes, so every click does not wait for Yahoo again
        if frame in ex.DAILY_FALLBACK and source == "live" and not _failed_recently(("intraday", symbol, frame)):
            try:
                bars = c.intraday(symbol, frame)
            except (ConnectionError, ValueError):
                _remember_failure(("intraday", symbol, frame))
                bars = None
        cl, note = ex.price_range(close, frame, bars)
        intraday = frame in ex.DAILY_FALLBACK and not note
        note = _range_note(frame, source, live, note)
        if intraday:            # close is Yahoo's daily prices here (source is live): one source on screen
            note = (f"Prices within the day from Yahoo Finance, New York time, up to "
                    f"{cl.index[-1]:%d %b %H:%M}. The 52-week numbers use Yahoo's daily prices to "
                    f"{close.index[-1]:%d %b %Y} plus these prices within the day.")
        if note or bursts_note:         # above the chart: read before the short line, not after
            st.caption(note or bursts_note)
        daily_range = ex.daily_within(close, cl) if intraday else cl
        st_ = ex.range_stats(cl, daily=close)
        chg, col = c.arrow(st_["change"])
        vol = st_["volatility"]
        ch = abs(st_["change"])
        # 1D bars start at the open, not at yesterday's close, so their change is "since open" (the
        # header's "on the day" is against yesterday's close)
        change_label = "Since open" if intraday and frame == "1D" else "Change"
        c.stats_row([(change_label, f"{'▲' if st_['change'] >= 0 else '▼'} {ch:,.0%}" if ch >= 10 else
                      f"{'▲' if st_['change'] >= 0 else '▼'} {ch:.2%}", col, None),
                     ("High", ex.price_text(st_["high"]), None, None), ("Low", ex.price_text(st_["low"]), None, None),
                     ("52-week high", ex.price_text(st_["high_52w"]), None, None),
                     ("52-week low", ex.price_text(st_["low_52w"]), None, None),
                     ("Volatility (per year)", "–" if np.isnan(vol) else f"{vol:.1%}", None, c.HELP["vol"]),
                     ("Max drawdown", f"{st_['drawdown']:.1%}", None, c.HELP["dd"])])
        if view == "Price":
            # the switches are in Settings (the sidebar): read here, so a switch reruns this chart
            show_big, show_dd = ss.get("t_big", True), ss.get("t_dd", True)
            log = ss.get("t_log", True)
            c.show(_price_fig(PAGE_VERSION, name, cl, r, daily_range, show_big, show_dd, log, intraday,
                              ex.trading_hours(symbol, bars)), key="price_chart")
            c.try_this(TRY["Price"])
            c.hint(HINTS["Price"])
        else:
            rr = ex.moves_in_range(r, daily_range)    # n prices, n - 1 moves: none from before the range
            herd_label = c.MIX["herding"][0] if a1 != 0 else f"{c.MIX['chartists'][0]} (herding strength 0)"
            herd_note = (f"{c.MIX['herding'][0]} {c.gone_text(mixes['herding'][1])}, so there is nothing to show"
                         if mixes["herding"][0] is None else "")
            c.show(_bursts_chart(name, rr if len(rr) > 1 else r, r, mixes["herding"][0], herd_label, herd_note),
                   key="bursts_chart")
            c.try_this(TRY["Bursts"])
            c.hint(HINTS["Bursts"])
    elif view == "Model market":
        _model_market_panel(a1)
    else:
        earlier = []
        others = _earlier_runs()
        if others and st.toggle("Show my earlier histories", value=False, key="t_earlier",
                                help="Adds the line with herding for each of your last four other histories, dotted."):
            gone = 0
            for prm in others:
                rr = c.run_mixes(**prm)["herding"][0]
                if rr is None:
                    gone += 1
                else:
                    earlier.append((_earlier_label(prm, ss.run), rr))
            if gone:
                missing += (" 1 earlier history ran away, so it has no line." if gone == 1 else
                            f" {gone} of your earlier histories ran away, so they have no line.")
        c.show(_model_chart(view, name, r, mixes, earlier, a1), key=f"model_chart_{view}")
        c.try_this(TRY[view].format(name=name))
        c.hint(HINTS[view])
        if missing:
            st.caption(missing.strip())


chart_panel = st.fragment(_chart_panel)
# with live prices on, the chart redraws itself every LIVE_REFRESH seconds with Yahoo's latest prices
chart_panel_live = st.fragment(run_every=c.LIVE_REFRESH)(_chart_panel)


SOURCE_TEXT = {"live": "live from Yahoo", "saved": "saved data",
               "fred": "saved data + daily closes from FRED (Yahoo is not answering)"}


def _quote_html(name, close, source, spot, quote=None):
    """The header: kind of market, where the prices come from, the last price and its day change.
    quote: Yahoo's latest quote (common.live_quote), shown instead of the last daily close."""
    price, change = (quote["price"], quote["change"]) if quote else (close.iloc[-1], ex.day_change(close))
    chg, _ = c.arrow(change)
    when = (f"quote at {datetime.fromtimestamp(quote['at']):%H:%M:%S}" if quote else
            f"last close {close.index[-1]:%d %b %Y}")
    live_hint = " · Live prices: in ⚙ Settings" if source == "saved" and not quote else ""
    return (f"<div class='quote'><h1 style='margin:0'>{ex.display_name(name)}</h1>"
            f"<span class='price'>{ex.price_text(price)}</span><span class='chg'>{chg} "
            f"<span class='muted'>on the day</span></span>"
            f"<span class='src'>{ex.asset_kind(name)} · "
            f"{SOURCE_TEXT.get(source, 'saved data')}{spot} · {when}{live_hint}</span></div>")


@st.fragment(run_every=c.QUOTE_REFRESH)
def _live_quote(name, close, fetch):
    """The header of a market shown with live prices: Yahoo's latest quote, asked for again every
    QUOTE_REFRESH seconds, so the price on screen is never more than about 15 s behind Yahoo. If the
    quote does not come, the latest daily close (refreshed every minute) is shown instead."""
    quote = c.live_quote(fetch)
    if quote is None:
        try:
            close = c.live_asset(fetch)[0]
        except (ConnectionError, ValueError):        # keep the last price the page had
            pass
    st.markdown(_quote_html(name, close, "live", "", quote), unsafe_allow_html=True)


def _model_table(name, s, r, sums, mixes, a1):
    """The Model vs market table: numbers from explore.mix_table, formatted for reading."""
    t = ex.mix_table(name, s, r, sums, mixes)
    rows = ["Days", "Tail exponent", "Excess kurtosis", "Memory of wild days (20 days)",
            "Worst day (sd)", "Big days (beyond 4 sd)", "Max drawdown"]
    fmt = ["{:,.0f}", "{:.2f}", "{:.2f}", "{:.2f}", "{:.1f}", "{:,.0f}", "{:.1%}"]

    def col(row):
        if not np.isnan(row["runaway_day"]):
            return [_gone_short(row["runaway_day"])] + [""] * (len(rows) - 1)
        return [f.format(row[k]) for f, k in zip(fmt, ex.TABLE_MEASURES[:7])]
    tab = {name: col(t.loc[name])}
    for m in _shown_mixes(a1):
        tab[c.MIX[m][0]] = col(t.loc[m])
    b = t.loc["bell"]
    tab["Bell curve"] = [f"{b['days']:,.0f}", "none", f"{b['kurtosis']:.2f}", f"{b['memory']:.2f}",
                                         "", f"{b['big_days']:.1f}", ""]
    return pd.DataFrame(tab, index=rows)


def _gone_short(day):
    """A ran-away history in a table cell, short enough not to be cut: "ran away (day 7,282)"."""
    return "ran away (warm-up)" if day < 1 else f"ran away (day {int(day):,})"


def _model_table_config(name, a1):
    """Column help for the Model vs market table: the measures (rows) explained on the first column,
    and what each column is."""
    help_rows = (f"Tail exponent: {c.HELP['tail']}\n\nExcess kurtosis: {c.HELP['kurtosis'].removeprefix('Excess kurtosis. ')}\n\n"
                 f"Memory of wild days: {c.HELP['memory']}")
    cfg = {"_index": st.column_config.TextColumn("Measure", help=help_rows),
           name: st.column_config.TextColumn(help=f"{name}'s daily moves."),
           "Bell curve": st.column_config.TextColumn(
               help="What a bell curve expects in as many days as the market has (market length).")}
    for m in _shown_mixes(a1):
        cfg[c.MIX[m][0]] = st.column_config.TextColumn(help=f"One simulated history ({c.MIX[m][0].lower()}: {c.MIX[m][1]}).")
    return cfg


def _log_run(new, res):
    """Put a history at the top of the history log. One already in the log moves up instead of being
    added twice (pressing Run model again with the same settings gave duplicate rows)."""
    ss = st.session_state
    ss.log = [e for e in ss.log if e[1] != new]
    ss.log.insert(0, (datetime.now().strftime("%H:%M:%S"), new, ex.mix_tails(res)))


def _run_log_table():
    """The run log with one number type per column: a run that ran away has no tail (empty), and the
    Ran away column says which mixes ran away (a text 'ran away' in a number column broke the table)."""
    ss = st.session_state
    out = []
    for t, prm, tl in ss.log:
        row = {"Time": t, "Herding strength (a1)": prm["a1"], "Reaction to price trends (a2)": prm["a2"],
               "Mood changes (v1)": prm["v1"],
               "Days": prm["days"], "Seed": prm["seed"]}
        # at a1 = 0 "trend chasers who herd" is the same history as "trend chasers, no herding": no second number
        shown = _shown_mixes(prm["a1"])
        row.update({c.MIX[m][0]: (np.nan if v is None or m not in shown else float(v)) for m, v in tl.items()})
        row["Ran away"] = ", ".join(c.MIX[m][0] for m, v in tl.items() if v is None and m in shown)
        out.append(row)
    return pd.DataFrame(out)


def _run_log_view(df):
    """The run log as shown: tail exponents with 2 decimals and "—" where there is none (a history
    that ran away, or trend chasers who herd at a1 = 0); the numbers underneath stay numbers."""
    tails = [c.MIX[m][0] for m in ex.MIXES]
    return df.style.format({**{k: "{:.2f}" for k in tails + RUN_LOG_SETTINGS}, "Days": "{:,}"}, na_rep="—")


RUN_LOG_SETTINGS = ["Herding strength (a1)", "Reaction to price trends (a2)", "Mood changes (v1)"]
SLIDERS = (("in_a1", "a1"), ("in_a2", "a2"), ("in_v1", "v1"), ("in_days", "days"), ("in_seed", "seed"))


def _control_values():
    """The run the model controls describe now."""
    ss = st.session_state
    return {"a1": float(ss.in_a1), "a2": float(ss.in_a2), "v1": float(ss.in_v1),
            "days": int(ss.in_days), "seed": int(ss.in_seed)}


def _run(new):
    """Run the model with these settings (cached: a run made before comes back at once), put it in
    the run log and make it the run shown."""
    with st.spinner("Running the model..."):
        res = c.run_mixes(**new)
    _log_run(new, res)
    st.session_state.run = new


def _controls_changed():
    """True while a model control differs from the run shown, so the page can say Run model is needed."""
    ss = st.session_state
    return any(abs(float(ss[k]) - float(ss.run.get(p, 0.0))) > 1e-9 for k, p in SLIDERS)


def _run_line(name, s, sums, run, log):
    """The line under Run model after a run: what ran and its tail exponent next to the market's.
    Empty until a run is made on this page."""
    if not log or log[0][1] != run:
        return ""
    head = f"Ran at herding strength {run['a1']:.2f}"
    herd = sums["herding"]
    if herd is None:
        return f"{head}: the history with herding ran away, so it has no tail exponent."
    return f"{head}: tail exponent {herd['tail']:.2f} ({name}: {s['tail']:.2f})"


def _dial_caption(name, market_tail, n_seeds, off_fit_now, earlier_differ):
    """The words under the dial chart: what the pink line, the dots and the flat line are, and why
    stronger herding means a higher exponent here (figures/RESULTS.md, Fig 10)."""
    text = (f"Pink line and dots: the tail exponent at each herding strength, the middle value of {n_seeds} saved "
            "simulated histories (strengths where most of them ran away are left out; the shaded band holds the "
            "middle half). Solid dot: the history shown now; hollow circles: your earlier histories. "
            f"The flat line marked \"this market\" is {name}'s own tail exponent ({market_tail:.2f}). "
            "Here stronger herding gives a higher exponent, so thinner tails. In the project's tests "
            "(Fig 10) the trend chasers locked into one mood and fewer traders stayed trend chasers, and either "
            "could calm the big days.")
    if off_fit_now:
        text += (" Your reaction to price trends (a2) or mood changes (v1) differs from the fit, so your "
                 "histories are not directly comparable to the pink line.")
    elif earlier_differ:
        text += (" Some earlier histories used an a2 or v1 different from the fit, so those hollow circles "
                 "are not directly comparable to the pink line.")
    return text


SCORE_MARK = {"match": ("✓", "UP", "matches"), "close": ("≈", "ACCENT", "close"), "off": ("✗", "DOWN", "far off")}
SCORE_FMT = {"Fat tails": "{:.2f}", "Big days (per 1,000 days)": "{:.1f}", "Bursts of wild days": "{:.2f}"}
SCORE_HELP = {"Fat tails": "tail exponent (lower = more big days; the S&P 500 is about 3)",
              "Big days (per 1,000 days)": f"days beyond 4 sd per 1,000 days (a bell curve: {ex.bell_big_days(1000):.2f})",
              "Bursts of wild days": "memory of wild days a month apart (0 = no bursts)"}


def _scorecard(name, s, sums, a1):
    """How close the shown model history is to the market, in three plain checks (explore.scorecard)."""
    mix = "herding" if a1 != 0 else "chartists"
    rows = ex.scorecard(s, sums[mix])
    st.markdown(f"**How close is the model to {ex.display_name(name)}?**")
    if not rows:
        st.caption(f"{c.MIX[mix][0]} ran away in this simulated history, so there is nothing to score. Try another seed "
                   "or a different herding strength.")
        return
    html = []
    for row in rows:
        mark, col, word = SCORE_MARK[row["verdict"]]
        f = SCORE_FMT[row["what"]]
        html.append(f"<div title='{SCORE_HELP[row['what']]}' style='display:flex;gap:.5rem;align-items:baseline;"
                    f"padding:.15rem 0'><b style='color:{getattr(c, col)};width:1rem'>{mark}</b>"
                    f"<span style='flex:1'>{row['what']} <span class='muted'>({word})</span></span>"
                    f"<span class='muted'>{f.format(row['model'])} vs {f.format(row['market'])}</span></div>")
    st.markdown("".join(html), unsafe_allow_html=True)
    st.caption("Model and market; hover a row for what the number is. ✓ is as close as luck allows.")


def _model_panel(name, s, sums, a1):
    """The model: presets, the herding-strength slider, what ran (Run model only when Run
    automatically is off) and the scorecard. The other settings are in Settings."""
    ss = st.session_state
    with st.container(key="model_panel", border=True):
        st.subheader("The model", anchor="model")     # the phone-only "Jump to the model" link lands here
        # not a form: a form keeps slider moves to itself until it is sent, so the page couldn't say
        # "Settings changed" while they differ from the run shown
        st.segmented_control("Presets", list(PRESETS), key="preset", on_change=_preset,
                             label_visibility="collapsed", width="stretch",
                             help="A preset fills in the controls and runs the model. With too little herding "
                                  "the price runs away. More herding calms it. The S&P 500's fat tails appear "
                                  "only near the edge between the two.")
        st.markdown("<div class='preset-notes'>" + "".join(
            f"<div><b>{n}</b> {_preset_note(n)}</div>" for n in PRESETS) + "</div>", unsafe_allow_html=True)
        st.slider("Herding strength (a1)", 0.0, 2.0, key="in_a1", step=0.01, format="%.2f",
                  help="How strongly trend chasers copy the crowd's mood.")
        st.markdown("<div class='a1-scale'><span>runs away</span><span>wilder</span><span>calmer</span></div>",
                    unsafe_allow_html=True)
        if not ss.auto_run:         # with Run automatically on (Settings), a button would only repeat a run
            if st.button("Run model", type="primary", width="stretch"):
                _run(_control_values())
                st.rerun()
        if _controls_changed():
            st.markdown("<p class='muted' style='margin:.2rem 0'>Settings changed: press Run model to see them.</p>",
                        unsafe_allow_html=True)
        else:
            line = _run_line(name, s, sums, ss.run, ss.log)
            if line:
                st.success(line)
            else:
                st.caption("The model reruns as you drag. The charts switch to the new simulated history.")
        _scorecard(name, s, sums, a1)
        st.caption("Reaction to price trends, mood changes, length, seed, other markets and the share link "
                   "are in ⚙ Settings (the sidebar; the » button at the top left when it is closed).")


def _dial_panel(name, s, sums, a1):
    """Every history on one chart: the tail exponent against herding strength, with every history made."""
    ss = st.session_state
    p, _ = c.fitted()
    runs, earlier_differ = ex.dial_runs(ss.log, p)
    cur = sums["herding"]["tail"] if sums["herding"] else None
    st.markdown("**Tail exponent vs herding strength**")
    c.show(c.dial_chart(s["tail"], runs, (a1, cur), height=320), key="dial")
    st.caption(_dial_caption(name, s["tail"], len(c.numbers()["sweep"]["seeds"]),
                             ex.differs_from_fit(ss.run, p), earlier_differ))


def _tasks_done(name, s, sums):
    """Which of the three tasks are done, from what the visitor has really done this session."""
    ss = st.session_state
    default_seed = c.default_run()["seed"]
    herd = sums.get("herding")
    if (name == "S&P 500" and ss.run["seed"] != default_seed and herd is not None
            and abs(herd["tail"] - s["tail"]) <= CHALLENGE_GAP):
        ss.challenge_done = True
    return [ss.get("mm_zoom") == "crash",
            any(prm["a1"] <= EDGE_A1 for _, prm, _ in ss.log),
            bool(ss.get("challenge_done"))]


EDGE_A1 = 0.8            # at or below this, half or more of the saved histories run away (the sweep)
CHALLENGE_GAP = 0.10     # the challenge: the herding tail within this of the S&P 500's


def _tasks(box, done, spx_tail):
    items = [
        "**Watch a crash.** Open the **Watch a crash** chart and press **Biggest crash**.",
        f"**Find the edge.** Slide **Herding strength** down to {EDGE_A1} or less. Does the market get "
        "wilder, or run away? (Copying less makes it less stable, not calmer.)",
        f"**Challenge.** In **⚙ Settings**, change the **seed** to draw a new simulated history. "
        f"Then move **Herding strength** until the model's tail exponent is within {CHALLENGE_GAP} of the S&P 500's "
        f"{spx_tail:.2f}. Is the fitted value still the answer?",
    ]
    n = sum(done)
    with box.container(key="tasks"):
        with st.expander(f"Three things to try · {n} of 3 done", expanded=n < 3, icon=":material/checklist:"):
            if n:                      # an empty bar only left a gap; the count above says 0 of 3
                st.progress(n / 3)
            for ok, text in zip(done, items):
                st.markdown(("✅ ~~" if ok else "⬜ ") + text + ("~~" if ok else ""))
            if n == 3:
                st.success(f"All three done. The Checks page has the same questions answered over "
                           f"{len(c.numbers()['recipe']['seeds'])} simulated histories.")
            st.caption("A simulation for learning, not a forecast or investment advice. "
                       "The Start here page explains every word.")


LOW_TABS = ["Model and {name}", "Big days", "History log", "Every history on one chart"]


def _style():
    """This page's own look, in the site's colors (web/common.py): slim task card, quote bar, chart
    tabs drawn as a segmented bar, the model panel and the one-line insight."""
    p = c.pal()
    a, line, panel, muted, text2, real = p["ACCENT"], p["LINE"], p["PANEL"], p["MUTED"], p["TEXT2"], p["REAL"]
    st.html(f"""<style>
    .lab-eyebrow {{font-family: {c.MONO}; font-size: .78rem; font-weight: 700; letter-spacing: .1em;
        color: {a}; margin: 0}}
    .lab-title {{font-family: {c.MONO}; font-size: 1.15rem; font-weight: 600; color: {real}; margin: 0 0 .5rem}}
    .st-key-tasks [data-testid="stExpander"] details {{border: 1px solid {line}; border-radius: 2px; background: {panel}}}
    .st-key-tasks [data-testid="stExpander"] summary p {{font-family: {c.MONO}; font-size: .86rem; font-weight: 700}}
    .st-key-tasks [data-testid="stProgress"] > div > div {{height: 3px}}
    .quote {{display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 14px; padding: 2px 0 8px;
        border-bottom: 1px solid {line}; margin-bottom: 6px}}
    .quote h1 {{font-size: 1.25rem !important; margin: 0 !important; color: {real}}}
    .quote h1::after {{display: none}}
    .quote .price {{font-size: 1.9rem; color: {real}; line-height: 1.1}}
    .quote .chg {{font-family: {c.MONO}; font-size: .95rem}}
    .quote .src {{flex-basis: 100%; font-family: {c.MONO}; font-size: .76rem; color: {muted}; letter-spacing: .02em}}
    .st-key-chart_tabs [data-testid="stButtonGroup"] button {{border-radius: 0 !important; min-height: 40px}}
    .st-key-chart_tabs [data-testid="stButtonGroup"] button p {{font-family: {c.MONO}; font-size: .82rem;
        font-weight: 600; letter-spacing: .02em}}
    .st-key-chart_tabs [data-testid="stButtonGroup"] div {{flex-wrap: wrap !important}}
    .st-key-chart_sub [data-testid="stButtonGroup"] button {{min-height: 32px; border-radius: 0 !important}}
    .st-key-chart_sub [data-testid="stButtonGroup"] button p {{font-family: {c.MONO}; font-size: .76rem}}
    .st-key-model_panel {{background: {panel}; border-radius: 2px !important; border-color: {line} !important}}
    .st-key-model_panel [data-testid="stButtonGroup"] button p {{font-family: {c.MONO}; font-size: .78rem;
        font-weight: 600}}
    .preset-notes {{font-size: .74rem; color: {muted}; line-height: 1.4; margin: -.2rem 0 .5rem}}
    .preset-notes b {{color: {text2}; font-family: {c.MONO}; font-weight: 600; margin-right: 4px}}
    .a1-scale {{display: flex; justify-content: space-between; font-family: {c.MONO}; font-size: .72rem;
        color: {muted}; margin: -.6rem 0 .5rem}}
    .st-key-insight p {{color: {text2}; font-size: .92rem; border-top: 1px dashed {line}; padding-top: .45rem}}
    .st-key-insight p::before {{content: "IN ONE LINE ▸  "; font-family: {c.MONO}; font-size: .74rem;
        font-weight: 700; letter-spacing: .05em; color: {muted}}}
    .set-title {{font-family: {c.MONO}; font-weight: 700; color: {real}; letter-spacing: .06em}}
    [data-testid="stSidebar"] h3 {{font-size: .78rem !important; margin: .9rem 0 .2rem !important}}
    @media (prefers-reduced-motion: no-preference) {{
        .stPlotlyChart {{animation: labfade .18s cubic-bezier(.2,.8,.2,1)}}
        @keyframes labfade {{from {{opacity: .35}} to {{opacity: 1}}}}
    }}
    </style>""")


def page():
    _init()
    _style()
    ss = st.session_state
    live = _settings_head()                  # Settings: the market and Live prices (decides the prices)
    close, r, s, source, name = _load(live)
    symbol = ex.yahoo_symbol(name)
    # the Yahoo symbol whose prices are on screen, or None for the saved files
    fetch = None if source != "live" else (name if ss.asset == ("live", name) else symbol)
    typed = ss.get("typed_symbol")
    spot = (" · showing the saved spot price" if ss.asset == ("saved", name) and name in ex.SAVED_SPOT
            and typed == ex.yahoo_symbol(name) and source == "saved" else "")
    _settings_rest(name)

    st.markdown("<p class='lab-eyebrow'>TRY IT YOURSELF</p><p class='lab-title'>Test program</p>",
                unsafe_allow_html=True)
    tasks_box = st.container()           # filled at the end, once this run's results are known
    # on a phone the model controls come after the chart, far down: a link straight to them
    st.markdown("<p class='phone-only'><a class='jump' href='#model'>Jump to the model controls ↓</a></p>",
                unsafe_allow_html=True)
    if ss.auto_run and _controls_changed():         # a slider moved: run it before anything is drawn
        _run(_control_values())
    with st.spinner("Running the model..."):
        mixes = c.run_mixes(**ss.run)
    sums = ex.mix_summaries(mixes)
    a1 = ss.run["a1"]

    # the chart and the model side by side; on a phone Streamlit stacks columns in order, so the
    # model comes right after the chart, and the tables after both
    main, right = st.columns([3.2, 1.25], gap="large")
    with right:                 # first in the code: a run made here reruns before the chart is drawn
        _model_panel(name, s, sums, a1)
    with main:
        if fetch is None:
            st.markdown(_quote_html(name, close, source, spot), unsafe_allow_html=True)
        else:
            _live_quote(name, close, fetch)
        if fetch is None:
            chart_panel(name, symbol, close, r, mixes, source, live)
        else:
            chart_panel_live(name, symbol, close, r, mixes, source, live, fetch)
        shown = {m: sums[m] for m in _shown_mixes(a1) if sums[m] is not None}
        if shown:
            with st.container(key="insight"):
                st.caption(ex.insight(name, s, ex.closest_mix(s, shown)))

    _write_link(name)
    spx_tail = c.saved_asset("S&P 500")[2]["tail"]
    _tasks(tasks_box, _tasks_done(name, s, sums), spx_tail)

    tabs = st.tabs([t.format(name=ex.display_name(name)) for t in LOW_TABS])
    with tabs[0]:
        st.dataframe(_model_table(name, s, r, sums, mixes, a1), width="stretch",
                     column_config=_model_table_config(name, a1))
        st.caption(f"Model columns: one simulated history (herding strength {a1:.2f}, seed {ss.run['seed']}, "
                   f"{ss.run['days']:,} days). The bell-curve column is as long as the market "
                   f"({s['days']:,} days). For as many days as the simulated history, a bell curve expects "
                   f"{ex.bell_big_days(ss.run['days']):.1f} days beyond 4 sd."
                   + (" With herding strength 0 trend chasers do not herd, so there is no separate herding column."
                      if a1 == 0 else ""))
    with tabs[1]:
        big = ex.big_day_list(r)
        if big.empty:
            st.caption("No day moved more than 4 sd.")
        else:
            _day_reader(r, close)
            t = pd.DataFrame({"Date": big.index.strftime("%Y-%m-%d"), "Move": big["move"].values,
                              "Size (sd)": big["sd"].values})
            # up/down colors go with an arrow
            st.dataframe(t.style.format({"Move": lambda v: f"{'▲' if v > 0 else '▼'} {abs(v):.2%}",
                                         "Size (sd)": "{:+.1f}"})
                         .map(lambda v: f"color: {c.UP if v > 0 else c.DOWN}", subset=["Move"]),
                         hide_index=True, width="stretch", height=300)
            st.caption(f"{len(big)} big days (beyond 4 sd). A bell curve expects "
                       f"{s['bell_big_days']:.1f} in {s['days']:,} days.")
    with tabs[2]:
        if not ss.log:
            st.caption("Move the herding slider (or pick a preset): each simulated history you run is added here.")
        else:
            num = st.column_config.NumberColumn
            st.dataframe(_run_log_view(_run_log_table()), hide_index=True, width="stretch",
                         column_config={**{c.MIX[m][0]: num(f"{c.MIX[m][0]}: tail exponent",
                                                            help="Tail exponent of this trader mix's history. A dash: it "
                                                                 "ran away, or (Trend chasers who herd at herding strength 0) it "
                                                                 "is the same history as Trend chasers, no herding.")
                                           for m in ex.MIXES},
                                        "Herding strength (a1)": num(format="%.2f"),
                                        "Reaction to price trends (a2)": num(format="%.2f"),
                                        "Mood changes (v1)": num(format="%.2f"), "Days": num(format="%d")})
    with tabs[3]:
        _dial_panel(name, s, sums, a1)


page()
