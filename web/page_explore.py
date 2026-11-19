"""Workspace: one market next to the model.

left sidebar  search any ticker, the saved markets, presets for the model
main          price and range numbers, a chart with ranges 1D-All and five views (a fragment: changing
              the view or range reruns only the chart), then big days, model vs market and the run log
right         model controls with Run model, and the herding chart that plots every run you make
"""
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from herding import explore as ex
from web import common as c

VIEWS = ["Price", "Tail", "Histogram", "Bursts", "Memory"]
# one sentence under each chart for first-time visitors: what it shows and what to look for
HINTS = {
    "Price": "The price over the range you picked. Triangles mark days that moved more than 4 standard "
             "deviations; a bell curve says those should almost never happen. The lower panel shows how far "
             "the price sits below its last high.",
    "Tail": "How often big days happen. Read left to right: the further a line stays up, the more often "
            "big moves happen. The dotted bell curve drops off fast; look for the line that follows this market best.",
    "Histogram": "How common each size of daily move is. Compared with the dotted bell curve, real markets have "
                 "more quiet days in the middle and more extreme days at the edges.",
    "Bursts": "Each day's move, measured in standard deviations. Notice that wild days come in clusters, both in "
              "the real market (top) and in the herding model (bottom, when its history did not run away).",
    "Memory": "Does a big move today mean big moves later? Above 0 means yes. Real markets stay above 0 for "
              "months; the bell-curve model drops to 0 at once.",
}
PRESETS = {"S&P 500 fit": None, "Strong herding": 1.6, "Weak herding": 0.6, "No herding": 0.0}
SAVED_BY_SYMBOL = {v: k for k, v in ex.YAHOO.items()}      # ^GSPC -> S&P 500, GC=F -> Gold, ...
NO_INTRADAY_NOTE = "Prices within the day need Live prices on, so this shows the last daily closes."


def _init():
    ss = st.session_state
    ss.setdefault("asset", ("saved", "S&P 500"))
    ss.setdefault("ticker_error", "")
    ss.setdefault("ticker_error_for", None)        # the market on screen when the error was made
    ss.setdefault("list_version", 0)
    ss.setdefault("run", c.default_run())
    ss.setdefault("log", [])
    for k in ("a1", "a2", "v1", "days", "seed"):
        ss.setdefault(f"in_{k}", ss.run[k])
    # the preset picked is the one the controls match now, or none: a radio that kept showing
    # "S&P 500 fit" after a slider moved could not be clicked again to restore the fit
    ss.preset = _matching_preset()


def _matching_preset():
    ss = st.session_state
    p, _ = c.fitted()
    for name, a1 in PRESETS.items():
        want = {"in_a1": float(p["a1"]) if a1 is None else a1, "in_a2": float(p["a2"]), "in_v1": float(p["v1"])}
        if all(abs(float(ss[k]) - v) < 1e-9 for k, v in want.items()):
            return name
    return None


def _pick_row(key, group):
    rows = st.session_state[key].selection.rows
    if rows:
        st.session_state.asset = ("saved", ex.catalog()[group][rows[0]])
        st.session_state.ticker_error = ""          # a market is picked: an old ticker error is stale


def _fetch():
    try:
        t = ex.clean_ticker(st.session_state.ticker_box)
        if t in ex.market_data.stock_tickers():
            st.session_state.asset = ("saved", t)
        elif t in SAVED_BY_SYMBOL:                  # ^GSPC, GC=F, ...: the saved market with that symbol
            st.session_state.asset = ("saved", SAVED_BY_SYMBOL[t])
        else:
            c.live_asset(t)
            st.session_state.asset = ("live", t)
        st.session_state.ticker_error = ""
        st.session_state.list_version += 1          # a new table key clears the old row tick
    except ConnectionError:
        st.session_state.ticker_error = "Couldn't reach Yahoo Finance just now. Pick a saved market."
    except ValueError as e:
        msg = str(e)                                # (str.capitalize would lower-case the ticker)
        st.session_state.ticker_error = msg[:1].upper() + msg[1:]
    st.session_state.ticker_error_for = st.session_state.asset


def _preset():
    p, _ = c.fitted()
    a1 = PRESETS[st.session_state.preset]
    st.session_state.in_a1 = float(p["a1"]) if a1 is None else a1
    st.session_state.in_a2, st.session_state.in_v1 = float(p["a2"]), float(p["v1"])


@st.cache_data(show_spinner=False)
def _watch(group):
    rows = []
    for name in ex.catalog()[group]:
        close, _, s = c.saved_asset(name)
        rows.append({"Market": name, "Tail": s["tail"], "Past year": ex.past_year(close)})
    return pd.DataFrame(rows)


def _sidebar():
    with st.sidebar:
        st.text_input("Search", key="ticker_box", placeholder="Any ticker, e.g. NVDA", on_change=_fetch)
        # shown only while the same market is on screen: picking another one makes the error stale
        if st.session_state.ticker_error and st.session_state.ticker_error_for == st.session_state.asset:
            st.error(st.session_state.ticker_error)
        group = c.pick_one("Markets", ex.catalog(), "Indexes", "group")
        key = f"rows_{group}_{st.session_state.list_version}"
        st.dataframe(_watch(group), key=key, hide_index=True, on_select=lambda: _pick_row(key, group),
                     selection_mode="single-row", width="stretch",
                     height=min(35 * (len(ex.catalog()[group]) + 1) + 3, 460),
                     column_config={"Past year": st.column_config.LineChartColumn(width="small"),
                                    "Tail": st.column_config.NumberColumn(format="%.2f", help=c.HELP["tail"],
                                                                          width="small")})
        st.radio("Model presets", list(PRESETS), index=None, key="preset", on_change=_preset,
                 captions=["the fitted settings", "a1 = 1.6", "a1 = 0.6", "a1 = 0, chartists don't copy"])
        st.caption("A preset fills in the controls. Press Run model to use it.")


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
            ss.ticker_error = f"Live prices for {name} aren't available right now. Showing the S&P 500."
            ss.ticker_error_for = ss.asset
            st.toast(ss.ticker_error)
            return (*c.saved_asset("S&P 500"), "saved", "S&P 500")   # not live again: Yahoo just failed
    if live:
        try:
            return (*c.live_asset(ex.yahoo_symbol(name)), "live", name)
        except (ConnectionError, ValueError):
            st.toast(f"Live prices for {name} aren't available right now, so this shows the saved data.")
    return (*c.saved_asset(name), "saved", name)


def _price_chart(name, cl, r, daily_range, show_big, show_dd, log, intraday, hours=None):
    color = c.UP if cl.iloc[-1] >= cl.iloc[0] else c.DOWN
    rows = 2 if show_dd else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[0.74, 0.26] if show_dd else [1])
    x, y = c.thin(cl.index, cl.values)
    fmt = "%b %d %H:%M" if intraday else "%b %d, %Y"
    fig.add_scatter(x=x, y=y, mode="lines", name=name, line=dict(color=color, width=1.8),
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
        fig.add_scatter(x=x, y=y, mode="lines", name="Drawdown", fill="tozeroy",
                        line=dict(color=c.DOWN, width=1), fillcolor=c.DOWN_FILL,
                        hovertemplate=f"%{{x|{fmt}}}<br>%{{y:.1%}} below the high<extra></extra>", row=2, col=1)
    c.style(fig, height=470)
    fig.update_yaxes(type="log" if log else "-", title="Price", row=1, col=1)
    if show_dd:
        fig.update_yaxes(tickformat=".0%", title="Drawdown", row=2, col=1)
    if intraday:            # hide weekends and nights, but only for markets closed then (crypto never is)
        hours = hours or {"nights": False, "weekends": False}
        breaks = ([] if hours["weekends"] else [dict(bounds=["sat", "mon"])]) + \
                 ([] if hours["nights"] else [dict(bounds=[16, 9.5], pattern="hour")])
        if breaks:
            fig.update_xaxes(rangebreaks=breaks)
    return fig


def _model_chart(view, name, r, mixes, earlier, a1):
    series = [(name, r.values, dict(color=c.REAL, width=3))] + \
             [(c.MIX[m][0], mixes[m][0], c.mix_line(m)) for m in _shown_mixes(a1) if mixes[m][0] is not None] + \
             [(label, rr, dict(color=c.MIX["herding"][2], width=1, dash="dot")) for label, rr in earlier]
    if view == "Tail":
        return c.tail_chart(series, height=470)
    fig = go.Figure()
    if view == "Histogram":
        for label, rr, line in series:
            cx, d = ex.histogram(rr, empty=np.nan)        # NaN: an empty bin can't be drawn on a log scale
            fig.add_scatter(x=cx, y=d, name=label, mode="lines", line=line)
        g = np.linspace(-8, 8, 200)
        fig.add_scatter(x=g, y=ex.bell_density(g), name="Bell curve", mode="lines",
                        line=dict(color=c.BELL, dash="dot"))
        fig.update_yaxes(range=[-4.5, 0])
        return c.style(fig, 470, logy=True, xtitle="Daily move (standard deviations)", ytitle="How common (log)")
    for label, rr, line in series:
        lx, cc = ex.memory_curve(rr)
        fig.add_scatter(x=lx, y=cc, name=label, mode="lines+markers", line=line, marker=dict(size=5))
    fig.add_hline(y=0, line=dict(color=c.BELL, dash="dot"))
    return c.style(fig, 470, logx=True, xtitle="Days apart", ytitle="Correlation of move sizes")


def _bursts_chart(name, rr, r, herd, herd_label, herd_note):
    """Daily moves in standard deviations: the market's (of all its days) on top, the herding
    model's below, or herd_note in the empty panel when its history ran away."""
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
    """The trader mixes worth drawing: with a1 = 0 "chartists who herd" do not herd, so their history
    is the same as "chartists, no herding" and a second, identical line would only confuse."""
    return [m for m in ex.MIXES if not (m == "herding" and a1 == 0)]


def _missing_note(mixes, a1):
    """One sentence for each trader mix the charts leave out, so the color key never names a line
    that silently isn't there."""
    notes = []
    for m in ex.MIXES:
        if m not in _shown_mixes(a1):
            notes.append(f"With a1 = 0 chartists don't herd, so \"{c.MIX[m][0]}\" is the same history as "
                         f"\"{c.MIX['chartists'][0]}\" and is drawn once.")
        elif mixes[m][0] is None:
            notes.append(f"\"{c.MIX[m][0]}\" {c.gone_text(mixes[m][1])}, so it has no line.")
    return " ".join(notes)


def _earlier_runs():
    """Up to four of your earlier runs that differ from the one shown now, newest first."""
    ss = st.session_state
    return [prm for _, prm, _ in ss.log if prm != ss.run][:4]


def _earlier_label(prm, now):
    label = f"Earlier run: a1 {prm['a1']:.2f}, seed {prm['seed']}"
    extra = [f"{k} {prm[k]:.2f}" for k in ("a2", "v1") if prm[k] != now[k]]
    if prm["days"] != now["days"]:
        extra.append(f"{prm['days']:,} days")
    return label + (", " + ", ".join(extra) if extra else "")


@st.fragment
def chart_panel(name, symbol, close, r, mixes, source):
    """Everything that only changes the chart. A fragment, so these clicks rerun just this part."""
    ss = st.session_state
    a1 = ss.run["a1"]
    # one row each: side by side, the two button rows were cut off on narrow screens
    view = c.pick_one("View", VIEWS, "Price", "view", label_visibility="collapsed")
    frame = c.pick_one("Range", ex.FRAMES, "All", "frame", label_visibility="collapsed",
                       disabled=view not in ("Price", "Bursts"))
    missing = _missing_note(mixes, a1)
    if view in ("Price", "Bursts"):
        bursts_note = ""
        if view == "Bursts" and frame in ex.DAILY_FALLBACK:  # bursts are daily moves: a day or a week is too short
            frame, bursts_note = "1Y", "Bursts use daily moves, so 1D and 7D show the past year."
        bars = None
        # prices within the day come only from Yahoo, so they are fetched only when the page shows
        # Yahoo's prices: with saved data, the header (saved close) and the chart (today) disagreed
        if frame in ex.DAILY_FALLBACK and source == "live":
            try:
                bars = c.intraday(symbol, frame)
            except (ConnectionError, ValueError):
                bars = None
        cl, note = ex.price_range(close, frame, bars)
        intraday = frame in ex.DAILY_FALLBACK and not note
        if frame in ex.DAILY_FALLBACK and source != "live":
            note = NO_INTRADAY_NOTE
        if intraday:            # close is Yahoo's daily prices here (source is live): one source on screen
            note = (f"Prices within the day from Yahoo Finance, New York time, up to "
                    f"{cl.index[-1]:%d %b %H:%M}. The 52-week numbers use Yahoo's daily prices to "
                    f"{close.index[-1]:%d %b %Y}.")
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
                     ("Volatility (yr)", "–" if np.isnan(vol) else f"{vol:.1%}", None, c.HELP["vol"]),
                     ("Max drawdown", f"{st_['drawdown']:.1%}", c.DOWN if st_["drawdown"] > 0 else None, c.HELP["dd"])])
        if view == "Price":
            t1, t2, t3 = st.columns(3)
            show_big = t1.toggle("Big days", value=True, help=c.HELP["big"], disabled=intraday, key="t_big")
            show_dd = t2.toggle("Drawdown", value=True, help=c.HELP["dd"], key="t_dd")
            # fixed key and default: a default that changed with the range made Streamlit rebuild
            # the switch (and the chart after it) on every new range
            log = t3.toggle("Log scale", value=True, key="t_log")
            c.show(_price_chart(name, cl, r, daily_range, show_big, show_dd, log, intraday,
                                hours=ex.trading_hours(symbol, bars)), key="price_chart")
            c.hint(HINTS["Price"])
        else:
            rr = ex.moves_in_range(r, daily_range)    # n prices, n - 1 moves: none from before the range
            herd_label = c.MIX["herding"][0] if a1 != 0 else f"{c.MIX['chartists'][0]} (a1 = 0)"
            herd_note = (f"{c.MIX['herding'][0]} {c.gone_text(mixes['herding'][1])}, so there is nothing to show"
                         if mixes["herding"][0] is None else "")
            c.show(_bursts_chart(name, rr if len(rr) > 1 else r, r, mixes["herding"][0], herd_label, herd_note),
                   key="bursts_chart")
            c.hint(HINTS["Bursts"])
        if note or bursts_note:
            st.caption(note or bursts_note)
    else:
        earlier = []
        others = _earlier_runs()
        if others and st.toggle("Show my earlier runs", value=False, key="t_earlier",
                                help="Adds the herding line of each of your last four other runs, dotted."):
            gone = 0
            for prm in others:
                rr = c.run_mixes(**prm)["herding"][0]
                if rr is None:
                    gone += 1
                else:
                    earlier.append((_earlier_label(prm, ss.run), rr))
            if gone:
                missing += f" {gone} of your earlier runs ran away, so they have no line."
        c.show(_model_chart(view, name, r, mixes, earlier, a1), key=f"model_chart_{view}")
        c.hint(HINTS[view])
        if missing:
            st.caption(missing.strip())


def _model_table(name, s, r, sums, mixes, a1):
    """The Model vs market table: numbers from explore.mix_table, formatted for reading."""
    t = ex.mix_table(name, s, r, sums, mixes)
    rows = ["Days", "Tail exponent", "Excess kurtosis", "Volatility memory (20 days)",
            "Worst day (sd)", "Days beyond 4 sd", "Max drawdown"]
    fmt = ["{:,.0f}", "{:.2f}", "{:.2f}", "{:.2f}", "{:.1f}", "{:,.0f}", "{:.1%}"]

    def col(row):
        if not np.isnan(row["runaway_day"]):
            return [c.gone_text(row["runaway_day"])] + [""] * (len(rows) - 1)
        return [f.format(row[k]) for f, k in zip(fmt, ex.TABLE_MEASURES[:7])]
    tab = {name: col(t.loc[name])}
    for m in _shown_mixes(a1):
        tab[c.MIX[m][0]] = col(t.loc[m])
    b = t.loc["bell"]
    tab["Bell curve (market length)"] = [f"{b['days']:,.0f}", "none", f"{b['kurtosis']:.2f}", f"{b['memory']:.2f}",
                                         "", f"{b['big_days']:.1f}", ""]
    return pd.DataFrame(tab, index=rows)


def _log_run(new, res):
    """Put a run at the top of the run log. A run already in the log moves up instead of being
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
        row = {"Time": t, "a1": prm["a1"], "a2": prm["a2"], "v1": prm["v1"], "Days": prm["days"],
               "Seed": prm["seed"]}
        row.update({c.MIX[m][0]: (np.nan if v is None else float(v)) for m, v in tl.items()})
        row["Ran away"] = ", ".join(c.MIX[m][0] for m, v in tl.items() if v is None)
        out.append(row)
    return pd.DataFrame(out)


def page():
    _init()
    _sidebar()
    ss = st.session_state
    source, name = ss.asset
    st.page_link("web/page_guide.py", label="New here? How to use this page", icon=":material/help:")

    head_l, head_r = st.columns([3, 2], vertical_alignment="bottom")
    live = head_r.toggle("Live prices", value=False,
                         help="On: today's prices from Yahoo Finance. Off: the saved files, which end 30 October 2026.")
    close, r, s, source, name = _load(live)
    symbol = ex.yahoo_symbol(name)
    chg, _ = c.arrow(ex.day_change(close))
    head_l.markdown(f"<div class='muted'>{ex.asset_kind(name)} · "
                    f"{'live from Yahoo' if source == 'live' else 'saved data'} · last close "
                    f"{close.index[-1]:%d %b %Y}</div><h1 style='margin:0'>{name}</h1>"
                    f"<span class='price'>{ex.price_text(close.iloc[-1])}</span> &nbsp;{chg} "
                    f"<span class='muted'>on the day</span>", unsafe_allow_html=True)

    with st.spinner("Running the model..."):
        mixes = c.run_mixes(**ss.run)
    sums = ex.mix_summaries(mixes)
    a1 = ss.run["a1"]

    main, right = st.columns([3.2, 1.25], gap="large")

    with right:
        st.subheader("Model", anchor=False)
        with st.form("controls", border=False):
            st.slider("Herding strength a1", 0.0, 2.0, key="in_a1", step=0.01, format="%.2f",
                      help="How strongly chartists copy the majority mood.")
            st.slider("Trend following a2", 0.0, 1.0, key="in_a2", step=0.01, format="%.2f",
                      help="How strongly chartists react to the price trend.")
            st.slider("Mood changes v1", 0.5, 5.0, key="in_v1", step=0.01, format="%.2f",
                      help="How often chartists reconsider their mood.")
            a, b = st.columns(2)
            sp_days = c.default_run()["days"]
            a.selectbox("Days", [5000, 10000, sp_days], key="in_days",
                        format_func=lambda d: f"{d:,}" + (" (S&P)" if d == sp_days else ""))
            b.number_input("Seed", 0, 10_000, key="in_seed", help="Which random history.")
            go_run = st.form_submit_button("Run model", type="primary", width="stretch")
        if go_run:
            new = {"a1": float(ss.in_a1), "a2": float(ss.in_a2), "v1": float(ss.in_v1),
                   "days": int(ss.in_days), "seed": int(ss.in_seed)}
            with st.spinner("Running..."):
                res = c.run_mixes(**new)
            _log_run(new, res)
            ss.run = new
            st.rerun()

        p, _ = c.fitted()
        runs, earlier_differ = ex.dial_runs(ss.log, p)
        cur = sums["herding"]["tail"] if sums["herding"] else None
        c.show(c.dial_chart(s["tail"], runs, (a1, cur), height=300), key="dial")
        n_seeds = len(c.numbers()["sweep"]["seeds"])
        st.caption(f"Pink: the tail exponent at each herding strength, the median of {n_seeds} saved histories "
                   "(herding strengths where most of them ran away are left out). Solid dot: the run shown now; "
                   "hollow circles: your earlier runs."
                   + (" Your a2 or v1 differs from the fit, so your runs aren't directly comparable to "
                      "the pink line." if ex.differs_from_fit(ss.run, p) else
                      " Some earlier runs used an a2 or v1 different from the fit, so those hollow circles "
                      "aren't directly comparable to the pink line." if earlier_differ else ""))

    with main:
        c.color_key("real", "bell", "fundamentalists", "chartists", "herding")
        chart_panel(name, symbol, close, r, mixes, source)
        shown = {m: sums[m] for m in _shown_mixes(a1) if sums[m] is not None}
        if shown:
            st.caption(ex.insight(name, s, ex.closest_mix(s, shown)))

        tabs = st.tabs(["Model vs market", "Big days", "Run log"])
        with tabs[0]:
            st.dataframe(_model_table(name, s, r, sums, mixes, a1), width="stretch")
            st.caption(f"Model columns: one simulated history (a1 {a1:.2f}, seed {ss.run['seed']}, "
                       f"{ss.run['days']:,} days). The bell-curve column is as long as the market "
                       f"({s['days']:,} days); as long as the model run it expects "
                       f"{ex.bell_big_days(ss.run['days']):.1f} days beyond 4 sd."
                       + (" With a1 = 0 chartists don't herd, so there is no separate herding column."
                          if a1 == 0 else ""))
        with tabs[1]:
            big = ex.big_day_list(r)
            if big.empty:
                st.caption("No day moved more than 4 standard deviations.")
            else:
                t = pd.DataFrame({"Date": big.index.strftime("%Y-%m-%d"), "Move": big["move"].values,
                                  "Standard deviations": big["sd"].values})
                st.dataframe(t.style.format({"Move": "{:+.2%}", "Standard deviations": "{:+.1f}"})
                             .map(lambda v: f"color: {c.UP if v > 0 else c.DOWN}", subset=["Move"]),
                             hide_index=True, width="stretch", height=300)
                st.caption(f"{len(big)} days beyond 4 standard deviations. A bell curve expects "
                           f"{s['bell_big_days']:.1f} in {s['days']:,} days.")
        with tabs[2]:
            if not ss.log:
                st.caption("Press Run model to start the log.")
            else:
                num = st.column_config.NumberColumn
                st.dataframe(_run_log_table(), hide_index=True, width="stretch",
                             column_config={**{c.MIX[m][0]: num(format="%.2f", help="Tail exponent")
                                               for m in ex.MIXES},
                                            "a1": num(format="%.2f"), "a2": num(format="%.2f"),
                                            "v1": num(format="%.2f"), "Days": num(format="%d")})


page()
