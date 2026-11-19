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

# Yahoo symbols for the saved markets (stocks use their own ticker)
YAHOO = {"S&P 500": "^GSPC", "Nasdaq-100": "^NDX", "Dow Jones": "^DJI", "Brent oil": "BZ=F",
         "Gold": "GC=F", "Copper": "HG=F"}
FRAMES = ["1D", "7D", "YTD", "1Y", "5Y", "20Y", "All"]
DAYS = {"1Y": 252, "5Y": 1260, "20Y": 5040}
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
              "the real market (top) and in the herding model (bottom).",
    "Memory": "Does a big move today mean big moves later? Above 0 means yes. Real markets stay above 0 for "
              "months; the bell-curve model drops to 0 at once.",
}
PRESETS = {"S&P 500 fit": None, "Strong herding": 1.6, "Weak herding": 0.6, "No herding": 0.0}


def _init():
    ss = st.session_state
    ss.setdefault("asset", ("saved", "S&P 500"))
    ss.setdefault("ticker_error", "")
    ss.setdefault("list_version", 0)
    ss.setdefault("run", c.default_run())
    ss.setdefault("log", [])
    for k in ("a1", "a2", "v1", "days", "seed"):
        ss.setdefault(f"in_{k}", ss.run[k])


def _pick_row(key, group):
    rows = st.session_state[key].selection.rows
    if rows:
        st.session_state.asset = ("saved", ex.catalog()[group][rows[0]])


def _fetch():
    try:
        t = ex.clean_ticker(st.session_state.ticker_box)
        if t in ex.market_data.stock_tickers():
            st.session_state.asset = ("saved", t)
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
        year = close.iloc[-252:]
        rows.append({"Market": name, "Tail": s["tail"], "Past year": year.iloc[::5].round(4).tolist()})
    return pd.DataFrame(rows)


def _sidebar():
    with st.sidebar:
        st.text_input("Search", key="ticker_box", placeholder="Any ticker, e.g. NVDA", on_change=_fetch)
        if st.session_state.ticker_error:
            st.error(st.session_state.ticker_error)
        group = st.segmented_control("Markets", list(ex.catalog()), default="Index funds", key="group") \
            or "Index funds"
        key = f"rows_{group}_{st.session_state.list_version}"
        st.dataframe(_watch(group), key=key, hide_index=True, on_select=lambda: _pick_row(key, group),
                     selection_mode="single-row", width="stretch",
                     height=min(35 * (len(ex.catalog()[group]) + 1) + 3, 460),
                     column_config={"Past year": st.column_config.LineChartColumn(width="small"),
                                    "Tail": st.column_config.NumberColumn(format="%.2f", help=c.HELP["tail"],
                                                                          width="small")})
        st.radio("Model presets", list(PRESETS), key="preset", on_change=_preset,
                 captions=["the fitted settings", "a1 = 1.6", "a1 = 0.6", "a1 = 0, chartists don't copy"])
        st.caption("A preset fills in the controls. Press Run model to use it.")


def _load(live):
    source, name = st.session_state.asset
    if source == "live":
        return (*c.live_asset(name), "live", name)
    if live:
        try:
            return (*c.live_asset(YAHOO.get(name, name)), "live", name)
        except (ConnectionError, ValueError):
            st.toast(f"Live prices for {name} aren't available right now, so this shows the saved data.")
    return (*c.saved_asset(name), "saved", name)


def _kind(source, name):
    if name in ex.INDEXES:
        return "Index fund"
    if name in ex.COMMODITIES:
        return "Commodity"
    return "Stock"


def _range(close, frame, symbol):
    """(prices in the chosen range, note) — 1D and 7D try Yahoo's intraday bars first."""
    if frame in ("1D", "7D"):
        try:
            return c.intraday(symbol, frame), ""
        except (ConnectionError, ValueError):
            n = 2 if frame == "1D" else 6
            return close.iloc[-n:], "Intraday prices aren't available right now, so this shows daily closes."
    if frame == "YTD":
        return close[close.index >= pd.Timestamp(close.index[-1].year, 1, 1)], ""
    if frame == "All":
        return close, ""
    return close.iloc[-(DAYS[frame] + 1):], ""


def _price_chart(name, cl, r, daily_range, show_big, show_dd, log, intraday, futures=False):
    color = c.UP if cl.iloc[-1] >= cl.iloc[0] else c.DOWN
    rows = 2 if show_dd else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[0.74, 0.26] if show_dd else [1])
    x, y = c.thin(cl.index, cl.values)
    fmt = "%b %d %H:%M" if intraday else "%b %d, %Y"
    fig.add_scatter(x=x, y=y, mode="lines", name=name, line=dict(color=color, width=1.8),
                    hovertemplate=f"%{{x|{fmt}}}<br>%{{y:,.2f}}<extra></extra>", row=1, col=1)
    if show_big and not intraday:
        big = ex.big_day_list(r)
        big = big[big.index.isin(daily_range.index)]
        for up, sym, col in ((True, "triangle-up", c.UP), (False, "triangle-down", c.DOWN)):
            b = big[(big["move"] > 0) == up]
            fig.add_scatter(x=b.index, y=daily_range[b.index].values, mode="markers",
                            name=f"Big {'rise' if up else 'fall'} (> 4 sd)",
                            marker=dict(symbol=sym, size=10, color=col, line=dict(color=c.BG, width=1)),
                            customdata=np.c_[b["move"], b["sd"]],
                            hovertemplate="%{x|%b %d, %Y}<br>%{customdata[0]:+.1%} (%{customdata[1]:+.1f} sd)<extra></extra>",
                            row=1, col=1)
    if show_dd:
        dd = cl / cl.cummax() - 1
        x, y = c.thin(dd.index, dd.values)
        fig.add_scatter(x=x, y=y, mode="lines", name="Drawdown", fill="tozeroy",
                        line=dict(color=c.DOWN, width=1), fillcolor=c.DOWN_FILL,
                        hovertemplate=f"%{{x|{fmt}}}<br>%{{y:.1%}} below the high<extra></extra>", row=2, col=1)
    c.style(fig, height=470)
    fig.update_yaxes(type="log" if log else "-", title="Price", row=1, col=1)
    if show_dd:
        fig.update_yaxes(tickformat=".0%", title="Drawdown", row=2, col=1)
    if intraday:            # hide weekends, and nights for markets that close (futures trade almost 24 h)
        breaks = [dict(bounds=["sat", "mon"])] + ([] if futures else [dict(bounds=[16, 9.5], pattern="hour")])
        fig.update_xaxes(rangebreaks=breaks)
    return fig


def _model_chart(view, name, r, mixes, earlier):
    series = [(name, r.values, dict(color=c.REAL, width=3))] + \
             [(c.MIX[m][0], mixes[m][0], c.mix_line(m)) for m in ex.MIXES if mixes[m][0] is not None] + \
             [(f"Earlier run: a1 {a1:.2f}, seed {sd}", rr, dict(color=c.MIX["herding"][2], width=1, dash="dot"))
              for a1, sd, rr in earlier]
    if view == "Tail":
        return c.tail_chart(series, height=470)
    fig = go.Figure()
    if view == "Histogram":
        for label, rr, line in series:
            cx, d = ex.histogram(rr)
            fig.add_scatter(x=cx, y=np.where(d > 0, d, np.nan), name=label, mode="lines", line=line)
        g = np.linspace(-8, 8, 200)
        fig.add_scatter(x=g, y=np.exp(-g ** 2 / 2) / np.sqrt(2 * np.pi), name="Bell curve", mode="lines",
                        line=dict(color=c.BELL, dash="dot"))
        fig.update_yaxes(range=[-4.5, 0])
        return c.style(fig, 470, logy=True, xtitle="Daily move (standard deviations)", ytitle="How common (log)")
    for label, rr, line in series:
        lx, cc = ex.memory_curve(rr)
        fig.add_scatter(x=lx, y=cc, name=label, mode="lines+markers", line=line, marker=dict(size=5))
    fig.add_hline(y=0, line=dict(color=c.BELL, dash="dot"))
    return c.style(fig, 470, logx=True, xtitle="Days apart", ytitle="Correlation of move sizes")


def _bursts_chart(name, rr, r, herd):
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.1)
    z = (rr - r.mean()) / r.std()
    x, y = c.thin(z.index, z.values)
    fig.add_scatter(x=x, y=y, mode="lines", line=dict(color=c.REAL, width=1), name=f"{name}, real days",
                    hovertemplate="%{x|%b %d, %Y}: %{y:+.1f} sd<extra></extra>", row=1, col=1)
    if herd is not None:
        hz = (herd - herd.mean()) / herd.std()
        x, y = c.thin(np.arange(len(hz)), hz)
        fig.add_scatter(x=x, y=y, mode="lines", line=dict(color=c.MIX["herding"][2], width=1),
                        name="Herding model, simulated days",
                        hovertemplate="day %{x}: %{y:+.1f} sd<extra></extra>", row=2, col=1)
    for i in (1, 2):
        for lim in (-4, 4):
            fig.add_hline(y=lim, line=dict(color=c.DOWN, dash="dot", width=1), row=i, col=1)
    c.style(fig, 470)
    fig.update_yaxes(title="Move (sd)")
    return fig


@st.fragment
def chart_panel(name, symbol, close, r, mixes):
    """Everything that only changes the chart. A fragment, so these clicks rerun just this part."""
    # one row each: side by side, the two button rows were cut off on narrow screens
    view = st.segmented_control("View", VIEWS, default="Price", key="view", label_visibility="collapsed") \
        or "Price"
    frame = st.segmented_control("Range", FRAMES, default="All", key="frame", label_visibility="collapsed",
                                 disabled=view not in ("Price", "Bursts")) or "All"
    if view in ("Price", "Bursts"):
        bursts_note = ""
        if view == "Bursts" and frame in ("1D", "7D"):      # bursts are daily moves: a day or a week is too short
            frame, bursts_note = "1Y", "Bursts use daily moves, so 1D and 7D show the past year."
        cl, note = _range(close, frame, symbol)
        intraday = frame in ("1D", "7D") and not note
        daily = close
        if intraday:            # keep every number on one source: live daily prices next to live intraday ones
            try:
                daily = c.live_asset(symbol)[0]
            except (ConnectionError, ValueError):
                pass
            note = (f"Prices within the day from Yahoo Finance, New York time, up to "
                    f"{cl.index[-1]:%d %b %H:%M}. The 52-week numbers use Yahoo's daily prices to "
                    f"{daily.index[-1]:%d %b %Y}.")
        daily_range = daily[daily.index >= cl.index[0].normalize()] if intraday else cl
        st_ = ex.range_stats(cl, daily=daily)
        chg, col = c.arrow(st_["change"])
        vol = st_["volatility"]
        ch = abs(st_["change"])
        c.stats_row([("Change", f"{'▲' if st_['change'] >= 0 else '▼'} {ch:,.0%}" if ch >= 10 else
                      f"{'▲' if st_['change'] >= 0 else '▼'} {ch:.2%}", col, None),
                     ("High", f"{st_['high']:,.2f}", None, None), ("Low", f"{st_['low']:,.2f}", None, None),
                     ("52-week high", f"{st_['high_52w']:,.2f}", None, None),
                     ("52-week low", f"{st_['low_52w']:,.2f}", None, None),
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
                                futures=symbol.endswith("=F")), key="price_chart")
            c.hint(HINTS["Price"])
        else:
            rr = r[r.index >= daily_range.index[0]] if len(daily_range) > 1 else r
            c.show(_bursts_chart(name, rr if len(rr) > 1 else r, r, mixes["herding"][0]), key="bursts_chart")
            c.hint(HINTS["Bursts"])
        if note or bursts_note:
            st.caption(note or bursts_note)
    else:
        ss = st.session_state
        earlier = []
        if ss.log and st.toggle("Show my earlier runs", value=False, key="t_earlier",
                                help="Adds the herding line of each of your last four runs, dotted."):
            for _, prm, _ in ss.log[:4]:
                if prm != ss.run:
                    rr = c.run_mixes(**prm)["herding"][0]
                    if rr is not None:
                        earlier.append((prm["a1"], prm["seed"], rr))
        c.show(_model_chart(view, name, r, mixes, earlier), key=f"model_chart_{view}")
        c.hint(HINTS[view])


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
    symbol = name if source == "live" and name not in YAHOO else YAHOO.get(name, name)
    day = close.iloc[-1] / close.iloc[-2] - 1
    chg, _ = c.arrow(day)
    head_l.markdown(f"<div class='muted'>{_kind(source, name)} · "
                    f"{'live from Yahoo' if source == 'live' else 'saved data'} · last close "
                    f"{close.index[-1]:%d %b %Y}</div><h1 style='margin:0'>{name}</h1>"
                    f"<span class='price'>{close.iloc[-1]:,.2f}</span> &nbsp;{chg} "
                    f"<span class='muted'>on the day</span>", unsafe_allow_html=True)

    with st.spinner("Running the model..."):
        mixes = c.run_mixes(**ss.run)
    sums = {m: (ex.summary(rv) if rv is not None else None) for m, (rv, _) in mixes.items()}

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
            ss.log.insert(0, (datetime.now().strftime("%H:%M:%S"), new,
                              {m: (ex.summary(rv)["tail"] if rv is not None else None) for m, (rv, _) in res.items()}))
            ss.run = new
            st.rerun()

        p, _ = c.fitted()
        runs = [(prm["a1"], tl["herding"]) for _, prm, tl in ss.log if tl["herding"] is not None]
        cur = sums["herding"]["tail"] if sums["herding"] else None
        c.show(c.dial_chart(s["tail"], runs, (ss.run["a1"], cur), height=300), key="dial")
        changed = abs(ss.run["a2"] - p["a2"]) > 1e-9 or abs(ss.run["v1"] - p["v1"]) > 1e-9
        st.caption("Pink: the tail exponent at each herding strength, from 10 saved histories. White dot: "
                   "the run shown now; circles: your earlier runs."
                   + (" Your a2 or v1 differs from the fit, so your runs aren't directly comparable to "
                      "the pink line." if changed else ""))

    with main:
        c.color_key("real", "bell", "fundamentalists", "chartists", "herding")
        chart_panel(name, symbol, close, r, mixes)
        if any(sums.values()):
            st.caption(ex.insight(name, s, ex.closest_mix(s, sums)))

        tabs = st.tabs(["Model vs market", "Big days", "Run log"])
        with tabs[0]:
            rows = ["Days", "Tail exponent", "Excess kurtosis", "Volatility memory (20 days)",
                    "Worst day (sd)", "Days beyond 4 sd", "Max drawdown"]

            def col(sm, rr, gone):
                if sm is None:
                    return [f"ran away on day {gone:,}"] + [""] * (len(rows) - 1)
                return [f"{sm['days']:,}", f"{sm['tail']:.2f}", f"{sm['kurtosis']:.2f}", f"{sm['memory']:.2f}",
                        f"{sm['worst']:.1f}", f"{sm['big_days']:,}", f"{ex.max_drawdown(rr):.1%}"]
            tab = {name: col(s, r.values, None)}
            for m in ex.MIXES:
                tab[c.MIX[m][0]] = col(sums[m], mixes[m][0], mixes[m][1])
            tab["Bell curve"] = ["", "none", "0.00", "0.00", "", f"{s['bell_big_days']:.1f}", ""]
            st.dataframe(pd.DataFrame(tab, index=rows), width="stretch")
            st.caption(f"Model columns: one simulated history (a1 {ss.run['a1']:.2f}, seed {ss.run['seed']}, "
                       f"{ss.run['days']:,} days).")
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
                st.dataframe(pd.DataFrame([{"Time": t, "a1": prm["a1"], "a2": prm["a2"], "v1": prm["v1"],
                                            "Days": prm["days"], "Seed": prm["seed"],
                                            **{c.MIX[m][0]: ("ran away" if v is None else round(v, 2))
                                               for m, v in tl.items()}}
                                           for t, prm, tl in ss.log]), hide_index=True, width="stretch")


page()
