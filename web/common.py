"""Shared look and cached data for every page of the web app (app.py).

Colors mean one thing each, on every page (Okabe-Ito colors, readable with color blindness):
    white          the real market
    blue           fundamentalists only (the bell-curve base case)
    orange         chartists without herding
    pink           chartists who herd (the full model)
    gray, dotted   the bell curve
    green / coral  only gains / losses, always with an arrow
Buttons, sliders and switches are gray, so color on a page always means data.

Smoothness: no line is drawn with more than about 1,500 points (explore.thin_indices keeps every
crash and spike), and controls that only change a chart rerun just that chart (st.fragment).
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from herding import explore as ex

REAL, BELL, UP, DOWN = "#F2F4F8", "#7D8594", "#34D399", "#FF7A6B"
GRID, CARD, MUTED = "#1F2632", "#11151C", "#8B93A3"
MONO = "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace"
MIX = {  # key: (name, one-line meaning, color, line style)
    "fundamentalists": ("Fundamentalists only", "trade on the true value", "#56B4E9", "dot"),
    "chartists": ("Chartists, no herding", "follow the trend alone", "#E69F00", "dash"),
    "herding": ("Chartists who herd", "follow the trend and copy the crowd", "#D98BBE", "solid"),
}
PICK = ["#F2F2F5", "#F0E442", "#4FD1C5", "#FF9F7F", "#A3E635", "#93C5FD"]   # labels for markets you pick
MAX_POINTS = 1500
PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
               "toImageButtonOptions": {"format": "png", "scale": 2}}

HELP = {
    "tail": "How fast big days get rarer. Real markets are near 3: a move twice as big is about "
            "2³ = 8 times rarer. Smaller means fatter tails. A bell curve, measured the same way, comes out near 6.",
    "kurtosis": "Excess kurtosis. 0 for a bell curve; bigger means more very calm and very wild days.",
    "memory": "Correlation of move sizes a month (20 trading days) apart. 0 means no memory; above 0 "
              "means wild days bunch together.",
    "big": "Days that moved more than 4 standard deviations, up or down.",
    "vol": "Annualized volatility: the standard deviation of daily returns in the range, times √252.",
    "dd": "The largest fall from a high point to a later low within the range.",
}


def setup():
    """Page-wide type and spacing. st.html, not st.markdown: markdown read the [kind="primary"]
    selectors as link syntax and silently dropped them."""
    st.html(f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');
    .block-container {{padding-top: 3.2rem; padding-bottom: 4rem; max-width: 1240px}}
    h1 {{font-size: 1.75rem !important; font-weight: 600 !important; margin-bottom: .1rem !important}}
    h2 {{font-size: 1.2rem !important; font-weight: 600 !important; margin-top: .8rem !important}}
    h3 {{font-size: 1.02rem !important; font-weight: 600 !important}}
    p, li {{line-height: 1.6}}
    hr {{margin: 1.6rem 0 !important; border-color: {GRID} !important}}
    [data-testid="stVerticalBlockBorderWrapper"] {{border-radius: 8px; border-color: {GRID} !important}}
    div[data-testid="stMetric"] {{padding: 2px 0}}
    div[data-testid="stMetricLabel"] p {{color: {MUTED}; font-size: .8rem}}
    div[data-testid="stMetricValue"] {{font-family: {MONO}; font-variant-numeric: tabular-nums;
        font-size: 1.3rem !important; font-weight: 500}}
    .num {{font-family: {MONO}; font-variant-numeric: tabular-nums}}
    .muted {{color: {MUTED}; font-size: .86rem}}
    .lead {{color: #C3C8D3; max-width: 720px; margin-bottom: .6rem}}
    .key {{display: flex; flex-wrap: wrap; gap: 16px; margin: .2rem 0 .8rem; font-size: .82rem; color: #B9BFCB}}
    .key span {{display: inline-flex; align-items: center; gap: 6px}}
    .sw {{display: inline-block; width: 20px; height: 0; border-top: 2.5px solid}}
    .stats {{display: flex; flex-wrap: wrap; gap: 6px 26px; margin: .2rem 0 .6rem}}
    .stats div {{min-width: 88px}}
    .stats .l {{color: {MUTED}; font-size: .74rem}}
    .stats .v {{font-family: {MONO}; font-variant-numeric: tabular-nums; font-size: .98rem}}
    .price {{font-family: {MONO}; font-variant-numeric: tabular-nums; font-size: 1.9rem; font-weight: 500}}
    .ok {{color: {UP}; font-weight: 600}} .no {{color: {DOWN}; font-weight: 600}}
    button[kind="primaryFormSubmit"], button[kind="primary"] {{background: #E6E8EE !important;
        color: #0B0E13 !important; border: 1px solid #E6E8EE !important; font-weight: 600}}
    button[kind="primaryFormSubmit"]:hover, button[kind="primary"]:hover {{background: #FFFFFF !important}}
    </style>""")


def color_key(*items):
    """One line saying what each color means (items: 'real', 'bell', 'updown' or a mix key)."""
    parts = []
    for it in items:
        if it == "real":
            parts.append(f"<span><i class='sw' style='border-color:{REAL};border-top-width:3.5px'></i>real market</span>")
        elif it == "bell":
            parts.append(f"<span><i class='sw' style='border-color:{BELL};border-top-style:dotted'></i>bell curve</span>")
        elif it == "updown":
            parts.append(f"<span style='color:{UP}'>▲ up</span><span style='color:{DOWN}'>▼ down</span>")
        else:
            name, _, color, dash = MIX[it]
            style = {"dot": "dotted", "dash": "dashed", "solid": "solid"}[dash]
            parts.append(f"<span><i class='sw' style='border-color:{color};border-top-style:{style}'></i>{name}</span>")
    st.markdown(f"<div class='key'>{''.join(parts)}</div>", unsafe_allow_html=True)


def lead(text):
    st.markdown(f"<p class='lead'>{text}</p>", unsafe_allow_html=True)


def stats_row(items):
    """A row of small label/number pairs: items = [(label, text, color or None, help or None)]."""
    cells = []
    for label, text, color, tip in items:
        style = f" style='color:{color}'" if color else ""
        title = f" title='{tip}'" if tip else ""
        cells.append(f"<div{title}><div class='l'>{label}</div><div class='v'{style}>{text}</div></div>")
    st.markdown(f"<div class='stats'>{''.join(cells)}</div>", unsafe_allow_html=True)


def verdict(passed):
    return "<span class='ok'>passes</span>" if passed else "<span class='no'>does not pass</span>"


def arrow(x, fmt="{:.2%}"):
    color = UP if x >= 0 else DOWN
    return f"<span class='num' style='color:{color}'>{'▲' if x >= 0 else '▼'} {fmt.format(abs(x))}</span>", color


# ------------------------------------------------------------------ cached data
@st.cache_data(show_spinner=False)
def saved_asset(name):
    close = ex.saved_prices(name)
    r = ex.log_returns(close)
    return close, r, ex.summary(r.values)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def live_asset(ticker):
    close = ex.live_prices(ticker)
    r = ex.log_returns(close)
    return close, r, ex.summary(r.values)


@st.cache_data(ttl=300, show_spinner=False)
def intraday(ticker, frame):
    return ex.intraday_prices(ticker, frame)


@st.cache_data(show_spinner=False)
def market_table():
    return ex.market_table()


@st.cache_data(show_spinner=False)
def fitted():
    return ex.fitted_settings()


@st.cache_data(show_spinner=False)
def numbers():
    """The saved results behind the Checks and Story pages (scripts/web_numbers.py)."""
    return ex.web_numbers()


def default_run():
    """The history the site shows unless you change it: the S&P fit, the first held-out seed and an
    S&P-length history, the same history as the project's figures (Fig 1)."""
    p, _ = fitted()
    rec = numbers()["recipe"]
    return dict(a1=float(p["a1"]), a2=float(p["a2"]), v1=float(p["v1"]), days=int(rec["days"]),
                seed=int(rec["seeds"][0]))


@st.cache_data(show_spinner=False, max_entries=60)
def run_mixes(a1, a2, v1, days, seed):
    """{mix: (returns or None, runaway day after burn-in or None)} for the three trader mixes."""
    p, burn = fitted()
    p = dict(p, a2=a2, v1=v1)
    out = {}
    for m in ex.MIXES:
        r, err = ex.run_mix(p, m, a1, days, seed, burn)
        out[m] = (r, None if err is None else int(err.day - burn))
    return out


@st.cache_data(show_spinner=False)
def runaway_path(seed, day):
    p, burn = fitted()
    return ex.runaway_path(p, "chartists", 0.0, seed, int(day) + burn)


@st.cache_data(show_spinner=False)
def sweep_table():
    return pd.DataFrame(numbers()["sweep"]["rows"])


# ------------------------------------------------------------------ charts
def style(fig, height=380, logx=False, logy=False, xtitle="", ytitle="", legend=True):
    """Dark chart with the legend in its own strip above the plot, so it never covers a line."""
    fig.update_layout(template="plotly_dark", height=height, margin=dict(l=8, r=8, t=44 if legend else 12, b=8),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=CARD, hovermode="closest",
                      showlegend=legend, font=dict(size=12), dragmode="zoom", uirevision="keep",
                      hoverlabel=dict(font=dict(family=MONO, size=12)),
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                                  font=dict(size=11), bgcolor="rgba(0,0,0,0)"))
    # "-" lets plotly choose (dates or numbers); forcing "linear" left the date axis empty
    fig.update_xaxes(type="log" if logx else "-", gridcolor=GRID, zeroline=False, title=xtitle)
    fig.update_yaxes(type="log" if logy else "-", gridcolor=GRID, zeroline=False, title=ytitle)
    return fig


def show(fig, key=None):
    # theme=None: the figure is styled here already; Streamlit's theme pass redid it on every rerun
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG, key=key, theme=None)


def mix_line(m, width=2.2):
    _, _, color, dash = MIX[m]
    return dict(color=color, dash=dash, width=width)


def thin(x, y):
    """A long series cut to at most MAX_POINTS points for drawing, keeping every crash and spike."""
    i = ex.thin_indices(np.asarray(y, dtype=float), MAX_POINTS)
    return np.asarray(x)[i], np.asarray(y)[i]


def tail_axes(fig, ymin=-5):
    """Readable labels for the log-log tail charts: percentages up the side, round sizes across
    (plotly's own log labels read '10µ' and crowded the corner)."""
    ys = [10.0 ** -k for k in range(0, -ymin + 1)]
    fig.update_yaxes(tickvals=ys, ticktext=[f"{100 * y:.{max(0, k - 2)}f}%" for k, y in enumerate(ys)],
                     range=[ymin, 0.05])
    xs = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
    fig.update_xaxes(tickvals=xs, ticktext=[f"{x:g}" for x in xs], range=[-1, 1.6])
    return fig


def tail_chart(series, height=380, bell=True, guide=True):
    """Share of days with a move at least this big, on log-log axes. series: [(label, returns, line)]."""
    fig = go.Figure()
    for label, r, line in series:
        x, y = ex.tail_curve(np.asarray(r))
        fig.add_scatter(x=x, y=y, name=label, mode="lines", line=line,
                        hovertemplate=f"{label}<br>%{{x:.1f}} sd or more on %{{y:.2%}} of days<extra></extra>")
    if bell:
        g = np.linspace(0.1, 6, 80)
        fig.add_scatter(x=g, y=[ex.bell_big_days(1, v) for v in g], name="Bell curve", mode="lines",
                        line=dict(color=BELL, dash="dot", width=2), hoverinfo="skip")
    if guide and series:
        x, y = ex.tail_curve(np.asarray(series[0][1]))
        y0 = float(np.interp(3.0, x, y))
        fig.add_scatter(x=[2, 30], y=[y0 * (2 / 3) ** -3, y0 * (30 / 3) ** -3], name="Slope 3",
                        mode="lines", line=dict(color="#5C5E6B", dash="longdash", width=1), hoverinfo="skip")
    style(fig, height, logx=True, logy=True, xtitle="Move size (standard deviations)",
          ytitle="Share of days at least this big")
    return tail_axes(fig)


def dial_chart(market_tail=None, runs=(), current=None, height=300):
    """Tail exponent against herding strength: median and middle half of the saved sweep, the
    market's own tail, and the runs made on this page (runs: [(a1, tail)], current: (a1, tail))."""
    t = sweep_table()
    g = t.dropna(subset=["tail"]).groupby("a1")["tail"]
    med, lo, hi = g.median(), g.quantile(0.25), g.quantile(0.75)
    fig = go.Figure()
    fig.add_scatter(x=list(hi.index) + list(lo.index[::-1]), y=list(hi.values) + list(lo.values[::-1]),
                    fill="toself", fillcolor="rgba(217,139,190,0.15)", line=dict(width=0), hoverinfo="skip",
                    name="Middle half of 10 histories")
    fig.add_scatter(x=med.index, y=med.values, mode="lines+markers", name="Model (median)",
                    line=mix_line("herding"), marker=dict(size=5),
                    hovertemplate="a1 %{x:.2f}: tail %{y:.2f}<extra></extra>")
    if market_tail is not None:
        fig.add_hline(y=market_tail, line=dict(color=REAL, width=1.5),
                      annotation_text="this market", annotation_position="top left",
                      annotation_font=dict(size=11, color=REAL))
    if runs:
        fig.add_scatter(x=[a for a, _ in runs], y=[v for _, v in runs], mode="markers", name="Your runs",
                        marker=dict(size=8, color=CARD, line=dict(color=REAL, width=1.5)),
                        hovertemplate="your run, a1 %{x:.2f}: tail %{y:.2f}<extra></extra>")
    if current is not None and current[1] is not None:
        fig.add_scatter(x=[current[0]], y=[current[1]], mode="markers", name="Shown now",
                        marker=dict(size=11, color=REAL),
                        hovertemplate="shown now, a1 %{x:.2f}: tail %{y:.2f}<extra></extra>")
    return style(fig, height, xtitle="Herding strength a1", ytitle="Tail exponent", legend=False)
