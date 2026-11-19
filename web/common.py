"""Shared look and cached data for every page of the web app (app.py).

Light and dark mode. Streamlit itself recolors the page, tables and controls from the two color
sets in .streamlit/config.toml. The button at the top right (mode_button) saves the choice where
Streamlit keeps it and reloads. Everything drawn here (charts, the small text styles) takes its
colors from PALETTES for the current visitor's mode: pages write c.REAL, c.MIX, ... and the module
__getattr__ below answers for that visitor, so two visitors in different modes never mix.

Colors mean one thing each, in both modes (Okabe-Ito colors, readable with color blindness):
    REAL           the real market (white in dark mode, ink in light mode)
    MIX            fundamentalists blue, chartists orange, herding pink
    BELL           the bell curve, gray and dotted
    UP / DOWN      only gains / losses, always with an arrow
Buttons, sliders and switches are gray, so color on a page always means data.

Smoothness: no line is drawn with more than about 1,500 points (explore.thin_indices keeps every
crash and spike), and controls that only change a chart rerun just that chart (st.fragment).
"""
import json

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from herding import explore as ex

MONO = "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace"
_NAMES = {"fundamentalists": ("Fundamentalists only", "trade on the true value", "dot"),
          "chartists": ("Chartists, no herding", "follow the trend alone", "dash"),
          "herding": ("Chartists who herd", "follow the trend and copy the crowd", "solid")}
PALETTES = {
    "dark": dict(REAL="#F2F4F8", BELL="#7D8594", UP="#34D399", DOWN="#FF7A6B", GRID="#1F2632",
                 CARD="#11151C", MUTED="#8B93A3", BG="#0B0E13", TEXT2="#C3C8D3", GUIDE="#5C5E6B",
                 SHADE_MID="#9AA3B5", SHADE_LOW="#5C6577", FAINT="rgba(255,255,255,0.04)",
                 OK_FILL="rgba(52,211,153,0.08)", DOWN_FILL="rgba(255,122,107,0.15)",
                 UP_FILL="rgba(52,211,153,0.10)", HERD_FILL="rgba(217,139,190,0.15)",
                 TEMPLATE="plotly_dark", BUTTON="#E6E8EE", BUTTON_TEXT="#0B0E13",
                 MIXC=dict(fundamentalists="#56B4E9", chartists="#E69F00", herding="#D98BBE"),
                 PICK=["#F2F2F5", "#F0E442", "#4FD1C5", "#FF9F7F", "#A3E635", "#93C5FD"]),
    "light": dict(REAL="#111827", BELL="#6B7280", UP="#059669", DOWN="#D9412F", GRID="#E5E7EB",
                  CARD="#FFFFFF", MUTED="#5F6673", BG="#FFFFFF", TEXT2="#374151", GUIDE="#9CA3AF",
                  SHADE_MID="#6B7280", SHADE_LOW="#C4C9D2", FAINT="rgba(17,24,39,0.04)",
                  OK_FILL="rgba(5,150,105,0.08)", DOWN_FILL="rgba(217,65,47,0.12)",
                  UP_FILL="rgba(5,150,105,0.10)", HERD_FILL="rgba(185,71,143,0.12)",
                  TEMPLATE="plotly_white", BUTTON="#111827", BUTTON_TEXT="#FFFFFF",
                  MIXC=dict(fundamentalists="#0072B2", chartists="#C77700", herding="#B9478F"),
                  PICK=["#111827", "#B58900", "#0F8B8D", "#D55E00", "#4D7C0F", "#2563EB"]),
}
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
    "vol": "Annualized volatility: the standard deviation of daily returns in the range, times √252 (√365 for a market open every day, like crypto).",
    "dd": "The largest fall from a high point to a later low within the range.",
}


# ------------------------------------------------------------------ light / dark mode
def mode():
    """'light' or 'dark' for this visitor: what the browser reports, else the mode the button just
    set (?mode= in the address), else the last mode seen in this session, else dark."""
    ss = st.session_state
    try:
        t = st.context.theme.type
    except Exception:                       # no browser (tests, scripts)
        t = None
    q = st.query_params.get("mode")
    if t in ("light", "dark"):
        ss["_mode"] = t
    elif q in ("light", "dark"):
        ss["_mode"] = q
    return ss.get("_mode", "dark")


def pal():
    return PALETTES[mode()]


def __getattr__(name):
    """c.REAL, c.MIX, ... for the current visitor's mode (see the module docstring)."""
    if name == "MIX":
        p = pal()
        return {k: (n, sub, p["MIXC"][k], dash) for k, (n, sub, dash) in _NAMES.items()}
    if name in PALETTES["dark"]:
        return pal()[name]
    raise AttributeError(name)


MODE_SCRIPT = """<script>
(function () {
  const doc = window.document;
  function current() {               // read the real background, so the label is always right
    const el = doc.querySelector('.stApp') || doc.body;
    const m = (getComputedStyle(el).backgroundColor.match(/\\d+/g) || [0, 0, 0]).map(Number);
    return (0.299 * m[0] + 0.587 * m[1] + 0.114 * m[2]) > 128 ? 'light' : 'dark';
  }
  function switchTo(next) {          // Streamlit keeps the choice as "Light"/"Dark" per page address
    const val = JSON.stringify(next === 'light' ? 'Light' : 'Dark');
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);
      if (k && k.startsWith('stActiveTheme-')) localStorage.setItem(k, val);
    }
    // also every page of the site, even ones this browser has not opened yet: without a saved key
    // a page opened in the system mode, so the site flipped mode on a first visit
    for (const p of PAGE_PATHS.concat([location.pathname])) {
      localStorage.setItem('stActiveTheme-' + p + '-v2', val);
    }
    const u = new URL(location.href);
    u.searchParams.set('mode', next);
    location.href = u.toString();
  }
  function place() {
    // top right: at the front of the group that holds Share/Deploy and the ⋮ menu
    const menu = doc.querySelector('[data-testid="stMainMenu"]');
    const group = menu ? menu.parentElement : doc.querySelector('[data-testid="stToolbar"]');
    if (!group) return;
    let b = doc.getElementById('mode-toggle');
    if (!b) {
      b = doc.createElement('button');
      b.id = 'mode-toggle';
      b.type = 'button';
      b.onclick = () => switchTo(current() === 'dark' ? 'light' : 'dark');
    }
    if (b.parentElement !== group) group.insertBefore(b, group.firstChild);
    const cur = current();
    const label = cur === 'dark' ? '\\u2600\\uFE0E  Light mode' : '\\u263E  Dark mode';
    if (b.textContent !== label) {
      b.textContent = label;
      b.title = 'Switch to ' + (cur === 'dark' ? 'light' : 'dark') +
                ' colors. The page reloads, so the Workspace run log starts over.';
      b.setAttribute('aria-label', b.title);
    }
  }
  place();
  if (!window.__modeObserver) {
    let queued = false;
    window.__modeObserver = new MutationObserver(() => {
      if (queued) return;
      queued = true;
      requestAnimationFrame(() => { queued = false; place(); });
    });
    window.__modeObserver.observe(doc.body, {childList: true, subtree: true});
  }
})();
</script>"""


def mode_script(paths=()):
    """The light/dark button's script, told every page address of the site (paths, like
    '/page_story'; '/' is the start page)."""
    return MODE_SCRIPT.replace("(function () {", "(function () {\n  const PAGE_PATHS = "
                               + json.dumps(list(paths)) + ";", 1)


def setup(paths=()):
    """Page-wide type, spacing and the light/dark button. st.html, not st.markdown: markdown read
    the [kind="primary"] selectors as link syntax and silently dropped them. paths: every page's
    address (app.py passes them, built from its st.Page list, so the list can't drift)."""
    p = pal()
    st.html(f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');
    .block-container {{padding-top: 3.2rem; padding-bottom: 4rem; max-width: 1240px}}
    h1 {{font-size: 1.75rem !important; font-weight: 600 !important; margin-bottom: .1rem !important}}
    h2 {{font-size: 1.2rem !important; font-weight: 600 !important; margin-top: .8rem !important}}
    h3 {{font-size: 1.02rem !important; font-weight: 600 !important}}
    p, li {{line-height: 1.6}}
    hr {{margin: 1.6rem 0 !important; border-color: {p['GRID']} !important}}
    .num {{font-family: {MONO}; font-variant-numeric: tabular-nums}}
    .muted {{color: {p['MUTED']}; font-size: .86rem}}
    .lead {{color: {p['TEXT2']}; max-width: 720px; margin-bottom: .6rem}}
    .hint {{color: {p['MUTED']}; font-size: .86rem; margin: -.2rem 0 .6rem}}
    .key {{display: flex; flex-wrap: wrap; gap: 16px; margin: .2rem 0 .8rem; font-size: .82rem; color: {p['TEXT2']}}}
    .key span {{display: inline-flex; align-items: center; gap: 6px}}
    .sw {{display: inline-block; width: 20px; height: 0; border-top: 2.5px solid}}
    .stats {{display: flex; flex-wrap: wrap; gap: 6px 26px; margin: .2rem 0 .6rem}}
    .stats div {{min-width: 88px}}
    .stats .l {{color: {p['MUTED']}; font-size: .74rem}}
    .stats .v {{font-family: {MONO}; font-variant-numeric: tabular-nums; font-size: .98rem}}
    .price {{font-family: {MONO}; font-variant-numeric: tabular-nums; font-size: 1.9rem; font-weight: 500}}
    .ok {{color: {p['UP']}; font-weight: 600}} .no {{color: {p['DOWN']}; font-weight: 600}}
    button[kind="primaryFormSubmit"], button[kind="primary"] {{background: {p['BUTTON']} !important;
        color: {p['BUTTON_TEXT']} !important; border: 1px solid {p['BUTTON']} !important; font-weight: 600}}
    #mode-toggle {{background: transparent; color: inherit; border: 1px solid {p['GRID']}; border-radius: 999px;
        padding: 3px 12px; margin-right: 10px; font-size: .82rem; cursor: pointer; white-space: nowrap;
        align-self: center}}
    #mode-toggle:hover {{border-color: {p['MUTED']}}}
    </style>""")
    st.html(mode_script(paths), unsafe_allow_javascript=True)


def color_key(*items):
    """One line saying what each color means (items: 'real', 'bell', 'updown' or a mix key)."""
    p, mix = pal(), __getattr__("MIX")
    parts = []
    for it in items:
        if it == "real":
            parts.append(f"<span><i class='sw' style='border-color:{p['REAL']};border-top-width:3.5px'></i>real market</span>")
        elif it == "bell":
            parts.append(f"<span><i class='sw' style='border-color:{p['BELL']};border-top-style:dotted'></i>bell curve</span>")
        elif it == "updown":
            parts.append(f"<span style='color:{p['UP']}'>▲ up</span><span style='color:{p['DOWN']}'>▼ down</span>")
        else:
            name, _, color, dash = mix[it]
            style = {"dot": "dotted", "dash": "dashed", "solid": "solid"}[dash]
            parts.append(f"<span><i class='sw' style='border-color:{color};border-top-style:{style}'></i>{name}</span>")
    st.markdown(f"<div class='key'>{''.join(parts)}</div>", unsafe_allow_html=True)


def pick_one(label, options, default, key, **kw):
    """A segmented control that always has one option picked. Clicking the picked segment again
    unselects it in Streamlit; the callback puts the last pick back, so the control and the content
    under it always agree (falling back to `default` showed one thing under an empty control)."""
    options = list(options)
    last = f"_last_{key}"

    def keep():
        if st.session_state.get(key) not in options:
            st.session_state[key] = st.session_state.get(last, default)
    value = st.segmented_control(label, options, default=default, key=key, on_change=keep, **kw)
    value = value if value in options else st.session_state.get(last, default)
    st.session_state[last] = value
    return value


def lead(text):
    st.markdown(f"<p class='lead'>{text}</p>", unsafe_allow_html=True)


def hint(text):
    """A one-line 'how to read this' note under a chart, for first-time visitors."""
    st.markdown(f"<p class='hint'>{text}</p>", unsafe_allow_html=True)


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
    p = pal()
    color = p["UP"] if x >= 0 else p["DOWN"]
    return f"<span class='num' style='color:{color}'>{'▲' if x >= 0 else '▼'} {fmt.format(abs(x))}</span>", color


# ------------------------------------------------------------------ cached data
@st.cache_data(show_spinner=False)
def saved_asset(name):
    close = ex.saved_prices(name)
    r = ex.log_returns(close)
    return close, r, ex.summary(r.values)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def live_asset(ticker):
    return ex.live_market(ticker)


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
    """{mix: (returns or None, runaway day after burn-in or None)} for the three trader mixes.
    The runaway day is negative when a history ran away during the burn-in (see gone_text)."""
    p, burn = fitted()
    return ex.run_mixes(p, burn, a1, a2, v1, days, seed)


@st.cache_data(show_spinner=False)
def runaway_story(seed, day):
    """The Story page's numbers for the chartists-only history that ran away on `day`."""
    p, burn = fitted()
    return ex.runaway_story(p, seed, int(day), burn)


def gone_text(day):
    """How a ran-away history is described. run_mixes counts days after the burn-in (the warm-up
    days the site never shows), so a history that ran away during the warm-up has a day below 1:
    "ran away on day -745" meant nothing to a reader."""
    if day < 1:
        return "ran away during the warm-up, before day 1"
    return f"ran away on day {int(day):,}"


# ------------------------------------------------------------------ charts
def style(fig, height=380, logx=False, logy=False, xtitle="", ytitle="", legend=True):
    """Chart in the visitor's mode, with the legend in its own strip above the plot so it never
    covers a line."""
    p = pal()
    fig.update_layout(template=p["TEMPLATE"], height=height, margin=dict(l=8, r=8, t=44 if legend else 12, b=8),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=p["CARD"], hovermode="closest",
                      showlegend=legend, font=dict(size=12), dragmode="zoom", uirevision="keep",
                      hoverlabel=dict(font=dict(family=MONO, size=12)),
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                                  font=dict(size=11), bgcolor="rgba(0,0,0,0)"))
    # "-" lets plotly choose (dates or numbers); forcing "linear" left the date axis empty
    fig.update_xaxes(type="log" if logx else "-", gridcolor=p["GRID"], zeroline=False, title=xtitle,
                     linecolor=p["GRID"])
    fig.update_yaxes(type="log" if logy else "-", gridcolor=p["GRID"], zeroline=False, title=ytitle,
                     linecolor=p["GRID"])
    return fig


def show(fig, key=None):
    # theme=None: the figure is styled here already; Streamlit's theme pass redid it on every rerun
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG, key=key, theme=None)


def mix_line(m, width=2.2):
    _, _, color, dash = __getattr__("MIX")[m]
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
    p = pal()
    fig = go.Figure()
    for label, r, line in series:
        x, y = ex.tail_curve(np.asarray(r))
        fig.add_scatter(x=x, y=y, name=label, mode="lines", line=line,
                        hovertemplate=f"{label}<br>%{{x:.1f}} sd or more on %{{y:.2%}} of days<extra></extra>")
    if bell:
        g = np.linspace(0.1, 6, 80)
        fig.add_scatter(x=g, y=ex.bell_tail(g), name="Bell curve", mode="lines",
                        line=dict(color=p["BELL"], dash="dot", width=2), hoverinfo="skip")
    if guide and series:
        gx, gy = ex.slope_guide(np.asarray(series[0][1]))
        fig.add_scatter(x=gx, y=gy, name="Slope 3",
                        mode="lines", line=dict(color=p["GUIDE"], dash="longdash", width=1), hoverinfo="skip")
    style(fig, height, logx=True, logy=True, xtitle="Move size (standard deviations)",
          ytitle="Share of days at least this big")
    return tail_axes(fig)


def dial_chart(market_tail=None, runs=(), current=None, height=300):
    """Tail exponent against herding strength: median and middle half of the saved sweep, the
    market's own tail, and the runs made on this page (runs: [(a1, tail)], current: (a1, tail)).
    Herding strengths where most saved histories ran away are left out (explore.sweep_band): the
    few that survive are the calm ones, so their median would be mostly luck."""
    p = pal()
    sw = numbers()["sweep"]
    b = ex.sweep_band(sw["rows"])
    b = b[b["shown"]]
    total = len(sw["seeds"])
    fig = go.Figure()
    if len(b):
        fig.add_scatter(x=list(b["a1"]) + list(b["a1"][::-1]), y=list(b["q3"]) + list(b["q1"][::-1]),
                        fill="toself", fillcolor=p["HERD_FILL"], line=dict(width=0), hoverinfo="skip",
                        name=f"Middle half of the histories that did not run away (of {total})")
        fig.add_scatter(x=b["a1"], y=b["median"], mode="lines+markers", name="Model (median)",
                        line=mix_line("herding"), marker=dict(size=5), customdata=np.c_[b["n"]],
                        hovertemplate=f"a1 %{{x:.2f}}: tail %{{y:.2f}}<br>median of %{{customdata[0]}} of "
                                      f"{total} histories<extra></extra>")
    else:
        fig.add_annotation(text="Every saved history ran away", showarrow=False, x=0.5, y=0.5,
                           xref="paper", yref="paper", font=dict(color=p["MUTED"], size=12))
    if market_tail is not None:
        fig.add_hline(y=market_tail, line=dict(color=p["REAL"], width=1.5),
                      annotation_text="this market", annotation_position="top left",
                      annotation_font=dict(size=11, color=p["REAL"]))
    if runs:
        fig.add_scatter(x=[a for a, _ in runs], y=[v for _, v in runs], mode="markers", name="Your runs",
                        marker=dict(size=8, color=p["CARD"], line=dict(color=p["REAL"], width=1.5)),
                        hovertemplate="your run, a1 %{x:.2f}: tail %{y:.2f}<extra></extra>")
    if current is not None and current[1] is not None:
        fig.add_scatter(x=[current[0]], y=[current[1]], mode="markers", name="Shown now",
                        marker=dict(size=11, color=p["REAL"]),
                        hovertemplate="shown now, a1 %{x:.2f}: tail %{y:.2f}<extra></extra>")
    return style(fig, height, xtitle="Herding strength a1", ytitle="Tail exponent", legend=False)
