"""Shared look and cached data for every page of the web app (app.py).

Dark mode only (the redesign, Nov 2026: docs/redesign/README.md). Streamlit colors the page, tables
and controls from the one color set in .streamlit/config.toml; everything drawn here (charts, the
small text styles) takes its colors from PALETTES["dark"]: pages write c.REAL, c.MIX, ...

Colors mean one thing each (checked for color blindness in the readability audit):
    REAL           the real market, white
    MIX            value traders blue, trend chasers orange, herding pink
    BELL           the bell curve, gray and dotted
    UP / DOWN      only gains / losses, always with an arrow
    ACCENT         amber: the site's own frame only (headings, labels, active tab, buttons, sliders),
                   in the style of a trading terminal. Never used inside a chart, so in a chart
                   color always means data.

Look: a trading terminal made readable for beginners. Black screen,
monospace amber labels and white monospace numbers, thin square panels; the explanations stay in a
plain sans font at full size, and every "how to read this" note is marked as such.

Smoothness: no line is drawn with more than about 1,500 points (explore.thin_indices keeps every
crash and spike), and controls that only change a chart rerun just that chart (st.fragment).
"""
import re
import threading
import time
from collections import OrderedDict, deque
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from herding import explore as ex

MONO = "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace"
_NAMES = {"fundamentalists": ("Value traders only", "trade on the true value", "dot"),
          "chartists": ("Trend chasers, no herding", "follow the trend alone", "dash"),
          "herding": ("Trend chasers who herd", "follow the trend and copy the crowd", "solid")}
# Dark colors from the readability audit (Nov 2026, docs/redesign/README.md): every chart line at least
# 3:1 against the chart, and herding, the bell curve and value traders told apart in colorblind views.
PALETTES = {
    "dark": dict(REAL="#F2F4F8", BELL="#737A87", UP="#34D399", DOWN="#FF7A6B", GRID="#2E2E2E",
                 CARD="#050505", MUTED="#A29F98", BG="#000000", TEXT2="#C9C6BF", GUIDE="#8A8D99",
                 SHADE_MID="#9AA3B5", SHADE_LOW="#7A8496", FAINT="rgba(255,255,255,0.04)",
                 OK_FILL="rgba(52,211,153,0.08)", DOWN_FILL="rgba(255,122,107,0.10)",
                 UP_FILL="rgba(52,211,153,0.10)", HERD_FILL="rgba(242,167,218,0.10)",
                 TEMPLATE="plotly_dark", BUTTON="#FFA028", BUTTON_TEXT="#000000", CHIP_TEXT="#000000",
                 ACCENT="#FFA028", PANEL="#0D0D0D", LINE="#4A4A4A",
                 MIXC=dict(fundamentalists="#3D9BE0", chartists="#E66A1F", herding="#F2A7DA"),
                 PICK=["#F2F2F5", "#F0E442", "#4FD1C5", "#FF9F7F", "#C084FC", "#93C5FD"]),
}
# line styles for lines told apart by color alone (the Compare page's markets): a second cue for
# readers who can't tell two of the colors apart. Same order as PICK.
PICK_DASH = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
MAX_POINTS = 1500
TEST_PROGRAM = "Test program"     # the hands-on page's name (app.py, the button under the tape)
PLOT_CONFIG = {"displayModeBar": False, "displaylogo": False}   # no toolbar: drag to zoom, double-click resets

HELP = {
    "tail": "A score for how common big days are (lower = more big days). The S&P 500 is near 3: a move "
            "twice as big is about 2³ = 8 times rarer. A bell curve, measured the same way, comes out near 6.",
    "kurtosis": "Excess kurtosis. 0 for a bell curve; bigger means more very calm and very wild days (fat tails).",
    "memory": "Correlation of move sizes a month (20 trading days) apart. 0 means no memory; above 0 "
              "means bursts of wild days.",
    "big": "Days that moved more than 4 sd (the usual size of a day's move), up or down.",
    "vol": "Annualized volatility: the standard deviation (sd) of daily returns in the range, times √252 (√365 for a market open every day, like crypto).",
    "dd": "The largest fall from a high point to a later low within the range.",
}


# ------------------------------------------------------------------ colors
def superscript(n):
    """A whole number as superscript digits: 86 -> '⁸⁶' (WORDING rule 10: 10⁸⁶, never 10^86)."""
    return str(int(n)).translate(str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"))


def pal():
    """The site's one color set (dark only since the redesign; light mode and mode() were removed)."""
    return PALETTES["dark"]


def __getattr__(name):
    """c.REAL, c.MIX, ... from the dark color set (see the module docstring)."""
    if name == "MIX":
        p = pal()
        return {k: (n, sub, p["MIXC"][k], dash) for k, (n, sub, dash) in _NAMES.items()}
    if name in PALETTES["dark"]:
        return pal()[name]
    raise AttributeError(name)


SITE_SCRIPT = """<script>
(function () {
  const doc = window.document;
  if (!window.__jumpLinks) {         // links like "Jump to the model controls": Streamlit swallows
    window.__jumpLinks = true;        // same-page #links, so scroll to the target here
    doc.addEventListener('click', (ev) => {
      const a = ev.target.closest && ev.target.closest('a.jump[href^="#"]');
      const el = a && doc.getElementById(a.getAttribute('href').slice(1));
      if (!el) return;
      ev.preventDefault();
      el.scrollIntoView({behavior: 'smooth', block: 'start'});
    }, true);
  }
})();
</script>"""


def nav_numbers(paths=()):
    """CSS that puts a number chip before each page in the top navigation, in the order a beginner
    should visit them. paths: every page's address in order, like ['/', '/page_story', ...] ('/' is
    the start page). Each link is numbered by its own address, so the pages that do not fit on a
    narrow screen and move into the "N more" list keep their numbers (a CSS counter restarted at 1
    there). Without paths (tests), a counter numbers the links in the bar. The number is decorative
    (the page name follows it), so screen readers skip it: content: "1" / "" (after a plain
    content: "1", which older browsers that don't know the "/ alt" form keep instead)."""
    if not paths:
        return ('[data-testid="stTopNav"] {counter-reset: fkey}\n'
                '[data-testid="stTopNavLink"] {counter-increment: fkey}\n'
                '[data-testid="stTopNavLink"]::before {content: counter(fkey); content: counter(fkey) / ""}\n')
    rules = []
    for i, path in enumerate(paths, start=1):
        sel = ", ".join(f'[data-testid="{t}"][href$="{path}"]::before' for t in ("stTopNavLink", "stTopNavDropdownLink"))
        rules.append(f'{sel} {{content: "{i}"; content: "{i}" / ""}}')
    return "\n".join(rules) + "\n"


def cta_links(page_title):
    """The links under the tape on a page called page_title: a list of (key, page file, label, icon).
    The Start here page points to the Story (the first thing to read); every other page but the
    Test program points to the Test program; every page but Start here also links back to the list
    of words used on the site."""
    links = []
    if page_title == "Start here":
        links.append(("cta", "web/page_story.py", "Start the story", ":material/auto_stories:"))
    elif page_title != TEST_PROGRAM:
        links.append(("cta", "web/page_explore.py", "Open the test program",
                      ":material/play_arrow:"))
    if page_title != "Start here":
        links.append(("words", "web/page_guide.py", "Words used on this site", ":material/menu_book:"))
    return links


def css(paths=()):
    """The page-wide style sheet (see setup)."""
    p = pal()
    a, line, chip = p["ACCENT"], p["LINE"], p["CHIP_TEXT"]
    return f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
    .block-container {{padding-top: 3.6rem; padding-bottom: 4rem; max-width: 1280px}}

    /* keyboard focus: a clear amber ring on anything you can tab to */
    :is(button, a, input, [role="tab"], [role="option"], [tabindex]):focus-visible {{outline: 2px solid {a} !important;
        outline-offset: 2px}}

    /* headings: monospace, like a terminal screen title; sentence case stays, so they read easily */
    h1 {{font-family: {MONO} !important; font-size: 1.6rem !important; font-weight: 600 !important;
        letter-spacing: -.01em; margin-bottom: .2rem !important}}
    h1::after {{content: ""; display: inline-block; width: .55em; height: 1em; margin-left: .25em;
        vertical-align: -.12em; background: {a}; animation: blink 1.1s steps(1) 4}}
    @keyframes blink {{50% {{opacity: 0}}}}
    @media (prefers-reduced-motion: reduce) {{h1::after {{animation: none}}}}
    h2 {{font-family: {MONO} !important; font-size: 1.02rem !important; font-weight: 600 !important;
        color: {a} !important; margin-top: 1rem !important; padding: 0 0 .35rem !important;
        border-bottom: 1px solid {line}}}
    h2::before {{content: "■  "; content: "■  " / ""; font-size: .8em}}
    h3 {{font-family: {MONO} !important; font-size: .92rem !important; font-weight: 600 !important;
        text-transform: uppercase; letter-spacing: .08em; color: {a} !important}}
    p, li {{line-height: 1.65}}
    hr {{margin: 1.6rem 0 !important; border-color: {line} !important}}

    /* the frame: square panels, thin lines */
    [data-testid="stSidebar"] {{border-right: 1px solid {line}}}
    /* control labels: amber monospace, but in normal case so they read like a sentence and Greek
       letters stay themselves (uppercase turned ε into Ε) */
    [data-testid="stWidgetLabel"] p {{font-family: {MONO}; font-size: .82rem !important; letter-spacing: .02em;
        color: {a}}}
    [data-testid="stTab"] p {{font-family: {MONO}; font-size: .8rem !important; text-transform: uppercase;
        letter-spacing: .06em}}
    [data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {{border: 1px solid {line}; border-radius: 2px}}
    [data-testid="stPlotlyChart"] {{padding: 4px}}

    /* top navigation: numbered keys in the order a beginner should visit the pages */
    [data-testid="stTopNavLink"], [data-testid="stTopNavDropdownLink"] {{border-radius: 2px !important}}
    [data-testid="stTopNavLink"] > span:has([data-testid="stIconMaterial"]),
    [data-testid="stTopNavDropdownLink"] > span:has([data-testid="stIconMaterial"]) {{display: none}}
    [data-testid="stTopNavLinkContainer"] {{margin-right: 6px}}
    {nav_numbers(paths)}
    [data-testid="stTopNavLink"]::before, [data-testid="stTopNavDropdownLink"]::before {{font-family: {MONO};
        font-size: .78rem; font-weight: 700; color: {chip}; background: {p['MUTED']}; padding: 1px 5px;
        margin-right: 7px; border-radius: 2px}}
    [data-testid="stTopNavLink"] p, [data-testid="stTopNavDropdownLink"] p {{font-family: {MONO};
        font-size: .8rem !important; text-transform: uppercase; letter-spacing: .05em}}
    [data-testid="stTopNavLink"][aria-current="page"]::before,
    [data-testid="stTopNavDropdownLink"][aria-current="page"]::before {{background: {a}}}
    [data-testid="stTopNavLink"][aria-current="page"] p,
    [data-testid="stTopNavDropdownLink"][aria-current="page"] p {{color: {a}}}

    /* the tape: one slim strip with the three saved indexes, a hairline under it (redesign, Nov 2026:
       no box, so the page starts with the content, not a panel) */
    .tape {{display: flex; flex-wrap: wrap; align-items: center; gap: 4px 22px; font-family: {MONO};
        font-size: .8rem; padding: 4px 0 7px; margin: 0 0 .6rem; border-bottom: 1px solid {line};
        font-variant-numeric: tabular-nums}}
    .tape .brand {{color: {a}; font-weight: 700; letter-spacing: .06em}}
    .tape .pg {{color: {p['TEXT2']}; text-transform: uppercase; letter-spacing: .06em}}
    .tape .tk {{color: {p['MUTED']}}} .tape .tk b {{font-weight: 500; color: {p['REAL']}}}
    .tape .asof {{margin-left: auto; color: {p['MUTED']}; text-transform: uppercase; font-size: .78rem}}

    .num {{font-family: {MONO}; font-variant-numeric: tabular-nums}}
    .muted {{color: {p['MUTED']}; font-size: .86rem}}
    .lead {{color: {p['TEXT2']}; max-width: 760px; margin-bottom: .8rem; font-size: 1.02rem}}
    .hint {{color: {p['MUTED']}; font-size: .86rem; margin: .1rem 0 .7rem}}
    .hint::before {{content: "HOW TO READ ▸  "; font-family: {MONO}; font-size: .78rem; font-weight: 600;
        letter-spacing: .06em; color: {a}}}
    .try {{color: {p['TEXT2']}; font-size: .92rem; margin: .5rem 0 .15rem; padding: .45rem .7rem;
        border-left: 3px solid {a}; background: {p['HERD_FILL']}}}
    .try::before {{content: "TRY THIS ▸  "; font-family: {MONO}; font-size: .78rem; font-weight: 600;
        letter-spacing: .06em; color: {a}}}
    .key {{display: flex; flex-wrap: wrap; gap: 6px 18px; margin: .2rem 0 .8rem; font-size: .78rem; color: {p['TEXT2']};
        font-family: {MONO}; padding: 5px 10px; border: 1px solid {line}; width: fit-content; max-width: 100%}}
    .key::before {{content: "KEY"; color: {a}; font-weight: 600; letter-spacing: .08em}}
    .key span {{display: inline-flex; align-items: center; gap: 6px}}
    .sw {{display: inline-block; width: 20px; height: 0; border-top: 2.5px solid}}

    /* numbers in a row of cells, label above value, like a terminal quote panel */
    .stats {{display: flex; flex-wrap: wrap; margin: .3rem 0 .7rem; padding: 1px 0 0 1px; max-width: 100%}}
    .stats > div {{min-width: 104px; padding: 5px 12px; border: 1px solid {line}; margin: -1px 0 0 -1px}}
    .stats .l {{color: {a}; font-family: {MONO}; font-size: .78rem; text-transform: uppercase; letter-spacing: .07em}}
    .stats .v {{font-family: {MONO}; font-variant-numeric: tabular-nums; font-size: .98rem}}
    details.stats-help {{margin: -.3rem 0 .8rem}}
    details.stats-help summary {{cursor: pointer; width: fit-content}}
    details.stats-help p {{margin: .2rem 0 0 1rem; line-height: 1.5}}
    .price {{font-family: {MONO}; font-variant-numeric: tabular-nums; font-size: 2rem; font-weight: 600}}
    /* a check's verdict: a mark and bold text, not green/red (those mean gains and losses only) */
    .ok, .no {{font-weight: 700; color: {p['REAL']}}}
    .ok::before {{content: "✓ "; content: "✓ " / ""}} .no::before {{content: "✗ "; content: "✗ " / ""}}
    button[kind="primaryFormSubmit"], button[kind="primary"] {{background: {p['BUTTON']} !important;
        color: {p['BUTTON_TEXT']} !important; border: 1px solid {a} !important; font-weight: 700;
        font-family: {MONO}; text-transform: uppercase; letter-spacing: .06em}}
    button[kind="primaryFormSubmit"] p, button[kind="primary"] p {{font-family: {MONO}; font-weight: 700}}
    .st-key-picker {{border-left: 3px solid {a} !important}}
    .st-key-picker [data-testid="stButtonGroup"] button p {{font-family: {MONO}; font-size: .86rem;
        letter-spacing: .03em}}
    .st-key-picker [data-testid="stButtonGroup"] div {{flex-wrap: wrap !important; overflow: visible !important}}

    /* the links under the tape: one big button (the next page to visit) and a small text link */
    .st-key-cta {{margin-bottom: .3rem; max-width: 100%}}
    .st-key-cta a {{background: {p['BUTTON']}; border: 1px solid {a}; border-radius: 2px; padding: 6px 14px;
        width: fit-content; max-width: 100%; height: auto; white-space: normal}}
    .st-key-cta a * {{white-space: normal !important; overflow: visible !important; text-overflow: clip !important}}
    .st-key-cta a:hover {{filter: brightness(1.08)}}
    .st-key-cta a [data-testid="stIconMaterial"] {{color: {p['BUTTON_TEXT']} !important}}
    .st-key-cta a p {{color: {p['BUTTON_TEXT']} !important; font-family: {MONO}; font-weight: 700; line-height: 1.35;
        text-transform: uppercase; letter-spacing: .05em; font-size: .82rem !important; overflow-wrap: anywhere}}
    .st-key-words {{margin-bottom: 1rem}}
    .st-key-words a {{padding: 0 2px; width: fit-content}}
    .st-key-words a p {{font-size: .86rem !important; text-decoration: underline; text-underline-offset: 3px}}
    .phone-only {{display: none}}
    @media (max-width: 640px) {{
        .phone-only {{display: block; font-family: {MONO}; font-size: .86rem; margin: .4rem 0}}
        .tape .asof {{margin-left: 0}}
    }}
    </style>"""


def setup(paths=(), page_title=""):
    """Page-wide type, spacing and the terminal tape at the top. st.html, not
    st.markdown: markdown read the [kind="primary"] selectors as link syntax and silently dropped
    them. paths: every page's address (app.py passes them, built from its st.Page list, so the list
    can't drift). page_title: the open page's name, shown in the tape."""
    st.html(css(paths))
    st.html(SITE_SCRIPT, unsafe_allow_javascript=True)
    tape(page_title)
    warm_model()


def _compile_model():
    try:
        p, _ = ex.fitted_settings()
        ex.model.simulate(p, 3, seed=0)
    except Exception:            # warming up is only a speed-up: never let it break a page
        return
    _warm_pages()


def _warm_pages():
    """After the model is compiled: fill the caches the slowest first visits need, through the same
    cached functions the pages call, so the numbers are the same: the Story's default history (its
    three mixes and the runaway story, about 1 s)."""
    try:
        run = default_run()
        mixes = run_mixes(**run)
        day = mixes["chartists"][1]
        if day is not None and day >= 1:
            runaway_story(run["seed"], day)
    except Exception:            # only a speed-up, as above
        pass


@st.cache_resource(show_spinner=False)
def warm_model():
    """Once per server: run a 3-day history in the background, so the model's compiled code is loaded
    (or compiled, about 2.6 s on a fresh server) before the first visitor presses anything."""
    t = threading.Thread(target=_compile_model, daemon=True, name="warm-model")
    t.start()
    return t


@st.cache_data(show_spinner=False)
def _tape_quotes(stamp=None):
    """[(name, last close, change on the last day, date of the last close)] for the tape. Only the
    closes are read (saved_asset also works out the tail and kurtosis, which the tape never shows),
    and cached, so the strip costs nothing after the first visitor."""
    out = []
    for name in ("S&P 500", "Nasdaq-100", "Dow Jones"):
        try:
            close = ex.saved_prices(name)
            out.append((name, float(close.iloc[-1]), ex.day_change(close), close.index[-1]))
        except Exception:                       # a missing file must not take the whole site down
            continue
    return out


def tape(page_title=""):
    """The strip at the top of every page: site name, the open page, and the last saved close of the
    three indexes with their change on the day (a gain or loss arrow in the usual colors), then the
    links under it (cta_links)."""
    items, last = [], None
    for name, close, change, day in _tape_quotes(data_stamp()):
        chg, _ = arrow(change)
        items.append(f"<span class='tk' title='{name}: last saved close and its change that day'>"
                     f"{name.upper()} <b>{ex.price_text(close)}</b> {chg}</span>")
        last = day
    asof = f"<span class='asof'>Saved data · last close {last:%d %b %Y}</span>" if last is not None else ""
    pg = f"<span class='pg'>▸ {page_title}</span>" if page_title else ""
    st.html(f"<div class='tape'><span class='brand'>HERDING</span>{pg}{''.join(items)}{asof}</div>")
    links = cta_links(page_title)
    for key, page, label, icon in links:
        with st.container(key=key):
            st.page_link(page, label=label, icon=icon)
    if not any(k == "cta" for k, *_ in links):
        st.html("<div style='height:.7rem'></div>")


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


def try_this(text):
    """One thing to do with a chart, with a question to answer (PhET, Nicky Case: act, then look),
    above its 'how to read' note. text is trusted site text (Markdown bold ** is turned into <b>)."""
    html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    st.markdown(f"<p class='try'>{html}</p>", unsafe_allow_html=True)


def stats_html(items):
    """The HTML of a stats row (see stats_row). The help shows on hover (title) and, because a phone
    or a keyboard can't hover, also in a "What these numbers mean" note that opens on a click."""
    cells, notes = [], []
    for label, text, color, tip in items:
        style = f" style='color:{color}'" if color else ""
        title = f" title='{tip}'" if tip else ""
        cells.append(f"<div{title}><div class='l'>{label}</div><div class='v'{style}>{text}</div></div>")
        if tip:
            notes.append(f"<p><b>{label}</b>: {tip}</p>")
    html = f"<div class='stats'>{''.join(cells)}</div>"
    if notes:
        html += (f"<details class='muted stats-help'><summary>What these numbers mean</summary>"
                 f"{''.join(notes)}</details>")
    return html


def stats_row(items):
    """A row of small label/number pairs: items = [(label, text, color or None, help or None)]."""
    st.markdown(stats_html(items), unsafe_allow_html=True)


def wait_range(cc):
    """How uncertain the model's Black-Monday wait is (explore.crash_check's bm_range, an exact Poisson
    95% range from only a few falls), in words for the pages."""
    lo, hi = cc["bm_range"]
    n = cc["bm_count"]
    top = "no upper limit" if hi == float("inf") else f"{hi:,.0f} years"
    return (f"very uncertain: from only {n} such fall{'s' if n != 1 else ''}, the true wait could be "
            f"anywhere from {lo:,.0f} years to {top}")


def verdict(passed):
    return "<span class='ok'>passes</span>" if passed else "<span class='no'>does not pass</span>"


def arrow(x, fmt="{:.2%}"):
    p = pal()
    color = p["UP"] if x >= 0 else p["DOWN"]
    return f"<span class='num' style='color:{color}'>{'▲' if x >= 0 else '▼'} {fmt.format(abs(x))}</span>", color


# ------------------------------------------------------------------ cached data
# Streamlit's cache knows when a function's CODE changes, not when the FILE it reads changes: after
# a refit, a data download or a caption edit, a running site kept serving the old copy (it crashed
# the Findings page once, KeyError 'paper'). So every cached function that reads a file takes that
# file's time ("stamp") as an extra argument: a changed file is a new cache key, read again.
DATA = ex.market_data.ROOT / "data"


def file_stamp(*paths):
    """The last-changed times of these files (0 for a missing one), as a cache key."""
    return tuple(p.stat().st_mtime if p.exists() else 0.0 for p in map(Path, paths))


def data_stamp():
    """The latest change anywhere in data/ (the saved prices, read by many functions)."""
    return max((p.stat().st_mtime for p in DATA.rglob("*") if p.is_file()), default=0.0)


@st.cache_data(show_spinner=False)
def _saved_asset(name, stamp):
    close = ex.saved_prices(name)
    r = ex.log_returns(close)
    return close, r, ex.summary(r.values)


def saved_asset(name):
    """(closes, daily log returns, summary) of one saved market, read again when data/ changes."""
    return _saved_asset(name, data_stamp())


LIVE_REFRESH = 60       # seconds: how far behind Yahoo a live price may be (it was up to a day)


# Limits that keep one visitor (or a script) from exhausting the server or getting it blocked by
# Yahoo: every cache keyed by a typed ticker keeps at most LIVE_TICKERS tickers (the oldest are
# dropped), and the whole server starts at most FETCH_LIMIT new downloads of a ticker's history per
# minute. Tickers already cached cost nothing.
LIVE_TICKERS = 40
FETCH_LIMIT = 20
_fetches = deque()
_fetch_lock = threading.Lock()


def allow_fetch(now=None, limit=None, window=60.0):
    """True (and counted) if the server may start one more download from Yahoo now: fewer than
    `limit` (FETCH_LIMIT) in the last `window` seconds."""
    now = time.time() if now is None else now
    limit = FETCH_LIMIT if limit is None else limit
    with _fetch_lock:
        while _fetches and now - _fetches[0] >= window:
            _fetches.popleft()
        if len(_fetches) >= limit:
            return False
        _fetches.append(now)
        return True


BUSY = "The site is fetching many prices right now. Try again in a minute, or pick a saved market."


@st.cache_data(ttl=24 * 3600, show_spinner=False, max_entries=LIVE_TICKERS)
def _live_history(ticker):
    if not allow_fetch():
        raise ConnectionError(BUSY)
    return ex.live_market(ticker)


@st.cache_data(ttl=LIVE_REFRESH, show_spinner=False, max_entries=LIVE_TICKERS)
def _live_recent(ticker):
    """The last few daily closes (today's price so far included), or None if Yahoo did not answer:
    then the day-old history is still shown rather than an error."""
    try:
        return ex.recent_prices(ticker)
    except (ConnectionError, ValueError):
        return None


@st.cache_data(show_spinner=False, max_entries=20)
def _spliced(close, recent):
    return ex.market_from_close(ex.splice_recent(close, recent))


def live_asset(ticker):
    """(closes, daily log returns, summary) from Yahoo. The long history is fetched once a day, its
    last days every LIVE_REFRESH seconds, so the last price is at most a minute behind Yahoo."""
    close, r, s = _live_history(ticker)
    recent = _live_recent(ticker)
    if recent is None:
        return close, r, s
    try:
        return _spliced(close, recent)
    except ValueError:              # a strange recent fetch must not hide the history
        return close, r, s


@st.cache_data(ttl=3600, show_spinner=False, max_entries=len(ex.FRED_SERIES))
def _fred_recent(symbol):
    try:
        return ex.fred_closes(symbol)
    except (ConnectionError, ValueError):
        return None


def fred_asset(name, symbol):
    """A saved index with its latest daily closes from FRED added on (FRED updates once a day, so
    asked at most once an hour): (closes, returns, summary), or None if FRED did not answer either."""
    recent = _fred_recent(symbol)
    if recent is None:
        return None
    close, _, _ = saved_asset(name)
    try:
        return _spliced(close, recent)
    except ValueError:
        return None


QUOTE_REFRESH = 15      # seconds between two asks for a live header's latest price


@st.cache_data(ttl=QUOTE_REFRESH, show_spinner=False, max_entries=LIVE_TICKERS)
def live_quote(ticker):
    """Yahoo's latest price for the live header (explore.live_quote) with the time it was asked for,
    or None if Yahoo did not answer (the header then shows the last daily close)."""
    try:
        q = ex.live_quote(ticker)
    except (ConnectionError, ValueError):
        return None
    return dict(q, at=time.time())


@st.cache_data(ttl=LIVE_REFRESH, show_spinner=False, max_entries=2 * LIVE_TICKERS)
def intraday(ticker, frame):
    return ex.intraday_prices(ticker, frame)


@st.cache_data(show_spinner=False)
def _market_table(stamp):
    return ex.market_table()


def market_table():
    return _market_table(data_stamp())


@st.cache_data(show_spinner=False)
def _fitted(stamp):
    return ex.fitted_settings()


def fitted():
    """The S&P 500 fit (settings, burn-in), read again when settings/sp500_calibrated.json changes."""
    return _fitted(file_stamp(ex.FIT))


@st.cache_data(show_spinner=False, max_entries=2)
def _numbers(changed):
    return ex.web_numbers()


def numbers():
    """The saved results behind the Checks, Story and Findings pages (scripts/web_numbers.py). Read
    again whenever the file changes (its time is part of the cache key): a cache of the old file
    crashed the Findings page after web_numbers.py added the paper comparison (KeyError 'paper')."""
    return _numbers(file_stamp(ex.WEB_NUMBERS))


def default_run():
    """The history the site shows unless you change it: the S&P fit, the first held-out seed and an
    S&P-length history, the same history as the project's figures (Fig 1)."""
    p, _ = fitted()
    rec = numbers()["recipe"]
    return dict(a1=float(p["a1"]), a2=float(p["a2"]), v1=float(p["v1"]), days=int(rec["days"]),
                seed=int(rec["seeds"][0]))


# Each trader mix's history, kept by the settings it really depends on (explore.mix_key): moving the
# herding slider then reruns only "trend chasers who herd", one history instead of three (on Streamlit
# Cloud's small server the three barely run side by side). Shared by all visitors, newest kept.
_MIX_RESULTS = OrderedDict()
_MIX_LOCK = threading.Lock()
MIX_RESULTS_MAX = 120          # about 120 x 160 kB = 20 MB at most
# The same histories day by day (explore.market_paths), for the Simulated market chart, so it needn't
# run a history the slider just ran (0.35 s a slider move, Nov 2026). Bigger (5 arrays), so fewer kept.
_MARKETS = OrderedDict()
MARKETS_MAX = 24               # about 24 x 0.8 MB = 19 MB at most


def _mix_store_key(m, a1, a2, v1, days, seed, stamp):
    return (m, ex.mix_key(m, a1, a2, v1), int(days), int(seed), stamp)


@st.cache_data(show_spinner=False, max_entries=60)
def _run_mixes(a1, a2, v1, days, seed, stamp):
    p, burn = fitted()
    keys = {m: _mix_store_key(m, a1, a2, v1, days, seed, stamp) for m in ex.MIXES}
    with _MIX_LOCK:
        known = {m: _MIX_RESULTS[k] for m, k in keys.items() if k in _MIX_RESULTS}
    paths = {}
    out = ex.run_mixes(p, burn, a1, a2, v1, days, seed, known=known, paths=paths)
    with _MIX_LOCK:
        for m, k in keys.items():
            if m in out:
                _MIX_RESULTS[k] = out[m]
                _MIX_RESULTS.move_to_end(k)
            if m in paths:
                _MARKETS[k] = paths[m]
                _MARKETS.move_to_end(k)
        while len(_MIX_RESULTS) > MIX_RESULTS_MAX:
            _MIX_RESULTS.popitem(last=False)
        while len(_MARKETS) > MARKETS_MAX:
            _MARKETS.popitem(last=False)
    return out


def run_mixes(a1, a2, v1, days, seed):
    """{mix: (returns or None, runaway day after burn-in or None)} for the three trader mixes.
    The runaway day is negative when a history ran away during the burn-in (see gone_text).
    Keyed on the fit's file too: after a refit, cached runs made with the old settings are not reused."""
    return _run_mixes(a1, a2, v1, days, seed, file_stamp(ex.FIT))


@st.cache_data(show_spinner=False, max_entries=30)
def _model_market(mix, a1, a2, v1, days, seed, stamp):
    with _MIX_LOCK:          # this history was just run for the other charts: reuse it, not rerun it
        kept = _MARKETS.get(_mix_store_key(mix, a1, a2, v1, days, seed, stamp))
    if kept is not None:
        return kept
    p, burn = fitted()
    try:
        return ex.model_market(p, burn, mix, a1, a2, v1, days, seed)
    except ValueError:          # ran away in the warm-up: nothing to draw
        return None


def model_market(mix, a1, a2, v1, days, seed):
    """The simulated market behind one trader mix (explore.model_market), or None when that history
    ran away in the warm-up. Same arguments and cache key as run_mixes."""
    return _model_market(mix, a1, a2, v1, days, seed, file_stamp(ex.FIT))


@st.cache_data(show_spinner=False)
def _runaway_story(seed, day, stamp):
    p, burn = fitted()
    return ex.runaway_story(p, seed, int(day), burn)


def runaway_story(seed, day):
    """The Story page's numbers for the history of trend chasers without herding that ran away on `day`."""
    return _runaway_story(seed, day, file_stamp(ex.FIT))


def gone_text(day):
    """How a ran-away history is described. run_mixes counts days after the burn-in (the warm-up
    days the site never shows), so a history that ran away during the warm-up has a day below 1:
    "ran away on day -745" meant nothing to a reader."""
    if day < 1:
        return "ran away during the warm-up, before day 1"
    return f"ran away on day {int(day):,}"


# ------------------------------------------------------------------ charts
_TEMPLATES = {}


def _template(name):
    """plotly's built-in template as an object, looked up once: given by name, plotly built and checked
    the whole template again for every chart (half the time it took to draw one, Nov 2026)."""
    if name not in _TEMPLATES:
        import plotly.io as pio
        _TEMPLATES[name] = pio.templates[name]
    return _TEMPLATES[name]


def style(fig, height=380, logx=False, logy=False, xtitle="", ytitle="", legend=True, hover="x unified",
          plain_log=False):
    """Chart in the site's colors, with the legend in its own strip above the plot so it never
    covers a line. hover:
      "x unified" (default, for lines over time that share their dates): one box reads every line
                  at the mouse's date, with a dotted crosshair, so the numbers follow the mouse;
      "closest"   for lines that don't share x values (the log-log tail curves, the herding dial)
                  and scatter maps: the box shows the one line or point under the mouse, no crosshair;
      "y unified" for horizontal bars.
    plain_log: label a log axis at powers of ten in plain numbers (1, 10, 100, 1,000) instead of
    plotly's 1k / 100μ (see log_ticks)."""
    p = pal()
    spikes = hover in ("x unified", "y unified")       # a crosshair only where one box reads every line
    fig.update_layout(template=_template(p["TEMPLATE"]), height=height, margin=dict(l=8, r=8, t=44 if legend else 12, b=8),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=p["CARD"], hovermode=hover,
                      hoverdistance=-1, spikedistance=-1,
                      showlegend=legend, font=dict(size=12, family=MONO, color=p["TEXT2"]),
                      dragmode="zoom", uirevision="keep",
                      # namelength -1: the whole line name in the hover box, never cut to "Chartists wh..."
                      hoverlabel=dict(font=dict(family=MONO, size=12), bgcolor=p["PANEL"], bordercolor=p["LINE"],
                                      namelength=-1),
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                                  font=dict(size=12), bgcolor="rgba(0,0,0,0)"))
    # "-" lets plotly choose (dates or numbers); forcing "linear" left the date axis empty
    axis = dict(showspikes=spikes, spikemode="across", spikesnap="cursor", spikethickness=1, spikedash="dot",
                spikecolor=p["MUTED"], gridcolor=p["GRID"], zeroline=False, linecolor=p["LINE"],
                tickfont=dict(size=12, color=p["MUTED"]), title_font=dict(size=12, color=p["MUTED"]))
    fig.update_xaxes(type="log" if logx else "-", title=xtitle, **axis)
    if logx:                                 # the hover box's header: 1.87, not 1.870863
        fig.update_xaxes(hoverformat=".2f")
    fig.update_yaxes(type="log" if logy else "-", title=ytitle, **axis)
    if plain_log:
        log_ticks(fig, x=logx, y=logy)
    return fig


def log_ticks(fig, x=False, y=True):
    """Label log axes only at powers of ten, written as plain numbers: 0.01, 0.1, 1, 10, 100, 1,000,
    10,000 (plotly's own labels read 1k, 2, 5 and 100μ, which a beginner can't read). dtick=1 puts a
    tick at every power of ten; exponentformat 'none' and a ',' separator write them out in full.
    x, y: which axes. Returns the figure."""
    ticks = dict(dtick=1, tickformat=",~g", exponentformat="none", minor=dict(showgrid=False))
    if x:
        fig.update_xaxes(**ticks)
    if y:
        fig.update_yaxes(**ticks)
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
    # each curve has its own x values, so "x unified" read the other lines at a point they don't have
    style(fig, height, logx=True, logy=True, xtitle="Size of the day's move (sd)",
          ytitle="Share of days at least this big", hover="closest")
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
                        name=f"Middle half of the simulated histories that did not run away (of {total})")
        fig.add_scatter(x=b["a1"], y=b["median"], mode="lines+markers", name="Model (median)",
                        line=mix_line("herding"), marker=dict(size=5), customdata=np.c_[b["n"]],
                        hovertemplate=f"herding strength %{{x:.2f}}: tail exponent %{{y:.2f}}<br>middle value of %{{customdata[0]}} of "
                                      f"{total} simulated histories<extra></extra>")
    else:
        fig.add_annotation(text="Every saved simulated history ran away", showarrow=False, x=0.5, y=0.5,
                           xref="paper", yref="paper", font=dict(color=p["MUTED"], size=12))
    if market_tail is not None:
        fig.add_hline(y=market_tail, line=dict(color=p["REAL"], width=1.5),
                      annotation_text="this market", annotation_position="top left",
                      annotation_font=dict(size=11, color=p["REAL"]))
    if runs:
        fig.add_scatter(x=[a for a, _ in runs], y=[v for _, v in runs], mode="markers", name="Your simulated histories",
                        marker=dict(size=8, color=p["CARD"], line=dict(color=p["REAL"], width=1.5)),
                        hovertemplate="your simulated history, herding strength %{x:.2f}: tail exponent %{y:.2f}<extra></extra>")
    if current is not None and current[1] is not None:
        fig.add_scatter(x=[current[0]], y=[current[1]], mode="markers", name="Shown now",
                        marker=dict(size=11, color=p["REAL"]),
                        hovertemplate="shown now, herding strength %{x:.2f}: tail exponent %{y:.2f}<extra></extra>")
    return style(fig, height, xtitle="Herding strength (a1)", ytitle="Tail exponent (lower = more big days)", legend=False, hover="closest")
