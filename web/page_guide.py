"""Start here: what the site is and how to use it."""
import base64
import json
import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from herding import explore as ex
from web import common as c

REPO = "https://github.com/avishj19/can-herding-explain-market-crashes"


def gh(path, text=None):
    """A Markdown link to a file of the project's public repository on GitHub."""
    return f"[`{text or path}`]({REPO}/blob/main/{path})"


def _power(years):
    """The power of ten of a very long wait, as superscript digits: 1.4e86 -> '⁸⁶'."""
    return c.superscript(math.floor(math.log10(years)))


def short_answer(nums, spx, spx_r):
    """The site's answer in four lines: what herding explains and what it misses. The model's tail is
    the median of the saved held-out histories (web/numbers.json); the S&P's is computed from its file."""
    tail = ex.spread(nums["checks"]["herding"]["tail"])
    cc = ex.crash_check(nums, spx_r)
    lo, hi = ex.PROPOSAL_TAIL

    def _where(x):
        return "inside" if lo <= x <= hi else "above" if x > hi else "below"

    return (
        "**Short answer: partly.**\n\n"
        f"- **Yes:** with plain bell-curve news, herding alone gives the S&P 500's fat tails and bursts of "
        f"wild days. The model's tail exponent is {tail['median']:.2f} (median of {tail['n']} simulated histories), "
        f"{_where(tail['median'])} the proposal's {lo} to {hi}; the S&P 500's is {spx['tail']:.2f}. The settings "
        f"were fitted to the S&P 500, so this match is partly by design; the bell curve without trend chasers was "
        f"never a target.\n"
        f"- **Crashes:** a bell curve puts 20-sd falls 10{_power(ex.bell_crash_years(20))} years apart, and Black "
        f"Monday ({cc['worst_sd']:.0f} sd) 10{_power(cc['bm_bell'])} years apart. "
        f"In the model, an {cc['size']}-sd fall comes about every {cc['row']['model']:.0f} years, and a fall the size "
        f"of Black Monday about every {cc['bm_model']:,.0f} years ({c.wait_range(cc)}). That is far closer, but "
        f"still rarer than in the S&P 500.\n"
        "- **No skew:** it does not make falls bigger than rises, as the S&P 500 does.\n"
        "- **Daily reversal:** its prices reverse a little from one day to the next, which the S&P 500's "
        "do not (Hurst H 0.39 against 0.50 in the project's "
        f"[report]({REPO}/blob/main/figures/RESULTS.md)).\n"
        "- **Weeks:** its fat tails fade over a few weeks somewhat faster than the S&P 500's.\n\n"
        "The **Checks** page has the details (see Beyond the paper).")


FIRST_STEPS = [
    (1, "Read the argument", "Eight short steps, one chart each. About three minutes.",
     "web/page_story.py", "Start the story"),
    (2, "Watch a crash", "In the Test program, open **Watch a crash** and press **Biggest crash**.",
     "web/page_explore.py", "Open the Test program"),
    (3, "See what passed", "The proposal's checks, with a verdict for each, including the ones that fail.",
     "web/page_checks.py", "Open the Checks"),
]


CROWD_TRADERS = 500


def crowd_data():
    """The crowd picture's numbers: the default simulated history (the S&P fit, the figures' first
    seed) around its biggest one-day fall. Per day: (value traders, optimists, pessimists) out of 500
    (explore.crowd_counts), the price, and the day number. None if that history ran away in the warm-up."""
    run = c.default_run()
    mm = c.model_market("herding", run["a1"], run["a2"], run["v1"], run["days"], run["seed"])
    if mm is None:
        return None
    a, b = ex.zoom_window(mm["r"], "crash")
    counts = ex.crowd_counts(mm["z"][a:b + 1], mm["x"][a:b + 1], CROWD_TRADERS)
    price = mm["price"][a:b + 1]
    return dict(counts=counts.tolist(), price=[round(float(v), 4) for v in price], first=int(a),
                crash=int(np.argmin(mm["r"][a:b])) + 1)       # the day the biggest fall ends, in the window


# a canvas the browser animates; Python only fills in the numbers and the colors. One trader per dot,
# grouped by what each is doing that day; the price line underneath, with a marker on the day shown.
# Reduced motion (the visitor's setting): no playing, one still frame on the crash day.
CROWD_HTML = """<div class="crowd" id="crowd-box">
<canvas id="crowd-cv" width="640" height="400" role="img"
  aria-label="__N__ simulated traders, colored by what each is doing, day by day through the model's biggest crash"></canvas>
<div class="crowd-key"><span><i style="background:__FUND__"></i>Value traders <b id="crowd-f"></b></span>
<span><i style="background:__CHART__"></i>Optimists (buy) <b id="crowd-o"></b></span>
<span><i style="border:2px solid __CHART__"></i>Pessimists (sell) <b id="crowd-p"></b></span>
<span class="crowd-day">simulated day <b id="crowd-d"></b></span>
<button type="button" id="crowd-play">Pause</button></div></div>
<script>
(function () {
  const D = __DATA__, C = __COLORS__, N = __N__;
  const cv = document.getElementById("crowd-cv"); if (!cv) return;
  const ctx = cv.getContext("2d"), W = cv.width, H = cv.height, days = D.counts.length;
  const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const ZONES = [{x: 12, label: "VALUE TRADERS", col: C.fund, ring: false},
                 {x: 222, label: "OPTIMISTS: BUY", col: C.chart, ring: false},
                 {x: 432, label: "PESSIMISTS: SELL", col: C.chart, ring: true}];
  const COLS = 20, GAP = 9.6, TOP = 46;
  const dots = []; for (let i = 0; i < N; i++) dots.push({g: 0, x: 0, y: 0, tx: 0, ty: 0});
  function regroup(target) {                      // move only as many traders as the counts changed by
    const have = [0, 1, 2].map(g => dots.filter(d => d.g === g));
    for (let g = 0; g < 3; g++) while (have[g].length > target[g]) {
      const to = [0, 1, 2].find(h => have[h].length < target[h]);
      const d = have[g].splice(have[g].length - 1, 1)[0]; d.g = to; have[to].push(d);
    }
    for (let g = 0; g < 3; g++) have[g].forEach((d, i) => {
      d.tx = ZONES[g].x + 8 + (i % COLS) * GAP; d.ty = TOP + Math.floor(i / COLS) * GAP; });
  }
  let day = still ? D.crash : 0, playing = !still, last = 0;
  dots.forEach((d, i) => { const c0 = D.counts[day]; d.g = i < c0[0] ? 0 : i < c0[0] + c0[1] ? 1 : 2; });
  regroup(D.counts[day]); dots.forEach(d => { d.x = d.tx; d.y = d.ty; });
  const lo = Math.min(...D.price), hi = Math.max(...D.price);
  const sx = i => 14 + i / (days - 1) * (W - 28), sy = v => H - 14 - (v - lo) / ((hi - lo) || 1) * 64;
  const btn = document.getElementById("crowd-play");
  btn.textContent = playing ? "Pause" : "Play";
  btn.onclick = () => { playing = !playing; btn.textContent = playing ? "Pause" : "Play"; };
  function frame(t) {
    if (!cv.isConnected) return;                  // the page was redrawn: this copy stops
    if (playing && t - last > 90) { last = t; day = (day + 1) % days; regroup(D.counts[day]); }
    ctx.fillStyle = C.card; ctx.fillRect(0, 0, W, H);
    ctx.font = "600 12px 'JetBrains Mono', monospace";
    ZONES.forEach(z => { ctx.fillStyle = z.col; ctx.fillText(z.label, z.x + 6, 28);
      ctx.strokeStyle = C.grid; ctx.strokeRect(z.x, 36, 200, 250); });
    const k = still ? 1 : 0.25;
    dots.forEach(d => { d.x += (d.tx - d.x) * k; d.y += (d.ty - d.y) * k; const z = ZONES[d.g];
      ctx.beginPath(); ctx.arc(d.x, d.y, 3.3, 0, 6.283);
      if (z.ring) { ctx.strokeStyle = z.col; ctx.lineWidth = 1.4; ctx.stroke(); ctx.lineWidth = 1; }
      else { ctx.fillStyle = z.col; ctx.fill(); } });
    ctx.fillStyle = C.muted; ctx.fillText("PRICE", 14, H - 84);
    ctx.strokeStyle = C.real; ctx.lineWidth = 1.8; ctx.beginPath();
    D.price.forEach((v, i) => i ? ctx.lineTo(sx(i), sy(v)) : ctx.moveTo(sx(i), sy(v))); ctx.stroke();
    ctx.lineWidth = 1; ctx.strokeStyle = C.accent; ctx.beginPath();
    ctx.moveTo(sx(day), H - 80); ctx.lineTo(sx(day), H - 6); ctx.stroke();
    ctx.fillStyle = C.accent; ctx.beginPath(); ctx.arc(sx(day), sy(D.price[day]), 4, 0, 6.283); ctx.fill();
    const n = D.counts[day];
    document.getElementById("crowd-f").textContent = n[0];
    document.getElementById("crowd-o").textContent = n[1];
    document.getElementById("crowd-p").textContent = n[2];
    document.getElementById("crowd-d").textContent = (D.first + day).toLocaleString();
    if (playing || !still) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
</script>"""


def crowd_html(data):
    """The crowd picture's HTML and script, with the numbers and the site's colors filled in."""
    p = c.pal()
    colors = dict(fund=p["MIXC"]["fundamentalists"], chart=p["MIXC"]["chartists"], card=p["CARD"],
                  grid=p["GRID"], muted=p["MUTED"], real=p["REAL"], accent=p["ACCENT"])
    return (CROWD_HTML.replace("__DATA__", json.dumps(data)).replace("__COLORS__", json.dumps(colors))
            .replace("__N__", str(CROWD_TRADERS)).replace("__FUND__", colors["fund"])
            .replace("__CHART__", colors["chart"]))


CROWD_CSS = """<style>
.crowd {border: 1px solid __LINE__; background: __CARD__}
.crowd canvas {display: block; width: 100%; height: auto}
.crowd-key {display: flex; flex-wrap: wrap; align-items: center; gap: 6px 16px; padding: 8px 12px;
  border-top: 1px solid __LINE__; font-size: .8rem}
.crowd-key i {display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px;
  vertical-align: -1px; box-sizing: border-box}
.crowd-day {margin-left: auto; color: __MUTED__}
#crowd-play {min-height: 30px; padding: 0 12px; font-size: .8rem; background: transparent; color: inherit;
  border: 1px solid __LINE__; border-radius: 2px; cursor: pointer}
#crowd-play:hover {border-color: __ACCENT__; color: __ACCENT__}
</style>"""


def _loglog_path(xs, ys, lo=0.1, hi=30.0, floor=1e-5):
    """An SVG path for a tail curve on a small log-log box (200 x 86): moves of lo..hi sd across,
    shares of days 1 .. floor down. Points outside the box are left out."""
    pts = [(10 + (math.log10(x) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * 180,
            8 + -math.log10(y) / -math.log10(floor) * 74)
           for x, y in zip(xs, ys) if lo <= x <= hi and y >= floor]
    return " ".join(("M" if i == 0 else "L") + f"{x:.1f} {y:.1f}" for i, (x, y) in enumerate(pts))


def steps_html(spx_r, first_counts):
    """How the program answers the question, in three small pictures. Step 2's dots are the model's
    trader mix on the first day the crowd picture shows (scaled to 84 dots); step 3's curves are the
    S&P 500's measured tail and a bell curve's (explore.tail_curve, explore.bell_tail)."""
    p = c.pal()
    fund, chart, line, muted, real, bell = (p["MIXC"]["fundamentalists"], p["MIXC"]["chartists"], p["LINE"],
                                            p["MUTED"], p["REAL"], p["BELL"])
    ants = "".join(f"<circle cx='{40 + (6 + (i % 4) * 4.5) * math.cos(i * 2.4):.1f}' "
                   f"cy='{43 + (6 + (i % 4) * 4.5) * math.sin(i * 2.4):.1f}' r='2.2'/>" for i in range(22))
    ants += "".join(f"<circle cx='{152 + i * 5}' cy='{40 + (i % 2) * 7}' r='2.2'/>" for i in range(4))
    n = 84
    f_n = round(first_counts[0] / sum(first_counts) * n)
    o_n = round(first_counts[1] / sum(first_counts) * n)
    dots = ""
    for i in range(n):
        x, y = 12 + (i % 14) * 13.5, 12 + (i // 14) * 12.5
        if i < f_n:
            dots += f"<circle cx='{x}' cy='{y}' r='3.6' fill='{fund}'/>"
        elif i < f_n + o_n:
            dots += f"<circle cx='{x}' cy='{y}' r='3.6' fill='{chart}'/>"
        else:
            dots += f"<circle cx='{x}' cy='{y}' r='3' fill='none' stroke='{chart}' stroke-width='1.4'/>"
    grid = np.geomspace(0.1, 30, 120)
    tx, ty = ex.tail_curve(spx_r)
    def svg(label, body):
        # an <img> holding the drawing: Streamlit's st.html filter removes inline <svg>, but keeps images
        # (and an image can't run code)
        pic = (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 200 86' "
               f"font-family='JetBrains Mono, monospace'>{body}</svg>")
        return (f"<img alt=\"{label}\" style='width:100%;height:auto;margin:.4rem 0' "
                f"src='data:image/svg+xml;base64,{base64.b64encode(pic.encode()).decode()}'/>")
    card = ("<div class='step3'><span class='k'>{}</span><b>{}</b>{}<p>{}</p></div>")
    return f"""<style>
.steps3 {{display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: .4rem 0 1rem}}
.step3 {{border: 1px solid {line}; padding: 12px 14px; background: {p['CARD']}}}
.step3 .k {{display: block; font-family: {c.MONO}; font-size: .78rem; letter-spacing: .06em; color: {p['ACCENT']}}}
.step3 b {{display: block; margin-top: .2rem}}
.step3 p {{margin: 0; color: {p['TEXT2']}; font-size: .9rem; line-height: 1.5}}
@media (max-width: 760px) {{.steps3 {{grid-template-columns: 1fr}}}}
</style><div class='steps3'>""" + card.format(
        "STEP 1 · CHECK THE HERDING RULE", "Kirman's ants",
        svg("Ants choosing between two food sources, A and B, most of them at A",
                   f"<circle cx='40' cy='43' r='26' fill='none' stroke='{line}'/><text x='40' y='48' fill='{muted}' "
                   f"font-size='13' text-anchor='middle'>A</text><circle cx='160' cy='43' r='26' fill='none' "
                   f"stroke='{line}'/><text x='160' y='48' fill='{muted}' font-size='13' text-anchor='middle'>B</text>"
                   f"<g fill='{p['MIXC']['herding']}'>{ants}</g><path d='M70 43 Q100 20 130 43' fill='none' "
                   f"stroke='{p['ACCENT']}' stroke-dasharray='3 3'/>"),
        "Ants copy each other when they choose between two food sources. The simulation must match "
        "Kirman's exact formula before the market is built on it.") + card.format(
        "STEP 2 · BUILD THE MARKET", f"{CROWD_TRADERS} traders",
        svg("Value traders (blue) and trend chasers, optimists filled and pessimists hollow (orange)", dots),
        "Value traders buy below the true value and sell above it. Trend chasers follow the trend and copy "
        "the crowd's mood. Each trader switches to whichever kind is making more money.") + card.format(
        "STEP 3 · COMPARE", "Against the S&amp;P 500",
        svg("How rare big days are: the S&P 500's fat tail against a bell curve's",
                   f"<path d='{_loglog_path(grid, ex.bell_tail(grid))}' fill='none' stroke='{bell}' stroke-width='2' "
                   f"stroke-dasharray='3 3'/><path d='{_loglog_path(tx, ty)}' fill='none' stroke='{real}' "
                   f"stroke-width='2'/><text x='150' y='20' fill='{real}' font-size='10'>S&amp;P 500</text>"
                   f"<text x='96' y='80' fill='{bell}' font-size='10'>bell curve</text>"),
        "Same measurements on both: how rare big days are, whether wild days come in bursts, and how long "
        "between crashes.") + "</div>"


def crowd():
    """The crowd picture with its caption (nothing if the history ran away in the warm-up)."""
    data = crowd_data()
    if data is None:
        return
    p = c.pal()
    st.html(CROWD_CSS.replace("__LINE__", p["LINE"]).replace("__CARD__", p["CARD"])
            .replace("__MUTED__", p["MUTED"]).replace("__ACCENT__", p["ACCENT"]))
    st.html(crowd_html(data), unsafe_allow_javascript=True)
    n, k = data["counts"], data["crash"]
    st.caption(f"The model's {CROWD_TRADERS} traders, day by day, around its biggest one-day fall. Each dot is "
               f"one trader. Trend chasers copy each other's mood, so pessimism spreads: on the day of the fall, "
               f"optimists go from {n[k - 1][1]} to {n[k][1]} and pessimists from {n[k - 1][2]} to {n[k][2]}. "
               f"Value traders, who buy below the true value, pull the price back up.")


def page():
    st.title("Can herding explain market crashes?", anchor=False)
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        c.lead("If the news is plain bell-curve noise, can traders copying each other (herding) alone cause the fat "
               "tails, crashes and bursts of wild days that real markets show? This site runs a simulated market of "
               "500 traders to find out, and compares it with the S&P 500 (the 500 biggest US companies' stocks). "
               "It is a student project for 02-120 Programming for Scientists at CMU (Fall 2026).")
    with right:
        crowd()
    _, spx_r, spx = c.saved_asset("S&P 500")
    nums = c.numbers()
    cc = ex.crash_check(nums, spx_r.values)
    # three waits for the SAME fall: one the size of Black Monday ({worst_sd} sd against the S&P 500's typical
    # day). Before 28 Nov the cards compared a 20-sd fall, the S&P's record and the model's 8-sd fall, which
    # flattered the model (Opus review).
    st.markdown(f"**How long to wait for a fall the size of Black Monday** (19 October 1987: "
                f"{cc['worst_sd']:.0f} sd, measured against the S&P 500's typical day)")
    a, b, d = st.columns(3)
    a.metric("If daily moves followed a bell curve", f"10{_power(cc['bm_bell'])} years",
             help=f"A bell curve makes a fall this big almost impossible. Even a 20-sd fall, the size the project "
                  f"proposal used, would come once in about 10{_power(ex.bell_crash_years(20))} years.",
             border=True)
    b.metric("The S&P 500", f"{spx['days'] / ex.YEAR_DAYS:.0f} years",
             help="Black Monday happened once in the S&P 500's daily record since 1950.", border=True)
    d.metric("The herding model", f"{cc['bm_model']:,.0f} years",
             help=f"Over {len(nums['crashes']['days'])} simulated histories fed only bell-curve news: "
                  f"{cc['bm_count']} falls this big, so the number is {c.wait_range(cc)}. Far closer than a "
                  f"bell curve, but still rarer than in the S&P 500. Smaller crashes come sooner: an "
                  f"{cc['size']}-sd fall about every "
                  f"{cc['row']['model']:.0f} years.",
             border=True)

    st.subheader("How the program answers it", anchor=False)
    data = crowd_data()
    st.html(steps_html(spx_r.values, data["counts"][0] if data else [CROWD_TRADERS, 0, 0]))

    st.subheader("Your first five minutes", anchor=False)
    steps = st.columns(3)
    for col, (n, title, text, page, label) in zip(steps, FIRST_STEPS):
        with col.container(border=True):
            st.markdown(f"<span class='muted'>STEP {n}</span>", unsafe_allow_html=True)
            st.markdown(f"**{title}**  \n{text}")
            st.page_link(page, label=label, icon=":material/arrow_forward:")

    with st.container(border=True):
        st.markdown(short_answer(c.numbers(), spx, spx_r.values))
        st.page_link("web/page_checks.py", label="Read the findings (Checks)", icon=":material/arrow_forward:")

    st.header("What the model is", anchor=False)
    st.markdown(
        "The model is from Lux and Marchesi (*Nature*, 1999). 500 simulated traders each follow one of "
        "two strategies:\n\n"
        "- **Value traders** (economists call them fundamentalists) buy when the price is below the true "
        "value and sell when it is above.\n"
        "- **Trend chasers** (economists call them chartists) ignore value and follow the trend. They are "
        "either optimists or pessimists, and they tend to copy whatever mood most trend chasers are in. "
        "That copying is **herding**.\n\n"
        "Each trader switches to whichever kind is making more money. News moves the true value by "
        "bell-curve amounts, so if big days show up, the traders made them.")

    st.header("How to use it", anchor=False)
    manual = st.expander("The full manual: every button and chart", icon=":material/menu_book:")
    manual.markdown(
        "1. **Story** walks through the argument in eight short steps. Read this first.\n"
        "2. **Test program** is where you try it yourself on stocks and other markets.\n"
        "    - Everything that is not on the page itself is in **⚙ Settings**, the sidebar on the left (the ☰ "
        "button at the top left on a phone).\n"
        "    - Pick a market in **⚙ Settings**: choose a group (Indexes, Stocks or Commodities), then click a market. "
        "Or type any stock symbol (a market's short code, for example `NVDA` for Nvidia) in the box beside it and press Enter.\n"
        "    - Use the range buttons (1D, 7D, YTD, 1Y, 5Y, 20Y, All) to change the time span. "
        "1D is one day and 7D is one week.\n"
        "    - 1D and 7D show prices within the day from Yahoo Finance: always for a ticker you type, and for a "
        "market from the list once **Live prices** is on. That switch is in **⚙ Settings** too.\n"
        "    - **Three things to try** at the top gives you three tasks, with a tick as you do each one.\n"
        "    - Switch the chart between four tabs: **Price**, **Watch a crash**, **How rare big days are** (with a "
        "Tail / Histogram switch) and **Bursts of wild days** (with a Bursts / Memory switch). Each has a **Try "
        "this** line under it. Move the mouse over a chart "
        "(or touch it on a phone) to read the exact values under it. Drag to zoom, double-click to zoom back "
        "out, and click a legend entry to hide a line.\n"
        "    - **Presets** in the model panel on the right fill in the model's controls for you: **Wilder**, "
        "**Best fit** (the S&P 500 fit), **Calmer** or **No herding**. Each says what to expect: with too little "
        "herding the price runs away, and more herding calms it. **Reset everything** (in ⚙ Settings) takes you "
        "back to the start.\n"
        "    - Move the herding strength on the right: the model reruns by itself (turn off **Run "
        "automatically** to use the **Run model** button instead). Each simulated history is added to the chart under "
        "**Every history on one chart**, so you can see how the tail exponent changes as you turn the dial.\n"
        "    - **Watch a crash** shows the model's own price, its true value and its traders day by day. "
        "**Biggest crash** and **Wildest year** jump straight to them.\n"
        "3. **Checks** shows the proposal's checks in its order, with a scoreboard at the top and the "
        "rule for each one. Check 4 asks whether crashes come within decades. Its tabs hold more: **Beyond the "
        "paper** compares herding with a formula that has no traders, **All markets** puts every saved market "
        "on one map next to the model, and **Figures** collects the project's figures, which are averaged over "
        "many simulated histories.")

    st.header("How to read the charts", anchor=False)
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown(
            "**Tail** is the main chart. Across: how big a day's move was, in sd "
            "(a standard deviation is the usual size of a day's move). Up: the share of days with a move at "
            "least that big. Both axes use a **log scale**: each step along an axis is 10 times the last, so "
            "the rarest days still fit on the chart.\n\n"
            "- A bell curve drops away: a 5-sd day should come once in thousands of years.\n"
            "- The S&P 500 falls along a straight line instead. Its slope is the **tail exponent**, near 3 "
            "(lower = more big days). That straight line is the *inverse cubic law*: a move twice as big is about "
            "2³ = 8 times rarer.\n\n"
            "**Histogram** shows the same thing as a shape: a market with fat tails has a taller middle and longer "
            "tails than the bell curve.\n\n"
            "**Bursts** shows each day's move over time. In real markets wild days come in bursts.\n\n"
            "**Memory** measures those bursts: how much a big move today predicts big moves "
            "1 to 200 days later.")
    with right:
        bell, fat = ex.example_returns()
        fig = go.Figure()
        for label, r, line in (("Bell curve", bell, dict(color=c.BELL, dash="dot", width=2.5)),
                               ("Fat tails (like a real market)", fat, dict(color=c.REAL, width=2.5))):
            x, y = ex.tail_curve(r)
            fig.add_scatter(x=x, y=y, name=label, mode="lines", line=line, hoverinfo="skip")
        c.style(fig, 330, logx=True, logy=True, xtitle="Size of the day's move (sd)",
                ytitle="Share of days at least this big", hover="closest")
        c.tail_axes(fig)     # down to 0.001%: 20,000 days reach 1/20,000 = 0.005%, which ymin=-4 cut off
        c.show(fig, key="guide_example")
        st.caption("An example with made-up numbers: the two curves agree for small moves and split apart "
                   "for big ones.")

    st.header("The controls", anchor=False)
    st.markdown(
        "- **Herding strength (a1):** how strongly trend chasers copy the crowd's mood. 0 means not at all.\n"
        "- **Reaction to price trends (a2):** how strongly trend chasers react to recent price moves.\n"
        "- **How often trend chasers change mood (v1):** how often trend chasers reconsider whether they are optimistic.\n"
        "- **Simulated trading days and seed:** how long a simulated history lasts, and the number that picks "
        "one simulated history. The same settings and seed always give exactly the same history.\n\n"
        f"Every other setting stays at the values fitted to the S&P 500 ({gh('settings/sp500_calibrated.json')}).")

    st.header("Words used on the site", anchor="words")
    st.markdown(
        "- **Index:** one number that follows many companies' prices at once, like the S&P 500 "
        "(500 large US companies).\n"
        "- **Ticker:** a market's short code on the stock exchange, like `AAPL` for Apple.\n"
        "- **Standard deviation (sd):** the usual size of a day's move. A 4-sd day moves four times as "
        "much as usual.\n"
        "- **Big day:** a day that moves more than 4 sd, up or down.\n"
        "- **Fat tails:** big days come far more often than a bell curve predicts.\n"
        f"- **Tail exponent:** {c.HELP['tail']}\n"
        f"- **Excess kurtosis:** {c.HELP['kurtosis'].removeprefix('Excess kurtosis. ')}\n"
        "- **Bursts of wild days:** wild days come in bunches, not alone (economists call this volatility "
        "clustering). A **wild day** is a day that moves much more than usual.\n"
        f"- **Memory of wild days:** {c.HELP['memory']}\n"
        "- **Hurst H:** another memory score, used on the Checks page. 0.5 means no memory; above 0.5 "
        "means what happened keeps happening; below 0.5 means it tends to reverse.\n"
        "- **Median:** the middle value: half the simulated histories are above it and half below.\n"
        "- **Seed / held-out:** the model is random, and a number called a seed picks one simulated history. "
        "The same seed always gives the same history. Held-out seeds were never used to fit the settings, "
        "so they test the model fairly.\n"
        "- **Log scale:** an axis where each step is 10 times the last (1, 10, 100, ...), so the rarest "
        "days still fit.\n"
        "- **Drawdown:** how far the price has fallen from its last high.\n"
        "- **Annualized volatility:** how much the price typically moves in a year.\n"
        "- **Runaway:** the simulated price leaves its true value and never comes back, a bubble or fall "
        "that does not end. We say when it happens instead of hiding it.")

    st.header("Things to keep in mind", anchor=False)
    st.markdown(
        "- **One simulated history is one random draw.** The Test program shows one simulated history, so its "
        f"numbers move around when you change the seed. The Checks page and the figures use {len(c.numbers()['recipe']['seeds'])} simulated "
        "histories and report the median, which is why their numbers are steadier.\n"
        "- **Everything starts from the S&P 500 fit.** Other markets are compared on shape only (tail "
        "exponent, kurtosis, memory), which does not depend on how volatile a market is.\n"
        "- **This is not investment advice.** The model explains the shape of returns, not where a price goes next.\n"
        "- **Data:** saved prices end on 30 October 2026 (Yahoo Finance, FRED, MeasuringWorth). Live and "
        "intraday prices come from Yahoo Finance and are sometimes unavailable.")

    st.header("Run it yourself", anchor=False)
    st.markdown(f"The code is public on GitHub: [{REPO.removeprefix('https://')}]({REPO}).")
    st.code(f"git clone {REPO}.git\ncd can-herding-explain-market-crashes\n"
            "pip install -r requirements.txt\nstreamlit run app.py", language="bash")
    st.markdown(f"Tests: `python3 -m pytest tests/ -q`. Every number on the site comes from "
                f"{gh('herding/explore.py')}, which is tested in {gh('tests/test_explore.py')}.")


page()
