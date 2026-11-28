"""Checks: the proposal's checks, each with its rule and its numbers, in four tabs (the redesign,
docs/redesign/README.md): the five checks; Beyond the paper (the textbook checklist against GARCH, the
paper comparison); All markets (was the Compare page); Figures (was Findings).

Model numbers come from web/numbers.json (scripts/web_numbers.py): the same 40 held-out histories as
the project's figures, so this page and figures/RESULTS.md agree. Kirman's ants run live.
"""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from herding import explore as ex
from herding import kirman
from web import common as c
from web import page_compare, page_findings

CHECK_NAMES = ["1. Kirman's ants", "2. No trend chasers", "3. Tail exponent", "4. Crashes",
               "5. Big days and bursts of wild days"]   # the scoreboard's rows, each a link to its check
TABS = ["The five checks", "Beyond the paper", "All markets", "Figures"]
RESULTS = "https://github.com/avishj19/can-herding-explain-market-crashes/blob/main/figures/RESULTS.md"

KIRMAN = dict(N=100, delta=0.1)
STEPS = 20_000_000


@st.cache_data(show_spinner=False)
def _ants(eps, seed=1):
    k = kirman.simulate(KIRMAN["N"], eps, KIRMAN["delta"], STEPS, seed=seed)
    sim, exact, tv = ex.kirman_match(k, KIRMAN["N"], eps, KIRMAN["delta"])
    return ex.kirman_path(k), sim, exact, tv


def _ant_words(eps, th):
    """What the ants do at this eps. Just above the threshold the exact distribution is almost flat
    (at eps = 0.01: as much time at 0 or 100 ants as at 50), so "split about evenly" only from about
    twice the threshold (at 0.02 the middle is 45 times likelier than the ends)."""
    if eps < th:
        return "herd"
    if eps < 2 * th:
        return "wander across the whole range: no strong herding, but no even split either"
    return "split about evenly"


def _dots(seeds, values, color, band, ref=None, ref_label="", ytitle=""):
    fig = go.Figure()
    fig.add_hrect(y0=band[0], y1=band[1], fillcolor=c.OK_FILL, line_width=0, layer="below")
    fig.add_scatter(x=seeds, y=values, mode="markers", marker=dict(size=8, color=color), name="One history",
                    hovertemplate="seed %{x}: %{y:.3f}<extra></extra>")
    fig.add_hline(y=ex.spread(values)["median"], line=dict(color=color, width=1.5))   # the median (stated beside the chart)
    if ref is not None:
        fig.add_hline(y=ref, line=dict(color=c.REAL, width=1.5, dash="dash"), annotation_text=ref_label,
                      annotation_position="top left", annotation_font=dict(size=11, color=c.REAL))
    return c.style(fig, 260, xtitle="History number (seed)", ytitle=ytitle, legend=False)


def _years(y):
    """A waiting time in plain words: '35 years', '1,534 years', '4 million years', '1.4 × 10⁸⁶ years'."""
    if math.isinf(y):
        return "never (none seen)"
    if y < 10:
        return f"{y:.1f} years"
    if y < 1e6:
        return f"{y:,.0f} years"
    if y < 1e12:
        for unit, size in (("trillion", 1e12), ("billion", 1e9), ("million", 1e6)):
            if y >= size:
                return f"{y / size:.0f} {unit} years"
    e = math.floor(math.log10(y))
    return f"{y / 10 ** e:.1f} × 10{c.superscript(e)} years"


AGE_OF_UNIVERSE = 1.38e10     # years


def _crash_chart(rows, cc):
    """Years between crashes against crash size: bell curve, S&P 500 and the model (log scale)."""
    fig = go.Figure()
    fig.add_hrect(y0=0.1, y1=cc["limit"], fillcolor=c.OK_FILL, line_width=0, layer="below")
    fig.add_annotation(x=0.01, y=math.log10(cc["limit"]), xref="x domain", yref="y", yanchor="top",
                       text=f"within {cc['limit']} years", showarrow=False, font=dict(size=11, color=c.MUTED))
    fig.add_hline(y=AGE_OF_UNIVERSE, line=dict(color=c.GUIDE, dash="dot", width=1),
                  annotation_text="age of the universe", annotation_position="top left",
                  annotation_font=dict(size=11, color=c.MUTED))
    sizes = [r["size"] for r in rows]
    fig.add_scatter(x=sizes, y=[r["bell"] for r in rows], mode="lines+markers", name="Bell curve",
                    line=dict(color=c.BELL, dash="dot", width=2.5),
                    hovertemplate="%{x} sd fall: every %{y:.2g} years<extra>Bell curve</extra>")
    for key, label, line in (("model", "Model", c.mix_line("herding", 2.5)),
                             ("sp500", "S&P 500", dict(color=c.REAL, width=2.5))):
        pts = [(r["size"], r[key]) for r in rows if not math.isinf(r[key])]
        fig.add_scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="lines+markers", name=label,
                        line=line, hovertemplate=f"%{{x}} sd fall: every %{{y:,.1f}} years<extra>{label}</extra>")
    top = 1e14                # just above the age of the universe; the bell curve leaves the chart here
    gone = next((r for r in rows if r["bell"] > top), None)
    if gone is not None:
        fig.add_annotation(x=gone["size"], y=math.log10(top), yref="y", text=(
            f"Bell curve goes off the chart:<br>a {rows[-1]['size']}-sd fall every {_years(rows[-1]['bell'])},<br>"
            f"Black Monday ({cc['worst_sd']:.0f} sd) every {_years(cc['bm_bell'])}"),
            showarrow=False, xanchor="left", yanchor="top", align="left", font=dict(size=11, color=c.MUTED))
    c.style(fig, 400, logy=True, xtitle="Size of the daily fall (sd)",
            ytitle="Years between such falls", hover="closest")
    fig.update_yaxes(range=[-1, math.log10(top)])
    return fig


@st.cache_data(show_spinner=False)
def _spx_tail_band(stamp):
    """The S&P 500's own 95% interval for its tail exponent (explore.tail_interval), once per data file."""
    return ex.tail_interval(ex.log_returns(ex.saved_prices("S&P 500")).values)


def _scoreboard(rows):
    """The proposal's checks in one table (wrapping its text): what the proposal said, what came out, the verdict."""
    st.table(pd.DataFrame(rows).set_index("Check"))


def page():
    nums = c.numbers()
    rec = nums["recipe"]
    _, _, spx = c.saved_asset("S&P 500")
    seeds = rec["seeds"]

    st.title("Checks", anchor=False)
    c.lead(f"Each check below says in one sentence what it tests, then gives its rule and the numbers. The "
           f"model numbers come from {len(seeds)} simulated histories of {rec['days']:,} days each. Each "
           f"history is picked by a number called a seed; these {len(seeds)} were never used to fit the "
           f"settings. They are the same histories as the project's figures. The checks follow the proposal's "
           f"order: Kirman's ants first, then the market model. The other tabs go beyond the proposal, compare "
           f"every saved market, and show the project's figures.")
    spx_r = c.saved_asset("S&P 500")[1].values
    rows = ex.crash_table(nums, spx_r)
    cc = ex.crash_check(nums, spx_r)
    pooled_years = sum(nums["crashes"]["days"]) / ex.YEAR_DAYS
    spx_years = len(spx_r) / ex.YEAR_DAYS
    tab_checks, tab_beyond, tab_markets, tab_figures = st.tabs(TABS)
    with tab_checks:
        st.subheader("Scoreboard", anchor=False)
        board = st.container()
        st.markdown("Open a check: " + " · ".join(
            f"<a class='jump' href='#check-{i}'>{name}</a>" for i, name in enumerate(CHECK_NAMES, 1)),
            unsafe_allow_html=True)
        st.divider()

        # 3 ------------------------------------------------------------------
        st.header("1. Kirman's ants match his exact formula", anchor="check-1")
        th = kirman.herding_threshold(KIRMAN["N"], KIRMAN["delta"])
        left, right = st.columns([1, 2], gap="large")
        with left:
            st.markdown("The proposal's first step: the herding rule inside the model should behave exactly as the "
                        "formula it comes from says.")
            st.markdown(f"The herding rule inside the model comes from Kirman's ant model. {KIRMAN['N']} ants "
                        f"choose between two food sources; each step one ant either switches on its own (chance ε) or "
                        f"copies another ant. Kirman worked out the exact long-run answer.")
            eps = st.select_slider("Chance an ant switches on its own (epsilon, ε)",
                                   [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1], value=0.002,
                                   key="eps")
            (steps, hist_k), sim, exact, tv = _ants(eps)
            st.markdown(f"Below ε = {th:.4f} the ants herd: most sit at one source and switch together. "
                        f"Now ε = {eps}, so they {_ant_words(eps, th)}.")
            st.markdown(f"**Rule:** over {STEPS / 1e6:.0f} million steps, the share of time the simulation spends at "
                        f"each count of ants is within {ex.KIRMAN_PASS} of the exact answer (0 = perfect match). "
                        f"Gap **{tv:.4f}**: the check {c.verdict(ex.kirman_passes(tv))}.",
                        unsafe_allow_html=True)
            with st.expander("How the rule is computed"):
                st.markdown("The distance is the *total variation distance*: add up the gaps between the simulated "
                            "and the exact share of time at every count (0 to "
                            f"{KIRMAN['N']} ants) and halve the sum. It is 0 for identical shares and 1 for shares "
                            "that never overlap.")
        with right:
            fig = make_subplots(rows=2, cols=1, vertical_spacing=0.18)
            fig.add_scatter(x=steps / 1e6, y=hist_k, mode="lines", name="Simulation",
                            line=dict(color=c.MIX["herding"][2], width=1), row=1, col=1)
            fig.add_bar(x=np.arange(KIRMAN["N"] + 1), y=sim, name="Simulated share of time",
                        marker_color=c.MIX["herding"][2], opacity=0.55, row=2, col=1)
            fig.add_scatter(x=np.arange(KIRMAN["N"] + 1), y=exact, mode="lines", name="Kirman's exact answer",
                            line=dict(color=c.REAL, width=2), row=2, col=1)
            c.style(fig, 420)
            fig.update_layout(bargap=0)
            fig.update_xaxes(title="Steps (millions)", row=1, col=1)
            fig.update_yaxes(title="Ants at source A", row=1, col=1)
            fig.update_xaxes(title="Ants at source A", row=2, col=1)
            fig.update_yaxes(title="Share of time", row=2, col=1)
            c.show(fig, key="c3")
        st.divider()

        # 1 ------------------------------------------------------------------
        k = nums["checks"]["fundamentalists"]["kurtosis"]
        res = ex.bell_check(k, rec["days"])
        st.header("2. With no trend chasers, daily moves follow a bell curve", anchor="check-2")
        left, right = st.columns([1, 2], gap="large")
        with left:
            st.markdown("With only value traders, the simulated market's daily moves should follow a bell curve, like its news.")
            st.markdown(f"**Rule:** every simulated history's excess kurtosis (0 for a bell curve) is within "
                        f"±{res['tol']:.3f} of 0.")
            with st.expander("How the rule is computed"):
                st.markdown(f"Even a perfect bell curve gives an excess kurtosis a little away from 0 in a finite sample. "
                            f"The usual size of that wobble (its standard error) is √(24/n) for n days. The rule "
                            f"allows 5 standard errors: for n = {rec['days']:,} days, ±5·√(24/n) = "
                            f"±{res['tol']:.3f}.")
            st.markdown(f"Median **{ex.spread(k)['median']:+.3f}**; farthest from 0: **{res['worst']:+.3f}**. "
                        f"The check {c.verdict(res['passed'])}.", unsafe_allow_html=True)
        with right:
            fig = _dots(seeds, k, c.MIX["fundamentalists"][2], (-res["tol"], res["tol"]), ytitle="Excess kurtosis (0 = bell curve)")
            fig.update_yaxes(range=[-1.6 * res["tol"], 1.6 * res["tol"]])
            c.show(fig, key="c1")
        st.divider()

        # 2 ------------------------------------------------------------------
        tails = nums["checks"]["herding"]["tail"]
        band = _spx_tail_band(c.file_stamp(ex.market_data.SPX))       # the S&P 500's own 95% interval
        tres = ex.tail_check(tails, close=band)   # skips the histories that ran away (None) and counts them
        lo, hi = ex.PROPOSAL_TAIL
        st.header("3. With herding, the tail exponent is about 3", anchor="check-3")
        left, right = st.columns([1, 2], gap="large")
        with left:
            st.markdown("When trend chasers copy each other, big days should become about as common as in the "
                        "S&P 500.")
            st.markdown(f"**Rule (the proposal's):** the median tail exponent (lower = more big days) lies between "
                        f"{lo} and {hi}. The inverse cubic law (Gopikrishnan et al. 1998) says about 3.")
            with st.expander("How the rule is computed"):
                st.markdown("Each history's tail exponent is the Hill estimate over its largest 5% of days (by "
                            "size, up or down). The rule uses the median over the simulated histories that did not "
                            "run away. **Close** means the median misses the proposal's range but sits inside the "
                            "S&P 500's own uncertainty: the range its tail exponent takes over 500 resamples of its "
                            "history (in runs of 50 days, so bursts stay together), middle 95%.")
            # passes / does not pass in the site's ✓ ✗ style (common.verdict); "close" in words, between the two
            verdict = (c.verdict(tres["passed"]) if tres["passed"] or not tres["close"]
                       else "is <b>close</b>")
            near = tres["passed"] or tres["close"]           # "just" only when it is inside the S&P's own range
            where = ("inside" if tres["passed"] else ("just " if near else "") + ("above" if tres["median"] > hi
                                                                                   else "below"))
            st.markdown(f"Model median **{tres['median']:.2f}** (range {tres['low']:.2f} to {tres['high']:.2f}) over "
                        f"all {tres['n']} simulated histories that did not run away; "
                        f"S&P 500 **{spx['tail']:.2f}**. The check {verdict}: the median is {where} the "
                        f"proposal's {lo} to {hi}; the S&P 500's own range is {band[0]:.2f} to {band[1]:.2f}.",
                        unsafe_allow_html=True)
            st.caption("Not a fully independent test: the model's settings were fitted to the S&P 500, and this "
                       "tail exponent was one of the fit's 11 targets (herding/calibrate.py, stats.FINGERPRINT: also "
                       "excess kurtosis, the typical move, how move sizes and directions correlate across days, Hurst "
                       "H, and kurtosis over 5 and 25 days). Fully independent of the fit: check 1 (Kirman's ants) and "
                       "check 2 (no trend chasers gives a bell curve). Check 4 (how often crashes come) was not a "
                       "target, but it follows largely from the fitted tail and kurtosis.")
            if tres["left_out"]:
                st.caption(f"{tres['left_out']} simulated histories ran away and are left out.")
        with right:
            fig = _dots([s for s, t in zip(seeds, tails) if t is not None], [t for t in tails if t is not None],
                        c.MIX["herding"][2], ex.PROPOSAL_TAIL,
                        ref=spx["tail"], ref_label="S&P 500", ytitle="Tail exponent")
            fig.update_yaxes(range=[2.2, 4.2])
            c.show(fig, key="c2")
        st.divider()

        # crashes ------------------------------------------------------------
        # the proposal's number, worked out (20 sd: about 1.4 × 10⁸⁶ years), never typed in
        p20 = c.superscript(math.floor(math.log10(ex.bell_crash_years(20))))
        st.header(f"4. Crashes come within decades, not 10{p20} years", anchor="check-4")
        left, right = st.columns([1, 2], gap="large")
        with left:
            # a 20-sd fall is the proposal's 10⁸⁶ years; Black Monday was bigger (about 23 sd), so rarer still
            st.markdown(f"The proposal's question: a bell curve says a 20-sd fall comes about once in 10{p20} "
                        f"years, and Black Monday ({cc['worst_sd']:.0f} sd) even more rarely: once in "
                        f"{_years(cc['bm_bell'])}. With herding, do falls like these come within a few decades, "
                        f"as in the S&P 500?")
            st.markdown(f"**Rule, in two parts:** (a) falls of {cc['size']} sd, which a bell "
                        f"curve puts {_years(cc['row']['bell'])} apart, come in the model at least once every "
                        f"{cc['limit']} years; (b) so does a fall as big as Black Monday.")
            st.markdown(f"(a) The model has {'an' if cc['size'] in (8, 11, 18) else 'a'} {cc['size']}-sd fall every **{_years(cc['row']['model'])}** "
                        f"(S&P 500: every {_years(cc['row']['sp500'])}): {c.verdict(cc['decades'])}.",
                        unsafe_allow_html=True)
            st.markdown(f"(b) Black Monday was a {cc['worst_sd']:.0f}-sd fall. Bell curve: once every "
                        f"**{_years(cc['bm_bell'])}**. Model: every **{_years(cc['bm_model'])}** "
                        f"({cc['bm_count']} such falls in {_years(pooled_years)} of simulated market; "
                        f"{c.wait_range(cc)}). Even the shortest of those waits is far more than {cc['limit']} years: "
                        f"{c.verdict(cc['black_monday'])}.", unsafe_allow_html=True)
            st.caption(f"The proposal's 10{p20} years is for a 20-sd fall ({_years(ex.bell_crash_years(20))} here). "
                       f"Measured against the S&P 500's typical day since 1950, Black Monday was "
                       f"{cc['worst_sd']:.0f} sd, which a bell curve makes rarer still.")
            if not cc["passed"] and cc["decades"] and not math.isinf(cc["bm_model"]):
                st.markdown(f"**Overall: partly.** Herding brings the wait for a Black Monday down by about "
                            f"{math.log10(cc['bm_bell'] / cc['bm_model']):.0f} powers of ten, but not all the way to "
                            f"the S&P 500's one in {_years(spx_years)}.")
            else:
                st.markdown(f"**Overall: {'passes' if cc['passed'] else 'does not pass'}.**")
            with st.expander("How the rule is computed"):
                st.markdown("A crash is a daily fall of at least that many sd (each history "
                            "measured in its own sd). Years between crashes = trading days ÷ 252 "
                            "÷ number of crashes, with all held-out histories pooled. For the bell curve it is "
                            "1 ÷ (chance of such a fall on one day × 252).")
        with right:
            c.show(_crash_chart(rows, cc), key="c_crash")
        st.divider()

        # 4 ------------------------------------------------------------------
        st.header("5. Big days and bursts of wild days, next to the S&P 500", anchor="check-5")
        st.markdown("The model's big days and bursts of wild days should be about as common and as strong as the S&P 500's.")
        st.markdown("The proposal asks for numbers comparable to the S&P 500, so there is no pass or fail line "
                    "here. The model column is the median over the same histories, with the middle 90% of them in "
                    "brackets (unlike check 2, whose range covers all of them).")
        t = ex.check_table(nums, spx)
        label = {"tail": ("Tail exponent", "{:.2f}"), "kurtosis": ("Excess kurtosis", "{:.1f}"),
                 "memory": ("Memory of wild days (20 days)", "{:.2f}"), "worst": ("Worst day (sd)", "{:.1f}")}
        rows = [{"Measure": label[m][0],
                 "Model: median (middle 90%)": f"{label[m][1].format(v['median'])} ({label[m][1].format(v['low'])} "
                                               f"to {label[m][1].format(v['high'])})",
                 "S&P 500": label[m][1].format(v["sp500"])} for m, v in t.iterrows()]
        meaning = "; ".join(f"{label[m][0]}: {c.HELP[m]}" for m in t.index if m in c.HELP)
        st.dataframe(rows, hide_index=True, width="stretch", column_config={
            "Measure": st.column_config.TextColumn("Measure", help=meaning),
            "Model: median (middle 90%)": st.column_config.TextColumn(
                "Model: median (middle 90%)", help="Median over the held-out histories; in brackets, the range "
                                                   "of the middle 90% of them (5th to 95th percentile)."),
            "S&P 500": st.column_config.TextColumn("S&P 500", help="The S&P 500 since 1950, measured the same way.")})
        st.markdown("\n".join(f"- **{label[m][0]}:** {c.HELP[m]}" for m in t.index if m in c.HELP))
        hk = t.loc["kurtosis", "median"]
        k_wo = ex.kurtosis_without_worst(spx_r)                       # the S&P without Black Monday
        wo = ex.spread(nums["checks"]["herding"]["kurtosis_wo_worst"])  # each history without its own worst fall
        where = "inside" if wo["low"] <= k_wo <= wo["high"] else "above" if k_wo > wo["high"] else "below"
        st.caption(f"Excess kurtosis is the weak spot: the S&P 500's {spx['kurtosis']:.0f} is driven mostly by one "
                   f"day (Black Monday: each day counts to the 4th power, so one {abs(spx['worst']):.0f}-sd day adds "
                   f"about {(abs(spx['worst']) ** 4) / len(spx_r):.0f}). Without that day it is {k_wo:.1f}. Compared "
                   f"like for like, with each simulated history also losing its own worst fall, the model's middle "
                   f"90% is {wo['low']:.1f} to {wo['high']:.1f} (median {wo['median']:.1f}): the S&P 500 is {where} "
                   f"it, so the model's days are still a little too tame. The project's [report]({RESULTS}) discusses this.")

    with tab_beyond:
        # 6 ------------------------------------------------------------------
        facts = ex.fact_table(nums, spx_r)
        gp = nums["facts"]["garch_params"]
        n_of = facts[0]["of"]
        herd_n = sum(f["herding_pass"] > n_of / 2 for f in facts)
        garch_n = sum(f["garch_pass"] > n_of / 2 for f in facts)
        st.header("The textbook checklist, against a formula with no traders", anchor=False)
        left, right = st.columns([1, 2], gap="large")
        with left:
            st.markdown("Beyond the proposal: economists keep a checklist of patterns found in nearly every "
                        "market (Cont, 2001). How many does herding reproduce?")
            st.markdown("And a fair rival: **GARCH**, a formula with no traders, fitted to the same S&P 500. "
                        "It also gets only bell-curve news, but its bursts are written into the formula: a big "
                        "move makes the next day's typical move bigger.")
            st.markdown(f"Out of {len(facts)}, the model passes **{herd_n}** (in most of its {n_of} "
                        f"histories), the formula **{garch_n}**, and the S&P 500 "
                        f"**{sum(f['sp500_pass'] for f in facts)}**.")
            st.markdown("**What that means:** herding does not beat the formula on these numbers. What it adds is "
                        "a reason: the bursts come out of traders copying each other, not out of an equation "
                        "chosen to have bursts. Both miss the same two patterns, which point to what the model "
                        "still lacks.")
            with st.expander("How the rules and the formula work"):
                st.markdown("Each rule's line is set where a plain bell curve cannot reach it. The formula is "
                            "GJR-GARCH(1,1) (Bollerslev 1986; Glosten, Jagannathan & Runkle 1993): tomorrow's "
                            f"variance = {gp['omega']:.2g} + ({gp['alpha']:.3f} + {gp['gamma']:.3f} after a fall) × "
                            f"today's squared move + {gp['beta']:.3f} × today's variance, fitted by maximum "
                            f"likelihood, then run on the same {n_of} seeds. A count like 37/40 is how many "
                            "histories pass.")
        with right:
            def mark(ok):
                return "✅" if ok else "❌"
            st.table(pd.DataFrame([{
                "Pattern": f["name"],
                "S&P 500": mark(f["sp500_pass"]),
                "Model": f"{mark(f['herding_pass'] > n_of / 2)} {f['herding_pass']}/{n_of}",
                "Formula (GARCH)": f"{mark(f['garch_pass'] > n_of / 2)} {f['garch_pass']}/{n_of}",
                "Rule": f["rule"]} for f in facts]).set_index("Pattern"))

        st.divider()
        page_findings.paper_section()
    with tab_markets:
        page_compare.body()
    with tab_figures:
        page_findings.figures()

    with board:
        _scoreboard([
            {"Check": "1. Kirman's ants", "The proposal expected": "Reproduce Kirman's ant model first, against his exact formula",
             "Result": f"Gap from his exact answer {tv:.4f} (needs under {ex.KIRMAN_PASS})",
             "Verdict": "✅ Passes" if ex.kirman_passes(tv) else "❌ Fails"},
            {"Check": "2. No trend chasers", "The proposal expected": "With zero trend chasers, a bell curve",
             "Result": f"Excess kurtosis {ex.spread(k)['median']:+.3f} (median), every history within ±{res['tol']:.3f}",
             "Verdict": "✅ Passes" if res["passed"] else "❌ Fails"},
            {"Check": "3. Tail exponent", "The proposal expected": "With trend chasers, about 2.6 to 3 (inverse cubic law)",
             "Result": f"Model {tres['median']:.2f} (median), S&P 500 {spx['tail']:.2f}",
             "Verdict": "✅ Passes" if tres["passed"] else ("🟡 Close" if tres["close"] else "❌ Fails")},
            {"Check": "4. Crashes", "The proposal expected": f"Crashes within a few decades, not 10{p20} years",
             "Result": f"{'An' if cc['size'] in (8, 11, 18) else 'A'} {cc['size']}-sd fall every {_years(cc['row']['model'])}; "
                       f"a fall the size of Black Monday every "
                       f"{_years(cc['bm_model'])} (bell curve {_years(cc['bm_bell'])})",
             "Verdict": "✅ Passes" if cc["passed"] else ("🟡 Partly" if cc["decades"] else "❌ Fails")},
            {"Check": "5. Big days and bursts of wild days", "The proposal expected": "Comparable to the S&P 500",
             "Result": f"Worst day {t.loc['worst', 'median']:.1f} sd vs {t.loc['worst', 'sp500']:.1f}; "
                       f"memory of wild days {t.loc['memory', 'median']:.2f} vs {t.loc['memory', 'sp500']:.2f}",
             "Verdict": "ℹ️ No pass mark set"}
        ])


page()
