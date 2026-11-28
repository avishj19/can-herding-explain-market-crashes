"""Story: the argument in 8 guided steps (the redesign, docs/redesign/README.md item 4).

One idea per screen: a headline, a few plain sentences, one big number and one chart. Next / Back
buttons, a clickable step list and a progress bar; the current step is st.session_state["story_step"]
(0 to 7). Every number comes from herding/explore.py, web/common.py or the saved histories
(web/numbers.json); nothing is typed in by hand. Wording follows docs/redesign/WORDING.md.
"""
import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from herding import explore as ex
from herding import kirman
from web import common as c

# step 7's question: each answer and what the saved sweep says about it
GUESSES = {
    "Wilder": "Not in this model: more copying makes it calmer. The wildest markets copy the least, "
              "and they run away.",
    "Wilder, then calmer": "Not quite: it gets calmer all the way up. But the S&P 500 does sit at "
                           "one point, as you will see.",
    "Calmer": "Yes, surprisingly. That is what the saved histories show.",
}

# the steps: (short name for the step list, headline that states the finding)
STEPS = [
    ("The puzzle", "Real markets have days a bell curve says should never happen"),
    ("The idea", "The idea: traders copy each other, like ants do"),
    ("The market", "So we build a market of simulated traders and feed it bell-curve news"),
    ("Experiment 1", "Experiment 1: with only value traders, daily moves follow a bell curve"),
    ("Experiment 2", "Experiment 2: without herding, trend chasers make the price run away"),
    ("Experiment 3", "Experiment 3: with herding, the model gets fat tails and bursts"),
    ("How much?", "How much herding? Guess first"),
    ("The verdict", "The verdict: herding explains a lot, not everything"),
]
LAST = len(STEPS) - 1

# Kirman's ants, as on the Checks page (check 1): 100 ants, herding strong enough to see them switch together
ANTS = dict(N=100, eps=0.002, delta=0.1, steps=20_000_000, seed=1)


def ten_to(years):
    """'10⁸⁶' for a number of years of about 10 to the 86, with the power taken from the number."""
    return f"10{c.superscript(math.floor(math.log10(years)))}"


def step_now():
    """The current step, 0 to LAST (a stray value in the session is pulled back into range)."""
    return min(max(int(st.session_state.get("story_step", 0)), 0), LAST)


def move(step=None, by=0):
    """Move to step `step`, or `by` steps from here; never outside 0..LAST (Back on step 1 and Next on
    step 8 stay where they are)."""
    here = step_now() if step is None else int(step)
    st.session_state["story_step"] = min(max(here + by, 0), LAST)


@st.cache_data(show_spinner=False)
def _ants(N, eps, delta, steps, seed):
    k = kirman.simulate(N, eps, delta, steps, seed=seed)
    sim, exact, tv = ex.kirman_match(k, N, eps, delta)
    return ex.kirman_path(k), sim, exact, tv


def story_css():
    p = c.pal()
    a, line, panel, mono = p["ACCENT"], p["LINE"], p["PANEL"], c.MONO
    return f"""<style>
    .st-key-story_steps button {{justify-content: flex-start; text-align: left; border-radius: 2px; min-height: 2.5rem;
        width: 100%; border: 1px solid transparent; background: transparent}}
    .st-key-story_steps button p {{font-family: {mono}; font-size: .82rem !important; color: {p['MUTED']};
        text-align: left}}
    .st-key-story_steps button:hover {{border-color: {line}}}
    .st-key-story_steps button[kind="primary"] {{background: {p['BUTTON']} !important}}
    .st-key-story_steps button[kind="primary"] p {{color: {p['BUTTON_TEXT']} !important}}
    .story-eyebrow {{font-family: {mono}; font-size: .78rem; letter-spacing: .08em; color: {a}; margin: .6rem 0 0}}
    .story-num {{margin: .8rem 0; padding: .6rem .9rem; background: {panel}; border: 1px solid {line};
        border-left: 3px solid {a}}}
    .story-num .n {{display: block; font-family: {mono}; font-variant-numeric: tabular-nums; font-size: 2rem;
        font-weight: 600; color: {p['REAL']}; line-height: 1.2}}
    .story-num .w {{display: block; color: {p['TEXT2']}; font-size: .86rem; margin-top: .15rem}}
    .story-note {{color: {p['TEXT2']}; font-size: .92rem; margin: .5rem 0; padding: .45rem .7rem;
        border-left: 3px solid {line}; background: {p['FAINT']}}}
    .story-note::before {{content: "WHAT TO NOTICE ▸  "; font-family: {mono}; font-size: .78rem; font-weight: 600;
        letter-spacing: .06em; color: {a}}}
    .story-kinds {{display: grid; gap: .6rem}}
    .story-kind {{display: grid; grid-template-columns: 120px 1fr; gap: .9rem; align-items: center; padding: .6rem .8rem;
        background: {panel}; border: 1px solid {line}}}
    .story-kind svg {{width: 120px; height: 46px}}
    .story-kind b {{display: block; font-family: {mono}; font-size: .9rem; color: {p['REAL']}}}
    .story-kind span {{color: {p['TEXT2']}; font-size: .86rem; line-height: 1.5}}
    ul.story-verdict {{list-style: none; margin: 0; padding: 0; display: grid; gap: .5rem}}
    ul.story-verdict li {{display: grid; grid-template-columns: 2rem 1fr; padding: .55rem .8rem; background: {panel};
        border: 1px solid {line}; line-height: 1.55}}
    ul.story-verdict li .m {{font-family: {mono}; font-weight: 700; color: {a}}}
    @media (prefers-reduced-motion: no-preference) {{
        .st-key-story_card {{animation: story-in .25s ease-out}}
        @keyframes story-in {{from {{opacity: 0; transform: translateY(6px)}} to {{opacity: 1; transform: none}}}}
    }}
    </style>"""


def bignum(number, words):
    """The step's one big number and what it counts (trusted site text; `number` is already formatted)."""
    st.markdown(f"<div class='story-num'><span class='n'>{number}</span><span class='w'>{words}</span></div>",
                unsafe_allow_html=True)


def notice(text):
    st.markdown(f"<div class='story-note'>{text}</div>", unsafe_allow_html=True)


def kinds_picture():
    """Step 3's picture: the three kinds of trader as rows of dots (blue = value traders; the trend
    chasers' orange, filled = optimists, hollow = pessimists, so the picture does not rely on red and green)."""
    fund, chart = c.MIX["fundamentalists"][2], c.MIX["chartists"][2]

    def dots(fill, stroke):
        return "".join(f"<circle cx='{8 + (i % 9) * 13}' cy='{12 + (i // 9) * 18}' r='4.5' fill='{fill}' "
                       f"stroke='{stroke}' stroke-width='1.5'/>" for i in range(18))
    rows = [("Value traders", "Economists call them fundamentalists. They buy when the price is below the true "
                              "value and sell when it is above. They pull the price back.", dots(fund, fund)),
            ("Optimists", "Trend chasers (economists call them chartists) who expect a rise, so they buy. "
                          "They copy the crowd's mood.", dots(chart, chart)),
            ("Pessimists", "Trend chasers who expect a fall, so they sell. They copy the crowd's mood too.",
             dots("none", chart))]
    html = "".join(f"<div class='story-kind'><svg viewBox='0 0 120 46' aria-hidden='true'>{d}</svg>"
                   f"<div><b>{name}</b><span>{text}</span></div></div>" for name, text, d in rows)
    st.markdown(f"<div class='story-kinds'>{html}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------------- the eight steps
def step_puzzle(ctx):
    spx, spx_r = ctx["spx"], ctx["spx_r"]
    years = ex.bell_crash_years(20)
    left, right = st.columns([1, 1.4], gap="large")
    with left:
        st.markdown("A **standard deviation** (sd) is the usual size of a day's move. A 4-sd day moves four times "
                    "as much as usual. A **big day** is a day that moves more than 4 sd.")
        st.markdown(f"If daily moves followed a **bell curve**, big days would almost never come. Since 1950 the "
                    f"S&P 500 (the 500 biggest US companies' stocks) has had **{spx['big_days']} big days**. A bell "
                    f"curve expects about **{spx['bell_big_days']:.1f}** in the same {spx['days']:,} days.")
        st.markdown(f"Black Monday (19 October 1987) fell **{abs(spx['worst']):.0f} sd**. On a bell curve a 20-sd "
                    f"fall comes once in about **{ten_to(years)} years** (a 1 followed by "
                    f"{math.floor(math.log10(years))} zeros), far longer than the age of the universe. Yet the "
                    f"S&P 500 had one within {spx['days'] / ex.YEAR_DAYS:.0f} years.")
        st.markdown("The chart's bell curve drops away. The S&P 500 falls along a straight line instead. "
                    "Its slope is the **tail exponent** (lower = more big days). The S&P 500 is near 3.")
        bignum(f"{spx['big_days']} days", f"beyond 4 sd since 1950. A bell curve expects {spx['bell_big_days']:.1f}.")
    with right:
        c.color_key("real", "bell")
        c.show(c.tail_chart([("S&P 500", spx_r.values, dict(color=c.REAL, width=3))], height=360), key="s1")
        notice("Right of 4 sd, the bell curve drops away. The white S&P 500 line keeps going.")
        c.hint("Across: size of the day's move in sd. Up: share of days at least that big. Each step down is 10 "
               "times rarer. Higher means more common.")


def step_idea(ctx):
    (steps, counts), sim, exact, tv = _ants(**ANTS)
    left, right = st.columns([1, 1.4], gap="large")
    with left:
        st.markdown("Biologists gave ants two identical food sources. The ants do not split evenly: they "
                    "**crowd onto one**, then suddenly switch together.")
        st.markdown(f"The economist Alan Kirman (1993) wrote the math for it. He did not watch ants himself: he built "
                    f"a model of an experiment that other biologists had done. In the model, {ANTS['N']} ants each "
                    f"mostly copy another ant, and sometimes (chance ε = {ANTS['eps']}) switch on their own.")
        st.markdown("Copying the crowd like this is called **herding**. It is the one rule this project adds to a "
                    "market.")
        bignum(f"{tv:.3f}", f"gap between the simulated ants and Kirman's exact formula (0 = a perfect match; "
                            f"check 1 on the Checks page needs under {ex.KIRMAN_PASS})")
    with right:
        fig = make_subplots(rows=2, cols=1, vertical_spacing=0.2)
        fig.add_scatter(x=steps / 1e6, y=counts, mode="lines", name="Ants at source A",
                        line=dict(color=c.MIX["herding"][2], width=1.2), row=1, col=1)
        fig.add_bar(x=np.arange(ANTS["N"] + 1), y=sim, name="Simulated share of time",
                    marker_color=c.MIX["herding"][2], opacity=0.55, row=2, col=1)
        fig.add_scatter(x=np.arange(ANTS["N"] + 1), y=exact, mode="lines", name="Kirman's exact answer",
                        line=dict(color=c.REAL, width=2), row=2, col=1)
        c.style(fig, 400)
        fig.update_layout(bargap=0)
        fig.update_xaxes(title="Steps (millions)", row=1, col=1)
        fig.update_yaxes(title="Ants at source A", row=1, col=1)
        fig.update_xaxes(title="Ants at source A", row=2, col=1)
        fig.update_yaxes(title="Share of time", row=2, col=1)
        c.show(fig, key="s_ants")
        notice("The top line jumps between near 0 and near 100: nearly all ants switch sources together. "
               "Nobody tells them to.")
        c.hint(f"Top: how many of {ANTS['N']} ants are at source A, over the first "
               f"{steps[-1] / 1e6:.0f} million steps. Bottom: the share of time at each count, "
               f"simulated against Kirman's formula.")


def step_market(ctx):
    p, run = ctx["p"], ctx["run"]
    left, right = st.columns([1, 1.4], gap="large")
    with left:
        st.markdown(f"The model is a market of **{int(p['N'])} simulated traders** (Lux and Marchesi, *Nature*, 1999). "
                    f"Every trader is one of three kinds, and switches to whichever kind is making more money.")
        st.markdown("The **news** (changes in the stock's **true value**) is plain **bell-curve news**: mostly "
                    "small, never extreme. So if big days appear, the traders made them.")
        st.markdown(f"The history on this page is the same one the project's figures use: settings fitted to the "
                    f"S&P 500, and a number called a seed (it picks one **simulated history**) that was never used "
                    f"to fit those settings. It has as many days as the S&P 500 record, "
                    f"{run['days']:,} simulated trading days (about {run['days'] / ex.YEAR_DAYS:.0f} years).")
        bignum(f"{int(p['N'])}", f"simulated traders, trading every day for {run['days'] / ex.YEAR_DAYS:.0f} years")
        notice("Only trend chasers copy the crowd's mood. Turning that copying up or down is the experiment.")
    with right:
        kinds_picture()


def step_value_traders(ctx):
    fund, spx = ctx["mixes"]["fundamentalists"][0], ctx["spx"]
    k = ctx["nums"]["checks"]["fundamentalists"]["kurtosis"]
    s, sk = ex.summary(fund), ex.spread(k)
    left, right = st.columns([1, 1.4], gap="large")
    with left:
        st.markdown("Switch the trend chasers off. Only **value traders** remain (the mix called "
                    "**Value traders only** in the Test program): they buy below the true value and sell above "
                    "it. News moves the true value by bell-curve amounts.")
        st.markdown(f"The daily moves come out as a bell curve too. Excess kurtosis (0 means a perfect bell curve) is "
                    f"**{s['kurtosis']:.2f}** in this simulated history and **{sk['median']:.2f}** (median) over {sk['n']} "
                    f"simulated histories. So bell-curve news alone does not make big days.")
        bignum(f"{s['big_days']} days", f"beyond 4 sd in {s['days']:,} simulated trading days. The S&P 500 "
                                        f"had {spx['big_days']} in {spx['days']:,}.")
        notice("The model's line sits right on top of the dotted bell curve.")
    with right:
        c.color_key("bell", "fundamentalists")
        c.show(c.tail_chart([("Value traders only", fund, c.mix_line("fundamentalists", 3))],
                            height=360, guide=False), key="s2")
        c.hint("Across: size of the day's move in sd. Up: share of days at least that big.")


def step_run_away(ctx):
    nums, run, mixes = ctx["nums"], ctx["run"], ctx["mixes"]
    n_gone, n_all = ex.runaway_count(nums, "chartists")
    day = mixes["chartists"][1]
    day = day if day is not None and day >= 1 else None     # (a run-away during the warm-up has no story)
    story = c.runaway_story(run["seed"], day) if day is not None else None
    left, right = st.columns([1, 1.4], gap="large")
    with left:
        st.markdown("Now add **trend chasers**: traders who buy after a rise and sell after a fall. In the mix "
                    "called **Trend chasers, no herding** they do not copy each other (herding strength 0).")
        st.markdown(f"In **{n_gone} of {n_all}** simulated histories the market runs away: the price leaves its "
                    f"true value and never comes back. Real markets do not do that.")
        if story is not None and story["calm_days"] == 0:       # ran away on day 1-3: no calm stretch to tell
            st.markdown("In the simulated history on the right the price runs away within its first few days, as "
                        "nearly everyone becomes a trend chaser at once.")
        elif story is not None:
            st.markdown(f"In the simulated history on the right the price stays close to its true value for about "
                        f"{story['calm_days']:,} days while the share of trend chasers drifts up to "
                        f"{story['share_before']:.0%} of traders. Then, within a few days, "
                        f"nearly everyone becomes a trend chaser and the price runs away.")
        bignum(f"{n_gone} of {n_all}", "simulated histories ran away")
        notice("Calm for a long stretch, then a vertical take-off: a bubble that never ends.")
    with right:
        if story is not None:
            z, gap = story["z"], story["gap"]        # day 0 .. the runaway day (warm-up left out)
            days_ = np.arange(len(z))
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
            x, y = c.thin(days_, z)
            fig.add_scatter(x=x, y=y, name="Share of trend chasers", line=c.mix_line("chartists"),
                            hovertemplate="day %{x}: %{y:.0%} trend chasers<extra></extra>", row=1, col=1)
            x, y = c.thin(days_, gap)
            fig.add_scatter(x=x, y=y, name="Price above or below true value", line=dict(color=c.REAL, width=1),
                            hovertemplate="day %{x}: %{y:+.1%}<extra></extra>", row=2, col=1)
            # a neutral color: red means a loss on this site, and this is a bubble (the price shoots up)
            fig.add_annotation(x=days_[-1], y=story["gap_cap"], text="off the chart", showarrow=True,
                               arrowcolor=c.GUIDE, font=dict(color=c.MUTED, size=11), ax=-60, ay=12, row=2, col=1)
            c.style(fig, 400)
            fig.update_yaxes(tickformat=".0%", range=[0, 1], title="Trend chasers", row=1, col=1)
            fig.update_yaxes(tickformat=".0%", title="Price above true value", row=2, col=1)
            fig.update_xaxes(title="Simulated trading day", row=2, col=1)
            c.show(fig, key="s3")
            c.hint("Top: the share of traders chasing the trend. Bottom: how far the price is above its true value.")
        else:
            st.info("In this simulated history the price ran away during the warm-up, before day 1, so there is no "
                    "calm stretch to draw.", icon=":material/info:")


def step_herding(ctx):
    nums, spx, spx_r, herd = ctx["nums"], ctx["spx"], ctx["spx_r"], ctx["mixes"]["herding"][0]
    hs = ex.summary(herd)
    tails = ex.spread(nums["checks"]["herding"]["tail"])
    cc = ex.crash_check(nums, spx_r.values)
    left, right = st.columns([1, 1.4], gap="large")
    with left:
        st.markdown("Now the trend chasers also copy the majority mood: the mix called **Trend chasers who herd**. "
                    "Optimism spreads, then collapses, so **bursts of wild days** come and end on their own.")
        st.markdown(f"The tail exponent (lower = more big days) is **{hs['tail']:.2f}** in this history and "
                    f"**{tails['median']:.2f}** (median) over {tails['n']} simulated histories. The S&P 500's is "
                    f"**{spx['tail']:.2f}**. (The settings were fitted to the S&P 500 with this number as one target, "
                    f"so the match is partly by design. The bell curve without trend chasers was never a target.)")
        st.markdown(f"Up to about 4 sd the two lines agree. The S&P 500's most extreme days go further: its worst "
                    f"day is {abs(spx['worst']):.0f} sd (Black Monday); this history's worst is "
                    f"{abs(hs['worst']):.0f} sd.")
        st.markdown(f"Over {len(nums['crashes']['days'])} simulated histories, the model has an {cc['size']}-sd fall about "
                    f"every **{cc['row']['model']:.0f} years** (a bell curve: every "
                    f"{ten_to(cc['row']['bell'])} years), and a fall the size of Black Monday about "
                    f"every **{cc['bm_model']:,.0f} years** ({c.wait_range(cc)}). Herding makes crashes ordinary, though the "
                    f"biggest ones stay rarer than in the S&P 500 (the **Checks** page, check 4).")
        bignum(f"{tails['median']:.2f}", f"tail exponent (lower = more big days), middle value of {tails['n']} "
                                         f"simulated histories. The S&P 500: {spx['tail']:.2f}.")
        notice("The model's line follows the S&P 500's far past 4 sd. The bell curve drops away long before.")
    with right:
        c.color_key("real", "bell", "herding")
        c.show(c.tail_chart([("S&P 500", spx_r.values, dict(color=c.REAL, width=3)),
                             ("Trend chasers who herd", herd, c.mix_line("herding", 3))], height=330), key="s4")
        z = ex.moves_in_sd(herd)
        x, y = c.thin(np.arange(len(z)), z)
        fig = go.Figure(go.Scatter(x=x, y=y, mode="lines", line=dict(color=c.MIX["herding"][2], width=1),
                                   name="Simulated daily moves",
                                   hovertemplate="day %{x}: %{y:+.1f} sd<extra></extra>"))
        for lim in (-4, 4):
            fig.add_hline(y=lim, line=dict(color=c.GUIDE, dash="dot", width=1))   # a threshold, not a loss
        c.show(c.style(fig, height=210, xtitle="Simulated trading day", ytitle="Move (sd)", legend=False),
               key="s4b")
        st.caption("Daily moves of the model with herding. Beyond the dotted lines a bell curve almost never goes.")


def step_how_much(ctx):
    nums, spx = ctx["nums"], ctx["spx"]
    p = ctx["p"]
    fit, strong = ex.sweep_at(nums, float(p["a1"])), ex.sweep_at(nums, 1.6)
    edge, none = ex.sweep_at(nums, 0.8), ex.sweep_at(nums, 0.0)
    left, right = st.columns([1, 1.4], gap="large")
    shown = False
    with left:
        st.markdown("**Guess first.** If trend chasers copy each other more and more (herding strength, a1, goes "
                    "up), does the market get wilder or calmer?")
        guess = st.pills("Your guess", list(GUESSES), key="story_guess", label_visibility="collapsed",
                           on_change=lambda: st.session_state.update(story_reveal=True))
        if guess is None and not st.session_state.get("story_reveal"):
            st.button("Show me the answer", key="story_show",
                      on_click=lambda: st.session_state.update(story_reveal=True))
        else:
            shown = True
            if guess is not None:
                st.markdown(f"You said: *{guess}*. {GUESSES[guess]}")
            st.markdown(f"With no herding the market runs away in all {none['n']} saved histories; at "
                        f"herding strength {edge['a1']:.1f}, in {edge['ran_away']} of {edge['n']}. Turn herding up and it "
                        f"settles: at herding strength {strong['a1']:.1f} the tail exponent is {strong['tail']:.1f}, thinner "
                        f"than any real market. Only near herding strength {fit['a1']:.2f}, at the edge between running "
                        f"away and calm, is the tail exponent {fit['tail']:.1f}, like the S&P 500's "
                        f"{spx['tail']:.2f}.")
            bignum(f"{fit['a1']:.2f}", "the herding strength that best matches the S&P 500, right at the edge "
                                       "between running away and calm")
            notice("Lux and Marchesi called their paper *Scaling and criticality*: fat tails appear near a "
                   "tipping point.")
    with right:
        if shown:
            c.color_key("real", "herding")
            c.show(c.dial_chart(spx["tail"], height=330), key="s5")
            st.caption(f"Each point: the middle value of the tail exponent over the saved simulated histories that did not run away "
                       f"(of {len(nums['sweep']['seeds'])}). Herding strengths where most ran away are left out.")
        else:
            st.info("The chart appears once you guess.", icon=":material/visibility_off:")


def step_verdict(ctx):
    nums, spx, spx_r = ctx["nums"], ctx["spx"], ctx["spx_r"]
    tails = ex.spread(nums["checks"]["herding"]["tail"])
    cc = ex.crash_check(nums, spx_r.values)
    mem = ex.check_table(nums, spx).loc["memory"]        # memory of wild days: model median vs S&P 500
    years = spx["days"] / ex.YEAR_DAYS
    rows = [
        ("✓", f"<b>Fat tails.</b> Big days about as common as in the S&amp;P 500 (tail exponent "
              f"{tails['median']:.2f} against {spx['tail']:.2f})."),
        ("✓", f"<b>Bursts of wild days</b>, like the S&amp;P 500's (memory of wild days {mem['median']:.2f} "
              f"against {mem['sp500']:.2f})."),
        ("✓", f"<b>Crashes within decades:</b> an {cc['size']}-sd fall about every {cc['row']['model']:.0f} years, "
              f"not {ten_to(cc['row']['bell'])} years."),
        ("≈", f"A <b>fall the size of Black Monday</b> comes about every {cc['bm_model']:,.0f} years in the model "
              f"({c.wait_range(cc)}). "
              f"The S&amp;P 500 had one in {years:.0f} years."),
        ("✗", "<b>Falls bigger than rises.</b> In the S&amp;P 500 the biggest falls are bigger than the biggest "
              "rises. In the model they are the same size."),
        # not "Memory": memory of wild days matches (row above); this limit is about direction and weeks
        ("✗", "<b>Daily reversal and fading tails.</b> The model's prices reverse a little from one day to the next, which the S&amp;P 500's "
              "do not, and its fat tails fade over a few weeks somewhat faster."),
    ]
    left, right = st.columns([1, 1.4], gap="large")
    with left:
        st.markdown("Fed only bell-curve news, a market with herding reproduces most of what makes real markets "
                    "wild. It still misses two things. Every number here is checked on the **Checks** page, with "
                    "the rule for each check.")
        bignum(f"{cc['row']['model']:.0f} years", f"between {cc['size']}-sd falls in the model. A bell curve: "
                                                  f"{ten_to(cc['row']['bell'])} years.")
        notice("The two crosses are real limits of the model. The Checks page shows them too.")
    with right:
        items = "".join(f"<li><span class='m'>{m}</span><span>{t}</span></li>" for m, t in rows)
        st.markdown(f"<ul class='story-verdict'>{items}</ul>", unsafe_allow_html=True)
    st.markdown("**Try it yourself.** The model was fitted to the S&P 500 only. Test it against markets it never "
                "saw: 5 stocks, two other indexes (Nasdaq-100, Dow Jones) and three commodities, or any stock "
                "symbol you like.")
    a, b, d = st.columns(3)
    a.page_link("web/page_explore.py", label="Open the test program", icon=":material/candlestick_chart:")
    b.page_link("web/page_checks.py", label="Compare all markets (Checks, All markets)",
                icon=":material/scatter_plot:")
    d.page_link("web/page_checks.py", label="See the checks", icon=":material/fact_check:")


RENDER = [step_puzzle, step_idea, step_market, step_value_traders, step_run_away, step_herding,
          step_how_much, step_verdict]


# ---------------------------------------------------------------------------------- the page
def page():
    st.html(story_css())
    st.title("The argument in 8 steps", anchor=False)
    c.lead("If the news is plain bell-curve noise, can traders copying each other (herding) alone cause the fat "
           "tails, crashes and bursts of wild days that real markets show? Each step adds one thing and shows "
           "what changes. About 3 minutes.")

    _, spx_r, spx = c.saved_asset("S&P 500")
    run = c.default_run()
    with st.spinner("Running the model..."):
        mixes = c.run_mixes(**run)
    p, _ = c.fitted()
    ctx = dict(spx=spx, spx_r=spx_r, run=run, mixes=mixes, nums=c.numbers(), p=p)

    st.session_state["story_step"] = i = step_now()
    menu, card = st.columns([1, 4], gap="large")
    with menu:
        with st.container(key="story_steps"):
            for j, (short, _) in enumerate(STEPS):
                label = f"{'✓ ' if j < i else ''}{j + 1}  {short}"
                st.button(label, key=f"story_go_{j}", on_click=move, kwargs=dict(step=j),
                          type="primary" if j == i else "secondary", width="stretch")
    with card:
        with st.container(key="story_card"):
            st.progress((i + 1) / len(STEPS))
            st.markdown(f"<div class='story-eyebrow'>STEP {i + 1} OF {len(STEPS)}</div>", unsafe_allow_html=True)
            st.header(STEPS[i][1], anchor=False)
            RENDER[i](ctx)
        back, mid, nxt = st.columns([1, 3, 1])
        back.button("Back", key="story_back", on_click=move, kwargs=dict(by=-1), disabled=i == 0,
                    icon=":material/arrow_back:", width="stretch")
        mid.caption("Use Back and Next, or click a step on the left.")
        nxt.button("Next", key="story_next", on_click=move, kwargs=dict(by=1), disabled=i == LAST,
                   type="primary", icon=":material/arrow_forward:", width="stretch")


page()
