"""Checks: the proposal's checks, each with its rule and its numbers.

Model numbers come from web/numbers.json (scripts/web_numbers.py): the same 40 held-out histories as
the project's figures, so this page and figures/RESULTS.md agree. Kirman's ants run live.
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from herding import explore as ex
from herding import kirman
from web import common as c

KIRMAN = dict(N=100, delta=0.1)
STEPS = 20_000_000
KIRMAN_PASS = 0.05     # measured Nov 2026 with 20M steps: distance 0.004-0.016 over seeds 0-3


@st.cache_data(show_spinner=False)
def _ants(eps, seed=1):
    k = kirman.simulate(KIRMAN["N"], eps, KIRMAN["delta"], STEPS, seed=seed)
    sim, exact, tv = ex.kirman_match(k, KIRMAN["N"], eps, KIRMAN["delta"])
    return k[:5_000_000:2_500], sim, exact, tv


def _dots(seeds, values, color, band, ref=None, ref_label="", ytitle=""):
    fig = go.Figure()
    fig.add_hrect(y0=band[0], y1=band[1], fillcolor=c.OK_FILL, line_width=0, layer="below")
    fig.add_scatter(x=seeds, y=values, mode="markers", marker=dict(size=8, color=color), name="One history",
                    hovertemplate="seed %{x}: %{y:.3f}<extra></extra>")
    fig.add_hline(y=float(np.median(values)), line=dict(color=color, width=1.5))   # the median (stated beside the chart)
    if ref is not None:
        fig.add_hline(y=ref, line=dict(color=c.REAL, width=1.5, dash="dash"), annotation_text=ref_label,
                      annotation_position="top left", annotation_font=dict(size=11, color=c.REAL))
    return c.style(fig, 260, xtitle="Seed (held-out history)", ytitle=ytitle, legend=False)


def page():
    nums = c.numbers()
    rec = nums["recipe"]
    _, _, spx = c.saved_asset("S&P 500")
    seeds = rec["seeds"]

    st.title("Checks", anchor=False)
    c.lead(f"Each check below says its rule first, then shows the numbers. The model numbers come from "
           f"{len(seeds)} histories of {rec['days']:,} days each, on seeds the fitted settings never saw: the "
           f"same histories as the project's figures.")

    # 1 ------------------------------------------------------------------
    k = nums["checks"]["fundamentalists"]["kurtosis"]
    res = ex.bell_check(k, rec["days"])
    st.header("1. With no chartists, returns follow a bell curve", anchor=False)
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.markdown(f"**Rule:** every history's excess kurtosis is within ±5 standard errors of 0. For "
                    f"{rec['days']:,} days that is ±5·√(24/n) = ±{res['tol']:.3f}.")
        st.markdown(f"Median **{np.median(k):+.3f}**; farthest from 0: **{res['worst']:+.3f}**. "
                    f"The check {c.verdict(res['passed'])}.", unsafe_allow_html=True)
    with right:
        fig = _dots(seeds, k, c.MIX["fundamentalists"][2], (-res["tol"], res["tol"]), ytitle="Excess kurtosis")
        fig.update_yaxes(range=[-1.6 * res["tol"], 1.6 * res["tol"]])
        c.show(fig, key="c1")
    st.divider()

    # 2 ------------------------------------------------------------------
    tails = nums["checks"]["herding"]["tail"]
    ok_tails = [t for t in tails if t is not None]
    tres = ex.tail_check(ok_tails)
    st.header("2. With herding, the tail exponent is about 3", anchor=False)
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.markdown("**Rule:** the median tail exponent lies between 2.5 and 3.5. The inverse cubic law "
                    "(Gopikrishnan et al. 1998) says about 3; the proposal expected 2.6 to 3.")
        st.markdown(f"Model median **{np.median(ok_tails):.2f}** (range {tres['low']:.2f} to {tres['high']:.2f}); "
                    f"S&P 500 **{spx['tail']:.2f}**. The check {c.verdict(2.5 <= np.median(ok_tails) <= 3.5)}.",
                    unsafe_allow_html=True)
        lost = len(tails) - len(ok_tails)
        if lost:
            st.caption(f"{lost} histories ran away and are left out.")
    with right:
        fig = _dots([s for s, t in zip(seeds, tails) if t is not None], ok_tails, c.MIX["herding"][2], (2.5, 3.5),
                    ref=spx["tail"], ref_label="S&P 500", ytitle="Tail exponent")
        fig.update_yaxes(range=[2.2, 4.2])
        c.show(fig, key="c2")
    st.divider()

    # 3 ------------------------------------------------------------------
    st.header("3. Kirman's ants match his exact formula", anchor=False)
    th = (1 - KIRMAN["delta"]) / (KIRMAN["N"] - 1)
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.markdown(f"The herding rule inside the model comes from Kirman's ant colony. 100 ants choose "
                    f"between two food sources; each step one ant either switches on its own (chance ε) or "
                    f"copies another ant. Kirman worked out the exact long-run answer.")
        eps = st.select_slider("Chance an ant switches on its own (ε)",
                               [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1], value=0.002)
        hist_k, sim, exact, tv = _ants(eps)
        st.markdown(f"Below ε = {th:.4f} the ants herd: most sit at one source and switch together. "
                    f"Now ε = {eps}, so they {'herd' if eps < th else 'split about evenly'}.")
        st.markdown(f"**Rule:** over {STEPS / 1e6:.0f} million steps, the simulated share of time at each count "
                    f"differs from the exact answer by less than {KIRMAN_PASS} (total variation distance; 0 means "
                    f"identical). Distance **{tv:.4f}**: the check {c.verdict(tv < KIRMAN_PASS)}.",
                    unsafe_allow_html=True)
    with right:
        fig = make_subplots(rows=2, cols=1, vertical_spacing=0.18)
        fig.add_scatter(x=np.arange(len(hist_k)) * 2500 / 1e6, y=hist_k, mode="lines", name="Simulation",
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

    # 4 ------------------------------------------------------------------
    st.header("4. Extreme days and clustering, next to the S&P 500", anchor=False)
    st.markdown("The proposal asks for numbers comparable to the S&P 500, so there is no pass or fail line "
                "here. The model column is the median over the same histories, with the middle 90% in brackets.")
    h = nums["checks"]["herding"]
    rows = []
    for label, key, f in (("Tail exponent", "tail", "{:.2f}"), ("Excess kurtosis", "kurtosis", "{:.1f}"),
                          ("Volatility memory (20 days)", "memory", "{:.2f}"), ("Worst day (sd)", "worst", "{:.1f}")):
        v = np.array([x for x in h[key] if x is not None])
        lo, hi = np.percentile(v, [5, 95])
        rows.append({"Measure": label, "Model": f"{f.format(np.median(v))} ({f.format(lo)} to {f.format(hi)})",
                     "S&P 500": f.format(spx[key])})
    st.dataframe(rows, hide_index=True, width="stretch")
    hk = np.median([x for x in h["kurtosis"] if x is not None])
    st.caption(f"Kurtosis is the weak spot: the S&P's {spx['kurtosis']:.0f} is driven mostly by one day "
               f"(Black Monday), and the model's median is {hk:.1f}. figures/RESULTS.md discusses this.")


page()
