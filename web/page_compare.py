"""All markets (a tab of the Checks page since the redesign; was the Compare page): every saved market
on one map, next to the model. web/page_checks.py calls body()."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from herding import explore as ex
from web import common as c

SYMBOL = {"Stocks": "circle", "Indexes": "diamond", "Commodities": "square"}
MIX_SYMBOL = {"fundamentalists": "circle-open", "chartists": "x", "herding": "star"}
LABELLED = ("Indexes", "Commodities")       # stocks are many and crowd the map: their names show on hover
X_RANGE, Y_RANGE = (2.2, 4.1), (-0.03, 0.29)  # zoomed to the markets; a star outside gets an arrow and its value


def group_color(g):
    return {"Stocks": c.SHADE_MID, "Indexes": c.REAL, "Commodities": c.SHADE_LOW}[g]


@st.fragment
def market_map(t, nums):
    groups = st.pills("Show", list(SYMBOL), default=list(SYMBOL), selection_mode="multi",
                      label_visibility="collapsed") or []
    fig = go.Figure()
    fig.add_vrect(x0=2.5, x1=3.5, fillcolor=c.FAINT, line_width=0, layer="below",
                  annotation_text="about 3 (inverse cubic law)", annotation_position="top left",
                  annotation_font=dict(size=11, color=c.MUTED))
    for g in groups:
        d = t[t["group"] == g]
        fig.add_scatter(x=d["tail"], y=d["memory"], mode="markers+text" if g in LABELLED else "markers",
                        name=g, text=d["name"], textposition="top center", textfont=dict(size=10, color=c.MUTED),
                        marker=dict(symbol=SYMBOL[g], size=10, color=group_color(g), line=dict(color=c.BG, width=1)),
                        customdata=np.c_[d["kurtosis"], d["times_bell"]],
                        hovertemplate="<b>%{text}</b><br>tail exponent %{x:.2f}, memory of wild days %{y:.2f}<br>excess kurtosis "
                                      "%{customdata[0]:.1f}, %{customdata[1]:.0f}× the bell curve's big days"
                                      "<extra></extra>")
    for key in ex.MIXES:
        v = nums["checks"][key]
        if all(x is None for x in v.get("tail", [])):
            continue                    # every history of this mix ran away: no star (the caption says so)
        sp, mem = ex.spread(v["tail"]), ex.spread(v["memory"])
        color = c.MIX[key][2]
        fig.add_scatter(x=[sp["median"]], y=[mem["median"]], mode="markers", name=f"Model: {c.MIX[key][0].lower()}",
                        marker=dict(symbol=MIX_SYMBOL[key], size=16 if key == "herding" else 13, color=color,
                                    line=dict(color=color if key != "herding" else c.BG, width=2 if key != "herding" else 1)),
                        error_x=dict(type="data", symmetric=False, color=color, thickness=1,
                                     array=[sp["high"] - sp["median"]],
                                     arrayminus=[sp["median"] - sp["low"]]),
                        hovertemplate=f"<b>{c.MIX[key][0]}</b> (median of {sp['n']} simulated histories)<br>"
                                      f"tail exponent %{{x:.2f}}, memory of wild days %{{y:.2f}}<extra></extra>")
        if not (X_RANGE[0] <= sp["median"] <= X_RANGE[1] and Y_RANGE[0] <= mem["median"] <= Y_RANGE[1]):
            fig.add_annotation(x=X_RANGE[1], y=min(max(mem["median"], Y_RANGE[0]), Y_RANGE[1]), xanchor="right",
                               text=f"{c.MIX[key][0]}: tail exponent {sp['median']:.2f}, memory {mem['median']:.2f} →",
                               showarrow=False, font=dict(size=11, color=color), yshift=10)
    c.style(fig, height=500, xtitle="Tail exponent (lower = more big days)",
            ytitle="Memory of wild days (move sizes 20 days apart)", hover="closest")   # a map: one point at a time
    fig.update_xaxes(range=list(X_RANGE))
    fig.update_yaxes(range=list(Y_RANGE))
    c.show(fig, key="map")
    st.caption("Stocks are unlabelled to keep the map readable: point at one to see its name.")
    if not groups:
        st.caption("Pick a group above to show its markets.")


@st.fragment
def tails_side_by_side(t):
    names = st.multiselect("Markets (up to 6)", list(t["name"]), default=["S&P 500", "Gold", "AAPL"],
                           max_selections=6)
    series = [(n, c.saved_asset(n)[1].values, dict(color=c.PICK[i], width=2.4, dash=c.PICK_DASH[i]))
              for i, n in enumerate(names)]
    if series:
        c.show(c.tail_chart(series, height=420, guide=False), key="tails")
        st.caption("Each market you picked has its own color and line style (they do not mean anything else "
                   "here). The grey dotted line is the bell curve.")
    else:
        st.caption("Pick up to 6 markets to compare.")


def body():
    nums = c.numbers()
    c.lead("Every saved market on one map. Across is the tail exponent (further left = more big days); up "
           "is memory of wild days (higher means stronger bursts of wild days). The model is shown in color "
           "(star: trend chasers who herd; open circle: value traders only; cross: trend chasers, no herding): the "
           f"median of {len(nums['recipe']['seeds'])} simulated histories, with a bar covering the middle 90%.")

    t = c.market_table()
    market_map(t, nums)
    gone, total = ex.runaway_count(nums, "chartists")
    chartists = ("Trend chasers, no herding have no marker: all {} of their simulated histories ran away." if gone == total
                 else "Trend chasers, no herding: {} of {} simulated histories ran away, so their cross is the median "
                      "of the rest.").format(*((gone,) if gone == total else (gone, total)))
    st.caption(f"Most markets sit near a tail exponent of 3 with memory between 0.1 and 0.25. The model with herding lands "
               f"among them. With no trend chasers (open circle) it sits far to the right at memory 0, off this map, where "
               f"a bell curve would be. "
               f"{chartists}")
    st.divider()

    st.header("How many more big days than a bell curve?", anchor=False)
    d = t.sort_values("times_bell")
    fig = go.Figure()
    for g in SYMBOL:
        dd = d[d["group"] == g]
        fig.add_bar(x=dd["times_bell"], y=dd["name"], orientation="h", name=g, marker_color=group_color(g),
                    customdata=np.c_[dd["big_days"], dd["bell_big_days"]],
                    hovertemplate="<b>%{y}</b><br>%{customdata[0]} big days; a bell curve expects "
                                  "%{customdata[1]:.1f}<extra></extra>")
    fig.add_vline(x=1, line=dict(color=c.BELL, dash="dot", width=1.5), annotation_text="bell curve = 1×",
                  annotation_position="top right", annotation_font=dict(size=11, color=c.MUTED))
    fig.update_yaxes(categoryorder="array", categoryarray=list(d["name"]))
    c.style(fig, height=700, xtitle="Big days (beyond 4 sd), as a multiple of what a bell "
                                    "curve expects", hover="y unified")   # bars run across: read by row
    fig.update_xaxes(type="linear", range=[0, 1.05 * float(d["times_bell"].max())], rangemode="tozero")
    c.show(fig, key="bars")
    st.caption(f"A bell curve would give a bar of length 1 (the dotted line). These markets have "
               f"{d['times_bell'].min():.0f} to {d['times_bell'].max():.0f} times more big days than that.")
    st.divider()

    st.header("Tails side by side", anchor=False)
    tails_side_by_side(t)
    st.divider()

    st.header("All numbers", anchor=False)
    show = t.rename(columns={"group": "Group", "name": "Market", "days": "Days", "tail": "Tail exponent",
                             "kurtosis": "Excess kurtosis", "memory": "Memory of wild days", "worst": "Worst day (sd)",
                             "big_days": "Big days (beyond 4 sd)", "times_bell": "Times a bell curve"}).drop(columns="bell_big_days")
    st.dataframe(show.style.format({"Tail exponent": "{:.2f}", "Excess kurtosis": "{:.1f}", "Memory of wild days": "{:.2f}",
                                    "Worst day (sd)": "{:.1f}", "Times a bell curve": "{:.0f}", "Days": "{:,}"}),
                 hide_index=True, width="stretch", height=420)
