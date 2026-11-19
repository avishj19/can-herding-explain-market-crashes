"""Compare: every saved market on one map, next to the model."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from herding import explore as ex
from web import common as c

SYMBOL = {"Stocks": "circle", "Indexes": "diamond", "Commodities": "square"}


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
        fig.add_scatter(x=d["tail"], y=d["memory"], mode="markers+text", name=g, text=d["name"],
                        textposition="top center", textfont=dict(size=10, color=c.MUTED),
                        marker=dict(symbol=SYMBOL[g], size=10, color=c.REAL, line=dict(color=c.BG, width=1)),
                        customdata=np.c_[d["kurtosis"], d["times_bell"]],
                        hovertemplate="<b>%{text}</b><br>tail %{x:.2f}, memory %{y:.2f}<br>kurtosis "
                                      "%{customdata[0]:.1f}, %{customdata[1]:.0f}x the bell curve's big days"
                                      "<extra></extra>")
    for key in ex.MIXES:
        v = nums["checks"][key]
        if all(x is None for x in v.get("tail", [])):
            continue                    # every history of this mix ran away: no star (the caption says so)
        sp, mem = ex.spread(v["tail"]), ex.spread(v["memory"])
        fig.add_scatter(x=[sp["median"]], y=[mem["median"]], mode="markers", name=f"Model: {c.MIX[key][0].lower()}",
                        marker=dict(symbol="star", size=18, color=c.MIX[key][2], line=dict(color=c.BG, width=1)),
                        error_x=dict(type="data", symmetric=False, color=c.MIX[key][2], thickness=1,
                                     array=[sp["high"] - sp["median"]],
                                     arrayminus=[sp["median"] - sp["low"]]),
                        hovertemplate=f"<b>{c.MIX[key][0]}</b> (median of {sp['n']} histories)<br>"
                                      f"tail %{{x:.2f}}, memory %{{y:.2f}}<extra></extra>")
    c.style(fig, height=500, xtitle="Tail exponent (smaller means fatter tails)",
            ytitle="Volatility memory (move sizes 20 days apart)")
    c.show(fig, key="map")
    if not groups:
        st.caption("Pick a group above to show its markets.")


@st.fragment
def tails_side_by_side(t):
    names = st.multiselect("Markets (up to 6)", list(t["name"]), default=["S&P 500", "Gold", "AAPL"],
                           max_selections=6)
    series = [(n, c.saved_asset(n)[1].values, dict(color=c.PICK[i], width=2.4)) for i, n in enumerate(names)]
    if series:
        c.show(c.tail_chart(series, height=420, guide=False), key="tails")
        st.caption("Here the colors only label the markets you picked.")
    else:
        st.caption("Pick up to 6 markets to compare.")


def page():
    st.title("Compare markets", anchor=False)
    c.lead("Every saved market on one map. Across is the tail exponent (further left means fatter tails); up "
           "is volatility memory (higher means wild days bunch together more). The stars are the model: the "
           "median of 40 histories, with a bar covering the middle 90%.")

    t = c.market_table()
    nums = c.numbers()
    market_map(t, nums)
    gone, total = ex.runaway_count(nums, "chartists")
    chartists = ("Chartists without herding have no star: all {} of their histories ran away." if gone == total
                 else "Chartists without herding: {} of {} histories ran away, so their star is the median "
                      "of the rest.").format(*((gone,) if gone == total else (gone, total)))
    st.caption(f"Most markets sit near a tail of 3 with memory between 0.1 and 0.25. The herding model lands "
               f"among them; with no chartists it sits far to the right at memory 0, where a bell curve would be. "
               f"{chartists}")
    st.divider()

    st.header("How many more big days than a bell curve?", anchor=False)
    d = t.sort_values("times_bell")
    shade = {"Indexes": c.REAL, "Stocks": c.SHADE_MID, "Commodities": c.SHADE_LOW}
    fig = go.Figure()
    for g, col in shade.items():
        dd = d[d["group"] == g]
        fig.add_bar(x=dd["times_bell"], y=dd["name"], orientation="h", name=g, marker_color=col,
                    customdata=np.c_[dd["big_days"], dd["bell_big_days"]],
                    hovertemplate="<b>%{y}</b><br>%{customdata[0]} big days; a bell curve expects "
                                  "%{customdata[1]:.1f}<extra></extra>")
    fig.update_yaxes(categoryorder="array", categoryarray=list(d["name"]))
    c.style(fig, height=700, logx=True, xtitle="Days beyond 4 standard deviations, as a multiple of what a bell "
                                                "curve expects (log scale)")
    c.show(fig, key="bars")
    st.divider()

    st.header("Tails side by side", anchor=False)
    tails_side_by_side(t)
    st.divider()

    st.header("All numbers", anchor=False)
    show = t.rename(columns={"group": "Group", "name": "Market", "days": "Days", "tail": "Tail",
                             "kurtosis": "Kurtosis", "memory": "Memory", "worst": "Worst day (sd)",
                             "big_days": "Days > 4 sd", "times_bell": "x bell curve"}).drop(columns="bell_big_days")
    st.dataframe(show.style.format({"Tail": "{:.2f}", "Kurtosis": "{:.1f}", "Memory": "{:.2f}",
                                    "Worst day (sd)": "{:.1f}", "x bell curve": "{:.0f}", "Days": "{:,}"}),
                 hide_index=True, width="stretch", height=420)


page()
