"""Story: the argument in five steps, one idea and one chart each."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from herding import explore as ex
from web import common as c


def page():
    st.title("The argument in five steps", anchor=False)
    c.lead("Each step adds one thing to the simulated market and shows what changes. The model history "
           "on this page is the same one used in the project's figures: settings fitted to the S&P 500, "
           "a seed the fit never saw, and as many days as the S&P 500 record.")
    c.color_key("real", "bell", "fundamentalists", "chartists", "herding")

    _, spx_r, spx = c.saved_asset("S&P 500")
    run = c.default_run()
    with st.spinner("Running the model..."):
        mixes = c.run_mixes(**run)
    nums = c.numbers()

    # 1 ------------------------------------------------------------------
    st.header("1. Real markets have too many big days", anchor=False)
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.markdown(f"Since 1950 the S&P 500 has had **{spx['big_days']} days** that moved more than four "
                    f"standard deviations. A bell curve expects about **{spx['bell_big_days']:.1f}** in the "
                    f"same {spx['days']:,} days.")
        st.markdown("On the chart, the bell curve drops off a cliff. The real market falls along a straight "
                    "line with a slope near 3 instead.")
    with right:
        c.show(c.tail_chart([("S&P 500", spx_r.values, dict(color=c.REAL, width=3))], height=330), key="s1")
    st.divider()

    # 2 ------------------------------------------------------------------
    fund = mixes["fundamentalists"][0]
    k = nums["checks"]["fundamentalists"]["kurtosis"]
    st.header("2. Value traders alone give a bell curve", anchor=False)
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.markdown("Start with only fundamentalists: they buy below the true value and sell above it. "
                    "News moves the value by bell-curve amounts.")
        st.markdown(f"The returns come out as a bell curve too. Excess kurtosis is "
                    f"**{ex.summary(fund)['kurtosis']:.2f}** in this history and **{np.median(k):.2f}** "
                    f"(median) over {len(k)} histories, where 0 means a perfect bell curve.")
    with right:
        c.show(c.tail_chart([("Fundamentalists only", fund, c.mix_line("fundamentalists", 3))],
                            height=330, guide=False), key="s2")
    st.divider()

    # 3 ------------------------------------------------------------------
    gone = nums["checks"]["chartists"]["runaway_day"]
    n_gone = sum(d is not None for d in gone)
    st.header("3. Trend-followers alone make the market run away", anchor=False)
    left, right = st.columns([1, 2], gap="large")
    day = mixes["chartists"][1]
    with left:
        st.markdown("Now add chartists, who buy after a rise and sell after a fall, but don't copy each "
                    "other (herding strength 0).")
        st.markdown(f"In **{n_gone} of {len(gone)}** histories the market runs away: the price leaves its "
                    f"value and never comes back.")
        if day is not None:
            path = c.runaway_path(run["seed"], day)
            st.markdown(f"In the history on the right the price stays close to its value for about "
                        f"{day - 3:,} days while chartists slowly grow to "
                        f"{path['z'][max(len(path['z']) - 6, 0)]:.0%} of traders. Then, within a few days, "
                        f"nearly everyone becomes a chartist and the price explodes.")
    with right:
        if day is not None:
            path = c.runaway_path(run["seed"], day)
            days_ = np.arange(len(path["z"]))
            gap = np.expm1(np.clip(path["lp"] - path["lf"], -0.5, 0.5))
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
            x, y = c.thin(days_, path["z"])
            fig.add_scatter(x=x, y=y, name="Share of chartists", line=c.mix_line("chartists"),
                            hovertemplate="day %{x}: %{y:.0%} chartists<extra></extra>", row=1, col=1)
            x, y = c.thin(days_, gap)
            fig.add_scatter(x=x, y=y, name="Price above or below value", line=dict(color=c.REAL, width=1),
                            hovertemplate="day %{x}: %{y:+.1%}<extra></extra>", row=2, col=1)
            fig.add_annotation(x=days_[-1], y=float(np.expm1(0.5)), text="off the chart", showarrow=True,
                               arrowcolor=c.DOWN, font=dict(color=c.DOWN, size=11), ax=-60, ay=12, row=2, col=1)
            c.style(fig, height=360)
            fig.update_yaxes(tickformat=".0%", range=[0, 1], title="Chartists", row=1, col=1)
            fig.update_yaxes(tickformat=".0%", title="Mispricing", row=2, col=1)
            fig.update_xaxes(title="Trading day", row=2, col=1)
            c.show(fig, key="s3")
    st.divider()

    # 4 ------------------------------------------------------------------
    herd = mixes["herding"][0]
    hs = ex.summary(herd)
    tails = [t for t in nums["checks"]["herding"]["tail"] if t is not None]
    st.header("4. Add herding: fat tails and bursts", anchor=False)
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.markdown("Now chartists also copy the majority mood. Optimism spreads, then collapses, so wild "
                    "days come in bursts that end on their own.")
        st.markdown(f"The tail exponent is **{hs['tail']:.2f}** in this history and **{np.median(tails):.2f}** "
                    f"(median) over {len(tails)} histories. The S&P 500's is **{spx['tail']:.2f}**.")
        st.markdown(f"Up to about four standard deviations the two lines agree. The S&P's most extreme days "
                    f"go further: its worst day is {abs(spx['worst']):.0f} standard deviations (Black "
                    f"Monday); this history's worst is {abs(hs['worst']):.0f}.")
    with right:
        c.show(c.tail_chart([("S&P 500", spx_r.values, dict(color=c.REAL, width=3)),
                             ("Chartists who herd", herd, c.mix_line("herding", 3))], height=330), key="s4")
    z = (herd - herd.mean()) / herd.std()
    x, y = c.thin(np.arange(len(z)), z)
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines", line=dict(color=c.MIX["herding"][2], width=1),
                               name="Simulated daily moves", hovertemplate="day %{x}: %{y:+.1f} sd<extra></extra>"))
    for lim in (-4, 4):
        fig.add_hline(y=lim, line=dict(color=c.DOWN, dash="dot", width=1))
    c.show(c.style(fig, height=210, xtitle="Simulated trading day", ytitle="Move (sd)", legend=False), key="s4b")
    st.caption("Daily moves of the herding model. Beyond the dotted lines a bell curve almost never goes.")
    st.divider()

    # 5 ------------------------------------------------------------------
    st.header("5. Try it yourself", anchor=False)
    st.markdown("The model was fitted to the S&P 500 only. Test it against 20 stocks, three index funds and "
                "three commodities it never saw, or any ticker you like.")
    a, b, d = st.columns(3)
    a.page_link("web/page_explore.py", label="Open the workspace", icon=":material/candlestick_chart:")
    b.page_link("web/page_compare.py", label="Compare all markets", icon=":material/scatter_plot:")
    d.page_link("web/page_checks.py", label="See the checks", icon=":material/fact_check:")


page()
