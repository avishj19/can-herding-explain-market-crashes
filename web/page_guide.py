"""Start here: what the site is and how to use it."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from herding import explore as ex
from web import common as c


def page():
    st.title("Can herding explain market crashes?", anchor=False)
    c.lead("Stock prices have far more huge days than a bell curve allows. On 19 October 1987 the S&P 500 "
           "fell 20% in a single day. This site runs a simulated market to test one explanation: traders "
           "copying each other. It is a student project for 02-120 Programming for Scientists at CMU (Fall 2026).")

    st.header("What the model is", anchor=False)
    st.markdown(
        "The model is from Lux and Marchesi (*Nature*, 1999). 500 simulated traders each follow one of "
        "two strategies:\n\n"
        "- **Fundamentalists** buy when the price is below the stock's true value and sell when it is above.\n"
        "- **Chartists** ignore value and follow the trend. They are either optimists or pessimists, "
        "and they tend to copy whatever mood most chartists are in. That copying is the **herding**.\n\n"
        "Traders switch strategy when the other one is making more money. News moves the true value by "
        "small random amounts that follow a bell curve, so if big days show up, the traders made them.")

    st.header("How to use it", anchor=False)
    st.markdown(
        "1. **Story** walks through the argument in five short steps. Read this first.\n"
        "2. **Workspace** is where you experiment.\n"
        "    - Pick a market from the list on the left, or type any ticker (for example `NVDA`) and press Enter.\n"
        "    - Use the range buttons (1D, 7D, YTD, 1Y, 5Y, 20Y, All) to change the time span. "
        "1D and 7D load prices within the day from Yahoo Finance.\n"
        "    - Switch the chart between Price, Tail, Histogram, Bursts and Memory. Hover for exact values, "
        "drag to zoom, double-click to zoom back out, and click a legend entry to hide a line.\n"
        "    - Change the herding strength on the right and press **Run model**. Each run is added to the "
        "herding chart beneath the controls, so you can see how the tail changes as you turn the dial.\n"
        "3. **Compare** puts every saved market on one map, next to the model.\n"
        "4. **Checks** shows the tests from the project proposal, with the rule for each one.\n"
        "5. **Findings** collects the project's figures, which are averaged over many simulated histories.")

    st.header("How to read the charts", anchor=False)
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown(
            "**Tail** is the main chart. Across: how big a day's move was, in standard deviations "
            "(a standard deviation is the size of a typical day). Up: the share of days with a move at "
            "least that big. Both axes are logarithmic.\n\n"
            "- A bell curve drops off a cliff: a 5-standard-deviation day should come once in thousands of years.\n"
            "- Real markets fall along a straight line instead, with a slope near 3. That straight line is "
            "the *inverse cubic law*: a move twice as big is about 2³ = 8 times rarer.\n\n"
            "**Histogram** shows the same thing as a shape: the real market has a taller middle and longer "
            "tails than the bell curve.\n\n"
            "**Bursts** shows each day's move over time. In real markets wild days come in clusters.\n\n"
            "**Memory** measures those clusters: how much a big move today predicts big moves "
            "1 to 200 days later.")
    with right:
        g = np.random.default_rng(3)
        bell = g.standard_normal(20000)
        fat = g.standard_t(3, 20000)
        fig = go.Figure()
        for label, r, line in (("Bell curve", bell, dict(color=c.BELL, dash="dot", width=2.5)),
                               ("Fat tails (a real market)", fat, dict(color=c.REAL, width=2.5))):
            x, y = ex.tail_curve(r)
            fig.add_scatter(x=x, y=y, name=label, mode="lines", line=line, hoverinfo="skip")
        c.style(fig, 330, logx=True, logy=True, xtitle="Move size (standard deviations)",
                ytitle="Share of days at least this big")
        c.tail_axes(fig)     # down to 0.001%: 20,000 days reach 1/20,000 = 0.005%, which ymin=-4 cut off
        c.show(fig, key="guide_example")
        st.caption("An example with made-up numbers: the two curves agree for small moves and split apart "
                   "for big ones.")

    st.header("The controls", anchor=False)
    st.markdown(
        "- **Herding strength (a1):** how strongly chartists copy the majority mood. 0 means not at all.\n"
        "- **Trend following (a2):** how strongly chartists react to the recent price trend.\n"
        "- **Mood changes (v1):** how often chartists reconsider whether they are optimistic.\n"
        "- **Days and seed:** how long a history to simulate and which random history to draw. "
        "The same settings and seed always give exactly the same history.\n\n"
        "Every other setting stays at the values fitted to the S&P 500 (`settings/sp500_calibrated.json`).")

    st.header("Words used on the site", anchor=False)
    st.markdown(
        "- **Fat tails:** big days happen far more often than a bell curve predicts.\n"
        "- **Tail exponent:** how fast big days get rarer. Real markets are near 3; smaller means fatter tails.\n"
        "- **Excess kurtosis:** 0 for a bell curve; bigger means more extreme days.\n"
        "- **Volatility clustering:** wild days come in bunches, and so do calm ones.\n"
        "- **Drawdown:** how far the price is below its last high.\n"
        "- **Annualized volatility:** how much the price typically moves in a year.\n"
        "- **Runaway:** the simulated price leaves its true value and never comes back, a bubble or crash "
        "that does not end. The site reports these instead of hiding them.")

    st.header("Things to keep in mind", anchor=False)
    st.markdown(
        "- **One run is one random draw.** The Workspace shows one simulated history, so its numbers move "
        "around when you change the seed. The Checks page and the figures use 40 histories and report the "
        "median, which is why their numbers are steadier.\n"
        "- **Everything starts from the S&P 500 fit.** Other markets are compared on shape only (tail, "
        "kurtosis, memory), which doesn't depend on how volatile a market is.\n"
        "- **This is not trading advice.** The model explains the shape of returns, not where a price goes next.\n"
        "- **Data:** saved prices end on 30 October 2026 (Yahoo Finance, FRED, MeasuringWorth). Live and "
        "intraday prices come from Yahoo Finance and are sometimes unavailable.")

    st.header("Run it yourself", anchor=False)
    st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")
    st.markdown("Tests: `python3 -m pytest tests/ -q`. Every number on the site comes from "
                "`herding/explore.py`, which is tested in `tests/test_explore.py`.")


page()
