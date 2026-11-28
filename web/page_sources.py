"""Sources: the papers, data and tools behind the site, each with what it was used for.
Links only where they were checked (DOIs, or free arXiv copies). Wording: docs/redesign/WORDING.md."""
import streamlit as st

from web import common as c

REPO = "https://github.com/avishj19/can-herding-explain-market-crashes"

# (group, [(citation in Markdown, what the site used it for)])
SOURCES = [
    ("The model", [
        ("Lux, T. & Marchesi, M. (1999). Scaling and criticality in a stochastic multi-agent model of a financial "
         "market. *Nature* 397, 498–500. [doi:10.1038/17290](https://doi.org/10.1038/17290)",
         "The model this site rebuilds, and the results it reproduces."),
        ("Samanidou, E., Zschischang, E., Stauffer, D. & Lux, T. (2007). Agent-based models of financial markets. "
         "*Rep. Prog. Phys.* 70, 409–450. [arXiv:physics/0701140](https://arxiv.org/abs/physics/0701140)",
         "Every equation of the model (docs/SPEC.md)."),
        ("Cristelli, M., Pietronero, L. & Zaccaria, A. (2011). Critical overview of agent-based models for economics. "
         "[arXiv:1101.1847](https://arxiv.org/abs/1101.1847)",
         "The same equations, checked a second time."),
        ("Lux, T. & Marchesi, M. (2000). Volatility clustering in financial markets: a microsimulation of interacting "
         "agents. *Int. J. Theor. Appl. Finance* 3, 675–702.",
         "The paper's settings tables (not yet obtained, so this site fits its own)."),
        ("Kirman, A. (1993). Ants, rationality, and recruitment. *Quarterly Journal of Economics* 108, 137–156.",
         "The ant model behind herding, checked against its exact formula first."),
        ("Dewachter, H. & Lyrio, M. The economic value of technical trading rules: a nonparametric utility-based "
         "approach.",
         "Why trend chasers (chartists) are worth modelling: traders really do trade on past price patterns."),
    ]),
    ("Measuring markets", [
        ("Gopikrishnan, P. et al. (1998). Inverse cubic law for the distribution of stock price variations. "
         "*Eur. Phys. J. B* 3, 139–140. [arXiv:cond-mat/9803374](https://arxiv.org/abs/cond-mat/9803374)",
         "The target for the tail exponent: about 3."),
        ("Cont, R. (2001). Empirical properties of asset returns: stylized facts and statistical issues. "
         "*Quantitative Finance* 1, 223–236. [doi:10.1080/713665670](https://doi.org/10.1080/713665670)",
         "The textbook checklist of patterns in real markets."),
        ("Peng, C.-K. et al. (1994). Mosaic organization of DNA nucleotides. *Phys. Rev. E* 49, 1685–1689.",
         "The method that measures how long wild spells last (DFA, Hurst H)."),
    ]),
    ("Beyond the paper", [
        ("Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *J. Econometrics* 31, "
         "307–327. [doi:10.1016/0304-4076(86)90063-1](https://doi.org/10.1016/0304-4076(86)90063-1)",
         "GARCH: a rival formula for bursts of wild days, with no traders in it."),
        ("Glosten, L., Jagannathan, R. & Runkle, D. (1993). On the relation between the expected value and the "
         "volatility of the nominal excess return on stocks. *J. Finance* 48, 1779–1801.",
         "The part of GARCH that makes falls bigger than rises."),
        ("Bouchaud, J.-P., Matacz, A. & Potters, M. (2001). Leverage effect in financial markets: the retarded "
         "volatility model. *Phys. Rev. Lett.* 87, 228701. "
         "[doi:10.1103/PhysRevLett.87.228701](https://doi.org/10.1103/PhysRevLett.87.228701)",
         "The \"panic\" idea, tested here. It does not make falls bigger than rises."),
    ]),
    ("Data", [
        ("Yahoo Finance.", "The S&P 500 (^GSPC), the Nasdaq-100, 5 stocks, gold and copper, and live prices."),
        ("FRED, Federal Reserve Bank of St. Louis.",
         "A second copy of the S&P 500 to check against, and Brent oil."),
        ("MeasuringWorth.", "The Dow Jones since 1896."),
    ]),
]
# the proposal planned Stooq for the S&P 500; by the time the data was downloaded Stooq answered programs with a
# browser check and gates its downloads behind a key, so Yahoo was used, with FRED as an independent copy
DATA_NOTE = ("The project proposal planned to take the S&P 500 from Stooq. Stooq now blocks automated downloads, so "
             "the site uses Yahoo Finance and checks it against an independent copy from FRED: they agree on every "
             "day both have (2016 to 2026) to within 0.12%.")
EDUCATION = ("**For educational purposes only.** This is a student project for 02-120 Programming for "
             "Scientists at Carnegie Mellon University. It is a simulation for learning, not a forecast, "
             "not trading advice and not investment advice.")


def page():
    st.title("Sources", anchor=False)
    st.warning(EDUCATION, icon=":material/school:")
    c.lead("The papers, data and tools behind this site, and what each one was used for.")
    n = 0
    for group, items in SOURCES:
        st.header(group, anchor=False)
        for cite, use in items:
            n += 1
            st.markdown(f"**[{n}]** {cite}  \n<span class='muted'>{use}</span>", unsafe_allow_html=True)
        if group == "Data":
            st.caption(DATA_NOTE)
    st.header("Tools", anchor=False)
    st.markdown(f"Python, NumPy, SciPy, pandas, Numba, Plotly and Streamlit. The code and its tests are public: "
                f"[{REPO.removeprefix('https://')}]({REPO}).")


page()
