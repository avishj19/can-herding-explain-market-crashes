"""Web app: can herding explain market crashes?

Run on this computer:  streamlit run app.py   (then open http://localhost:8501)

Pages live in web/ (one file each); web/common.py holds the shared colors, spacing and cached data.
The pages only draw. Every number comes from herding/explore.py (tested in tests/test_explore.py),
which runs the model in herding/model.py (docs/SPEC.md).
"""
import streamlit as st

from web import common

st.set_page_config(page_title="Herding and market crashes", page_icon="📉", layout="wide",
                   initial_sidebar_state="auto")   # open on wide screens; "expanded" covered the page on phones
PAGES = [
    st.Page("web/page_guide.py", title="Start here", icon=":material/home:", default=True),
    st.Page("web/page_story.py", title="Story", icon=":material/auto_stories:"),
    st.Page("web/page_explore.py", title="Workspace", icon=":material/candlestick_chart:"),
    st.Page("web/page_compare.py", title="Compare", icon=":material/scatter_plot:"),
    st.Page("web/page_checks.py", title="Checks", icon=":material/fact_check:"),
    st.Page("web/page_findings.py", title="Findings", icon=":material/photo_library:"),
]
nav = st.navigation(PAGES, position="top")
# every page's address, so the light/dark button can set the mode for pages not opened yet.
# After st.navigation: a page only gets its url_path when navigation registers it (reading it
# first crashed the live site on a fresh start, Nov 2026)
common.setup(["/" + (pg.url_path or "") for pg in PAGES])
nav.run()
