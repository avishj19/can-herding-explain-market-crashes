"""Web app: can herding explain market crashes?

Run on this computer:  streamlit run app.py   (then open http://localhost:8501)

Pages live in web/ (one file each); web/common.py holds the shared colors, spacing and cached data.
The pages only draw. Every number comes from herding/explore.py (tested in tests/test_explore.py),
which runs the model in herding/model.py (docs/SPEC.md).
"""
import streamlit as st

from web import common

st.set_page_config(page_title="Herding and market crashes", page_icon="📉", layout="wide",
                   initial_sidebar_state="expanded")
common.setup()

nav = st.navigation([
    st.Page("web/page_guide.py", title="Start here", icon=":material/home:", default=True),
    st.Page("web/page_story.py", title="Story", icon=":material/auto_stories:"),
    st.Page("web/page_explore.py", title="Workspace", icon=":material/candlestick_chart:"),
    st.Page("web/page_compare.py", title="Compare", icon=":material/scatter_plot:"),
    st.Page("web/page_checks.py", title="Checks", icon=":material/fact_check:"),
    st.Page("web/page_findings.py", title="Findings", icon=":material/photo_library:"),
], position="top")
nav.run()
