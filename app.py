"""Web app: can herding explain market crashes?

Run on this computer:  streamlit run app.py   (then open http://localhost:8501)

Pages live in web/ (one file each); web/common.py holds the shared colors, spacing and cached data.
The pages only draw. Every number comes from herding/explore.py (tested in tests/test_explore.py),
which runs the model in herding/model.py (docs/SPEC.md).
"""
import importlib
import os

import streamlit as st

# The site's own modules, each after the ones it imports. Streamlit reruns app.py and the pages on
# every visit, but keeps imported modules in memory: after a push the live server kept the old
# herding/explore.py and web/common.py and crashed ("setup() takes 0 positional arguments",
# "no attribute KIRMAN_PASS", Nov 2026) until someone rebooted it by hand.
SITE_MODULES = ["herding.checks", "herding.stats", "herding.model", "herding.market_data",
                "herding.factors", "herding.kirman", "herding.explore", "web.common"]


def _loaded_at(m):
    """When this copy of module m was loaded. After a reload here it is recorded in m._site_mtime.
    Before that (a module imported before this check ran, like on the live server the first time):
    the time of its compiled .pyc file, which Python writes or checks when it imports the module,
    so a source file newer than its .pyc changed after the module was loaded."""
    if hasattr(m, "_site_mtime"):
        return m._site_mtime
    cached = getattr(m, "__cached__", None)
    return os.path.getmtime(cached) if cached and os.path.exists(cached) else os.path.getmtime(m.__file__)


def reload_if_changed(names=SITE_MODULES):
    """Reload all the site's modules, in order, when any of their files changed since it was loaded.
    Returns the names reloaded (empty when nothing changed)."""
    mods = [importlib.import_module(n) for n in names]
    if not any(os.path.getmtime(m.__file__) > _loaded_at(m) for m in mods):
        for m in mods:
            m._site_mtime = _loaded_at(m)
        return []
    for n in names:               # all of them, in order, so nobody keeps a reference to an old copy
        m = importlib.reload(importlib.import_module(n))
        m._site_mtime = os.path.getmtime(m.__file__)
    return list(names)


reload_if_changed()
from web import common  # noqa: E402  (after the reload, so this is the current copy)

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
