"""Findings: every project figure with its plain caption (from figures/GALLERY.md)."""
import streamlit as st

from herding import explore as ex
from herding.market_data import ROOT
from web import common as c

FIG = ROOT / "figures"


@st.cache_data(show_spinner=False)
def _entries():
    return ex.gallery_entries((FIG / "GALLERY.md").read_text())


@st.fragment
def gallery(entries):
    sections = list(dict.fromkeys(e["section"] for e in entries))
    pick = c.pick_one("Section", sections, sections[0], "findings_section", label_visibility="collapsed")
    for e in (e for e in entries if e["section"] == pick):
        st.image(str(FIG / e["file"]), width="stretch")
        st.markdown(e["caption"].replace(" ✅", ""))
        st.divider()


def page():
    st.title("Findings", anchor=False)
    c.lead("The project's figures, grouped by the question each one answers. They average over many "
           "simulated histories, so they are steadier than a single run in the Workspace. Full numbers are "
           "in figures/RESULTS.md.")
    gallery(_entries())


page()
