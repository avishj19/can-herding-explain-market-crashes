"""Findings (tabs of the Checks page since the redesign): what we found about Lux & Marchesi (1999),
then every project figure with its plain caption (from figures/GALLERY.md). Every number comes from
web/numbers.json (scripts/web_numbers.py, the same recipes as the figures) through herding/explore.py."""
import numpy as np
import streamlit as st

from herding import explore as ex
from herding.market_data import ROOT
from web import common as c

FIG = ROOT / "figures"
REPO = "https://github.com/avishj19/can-herding-explain-market-crashes/blob/main/"
RESULTS = REPO + "figures/RESULTS.md"
LIMITS = REPO + "docs/KEY_POINTS.md#7-what-the-model-gets-wrong-honest-limitations"
# short names for the gallery's section buttons (the full title is shown under them), so all five fit a phone
SHORT = {"0": "0. Quick check", "1": "1. Checks", "2": "2. Fits S&P 500?", "3": "3. Crashes", "4": "4. Robustness"}


@st.cache_data(show_spinner=False)
def _entries_of(stamp):
    return ex.gallery_entries((FIG / "GALLERY.md").read_text())


def _entries():
    """The gallery's figures and captions, read again when figures/GALLERY.md changes."""
    return _entries_of(c.file_stamp(FIG / "GALLERY.md"))


@st.cache_data(show_spinner=False)
def _spx_paper_tail_of(stamp):
    _, r, _ = c.saved_asset("S&P 500")
    return ex.paper_tail(r.values)


def _spx_paper_tail():
    """The S&P 500's tail exponent by the paper's method (regression over the largest 30% of days)."""
    return _spx_paper_tail_of(c.data_stamp())


def short_label(section):
    """The gallery button's label: a short name for a known section number, else the title itself."""
    return SHORT.get(section.split(".")[0].strip(), section)


@st.fragment
def gallery(entries):
    sections = list(dict.fromkeys(e["section"] for e in entries))
    pick = c.pick_one("Section", sections, sections[0], "findings_section", label_visibility="collapsed",
                      format_func=short_label)
    st.subheader(pick, anchor=False)
    for e in (e for e in entries if e["section"] == pick):
        st.image(str(FIG / e["file"]), width="stretch")
        st.markdown(e["caption"].replace(" ✅", ""))
        st.divider()


def paper_rows(nums):
    """The comparison table's rows, as text, from explore.paper_comparison."""
    out = []
    for r in ex.paper_comparison(nums):
        tol = r["tolerance"]
        out.append({"Measure": r["measure"],
                    "Lux & Marchesi": f"{r['paper']:.2f} ± {r['paper_err']:.3f}",
                    "Our rebuild": f"{r['ours']:.2f} ± {r['ours_sd']:.3f}",
                    "Within our tolerance?": f"{'yes' if r['close'] else 'no'} (± {tol:.2f})",
                    "Inside the paper's error bar?":
                        "yes" if r["inside_err"] else "no"})
    return out


def html_table(rows):
    """A small table that wraps its text, so every column fits a phone (st.dataframe showed only the
    first column at 375 pixels)."""
    cols = list(rows[0])
    cell = "padding:.35rem .5rem;border-bottom:1px solid rgba(128,128,128,.3);vertical-align:top"
    head = "".join(f"<th style='{cell};text-align:left;font-weight:600'>{h}</th>" for h in cols)
    body = "".join("<tr>" + "".join(f"<td style='{cell}'>{r[h]}</td>" for h in cols) + "</tr>" for r in rows)
    return (f"<table class='paper' style='width:100%;border-collapse:collapse;font-size:.85rem;"
            f"table-layout:auto'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")


def paper_section():
    nums = c.numbers()
    rows = ex.paper_comparison(nums)
    pp = nums["paper"]
    st.header("The paper we tested: Lux & Marchesi (1999)", anchor=False)
    c.lead("In <i>Nature</i> (vol. 397, pages 498-500), Thomas Lux and Michele Marchesi built a market of "
           "traders who switch between two strategies: <b>value traders</b> (economists call them fundamentalists), "
           "who trade on the true value, and <b>trend chasers</b> (economists call them chartists), who follow "
           "the trend and copy each other. The news in their model is plain bell-curve noise. Yet the prices "
           "came out with fat tails and bursts of wild days. Their claim: those come from the traders, not from "
           "the news. This project rebuilt their model from its equations and tested that claim in two ways.")
    st.markdown("This project fits the model's settings in two ways. **Paper-matched** settings are fitted "
                "to the numbers the paper reports; **S&P-calibrated** settings are fitted to the S&P 500. "
                "Section 1 uses the first, section 2 the second.")

    st.subheader("1. Does our rebuild give the paper's numbers?", anchor=False)
    st.markdown(html_table(paper_rows(nums)), unsafe_allow_html=True)
    st.caption(f"The paper's ± is the fit error of its single long run; ours is how much {len(pp['seeds'])} "
               f"simulated histories differ from each other. Our tolerance is the gap this project accepts when fitting.")
    k1, k25 = np.median(pp["kurtosis"]), np.median(pp["kurt_25"])
    miss = [r for r in rows if not r["close"]]
    missed = "; ".join(f"{r['measure'].split(' (')[0]}: {r['ours']:.2f} against {r['paper']:.2f}"
                       for r in miss)
    st.markdown(
        f"Ours: the mean of {len(pp['seeds'])} simulated histories of {pp['days']:,} days each, on seeds never used to "
        f"fit the settings (they were used to choose between fits), ± how much they vary from one history to "
        f"the next. The tail exponent here is measured the paper's way: a straight line fitted to the largest "
        f"30% of days. The *Nature* letter gives no settings (the follow-up paper with its tables was not "
        f"yet obtained), so ours are fitted to its numbers. The match holds with the model's time step of 0.01 "
        f"day, as in the paper. With a finer time step these settings lose most of their fat tails. Like the "
        f"paper, the fat tails fade over longer spans: excess kurtosis (0 for a bell curve) falls from {k1:.0f} for "
        f"one day to {k25:.1f} for 25 days."
        + (f" **Not reproduced:** {missed}." if miss else "")
        + (" Our bursts of wild days are a little weaker than the paper's." if "H_abs" in [r["key"] for r in miss]
           else ""))
    st.markdown("Hurst H is a memory score where 0.5 means no memory. It is a different score from the "
                "*memory of wild days* on the other pages, where 0 means no memory.")

    st.subheader("2. Does herding explain the S&P 500?", anchor=False)
    chk = nums["checks"]
    herd, fund = ex.spread(chk["herding"]["tail"]), ex.spread(chk["fundamentalists"]["kurtosis"])
    gone, total = ex.runaway_count(nums, "chartists")
    _, _, spx = c.saved_asset("S&P 500")
    st.markdown(
        f"Settings fitted to the S&P 500, {herd['n']} simulated histories as long as its record. Here the tail "
        f"exponent is the Hill estimate over the largest 5% of days, as on the rest of the site:\n"
        f"- **Value traders only:** a bell curve (median excess kurtosis {fund['median']:.2f}, where a bell curve "
        f"gives 0). Bell-curve news alone gives bell-curve moves.\n"
        f"- **Trend chasers, no herding:** the price runs away in {gone} of {total} simulated histories.\n"
        f"- **Trend chasers who herd:** fat tails with a median tail exponent of **{herd['median']:.2f}**, "
        f"against **{spx['tail']:.2f}** for the S&P 500 and about 3 for markets in general (lower = more big days).")
    st.markdown(
        f"By the paper's own method (largest 30% of days), the S&P 500's tail exponent is "
        f"{_spx_paper_tail():.2f}: fatter than every one of the model's simulated histories in the project's "
        f"[report]({RESULTS}#fig-7--is-the-sp-500-a-typical-history-of-the-model) (Fig 7). Its middle-to-large "
        f"days are more common than the model's.")
    st.markdown(
        "<p class='lead'><b>Our finding:</b> the paper's main claim holds in our rebuild. Herding alone, "
        "with plain bell-curve news, gives the fat tails and bursts of wild days of a real market. It does not "
        "explain everything. The S&P 500 falls harder than it rises, and the model does not. The model's "
        "prices reverse a little from one day to the next, and the S&P 500's do not. The model loses its fat "
        "tails over a few weeks somewhat faster than the S&P 500. Its settings are fitted, not the paper's "
        "own, and they rarely run away into an endless bubble. "
        f"<a href='{LIMITS}'>All the limitations</a> (KEY_POINTS, section 7).</p>", unsafe_allow_html=True)


def figures_header():
    st.header("The project's figures", anchor=False)
    st.markdown("**In these figures:** black = S&P 500, blue = model fitted to the S&P 500, orange = model "
                "fitted to the paper's numbers, green = news only / no trend chasers. This differs from the colors "
                "on the rest of the site.")


def figures():
    """The Figures tab: every project figure, grouped by the question each one answers."""
    c.lead("The project's figures, grouped by the question each one answers. Most figures average over many "
           "simulated histories, so they are steadier than a single simulated history in the Test program. Full numbers are in the "
           f"project's <a href='{RESULTS}'>report</a> (figures/RESULTS.md).")
    figures_header()
    gallery(_entries())
