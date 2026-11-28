"""The fixes from the Opus review of the site rebuild (28 Nov 2026): each test fails if its problem comes back."""
import inspect
import re

import numpy as np
import pytest

from herding import explore as ex, stats
from herding.market_data import ROOT
from test_web_pages import open_page  # noqa: F401


def _text(at):
    return " ".join([m.value for m in at.markdown] + [x.value for x in at.caption])


# Finding 2: the site's kurtosis is EXCESS kurtosis (0 for a bell curve), so it is never labelled plain "Kurtosis".
def test_kurtosis_is_excess_and_labelled_so():
    r = np.random.default_rng(1).standard_normal(200_000)
    assert abs(stats.kurtosis(r)) < 0.05                        # a bell curve gives 0: excess kurtosis
    from web import common as c
    assert c.HELP["kurtosis"].startswith("Excess kurtosis. 0 for a bell curve")


@pytest.mark.parametrize("page", ["web/page_guide.py", "web/page_story.py", "web/page_explore.py",
                                  "web/page_checks.py"])
def test_no_page_says_plain_kurtosis_next_to_a_number(open_page, page):
    at = open_page(page)
    text = _text(at)
    bare = re.findall(r"(?<![Ee]xcess )\b[Kk]urtosis\b(?: \(0[^)]*\))? \*{0,2}[+-]?\d", text)
    assert not bare, bare


# Finding 3: the Story's verdict marks direction and weeks as the miss, and memory of wild days as a match.
def test_story_verdict_rows(open_page):
    at = open_page("web/page_story.py")
    at.session_state["story_step"] = 7
    at.run()
    html = " ".join(h.proto.body for h in at.get("html")) + _text(at)
    assert "<b>Memory.</b>" not in html and "Daily reversal and fading tails" in html
    spx = ex.summary(ex.log_returns(ex.saved_prices("S&P 500")).values)
    mem = ex.check_table(ex.web_numbers(), spx).loc["memory"]
    assert f"memory of wild days {mem['median']:.2f} against {mem['sp500']:.2f}" in html


# Finding 5: the party study points at the Figures tab, not "below".
def test_party_study_points_at_the_figures_tab(open_page):
    at = open_page("web/page_checks.py")
    text = _text(at)
    assert "Figures 12 to 15 (the Figures tab)" in text and "Figures 12 to 15 below" not in text


# Finding 6: numbers on the pages are worked out, or pinned to the report they quote.
def test_party_gap_year_matches_party_view():
    dow = ex.log_returns(ex.saved_prices("Dow Jones"))
    assert ex.party_gap_year(dow) == pytest.approx(ex.party_view(dow)["gap_year"], rel=1e-12)
    with pytest.raises(ValueError):
        ex.party_gap_year(dow.values)


def test_tail_range_is_the_checks_rule():
    sig = inspect.signature(ex.tail_check)
    assert (sig.parameters["lo"].default, sig.parameters["hi"].default) == ex.TAIL_RANGE == (2.5, 3.5)


def test_explore_numbers_are_worked_out(open_page):
    at = open_page("web/page_explore.py")
    src = (ROOT / "web" / "page_explore.py").read_text()
    assert "+3.5% a year" not in src and "(a bell curve: 0.06)" not in src and "over 40 simulated" not in src
    assert f"{ex.bell_big_days(1000):.2f}" == "0.06"
    assert not at.exception


def test_quoted_numbers_match_their_sources():
    results = (ROOT / "figures" / "RESULTS.md").read_text()
    guide = (ROOT / "web" / "page_guide.py").read_text()
    assert "Hurst H 0.39 against 0.50" in guide
    assert "(H 0.39), while the S&P's do not (0.50)" in results           # the report the Start page quotes
    from scripts.analysis import party_scale
    assert inspect.signature(party_scale.main).parameters["n"].default == 1000
    assert "1,000 model histories" in (ROOT / "web" / "page_findings.py").read_text()
