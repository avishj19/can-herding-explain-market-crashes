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


# Finding 6: numbers on the pages are worked out, or pinned to the report they quote.
def test_tail_range_is_the_proposals():
    # check 3 uses the proposal's own range (written before any result), not one widened afterwards
    sig = inspect.signature(ex.tail_check)
    assert (sig.parameters["lo"].default, sig.parameters["hi"].default) == ex.PROPOSAL_TAIL == (2.6, 3.0)
    assert not hasattr(ex, "TAIL_RANGE")


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


# ---------------------------------------------------------------- Phase A review (Opus, 28 Nov): honesty lines
def test_checks_page_states_the_uncertain_wait_and_the_fit(open_page):
    at = open_page("web/page_checks.py")
    text = _text(at)
    nums = ex.web_numbers()
    spx_r = ex.log_returns(ex.saved_prices("S&P 500")).values
    cc = ex.crash_check(nums, spx_r)
    lo, hi = cc["bm_range"]
    assert f"anywhere from {lo:,.0f} years to {hi:,.0f} years" in text           # Black Monday: few falls, wide range
    assert "this tail exponent was one of the fit's 11 targets" in text            # check 3 is partly fitted
    assert "Fully independent of the fit: check 1" in text and "not a target, but it follows largely" in text
    k_wo = ex.kurtosis_without_worst(spx_r)
    assert f"Without that day it is {k_wo:.1f}" in text and round(k_wo, 1) == 11.4   # one day drives the 25
    wo = ex.spread(nums["checks"]["herding"]["kurtosis_wo_worst"])               # like for like (Opus review B)
    assert f"middle 90% is {wo['low']:.1f} to {wo['high']:.1f}" in text and k_wo > wo["high"]
    assert "the S&P 500 is above it" in text


def test_start_and_story_say_the_tail_is_partly_fitted(open_page):
    at = open_page("web/page_guide.py")
    short = [m.value for m in at.markdown if "Short answer" in m.value][0]
    assert "above the proposal's 2.6 to 3.0" in short and "partly by design" in short
    assert "very uncertain" in short
    at = open_page("web/page_story.py")
    at.session_state["story_step"] = 5
    at.run()
    assert "partly by design" in _text(at) and "very uncertain" in _text(at)


# Spec: Sources cites Dewachter & Lyrio (named in the proposal) and says why Yahoo replaced the proposal's Stooq.
def test_sources_page_names_the_proposals_source_change(open_page):
    at = open_page("web/page_sources.py")
    text = _text(at)
    assert "Dewachter, H. & Lyrio, M." in text
    assert "planned to take the S&P 500 from Stooq" in text and "within 0.12%" in text
