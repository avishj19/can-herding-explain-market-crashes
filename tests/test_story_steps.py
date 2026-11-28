"""The Story page (web/page_story.py): 8 guided steps with Next / Back, a step list and a progress bar.

Run headless with Streamlit's AppTest, no network:  python3 -m pytest tests/test_story_steps.py -q
Expected numbers are recomputed here from the saved files with explore functions or plain numpy, not read
from the page code, so a page that shows a wrong number fails."""
import json
import math

import numpy as np
import pytest

from conftest import ROOT
from herding import explore as ex
from herding import kirman
from test_web_pages import open_page  # noqa: F401

pytestmark = pytest.mark.slow

NUMBERS = json.loads((ROOT / "web" / "numbers.json").read_text())
TITLES = ["Real markets have days a bell curve says should never happen",
          "The idea: traders copy each other, like ants do",
          "So we build a market of simulated traders and feed it bell-curve news",
          "Experiment 1: with only value traders, daily moves follow a bell curve",
          "Experiment 2: without herding, trend chasers make the price run away",
          "Experiment 3: with herding, the model gets fat tails and bursts",
          "How much herding? Guess first",
          "The verdict: herding explains a lot, not everything"]
CHART_KEYS = {1: ["s1"], 2: ["s_ants"], 3: [], 4: ["s2"], 5: ["s3"], 6: ["s4", "s4b"], 7: [], 8: []}


def _step(at):
    return at.session_state["story_step"] + 1


def _text(at):
    return "\n".join([m.value for m in at.markdown] + [c.value for c in at.caption])


def _next(at):
    at.button(key="story_next").click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _chart_keys(at):
    return {pc.proto.id.split("-")[-1] for pc in at.get("plotly_chart")}


def test_all_eight_steps_render(open_page):
    at = open_page("web/page_story.py")
    for n in range(1, 9):
        assert not at.exception, (n, [e.value for e in at.exception])
        assert _step(at) == n
        assert [h.value for h in at.header] == [TITLES[n - 1]], n
        assert f"STEP {n} OF 8" in _text(at)
        assert set(CHART_KEYS[n]) <= _chart_keys(at), (n, _chart_keys(at))
        if n < 8:
            _next(at)


def test_progress_bar_and_step_list(open_page):
    at = open_page("web/page_story.py")
    labels = [b.label for b in at.button if b.key.startswith("story_go_")]
    assert len(labels) == 8 and labels[0].endswith("The puzzle") and labels[-1].endswith("The verdict")
    for n in (1, 2, 8):
        if n > 1:
            at.button(key=f"story_go_{n - 1}").click().run()
        assert _step(at) == n
        assert at.get("progress")[0].proto.value == round(n / 8 * 100)
        marked = [b.key for b in at.button if b.key.startswith("story_go_") and b.proto.type == "primary"]
        assert marked == [f"story_go_{n - 1}"]


def test_back_on_step_1_and_next_on_step_8_stay_in_range(open_page):
    at = open_page("web/page_story.py")
    assert at.button(key="story_back").disabled
    assert at.session_state["story_step"] == 0          # a visitor cannot click a disabled button
    for _ in range(7):
        _next(at)
    assert at.session_state["story_step"] == 7 and at.button(key="story_next").disabled
    at.button(key="story_back").click().run()
    assert at.session_state["story_step"] == 6


def test_a_stray_step_value_is_pulled_into_range(open_page):
    at = open_page("web/page_story.py")
    at.session_state["story_step"] = 99
    at.run()
    assert not at.exception and [h.value for h in at.header] == [TITLES[7]]
    at.session_state["story_step"] = -4
    at.run()
    assert not at.exception and [h.value for h in at.header] == [TITLES[0]]


def test_numbers_on_steps_match_the_backend(open_page):
    at = open_page("web/page_story.py")
    from web import common as c
    _, _, spx = c.saved_asset("S&P 500")
    # step 1: big days and the bell curve's wait for a 20-sd fall (10 to the 86)
    t = _text(at)
    assert f"**{spx['big_days']} big days**" in t and f"**{spx['bell_big_days']:.1f}**" in t
    e = math.floor(math.log10(ex.bell_crash_years(20)))
    assert e == 86 and "10⁸⁶ years" in t and "10^86" not in t and "1e86" not in t
    assert f"a 1 followed by {e} zeros" in t
    # step 2: Kirman's distance equals the one the check computes
    _next(at)
    k = kirman.simulate(100, 0.002, 0.1, 20_000_000, seed=1)
    tv = ex.kirman_match(k, 100, 0.002, 0.1)[2]
    assert f"{tv:.3f}" in _text(at) and "did not watch ants himself" in _text(at)
    # step 3: the trader count
    _next(at)
    assert f"**{int(c.fitted()[0]['N'])} simulated traders**" in _text(at)
    # step 4: median kurtosis of the held-out histories
    _next(at)
    kur = NUMBERS["checks"]["fundamentalists"]["kurtosis"]
    assert f"**{np.median(kur):.2f}** (median) over {len(kur)} simulated histories" in _text(at)
    # step 5: how many chartist histories ran away
    _next(at)
    gone = sum(d is not None for d in NUMBERS["checks"]["chartists"]["runaway_day"])
    assert f"{gone} of 40" in _text(at)
    # step 6: median tail exponent and crash waits
    _next(at)
    tails = [x for x in NUMBERS["checks"]["herding"]["tail"] if x is not None]
    cc = ex.crash_check(NUMBERS, c.saved_asset("S&P 500")[1].values)
    t = _text(at)
    assert f"**{np.median(tails):.2f}** (median) over {len(tails)} simulated histories" in t
    assert f"**{cc['row']['model']:.0f} years**" in t and f"**{cc['bm_model']:,.0f} years**" in t
    # step 8: the verdict repeats those numbers
    _next(at)
    _next(at)
    t = _text(at)
    assert f"{np.median(tails):.2f} against {spx['tail']:.2f}" in t
    assert f"every {cc['row']['model']:.0f} years" in t


def test_guess_is_hidden_until_answered_and_revealed_after(open_page):
    at = open_page("web/page_story.py")
    for _ in range(6):
        _next(at)
    assert "s5" not in _chart_keys(at)
    at.get("button_group")[0].set_value("Wilder").run()
    assert "s5" in _chart_keys(at)
    assert "You said: *Wilder*" in _text(at)
    sweep = ex.sweep_at(NUMBERS, 0.0)
    assert f"runs away in all {sweep['n']} saved histories" in _text(at)


def test_step_8_links_point_at_real_pages(open_page):
    at = open_page("web/page_story.py")
    at.session_state["story_step"] = 7
    at.run()
    pages = {pl.proto.page for pl in at.get("page_link")}
    assert {"page_explore", "page_checks"} <= pages
    assert not any(p in ("page_compare", "page_findings") for p in pages)


def test_wording_rules_on_every_step(open_page):
    at = open_page("web/page_story.py")
    bad = ["fundamental value", "volatility clustering", "SD ", "Gaussian", "normal distribution", "10^86", "1e86",
           "yrs", "tail score", "can't", "don't", "doesn't"]
    for n in range(1, 9):
        t = _text(at)
        for b in bad:
            assert b not in t, (n, b)
        if n < 8:
            _next(at)
