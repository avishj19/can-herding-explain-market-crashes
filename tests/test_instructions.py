"""Tests of the Nov 2026 instructions (research: PhET, Stripe, Linear, Titan, Brilliant): the Test program's
three tasks with progress, its Try-this lines, presets that say what to expect (explore.sweep_at),
Reset everything; the Story's guess-then-reveal step; the Start here numbers and first steps. No chat box."""
import math
import re

import numpy as np
import pytest

from conftest import ROOT
from herding import explore as ex
from test_web_explore_fixes import H, _group
from test_web_pages import app, open_page  # noqa: F401

NUMS = ex.web_numbers()
P, BURN = ex.fitted_settings()


# ------------------------------------------------------------------ sweep_at

# Spec: by hand from the saved rows: count the runaways and take the median of the other tails.
@pytest.mark.parametrize("a1", [0.0, 0.6, 0.8, 1.6, float(P["a1"])])
def test_sweep_at_by_hand(a1):
    rows = [r for r in NUMS["sweep"]["rows"] if abs(r["a1"] - a1) < 1e-9]
    tails = [r["tail"] for r in rows if r["tail"] is not None]
    got = ex.sweep_at(NUMS, a1)
    assert got["n"] == len(rows) == 10
    assert got["ran_away"] == len(rows) - len(tails)
    assert got["tail"] == (float(np.median(tails)) if tails else None)


def test_sweep_at_rejects_strengths_not_saved():
    with pytest.raises(ValueError):
        ex.sweep_at(NUMS, 0.7)


# Spec: the science the instructions teach: runaways fall and the tail exponent rises (calmer) as
# herding grows, in the saved sweep, wherever at least half the histories survive.
def test_more_herding_is_calmer_in_the_saved_sweep():
    a1s = sorted({r["a1"] for r in NUMS["sweep"]["rows"]})
    at = [ex.sweep_at(NUMS, a) for a in a1s]
    gone = [w["ran_away"] for w in at]
    assert all(x >= y for x, y in zip(gone, gone[1:]))
    tails = [w["tail"] for w in at if w["ran_away"] <= w["n"] / 2]
    assert tails[0] < 3.1 < tails[-1] and tails[-1] > 4


# ------------------------------------------------------------------ Test program

# Spec: each preset's note is the sweep's numbers in words.
def test_preset_notes_say_what_to_expect(app):
    at = app()
    html = [m.value for m in at.markdown if "class='preset-notes'" in m.value][0]
    notes = dict(re.findall(r"<div><b>([^<]+)</b> ([^<]+)</div>", html))
    assert list(notes) == ["Wilder", "Best fit", "Calmer", "No herding"]      # the order of the buttons
    assert notes["Best fit"] == f"herding strength {P['a1']:.2f}: tail exponent about {ex.sweep_at(NUMS, P['a1'])['tail']:.1f} (like the S&P 500)"
    assert notes["Calmer"] == "herding strength 1.60: tail exponent about 4.7 (calmer: thin tails)"
    assert notes["Wilder"] == "herding strength 0.60: runs away in 9 of 10; wild when it does not"
    assert notes["No herding"] == "herding strength 0.00: runs away in all 10 saved histories"


# Spec: the tasks tick only when really done: zoom to the biggest crash; run at a1 <= 0.8.
def test_tasks_tick_as_they_are_done(app):
    at = app()
    assert at.expander[0].label == "Three things to try · 0 of 3 done"
    _group(at, "view").set_value("Model market").run()
    _group(at, "mm_zoom").set_value("burst").run()
    assert at.expander[0].label == "Three things to try · 0 of 3 done"      # the wrong zoom doesn't count
    _group(at, "mm_zoom").set_value("crash").run()
    assert at.expander[0].label == "Three things to try · 1 of 3 done"
    at.slider(key="in_a1").set_value(0.81).run()
    assert at.expander[0].label == "Three things to try · 1 of 3 done"      # 0.81 is not the edge
    at.slider(key="in_a1").set_value(0.8).run()
    assert at.expander[0].label == "Three things to try · 2 of 3 done"
    assert not at.exception


# Spec: the challenge needs another seed AND the herding tail within 0.10 of the S&P 500's.
def test_challenge_rule():
    import streamlit as st
    ss = st.session_state
    spx = {"tail": 2.92}
    for seed, tail, want in ((1000, 2.95, False), (7, 2.95, True), (7, 3.05, False), (7, 2.83, True)):
        ss.clear()
        ss.run = dict(seed=seed)
        ss.log = []
        got = H["_tasks_done"]("S&P 500", spx, {"herding": {"tail": tail}})
        assert got[2] is want, (seed, tail)
    ss.clear()
    ss.run, ss.log = dict(seed=7), []
    assert H["_tasks_done"]("Apple", spx, {"herding": {"tail": 2.92}})[2] is False   # on the S&P 500 only
    ss.clear()


# Spec: Reset everything goes back to the S&P 500 and the fitted settings, and keeps the run log.
def test_reset_everything(app):
    at = app()
    at.slider(key="in_a1").set_value(1.5).run()
    n_log = len(at.session_state["log"])
    at.sidebar.button[0].click().run()
    assert at.session_state["in_a1"] == pytest.approx(float(P["a1"]), abs=1e-12)
    assert at.session_state["asset"] == ("saved", "S&P 500")
    assert len(at.session_state["log"]) >= n_log
    assert not at.exception


# Spec: every chart view has a Try-this line, and it names the market on screen where it says {name}.
def test_every_view_has_a_try_line():
    assert set(H["TRY"]) == set(H["LEAVES"]) == set(H["HINTS"])
    assert "{name}" in H["TRY"]["Tail"]


# Spec: no chat box anywhere on the site (the owner's rule: no AI usage on a public site).
def test_no_chat_box_on_any_page():
    for f in [ROOT / "app.py", *sorted((ROOT / "web").glob("*.py"))]:
        text = f.read_text()
        assert "chat_input" not in text and "chat_message" not in text and "anthropic" not in text.lower(), f


# ------------------------------------------------------------------ Story

# Spec: the chart of step 5 is hidden until a guess (or Just show me); each guess gets its verdict.
def test_story_guess_then_reveal(open_page):
    def step7():
        at = open_page("web/page_story.py")
        for _ in range(6):
            at.button(key="story_next").click().run()
        return at
    at = step7()
    assert not at.exception
    keys = [pc.proto.id for pc in at.get("plotly_chart")]
    assert not any(k.endswith("-s5") for k in keys)
    at.button(key="story_show").click().run()
    assert any(pc.proto.id.endswith("-s5") for pc in at.get("plotly_chart"))
    at = step7()
    at.get("button_group")[0].set_value("Calmer").run()
    text = " ".join(m.value for m in at.markdown)
    assert "Yes, surprisingly" in text and "runs away in all 10" in text
    assert any(pc.proto.id.endswith("-s5") for pc in at.get("plotly_chart"))


# ------------------------------------------------------------------ Start here

# Spec: the three cards are waits for the SAME fall, one the size of Black Monday, each computed: a bell
# curve's (10¹¹⁴ years), the S&P 500's record (77 years) and the herding model's (1,534 years). They used to
# compare a 20-sd fall, the S&P's record and the model's 8-sd fall (Opus review, 28 Nov).
def test_start_here_numbers(open_page):
    at = open_page("web/page_guide.py")
    assert not at.exception
    cc = ex.crash_check(ex.web_numbers(), ex.log_returns(ex.saved_prices("S&P 500")).values)
    vals = [m.value for m in at.metric]
    p = math.floor(math.log10(cc["bm_bell"]))
    assert p == 114 and vals[0] == "10¹¹⁴ years"
    assert vals[1] == "77 years" and vals[2] == f"{cc['bm_model']:,.0f} years" == "1,534 years"
    labels = [m.label for m in at.metric]
    assert labels == ["If daily moves followed a bell curve", "The S&P 500", "The herding model"]
    assert "The full manual: every button and chart" in [e.label for e in at.expander]
