"""Tests of the wording and chart fixes on Start here, Story, Checks, Findings and Compare (reviewers'
findings, Nov 23), run headless with Streamlit's AppTest and no network.

    python3 -m pytest tests/test_web_text_fixes.py -q          (slow: each test opens a page)

Expected numbers are recomputed here from web/numbers.json and the saved S&P 500 file with plain
numpy, not read from the page code, so a page that shows the wrong number fails.
"""
import json
import re
import socket

import numpy as np
import pytest

from conftest import ROOT
from herding import explore as ex

pytestmark = pytest.mark.slow

NUMBERS = json.loads((ROOT / "web" / "numbers.json").read_text())
REPO = "https://github.com/avishj19/can-herding-explain-market-crashes"
LINK = re.compile(re.escape(REPO) + r"(?:/blob/main/([^\s)'\"#]+))?")


def _no_daily(t, s):
    raise OSError("no daily prices in this test")


def _no_intraday(t, p, i):
    raise OSError("no intraday bars in this test")


@pytest.fixture
def open_page(monkeypatch):
    """Any page of the app, with no network (a copy of tests/test_web_pages.py's fixture)."""
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(socket.socket, "connect", lambda *a, **k: (_ for _ in ()).throw(OSError("blocked")))
    monkeypatch.setattr(ex, "_yahoo", _no_daily)
    monkeypatch.setattr(ex, "_yahoo_intraday", _no_intraday)
    st.cache_data.clear()

    def go(path):
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
        at.run()
        at.switch_page(path).run()
        assert not at.exception, [e.value for e in at.exception]
        return at
    yield go
    st.cache_data.clear()


def _text(at):
    return "\n".join([m.value for m in at.markdown] + [c.value for c in at.caption])


def _fig(at, key):
    for pc in at.get("plotly_chart"):
        if pc.proto.id.endswith(f"-{key}"):
            return json.loads(pc.proto.spec)
    raise AssertionError(f"no chart with key {key!r}")


def _trace(fig, name):
    for t in fig["data"]:
        if t.get("name") == name:
            return t
    raise AssertionError(f"no trace {name!r} in {[t.get('name') for t in fig['data']]}")


def _hill(r, tail=0.05):
    """Hill tail exponent over the largest `tail` share of |returns| (zero days dropped), by hand."""
    z = np.sort(np.abs(r[r != 0]))[::-1]
    k = int(len(z) * tail)
    return k / np.sum(np.log(z[:k] / z[k]))


def _links_ok(text):
    """Every GitHub link in the text is well formed and points at a file or folder that exists here."""
    found = LINK.findall(text)
    for path in found:
        if path:
            assert (ROOT / path).exists(), f"GitHub link to a missing file: {path}"
    return len(re.findall(re.escape(REPO), text))


# ---------------------------------------------------------------------------------- Start here
def test_guide_short_answer_numbers_match_the_saved_histories(open_page):
    at = open_page("web/page_guide.py")
    text = _text(at)
    tails = [t for t in NUMBERS["checks"]["herding"]["tail"] if t is not None]
    spx_tail = ex.summary(ex.log_returns(ex.saved_prices("S&P 500")).values)["tail"]
    short = [m.value for m in at.markdown if "Short answer" in m.value]
    assert len(short) == 1
    s = short[0]
    assert f"tail exponent is {np.median(tails):.2f} (median of {len(tails)} simulated histories)" in s
    assert f"the S&P 500's is {spx_tail:.2f}" in s
    assert "H 0.39 against 0.50" in s and f"{REPO}/blob/main/figures/RESULTS.md" in s
    assert "falls bigger than rises" in s and "few weeks" in s
    assert any(p.proto.page.endswith("page_checks.py") or "checks" in p.proto.page.lower()
               for p in at.get("page_link"))      # the findings are tabs of the Checks page since the redesign
    # the manual describes the Test program as it is now (Opus review, 28 Nov): its four chart tabs, the
    # presets in the model panel, the settings in the sidebar; never the old "Simulated market" view
    for words in ("**Watch a crash**", "**How rare big days are**", "**Bursts of wild days**", "**Wilder**",
                  "**Best fit**", "**Calmer**", "**No herding**", "**⚙ Settings**", "☰"):
        assert words in text, words
    assert "Simulated market" not in text and "Model presets" not in text and "top right" not in text


def test_guide_hill_tail_is_independent_of_explore():
    # the short answer's S&P number is explore.summary's Hill estimate; check it by hand once
    r = ex.log_returns(ex.saved_prices("S&P 500")).values
    assert ex.summary(r)["tail"] == pytest.approx(_hill(r), rel=1e-9)


def test_guide_glossary_and_links(open_page):
    at = open_page("web/page_guide.py")
    text = _text(at)
    for term in ("Standard deviation (sd)", "Tail exponent", "Seed / held-out", "Median", "Log scale",
                 "Ticker", "Index", "Hurst H", "Excess kurtosis", "Memory of wild days"):
        assert f"**{term}:**" in text, term
    from web import common as c
    assert c.HELP["tail"] in text and c.HELP["memory"] in text       # one definition, shared with the tooltips
    assert "each step along an axis is 10 times the last" in text
    assert _links_ok(text) >= 4
    assert f"{REPO}/blob/main/settings/sp500_calibrated.json" in text
    assert any(REPO in cb.value for cb in at.code)                     # the clone command


def test_guide_intraday_sentence_is_split(open_page):
    at = open_page("web/page_guide.py")
    line = [x for x in _text(at).split("\n") if "1D and 7D show prices" in x][0]
    assert line.count(". ") <= 2 and len(line) < 260


# ---------------------------------------------------------------------------------- Story

def story_step(open_page, n):
    """The Story page at step n (1 to 8), reached by clicking Next like a visitor."""
    at = open_page("web/page_story.py")
    for _ in range(n - 1):
        at.button(key="story_next").click().run()
        assert not at.exception, [e.value for e in at.exception]
    return at


def test_story_defines_sd_and_names_groups(open_page):
    # the Story is now 8 steps (docs/redesign/WORDING.md): sd and the tail exponent are defined on step 1,
    # the three trader mixes are named on steps 4 to 6, and the other markets are offered on step 8
    text1 = _text(story_step(open_page, 1))
    assert "usual size of a day's move. A 4-sd day moves four times as much as usual" in text1
    assert "**tail exponent** (lower = more big days)" in text1 and "The S&P 500 is near 3" in text1
    for n, name in ((4, "**Value traders only**"), (5, "**Trend chasers, no herding**"), (6, "**Trend chasers who herd**")):
        assert name in _text(story_step(open_page, n))
    assert "slowly grow" not in _text(story_step(open_page, 5))
    assert "two other indexes (Nasdaq-100, Dow Jones)" in _text(story_step(open_page, 8))   # tests/test_web_pages.py relies on it


# ---------------------------------------------------------------------------------- Checks
def test_checks_rules_have_plain_openers_and_expanders(open_page):
    at = open_page("web/page_checks.py")
    text = _text(at)
    assert [e.label for e in at.expander] == ["How the rule is computed"] * 4 + ["How the rules and the formula work"]   # + crashes, checklist (Nov 2026)
    outside = "\n".join(m.value for m in at.markdown if not any(m.value == x.value for e in at.expander
                                                                   for x in e.markdown))
    assert "√(24/n)" not in outside and "total variation" not in outside.lower()
    assert "√(24/n)" in "\n".join(x.value for e in at.expander for x in e.markdown)
    assert "a number called a seed" in text and "never used to fit" in text
    assert "figures/RESULTS.md discusses" not in text and f"{REPO}/blob/main/figures/RESULTS.md" in text
    assert "epsilon" in at.select_slider[0].label.lower()
    assert _fig(at, "c1")["layout"]["xaxis"]["title"]["text"] == "History number (seed)"
    n = sum(t is not None for t in NUMBERS["checks"]["herding"]["tail"])
    assert "middle 90%" in text and f"over all {n} simulated histories" in text


def test_checks_table_has_column_help(open_page):
    at = open_page("web/page_checks.py")
    df = at.dataframe[0]
    cols = json.loads(df.proto.columns) if df.proto.columns else {}
    assert any("help" in v for v in cols.values()), cols


# ---------------------------------------------------------------------------------- Findings
def _paper_table(html):
    head = re.findall(r"<th(?:\s[^>]*)?>(.*?)</th>", html)
    rows = [re.findall(r"<td[^>]*>(.*?)</td>", r) for r in re.findall(r"<tr>(.*?)</tr>", html)]
    return head, [r for r in rows if r]


def test_findings_paper_table_columns_and_verdicts(open_page):
    at = open_page("web/page_checks.py")
    html = [m.value for m in at.markdown if "<table" in m.value][0]
    head, rows = _paper_table(html)
    assert head == ["Measure", "Lux & Marchesi", "Our rebuild", "Within our tolerance?",
                    "Inside the paper's error bar?"]
    published = {"alpha_reg": (2.64, 0.077, 0.15), "H": (0.48, 0.003, 0.03), "H_abs": (0.85, 0.010, 0.03)}
    assert len(rows) == 3
    for row, (key, (paper, err, tol)) in zip(rows, published.items()):
        v = np.array([x for x in NUMBERS["paper"][key] if x is not None])
        mean, sd = v.mean(), v.std(ddof=1)
        assert row[1] == f"{paper:.2f} ± {err:.3f}"
        assert row[2] == f"{mean:.2f} ± {sd:.3f}"                  # same decimals as the paper's column
        assert row[3] == f"{'yes' if abs(mean - paper) <= tol else 'no'} (± {tol:.2f})"
        assert row[4] == ("yes" if abs(mean - paper) <= err else "no")
    assert [r[4] for r in rows] == ["yes", "no", "no"]                 # RESULTS.md: only alpha is inside them


def test_findings_paper_section_words(open_page):
    at = open_page("web/page_checks.py")
    text = _text(at)
    from herding import stats
    r = ex.log_returns(ex.saved_prices("S&P 500")).values
    assert f"S&P 500's tail exponent is {stats.alpha_regression(r):.2f}" in text
    assert f"{stats.alpha_regression(r):.2f}" == "2.35"                 # RESULTS.md, Fig 7
    k1, k25 = np.median(NUMBERS["paper"]["kurtosis"]), np.median(NUMBERS["paper"]["kurt_25"])
    assert f"falls from {k1:.0f} for one day to {k25:.1f} for 25 days" in text
    for words in ("fit error of its single long run", f"how much {len(NUMBERS['paper']['seeds'])} simulated histories differ",
                  "time step of 0.01 day", "never used to fit the settings (they were used to choose between fits)",
                  "gives no settings (the follow-up paper with its tables was not yet obtained)",
                  "largest 30% of days", "Hill estimate over the largest 5% of days",
                  "loses its fat tails over a few weeks somewhat faster", "falls harder than it rises",
                  "reverse a little", "rarely run away", "0.5 means no memory", "0 means no memory",
                  "black = S&P 500", "Paper-matched", "S&P-calibrated", "Most figures average"):
        assert words in text, words
    assert "keeps fat tails over months" not in text
    assert f"{REPO}/blob/main/docs/KEY_POINTS.md#7-" in text
    _links_ok(text)


def _gallery_picker(at):
    """The Figures tab's section picker (by its key: the Checks page has other button groups)."""
    return [b for b in at.get("button_group") if "findings_section" in b.proto.id][0]


def test_findings_gallery_buttons_are_short(open_page):
    at = open_page("web/page_checks.py")
    labels = [o.content for o in _gallery_picker(at).proto.options]
    sections = list(dict.fromkeys(e["section"] for e in ex.gallery_entries((ROOT / "figures" / "GALLERY.md").read_text())))
    assert len(labels) == len(sections) and all(len(x) <= 16 for x in labels)
    assert [x.split(".")[0] for x in labels] == [s.split(".")[0] for s in sections]


# ---------------------------------------------------------------------------------- Compare
def test_compare_bars_are_linear_from_zero(open_page):
    at = open_page("web/page_checks.py")
    fig = _fig(at, "bars")
    ax = fig["layout"]["xaxis"]
    assert ax.get("type") == "linear" and ax["range"][0] == 0
    t = ex.market_table()
    assert ax["range"][1] >= t["times_bell"].max()
    assert any("bell curve = 1×" in a.get("text", "") for a in fig["layout"]["annotations"])
    assert "(log scale)" not in ax["title"]["text"]


def test_compare_map_zoom_symbols_and_labels(open_page):
    at = open_page("web/page_checks.py")
    fig = _fig(at, "map")
    assert fig["layout"]["hovermode"] == "closest"
    assert _trace(fig, "Stocks")["mode"] == "markers"
    assert _trace(fig, "Indexes")["mode"] == "markers+text"
    syms = {n: _trace(fig, n)["marker"]["symbol"] for n in ("Stocks", "Indexes", "Commodities")}
    assert len(set(syms.values())) == 3
    colors = {_trace(fig, n)["marker"]["color"] for n in ("Stocks", "Indexes", "Commodities")}
    assert len(colors) == 3
    fund = _trace(fig, "Model: value traders only")
    herd = _trace(fig, "Model: trend chasers who herd")
    assert fund["marker"]["symbol"] == "circle-open" and herd["marker"]["symbol"] == "star"
    lo, hi = fig["layout"]["xaxis"]["range"]
    med = np.median(NUMBERS["checks"]["fundamentalists"]["tail"])
    assert med > hi                                                     # off the zoomed map ...
    assert any(f"tail exponent {med:.2f}" in a.get("text", "") for a in fig["layout"]["annotations"])   # ... and labelled


def test_compare_tails_have_distinct_dashes(open_page):
    at = open_page("web/page_checks.py")
    fig = _fig(at, "tails")
    picked = [t for t in fig["data"] if t.get("name") in ("S&P 500", "Gold", "AAPL")]
    assert len({t["line"]["dash"] for t in picked}) == 3
    assert "colors only label" not in _text(at)


# ---------------------------------------------------------------------------------- GALLERY.md
def test_gallery_captions_reworded_and_parseable():
    text = (ROOT / "figures" / "GALLERY.md").read_text()
    for gone in ("future web app", "shows the herding code is right", "variance", "(made by"):
        assert gone not in text, gone
    assert "matches the math the model's herding rule is" in text
    g = ex.gallery_entries(text)
    assert {x["file"] for x in g} == {p.name for p in (ROOT / "figures").glob("fig*.png")}
    assert all(x["caption"] for x in g)
