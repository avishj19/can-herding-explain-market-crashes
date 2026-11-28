"""Tests of the site's shared look (web/common.py, .streamlit/config.toml, app.py): readable color
pairs in both modes, hover boxes that read the right line, the links under the tape, and the help
under a row of numbers.

    python3 -m pytest tests/test_web_style.py -q

Contrast is WCAG 2's ratio (L1 + 0.05) / (L2 + 0.05) of the two colors' relative luminances L:
4.5 is the minimum for text, 3 for lines, borders and other graphics.
"""
import re
import tomllib
from pathlib import Path

import plotly.graph_objects as go
import pytest

from conftest import ROOT
from web import common as c


def _lin(v):
    v /= 255
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4


def luminance(hex_color):
    h = hex_color.lstrip("#")
    r, g, b = (_lin(int(h[i:i + 2], 16)) for i in (0, 2, 4))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    hi, lo = sorted([luminance(a), luminance(b)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)


# ------------------------------------------------------------------ the contrast formula itself
def test_contrast_known_answers():
    # black on white is (1 + 0.05) / (0 + 0.05) = 21 exactly; a color on itself is 1
    assert contrast("#000000", "#FFFFFF") == pytest.approx(21.0)
    assert contrast("#FFFFFF", "#000000") == pytest.approx(21.0)
    assert contrast("#A35100", "#A35100") == pytest.approx(1.0)
    # #777777: channel 119/255 -> ((0.4667 + 0.055) / 1.055) ** 2.4, worked out exactly here
    lum = ((119 / 255 + 0.055) / 1.055) ** 2.4
    assert contrast("#777777", "#FFFFFF") == pytest.approx(1.05 / (lum + 0.05))
    # the reviewers' measured failures, before this fix
    assert round(contrast("#141414", "#5F5B52"), 2) == 2.72
    assert round(contrast("#141414", "#A35100"), 2) == 3.28


# ------------------------------------------------------------------ readable pairs in both modes
TEXT_PAIRS = [("CHIP_TEXT", "MUTED"), ("CHIP_TEXT", "ACCENT"),            # nav number chips
              ("BUTTON_TEXT", "BUTTON"),                                   # primary buttons, the CTA
              ("ACCENT", "BG"), ("ACCENT", "PANEL"),                       # headings, labels, tape
              ("MUTED", "BG"), ("MUTED", "PANEL"), ("TEXT2", "BG"), ("TEXT2", "PANEL"),
              ("REAL", "PANEL"), ("UP", "BG"), ("DOWN", "BG"), ("UP", "PANEL"), ("DOWN", "PANEL"),
              ("UP", "CARD"), ("DOWN", "CARD")]
GRAPHIC_PAIRS = [("GUIDE", "CARD"), ("SHADE_LOW", "CARD"), ("BELL", "CARD"), ("ACCENT", "BG")]


@pytest.mark.parametrize("mode", ["dark"])
@pytest.mark.parametrize("fg,bg", TEXT_PAIRS)
def test_text_colors_are_readable(mode, fg, bg):
    p = c.PALETTES[mode]
    assert contrast(p[fg], p[bg]) >= 4.5, (mode, fg, p[fg], bg, p[bg], contrast(p[fg], p[bg]))


@pytest.mark.parametrize("mode", ["dark"])
def test_chart_lines_stand_out_from_the_chart(mode):
    p = c.PALETTES[mode]
    lines = [(k, p[k]) for k, _ in GRAPHIC_PAIRS] + list(p["MIXC"].items()) + list(enumerate(p["PICK"]))
    weak = [(k, v, round(contrast(v, p["CARD"]), 2)) for k, v in lines if contrast(v, p["CARD"]) < 3]
    assert not weak, weak


def test_palette_values_from_the_review():
    d = c.PALETTES["dark"]
    assert list(c.PALETTES) == ["dark"] and c.pal() is d and not hasattr(c, "mode")   # light mode removed (Nov 2026)
    assert (d["BELL"], d["GUIDE"], d["GRID"], d["LINE"], d["MUTED"], d["SHADE_LOW"]) == (
        "#737A87", "#8A8D99", "#2E2E2E", "#4A4A4A", "#A29F98", "#7A8496")     # audit colors (Nov 2026)
    assert d["MIXC"] == dict(fundamentalists="#3D9BE0", chartists="#E66A1F", herding="#F2A7DA")
    assert (d["HERD_FILL"], d["DOWN_FILL"]) == ("rgba(242,167,218,0.10)", "rgba(255,122,107,0.10)")
    assert d["CHIP_TEXT"] == "#000000" and d["PICK"][4] == "#C084FC"
    assert c.PICK_DASH == ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
    assert len(c.PICK_DASH) == len(d["PICK"])


@pytest.mark.parametrize("mode", ["dark"])
def test_fills_match_their_line_color(mode):
    # UP_FILL / DOWN_FILL are the UP / DOWN color, see-through
    p = c.PALETTES[mode]
    for line, fill in (("UP", "UP_FILL"), ("DOWN", "DOWN_FILL"), ("UP", "OK_FILL")):
        rgb = tuple(int(p[line][i:i + 2], 16) for i in (1, 3, 5))
        assert tuple(int(v) for v in re.match(r"rgba\((\d+),(\d+),(\d+),", p[fill]).groups()) == rgb


def test_config_colors():
    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())["theme"]
    assert cfg["base"] == "dark" and "light" not in cfg and "dark" not in cfg   # one color set
    assert cfg["primaryColor"] == c.PALETTES["dark"]["ACCENT"]
    assert contrast(cfg["borderColor"], cfg["backgroundColor"]) >= 3     # input borders: 3:1 on the page
    assert contrast(cfg["borderColor"], cfg["secondaryBackgroundColor"]) >= 3   # and on the panels
    assert cfg["dataframeBorderColor"] != cfg["borderColor"]              # table lines stay faint


# ------------------------------------------------------------------ the style sheet
@pytest.fixture(params=["dark"])
def sheet(request, monkeypatch):
    return request.param, c.PALETTES[request.param], c.css(["/", "/page_story", "/page_explore"])


def test_chips_and_go_use_chip_text(sheet):
    _, p, s = sheet
    chip = re.search(r'\[data-testid="stTopNavLink"\]::before, \[data-testid="stTopNavDropdownLink"\]::before'
                     r' \{([^}]*)\}', s).group(1)
    assert f"color: {p['CHIP_TEXT']}" in chip and f"background: {p['MUTED']}" in chip
    assert ".tape .brand b" not in s and "&lt;GO&gt;" not in s       # the <GO> chip is gone (redesign)
    tape = re.search(r"\.tape \{([^}]*)\}", s).group(1)
    assert f"border-bottom: 1px solid {p['LINE']}" in tape and "background" not in tape   # slim strip, no box


def test_nav_numbers_follow_the_page_addresses():
    s = c.nav_numbers(["/", "/page_story", "/page_explore"])
    for i, path in enumerate(["/", "/page_story", "/page_explore"], start=1):
        for tid in ("stTopNavLink", "stTopNavDropdownLink"):     # the bar and the "N more" list
            assert f'[data-testid="{tid}"][href$="{path}"]::before' in s
        assert f'content: "{i}"; content: "{i}" / ""' in s
    assert "counter" not in s
    assert "counter(fkey)" in c.nav_numbers()                    # no addresses: count the links


def test_sheet_rules(sheet):
    _, p, s = sheet
    assert "steps(1) 4}" in s and "infinite" not in s                    # the cursor blinks 4 times
    assert "prefers-reduced-motion: reduce" in s
    assert f':is(button, a, input, [role="tab"], [role="option"], [tabindex]):focus-visible {{outline: 2px solid {p["ACCENT"]} !important' in s
    assert "cursor: help" not in s
    assert 'content: "■  " / ""' in s                                    # decorative: silent
    label = re.search(r'\[data-testid="stWidgetLabel"\] p \{([^}]*)\}', s).group(1)
    assert "text-transform" not in label                                  # ε stays ε
    assert f"border: 1px solid {p['ACCENT']} !important" in s            # primary buttons
    assert "white-space: normal" in re.search(r"\.st-key-cta a \{([^}]*)\}", s).group(1)
    assert "max-width: 100%" in re.search(r"\.st-key-cta a \{([^}]*)\}", s).group(1)


def _rem(rule, s):
    return float(re.search(re.escape(rule) + r" \{[^}]*?font-size: ([.\d]+)rem", s).group(1))


def test_small_text_is_at_least_078rem(sheet):
    _, _, s = sheet
    for rule in (".stats .l", ".tape .asof", ".hint::before",
                 '[data-testid="stTopNavLink"]::before, [data-testid="stTopNavDropdownLink"]::before'):
        assert _rem(rule, s) >= 0.78, rule
    assert _rem("h3", s) >= 0.92
    assert "mode-toggle" not in s                                    # no light/dark button any more


def test_up_and_down_colors_only_come_with_an_arrow(sheet):
    _, p, s = sheet
    assert p["UP"] not in s and p["DOWN"] not in s                        # the sheet never uses them
    for x in (0.0123, -0.0123):
        html, color = c.arrow(x)
        assert ("▲" if x >= 0 else "▼") in html and color == (p["UP"] if x >= 0 else p["DOWN"])


def test_css_test_ids_are_complete():
    # every data-testid the sheet styles is written out in full (test_web_pages checks they exist)
    ids = set(re.findall(r'data-testid="(\w+)"', c.css(["/"])))
    assert {"stTopNavLink", "stTopNavDropdownLink", "stWidgetLabel"} <= ids


# ------------------------------------------------------------------ hover boxes
def _tail_fig():
    r1 = [0.01 * ((-1) ** i) * (1 + (i % 7)) for i in range(800)]
    r2 = [0.02 * ((-1) ** i) * (1 + (i % 5)) for i in range(800)]
    return c.tail_chart([("A", r1, dict(color="#000000")), ("B", r2, dict(color="#FFFFFF"))])


def test_tail_chart_hover_reads_one_line():
    fig = _tail_fig()
    assert fig.layout.hovermode == "closest"
    assert fig.layout.xaxis.showspikes is False and fig.layout.yaxis.showspikes is False
    assert fig.layout.hoverlabel.namelength == -1
    skip = {t.name: t.hoverinfo for t in fig.data}
    assert skip["Bell curve"] == "skip" and skip["Slope 3"] == "skip"
    assert skip["A"] != "skip" and "A<br>" in fig.data[0].hovertemplate


def test_dial_chart_hover_reads_one_point():
    fig = c.dial_chart(3.0, runs=[(1.0, 3.2)], current=(1.0, 3.2))
    assert fig.layout.hovermode == "closest" and fig.layout.xaxis.showspikes is False


def test_style_keeps_x_unified_for_time_series():
    fig = c.style(go.Figure(go.Scatter(x=["2026-01-02", "2026-01-05"], y=[1, 2])))
    assert fig.layout.hovermode == "x unified" and fig.layout.xaxis.showspikes is True
    assert fig.layout.hoverlabel.namelength == -1
    assert fig.layout.xaxis.tickfont.size == fig.layout.legend.font.size == fig.layout.yaxis.title.font.size == 12


def test_log_ticks_are_plain_powers_of_ten():
    fig = c.style(go.Figure(go.Scatter(x=[1, 2], y=[1, 20000])), logy=True, plain_log=True)
    y = fig.layout.yaxis
    assert y.type == "log" and y.dtick == 1 and y.exponentformat == "none" and y.tickformat == ",~g"
    assert fig.layout.xaxis.dtick is None                                 # only the axis asked for
    fig2 = c.log_ticks(go.Figure(), x=True, y=False)
    assert fig2.layout.xaxis.dtick == 1 and fig2.layout.yaxis.dtick is None


# ------------------------------------------------------------------ help under the numbers
def test_stats_help_is_readable_without_hovering():
    html = c.stats_html([("Tail", "3.10", None, "How fast big days get rarer."), ("Days", "500", None, None)])
    assert "title='How fast big days get rarer.'" in html                 # hover still works
    details = re.search(r"<details class='muted stats-help'><summary>What these numbers mean</summary>(.*)</details>",
                        html).group(1)
    assert details == "<p><b>Tail</b>: How fast big days get rarer.</p>"  # one line per item with help
    assert "<details" not in c.stats_html([("Days", "500", None, None)])  # no help, no note


# ------------------------------------------------------------------ links under the tape
@pytest.mark.parametrize("title,want", [
    ("Start here", [("cta", "web/page_story.py", "Start the story")]),
    ("Story", [("cta", "web/page_explore.py", "Open the test program"),
               ("words", "web/page_guide.py", "Words used on this site")]),
    (c.TEST_PROGRAM, [("words", "web/page_guide.py", "Words used on this site")]),
    ("Checks", [("cta", "web/page_explore.py", "Open the test program"),
                  ("words", "web/page_guide.py", "Words used on this site")]),
])
def test_cta_links_per_page(title, want):
    assert [(k, page, label) for k, page, label, _ in c.cta_links(title)] == want


def test_app_page_titles_match_common():
    src = (ROOT / "app.py").read_text()
    assert f'TEST_PROGRAM = "{c.TEST_PROGRAM}"' in src
    assert 'title="Start here"' in src
    # the navigation goes out before the site's modules are (re)loaded
    assert src.index("st.navigation(") < src.index("reload_if_changed()\nfrom web import common")


@pytest.mark.slow
@pytest.mark.parametrize("page,title", [(None, "Start here"), ("web/page_story.py", "Story"),
                                        ("web/page_explore.py", c.TEST_PROGRAM)])
def test_cta_in_the_running_app(page, title, monkeypatch):
    # the links the running app draws under the tape are the ones cta_links gives for that page
    import socket
    from streamlit.testing.v1 import AppTest
    from herding import explore as ex
    monkeypatch.setattr(socket.socket, "connect", lambda *a, **k: (_ for _ in ()).throw(OSError("blocked")))
    monkeypatch.setattr(ex, "_yahoo", lambda t, s: (_ for _ in ()).throw(OSError("offline")))
    monkeypatch.setattr(ex, "_yahoo_intraday", lambda t, p, i: (_ for _ in ()).throw(OSError("offline")))
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
    at.run()
    if page:
        at.switch_page(page).run()
    assert not at.exception, [e.value for e in at.exception]
    want = c.cta_links(title)
    drawn = [(e.proto.label, e.proto.page) for e in at.get("page_link")][:len(want)]
    # a page link stores the page's address: "page_story" for web/page_story.py, "" for the start page
    addr = {"web/page_guide.py": ""}
    assert drawn == [(label, addr.get(f, Path(f).stem)) for _, f, label, _ in want], drawn


# ------------------------------------------------------------------ caches follow the files they read
# Spec: a cached value that comes from a file is read again when that file changes (its time is part
# of the cache key), and the site's reload list covers every herding module the pages load.

def test_file_stamp_and_data_stamp(tmp_path):
    import os
    from web import common as c
    a, missing = tmp_path / "a.json", tmp_path / "nope.json"
    a.write_text("{}")
    os.utime(a, (1_000_000.0, 1_000_000.0))
    assert c.file_stamp(a, missing) == (1_000_000.0, 0.0)
    os.utime(a, (1_000_060.0, 1_000_060.0))
    assert c.file_stamp(a) == (1_000_060.0,)
    newest = max(p.stat().st_mtime for p in c.DATA.rglob("*") if p.is_file())
    assert c.data_stamp() == newest


@pytest.mark.parametrize("public, private, path_attr", [
    ("fitted", "_fitted", "FIT"),
])
def test_fit_is_read_again_when_the_settings_file_changes(monkeypatch, public, private, path_attr):
    import os
    import streamlit as st
    from herding import explore as ex
    from web import common as c
    st.cache_data.clear()
    calls = []
    monkeypatch.setattr(ex, "fitted_settings", lambda: calls.append(1) or ({"n": len(calls)}, 1000))
    path = getattr(ex, path_attr)
    before = os.stat(path)
    try:
        assert getattr(c, public)()[0] == {"n": 1} and getattr(c, public)()[0] == {"n": 1}
        os.utime(path, (before.st_atime, before.st_mtime + 60))       # what a refit + push does
        assert getattr(c, public)()[0] == {"n": 2}
    finally:
        os.utime(path, (before.st_atime, before.st_mtime))
        st.cache_data.clear()


def test_model_runs_are_not_reused_after_a_refit(monkeypatch):
    # run_mixes was cached on (a1, a2, v1, days, seed) only: after a refit it kept serving runs made
    # with the old settings while numbers.json showed the new ones
    import os
    import streamlit as st
    from herding import explore as ex
    from web import common as c
    st.cache_data.clear()
    calls = []
    monkeypatch.setattr(ex, "fitted_settings", lambda: ({"fit": 1}, 1000))
    monkeypatch.setattr(ex, "run_mixes", lambda p, burn, *a, **k: calls.append(a) or {"call": len(calls)})
    before = os.stat(ex.FIT)
    try:
        assert c.run_mixes(1.0, 0.1, 2.0, 5000, 1) == {"call": 1}
        assert c.run_mixes(1.0, 0.1, 2.0, 5000, 1) == {"call": 1}          # same settings file: cached
        os.utime(ex.FIT, (before.st_atime, before.st_mtime + 60))
        assert c.run_mixes(1.0, 0.1, 2.0, 5000, 1) == {"call": 2}          # refit: run again
    finally:
        os.utime(ex.FIT, (before.st_atime, before.st_mtime))
        st.cache_data.clear()


def test_reload_list_covers_every_herding_module_the_site_loads():
    # herding.calibrate (used by the Findings paper table) was missing, so after a push changing it
    # the live site would have kept the old copy (the crash of Nov 2026 came from the same gap).
    # Run in a fresh Python: in the test session other tests have imported more herding modules.
    import json
    import re
    import subprocess
    import sys
    script = (
        "import json, sys\n"
        "from streamlit.testing.v1 import AppTest\n"
        f"at = AppTest.from_file({str(ROOT / 'app.py')!r}, default_timeout=300)\n"
        "at.run()\n"
        "for p in ['web/page_story.py', 'web/page_explore.py', 'web/page_checks.py', 'web/page_sources.py']:\n"
        "    at.switch_page(p).run()\n"
        "    assert not at.exception, (p, [e.value for e in at.exception])\n"
        "print(json.dumps(sorted(m for m in sys.modules if m.startswith('herding.'))))\n")
    out = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    loaded = json.loads(out.stdout.strip().splitlines()[-1])
    listed = eval(re.search(r"SITE_MODULES = (\[.*?\])", (ROOT / "app.py").read_text(), re.S).group(1))
    assert "herding.explore" in loaded and [m for m in loaded if m not in listed] == []
