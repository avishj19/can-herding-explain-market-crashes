"""Security checks of the public web app: limits that stop one visitor (or a script) from exhausting
the server or getting it blocked by Yahoo, text that must not turn into markup, and the settings
that keep error details and secrets away from visitors. No test uses the internet."""
import re
import subprocess
import tomllib

import pytest

from conftest import ROOT
from herding import explore as ex
from web import common as c
from test_web_explore_fixes import H
from test_web_pages import app, open_page  # noqa: F401  (pytest fixtures)


# ------------------------------------------------------------------ limits

# Spec: at most `limit` downloads in any `window` seconds, for the whole server.
def test_allow_fetch_counts_per_window(monkeypatch):
    monkeypatch.setattr(c, "_fetches", c.deque())
    assert [c.allow_fetch(now=100.0 + i, limit=3) for i in range(4)] == [True, True, True, False]
    assert c.allow_fetch(now=159.9, limit=3) is False         # the first is still inside the minute
    assert c.allow_fetch(now=160.0, limit=3) is True          # ...and now it isn't


# Spec: when the server is busy, a new ticker's history is not downloaded at all.
def test_busy_server_downloads_nothing(monkeypatch):
    calls = []
    monkeypatch.setattr(ex, "_yahoo", lambda t, s: calls.append(t))
    monkeypatch.setattr(c, "allow_fetch", lambda: False)
    c._live_history.clear()
    with pytest.raises(ConnectionError, match="many prices"):
        c._live_history("ZZQQ")
    assert calls == []


# Spec: every cache keyed by a typed ticker is bounded, so many tickers can't fill the memory.
@pytest.mark.parametrize("fn", ["_live_history", "_live_recent", "live_quote", "intraday", "_spliced"])
def test_ticker_caches_are_bounded(fn):
    src = (ROOT / "web" / "common.py").read_text()
    deco = re.search(r"@st\.cache_data\(([^)]*)\)\s*\ndef " + fn + r"\(", src)
    assert deco and "max_entries" in deco.group(1)


# Spec: one visitor may look up TYPED_PER_MINUTE tickers a minute; older lookups stop counting.
def test_typing_limit(app):
    at = app()
    import streamlit as st
    allowed = H["_typing_allowed"]
    # the helper reads st.session_state: give it this test app's state
    H["st"] = type("S", (), {"session_state": _State()})
    try:
        got = [allowed(now=1000.0 + i) for i in range(H["TYPED_PER_MINUTE"] + 1)]
        assert got == [True] * H["TYPED_PER_MINUTE"] + [False]
        assert allowed(now=1061.0) is True
    finally:
        H["st"] = st
    assert at is not None


class _State(dict):
    __getattr__ = dict.get

    def __setattr__(self, k, v):
        self[k] = v


# ------------------------------------------------------------------ text that must stay text

@pytest.mark.parametrize("text,want", [
    ("![x](https://e/x.png)", "\\!\\[x\\](https://e/x.png)"),
    ("<img src=x onerror=alert(1)>", "\\<img src=x onerror=alert(1)\\>"),
    ("*bold* _it_ `code` ~s~ \\", "\\*bold\\* \\_it\\_ \\`code\\` \\~s\\~ \\\\"),
    ("Use only . - = ^ (up to 12), like BRK-B.", "Use only . - = ^ (up to 12), like BRK-B."),
])
def test_plain_escapes_only_markup(text, want):
    assert H["_plain"](text) == want


# Spec: what a visitor types can only reach Yahoo, or the page, as letters, digits and . - = ^.
@pytest.mark.parametrize("bad", ["<script>", "\"><b>", "a b", "../etc", "NVDA\n", "x" * 13, "ＮＶＤＡ", "$$", ""])
def test_clean_ticker_turns_away_anything_else(bad):
    with pytest.raises(ValueError):
        ex.clean_ticker(bad if bad != "NVDA\n" else "NV\nDA")


# ------------------------------------------------------------------ settings and secrets

def test_streamlit_config_hides_error_details_and_keeps_xsrf():
    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())
    assert cfg["client"]["showErrorDetails"] in ("type", "none")
    assert cfg["server"].get("enableXsrfProtection", True) is True


def test_no_secrets_are_tracked():
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    assert ".streamlit/secrets.toml" not in files
    assert not any(f.endswith((".env", ".pem", ".key")) for f in files)
    pattern = r"ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|sk-ant-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|" \
              r"-----BEGIN [A-Z ]*PRIVATE KEY"
    code = [f for f in files if f.endswith((".py", ".toml", ".json", ".txt", ".md", ".yml", ".yaml", ".cfg"))
            and not f.startswith("journal/")]                # journal/: checked by its own hook
    hits = [f for f in code if re.search(pattern, (ROOT / f).read_text(errors="ignore"))]
    assert hits == []


def test_secret_files_are_git_ignored():
    for f in (".streamlit/secrets.toml", ".env"):
        assert subprocess.run(["git", "check-ignore", "-q", f], cwd=ROOT).returncode == 0, f
