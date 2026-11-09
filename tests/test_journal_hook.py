"""Tests for .claude/hooks/save_journal.py: the journal is committed to a public repo,
so it must never carry secrets or personal info such as an email address."""
import importlib.util
import json
import os

import pytest

from conftest import ROOT

spec = importlib.util.spec_from_file_location("save_journal", ROOT / ".claude" / "hooks" / "save_journal.py")
hook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hook)


@pytest.mark.parametrize("email", ["student@example.edu", "first.last+tag@example.co.uk",
                                   "noreply@anthropic.com", "A_B-9@sub.domain.org"])
def test_scrub_hides_email_addresses(email):
    out = hook.scrub(f'{{"userEmail":"The user\'s email address is {email}. Use it"}}')
    assert email not in out and "[REDACTED]" in out
    json.loads(out)                                   # the log line is still valid JSON


def test_scrub_hides_an_email_right_after_an_escaped_newline():
    # in the saved JSON a newline is the two characters backslash + n
    out = hook.scrub(r'"line one\nstudent@example.edu and\tx.y@z.org"')
    assert "andrew.cmu.edu" not in out and "z.org" not in out


# Found in the real journal before the fix went live: in JSON, "\n@pytest.mark..." and diff
# lines "+@pytest.mark.slow" looked like addresses (n@pytest.mark.parametrize, ...).
@pytest.mark.parametrize("text", ["@pytest.mark.parametrize", "react@18.3.1", "numba>=0.60",
                                  "ratio 2.7x @ day 250", "user@localhost",
                                  r'"\n@pytest.mark.parametrize("seed", SEEDS)\n"', r"\t@pytest.mark.slow",
                                  "+@pytest.mark.slow", "-@pytest.fixture", r"\n219-@pytest.mark.parametrize",
                                  r"\n@numba.njit(cache=True)"])
def test_scrub_keeps_text_that_is_not_an_email(text):
    assert hook.scrub(text) == text


def test_scrub_still_hides_keys():
    assert "ghp_" not in hook.scrub("token ghp_" + "a" * 30)


def test_existing_journal_files_are_cleaned_even_without_their_source(tmp_path):
    # old sessions whose original log is gone must be cleaned too
    old = tmp_path / "old.jsonl"
    old.write_text('{"x": "mail me at someone@example.com"}\n')
    os.utime(old, (1_000_000, 1_000_000))
    assert hook.scrub_existing(tmp_path) == 1
    assert "someone@example.com" not in old.read_text()
    assert old.stat().st_mtime == 1_000_000           # time kept, so copies stay in step
    assert hook.scrub_existing(tmp_path) == 0         # nothing left to clean
