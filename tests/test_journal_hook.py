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


# ---------------------------------------------------------------- private documents
# Copyrighted papers and the course handout stay off GitHub, but a session that reads one
# puts its text in the journal. The hook keeps only fingerprints (hashes of sampled 8-word
# phrases, which cannot be turned back into text) and removes any passage that matches.

PRIVATE = " ".join(f"sentence {i} of the private document talks about herding traders and "
                   f"crowd number {i * 7} in a market that nobody outside should read" for i in range(60))


def test_a_copied_passage_of_a_private_document_is_removed():
    prints = hook.fingerprints(PRIVATE)
    line = json.dumps({"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "x", "content": "Page 1\n" + PRIVATE[:3000]},
        {"type": "text", "text": "an ordinary remark"}]}})
    out = hook.remove_private_text(line, prints)
    rec = json.loads(out)                                   # still valid JSON
    assert "nobody outside should read" not in out
    assert rec["message"]["content"][1]["text"] == "an ordinary remark"
    assert "removed" in rec["message"]["content"][0]["content"]


def test_a_short_quote_and_unrelated_text_are_kept():
    prints = hook.fingerprints(PRIVATE)
    quote = " ".join(PRIVATE.split()[:12])                  # a 12-word quote
    for text in (quote, "the model's tails are fatter than a bell curve " * 40):
        line = json.dumps({"message": {"content": text}})
        assert hook.remove_private_text(line, prints) == line


def test_a_cut_off_line_with_private_text_is_replaced_too():
    prints = hook.fingerprints(PRIVATE)
    out = hook.remove_private_text('{"message": {"content": "' + PRIVATE[:2000], prints)
    assert "nobody outside should read" not in out
    json.loads(out)


def test_fingerprints_hold_no_readable_text():
    prints = hook.fingerprints(PRIVATE)
    assert prints and all(len(p) == 12 and all(c in "0123456789abcdef" for c in p) for p in prints)
