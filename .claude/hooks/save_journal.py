"""Copy this project's AI coding session logs into journal/.

02-120 build journal script. It copies every Claude Code and Codex session log
that belongs to this folder, including sessions from before you set this up,
and writes a readable version of each one to journal/readable/.

Claude Code runs it at the start and end of every session, and git runs it
before every commit. You can also run it by hand: python save_journal.py
"""

import hashlib
import json
import os
import re
import sys
import unicodedata
import zlib
from pathlib import Path


def project_dir() -> Path:
    env_dir = os.environ.get("CLAUDE_PROJECT_DIR")
    if env_dir:
        return Path(env_dir)
    return Path(__file__).resolve().parent.parent.parent


def read_hook_input() -> dict:
    if sys.stdin is None or sys.stdin.isatty():
        return {}
    try:
        return json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        return {}


def claude_log_folder(hook_input: dict, project: Path) -> Path:
    transcript_path = hook_input.get("transcript_path")
    if transcript_path:
        return Path(transcript_path).expanduser().parent
    slug = re.sub(r"[^A-Za-z0-9]", "-", str(project.resolve()))
    return Path.home() / ".claude" / "projects" / slug


SECRET_PATTERN = re.compile(
    r"(sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
    r"|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{35}|xox[abprs]-[A-Za-z0-9-]{10,}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    # email addresses (personal info; Claude Code puts the user's email in each session's
    # context). The name before @ starts and ends with a letter or digit, and starts
    # either after an escaped newline/tab (\n, \t, \r in the saved JSON) or after a
    # character that cannot be part of it. So "\n@pytest.mark.slow" (JSON for a newline
    # then a decorator), "+@pytest.mark.slow" (a diff line) and "react@18.3.1" are left alone.
    r"|(?:(?<=\\[ntr])|(?<![\\A-Za-z0-9._%+-]))"
    r"[A-Za-z0-9](?:[A-Za-z0-9._%+-]*[A-Za-z0-9])?@(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,})",
    re.DOTALL,
)


def scrub(text: str) -> str:
    return SECRET_PATTERN.sub("[REDACTED]", text)


# Private documents (copyrighted papers, the course handout) are git-ignored, but a session
# that reads one puts its text in the journal. private_fingerprints.txt holds only hashes of
# every ~8th 8-word phrase of each document (make them with make_private_fingerprints.py);
# a hash cannot be turned back into words. Any text in a log line with MIN_HITS or more
# matching phrases is a copied passage and is replaced; a short quote has too few to match.
FINGERPRINTS = Path(__file__).with_name("private_fingerprints.txt")
SHINGLE, SAMPLE, MIN_HITS = 8, 8, 3
REMOVED = "[text of a private document removed: copyrighted paper or course material (save_journal.py)]"


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", text).lower())


def fingerprints(text: str) -> set[str]:
    """Hashes (12 hex digits) of about one in SAMPLE of the 8-word phrases of text."""
    words, out = _words(text), set()
    for i in range(len(words) - SHINGLE + 1):
        phrase = " ".join(words[i:i + SHINGLE]).encode()
        if zlib.crc32(phrase) % SAMPLE == 0:
            out.add(hashlib.blake2b(phrase, digest_size=6).hexdigest())
    return out


def load_fingerprints() -> set[str]:
    if not FINGERPRINTS.exists():
        return set()
    return set(FINGERPRINTS.read_text().split())


def remove_private_text(line: str, prints: set[str]) -> str:
    """One log line with every string that copies a private document replaced by REMOVED
    (a line that is not valid JSON, e.g. cut off mid-write, is replaced whole)."""
    if not prints or len(line) < 200:
        return line
    is_private = lambda s: len(s) >= 200 and len(fingerprints(s) & prints) >= MIN_HITS
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return json.dumps({"type": "removed", "note": REMOVED}) if is_private(line) else line
    changed = False

    def walk(x):
        nonlocal changed
        if isinstance(x, str):
            if is_private(x):
                changed = True
                return REMOVED
            return x
        if isinstance(x, list):
            return [walk(v) for v in x]
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        return x

    cleaned = walk(record)
    return json.dumps(cleaned, ensure_ascii=False) if changed else line


def scrub_log(text: str, prints: set[str]) -> str:
    """A whole log: private passages removed line by line, then secrets and emails."""
    if prints:
        text = "\n".join(remove_private_text(line, prints) for line in text.split("\n"))
    return scrub(text)


def copy_if_changed(source: Path, destination: Path) -> int:
    source_time = source.stat().st_mtime
    if destination.exists() and destination.stat().st_mtime == source_time:
        existing = destination.read_text(encoding="utf-8", errors="replace")
        if not SECRET_PATTERN.search(existing):
            return 0
    text = source.read_text(encoding="utf-8", errors="replace")
    destination.write_text(scrub_log(text, load_fingerprints()), encoding="utf-8")
    os.utime(destination, (source_time, source_time))
    return 1


def scrub_existing(journal: Path, private: bool = False) -> int:
    """Clean journal files already saved, including old sessions whose original log is
    gone (copy_if_changed only revisits files that still have a source). Keeps each
    file's time, so copy_if_changed still sees it as up to date. private=True also
    removes private-document text from every file (slow: run by hand with --private,
    once after adding fingerprints; new logs are checked as they are copied)."""
    cleaned = 0
    prints = load_fingerprints() if private else set()
    for path in sorted(journal.glob("*.jsonl")) + sorted(journal.glob("codex/*.jsonl")):
        text = path.read_text(encoding="utf-8", errors="replace")
        new = scrub_log(text, prints) if prints else scrub(text) if SECRET_PATTERN.search(text) else text
        if new != text:
            stat = path.stat()
            path.write_text(new, encoding="utf-8")
            os.utime(path, (stat.st_atime, stat.st_mtime))
            cleaned += 1
    return cleaned


def copy_claude_logs(source: Path, journal: Path) -> int:
    if not source.is_dir():
        print(f"save_journal: no Claude Code logs found at {source}", file=sys.stderr)
        return 0
    copied_count = 0
    for log_file in sorted(source.glob("*.jsonl")):
        copied_count += copy_if_changed(log_file, journal / log_file.name)
    return copied_count


def codex_session_folder(log_file: Path) -> str:
    try:
        with open(log_file, encoding="utf-8") as handle:
            first = json.loads(handle.readline() or "{}")
    except (OSError, json.JSONDecodeError):
        return ""
    return first.get("payload", {}).get("cwd", "")


def is_inside(folder: str, project: Path) -> bool:
    if not folder:
        return False
    try:
        Path(folder).resolve().relative_to(project.resolve())
    except ValueError:
        return False
    return True


def copy_codex_logs(project: Path, journal: Path) -> int:
    codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    log_files = sorted((codex_home / "sessions").glob("**/rollout-*.jsonl"))
    log_files += sorted((codex_home / "archived_sessions").glob("**/rollout-*.jsonl"))
    copied_count = 0
    for log_file in log_files:
        if is_inside(codex_session_folder(log_file), project):
            codex_journal = journal / "codex"
            codex_journal.mkdir(exist_ok=True)
            copied_count += copy_if_changed(log_file, codex_journal / log_file.name)
    return copied_count


def short(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def tool_note(agent: str, name: str, details: dict) -> str:
    target = details.get("file_path") or details.get("command") or details.get("cmd") or ""
    if isinstance(target, list):
        target = " ".join(target)
    if target:
        return f"*{agent} used {name}: `{short(scrub(str(target)), 100)}`*"
    return f"*{agent} used {name}*"


def claude_entry(record: dict) -> list[str]:
    lines = []
    if record.get("isMeta") or record.get("isSidechain"):
        return lines
    role = record.get("type")
    content = record.get("message", {}).get("content")
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    if role not in ("user", "assistant") or not isinstance(content, list):
        return lines
    for block in content:
        kind = block.get("type")
        text = block.get("text", "").strip()
        if kind == "text" and text and not text.startswith("<"):
            speaker = "Student" if role == "user" else "Claude"
            lines.append(f"**{speaker}:** {text}\n")
        elif kind == "image":
            lines.append("**Student:** [image]\n")
        elif kind == "tool_use":
            lines.append(tool_note("Claude", block.get("name", "a tool"), block.get("input", {})) + "\n")
    return lines


def codex_entry(record: dict) -> list[str]:
    lines = []
    payload = record.get("payload", {})
    if record.get("type") != "response_item":
        return lines
    kind = payload.get("type")
    role = payload.get("role")
    if kind == "message" and role in ("user", "assistant"):
        for block in payload.get("content", []):
            text = block.get("text", "").strip()
            if text and not text.startswith("<") and not text.startswith("# AGENTS.md"):
                speaker = "Student" if role == "user" else "Codex"
                lines.append(f"**{speaker}:** {text}\n")
    elif kind in ("function_call", "custom_tool_call", "local_shell_call"):
        try:
            details = json.loads(payload.get("arguments") or "{}")
        except json.JSONDecodeError:
            details = {}
        if not isinstance(details, dict):
            details = {}
        lines.append(tool_note("Codex", payload.get("name", "a tool"), details) + "\n")
    return lines


def collapse_repeats(lines: list[str]) -> list[str]:
    collapsed = []
    repeat_count = 1
    for index, line in enumerate(lines):
        is_last = index == len(lines) - 1
        if not is_last and line.startswith("*") and line == lines[index + 1]:
            repeat_count += 1
        elif repeat_count > 1:
            collapsed.append(line.rstrip("\n") + f" (x{repeat_count})\n")
            repeat_count = 1
        else:
            collapsed.append(line)
    return collapsed


def write_readable(log_file: Path, agent: str, readable: Path) -> None:
    lines = [f"# {agent} session {log_file.stem}\n"]
    with open(log_file, encoding="utf-8") as handle:
        for raw_line in handle:
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                record = {}
            if agent == "Codex":
                lines += codex_entry(record)
            else:
                lines += claude_entry(record)
    if len(lines) > 1:
        text = scrub("\n".join(collapse_repeats(lines)))
        (readable / f"{log_file.stem}.md").write_text(text, encoding="utf-8")


def write_all_readable(journal: Path) -> None:
    readable = journal / "readable"
    readable.mkdir(exist_ok=True)
    for log_file in sorted(journal.glob("*.jsonl")):
        write_readable(log_file, "Claude Code", readable)
    for log_file in sorted(journal.glob("codex/*.jsonl")):
        write_readable(log_file, "Codex", readable)


def main() -> None:
    project = project_dir()
    journal = project / "journal"
    journal.mkdir(exist_ok=True)
    copied_count = copy_claude_logs(claude_log_folder(read_hook_input(), project), journal)
    copied_count += copy_codex_logs(project, journal)
    scrub_existing(journal, private="--private" in sys.argv)
    write_all_readable(journal)
    print(f"save_journal: {copied_count} session log(s) updated in {journal}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # a journal problem must never block a session or a commit
        print(f"save_journal: {error}", file=sys.stderr)
