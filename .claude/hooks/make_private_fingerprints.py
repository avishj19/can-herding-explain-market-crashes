"""Make private_fingerprints.txt for save_journal.py from the private documents on this computer.

The documents themselves are git-ignored (copyrighted papers, the course handout). The
fingerprint file holds only hashes of sampled 8-word phrases, which cannot be turned back
into text, so it is safe to commit. Run after adding a private document:
    python3 .claude/hooks/make_private_fingerprints.py
    python3 .claude/hooks/save_journal.py --private      # clean the journal once
Needs pdftotext (poppler: brew install poppler).
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from save_journal import FINGERPRINTS, fingerprints  # noqa: E402

PROJECT = HERE.parent.parent
PRIVATE = sorted((PROJECT / "docs" / "papers").glob("*.pdf")) + [PROJECT / "docs" / "course_project_guide.pdf"]


def main():
    prints = set()
    for pdf in PRIVATE:
        if not pdf.exists():
            print(f"skipped (not on this computer): {pdf.relative_to(PROJECT)}")
            continue
        text = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True, check=True).stdout
        new = fingerprints(text)
        prints |= new
        print(f"{pdf.relative_to(PROJECT)}: {len(new)} fingerprints")
    old = set(FINGERPRINTS.read_text().split()) if FINGERPRINTS.exists() else set()
    FINGERPRINTS.write_text("\n".join(sorted(prints | old)) + "\n")   # never forget an old document
    print(f"saved {len(prints | old)} fingerprints to {FINGERPRINTS.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
