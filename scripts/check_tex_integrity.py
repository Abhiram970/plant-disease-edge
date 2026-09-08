"""Catch LaTeX corruption that compiles cleanly and therefore escapes every other check.

A backslash lost to Python string escaping does not raise a LaTeX error: `\times` written
as TAB+"imes" typesets as italic "imes", and `\ref{x}` written as CR+"ef{x}" typesets as
the literal text "ef{x}". Neither produces a warning, so check_tex_refs.py -- which counts
\label and \ref -- reports "all references resolve" while the compiled page 1 reads
"3.4--12.2imes ... (Sections efsec:flat--efsec:desc)".

That is exactly what happened on 2026-09-08, in the first technical content an editor reads.
This script fails loudly on the byte patterns that cause it.

    python scripts/check_tex_integrity.py
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

TEX = Path(__file__).resolve().parent.parent / "docs" / "paper" / "tex"

# (name, compiled regex over BYTES, why it matters)
CHECKS = [
    ("TAB byte", re.compile(rb"\t"),
     "a tab in a .tex file is almost always a backslash eaten by Python string escaping "
     "(\t -> TAB); LaTeX renders the rest of the command as ordinary text"),
    ("stray control byte", re.compile(rb"[\x00-\x08\x0b\x0c\x0e-\x1f]"),
     "same cause as above for \a \b \f \v and friends"),
    # Built with chr(92) so this file cannot itself fall to the bug it detects.
    ("orphan ref/label/cite",
     re.compile(("(?<![A-Za-z" + chr(92) + chr(92) + "])"
                 "(ef|abel|ite|extbf|emph)" + chr(92) + "{").encode()),
     "a command whose leading backslash was lost; typesets as literal text and is "
     "invisible to a label/ref counter"),
]


def main() -> int:
    bad = 0
    files = sorted(TEX.glob("*.tex"))
    for f in files:
        data = f.read_bytes()
        for name, pat, why in CHECKS:
            for m in pat.finditer(data):
                line = data[: m.start()].count(b"\n") + 1
                ctx = data[max(0, m.start() - 40): m.start() + 40]
                print(f"[FAIL] {f.name}:{line}  {name}")
                print(f"       {ctx!r}")
                print(f"       {why}")
                bad += 1
    print(f"\nchecked {len(files)} .tex files")
    if bad:
        print(f"[FAIL] {bad} integrity problem(s) -- these compile silently, so fix before build")
        return 1
    print("[OK] no corrupted escapes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
