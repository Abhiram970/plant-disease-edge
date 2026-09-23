#!/usr/bin/env python3
"""Structural validation of the manuscript source without a LaTeX engine.

Catches the failure modes that reach a compiled PDF silently: unbalanced braces,
unmatched environments, undefined \\ref targets, and citations with no bib entry.
LaTeX warns about some of these and is silent about others, so this runs regardless
of whether a TeX distribution is installed.

    python scripts/paper_fixes/check_structure.py
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "docs" / "paper" / "manuscript" / "revision"

INPUT_RE = re.compile(r"\\input\{([^}]+)\}")


def inline(text: str, depth: int = 0) -> str:
    """Splice \\input{} files in, the way LaTeX would.

    Built with an explicit scan rather than re.sub: a function replacement still
    hands the matched text back through the regex machinery in some paths, and
    table bodies are full of backslashes.
    """
    if depth > 4:
        return text
    out, pos = [], 0
    for m in INPUT_RE.finditer(text):
        out.append(text[pos:m.start()])
        f = TEX / (m.group(1) + ".tex")
        out.append(inline(f.read_text(encoding="utf-8"), depth + 1) if f.exists() else m.group(0))
        pos = m.end()
    out.append(text[pos:])
    return "".join(out)


def strip_comments(text: str) -> str:
    """Drop % comments but keep escaped \\%."""
    out = []
    for line in text.split("\n"):
        i, esc = 0, False
        while i < len(line):
            if line[i] == "\\":
                esc = not esc
            elif line[i] == "%" and not esc:
                line = line[:i]
                break
            else:
                esc = False
            i += 1
        out.append(line)
    return "\n".join(out)


def main() -> int:
    doc = strip_comments(inline((TEX / "main.tex").read_text(encoding="utf-8")))
    problems = 0

    # --- braces ---------------------------------------------------------------
    bal, line, first_bad = 0, 1, None
    for i, ch in enumerate(doc):
        if ch == "\n":
            line += 1
        elif ch in "{}" and (i == 0 or doc[i - 1] != "\\"):
            bal += 1 if ch == "{" else -1
            if bal < 0 and first_bad is None:
                first_bad = line
    if bal != 0 or first_bad:
        print(f"FAIL brace balance = {bal}" + (f", first stray close near line {first_bad}" if first_bad else ""))
        problems += 1
    else:
        print("PASS braces balanced")

    # --- environments ---------------------------------------------------------
    opened = Counter(re.findall(r"\\begin\{([a-zA-Z*]+)\}", doc))
    closed = Counter(re.findall(r"\\end\{([a-zA-Z*]+)\}", doc))
    unmatched = {k: v for k, v in (opened - closed).items()}
    unmatched.update({k: -v for k, v in (closed - opened).items()})
    if unmatched:
        print(f"FAIL unmatched environments: {unmatched}")
        problems += 1
    else:
        print(f"PASS {sum(opened.values())} environments matched")

    # --- refs -----------------------------------------------------------------
    labels = set(re.findall(r"\\label\{([^}]+)\}", doc))
    refs = set(re.findall(r"\\(?:ref|autoref|eqref)\{([^}]+)\}", doc))
    if refs - labels:
        print(f"FAIL undefined refs: {sorted(refs - labels)}")
        problems += 1
    else:
        print(f"PASS all {len(refs)} refs resolve ({len(labels)} labels)")
    if labels - refs:
        print(f"note  unreferenced labels: {sorted(labels - refs)}")

    # --- orphaned control words -----------------------------------------------
    # A LaTeX command that lost its backslash (Section~<CR>ef{...} instead of
    # Section~\ref{...}) is INVISIBLE to every other check here: it is not a \ref, so
    # it cannot be undefined; it is not a command, so LaTeX does not complain; and the
    # label it should have pointed at may still be referenced elsewhere, so it does not
    # even show up as unused. It just prints as garbage in the PDF. Delegated to
    # fix_orphaned_refs.py so the detection lives in one place.
    import subprocess
    rc = subprocess.run([sys.executable, str(Path(__file__).with_name("fix_orphaned_refs.py")),
                         "--check"], capture_output=True, text=True)
    if rc.returncode != 0:
        print("FAIL orphaned control words (run fix_orphaned_refs.py):")
        print("".join(f"  {ln}\n" for ln in rc.stdout.splitlines() if ln.strip()))
        problems += 1
    else:
        print("PASS no orphaned control words")

    # --- citations ------------------------------------------------------------
    bib = (TEX / "refs.bib").read_text(encoding="utf-8")
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    cited: set[str] = set()
    for m in re.findall(r"\\cite[a-zA-Z]*\{([^}]+)\}", doc):
        cited |= {k.strip() for k in m.split(",")}
    if cited - keys:
        print(f"FAIL citations with no bib entry: {sorted(cited - keys)}")
        problems += 1
    else:
        print(f"PASS all {len(cited)} citations resolve ({len(keys)} bib entries)")
    if keys - cited:
        print(f"note  uncited bib entries: {sorted(keys - cited)}")

    print(f"\n{problems} structural problem(s)")
    return problems


if __name__ == "__main__":
    sys.exit(main())
