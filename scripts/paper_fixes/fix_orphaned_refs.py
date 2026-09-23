#!/usr/bin/env python3
"""Repair LaTeX control words that lost their backslash, and detect the whole class.

HOW THIS HAPPENS. A single-backslash escape inside a NON-raw Python string is
interpreted by Python before it ever reaches the file: "\\ref" written as "\ref" is
CR + "ef", "\times" is TAB + "imes", "\newline" is LF + "ewline". LaTeX then sees a
bare word where a command should be. It does not warn -- `ef{sec:foo}` is ordinary
text, so there is no "undefined control sequence" and no "undefined reference" either,
because the reference was never made. The 2026-09-08 audit found three of these on
page 1; one more (Section~<CR>ef{sec:desc-gen}) was introduced on 2026-09-11 by a
patch script that used a non-raw string.

The fix is always the same, and so is the prevention: write LaTeX with r"" strings.

    python scripts/paper_fixes/fix_orphaned_refs.py            # repair + report
    python scripts/paper_fixes/fix_orphaned_refs.py --check    # report only, exit 1 if any
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "docs" / "paper" / "manuscript" / "submitted"

BS = chr(92)          # a literal backslash, built without writing one in a string
CR, TAB, LF, FF = chr(13), chr(9), chr(10), chr(12)

# (mangled prefix, control word) for the escapes Python actually swallows.
MANGLES = [
    (CR, "ref"), (CR, "right"), (CR, "rule"),
    (TAB, "times"), (TAB, "textbf"), (TAB, "textit"), (TAB, "texttt"), (TAB, "top"),
    (LF, "newline"), (LF, "node"), (LF, "num"),
    (FF, "frac"), (FF, "footnote"), (FF, "figure"),
    (chr(8), "begin"), (chr(8), "bf"), (chr(8), "bottomrule"),
    (chr(7), "alpha"), (chr(7), "and"),
    (chr(11), "vspace"),
]


def scan(text: str) -> list[tuple[str, str, int]]:
    """Return (control_word, mangled_char_name, count) for every hit."""
    names = {CR: "CR", TAB: "TAB", LF: "LF", FF: "FF",
             chr(8): "BS", chr(7): "BEL", chr(11): "VT"}
    hits = []
    for ch, word in MANGLES:
        n = text.count(ch + word[1:])
        if n:
            hits.append((word, names.get(ch, repr(ch)), n))
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report only; exit 1 if any found")
    args = ap.parse_args()

    total = 0
    for f in sorted(TEX.glob("*.tex")):
        # newline="" so a stray CR survives the read and can be seen; Path.read_text()
        # has no newline argument before 3.13, and universal-newline mode would silently
        # turn the CR into LF -- hiding the very byte this script exists to find.
        with open(f, encoding="utf-8", newline="") as fh:
            text = fh.read()
        hits = scan(text)
        if not hits:
            continue
        for word, chname, n in hits:
            print(f"  {f.name}: {n} x {chname}+{word[1:]!r}  -> should be {BS}{word}")
            total += n
        if not args.check:
            for ch, word in MANGLES:
                text = text.replace(ch + word[1:], BS + word)
            with open(f, "w", encoding="utf-8", newline="") as fh:
                fh.write(text)
            print(f"  {f.name}: repaired")

    if total == 0:
        print("no orphaned control words found")
        return 0
    print(f"\n{total} orphaned control word(s) {'found' if args.check else 'repaired'}")
    return 1 if args.check else 0


if __name__ == "__main__":
    sys.exit(main())
