#!/usr/bin/env python3
"""Verify the assembled submission package is self-contained.

The package is what actually gets uploaded, and it is a COPY with figures renamed and
\\includegraphics keys rewritten. A mistake in that rewrite would not show up in any
check run against docs/paper/manuscript/submitted/, so it is checked here, on the copy, in isolation.

    python scripts/paper_fixes/verify_package.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "docs" / "paper" / "submission"

def _strip_comments(text: str) -> str:
    """Blank out LaTeX % comments, keeping escaped \\%."""
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


INCLUDE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
INPUT = re.compile(r"\\input\{([^}]+)\}")
BIBKEY = re.compile(r"@\w+\{([^,]+),")
CITE = re.compile(r"\\cite[a-zA-Z]*\{([^}]+)\}")


def main() -> int:
    if not PKG.exists():
        raise SystemExit(f"no package at {PKG}; run scripts/build_submission.py first")

    raw = (PKG / "main.tex").read_text(encoding="utf-8")
    # Scan with comments stripped, for the same reason build_submission.py does: a %%
    # note that quotes \includegraphics{thumbnails/...} is documentation, not a figure.
    main_tex = _strip_comments(raw)
    problems = 0

    # figures: the key may or may not carry its extension
    figs = INCLUDE.findall(main_tex)
    missing = [k for k in figs
               if not (PKG / k).exists() and not (PKG / (k + ".png")).exists()]
    print(f"{'FAIL' if missing else 'PASS'} figures: {len(figs)} keys, {len(missing)} missing"
          + (f" -> {missing}" if missing else ""))
    problems += bool(missing)

    # every figure shipped should also be referenced, or it is dead weight in the upload
    shipped = {f"figures/{p.name}" for p in (PKG / "figures").glob("*.png")}
    used = {k if k.endswith(".png") else k + ".png" for k in figs}
    orphan = sorted(shipped - used)
    print(f"{'WARN' if orphan else 'PASS'} no unused figures shipped"
          + (f" -> {orphan}" if orphan else ""))

    # \input targets
    inputs = INPUT.findall(main_tex)
    missing_in = [i for i in inputs if not (PKG / (i + ".tex")).exists()]
    print(f"{'FAIL' if missing_in else 'PASS'} inputs: {len(inputs)} files, "
          f"{len(missing_in)} missing" + (f" -> {missing_in}" if missing_in else ""))
    problems += bool(missing_in)

    # citations against the packaged bib
    body = main_tex + "".join((PKG / (i + ".tex")).read_text(encoding="utf-8")
                              for i in inputs if (PKG / (i + ".tex")).exists())
    keys = set(BIBKEY.findall((PKG / "refs.bib").read_text(encoding="utf-8")))
    cited: set[str] = set()
    for m in CITE.findall(body):
        cited |= {c.strip() for c in m.split(",")}
    missing_cite = sorted(cited - keys)
    print(f"{'FAIL' if missing_cite else 'PASS'} citations: {len(cited)} cited, "
          f"{len(keys)} in refs.bib" + (f", missing -> {missing_cite}" if missing_cite else ""))
    problems += bool(missing_cite)

    # the package must carry the CURRENT manuscript, not a stale copy
    live = (REPO / "docs" / "paper" / "manuscript" / "submitted" / "main.tex").read_text(encoding="utf-8")
    for marker, label in (("de-duplicated re-run", "2026-09-11 scaling revision"),
                          ("Ryzen~7 7840HS", "CPU disclosure"),
                          ("Claude Sonnet 4.6", "generator disclosure")):
        in_pkg, in_live = marker in main_tex, marker in live
        ok = in_pkg == in_live and in_live
        print(f"{'PASS' if ok else 'FAIL'} package carries the {label}")
        problems += not ok

    # things that must NOT ship
    for bad, label in (("PENDING-ZENODO-DOI", "placeholder DOI"),
                       ("[TO SUPPLY", "unfilled author fact")):
        hit = bad in main_tex
        print(f"{'FAIL' if hit else 'PASS'} no {label}")
        problems += hit

    print(f"\n{problems} problem(s) in the package")
    return problems


if __name__ == "__main__":
    sys.exit(main())
