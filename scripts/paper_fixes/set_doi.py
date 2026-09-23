"""
Stamp the minted Zenodo DOI into every place that carries the placeholder, then re-verify.

Run this once, after the deposit is published:

    python scripts/paper_fixes/set_doi.py 10.5281/zenodo.1234567
    python scripts/paper_fixes/set_doi.py 10.5281/zenodo.1234567 --check   # show, write nothing

It rewrites the placeholder in the revision manuscript, CITATION.cff and .zenodo.json, and
refuses a DOI that does not look like one. It deliberately does NOT touch
docs/paper/manuscript/submitted/, which is the frozen 2026-09-22 submission.

Afterwards, rebuild and re-check:

    cd docs/paper/manuscript/revision && latexmk -pdf main.tex
    python scripts/paper_fixes/preflight.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
PLACEHOLDER = "PENDING-ZENODO-DOI"

# Each must contain the placeholder, or already the DOI. .zenodo.json is deliberately absent:
# Zenodo assigns the DOI on publication, so that file must not assert one.
TARGETS = [
    "docs/paper/manuscript/revision/main.tex",
    "CITATION.cff",
]

DOI_RE = re.compile(r"^10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+$")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("doi", help="the minted DOI, e.g. 10.5281/zenodo.1234567")
    ap.add_argument("--check", action="store_true", help="report what would change, write nothing")
    args = ap.parse_args()

    doi = args.doi.strip().removeprefix("https://doi.org/").removeprefix("doi:")
    if not DOI_RE.match(doi):
        print(f"ABORT: {doi!r} does not look like a DOI (expected 10.xxxx/suffix)")
        return 1

    plan, missing = [], []
    for rel in TARGETS:
        p = REPO / rel
        if not p.exists():
            missing.append(rel)
            continue
        t = p.read_text(encoding="utf-8")
        n = t.count(PLACEHOLDER)
        if n:
            plan.append((p, rel, n))
        elif doi in t:
            print(f"  [done] {rel} already carries {doi}")
        else:
            missing.append(f"{rel} (no placeholder and no DOI)")

    if missing:
        print("ABORT: nowhere to write the DOI in:")
        for m in missing:
            print(f"  - {m}")
        return 1
    if not plan:
        print("Nothing to do: every target already carries the DOI.")
        return 0

    for _p, rel, n in plan:
        print(f"  {rel}: {n} occurrence(s) of {PLACEHOLDER} -> {doi}")
    if args.check:
        print("\n--check only, nothing written")
        return 0

    for p, rel, _n in plan:
        t = p.read_text(encoding="utf-8").replace(PLACEHOLDER, doi)
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(t)
        print(f"[doi] wrote {rel}")

    print("\nNow rebuild and re-check:")
    print("  cd docs/paper/manuscript/revision && latexmk -pdf main.tex")
    print("  python scripts/paper_fixes/preflight.py        # the DOI check should pass")
    print("\nThe frozen submission at docs/paper/manuscript/submitted/ is intentionally untouched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
