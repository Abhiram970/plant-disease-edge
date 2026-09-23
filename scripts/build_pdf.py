#!/usr/bin/env python3
"""Compile docs/paper/tex/main.tex to a PDF, and report what the log actually says.

Runs pdflatex -> bibtex -> pdflatex -> pdflatex into docs/paper/build/, then surfaces
the three things that matter and that a silent "Output written" hides: unresolved
references, missing citations, and overfull boxes.

Two gotchas this handles, both of which produced a PDF that looked fine:

  BIBINPUTS   bibtex runs with the .aux in the output directory and resolves \\bibdata
              relative to its own cwd, so refs.bib in tex/ is invisible to it. Every
              citation then silently renders as [?] while pdflatex still exits 0.
  rerun       cross-references need two passes after bibtex; stopping early leaves
              "??" in the text with no error.

    python scripts/build_pdf.py
    python scripts/build_pdf.py --open
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEX = REPO / "docs" / "paper" / "tex"
BUILD = REPO / "docs" / "paper" / "build"
PDF = BUILD / "main.pdf"

# MiKTeX's per-user install is not on PATH in every shell.
EXTRA_PATHS = [
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "MiKTeX" / "miktex" / "bin" / "x64",
    Path("C:/Program Files/MiKTeX/miktex/bin/x64"),
]


def engine(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    for d in EXTRA_PATHS:
        cand = d / f"{name}.exe"
        if cand.exists():
            return str(cand)
    raise SystemExit(
        f"{name} not found. Install a TeX distribution "
        f"(winget install MiKTeX.MiKTeX) or compile docs/paper/compag_submission.zip "
        f"on Overleaf instead.")


def run(cmd: list[str], cwd: Path, env: dict) -> str:
    p = subprocess.run(cmd, cwd=cwd, env=env, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p.stdout or ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true", help="open the PDF when done")
    ap.add_argument("--keep", action="store_true", help="keep the previous build directory")
    args = ap.parse_args()

    pdflatex, bibtex = engine("pdflatex"), engine("bibtex")
    if not args.keep and BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    # Trailing separator means "and the default search path too".
    env["BIBINPUTS"] = f"{TEX}{os.pathsep}"
    env["TEXINPUTS"] = f"{TEX}{os.pathsep}"

    common = [pdflatex, "-interaction=nonstopmode", "-file-line-error",
              f"-output-directory={BUILD}", "main.tex"]

    print("pass 1 (pdflatex)")
    run(common, TEX, env)
    print("bibtex")
    blg = run([bibtex, "main"], BUILD, env)
    for line in blg.splitlines():
        if "couldn't open" in line or "I found no database" in line:
            print(f"  !! {line.strip()}")
    print("pass 2 (pdflatex)")
    run(common, TEX, env)
    print("pass 3 (pdflatex)")
    log = run(common, TEX, env)

    if not PDF.exists():
        print("\nFAILED: no PDF produced. Last errors:")
        for line in log.splitlines():
            if re.search(r"^main\.tex:\d+:|Fatal|! ", line):
                print(f"  {line}")
        return 1

    logtext = (BUILD / "main.log").read_text(encoding="utf-8", errors="replace")
    # pdflatex hard-wraps the log at ~79 columns, so "Output written on ... (16 pages"
    # is routinely split mid-phrase. Match across the break rather than reporting "?".
    flat = re.sub(r"\s*\n\s*", " ", logtext)
    pages = re.search(r"Output written on .*?\((\d+) pages?", flat)
    size_kb = PDF.stat().st_size / 1024

    print(f"\nPDF: {PDF}")
    print(f"     {pages.group(1) if pages else '?'} pages, {size_kb:.0f} KB")

    # The failure modes that still produce a PDF.
    undefined = re.findall(r"LaTeX Warning: (Citation|Reference) `([^']+)' (?:on page \d+ )?undefined",
                           logtext)
    if undefined:
        kinds = {k for k, _ in undefined}
        names = sorted({n for _, n in undefined})
        print(f"\n!! {len(names)} undefined {'/'.join(sorted(kinds)).lower()}: {names[:8]}"
              + (" ..." if len(names) > 8 else ""))
    else:
        print("     all citations and references resolved")

    if "Rerun to get" in logtext or "Rerun LaTeX" in logtext:
        print("!! LaTeX asked for another pass -- run again")

    overfull = re.findall(r"Overfull \\[hv]box \(([\d.]+)pt too (?:wide|high)\)", logtext)
    bad = [float(x) for x in overfull if float(x) > 5.0]
    print(f"     {len(overfull)} overfull boxes"
          + (f", {len(bad)} over 5pt (worst {max(bad):.1f}pt)" if bad else ""))

    missing_gfx = re.findall(r"File `([^']+)' not found", logtext)
    if missing_gfx:
        print(f"!! missing files: {sorted(set(missing_gfx))}")

    if args.open:
        os.startfile(PDF)  # noqa: S606
    return 0


if __name__ == "__main__":
    sys.exit(main())
