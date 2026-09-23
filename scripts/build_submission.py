"""
Assemble an Overleaf-ready submission package for Computers and Electronics in Agriculture.

Produces docs/paper/submission/ and a zip of it containing:

    main.tex, refs.bib, tab_*.tex        the manuscript and its generated tables
    figures/Figure_1.png ... Figure_N    renamed in order of first appearance, as the guide asks
    highlights.txt                       the journal wants highlights as a SEPARATE editable file
    README_SUBMISSION.md                 build instructions and the remaining author actions

Figures are renamed rather than copied verbatim because the guide asks for a logical naming
convention in order of appearance ("Figure_1, Figure_2 etc."); \\includegraphics keys in the copied
main.tex are rewritten to match, so the package compiles as-is on Overleaf.

    python scripts/build_submission.py
"""
from __future__ import annotations
import re
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

def _strip_comments(text: str) -> str:
    """Blank out LaTeX % comments, keeping escaped \\% and preserving line count."""
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


TEXDIR = C.REPO_ROOT / "docs" / "paper" / "manuscript" / "submitted"
FIGDIR = C.REPO_ROOT / "docs" / "paper" / "figures"
OUT = C.REPO_ROOT / "docs" / "paper" / "submission"


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "figures").mkdir(parents=True)

    tex = (TEXDIR / "main.tex").read_text(encoding="utf-8")

    # Scan for figures with comments STRIPPED. A %% comment that quotes a macro -- the
    # note above \maketitle explains the cas-common \includegraphics{thumbnails/...}
    # call -- was picked up as a real figure, shifting every later figure's number by one
    # and reporting a missing source for a file the manuscript never uses.
    scan = _strip_comments(tex)

    # Figures, in order of first \includegraphics appearance.
    used = []
    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", scan):
        name = m.group(1)
        if name not in used:
            used.append(name)

    missing, mapping = [], {}
    for i, name in enumerate(used, 1):
        src = FIGDIR / (name if name.endswith(".png") else name + ".png")
        if not src.exists():
            missing.append(name)
            continue
        dst = f"Figure_{i}.png"
        shutil.copy(src, OUT / "figures" / dst)
        mapping[name] = f"figures/{dst}"

    for old, new in mapping.items():
        tex = tex.replace("{" + old + "}", "{" + new + "}")
    # graphicspath pointed at ../figures in the repo layout; the package is self-contained.
    tex = tex.replace("\\graphicspath{{../figures/}}", "\\graphicspath{{figures/}}")
    (OUT / "main.tex").write_text(tex, encoding="utf-8")

    shutil.copy(TEXDIR / "refs.bib", OUT / "refs.bib")
    # Ship only the tables the manuscript \input's. Globbing tab_*.tex also shipped six tables
    # left over from the pre-salvage draft, among them a WiSE-FT table for a section that was cut.
    tables = []
    for m in re.finditer(r"\\input\{(tab_[^}]+?)(?:\.tex)?\}", scan):
        if m.group(1) not in tables:
            tables.append(m.group(1))
    for name in tables:
        src = TEXDIR / f"{name}.tex"
        if not src.exists():
            missing.append(f"{name}.tex")
            continue
        shutil.copy(src, OUT / src.name)

    # Highlights ship as a separate editable file with "highlights" in the name, per the guide.
    hl_src = C.REPO_ROOT / "docs" / "paper" / "highlights.txt"
    if hl_src.exists():
        shutil.copy(hl_src, OUT / "highlights.txt")

    readme = f"""# Submission package — Computers and Electronics in Agriculture

Upload the whole folder (or the zip) to Overleaf and set **main.tex** as the compile target.

## Build
Requires the Elsevier CAS class (`cas-dc.cls`, `cas-model2-names.bst`) - on Overleaf
start from the **Elsevier CAS** template, or the files are on CTAN. Compile order:

    pdflatex main -> bibtex main -> pdflatex main -> pdflatex main

Two BibTeX-dependent passes are needed or citations render as `?`.

## Contents
- `main.tex` — manuscript (cas-dc, double column, same layout as the authors' ASR submission)
- `refs.bib` — {len(re.findall(r'^@', (TEXDIR / 'refs.bib').read_text(encoding='utf-8'), re.M))} entries (BibTeX prints only the cited ones)
- `tab_*.tex` — {len(tables)} tables ({', '.join(tables)}), generated from the result files; do not hand-edit
- `figures/Figure_1..{len(mapping)}.png` — 300 dpi, renamed in order of appearance
- `highlights.txt` — upload separately in Editorial Manager, as the guide requires

## Remaining author actions
1. **Data availability** — the journal applies Option C, which *requires* the data to be deposited,
   cited and linked. Replace `PENDING-ZENODO-DOI` in `main.tex` with a real DOI (mint one by linking
   the GitHub repo to Zenodo), or make the repository public and use that URL.
2. **References** — check every entry against the published record before upload. Two entries
   still end in `and others` and four selective-prediction entries are marked VERIFY in `refs.bib`.
3. **Graphical abstract** — encouraged, not required. 531 x 1328 px minimum, submitted separately.
4. **Corresponding author** needs a full postal address and phone number in Editorial Manager.

## Regenerating
Tables and figures are generated, never typed:

    python docs/paper/make_tex_tables.py
    python docs/paper/make_figures.py
    python scripts/build_submission.py
"""
    (OUT / "README_SUBMISSION.md").write_text(readme, encoding="utf-8")

    zip_path = C.REPO_ROOT / "docs" / "paper" / "compag_submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(OUT))

    print(f"[pkg] {OUT}")
    print(f"[pkg] figures renamed: {len(mapping)}")
    for old, new in mapping.items():
        print(f"       {old:34s} -> {new}")
    if missing:
        print(f"[pkg] MISSING figure sources: {missing}")
    print(f"[pkg] zip: {zip_path} ({zip_path.stat().st_size/1e6:.1f} MB)")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
