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

TEXDIR = C.REPO_ROOT / "docs" / "paper" / "tex"
FIGDIR = C.REPO_ROOT / "docs" / "paper" / "figures"
OUT = C.REPO_ROOT / "docs" / "paper" / "submission"


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "figures").mkdir(parents=True)

    tex = (TEXDIR / "main.tex").read_text(encoding="utf-8")

    # Figures, in order of first \includegraphics appearance.
    used = []
    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex):
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
    for t in sorted(TEXDIR.glob("tab_*.tex")):
        shutil.copy(t, OUT / t.name)

    # Highlights as a separate editable file (required by the guide, with "highlights" in the name).
    hl = re.search(r"\\begin\{highlights\}(.*?)\\end\{highlights\}", tex, re.S)
    if hl:
        items = [re.sub(r"\\[a-zA-Z]+\{?|\}", "", i).replace("\\%", "%").strip()
                 for i in re.findall(r"\\item(.*)", hl.group(1)) if i.strip()]
        (OUT / "highlights.txt").write_text(
            "\n".join(items) + "\n", encoding="utf-8")

    readme = f"""# Submission package — Computers and Electronics in Agriculture

Upload the whole folder (or the zip) to Overleaf and set **main.tex** as the compile target.

## Build
Requires `elsarticle` (bundled with Overleaf). Compile order:

    pdflatex main -> bibtex main -> pdflatex main -> pdflatex main

Two BibTeX-dependent passes are needed or citations render as `?`.

## Contents
- `main.tex` — manuscript (elsarticle, `preprint,3p,times`)
- `refs.bib` — {len(re.findall(r'^@', (TEXDIR / 'refs.bib').read_text(encoding='utf-8'), re.M))} entries, all verified against the published record
- `tab_*.tex` — {len(list(TEXDIR.glob('tab_*.tex')))} tables, generated from the result files; do not hand-edit
- `figures/Figure_1..{len(mapping)}.png` — 300 dpi, renamed in order of appearance
- `highlights.txt` — upload separately in Editorial Manager, as the guide requires

## Remaining author actions
1. **Data availability** — the journal applies Option C, which *requires* the data to be deposited,
   cited and linked. Replace `PENDING-ZENODO-DOI` in `main.tex` with a real DOI (mint one by linking
   the GitHub repo to Zenodo), or make the repository public and use that URL.
2. **Graphical abstract** — encouraged, not required. 531 x 1328 px minimum, submitted separately.
3. **Corresponding author** needs a full postal address and phone number in Editorial Manager.

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
