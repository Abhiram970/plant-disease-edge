"""
Resolve \\ref and \\cite across the LaTeX sources the way LaTeX actually does.

WHY: linters that read main.tex alone report every table reference as undefined, because the tables
live in generated tab_*.tex files pulled in with \\input. That produced 8 false "blocking" errors.
Expanding \\input first removes them — and exposed the real problem underneath: 14 of 19 bibliography
entries were never cited, so the compiled paper would have had a 5-item reference list while
discussing SAGE, SigLIP2, BioCLIP2, SCOLD, ResNet and MobileNet with no attribution.

    python scripts/check_tex_refs.py        # exit 1 on undefined refs/cites or uncited entries
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

TEX = C.REPO_ROOT / "docs" / "paper" / "tex"


def expand(path: Path, depth: int = 0) -> str:
    """Inline \\input{} recursively, as LaTeX does before resolving cross-references."""
    if depth > 8 or not path.exists():
        return ""
    txt = path.read_text(encoding="utf-8")

    def sub(m):
        name = m.group(1)
        child = TEX / (name if name.endswith(".tex") else name + ".tex")
        return expand(child, depth + 1) if child.exists() else f"%% MISSING INPUT {name}"

    return re.sub(r"\\input\{([^}]+)\}", sub, txt)


def main():
    main_tex = TEX / "main.tex"
    if not main_tex.exists():
        print(f"[skip] {main_tex} not found")
        return 0

    full = re.sub(r"(?<!\\)%.*", "", expand(main_tex))   # strip comments, keep escaped \%
    labels = set(re.findall(r"\\label\{([^}]+)\}", full))
    refs = set(re.findall(r"\\(?:ref|autoref|eqref)\{([^}]+)\}", full))
    cites = {k.strip()
             for grp in re.findall(r"\\cite[pt]?\*?(?:\[[^\]]*\])*\{([^}]+)\}", full)
             for k in grp.split(",") if k.strip()}

    bib = TEX / "refs.bib"
    bibkeys = set(re.findall(r"@\w+\{([^,]+),", bib.read_text(encoding="utf-8"))) if bib.exists() \
        else set()

    bad_ref = sorted(refs - labels)
    bad_cite = sorted(cites - bibkeys)
    uncited = sorted(bibkeys - cites)

    print(f"labels {len(labels)} | refs {len(refs)} | cites {len(cites)} | bib {len(bibkeys)}")
    ok = True
    if bad_ref:
        print(f"[FAIL] undefined \\ref: {bad_ref}"); ok = False
    if bad_cite:
        print(f"[FAIL] \\cite with no bib entry: {bad_cite}"); ok = False
    if uncited:
        # Not fatal on its own, but an uncited entry never appears in the reference list, so a work
        # discussed in the prose ends up with no attribution at all.
        print(f"[FAIL] bib entries never cited ({len(uncited)}): {uncited}"); ok = False

    unused_labels = sorted(labels - refs)
    if unused_labels:
        print(f"[info] labels defined but never referenced: {unused_labels}")

    print("[OK] all references and citations resolve" if ok else "[FAIL] see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
