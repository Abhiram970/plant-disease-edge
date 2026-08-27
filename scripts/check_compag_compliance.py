"""
Check docs/paper/tex/main.tex against the Computers and Electronics in Agriculture guide for authors.

Encodes the mechanical rules from the journal's Guide for Authors (retrieved 19 Aug 2026) so they are
checked every time rather than remembered:

  * Abstract    <= 250 words, no citations inside it.
  * Keywords    1-7, and avoid multi-word keywords joined by "and"/"of".
  * Highlights  3-5 bullets, each <= 85 characters including spaces, supplied as a separate file.
  * Sections    numbered; the abstract is not part of the numbering.
  * Front matter: CRediT, competing interests, funding, data availability, generative-AI disclosure.
  * Figures     >= 300 dpi, and single-column art at least 1063 px wide.

    python scripts/check_compag_compliance.py
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
C_REPO = C.REPO_ROOT

TEX = C.REPO_ROOT / "docs" / "paper" / "tex" / "main.tex"
FIGS = C.REPO_ROOT / "docs" / "paper" / "figures"

ABSTRACT_MAX = 250
HL_MIN, HL_MAX, HL_CHARS = 3, 5, 85
KW_MAX = 7
MIN_PX_SINGLE_COL = 1063          # Elsevier: single column at 300 dpi
# cas-dc prints CRediT from per-author \credit{} macros via \printcredits, so accept either that
# or an explicit section (elsarticle style).
REQUIRED_SECTIONS = [
    (("CRediT authorship contribution statement", "\\printcredits"), "CRediT"),
    (("Declaration of competing interest",), "competing interests"),
    (("Funding",), "funding"),
    (("Data availability",), "data statement (Option C applies to this journal)"),
    (("generative AI",), "generative-AI disclosure"),
]

# The highlights file the journal wants uploaded separately.
HIGHLIGHTS_FILE = C_REPO / "docs" / "paper" / "highlights.txt"

ok, warn, fail = [], [], []


def words_in(tex: str) -> int:
    s = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", tex)   # drop macros w/ args
    s = re.sub(r"[{}$~\\]", " ", s)
    return len([w for w in s.split() if any(c.isalnum() for c in w)])


def block(tex: str, env: str):
    m = re.search(rf"\\begin\{{{env}\}}(.*?)\\end\{{{env}\}}", tex, re.S)
    return m.group(1) if m else None


def main():
    if not TEX.exists():
        print(f"[skip] {TEX} not found")
        return 0
    t = TEX.read_text(encoding="utf-8")
    t_nocomment = re.sub(r"(?<!\\)%.*", "", t)

    # ---- abstract
    abs_ = block(t_nocomment, "abstract")
    if abs_ is None:
        fail.append("no abstract found")
    else:
        n = words_in(abs_)
        (ok if n <= ABSTRACT_MAX else fail).append(
            f"abstract is {n} words (limit {ABSTRACT_MAX})")
        if re.search(r"\\cite[pt]?\*?\{", abs_):
            fail.append("abstract contains \\cite -- the guide asks for citations in full, so avoid "
                        "them in the abstract")

    # ---- keywords (elsarticle uses "keyword", cas-dc uses "keywords")
    kw = block(t_nocomment, "keywords") or block(t_nocomment, "keyword")
    if kw is None:
        fail.append("no \\begin{keyword} block")
    else:
        items = [k.strip() for k in re.split(r"\\sep", kw) if k.strip()]
        (ok if 1 <= len(items) <= KW_MAX else fail).append(
            f"{len(items)} keywords (allowed 1-{KW_MAX})")
        bad = [k for k in items if re.search(r"\b(and|of)\b", k)]
        if bad:
            warn.append(f"keywords joined by and/of, which the guide discourages: {bad}")

    # ---- highlights: a standalone file is the submission form the journal asks for
    hl = block(t_nocomment, "highlights")
    if hl is None and HIGHLIGHTS_FILE.exists():
        items = [ln.strip() for ln in
                 HIGHLIGHTS_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()]
        (ok if HL_MIN <= len(items) <= HL_MAX else fail).append(
            f"{len(items)} highlights in {HIGHLIGHTS_FILE.name} (allowed {HL_MIN}-{HL_MAX})")
        long_ = [i for i in items if len(i) > HL_CHARS]
        (ok if not long_ else fail).append(
            f"all highlights within {HL_CHARS} characters" if not long_
            else f"highlights over {HL_CHARS} chars: {long_}")
        hl = None
    elif hl is None:
        fail.append("no highlights: neither a \\begin{highlights} block nor docs/paper/highlights.txt")
    if hl is not None:
        items = [i.strip() for i in re.findall(r"\\item(.*)", hl) if i.strip()]
        (ok if HL_MIN <= len(items) <= HL_MAX else fail).append(
            f"{len(items)} highlights (allowed {HL_MIN}-{HL_MAX})")
        for i in items:
            plain = i.replace("\\%", "%").replace("\\&", "&").strip()
            if len(plain) > HL_CHARS:
                fail.append(f"highlight is {len(plain)} chars (limit {HL_CHARS}): {plain[:60]}...")
        if all(len(i.replace("\\%", "%").strip()) <= HL_CHARS for i in items):
            ok.append(f"all {len(items)} highlights within {HL_CHARS} characters")

    # ---- required front/back matter
    for needles, label in REQUIRED_SECTIONS:
        hit = any(n.lower() in t_nocomment.lower() for n in needles)
        (ok if hit else fail).append(f"{label} statement present")

    # ---- unresolved placeholders
    # PENDING-/TBD/TODO are in this pattern deliberately. The DOI placeholder in the
    # data-availability statement used to slip through a check that only looked for FILL/XXXX, so
    # the suite reported a clean 11/11 while the single hardest submission blocker was still in the
    # manuscript. A compliance check that misses the blocker is worse than no check at all.
    holes = re.findall(r"(FILL[^}\n]*|\[fill[^\]]*\]|XXXX|PENDING[-\w]*|\bTBD\b|\bTODO\b)",
                       t_nocomment)
    (ok if not holes else fail).append(
        "no unresolved placeholders" if not holes else f"unresolved placeholders: {sorted(set(holes))}")

    # ---- figures
    if FIGS.exists():
        try:
            from PIL import Image
            small = []
            for f in sorted(FIGS.glob("*.png")):
                with Image.open(f) as im:
                    # PNG stores resolution as pixels-per-metre, so a 300 dpi figure round-trips as
                    # 299.9994. Compare on the rounded value or every figure false-fails.
                    dpi = round(im.info.get("dpi", (0, 0))[0])
                    if im.width < MIN_PX_SINGLE_COL or dpi < 300:
                        small.append(f"{f.name} ({im.width}px, {dpi}dpi)")
            (ok if not small else warn).append(
                f"all {len(list(FIGS.glob('*.png')))} figures >= {MIN_PX_SINGLE_COL}px and 300dpi"
                if not small else f"figures below spec: {small}")
        except ImportError:
            warn.append("Pillow not installed; figure resolution not checked")

    for m in ok:
        print(f"  [OK]   {m}")
    for m in warn:
        print(f"  [WARN] {m}")
    for m in fail:
        print(f"  [FAIL] {m}")
    print(f"\n{len(ok)} passed, {len(warn)} warnings, {len(fail)} failures")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
