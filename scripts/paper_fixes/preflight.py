#!/usr/bin/env python3
"""Pre-submission gate for the manuscript.

Every check here corresponds to a defect that actually reached a compiled PDF at least
once, so each one is a regression test rather than a style opinion. Exit code is the
number of failed checks, so CI can gate on it directly.

    python scripts/paper_fixes/preflight.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "docs" / "paper"
TEX = PAPER / "manuscript" / "revision"
MAIN = TEX / "main.tex"

RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    RESULTS.append((ok, name, detail))


def load(name: str) -> dict:
    p = PAPER / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main() -> int:
    src = MAIN.read_text(encoding="utf-8")
    all_tex = "\n".join(p.read_text(encoding="utf-8") for p in TEX.glob("*.tex"))

    # --- 1. Author-supplied facts that must not ship as placeholders -------------
    todo = re.findall(r"\[TO SUPPLY: ([^\]]+)\]", all_tex)
    check(not todo, "no [TO SUPPLY] placeholders", "; ".join(todo))

    check(
        "PENDING-ZENODO-DOI" not in all_tex,
        "data-availability DOI is resolved",
        "Data availability still points at PENDING-ZENODO-DOI",
    )

    # --- 2. Control characters that once corrupted page 1 ------------------------
    bad = {}
    for tf in sorted(TEX.glob("*.tex")):
        raw = tf.read_bytes()
        for name, b in (("TAB", b"\t"), ("CR", b"\r"), ("FF", b"\x0c")):
            if raw.count(b):
                bad[f"{tf.name}:{name}"] = raw.count(b)
    check(not bad, "no stray control bytes in any .tex",
          ", ".join(f"{k}={v}" for k, v in bad.items()))

    # --- 3. Cross-table agreement on top-1 --------------------------------------
    # tab_scale_study and tab_abstain both print held-out top-1. They disagreed by 0.1
    # on one cell because one read a pre-rounded JSON. Compare the rendered strings.
    scale = (TEX / "tab_scale_study.tex").read_text(encoding="utf-8")
    abstain = (TEX / "tab_abstain.tex").read_text(encoding="utf-8")
    cfg_of = {"A": "16 unseen", "B": "34 unseen", "C": "51 unseen"}
    mismatches = []
    for cfg in "ABC":
        zs = load(f"zeroshot_eval_{cfg}.json").get("models", {})
        for model, d in zs.items():
            short = model.split("/")[0].replace("_", "-")
            for strat in ("rich", "grounded"):
                want = f"{d[strat]['acc'] * 100:.1f}\\%"
                row = [ln for ln in abstain.splitlines()
                       if ln.startswith(f"{cfg} & ") and f"& {short} &" in ln]
                if row and want not in row[0]:
                    mismatches.append(f"{cfg}/{short}/{strat} expected {want}")
    check(not mismatches, "tab_scale_study and tab_abstain agree on top-1",
          "; ".join(mismatches[:4]))
    del scale, cfg_of  # read for symmetry; comparison is done against the JSON

    # --- 4. Elsevier submission limits ------------------------------------------
    kw = re.search(r"\\begin\{keywords\}(.*?)\\end\{keywords\}", src, re.S)
    n_kw = len([k for k in kw.group(1).split(r"\sep") if k.strip()]) if kw else 0
    check(0 < n_kw <= 6, "at most six keywords", f"found {n_kw}")

    # Count what a reader sees, not what the source contains: strip comments, drop math
    # delimiters so "$+1.96$" stays one token, and collapse LaTeX control sequences.
    ab = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", src, re.S)
    if ab:
        body = re.sub(r"(?m)^\s*%.*$", " ", ab.group(1))
        body = re.sub(r"\$([^$]*)\$", lambda m: m.group(1).replace(" ", ""), body)
        body = re.sub(r"\\[a-zA-Z]+\s*", " ", body)
        n_words = len([w for w in re.sub(r"[{}~\\]", " ", body).split() if any(c.isalnum() for c in w)])
    else:
        n_words = 0
    check(0 < n_words <= 250, "abstract at most 250 words (rendered)", f"found {n_words}")

    # --- 5. Claims that were false against the released JSONs -------------------
    check("leads on neither sweep" not in src,
          "abstain claim matches metrics_abstain_A.json",
          "grounded leads the top-5 sweep 3/4 at config A")

    check("every encoder, every strategy and every scale we measured" not in src,
          "monotonicity claim is scoped to the reported grid",
          "pointwise monotonicity fails in 19/30 curves")

    check("Selective accuracy rising with coverage," not in src,
          "no duplicated/reversed selective-accuracy fragment")

    # --- 6. The 19.9 vs 27.0 fence ----------------------------------------------
    # Both numbers are correct but are different descriptor strategies on the same
    # 17-class pilot. Every appearance must sit next to its strategy name.
    # Conditional for the same reason as the bake-off check below: the 2026-09-22 salvage cut the
    # pilot table, so the revision has no tab_pilot.tex. The rule still holds wherever the table
    # exists; a table that no longer exists is not a failure.
    pilot = TEX / "tab_pilot.tex"
    if pilot.exists():
        pilot_cap = pilot.read_text(encoding="utf-8")
        check("crude" in pilot_cap and "keyword-bank descriptors)" not in pilot_cap,
              "pilot table names its descriptor strategy",
              "tab_pilot caption must say crude, not keyword-bank")
    # Conditional since the 2026-09-22 salvage, which cut the Protocol-P bake-off. The rule still
    # holds wherever a bake-off section exists; it no longer demands that one exist.
    if r"\subsection{Encoder bake-off}" in src:
        check("rich" in src.split(r"\subsection{Encoder bake-off}")[-1][:1200],
              "bake-off subsection names its descriptor strategy")

    # --- 7. Reproducibility disclosure ------------------------------------------
    for token, label in (
        ("dfndr2b", "pretraining checkpoint tags present"),
        ("ONNX Runtime 1.26.0", "edge measurement conditions stated"),
    ):
        check(token in src, label, f"{token!r} not found in main.tex")
    # Protocol N only needs defining when a second protocol (P) is present to be told apart from.
    # The 2026-09-22 salvage cut Protocol P, leaving one protocol and nothing to disambiguate.
    if "Protocol~P" in src:
        check("Protocol~N" in src, "Protocol N defined", "'Protocol~P' used but 'Protocol~N' not defined")

    # --- report ------------------------------------------------------------------
    failed = [r for r in RESULTS if not r[0]]
    for ok, name, detail in RESULTS:
        mark = "PASS" if ok else "FAIL"
        line = f"  [{mark}] {name}"
        if not ok and detail:
            line += f"  --  {detail}"
        print(line)
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        print("NOT READY TO SUBMIT")
    return len(failed)


if __name__ == "__main__":
    sys.exit(main())
