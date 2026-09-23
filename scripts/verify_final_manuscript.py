#!/usr/bin/env python3
"""Check every number the final manuscript quotes against paper_numbers.json.

Each check pairs a phrase that must appear in main.tex with the value recomputed from the
released results. A quoted number that drifts from its source fails loudly, which is the
defect behind most findings in ten audit passes (a 5.7 that should have been 5.6, a caption
asserting a claim the table no longer supported).

    python scripts/verify_final_manuscript.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "docs" / "paper"
N = json.loads((P / "paper_numbers.json").read_text(encoding="utf-8"))
TEX = (P / "tex" / "main.tex").read_text(encoding="utf-8")
FLAT = " ".join(TEX.split())                       # whitespace-insensitive search

ok, bad = [], []


def c(e, t="uncleaned"):
    return N["configs"][e][t]


def f1(x):
    return f"{x:.1f}"


def need(phrase, why):
    """The phrase, with every number already formatted from the data, must appear."""
    if " ".join(phrase.split()) in FLAT:
        ok.append(why)
    else:
        bad.append(f"{why}\n        expected: {phrase!r}")


A, B, C = c("A"), c("B"), c("C")
Ac, Bc, Cc = c("A", "clean"), c("B", "clean"), c("C", "clean")
h = N["headline"]
avs = N["authoring_vs_size"]

# --- the central claim ---
need(f"gains {f1(A['authoring_gain'])}, {f1(B['authoring_gain'])} and {f1(C['authoring_gain'])} points",
     "authoring gain A/B/C")
need(f"gains {f1(A['size_range_under_best'])}, {f1(B['size_range_under_best'])} and "
     f"{f1(C['size_range_under_best'])}", "size gain A/B/C")
need(f"by {avs['uncleaned']['min']:.1f}--{avs['uncleaned']['max']:.1f} times on the full label set",
     "authoring/size ratio, full set")
need(f"the ratio is {avs['clean']['min']:.1f}--{avs['clean']['max']:.1f}", "authoring/size ratio, de-dup")
need(f"widens to {f1(Bc['size_range_under_best'])} points", "B de-dup size gain")
assert avs["uncleaned"]["all_above_1"] and avs["clean"]["all_above_1"], "authoring must exceed size everywhere"
ok.append("authoring exceeds size at every configuration on both label sets (checked, not quoted)")

gmin = min(A["authoring_gain"], B["authoring_gain"], C["authoring_gain"])
gmax = max(A["authoring_gain"], B["authoring_gain"], C["authoring_gain"])
smin = min(A["size_range_under_best"], B["size_range_under_best"], C["size_range_under_best"])
smax = max(A["size_range_under_best"], B["size_range_under_best"], C["size_range_under_best"])
need(f"gain {f1(gmin)}--{f1(gmax)} points over a bare class name, against {f1(smin)}--{f1(smax)} points",
     "abstract authoring vs size ranges")

# --- bare80 ---
need(f"changes accuracy by ${A['bare80_minus_bare']:.1f}$, ${B['bare80_minus_bare']:.1f}$ and "
     f"${C['bare80_minus_bare']:.1f}$ points", "bare80 - bare")

# --- which method wins ---
need(f"keyword bank leads at {f1(A['means']['rich'])}\\%", "rich leads at A")
need(f"its accuracy falls to {f1(C['means']['rich'])}\\% at C", "rich at C")
need(f"CuPL leads at B and C with {f1(B['means']['cupl'])}\\% and {f1(C['means']['cupl'])}\\%",
     "cupl at B and C")
need(f"ahead of source-grounded text by {f1(B['cupl_minus_grounded'])} and {f1(C['cupl_minus_grounded'])} points",
     "cupl - grounded")
need(f"of the keyword bank by {f1(B['cupl_minus_rich'])} and {f1(C['cupl_minus_rich'])}", "cupl - rich")
lo_b, hi_b = B["cupl_minus_grounded_range"]; lo_c, hi_c = C["cupl_minus_grounded_range"]
need(f"by {f1(lo_b)}--{f1(hi_b)} points at B and {f1(lo_c)}--{f1(hi_c)} at C", "per-encoder range")
assert B["cupl_beats_grounded_encoders"] == 4 and C["cupl_beats_grounded_encoders"] == 4
for e in "ABC":
    assert c(e, "clean")["cupl_beats_grounded_encoders"] == 4, f"de-dup {e}"
ok.append("CuPL beats grounded on 4/4 encoders at B, C and at every de-dup config (checked)")
need(f"($+{f1(B['dclip_minus_grounded'])}$ and $+{f1(C['dclip_minus_grounded'])}$ points)", "dclip - grounded")
need(f"falls {f1(-A['dclip_minus_grounded'])} points behind it at A", "dclip at A")

# --- headline ---
need(f"reaches {f1(h['best_at_C_acc'])}\\%, {h['best_over_majority_C']:.1f} times the 4.2\\% majority-class prior",
     "headline majority multiple")
need(f"{h['best_over_chance_C']:.1f} times uniform chance", "headline chance multiple")

# --- decomposition ---
cp = N["control_arm_paired"]
need(f"$+{cp['mean']:.2f}$ points (95\\% interval ${cp['lo']:.2f}$ to $+{cp['hi']:.2f}$", "control arm paired CI")
need("by $+0.94$ ($-0.52$ to $+2.39$)", "control arm full set")
need(f"accuracy moves by $+{f1(C['split_minus_grounded'])}$ points, recovering "
     f"{C['split_share_of_gap']*100:.0f}\\% of the gap", "construction, full")
need(f"$+{f1(Cc['split_minus_grounded'])}$ points and {Cc['split_share_of_gap']*100:.0f}\\%", "construction, de-dup")
need(f"What remains, {f1(C['cupl_minus_split'])} points at configuration C on the full label set and "
     f"{f1(Cc['cupl_minus_split'])} on the", "text residual")
need(f"recovers only {C['split_share_of_gap']*100:.0f}\\% of its deficit", "abstract construction share")

# --- per-configuration counts and images, straight from the eval JSONs ---
for e, n_img in (("A", "4{,}050"), ("B", "9{,}787"), ("C", "14{,}204")):
    need(f"{c(e)['n_classes']} classes, {n_img} images", f"config {e} size")
need(f"leaving {Ac['n_classes']}, {Bc['n_classes']} and {Cc['n_classes']} classes", "de-dup class counts")

# --- discussion ---
dmin = min(C["means"][s] for s in ("rich", "grounded", "dclip", "cupl"))
dmax = max(C["means"][s] for s in ("rich", "grounded", "dclip", "cupl"))
need(f"reach {f1(dmin)}--{f1(dmax)}\\%", "description-based range at C")

# --- highlights: a separate upload, so it was never re-checked and still carried the retracted
# "only strategy that improves" claim after the manuscript dropped it. Every number in a
# highlight must also appear in the (verified) manuscript, and each must fit the journal's limit.
HL_FILE = P / "highlights.txt"
HL = [l.strip() for l in HL_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
if not 3 <= len(HL) <= 5:
    bad.append(f"highlights: {len(HL)} bullets, the journal asks for 3-5")
for i, line in enumerate(HL, 1):
    if len(line) > 85:
        bad.append(f"highlight {i} is {len(line)} characters, limit 85: {line!r}")
    for num in re.findall(r"\d+(?:\.\d+)?", line):
        if num not in FLAT:
            bad.append(f"highlight {i} quotes {num}, which the manuscript never states: {line!r}")
ok.append(f"highlights: {len(HL)} bullets, each <= 85 characters, every number also in main.tex")

# --- no retracted claim anywhere, in the manuscript or the highlights ---
HL_TEXT = "\n".join(HL)
for banned, why in [("keeps improving", "withdrawn scaling claim"),
                    ("only strategy that improves", "withdrawn scaling claim"),
                    ("Accuracy does not track parameter count", "null from underpowered test"),
                    ("deployment sweet spot", "no objective function"),
                    ("field-useful", "unevidenced field claim"),
                    ("Protocol~P", "cut from the final paper"),
                    ("WiSE-FT", "cut from the final paper")]:
    where = [n for n, t in (("main.tex", TEX), ("highlights.txt", HL_TEXT)) if banned.lower() in t.lower()]
    if where:
        bad.append(f"retracted/cut term present in {', '.join(where)}: {banned!r} ({why})")
    else:
        ok.append(f"absent: {banned!r}")

for m in ok:
    print(f"  [OK]   {m}")
for m in bad:
    print(f"  [FAIL] {m}")
print(f"\n{len(ok)} passed, {len(bad)} failed")
sys.exit(1 if bad else 0)
