"""
Cross-check the manuscript's headline numbers against the result JSONs.

WHY: the tables are generated from the JSONs but the prose is not. main.tex silently kept 82.6% for
the seen probe and 88.4% for the best CNN after both were re-measured — the kind of drift that
survives every proofread because the sentence still reads fine. This recomputes each headline from
the JSONs and greps for contradicting values.

    python scripts/check_manuscript_numbers.py        # exit 1 if any manuscript disagrees
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

DOC = C.REPO_ROOT / "docs" / "paper"
# tex/main.tex is the single manuscript. The Markdown rendering (paper.md) was deleted in v2:
# maintaining two renderings of the same prose is exactly the drift this checker exists to catch,
# and the Markdown copy was already carrying four superseded values.
MANUSCRIPTS = [DOC / "tex" / "main.tex"]


def load(name):
    p = DOC / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def facts():
    """(label, correct_value_pct, [stale_values_that_must_not_appear])"""
    out = []

    probe = load("probe_seen_C.json")
    if probe:
        s0 = probe["models"].get("MobileCLIP2-S0", {}).get("seen_probe_top1")
        best = max(m["seen_probe_top1"] for m in probe["models"].values())
        out.append(("seen probe, MobileCLIP2-S0 @ config C", s0 * 100, [82.6, 82.8, 67.0]))
        out.append(("seen probe, best encoder @ config C", best * 100, []))

    cnn = []
    for f in sorted(DOC.glob("supervised_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("seen_top1"):
            cnn.append((d["arch"], d["seen_top1"] * 100, d.get("params_M")))
    if cnn:
        best_arch, best_acc, _ = max(cnn, key=lambda r: r[1])
        out.append((f"best supervised CNN ({best_arch})", best_acc, [88.4, 84.1]))

    w = load("run_all_exp3_lw11_full.json")
    if w:
        half = next((s for s in w["sweep"] if s["alpha"] == 0.5), None)
        if half:
            out.append(("WiSE-FT alpha=0.5 seen", half["seen"] * 100, []))

    edge = load("edge_quant_benchmark.json")
    if edge:
        for _k, d in edge["models"].items():
            if d.get("img_params_M", 0) < 12:
                ms = d["variants"]["onnx_fp32"]["p50_ms"]
                mb = d["variants"]["onnx_int8_static"]["size_mb"]
                out.append(("S0 ONNX FP32 latency (ms)", ms, [21.5, 15.8]))
                out.append(("S0 INT8 size (MB)", mb, []))
                break
    return out


def main():
    # Manuscript lines contain en-dashes and arrows. Printing one to a cp1252 Windows console
    # raised UnicodeEncodeError and killed this checker mid-audit, so it could report nothing while
    # having examined almost nothing. A guard that dies silently is worse than no guard, so force a
    # lossy-but-surviving stdout.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    """Prints every match with its line, because a value that is stale in one sentence is often
    legitimate in another — 82.6% is wrong for the S0 probe but right for MobileCLIP-B and for the
    WiSE-FT alpha=0 row. This flags candidates for a human to adjudicate; it does not auto-fix."""
    hits = 0
    for label, val, stale in facts():
        print(f"\n### {label} = {val:.1f}%")
        if not stale:
            print("    (no known-stale predecessors tracked)")
            continue
        for m in MANUSCRIPTS:
            if not m.exists():
                continue
            for i, line in enumerate(m.read_text(encoding="utf-8").splitlines(), 1):
                for s in stale:
                    if re.search(rf"(?<!\d){s:.1f}\s*\\?%", line):
                        hits += 1
                        snippet = line.strip()[:110]
                        print(f"    [{m.name}:{i}] {s:.1f}% -> {snippet}")

    print()
    if hits:
        print(f"[REVIEW] {hits} occurrence(s) of a superseded value. Each needs a human decision: "
              f"the same number can be stale in the abstract and correct in a table row.")
        return 1
    print("[OK] no superseded headline values found in the manuscript.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
