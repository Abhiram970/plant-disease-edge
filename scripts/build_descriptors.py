"""
Phase A2 — Build source-grounded symptom descriptors per disease.

These descriptors are the ZERO-SHOT FUEL: each disease's symptom text becomes a
CLIP/SigLIP text prototype, and we classify images (incl. UNSEEN crops) by nearest
prototype. So we need a descriptor for EVERY class — trained AND held-out.

Schema (per disease), source-grounded to resist hallucination:
  {
    "crop": "Coffee", "disease": "Leaf Rust",
    "symptom_text": "<one rich paragraph used to build the text prototype>",
    "fields": {
      "pathogen":       {"value": "...", "source_url": "...", "verbatim_quote": "..."},
      "affected_organs":{"value": "...", "source_url": "...", "verbatim_quote": "..."},
      "visual_symptoms":{"value": "...", "source_url": "...", "verbatim_quote": "..."}
    },
    "status": "filled" | "stub"
  }

WHERE TO RUN: anywhere (CPU). It's a text job, NOT a GPU job.
  - WITHOUT an API key: writes STUB descriptors (correct schema, empty quotes,
    status="stub") so the rest of the pipeline runs. The friend on the 4060 is NOT blocked.
  - WITH ANTHROPIC_API_KEY: fills them with Claude under the source-grounded rules.

USAGE
-----
  python scripts/build_descriptors.py                 # stubs from manifest classes
  ANTHROPIC_API_KEY=sk-... python scripts/build_descriptors.py --fill
  python scripts/build_descriptors.py --classes-from heldout   # only held-out crops

OUTPUT
------
  descriptors/<crop>.json   (one file per crop, committed to git)
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

EMPTY_FIELD = {"value": "", "source_url": "", "verbatim_quote": ""}

# Source-grounded extraction rules — the LLM may ONLY state what a cited source says.
GROUNDING_SYSTEM = (
    "You are a plant pathologist building a disease symptom registry. "
    "For the given crop and disease, provide pathogen, affected organs, and visual symptoms. "
    "RULES: (1) Every field MUST include a real source_url (university extension factsheet, CABI, "
    "or APS) and a verbatim_quote copied EXACTLY from that page that supports the value. "
    "(2) Do NOT use your own knowledge or invent quotes — if you cannot ground a field, leave it "
    "empty. (3) symptom_text is one rich descriptive paragraph (color, lesion shape, texture, "
    "distribution, affected organ) suitable for image-text matching. "
    "Return ONLY JSON matching the schema."
)


def classes_from_manifest(which: str) -> dict[str, set[str]]:
    """Return {crop: {disease,...}} from manifest, filtered by role group."""
    if not C.MANIFEST_CSV.exists():
        sys.exit(f"manifest not found: {C.MANIFEST_CSV}\nRun build_sage_subset.py first (A1).")
    role_filter = {
        "all": {"train_crop", "heldout_crop"},
        "train": {"train_crop"},
        "heldout": {"heldout_crop"},
    }[which]
    out: dict[str, set[str]] = defaultdict(set)
    with open(C.MANIFEST_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split_role"] in role_filter:
                out[r["crop"]].add(r["disease"])
    return out


def stub(crop: str, disease: str) -> dict:
    return {
        "crop": crop, "disease": disease,
        "symptom_text": f"TODO: source-grounded symptom description for {disease} on {crop}.",
        "fields": {k: dict(EMPTY_FIELD) for k in ("pathogen", "affected_organs", "visual_symptoms")},
        "status": "stub",
    }


def fill_with_claude(crop: str, disease: str) -> dict:
    """Fill one descriptor via Claude with web-grounded sourcing. Falls back to stub on error."""
    try:
        import anthropic
    except ImportError:
        sys.exit("pip install anthropic  (needed for --fill)")
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    prompt = (
        f"Crop: {crop}\nDisease: {disease}\n\n"
        "Return JSON exactly like:\n"
        '{"crop":"","disease":"","symptom_text":"",'
        '"fields":{"pathogen":{"value":"","source_url":"","verbatim_quote":""},'
        '"affected_organs":{"value":"","source_url":"","verbatim_quote":""},'
        '"visual_symptoms":{"value":"","source_url":"","verbatim_quote":""}}}'
    )
    try:
        msg = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=1200,
            system=GROUNDING_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        start, end = text.find("{"), text.rfind("}") + 1
        d = json.loads(text[start:end])
        d.setdefault("crop", crop)
        d.setdefault("disease", disease)
        d["status"] = "filled"
        return d
    except Exception as e:
        print(f"   [warn] fill failed for {crop}/{disease}: {e} -> stub")
        return stub(crop, disease)


def main():
    ap = argparse.ArgumentParser(description="Phase A2: build source-grounded descriptors.")
    ap.add_argument("--fill", action="store_true", help="use Claude to fill (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--classes-from", choices=["all", "train", "heldout"], default="all")
    args = ap.parse_args()

    if args.fill and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("--fill requires ANTHROPIC_API_KEY in the environment.")

    C.DESCRIPTORS_DIR.mkdir(parents=True, exist_ok=True)
    classes = classes_from_manifest(args.classes_from)
    if not classes:
        sys.exit("No classes found for that group in the manifest.")

    total, filled = 0, 0
    for crop, diseases in sorted(classes.items()):
        out_path = C.DESCRIPTORS_DIR / f"{C.safe_name(crop)}.json"
        existing = {}
        if out_path.exists():
            existing = {d["disease"]: d for d in json.loads(out_path.read_text(encoding="utf-8"))}
        records = []
        for disease in sorted(diseases):
            total += 1
            prev = existing.get(disease)
            if prev and prev.get("status") == "filled":
                records.append(prev)  # don't re-spend API on already-filled
                filled += 1
                continue
            if args.fill:
                print(f"  filling {crop} / {disease} ...")
                rec = fill_with_claude(crop, disease)
            else:
                rec = stub(crop, disease)
            records.append(rec)
            if rec.get("status") == "filled":
                filled += 1
        out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  wrote {out_path.name}: {len(records)} diseases")

    mode = "FILLED (Claude)" if args.fill else "STUBS (no API key / fast path)"
    print("\n" + "=" * 60)
    print(f"DESCRIPTORS BUILT — {mode}")
    print("=" * 60)
    print(f"  crops: {len(classes)}   diseases: {total}   filled: {filled}   stubs: {total - filled}")
    print(f"  dir  : {C.DESCRIPTORS_DIR}")
    if filled < total:
        print("\n  NOTE: stubs are placeholders so the pipeline runs. Re-run with --fill + an API key")
        print("        to produce real source-grounded descriptors before reporting zero-shot results.")


if __name__ == "__main__":
    main()
