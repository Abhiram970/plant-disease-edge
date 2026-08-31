"""
Generate UNGROUNDED descriptors — the control arm for the grounding claim.

WHY THIS EXISTS
---------------
Every published comparison pits `grounded` against `rich`, a keyword-retrieved bank in which most
held-out classes share text with another class. Any advantage `grounded` shows is therefore
attributable to per-class distinctness, not to sourcing. This arm removes that confound: same model,
same schema, same one-paragraph symptom_text, same fall-through behaviour — the ONLY difference is
that the "cite a retrievable source, never use your own knowledge" constraint is dropped.

Interpreting the result:
  ungrounded ~= grounded  -> grounding costs nothing and buys auditability. Clean, defensible.
  ungrounded <  grounded  -> the sourcing constraint itself improves the text. A real finding.
  ungrounded >  grounded  -> grounding costs accuracy; the paper must say so.

SEEDS: LLM text varies run to run, and a single sample cannot separate "grounding helps" from "this
generation was lucky". Generate at least three seeds (--seed 0,1,2) and report mean and spread. The
generation is sampled at non-zero temperature precisely so seeds differ; the grounded registry was
generated at temperature 0.

    ANTHROPIC_API_KEY=... python scripts/build_ungrounded.py --seed 0 --which heldout
    LAVA_API_KEY=...      python scripts/build_ungrounded.py --seed 1 --which heldout

Writes descriptors_ungrounded/<seed>/<Crop>.json in the same schema the grounded loader reads.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import build_descriptors as BD

# Deliberately parallel to BD.GROUNDING_SYSTEM, minus the sourcing rules. Everything else --- the
# role, the field set, and the symptom_text specification --- is word-for-word identical, so the two
# arms differ in exactly one respect.
UNGROUNDED_SYSTEM = (
    "You are a plant pathologist building a disease symptom registry. "
    "For the given crop and disease, provide pathogen, affected organs, and visual symptoms. "
    "RULES: (1) Answer from your own knowledge; you do NOT need to cite sources, and source_url and "
    "verbatim_quote may be left empty. (2) symptom_text is one rich descriptive paragraph (color, "
    "lesion shape, texture, distribution, affected organ) suitable for image-text matching. "
    "Return ONLY JSON matching the schema."
)


DEFAULT_MODEL = "claude-sonnet-5"


def model_name() -> str:
    return os.environ.get("PDE_LLM_MODEL", DEFAULT_MODEL)


def fill_one(crop: str, disease: str, seed: int, arm: str = "ungrounded") -> dict:
    """One descriptor for `arm`. Temperature is non-zero so seeds actually differ.

    `arm` selects only the system prompt. Everything else -- model, schema, temperature, seed,
    parsing, fall-through -- is identical, which is the whole point: the two arms must differ in
    exactly one respect or the comparison measures something other than grounding."""
    system = BD.GROUNDING_SYSTEM if arm == "grounded" else UNGROUNDED_SYSTEM
    model = model_name()
    lava = os.environ.get("LAVA_API_KEY", "")
    prompt = BD._user_prompt(crop, disease)
    if lava:
        from openai import OpenAI
        client = OpenAI(api_key=lava,
                        base_url=os.environ.get("LAVA_BASE_URL", "https://api.lava.so/v1"))
        resp = client.chat.completions.create(
            model=model, max_tokens=1200, temperature=1.0, seed=seed,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
        )
        rec = BD._parse_descriptor(resp.choices[0].message.content, crop, disease)
    else:
        client = BD.anthropic_client()   # handles workspace-scoped keys; see build_descriptors
        msg = client.messages.create(
            model=model, max_tokens=1200, temperature=1.0, system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        rec = BD._parse_descriptor(msg.content[0].text, crop, disease)
    # Provenance, stamped per record. The shipped grounded registry carries none, so there is no way
    # to prove which model wrote it -- which is exactly why this arm regenerates its own matched
    # grounded set instead of comparing against text of unknown origin.
    if isinstance(rec, dict):
        rec["model"] = model
        rec["arm"] = arm
        rec["seed"] = seed
        rec["temperature"] = 1.0
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, help="0, 1, 2 ... one directory per seed")
    ap.add_argument("--which", default="heldout", choices=["all", "train", "heldout"],
                    help="heldout is enough for the zero-shot comparison and is ~4x cheaper")
    ap.add_argument("--limit", type=int, default=0, help="stop after N classes (smoke test)")
    ap.add_argument("--arm", default="ungrounded", choices=["ungrounded", "grounded"],
                    help="ungrounded = sourcing constraint removed. grounded = the SAME model and "
                         "schema WITH the constraint, written to descriptors_grounded_matched/. "
                         "Generate both from one run: the shipped registry records no model, so "
                         "comparing against it would confound grounding with model version.")
    args = ap.parse_args()

    if not (os.environ.get("LAVA_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")):
        sys.exit("Set LAVA_API_KEY or ANTHROPIC_API_KEY. This step needs an LLM; it cannot be "
                 "reproduced offline, which is why the generated descriptors are released.")

    root = "descriptors_ungrounded" if args.arm == "ungrounded" else "descriptors_grounded_matched"
    out_dir = C.REPO_ROOT / root / str(args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_crop = BD.classes_from_manifest(args.which)
    total = sum(len(v) for v in by_crop.values())
    print(f"[{args.arm}] seed {args.seed}, model {model_name()}: {total} classes over "
          f"{len(by_crop)} crops -> {out_dir}")

    n = 0
    for crop in sorted(by_crop):
        path = out_dir / f"{C.safe_name(crop)}.json"
        recs = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        have = {r.get("disease") for r in recs if r.get("status") == "filled"}
        for disease in sorted(by_crop[crop]):
            if args.limit and n >= args.limit:
                break
            if disease in have:
                continue
            try:
                rec = fill_one(crop, disease, args.seed, args.arm)
            except Exception as e:
                print(f"  [fail] {crop}/{disease}: {type(e).__name__}: {str(e)[:70]}")
                rec = BD.stub(crop, disease)
            recs = [r for r in recs if r.get("disease") != disease] + [rec]
            path.write_text(json.dumps(recs, indent=2), encoding="utf-8")
            n += 1
            status = "ok" if rec.get("status") == "filled" else "STUB"
            print(f"  [{n:3d}/{total}] {crop}/{disease}: {status}")
            time.sleep(0.4)          # be gentle with the endpoint

    filled = sum(1 for p in out_dir.glob("*.json")
                 for r in json.loads(p.read_text(encoding="utf-8"))
                 if r.get("status") == "filled")
    print(f"\n[ungrounded] seed {args.seed}: {filled} filled records in {out_dir}")
    print("Evaluate with:  python scripts/evaluate.py --exp C --strategies rich grounded ungrounded "
          f"--ungrounded-seed {args.seed} --heavy")


if __name__ == "__main__":
    main()
