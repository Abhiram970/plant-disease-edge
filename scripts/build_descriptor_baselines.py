#!/usr/bin/env python3
"""Generate the two established descriptor-generation baselines: DCLIP and CuPL.

Section 6 concedes the paper has no comparison against the descriptor-authoring methods it is
positioned against. This closes that gap. Both are TEXT-ONLY: no new images, no training, so
scoring them reuses the cached image embeddings and costs minutes of GPU time.

  dclip   Menon and Vondrick, "Visual Classification via Description from Large Language
          Models", ICLR 2023. Their prompt asks what visual features distinguish the category;
          the answer is a list of short descriptor phrases. Scoring is the mean of per-descriptor
          similarities (see the fidelity note in descriptors.py -- this is why `dclip` does not
          re-normalise its class prototype).

  cupl    Pratt et al., "What does a platypus look like? Generating Customized Prompts for
          Zero-shot Image Classification", ICCV 2023. Several question templates per class, each
          answered with full sentences; the sentence embeddings are averaged and no hand-written
          template is applied afterwards.

Faithfulness and its limits, stated plainly because the paper must not overclaim a second time:
both methods were written for ImageNet-style object categories. The prompts below are their
published question forms with the category slot filled by "<disease> on <crop> leaf". Nothing
else is adapted -- no agricultural hints, no symptom vocabulary, no schema. Descriptor COUNTS
are left to the model, as in both papers. What this yields is a faithful port, not a
re-tuning; if these baselines win, the paper's authoring claim does not survive, and that is
the point of running them.

Same on-disk shape as the ungrounded arm (one JSON per crop under a per-seed directory), so the
seed plumbing, the integrity gate and descriptors.texts_for all work unchanged.

    export LAVA_API_KEY=...            # or ANTHROPIC_API_KEY
    python scripts/build_descriptor_baselines.py --method dclip --seed 0
    python scripts/build_descriptor_baselines.py --method cupl  --seed 0
    python scripts/build_descriptor_baselines.py --method both  --seed 0 --which all

Re-running is a no-op for classes that already hold real text, so an interrupted run resumes.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import build_descriptors as BD
import descriptors as D

DEFAULT_MODEL = "claude-sonnet-5"
MAX_TOKENS = int(os.environ.get("PDE_MAX_TOKENS", "1200"))

# --- DCLIP -------------------------------------------------------------------------------------
# Menon and Vondrick's prompt, verbatim in form. Their paper asks:
#     "What are useful visual features for distinguishing a {category} in a photo?"
# and shows the answer as a bulleted list of short phrases. We ask for JSON so the list parses
# deterministically; the question itself is unchanged.
DCLIP_SYSTEM = (
    "You answer questions about the visual appearance of categories, for use as text prompts in "
    "an image classifier. Reply with ONLY a JSON array of short strings, no prose, no keys."
)
DCLIP_USER = (
    "Q: What are useful visual features for distinguishing a {category} in a photo?\n"
    "A: There are several useful visual features to tell there is a {category} in a photo. "
    "List them as short noun phrases, each one a single visible feature."
)

# --- CuPL --------------------------------------------------------------------------------------
# Pratt et al. use a handful of question templates per class and pool the generated sentences.
# These are their published forms with the category slot filled.
CUPL_SYSTEM = (
    "You describe what things look like in photographs, for use as text prompts in an image "
    "classifier. Reply with ONLY a JSON array of complete sentences, no prose, no keys."
)
CUPL_QUESTIONS = [
    "Describe what a {category} looks like in a photo.",
    "Describe a photo of a {category}.",
    "What does a {category} look like in a photo?",
    "Describe the appearance of a {category} in a photo.",
]
CUPL_PER_QUESTION = 3        # Pratt et al. pool ~10 sentences per class; 4 x 3 = 12


def category_of(crop: str, disease: str) -> str:
    """The class name as both methods see it. Identical to the `bare` strategy's text, so the
    baselines are asked about exactly the class the paper's own bare row describes."""
    return f"{disease} on {crop} leaf".replace("_", " ")


def _parse_list(raw: str) -> list[str]:
    """Pull a JSON array of strings out of a model reply, tolerating fences and stray prose."""
    if not raw:
        return []
    s = raw.strip()
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.M).strip()
    start, end = s.find("["), s.rfind("]")
    if start != -1 and end > start:
        try:
            items = json.loads(s[start:end + 1])
            if isinstance(items, list):
                return [str(x).strip() for x in items if str(x).strip()]
        except json.JSONDecodeError:
            pass
    # Fall back to bullet/numbered lines rather than losing a paid call.
    out = []
    for line in s.splitlines():
        line = line.strip().lstrip("-*•").strip()
        line = re.sub(r"^\d+[.)]\s*", "", line)
        line = line.strip().strip('",')
        if len(line) > 3 and not line.lower().startswith(("q:", "a:", "here", "sure")):
            out.append(line)
    return out


# Provider lock. "lava" is the default and the only one this project has credits for; the
# direct-Anthropic path is kept reachable only by an explicit PDE_PROVIDER=anthropic, so a
# missing or misspelled LAVA_API_KEY can never silently fall through to an endpoint the
# operator does not have a key for and fail 401 halfway through a paid run.
PROVIDER = (os.environ.get("PDE_PROVIDER") or "lava").strip().lower()

# Live token accounting. An estimate is worth less than a meter: this records what the provider
# actually billed, so a run can be stopped early if the burn rate is not what was projected.
USAGE = {"calls": 0, "in": 0, "out": 0, "unreported": 0}


def _note_usage(u) -> None:
    """Record one response's usage. Field names differ by SDK, so try both shapes."""
    if u is None:
        USAGE["unreported"] += 1
        return
    i = getattr(u, "prompt_tokens", None)
    if i is None:
        i = getattr(u, "input_tokens", None)
    o = getattr(u, "completion_tokens", None)
    if o is None:
        o = getattr(u, "output_tokens", None)
    if i is None and o is None:
        USAGE["unreported"] += 1
        return
    USAGE["in"] += int(i or 0)
    USAGE["out"] += int(o or 0)


def usage_line() -> str:
    u = USAGE
    tot = u["in"] + u["out"]
    s = (f"{u['calls']} calls, {u['in']:,} in + {u['out']:,} out = {tot:,} tokens")
    if u["unreported"]:
        s += f" ({u['unreported']} call(s) reported no usage)"
    return s


def _ask(system: str, user: str, seed: int) -> str:
    """One completion, routed by PROVIDER. Key handling is shared with build_ungrounded."""
    model = os.environ.get("PDE_BASELINE_MODEL", DEFAULT_MODEL)
    # Sampling must be stochastic for a seed to mean anything, exactly as in the control arm.
    kw = dict(model=model, max_tokens=MAX_TOKENS, temperature=1.0)

    if PROVIDER == "lava":
        if not (os.environ.get("LAVA_API_KEY") or "").strip():
            sys.exit("[fatal] PDE_PROVIDER=lava but LAVA_API_KEY is empty or unset. Set it, or "
                     "set PDE_PROVIDER=anthropic to use ANTHROPIC_API_KEY instead. Refusing to "
                     "fall back to an endpoint you may have no key for.")
        # A Lava key is locked to ONE request shape and rejects the other with
        # 403 request_shape_not_allowed. LAVA_SHAPE selects it; openai is Lava's default.
        if not BD.lava_is_openai_shape():
            client = BD.anthropic_client()          # anthropic SDK pointed at Lava's base URL
            USAGE["calls"] += 1
            r = client.messages.create(system=system,
                                       messages=[{"role": "user", "content": user}], **kw)
            _note_usage(getattr(r, "usage", None))
            return "".join(getattr(b, "text", "") for b in r.content)
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["LAVA_API_KEY"],
                        base_url=os.environ.get("LAVA_BASE_URL", "https://api.lava.build/v1"))
        USAGE["calls"] += 1
        r = client.chat.completions.create(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            model=os.environ.get("LAVA_MODEL", "anthropic/claude-sonnet-4-6"),
            max_tokens=MAX_TOKENS, temperature=1.0, seed=seed)
        _note_usage(getattr(r, "usage", None))
        return r.choices[0].message.content or ""

    if not (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
        sys.exit("[fatal] PDE_PROVIDER=anthropic but ANTHROPIC_API_KEY is empty or unset.")
    client = BD.anthropic_client()
    USAGE["calls"] += 1
    r = client.messages.create(system=system, messages=[{"role": "user", "content": user}], **kw)
    _note_usage(getattr(r, "usage", None))
    return "".join(getattr(b, "text", "") for b in r.content)


def fill_dclip(crop: str, disease: str, seed: int) -> dict:
    cat = category_of(crop, disease)
    try:
        items = _parse_list(_ask(DCLIP_SYSTEM, DCLIP_USER.format(category=cat), seed))
    except Exception as e:
        print(f"   [warn] dclip {crop}/{disease}: {e} -> stub", flush=True)
        items = []
    return {"crop": crop, "disease": disease, "category": cat, "method": "dclip",
            "model": os.environ.get("PDE_BASELINE_MODEL", DEFAULT_MODEL), "seed": seed,
            "status": "filled" if items else "stub", "descriptors": items}


def fill_cupl(crop: str, disease: str, seed: int) -> dict:
    cat = category_of(crop, disease)
    sents: list[str] = []
    for q in CUPL_QUESTIONS:
        user = (f"{q.format(category=cat)}\n"
                f"Give {CUPL_PER_QUESTION} different complete sentences.")
        try:
            sents += _parse_list(_ask(CUPL_SYSTEM, user, seed))
        except Exception as e:
            print(f"   [warn] cupl {crop}/{disease} ({q[:24]}...): {e}", flush=True)
    seen, uniq = set(), []
    for s in sents:
        k = s.lower()
        if k not in seen:
            seen.add(k); uniq.append(s)
    return {"crop": crop, "disease": disease, "category": cat, "method": "cupl",
            "model": os.environ.get("PDE_BASELINE_MODEL", DEFAULT_MODEL), "seed": seed,
            "status": "filled" if uniq else "stub", "sentences": uniq}


FILLERS = {"dclip": (fill_dclip, "descriptors"), "cupl": (fill_cupl, "sentences")}


def run_method(method: str, seed: int, which: str, limit: int) -> int:
    fill, field = FILLERS[method]
    out_dir = C.REPO_ROOT / D.BASELINE_DIRS[method] / str(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_crop = BD.classes_from_manifest(which)
    total = sum(len(v) for v in by_crop.values())
    print(f"\n[{method}] seed {seed}: {total} classes over {len(by_crop)} crops -> {out_dir}")

    n = 0
    for crop in sorted(by_crop):
        path = out_dir / f"{C.safe_name(crop)}.json"
        recs = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        # Key resume on real content, not on the status flag. The matched grounded arm once wrote
        # 33 records flagged "filled" with empty text, and every later run skipped them.
        have = {r.get("disease") for r in recs
                if r.get("status") == "filled" and [x for x in (r.get(field) or []) if str(x).strip()]}
        for disease in sorted(by_crop[crop]):
            if limit and n >= limit:
                break
            if disease in have:
                continue
            rec = fill(crop, disease, seed)
            recs = [r for r in recs if r.get("disease") != disease] + [rec]
            path.write_text(json.dumps(recs, indent=2, ensure_ascii=False), encoding="utf-8")
            n += 1
            k = len(rec.get(field) or [])
            print(f"   {crop}/{disease}: {rec['status']}, {k} {field}", flush=True)
            time.sleep(0.4)      # courtesy rate limit, same as the other generators

    filled = 0
    for p in out_dir.glob("*.json"):
        for r in json.loads(p.read_text(encoding="utf-8")):
            if r.get("status") == "filled" and [x for x in (r.get(field) or []) if str(x).strip()]:
                filled += 1
    print(f"[{method}] seed {seed}: {filled} filled records in {out_dir}")
    print(f"[usage] cumulative: {usage_line()}")
    return filled


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="both", choices=["dclip", "cupl", "both"])
    ap.add_argument("--seed", type=int, default=0, help="one directory per seed")
    ap.add_argument("--which", default="heldout", choices=["all", "train", "heldout"],
                    help="heldout is what the cross-crop comparison needs and is ~4x cheaper")
    ap.add_argument("--limit", type=int, default=0, help="stop after N classes (smoke test)")
    ap.add_argument("--estimate", action="store_true",
                    help="print the call/token projection and exit WITHOUT calling the API")
    ap.add_argument("--seeds", type=int, default=1,
                    help="how many seeds the projection assumes (--estimate only)")
    args = ap.parse_args()

    if args.estimate:
        by_crop = BD.classes_from_manifest(args.which)
        n_cls = sum(len(v) for v in by_crop.values())
        if args.limit:
            n_cls = min(n_cls, args.limit)
        methods = ["dclip", "cupl"] if args.method == "both" else [args.method]
        # 4 chars/token is the usual English heuristic; treat the result as +/-20%.
        cat = "Cigar End Rot on Banana leaf"
        per = {
            "dclip": (1, (len(DCLIP_SYSTEM) + len(DCLIP_USER.format(category=cat))) / 4, 110),
            "cupl": (len(CUPL_QUESTIONS),
                     (len(CUPL_SYSTEM) + len(CUPL_QUESTIONS[0].format(category=cat)) + 40) / 4, 90),
        }
        tot_calls = tot_in = tot_out = 0
        print(f"projection: {n_cls} classes x {args.seeds} seed(s), which={args.which}\n")
        print(f"{'arm':6s} {'calls':>7s} {'in/call':>9s} {'out/call':>9s} {'tokens':>11s}")
        for m in methods:
            k, i, o = per[m]
            calls = n_cls * k * args.seeds
            print(f"{m:6s} {calls:7d} {i:9.0f} {o:9.0f} {calls * (i + o):11,.0f}")
            tot_calls += calls; tot_in += calls * i; tot_out += calls * o
        print(f"{'TOTAL':6s} {tot_calls:7d} {'':9s} {'':9s} {tot_in + tot_out:11,.0f}")
        print(f"\n  {tot_calls} calls, ~{tot_in:,.0f} input + ~{tot_out:,.0f} output "
              f"= ~{tot_in + tot_out:,.0f} tokens (~{(tot_in + tot_out)/1e6:.3f} M)")
        print(f"  output is capped at PDE_MAX_TOKENS={MAX_TOKENS}/call, so the worst case is "
              f"{tot_calls * MAX_TOKENS:,} output tokens; measured runs land far below it.")
        print("  multiply by your provider's per-token rate for the credit cost.")
        return 0

    need = "LAVA_API_KEY" if PROVIDER == "lava" else "ANTHROPIC_API_KEY"
    if not (os.environ.get(need) or "").strip():
        sys.exit(f"Set {need} (PDE_PROVIDER={PROVIDER}). Generation needs an LLM and cannot be "
                 f"reproduced offline, which is why the generated registries are released.")
    if PROVIDER == "lava":
        shape = "openai" if BD.lava_is_openai_shape() else "anthropic"
        base = (os.environ.get("LAVA_BASE_URL") or "https://api.lava.build/v1"
                if shape == "openai" else os.environ.get("LAVA_BASE_URL") or "https://api.lava.so")
        print(f"[provider] lava, {shape} shape, model "
              f"{os.environ.get('LAVA_MODEL', 'anthropic/claude-sonnet-4-6') if shape == 'openai' else os.environ.get('PDE_BASELINE_MODEL', DEFAULT_MODEL)}, "
              f"base {base}")
        print("[provider] if this 403s with request_shape_not_allowed, set LAVA_SHAPE=anthropic")

    methods = ["dclip", "cupl"] if args.method == "both" else [args.method]
    counts = {m: run_method(m, args.seed, args.which, args.limit) for m in methods}
    print("\n" + "=" * 70)
    for m, k in counts.items():
        print(f"  {m:6s} seed {args.seed}: {k} filled")
    print(f"  TOKENS BILLED: {usage_line()}")
    print("Next: score them with\n"
          "  python scripts/evaluate.py --exp C --strategies bare80 dclip cupl --heavy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
