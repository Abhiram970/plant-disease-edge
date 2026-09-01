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

try:                                   # load .env (LAVA_API_KEY, etc.) if python-dotenv is present
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

MAX_TOKENS = int(os.environ.get("PDE_MAX_TOKENS", "2000"))

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


def _user_prompt(crop: str, disease: str) -> str:
    return (
        f"Crop: {crop}\nDisease: {disease}\n\n"
        "Return JSON exactly like:\n"
        '{"crop":"","disease":"","symptom_text":"",'
        '"fields":{"pathogen":{"value":"","source_url":"","verbatim_quote":""},'
        '"affected_organs":{"value":"","source_url":"","verbatim_quote":""},'
        '"visual_symptoms":{"value":"","source_url":"","verbatim_quote":""}}}'
    )


def _repair_json(s: str) -> str:
    """Escape raw control characters and stray quotes inside JSON string values.

    The GROUNDED schema asks for a verbatim_quote copied EXACTLY from a source page, and exact
    page text routinely contains a double quote or a line break. Emitted unescaped, those
    terminate the JSON string early and the object fails to parse -- which is why the grounded
    arm stubbed 8 of 51 classes per seed while the ungrounded arm, whose quotes may be empty,
    filled 51/51. Walk the text tracking whether we are inside a string, and escape what would
    otherwise break it."""
    out, in_str, esc = [], False, False
    for i, ch in enumerate(s):
        if esc:
            out.append(ch)
            esc = False
            continue
        if ch == "\\":
            out.append(ch)
            esc = True
            continue
        if ch == '"':
            if not in_str:
                in_str = True
                out.append(ch)
            else:
                # closing quote only if the next non-space char can legally follow one
                nxt = next((c for c in s[i + 1:] if not c.isspace()), "")
                if nxt in ",:}]" or nxt == "":
                    in_str = False
                    out.append(ch)
                else:
                    out.append('\\"')          # a quote INSIDE the value
            continue
        if in_str and ch in "\n\r\t":
            out.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}[ch])
            continue
        out.append(ch)
    return "".join(out)


def _parse_descriptor(text: str, crop: str, disease: str) -> dict:
    """Extract the JSON object from an LLM reply (tolerating ```json fences)."""
    text = (text or "").strip()
    if not text:
        raise ValueError("empty response body from the API")
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError(f"no JSON object in a {len(text)}-char response")
    raw = text[start:end]
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        d = json.loads(_repair_json(raw))      # unescaped quote/newline inside a verbatim_quote
    d["crop"] = crop            # force to the manifest's names (the model sometimes renames the disease),
    d["disease"] = disease      # so the grounded lookup keys by folder name always match
    d["status"] = "filled"
    return d


def fill_with_lava(crop: str, disease: str) -> dict:
    """Fill one descriptor via Claude through Lava's OpenAI-compatible endpoint (spend key).
    NOTE: a plain chat call cannot browse — source_url/verbatim_quote are model-recalled and MUST
    be human spot-audited (see audit_descriptors.py). symptom_text is the functional prototype text."""
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("pip install openai python-dotenv  (needed for --provider lava)")
    key = os.environ.get("LAVA_API_KEY", "")
    if not key:
        sys.exit("Set LAVA_API_KEY (your Lava spend key) in .env or the environment.")
    client = OpenAI(api_key=key, base_url=os.environ.get("LAVA_BASE_URL", "https://api.lava.so/v1"))
    model = os.environ.get("LAVA_MODEL", "anthropic/claude-sonnet-4-6")
    try:
        resp = client.chat.completions.create(
            model=model, max_tokens=MAX_TOKENS, temperature=0.0,
            messages=[{"role": "system", "content": GROUNDING_SYSTEM},
                      {"role": "user", "content": _user_prompt(crop, disease)}],
        )
        return _parse_descriptor(resp.choices[0].message.content, crop, disease)
    except Exception as e:
        print(f"   [warn] lava fill failed for {crop}/{disease}: {e} -> stub")
        return stub(crop, disease)


def anthropic_client():
    """An Anthropic-shape client, pointed at Lava when a Lava key is present.

    Lava issues two kinds of key and they are NOT interchangeable:
      * OpenAI-shape  -- use the openai SDK against <base>/chat/completions
      * Anthropic-shape -- refuses an OpenAI-shaped call outright with
            403 request_shape_not_allowed: "This key only allows anthropic request shape,
            but received openai"
        and instead wants the native Messages API at <base>/messages with x-api-key auth.
    The anthropic SDK speaks exactly that shape, so an Anthropic-shape Lava key is just this
    client with base_url set. Probed against the live endpoint; do not swap the transports.

    A direct Anthropic key that is WORKSPACE-SCOPED additionally needs the workspace id as a
    header (400: "anthropic-workspace-id is required..."). The key is fine, the header is
    missing; set ANTHROPIC_WORKSPACE_ID and it is sent automatically."""
    import anthropic
    kw = {}
    lava = (os.environ.get("LAVA_API_KEY") or "").strip()
    if lava:
        kw["api_key"] = lava
        # The anthropic SDK appends /v1 itself, so the base must NOT already end in /v1 or every
        # call 404s on /v1/v1/messages. LAVA_BASE_URL is written for the OpenAI SDK (which does
        # want the /v1), so strip it here rather than making the operator keep two variables.
        base = os.environ.get("LAVA_BASE_URL", "https://api.lava.so").rstrip("/")
        kw["base_url"] = base[:-3].rstrip("/") if base.endswith("/v1") else base
    ws = (os.environ.get("ANTHROPIC_WORKSPACE_ID") or "").strip()
    if ws and not lava:                       # Lava does not want the workspace header
        kw["default_headers"] = {"anthropic-workspace-id": ws}
    return anthropic.Anthropic(**kw)


def lava_is_openai_shape() -> bool:
    """True when the Lava key speaks the OpenAI protocol (the default).

    A Lava service key is locked to ONE request shape and rejects the other outright:
        403 request_shape_not_allowed -- "This key only allows anthropic request shape,
                                          but received openai"
    Set LAVA_SHAPE=anthropic for such a key; it then goes through anthropic_client(), which is
    the same SDK pointed at Lava's base URL."""
    return (os.environ.get("LAVA_SHAPE") or "openai").strip().lower() == "openai"


def fill_with_claude(crop: str, disease: str) -> dict:
    """Fill one descriptor via the native Anthropic SDK (needs ANTHROPIC_API_KEY)."""
    try:
        client = anthropic_client()   # ANTHROPIC_API_KEY (+ workspace id if the key needs one)
    except ImportError:
        sys.exit("pip install anthropic  (needed for --provider anthropic)")
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
    try:
        msg = client.messages.create(
            model=model, max_tokens=MAX_TOKENS, system=GROUNDING_SYSTEM,
            messages=[{"role": "user", "content": _user_prompt(crop, disease)}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return _parse_descriptor(text, crop, disease)
    except Exception as e:
        print(f"   [warn] anthropic fill failed for {crop}/{disease}: {e} -> stub")
        return stub(crop, disease)


def _usable(rec: dict | None) -> bool:
    """A record counts as a real fill only if it has a substantive symptom_text (the prototype text).
    Empty/degenerate 'filled' records (model didn't know the disease) are treated as NOT filled so
    they retry next run and don't inflate the filled count."""
    if not rec or rec.get("status") != "filled":
        return False
    t = (rec.get("symptom_text") or "").strip()
    return len(t) >= 40 and not t.startswith("TODO")


def fill_one(crop: str, disease: str, provider: str) -> dict:
    rec = fill_with_lava(crop, disease) if provider == "lava" else fill_with_claude(crop, disease)
    if not _usable(rec):
        print(f"   [warn] {crop}/{disease}: empty/short symptom_text -> keeping as stub (will retry)")
        return stub(crop, disease)
    return rec


def main():
    ap = argparse.ArgumentParser(description="Phase A2: build source-grounded descriptors.")
    ap.add_argument("--fill", action="store_true", help="fill via LLM (else write stubs)")
    ap.add_argument("--provider", choices=["auto", "lava", "anthropic"], default="auto",
                    help="auto = Lava if LAVA_API_KEY is set, else Anthropic")
    ap.add_argument("--classes-from", choices=["all", "train", "heldout"], default="all")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap NEW fills this run (0 = no cap) — a spend guard for the $10 budget")
    args = ap.parse_args()

    provider = args.provider
    if args.fill:
        if provider == "auto":
            provider = "lava" if os.environ.get("LAVA_API_KEY") else "anthropic"
        if provider == "lava" and not os.environ.get("LAVA_API_KEY"):
            sys.exit("--provider lava needs LAVA_API_KEY (put it in .env or the environment).")
        if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("--provider anthropic needs ANTHROPIC_API_KEY.")

    C.DESCRIPTORS_DIR.mkdir(parents=True, exist_ok=True)
    classes = classes_from_manifest(args.classes_from)
    if not classes:
        sys.exit("No classes found for that group in the manifest.")
    n_want = sum(len(v) for v in classes.values())
    if args.fill:
        mdl = os.environ.get("LAVA_MODEL", "anthropic/claude-sonnet-4-6") if provider == "lava" \
            else os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
        print(f"[fill] provider={provider}  model={mdl}  candidates={n_want}"
              + (f"  (cap {args.limit} new this run)" if args.limit else "")
              + "  (already-filled are skipped)")

    total, filled, new_fills = 0, 0, 0
    for crop, diseases in sorted(classes.items()):
        out_path = C.DESCRIPTORS_DIR / f"{C.safe_name(crop)}.json"
        existing = {}
        if out_path.exists():
            existing = {d["disease"]: d for d in json.loads(out_path.read_text(encoding="utf-8"))}
        records = []
        for disease in sorted(diseases):
            total += 1
            prev = existing.get(disease)
            if _usable(prev):
                records.append(prev)  # don't re-spend on a good existing fill
                filled += 1
                continue
            if args.fill and (args.limit == 0 or new_fills < args.limit):
                print(f"  filling {crop} / {disease} ...")
                rec = fill_one(crop, disease, provider)
                new_fills += 1
            else:
                rec = stub(crop, disease)
            records.append(rec)
            if rec.get("status") == "filled":
                filled += 1
        out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  wrote {out_path.name}: {len(records)} diseases")

    mode = f"FILLED ({provider})" if args.fill else "STUBS (no LLM / fast path)"
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
