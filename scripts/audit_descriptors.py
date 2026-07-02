"""
Phase A2 audit — spot-check the source-grounded descriptors before trusting them in the paper.

The anti-hallucination claim only holds if the descriptors are actually grounded. This script:
  1. counts filled vs stub per crop, flags any held-out disease that is still a stub (a coverage gap);
  2. validates that every 'filled' field has a non-empty value + source_url + verbatim_quote;
  3. (optional) --check-urls does a HEAD request per unique source_url to catch dead links;
  4. prints a random sample of N filled descriptors for a human to eyeball the quote-vs-value match.

USAGE
  python scripts/audit_descriptors.py                 # offline structural audit + sample
  python scripts/audit_descriptors.py --sample 8 --check-urls
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

FIELDS = ("pathogen", "affected_organs", "visual_symptoms")


def load_all():
    recs = []
    for p in sorted(C.DESCRIPTORS_DIR.glob("*.json")):
        try:
            for d in json.loads(p.read_text(encoding="utf-8")):
                d["_file"] = p.name
                recs.append(d)
        except Exception as e:
            print(f"  [warn] {p.name}: {e}")
    return recs


def field_complete(rec) -> bool:
    f = rec.get("fields", {})
    return all(f.get(k, {}).get("value") and f.get(k, {}).get("source_url") and
               f.get(k, {}).get("verbatim_quote") for k in FIELDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=5)
    ap.add_argument("--check-urls", action="store_true", help="HEAD-request every source_url")
    args = ap.parse_args()

    recs = load_all()
    if not recs:
        sys.exit(f"No descriptors in {C.DESCRIPTORS_DIR} — run build_descriptors.py --fill first.")

    status = Counter(r.get("status", "?") for r in recs)
    by_crop = Counter(r["crop"] for r in recs)
    filled_by_crop = Counter(r["crop"] for r in recs if r.get("status") == "filled")
    n_verified = sum(1 for r in recs if r.get("verified"))
    print(f"[audit] {len(recs)} descriptors across {len(by_crop)} crops")
    print(f"[audit] status: " + ", ".join(f"{k}={v}" for k, v in status.items())
          + f"  |  page-verified citations: {n_verified}")

    # held-out coverage gaps (these MUST be filled — they are the zero-shot fuel)
    held_stubs = [f"{r['crop']}/{r['disease']}" for r in recs
                  if r["crop"] in C.HELDOUT_CROPS and r.get("status") != "filled"]
    print(f"[audit] held-out crops filled: " +
          ", ".join(f"{c}={filled_by_crop[c]}/{by_crop[c]}" for c in C.HELDOUT_CROPS if c in by_crop))
    if held_stubs:
        print(f"[audit] !! {len(held_stubs)} HELD-OUT diseases still stubs (zero-shot gap): "
              f"{held_stubs[:12]}{' ...' if len(held_stubs) > 12 else ''}")
    else:
        print("[audit] OK: every held-out disease has a filled descriptor.")

    # THE real problem: a filled record with an empty/degenerate symptom_text (no prototype text)
    empty = [f"{r['crop']}/{r['disease']}" for r in recs if r.get("status") == "filled"
             and len((r.get("symptom_text") or "").strip()) < 40]
    if empty:
        print(f"[audit] !! {len(empty)} 'filled' records have EMPTY symptom_text (re-run --fill to retry; "
              f"obscure diseases may never ground and will fall back to rich): {empty[:12]}")
    else:
        print("[audit] OK: every filled record has a substantive symptom_text.")

    # partial field grounding is ACCEPTABLE (the prompt tells the model to leave ungroundable fields
    # empty) — report as info, not an error.
    incomplete = sum(1 for r in recs if r.get("status") == "filled" and not field_complete(r))
    print(f"[audit] info: {incomplete}/{len(recs)} records have >=1 ungrounded field "
          f"(acceptable — model left what it couldn't cite empty).")

    if args.check_urls:
        import urllib.request
        import urllib.error
        urls = sorted({r["fields"][k]["source_url"] for r in recs if r.get("status") == "filled"
                       for k in FIELDS if r["fields"].get(k, {}).get("source_url")})
        print(f"\n[audit] checking {len(urls)} unique source URLs (GET, browser agent) ...")
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                 "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
        ok = blocked = 0
        dead = []
        for u in urls:
            try:
                urllib.request.urlopen(urllib.request.Request(u, headers=headers), timeout=12)
                ok += 1
            except urllib.error.HTTPError as e:
                if e.code in (401, 403, 405, 406, 429):
                    blocked += 1          # page exists but blocks automated requests — likely real
                else:
                    dead.append((u, e.code))   # 404/410/5xx = probably wrong/hallucinated
            except Exception as e:
                dead.append((u, type(e).__name__))
        print(f"[audit] URLs: {ok} reachable · {blocked} blocked-but-likely-real (403/429) · "
              f"{len(dead)} DEAD (verify/replace by hand)")
        if dead:
            for u, why in dead[:12]:
                print(f"         DEAD [{why}] {u}")
        print("[audit] NOTE: Lava can't browse, so citations are model-recalled. Treat ONLY the "
              "'reachable' ones as trustworthy; hand-verify a sample before claiming 'source-grounded'.")

    filled = [r for r in recs if r.get("status") == "filled"]
    if filled and args.sample:
        print(f"\n===== RANDOM SAMPLE OF {min(args.sample, len(filled))} (eyeball quote-vs-value) =====")
        random.seed(0)
        for r in random.sample(filled, min(args.sample, len(filled))):
            print(f"\n--- {r['crop']} / {r['disease']}  [{r['_file']}] ---")
            print(f"  symptom_text: {r.get('symptom_text','')[:220]}")
            for k in FIELDS:
                fk = r.get("fields", {}).get(k, {})
                print(f"  {k}: {str(fk.get('value',''))[:80]}")
                print(f"      src : {fk.get('source_url','')}")
                print(f"      quote: {str(fk.get('verbatim_quote',''))[:140]}")


if __name__ == "__main__":
    main()
