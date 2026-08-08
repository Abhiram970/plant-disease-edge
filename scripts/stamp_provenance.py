"""
Stamp every descriptor field with the provenance of its citation.

WHY THIS EXISTS
---------------
A reader opening descriptors/Corn.json sees `verbatim_quote` on most records and would reasonably
assume all of them were checked against the cited page. They were not. Only records processed by
apply_verified_citations.py had their page fetched and the sentence copied off it; the rest carry
quotes the generating model recalled from memory, which the endpoint could not verify because it
cannot browse.

That distinction is stated in the paper, but the data files themselves should not be able to mislead
anyone who reads them without the paper in hand. This writes it into each field:

    provenance = "page-verified"   quote copied from a page we retrieved and read
                 "model-recalled"  quote produced by the LLM; treat as UNVERIFIED provenance
                 "none"            no quote on this field

Re-runnable and idempotent.

    python scripts/stamp_provenance.py
"""
from __future__ import annotations
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

FIELDS = ("pathogen", "affected_organs", "visual_symptoms")


def main():
    tally = Counter()
    for p in sorted(C.DESCRIPTORS_DIR.glob("*.json")):
        recs = json.loads(p.read_text(encoding="utf-8"))
        for rec in recs:
            verified = bool(rec.get("verified"))
            for f in FIELDS:
                fld = (rec.get("fields") or {}).get(f)
                if not isinstance(fld, dict):
                    continue
                if not (fld.get("verbatim_quote") or "").strip():
                    prov = "none"
                elif verified:
                    prov = "page-verified"
                else:
                    prov = "model-recalled"
                fld["provenance"] = prov
                tally[prov] += 1
        p.write_text(json.dumps(recs, indent=2), encoding="utf-8")

    total = sum(tally.values())
    print(f"[provenance] stamped {total} descriptor fields across "
          f"{len(list(C.DESCRIPTORS_DIR.glob('*.json')))} crops")
    for k, v in tally.most_common():
        print(f"    {k:15s} {v:5d}  ({v / total:.0%})")
    print("\n  'model-recalled' means the LLM produced the sentence and it was NOT checked against")
    print("  the cited page. It is unverified provenance, not evidence.")


if __name__ == "__main__":
    main()
