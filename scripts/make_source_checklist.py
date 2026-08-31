"""
Build a human-checkable audit list of every source URL in the descriptor registry.

For each URL: whether it resolves, which descriptor records depend on it, what the page must actually
contain for the citation to be honest, and whether that quote was read off the page (page-verified) or
produced from model memory (model-recalled, i.e. UNVERIFIED and the ones worth checking first).

    python scripts/make_source_checklist.py            # writes docs/paper/SOURCE_CHECKLIST.md
    python scripts/make_source_checklist.py --no-net   # skip the reachability probe
"""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import json
import re
import socket
import ssl
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
FIELDS = ("pathogen", "affected_organs", "visual_symptoms")
HELD = set(C.EXPERIMENTS["C"]["held"])


def probe(url, _attempt=0):
    """Classify a URL. Distinguishing these matters: a 403 and an SSL trust failure both LOOK dead
    from a script but the page is fine, and reporting them as dead sends you hunting for
    replacements you do not need. Only a real 404/410/DNS failure is dead.

    Note many extension sites answer HEAD with 404 while serving GET correctly, so GET decides.

    A DEAD verdict is retried once, serially: under concurrency these servers intermittently drop
    connections, and a transient reset previously reported a live 403-protected page as dead.
    """
    last = None
    for method in ("HEAD", "GET"):
        try:
            req = urllib.request.Request(url, method=method, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=25) as r:
                if 200 <= r.status < 400:
                    return "OK"
                last = str(r.status)
        except ssl.SSLCertVerificationError:                    # local trust store, not the site
            return "SSL-unverifiable (page likely fine)"
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                return f"{e.code} bot-blocked (page exists)"
            last = f"DEAD ({e.code})"
        except Exception as e:
            if "CERTIFICATE_VERIFY_FAILED" in str(e):
                return "SSL-unverifiable (page likely fine)"
            # A timeout is not a missing page. Two large government sites (agriculture.gov.au,
            # aphis.usda.gov) are simply slow and were being labelled DEAD, which is the same false
            # alarm as calling a 403 dead -- and it sends you hunting for a replacement for a page
            # that is fine. Only an HTTP status can establish that a page is gone.
            if isinstance(e, (TimeoutError, socket.timeout)) or "timed out" in str(e).lower():
                return "slow (timed out; page likely fine)"
            last = f"unreachable ({type(e).__name__})"
    verdict = last or "DEAD"
    if verdict.startswith("DEAD") and _attempt == 0:
        time.sleep(2)
        return probe(url, _attempt=1)
    return verdict


def _keep_notes(dst: Path, out: list[str]) -> list[str]:
    """Carry hand-written `**Additional info:**` blocks across a regeneration, keyed by URL.

    This file is generated, but the verification pass is done BY HAND inside it -- page text is
    pasted under each URL as evidence. A plain overwrite destroyed 181 such blocks and took the file
    from 524 KB to 92 KB, i.e. it deleted the entire audit trail the checklist exists to hold. The
    notes are re-attached to whichever entry now cites the same URL, so a URL that moved keeps its
    evidence and a URL that disappeared drops its note with it.
    """
    if not dst.exists():
        return out
    prev = dst.read_text(encoding="utf-8", errors="replace")
    notes: dict[str, str] = {}
    for entry in prev.split("\n### ")[1:]:
        head = entry.split("- **Cited by:**")[0]
        # An entry may carry several URLs -- during the verification pass a replacement was written
        # beside the original as `<old> - <new>`. Key the note under EVERY url in the entry, because
        # the descriptor may since have been repointed from one to the other and the evidence
        # belongs to whichever is cited now.
        urls_here = re.findall(r"<(https?://[^>]+)>", head)
        if not urls_here:
            continue
        note = re.search(r"(?m)^-?\s*-?\s*\*\*Additional info:\*\*(.*?)(?=\n### |\Z)", entry, re.S)
        if note and note.group(1).strip():
            for u in urls_here:
                notes.setdefault(u, note.group(1).rstrip())
    if not notes:
        return out

    merged, cur, kept = [], None, 0
    for line in out:
        m = re.match(r"<(https?://[^>]+)>$", line.strip())
        if m:
            cur = m.group(1)
        if line.startswith("### ") and cur and cur in notes:
            cur = None
        merged.append(line)
        # append the note right after the last bullet of this entry (the blank line before the next)
        if cur and line == "" and merged[-2:-1] and merged[-2].startswith("- **"):
            if cur in notes:
                merged.insert(len(merged) - 1, f"- **Additional info:**{notes[cur]}")
                kept += 1
            cur = None
    print(f"[checklist] preserved {kept} hand-written note block(s) from the previous file")
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-net", action="store_true")
    args = ap.parse_args()

    cites = defaultdict(list)
    for p in sorted(C.DESCRIPTORS_DIR.glob("*.json")):
        for rec in json.loads(p.read_text(encoding="utf-8")):
            crop, dis = rec.get("crop"), rec.get("disease")
            for f in FIELDS:
                fl = (rec.get("fields") or {}).get(f)
                if not isinstance(fl, dict):
                    continue
                url = (fl.get("source_url") or "").strip()
                if not url.startswith("http"):
                    continue
                cites[url].append({
                    "crop": crop, "disease": dis, "field": f,
                    "value": (fl.get("value") or "").strip(),
                    "quote": (fl.get("verbatim_quote") or "").strip(),
                    "prov": fl.get("provenance", "?"),
                    "held": crop in HELD,
                })

    urls = sorted(cites)
    status = {u: "not checked" for u in urls}
    if not args.no_net:
        print(f"probing {len(urls)} urls ...")
        with cf.ThreadPoolExecutor(max_workers=6) as ex:   # gentler; these servers rate-limit
            for u, s in zip(urls, ex.map(probe, urls)):
                status[u] = s

    def sort_key(u):
        rows = cites[u]
        held = any(r["held"] for r in rows)
        unver = any(r["prov"] == "model-recalled" for r in rows)
        dead = status[u].startswith("DEAD")
        return (not dead, not held, not unver, u)   # dead first, then held-out, then unverified

    out = [
        "# Source URL checklist",
        "",
        "> Generated by `scripts/make_source_checklist.py`. Re-run after editing any descriptor.",
        "",
        "**How to use this.** For each URL, open it and confirm the page actually contains the "
        "sentence under *Must contain*. If it does not, the citation is wrong — replace the URL "
        "**and** the quote together (a live link whose quote is absent is worse than a dead link, "
        "because it reads as fabricated evidence).",
        "",
        "**Priority order.** Rows are sorted: dead links first, then **HELD-OUT** crops (these carry "
        "the paper's cross-crop claim), then `model-recalled` quotes (never checked against the "
        "page). Rows marked `page-verified` were already read off the page — lowest priority.",
        "",
        f"- URLs: **{len(urls)}**",
        f"- Citing fields: **{sum(len(v) for v in cites.values())}**",
        f"- Fields on held-out crops: "
        f"**{sum(1 for v in cites.values() for r in v if r['held'])}**",
        "",
        "---",
        "",
    ]

    for i, u in enumerate(sorted(urls, key=sort_key), 1):
        rows = cites[u]
        held = "HELD-OUT" if any(r["held"] for r in rows) else "seen"
        provs = sorted({r["prov"] for r in rows})
        who = ", ".join(sorted({f"{r['crop']}/{r['disease']}" for r in rows}))
        flds = ", ".join(sorted({r["field"] for r in rows}))
        out.append(f"### {i}. [{status[u]}] `{held}` — {', '.join(provs)}")
        out.append("")
        out.append(f"<{u}>")
        out.append("")
        out.append(f"- **Cited by:** {who}  (fields: {flds})")
        best = max(rows, key=lambda r: len(r["quote"]))
        if best["quote"]:
            out.append(f"- **Must contain:** \"{best['quote'][:260]}\"")
        else:
            out.append("- **Must contain:** _(no quote on this citation — it asserts only the value "
                       "below, so the page must at least support that)_")
        vals = sorted({r["value"][:110] for r in rows if r["value"]})
        if vals:
            out.append(f"- **Should support:** {'; '.join(vals)}")
        out.append("")

    dst = C.REPO_ROOT / "docs" / "paper" / "SOURCE_CHECKLIST.md"
    out = _keep_notes(dst, out)
    dst.write_text("\n".join(out), encoding="utf-8")
    print(f"[checklist] wrote {dst}  ({len(urls)} urls)")
    bad = [u for u in urls if status[u].startswith("DEAD")]
    print(f"[checklist] dead: {len(bad)}")
    for u in bad:
        print(f"    {u}")


if __name__ == "__main__":
    main()
