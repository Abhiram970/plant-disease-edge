"""
Recover result JSONs that were pasted into a .docx (e.g. copied out of a Kaggle notebook by hand).

Parses the document text, brace-matches every top-level JSON object, validates it looks like a
supervised_baseline.py result, and writes it to RESULTS_DIR/supervised_<arch>.json. Nothing is
transcribed by hand -- the whole point is that numbers reach the paper mechanically.

    python scripts/import_docx_results.py "C:\\path\\archs.docx" [--dest docs/paper]
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


def docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", "replace")
    xml = xml.replace("</w:p>", "\n").replace("</w:tc>", "\t")
    txt = re.sub(r"<[^>]+>", "", xml)
    for a, b in (("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"')):
        txt = txt.replace(a, b)
    return txt


def json_blocks(text: str):
    """Yield one brace-balanced span per record, anchored on the "arch" key.

    Plain brace matching is not enough: a document assembled by copy-paste can contain a stray '{'
    between records (this one does), and a single unbalanced brace desynchronises every record after
    it -- which silently dropped 4 of 10 results on the first attempt. Anchoring on "arch" and
    scanning back to the nearest '{' resynchronises at every record instead.
    """
    for m in re.finditer(r'"arch"\s*:', text):
        open_i = text.rfind("{", 0, m.start())
        if open_i < 0:
            continue
        depth = 0
        for i in range(open_i, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    yield text[open_i:i + 1]
                    break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("docx")
    ap.add_argument("--dest", default=None, help="default: config.RESULTS_DIR")
    args = ap.parse_args()

    dest = Path(args.dest) if args.dest else C.RESULTS_DIR
    dest.mkdir(parents=True, exist_ok=True)
    text = docx_text(Path(args.docx))

    found, skipped = [], 0
    for blk in json_blocks(text):
        try:
            d = json.loads(blk)
        except Exception:
            skipped += 1
            continue
        if not (isinstance(d, dict) and d.get("arch") and d.get("seen_top1") is not None):
            continue          # inner objects (epoch entries) and anything else
        out = dest / f"supervised_{d['arch']}.json"
        out.write_text(json.dumps(d, indent=2), encoding="utf-8")
        found.append((d["arch"], d["seen_top1"], d.get("seen_classes"),
                      len(d.get("epoch_log") or [])))

    print(f"[import] wrote {len(found)} result files to {dest}")
    for arch, acc, ncls, neps in sorted(found, key=lambda r: -r[1]):
        print(f"    {arch:26s} {acc:.1%}  classes={ncls}  epochs={neps}")
    if skipped:
        print(f"[import] {skipped} non-JSON braced spans ignored")

    bad = [f for f in found if f[2] != 166 or f[3] != 8]
    if bad:
        print("\n[warn] these do NOT share the 166-class / 8-epoch protocol -- do not put them in "
              f"the same table: {[b[0] for b in bad]}")


if __name__ == "__main__":
    main()
