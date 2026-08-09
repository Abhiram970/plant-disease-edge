"""
Backfill params_M into supervised_*.json results produced before the field existed.

Parameter counts are architecture constants, so they can be recovered exactly by instantiating the
model — no need to re-train, and no need to type a number into the paper by hand.

    python scripts/backfill_params.py [--dest docs/paper]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=None, help="default: docs/paper")
    args = ap.parse_args()
    dest = Path(args.dest) if args.dest else C.REPO_ROOT / "docs" / "paper"

    import timm
    n = 0
    for f in sorted(dest.glob("supervised_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        arch = d.get("arch")
        if not arch:
            continue
        if d.get("params_M") is not None and d.get("batch") is not None:
            continue
        try:
            m = timm.create_model(arch, pretrained=False,
                                  num_classes=d.get("seen_classes", 166))
        except Exception as e:
            print(f"  [skip] {arch}: {type(e).__name__}: {e}")
            continue
        d["params_M"] = round(sum(p.numel() for p in m.parameters()) / 1e6, 2)
        d.setdefault("img_size", C.IMG_SIZE)
        # Run 1 (8 Aug 2026) predates the batch/epochs fields. Its log records every architecture as
        # "[run] <arch> (epochs=8 batch=128)", so backfill that rather than leave the field absent --
        # the four re-run architectures used batch 64 after the OOMs, and a table that cannot show
        # the difference would hide a genuine protocol split.
        d.setdefault("batch", 128)
        d.setdefault("epochs", len(d.get("epoch_log") or []) or 8)
        f.write_text(json.dumps(d, indent=2), encoding="utf-8")
        print(f"  {arch:26s} {d['params_M']:7.2f} M   seen {d.get('seen_top1', 0):.1%}")
        n += 1
    print(f"[backfill] updated {n} files in {dest}")


if __name__ == "__main__":
    main()
