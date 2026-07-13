"""
Batch runner for the supervised CNN baseline family (Phase 5) — meant for a paid / cloud GPU.

Runs supervised_baseline.py for each architecture, SKIPPING any whose result JSON already exists
(so it is fully resumable), then prints a ranked summary. Training is on the seen crops
(~70k images, 166 classes in Experiment C), which is why this belongs on a rented GPU, not the laptop.

The point of the family: every CNN is strong on SEEN crops but STRUCTURALLY 0 on unseen (no output
neuron) — the contrast that motivates the frozen-VLM + descriptor head. See probe_seen.py for the VLM
half of the table.

------------------------------------------------------------------------------------------------
SETUP ON A CLOUD GPU (Linux)
  pip install torch torchvision timm pillow numpy
  # get the data + manifest onto the box (pick one):
  #   (a) re-fetch (fast on cloud bandwidth):
  export PDE_DATA_ROOT=/workspace/working
  export PDE_DATASET_DIR=/workspace/working/exp_data
  python scripts/sage_data.py --role all
  python scripts/build_manifest.py --min-images 25
  #   (b) or upload your local exp_data/ + manifest.csv and just set the two env vars above.
  # then run the family:
  python scripts/run_cnn_baselines.py --epochs 12 --batch 128 --workers 8
------------------------------------------------------------------------------------------------

USAGE
  python scripts/run_cnn_baselines.py                       # default family (skips already-done)
  python scripts/run_cnn_baselines.py --epochs 15 --batch 256 --workers 12
  python scripts/run_cnn_baselines.py --archs resnet101 convnextv2_tiny   # only these
  python scripts/run_cnn_baselines.py --force              # re-run even if a JSON exists
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

# Curated timm architectures with stable pretrained weights, spanning mobile -> mid -> large.
# (mobilenetv3_small_100 + resnet50 usually already run on the laptop; kept here for a complete table
#  and auto-skipped if their JSON exists.)
DEFAULT_ARCHS = [
    "mobilenetv3_small_100", "resnet50",                # already run on laptop (skipped if present)
    # --- edge / mobile (the paper's deployment-relevant references) ---
    "mobilenetv3_large_100", "mobilenetv4_conv_small", "mobilenetv4_conv_medium",
    "efficientnet_b0", "fastvit_t8", "fastvit_sa12",
    # --- modern mid / large CNNs (strong supervised references) ---
    "tf_efficientnetv2_s", "convnextv2_nano", "convnextv2_tiny",
    "resnet101", "regnety_040", "densenet121",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=DEFAULT_ARCHS)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="re-run even if a result JSON exists")
    args = ap.parse_args()

    script = str(Path(__file__).resolve().parent / "supervised_baseline.py")
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ran, failed = [], []
    for arch in args.archs:
        out = C.RESULTS_DIR / f"supervised_{arch}.json"
        if out.exists() and not args.force:
            print(f"[skip] {arch} (result exists: {out.name})")
            continue
        cmd = [sys.executable, script, "--arch", arch, "--epochs", str(args.epochs),
               "--batch", str(args.batch), "--workers", str(args.workers)]
        print(f"\n{'=' * 72}\n[run] {arch}   (epochs={args.epochs} batch={args.batch})\n{'=' * 72}")
        rc = subprocess.run(cmd).returncode
        (ran if rc == 0 else failed).append(arch)

    print(f"\n{'=' * 72}\nSUMMARY (seen top-1, 166-class trained head)\n{'=' * 72}")
    rows = []
    for arch in args.archs:
        out = C.RESULTS_DIR / f"supervised_{arch}.json"
        if out.exists():
            try:
                rows.append((arch, json.loads(out.read_text()).get("seen_top1")))
            except Exception:
                rows.append((arch, None))
    for arch, acc in sorted(rows, key=lambda x: -(x[1] or 0)):
        print(f"  {arch:26s} {'' if acc is None else f'{acc:.1%}'}")
    print("  (all of these are structurally 0% on UNSEEN crops — that is the point)")
    if failed:
        print(f"\n  FAILED (likely arch name not in this timm version): {failed}")


if __name__ == "__main__":
    main()
