"""
KAGGLE NOTEBOOK — full supervised CNN baseline sweep (paste as ONE cell)
=======================================================================

Runs every architecture in the baseline table at IDENTICAL settings, so the numbers are comparable.
Designed to be launched with "Save Version -> Save & Run All (Commit)" and left alone: it survives a
closed tab and the 40-minute idle killer, and it stops itself cleanly before Kaggle's 9-hour wall so
the results are actually written.

NOTEBOOK SETTINGS (right pane) -- all four matter:
    Accelerator = GPU T4 x2 (or P100)
    Internet    = ON              <- git clone and HuggingFace both fail silently-ish without it
    Persistence = Files only
    Add data    = pde-sage-data   <- if you already built it (see kaggle/RUNBOOK_KAGGLE.md Session 1)

SECRET (Add-ons -> Secrets): add GH_TOKEN = a fine-grained GitHub PAT with read-only Contents.
The repo is private, so a plain clone or ZIP download returns 404. Using a Secret keeps the token out
of the saved notebook.

RESUMING A SECOND SESSION: attach this notebook's own output as a dataset next time and set
PRIOR_OUTPUT below to its path. Completed architectures are skipped and a partially-trained one
resumes from its last epoch checkpoint.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ----------------------------------------------------------------- configuration
EPOCHS = 8          # MUST match across all archs or the table is not a comparison
BATCH = 128
WORKERS = 4         # Kaggle VMs have ~4 vCPU; more just thrashes
BUDGET_H = 8.3      # stop before Kaggle's 9 h wall so results get written
PRIOR_OUTPUT = None  # e.g. "/kaggle/input/cnn-baselines-run1" to resume a previous session

# Priority order: the scientifically valuable ones first, so a truncated run still helps.
# FastViT is deliberately near the front -- it is the same architecture family as MobileCLIP's image
# encoder, making it the fairest supervised-vs-VLM comparison in the paper.
ARCHS = [
    "tf_efficientnetv2_s",        # abandoned mid-run locally at 85.2% after 1 epoch
    "fastvit_t8", "fastvit_sa12",  # same family as the VLM backbone
    "convnextv2_tiny", "convnextv2_nano",
    "resnet50", "mobilenetv3_small_100", "mobilenetv4_conv_small",  # rerun for a uniform protocol
    "mobilenetv3_large_100", "mobilenetv4_conv_medium", "efficientnet_b0",
    "resnet101", "regnety_040", "densenet121",
]

T0 = time.time()
WORK = Path("/kaggle/working")
os.chdir(WORK)

# ----------------------------------------------------------------- repo (private -> needs a token)
tok = None
try:
    from kaggle_secrets import UserSecretsClient
    tok = UserSecretsClient().get_secret("GH_TOKEN")
except Exception as e:
    print(f"[warn] no GH_TOKEN secret ({type(e).__name__}). Falling back to an anonymous clone, "
          f"which only works if the repo is public.")

repo = WORK / "pde"
if not repo.exists():
    url = (f"https://{tok}@github.com/Abhiram970/plant-disease-edge.git" if tok
           else "https://github.com/Abhiram970/plant-disease-edge.git")
    rc = subprocess.run(["git", "clone", "--depth", "1", url, str(repo)]).returncode
    if rc != 0:
        sys.exit("[fatal] clone failed. Check: Internet=ON, and GH_TOKEN set under Add-ons -> Secrets.")
print(f"[ok] repo at {repo}")

# --no-deps keeps pip from dragging in a 2 GB torch wheel that can break Kaggle's CUDA build.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "timm"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "safetensors", "pyyaml", "huggingface_hub", "pyarrow", "tqdm"], check=False)

# ----------------------------------------------------------------- data
# Prefer the prebuilt Kaggle Dataset. Re-fetching SAGE shards costs hours and a single shard is ~10 GB
# against a ~20 GB /kaggle/working, so the fetch path is a last resort.
os.environ["PDE_DATA_ROOT"] = str(WORK)          # results MUST land in working, never in input
attached = Path("/kaggle/input/pde-sage-data")
if (attached / "exp_data").exists():
    os.environ["PDE_DATASET_DIR"] = str(attached / "exp_data")
    src = attached / "manifest.csv"
    if src.exists():
        (WORK / "manifest.csv").write_bytes(src.read_bytes())
    print(f"[ok] using attached dataset {attached}")
else:
    print("[warn] pde-sage-data not attached -- fetching SAGE from HuggingFace. This takes HOURS and "
          "may exhaust /kaggle/working. Prefer RUNBOOK_KAGGLE.md Session 1, then attach it.")
    os.environ["PDE_DATASET_DIR"] = str(WORK / "exp_data")
    subprocess.run([sys.executable, str(repo / "scripts" / "sage_data.py"), "--role", "all"],
                   check=False)
    subprocess.run([sys.executable, str(repo / "scripts" / "build_manifest.py"),
                    "--min-images", "25"], check=False)

results = WORK / "results"
results.mkdir(parents=True, exist_ok=True)

# Seed from a previous session's output so we resume rather than repeat.
if PRIOR_OUTPUT:
    prior = Path(PRIOR_OUTPUT)
    for sub in ("results", "checkpoints"):
        s = prior / sub
        if s.exists():
            d = WORK / sub
            d.mkdir(parents=True, exist_ok=True)
            for f in s.iterdir():
                if not (d / f.name).exists():
                    (d / f.name).write_bytes(f.read_bytes())
            print(f"[ok] seeded {sub}/ from {s}")

import torch
print(f"[env] cuda={torch.cuda.is_available()} "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")

# ----------------------------------------------------------------- the sweep
script = str(repo / "scripts" / "supervised_baseline.py")
done, failed, skipped = [], [], []

for arch in ARCHS:
    out = results / f"supervised_{arch}.json"
    if out.exists():
        print(f"[skip] {arch} (already complete)")
        continue

    elapsed_h = (time.time() - T0) / 3600
    if elapsed_h > BUDGET_H:
        print(f"[budget] {elapsed_h:.1f} h elapsed -- stopping before the 9 h wall. "
              f"Remaining: {', '.join(ARCHS[ARCHS.index(arch):])}")
        skipped = ARCHS[ARCHS.index(arch):]
        break

    print(f"\n{'=' * 72}\n[run] {arch}  (epochs={EPOCHS} batch={BATCH})  t+{elapsed_h:.1f} h\n{'=' * 72}",
          flush=True)
    rc = subprocess.run([sys.executable, "-u", script, "--arch", arch, "--epochs", str(EPOCHS),
                         "--batch", str(BATCH), "--workers", str(WORKERS), "--resume"]).returncode
    (done if rc == 0 else failed).append(arch)

# ----------------------------------------------------------------- summary
print(f"\n{'=' * 72}\nSUPERVISED CNN BASELINES — seen top-1 (166-class head)\n{'=' * 72}")
rows = []
for arch in ARCHS:
    p = results / f"supervised_{arch}.json"
    if p.exists():
        try:
            d = json.loads(p.read_text())
            rows.append((arch, d.get("seen_top1"), d.get("seen_classes"), len(d.get("epoch_log") or [])))
        except Exception:
            rows.append((arch, None, None, 0))
for arch, acc, ncls, neps in sorted(rows, key=lambda r: -(r[1] or 0)):
    print(f"  {arch:26s} {'--' if acc is None else f'{acc:.1%}'}   "
          f"classes={ncls}  epochs_logged={neps}")
print("  (every one of these is structurally 0% on UNSEEN crops -- that is the point)")
print(f"\n  completed this session: {done}")
if failed:
    print(f"  FAILED (arch name may not exist in this timm version): {failed}")
if skipped:
    print(f"  NOT STARTED (ran out of budget): {skipped}")
print(f"  wall time: {(time.time() - T0) / 3600:.2f} h")
print("\nCollect: right pane -> Output -> download results/supervised_*.json into docs/paper/,")
print("then run  python docs/paper/make_tables.py --write  and  make_figures.py  locally.")
