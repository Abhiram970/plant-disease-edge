"""
KAGGLE — build the SAGE subset AND run the full CNN baseline sweep, in one committed run.
==========================================================================================

Paste this whole file as ONE cell, then use  Save Version -> "Save & Run All (Commit)".
It runs on Kaggle's servers, survives a closed tab, and stops itself before the 9-hour wall so the
results are actually written.

NOTEBOOK SETTINGS (right pane) -- all of these matter:
    Accelerator = GPU T4 x2  (or P100)
    Internet    = ON              <- git clone and HuggingFace both fail without it
    Persistence = Files only

SECRET (Add-ons -> Secrets): GH_TOKEN = a GitHub fine-grained PAT with read-only Contents.
The repo is private, so an anonymous clone returns 404.

WHY THIS DOES NOT GET STUCK (it did before):
  1. hf_hub_download returns a SYMLINK into the HF cache; the old code deleted the symlink but not the
     ~10 GB blob behind it, so every shard permanently consumed disk until /kaggle/working filled and
     the fetch died. Fixed in sage_data.py.
  2. HF_HOME now points at /kaggle/temp (73 GB scratch) instead of the small working volume.
  3. Images are downscaled to MAX_SIDE. At full resolution the subset is 19.8 GB against a ~20 GB
     limit -- it does not fit. Everything trains at 224, so this costs nothing.
  4. Disk is checked after the fetch and the run aborts early with a clear message rather than dying
     mid-training.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ----------------------------------------------------------------- configuration
MAX_SIDE = 288       # store at 288 px; models train at 224. Keeps the subset ~4 GB instead of ~20 GB.
EPOCHS = 8           # MUST be identical across archs or the table is not a comparison
BATCH = 128
MIN_BATCH = 16       # retry floor; batch is halved on failure (T4 OOMs at 128 on the heavy models)
WORKERS = 4          # Kaggle VMs have ~4 vCPU
BUDGET_H = 8.3       # checked BEFORE starting an arch, so the run can overshoot by one arch's length
ROLE = "all"         # "all" = seen + held-out crops (reusable); "train" = seen only, smaller/faster

# Priority order, so a truncated run still yields the valuable ones. FastViT is near the front because
# it is the same architecture family as MobileCLIP's image encoder -- the fairest supervised-vs-VLM
# comparison in the paper.
ARCHS = [
    # Run 1 (8 Aug 2026) completed everything except these four, so they come first on a re-run:
    # three died of CUDA OOM at batch 128 on a T4, one ran out of budget. Finished archs are skipped
    # automatically, so this list can stay complete.
    "tf_efficientnetv2_s", "convnextv2_tiny", "resnet101", "densenet121",
    "fastvit_t8", "fastvit_sa12",  # same family as the VLM backbone
    "convnextv2_nano",
    "resnet50", "mobilenetv3_small_100", "mobilenetv4_conv_small",
    "mobilenetv3_large_100", "mobilenetv4_conv_medium", "efficientnet_b0",
    "regnety_040",
]

T0 = time.time()
WORK = Path("/kaggle/working")
os.chdir(WORK)


def sh(cmd, **kw):
    return subprocess.run(cmd, **kw).returncode


def free_gb(p="/kaggle/working"):
    return shutil.disk_usage(p).free / 1e9


def used_gb(p):
    p = Path(p)
    if not p.exists():
        return 0.0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9


# ----------------------------------------------------------------- repo
tok = None
try:
    from kaggle_secrets import UserSecretsClient
    tok = UserSecretsClient().get_secret("GH_TOKEN")
except Exception as e:
    print(f"[warn] no GH_TOKEN secret ({type(e).__name__}); anonymous clone only works if public.")

repo = WORK / "pde"
if not repo.exists():
    url = (f"https://{tok}@github.com/Abhiram970/plant-disease-edge.git" if tok
           else "https://github.com/Abhiram970/plant-disease-edge.git")
    if sh(["git", "clone", "--depth", "1", url, str(repo)]) != 0:
        sys.exit("[fatal] clone failed -> check Internet=ON and the GH_TOKEN secret.")
print(f"[ok] repo at {repo}")

sh([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "timm"])
sh([sys.executable, "-m", "pip", "install", "-q",
    "safetensors", "pyyaml", "huggingface_hub", "pyarrow", "tqdm"])

# ----------------------------------------------------------------- environment
# HF blobs go to scratch, NOT the 20 GB working volume. This is the single most important line here.
os.environ["HF_HOME"] = "/kaggle/temp/hf"
os.environ["HF_HUB_CACHE"] = "/kaggle/temp/hf/hub"
Path("/kaggle/temp/hf/hub").mkdir(parents=True, exist_ok=True)
os.environ["PDE_DATA_ROOT"] = str(WORK)                    # results land here
os.environ["PDE_DATASET_DIR"] = str(WORK / "exp_data")     # images land here

import torch
print(f"[env] cuda={torch.cuda.is_available()} "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''} "
      f"| free={free_gb():.1f} GB")

# ----------------------------------------------------------------- data
manifest = WORK / "manifest.csv"
if not manifest.exists():
    print(f"\n[data] fetching SAGE (role={ROLE}, max_side={MAX_SIDE}). Expect 1-3 h.", flush=True)
    rc = sh([sys.executable, "-u", str(repo / "scripts" / "sage_data.py"),
             "--role", ROLE, "--max-side", str(MAX_SIDE)])
    if rc != 0:
        print(f"[warn] sage_data exited {rc} -- continuing if enough images landed.")
    sh([sys.executable, "-u", str(repo / "scripts" / "build_manifest.py"), "--min-images", "25"])
else:
    print("[ok] manifest.csv already present -- skipping fetch.")

size = used_gb(WORK / "exp_data")
print(f"[data] exp_data = {size:.2f} GB, free = {free_gb():.1f} GB")
if not manifest.exists():
    sys.exit("[fatal] no manifest.csv -- the fetch did not produce a usable dataset. Check the log "
             "above for shard errors, and confirm Internet=ON.")
if free_gb() < 2:
    sys.exit(f"[fatal] only {free_gb():.1f} GB free; training would die mid-run. Lower MAX_SIDE or "
             f"set ROLE='train' (seen crops only) and re-run.")

results = WORK / "results"
results.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------- the sweep
script = str(repo / "scripts" / "supervised_baseline.py")
done, failed, skipped = [], [], []

for i, arch in enumerate(ARCHS):
    out = results / f"supervised_{arch}.json"
    if out.exists():
        print(f"[skip] {arch} (already complete)")
        continue
    elapsed_h = (time.time() - T0) / 3600
    if elapsed_h > BUDGET_H:
        skipped = ARCHS[i:]
        print(f"[budget] {elapsed_h:.1f} h elapsed -- stopping cleanly. Not started: {skipped}")
        break
    # Halve the batch and retry on failure. A T4 has 14.56 GB and batch 128 does NOT fit
    # tf_efficientnetv2_s, convnextv2_tiny or resnet101 -- all three died with CUDA OOM on the first
    # full run, and were reported as "arch may not exist" because the runner could not see why the
    # child exited. Retrying costs nothing when the first attempt succeeds.
    rc, batch = 1, BATCH
    while rc != 0 and batch >= MIN_BATCH:
        print(f"\n{'=' * 72}\n[run] {arch}  (epochs={EPOCHS} batch={batch})  "
              f"t+{(time.time() - T0) / 3600:.1f} h  free={free_gb():.1f} GB\n{'=' * 72}", flush=True)
        rc = sh([sys.executable, "-u", script, "--arch", arch, "--epochs", str(EPOCHS),
                 "--batch", str(batch), "--workers", str(WORKERS), "--resume"])
        if rc != 0:
            batch //= 2
            if batch >= MIN_BATCH:
                print(f"  [retry] {arch} failed (likely CUDA OOM) -> batch {batch}", flush=True)
    (done if rc == 0 else failed).append(arch)

    # Drop the checkpoint once the result JSON exists: it is only needed to resume an interrupted
    # arch, and 14 of them (up to ~400 MB each for resnet101/convnextv2) would eat several GB of both
    # disk and the notebook's ~20 GB output allowance. Checkpoints for FAILED archs are kept so a
    # follow-up session can resume them.
    if rc == 0 and out.exists():
        ck = WORK / "checkpoints" / f"{arch}_ckpt.pt"
        if ck.exists():
            ck.unlink()
            print(f"  [clean] removed {ck.name} (result saved)")

# ----------------------------------------------------------------- summary
print(f"\n{'=' * 72}\nSUPERVISED CNN BASELINES — seen top-1\n{'=' * 72}")
rows = []
for arch in ARCHS:
    p = results / f"supervised_{arch}.json"
    if p.exists():
        try:
            d = json.loads(p.read_text())
            rows.append((arch, d.get("seen_top1"), d.get("seen_classes"),
                         len(d.get("epoch_log") or [])))
        except Exception:
            rows.append((arch, None, None, 0))
for arch, acc, ncls, neps in sorted(rows, key=lambda r: -(r[1] or 0)):
    print(f"  {arch:26s} {'--' if acc is None else f'{acc:.1%}'}   classes={ncls}  epochs={neps}")
print("  (all are structurally 0% on UNSEEN crops -- that is the point)")
print(f"\n  completed: {done}")
if failed:
    print(f"  FAILED even at batch {MIN_BATCH} (check the traceback above -- CUDA OOM, or the arch "
          f"name may not exist in this timm version): {failed}")
if skipped:
    print(f"  NOT STARTED: {skipped}")
print(f"  wall: {(time.time() - T0) / 3600:.2f} h   free: {free_gb():.1f} GB")
print("\nNEXT: right pane -> Output -> download results/supervised_*.json into docs/paper/, then")
print("  python docs/paper/make_tables.py --write && python docs/paper/make_tex_tables.py")
print("ALSO: Output -> Create Dataset ('pde-sage-data') to keep exp_data + manifest.csv for reuse,")
print("so no future session ever refetches from HuggingFace.")
