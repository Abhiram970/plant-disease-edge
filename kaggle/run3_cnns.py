"""
KAGGLE RUN 3 of 3 — THE 14 SUPERVISED CNN BASELINES.  Paste this whole file as ONE cell.
=======================================================================================

NOTEBOOK SETTINGS
  Accelerator = GPU T4 x2   ·   Internet = ON   ·   Persistence = Files only
  Add data    -> **pde-sage-data** (run 1) and **pde-results-2** (run 2)
  Secret      -> GH_TOKEN

Then: Save Version -> "Save & Run All (Commit)" -> close the tab.
When it finishes: Output -> "Create Dataset" -> **pde-results-3**.

This is the long one (~8 h for all 14). It is also the most restartable: every architecture writes
its own results/supervised_<arch>.json, so a finished architecture is never redone, and an
architecture interrupted mid-training resumes from its last epoch checkpoint. If the budget stops
the run, publish the output, attach it to a fresh copy, and run again -- it picks up the first
architecture with no JSON.

TWO FAILURE MODES THIS HANDLES, BOTH OF WHICH COST A PREVIOUS SWEEP
------------------------------------------------------------------
  * CUDA OOM. A T4 has 14.56 GB and batch 128 does not fit the heavier models. Three architectures
    were lost to this and the cause was misdiagnosed at the time as "arch may not exist in this timm
    version" -- all three tracebacks were torch.OutOfMemoryError. The batch is halved and retried
    down to MIN_BATCH instead of losing the architecture.
  * --resume was never actually passed by the old runner, so an interrupted architecture restarted
    from epoch 0 every time. That is what lost the EfficientNetV2-S run at 85.2%. It is passed here.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ================================ SETTINGS ================================
EPOCHS, BATCH, WORKERS = 8, 128, 4
MIN_BATCH  = 16      # CNN batch is halved on CUDA OOM down to this
BUDGET_H   = 8.5     # do not START a new architecture past this
ARCH_MAX_H = 1.5     # kill any single architecture that exceeds this
REPO_URL   = "github.com/Abhiram970/plant-disease-edge.git"

# Ordered so a truncated run still yields the informative ones. FastViT is near the front because it
# is the same architecture family as the MobileCLIP image encoder, which makes it the fairest
# supervised-versus-VLM comparison in the paper.
ARCHS = [
    "tf_efficientnetv2_s", "convnextv2_tiny", "resnet101", "densenet121",
    "fastvit_t8", "fastvit_sa12", "convnextv2_nano",
    "resnet50", "mobilenetv3_small_100", "mobilenetv4_conv_small",
    "mobilenetv3_large_100", "mobilenetv4_conv_medium", "efficientnet_b0", "regnety_040",
]
# ==========================================================================

T0 = time.time()
WORK = Path("/kaggle/working")
os.chdir(WORK)
elapsed_h = lambda: (time.time() - T0) / 3600
free_gb = lambda p="/kaggle/working": shutil.disk_usage(p).free / 1e9


def secret(name):
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return None


def sh(cmd, timeout_h):
    try:
        return subprocess.run(cmd, timeout=timeout_h * 3600).returncode
    except subprocess.TimeoutExpired:
        print(f"\n[TIMEOUT] killed after {timeout_h} h: {' '.join(map(str, cmd))}\n", flush=True)
        return 124


print(f"[start] free disk {free_gb():.1f} GB")

gh_tok = secret("GH_TOKEN")
REPO = WORK / "pde"
if not REPO.exists():
    url = f"https://{gh_tok}@{REPO_URL}" if gh_tok else f"https://{REPO_URL}"
    if subprocess.run(["git", "clone", "--depth", "1", url, str(REPO)]).returncode != 0:
        sys.exit("[fatal] clone failed -> check Internet=ON and the GH_TOKEN secret.")
print(f"[ok] repo at {REPO}")
S = REPO / "scripts"

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "timm", "open_clip_torch"])
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "safetensors", "pyyaml",
                "huggingface_hub", "pyarrow", "tqdm", "regex", "ftfy"])

os.environ["HF_HOME"] = "/kaggle/temp/hf"
os.environ["HF_HUB_CACHE"] = "/kaggle/temp/hf/hub"
Path("/kaggle/temp/hf/hub").mkdir(parents=True, exist_ok=True)
os.environ["PDE_DATA_ROOT"] = str(WORK)

# --------------------------------------------------------------- data + prior results
inp = Path("/kaggle/input")
attached = None
if inp.exists():
    for cand in sorted(inp.iterdir()):
        if (cand / "exp_data").is_dir():
            attached = cand / "exp_data"
            break
        if cand.is_dir() and any(cand.glob("*___*")):
            attached = cand
            break
if not attached:
    sys.exit("[fatal] no image dataset attached. Add data -> pde-sage-data.")
os.environ["PDE_DATASET_DIR"] = str(attached)
print(f"[ok] images at {attached} ({len(list(attached.glob('*___*')))} class folders)")

RESULTS = WORK / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
if inp.exists():
    for cand in inp.iterdir():
        src = cand / "results"
        if src.is_dir():
            for f in src.glob("*.json"):
                if not (RESULTS / f.name).exists():
                    shutil.copy(f, RESULTS / f.name)
            print(f"[ok] seeded prior results from {src}")
        ck = cand / "checkpoints"
        if ck.is_dir():
            shutil.copytree(ck, WORK / "checkpoints", dirs_exist_ok=True)
            print(f"[ok] restored checkpoints from {ck} -- interrupted architectures resume")

# The manifest stores ABSOLUTE paths, so it is rebuilt here rather than carried between runs.
sh([sys.executable, "-u", str(S / "build_manifest.py"), "--min-images", "25"], 0.5)
if not (WORK / "manifest.csv").exists():
    sys.exit("[fatal] no manifest.csv")

import torch
print(f"\n[env] cuda={torch.cuda.is_available()} "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''} "
      f"| free {free_gb():.1f} GB")

# --------------------------------------------------------------- the sweep
script = str(S / "supervised_baseline.py")
done, failed, skipped = [], [], []
for i, arch in enumerate(ARCHS):
    out = RESULTS / f"supervised_{arch}.json"
    if out.exists():
        print(f"[skip] {arch} (already complete)")
        continue
    if elapsed_h() > BUDGET_H:
        skipped = ARCHS[i:]
        print(f"\n[budget] {elapsed_h():.1f} h elapsed -- stopping cleanly so this session commits.")
        print(f"         Not started: {skipped}")
        break

    rc, batch = 1, BATCH
    while rc != 0 and batch >= MIN_BATCH:
        print(f"\n{'=' * 72}\n[cnn {i + 1}/{len(ARCHS)}] {arch}  (epochs={EPOCHS} batch={batch})  "
              f"t+{elapsed_h():.1f} h  free {free_gb():.1f} GB\n{'=' * 72}", flush=True)
        rc = sh([sys.executable, "-u", script, "--arch", arch, "--epochs", str(EPOCHS),
                 "--batch", str(batch), "--workers", str(WORKERS), "--resume"], ARCH_MAX_H)
        if rc != 0:
            batch //= 2
            if batch >= MIN_BATCH:
                print(f"  [retry] {arch} failed (likely CUDA OOM) -> batch {batch}", flush=True)
    (done if rc == 0 else failed).append(arch)

    # 14 checkpoints would otherwise eat several GB of the 20 GB output quota. Only delete the
    # checkpoint of an architecture that COMPLETED -- a failed one still needs it to resume.
    if rc == 0:
        ck = WORK / "checkpoints" / f"{arch}_ckpt.pt"
        if ck.exists():
            ck.unlink()

# --------------------------------------------------------------- summary
sup = []
for f in sorted(RESULTS.glob("supervised_*.json")):
    try:
        d = json.loads(f.read_text())
        sup.append((d.get("arch"), d.get("params_M"), d.get("seen_top1"),
                    d.get("seen_classes"), len(d.get("epoch_log") or [])))
    except Exception:
        pass
if sup:
    print(f"\n{'=' * 72}\nSUPERVISED BASELINES ({len(sup)}/{len(ARCHS)})\n{'=' * 72}")
    for a, p, acc, ncls, neps in sorted(sup, key=lambda r: -(r[2] or 0)):
        print(f"  {a:26s} {'  --  ' if p is None else f'{p:6.2f}M'}  {(acc or 0):.1%}"
              f"   classes={ncls} epochs={neps}")
    best = max(sup, key=lambda r: r[2] or 0)
    big = max(sup, key=lambda r: r[1] or 0)
    print(f"\n  best  {best[0]} at {best[1]:.1f}M -> {best[2]:.1%}")
    print(f"  largest {big[0]} at {big[1]:.1f}M -> {big[2]:.1%}   "
          f"({(best[2] - big[2]) * 100:+.1f} pp for {big[1] / best[1]:.1f}x the parameters)")

if done:
    print(f"\n  trained this session: {done}")
if failed:
    print(f"  STILL FAILING at batch {MIN_BATCH}: {failed}  (read the traceback above)")
if skipped:
    print(f"  NOT STARTED (budget): {skipped}")

print(f"""
{'=' * 72}
NEXT
{'=' * 72}
  {'Re-run with this output attached to finish: ' + str(skipped + failed)
     if (skipped or failed) else 'All 14 baselines complete.'}

  1. Output -> "Create Dataset" -> pde-results-3
  2. Download results/*.json into docs/paper/ and regenerate (nothing is typed by hand):
       python docs/paper/make_tables.py --write
       python docs/paper/make_tex_tables.py
       python docs/paper/make_figures.py
       python scripts/collect_results.py && python scripts/build_submission.py

  wall {elapsed_h():.2f} h | free disk {free_gb():.1f} GB
""")
