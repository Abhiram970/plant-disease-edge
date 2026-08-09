"""
KAGGLE — finish the 4 architectures the first sweep did not complete (paste as ONE cell).
==========================================================================================

Run 1 completed 10 of 14. These four remain:

    tf_efficientnetv2_s   CUDA OOM at batch 128 on a T4 (14.56 GB)
    convnextv2_tiny       CUDA OOM at batch 128
    resnet101             CUDA OOM at batch 128
    densenet121           never started -- 10.5 h budget ran out

So this starts at batch 64 and halves on failure down to 16. Everything else matches run 1 exactly
(166 classes, 8 epochs) so the results go in the same table.

NOTEBOOK SETTINGS: Accelerator = GPU T4 x2 (or P100) · Internet = ON · Persistence = Files only
SECRET (Add-ons -> Secrets): GH_TOKEN = GitHub fine-grained PAT, read-only Contents.

SAVE 40 MINUTES -- ATTACH THE DATA (optional but recommended):
  Open run 1's notebook -> Output -> "Create Dataset" (name it e.g. pde-sage-data), then in THIS
  notebook use Add data to attach it. The script finds any attached folder containing exp_data and
  skips the HuggingFace fetch entirely. Without it, it refetches (~40 min) which is still fine.

  Note it rebuilds manifest.csv rather than reusing the attached one: the manifest stores ABSOLUTE
  image paths, and an attached dataset mounts at a different path than it was built at, so a copied
  manifest would point at files that are not there.

Then: Save Version -> "Save & Run All (Commit)" and close the tab.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ----------------------------------------------------------------- configuration
ARCHS = ["tf_efficientnetv2_s", "convnextv2_tiny", "resnet101", "densenet121"]
EPOCHS = 8           # MUST match run 1 or the table is not a comparison
BATCH = 64           # run 1 used 128 and these four OOMed; halves automatically on failure
MIN_BATCH = 16
WORKERS = 4
BUDGET_H = 8.0       # checked BEFORE each arch, so the run can overshoot by one arch's length
MAX_SIDE = 288       # must match run 1: images were stored at 288 px
ROLE = "all"

T0 = time.time()
WORK = Path("/kaggle/working")
os.chdir(WORK)


def sh(cmd):
    return subprocess.run(cmd).returncode


def free_gb(p="/kaggle/working"):
    return shutil.disk_usage(p).free / 1e9


# ----------------------------------------------------------------- repo
tok = None
try:
    from kaggle_secrets import UserSecretsClient
    tok = UserSecretsClient().get_secret("GH_TOKEN")
except Exception as e:
    print(f"[warn] no GH_TOKEN secret ({type(e).__name__}); anonymous clone needs a public repo.")

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

# ----------------------------------------------------------------- data
os.environ["HF_HOME"] = "/kaggle/temp/hf"
os.environ["HF_HUB_CACHE"] = "/kaggle/temp/hf/hub"
Path("/kaggle/temp/hf/hub").mkdir(parents=True, exist_ok=True)
os.environ["PDE_DATA_ROOT"] = str(WORK)

# Any attached dataset that contains an exp_data/ folder will do -- run 1's output, pde-sage-data,
# whatever it was named.
attached = None
inp = Path("/kaggle/input")
if inp.exists():
    for cand in sorted(inp.iterdir()):
        if (cand / "exp_data").is_dir():
            attached = cand / "exp_data"
            break
        if cand.is_dir() and any(cand.glob("*___*")):     # dataset published as the class folders
            attached = cand
            break

if attached:
    os.environ["PDE_DATASET_DIR"] = str(attached)
    n = len(list(attached.glob("*___*")))
    print(f"[ok] using attached images at {attached}  ({n} class folders) -- skipping fetch")
else:
    os.environ["PDE_DATASET_DIR"] = str(WORK / "exp_data")
    print("[data] no attached dataset found -> fetching SAGE (~40 min).", flush=True)
    sh([sys.executable, "-u", str(repo / "scripts" / "sage_data.py"),
        "--role", ROLE, "--max-side", str(MAX_SIDE)])

# Always rebuild: the manifest stores absolute paths, so one built elsewhere would be wrong here.
sh([sys.executable, "-u", str(repo / "scripts" / "build_manifest.py"), "--min-images", "25"])
if not (WORK / "manifest.csv").exists():
    sys.exit("[fatal] no manifest.csv -- check the fetch/attach step above.")

# Seed previously completed results so finished archs are skipped rather than retrained.
results = WORK / "results"
results.mkdir(parents=True, exist_ok=True)
if inp.exists():
    for cand in inp.iterdir():
        src = cand / "results"
        if src.is_dir():
            for f in src.glob("supervised_*.json"):
                if not (results / f.name).exists():
                    shutil.copy(f, results / f.name)
            print(f"[ok] seeded prior results from {src}")

import torch
print(f"[env] cuda={torch.cuda.is_available()} "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''} "
      f"| free={free_gb():.1f} GB")

# ----------------------------------------------------------------- the four
script = str(repo / "scripts" / "supervised_baseline.py")
done, failed, skipped = [], [], []

for i, arch in enumerate(ARCHS):
    out = results / f"supervised_{arch}.json"
    if out.exists():
        print(f"[skip] {arch} (already complete)")
        continue
    if (time.time() - T0) / 3600 > BUDGET_H:
        skipped = ARCHS[i:]
        print(f"[budget] stopping cleanly. Not started: {skipped}")
        break

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

    if rc == 0:
        ck = WORK / "checkpoints" / f"{arch}_ckpt.pt"
        if ck.exists():
            ck.unlink()

# ----------------------------------------------------------------- summary
print(f"\n{'=' * 72}\nALL SUPERVISED BASELINES — seen top-1 (166 classes, {EPOCHS} epochs)\n{'=' * 72}")
rows = []
for p in sorted(results.glob("supervised_*.json")):
    try:
        d = json.loads(p.read_text())
        rows.append((d.get("arch"), d.get("params_M"), d.get("seen_top1"),
                     d.get("seen_classes"), len(d.get("epoch_log") or [])))
    except Exception:
        pass
for arch, pm, acc, ncls, neps in sorted(rows, key=lambda r: -(r[2] or 0)):
    p = "  --  " if pm is None else f"{pm:6.2f}M"
    print(f"  {arch:26s} {p}  {acc:.1%}   classes={ncls} epochs={neps}")
print(f"\n  this session: {done}")
if failed:
    print(f"  STILL FAILING at batch {MIN_BATCH}: {failed}  (read the traceback above)")
if skipped:
    print(f"  NOT STARTED: {skipped}")
print(f"  wall: {(time.time() - T0) / 3600:.2f} h")
print("\nDownload results/supervised_*.json -> docs/paper/, then locally:")
print("  python docs/paper/make_tables.py --write && python docs/paper/make_tex_tables.py")
