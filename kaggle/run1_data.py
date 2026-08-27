"""
KAGGLE RUN 1 (FALLBACK ONLY) — FETCH SAGE ON KAGGLE.  Paste this whole file as ONE cell.
========================================================================================

YOU PROBABLY DO NOT NEED THIS RUN.
----------------------------------
The primary path builds the dataset locally and uploads it, because the images are already on disk
at full resolution and the pinned SAGE release is 114 GB:

    python scripts/prepare_upload.py --dry-run     # check the plan
    python scripts/prepare_upload.py               # ~2.2 GB at 288 px
    then upload it to Kaggle as the dataset  pde-sage-data

Use THIS notebook only if you need a clean-room fetch straight from HuggingFace -- to prove the
build is reproducible from the published source, or because the local copy is gone.

WHAT IT FETCHES
  config.py pins SAGE to the MAY 2026 release (bc9bd2899f), which is what every published number in
  this paper was measured on: 13 shards, 114 GB, 8 held-out crops, 51 classes at scale C. The
  AUGUST 2026 release re-canonicalised the crop names and DROPPED Cotton entirely, which would give
  7 held-out crops and 48 classes. Set PDE_SAGE_REVISION to override; do not do so casually.

BUDGET: 114 GB in ~10.7 GB shards. At the ~6 MB/s an unauthenticated puller sees, that is far more
than one session. With HF_TOKEN it is roughly 1-3 h. Expect to run this notebook TWICE and let the
resume marker carry the fetch across sessions -- that is designed for, not a failure.

NOTEBOOK SETTINGS
  Accelerator = **None (CPU)**   <- on purpose: this stage never touches the GPU, and a CPU session
                                    does not spend the 30 h/week GPU quota that runs 2 and 3 need.
  Internet    = ON
  Persistence = Files only
  Secrets (Add-ons -> Secrets):
      HF_TOKEN  = a HuggingFace **read** token   (https://huggingface.co/settings/tokens)
      GH_TOKEN  = a GitHub fine-grained PAT, read-only Contents  (the repo is private)

Then: Save Version -> "Save & Run All (Commit)" -> close the tab.
When it finishes: Output -> "Create Dataset" -> name it **pde-sage-data**.

WHY THE PREVIOUS ATTEMPT DIED AT 12 h WITH NOTHING SAVED
--------------------------------------------------------
It spent 11.8 hours inside one call: a shard was requested at t+548 s and never returned a byte.
Four causes, all now fixed:

  1. A FLOATING REVISION. SHARD_REVISION was `refs/convert/parquet`, an auto-generated branch that
     HuggingFace REGENERATED when SAGE was reorganised on 2026-08-24. The run was resuming a
     May-built .shards_done.json against August data. It is now pinned to a commit SHA.
  2. NO TOKEN. The log carried "You are sending unauthenticated requests to the HF Hub". Anonymous
     pulls are rate-limited, and a throttled HF connection does not fail -- it stalls.
  3. NO DEADLINE. hf_hub_download has no usable wall-clock bound and its internal backoff will retry
     a dead endpoint indefinitely. Downloads now run in a child process that is killed at 15 min and
     retried, so one bad shard costs minutes, not a session.
  4. NO BUDGET. A cell that overruns the limit is killed and commits NOTHING. This run stops itself
     at BUDGET_H and exits cleanly with whatever it has; .shards_done.json resumes the rest.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

# ================================ SETTINGS ================================
BUDGET_H       = 8.5    # stop starting new shards after this; MUST stay under the session limit
SHARD_TIMEOUT  = 900    # kill + retry a shard download that stalls this many seconds
MAX_SIDE       = 288    # store at 288 px (everything trains at 224); full-res does not fit in 20 GB
ROLE           = "all"  # seen + held-out crops
MIN_IMAGES     = 60_000 # sanity floor; prepare_upload.py yields 84,123 over 18 crops
MIN_CROPS      = 18   # all 18 must appear; Cotton only exists in the pinned May release
REPO_URL       = "github.com/Abhiram970/plant-disease-edge.git"
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


print(f"[start] free disk {free_gb():.1f} GB")

# --------------------------------------------------------------- secrets
hf_tok = secret("HF_TOKEN")
if hf_tok:
    os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_tok
    print("[ok] HF_TOKEN loaded")
else:
    print("[WARN] no HF_TOKEN secret. The fetch will still run, but anonymous pulls are throttled "
          "and are what stalled the previous attempt. Add the secret and re-run if this is slow.")

gh_tok = secret("GH_TOKEN")

# --------------------------------------------------------------- repo
REPO = WORK / "pde"
if not REPO.exists():
    url = f"https://{gh_tok}@{REPO_URL}" if gh_tok else f"https://{REPO_URL}"
    if subprocess.run(["git", "clone", "--depth", "1", url, str(REPO)]).returncode != 0:
        sys.exit("[fatal] clone failed -> check Internet=ON and the GH_TOKEN secret.")
print(f"[ok] repo at {REPO}")
S = REPO / "scripts"

# --no-deps keeps pip from pulling a 2 GB torch wheel over Kaggle's working CUDA build.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "timm", "open_clip_torch"])
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "safetensors", "pyyaml",
                "huggingface_hub", "hf_transfer", "pyarrow", "tqdm", "regex", "ftfy", "pillow"])

# --------------------------------------------------------------- paths
# HF blobs go to /kaggle/temp (73 GB scratch), NOT the ~20 GB working volume. Without this the
# fetch fills the disk partway through and dies.
os.environ["HF_HOME"] = "/kaggle/temp/hf"
os.environ["HF_HUB_CACHE"] = "/kaggle/temp/hf/hub"
Path("/kaggle/temp/hf/hub").mkdir(parents=True, exist_ok=True)
os.environ["PDE_DATA_ROOT"] = str(WORK)
os.environ["PDE_DATASET_DIR"] = str(WORK / "exp_data")
DATA = WORK / "exp_data"
DATA.mkdir(parents=True, exist_ok=True)

# Resume from a previous run's output if it was attached: copy the class folders forward so this
# run only downloads the shards the last one did not reach.
inp = Path("/kaggle/input")
if inp.exists():
    for cand in sorted(inp.iterdir()):
        src = cand / "exp_data" if (cand / "exp_data").is_dir() else cand
        if src.is_dir() and any(src.glob("*___*")):
            n = len(list(src.glob("*___*")))
            print(f"[resume] copying {n} class folders from {src} ...", flush=True)
            subprocess.run(["cp", "-rn", f"{src}/.", str(DATA)])
            mk = src / ".shards_done.json"
            if mk.exists():
                shutil.copy(mk, DATA / ".shards_done.json")
                print(f"[resume] {len(json.loads(mk.read_text()))} shards already done")
            break

# --------------------------------------------------------------- fetch
# Read the pin from the repo rather than restating it here, so this notebook can never disagree
# with config.py about which SAGE release the study is built on.
sys.path.insert(0, str(S))
import config as _C
REV, N_SHARDS = _C.SHARD_REVISION, _C.N_SHARDS
print(f"\n{'=' * 72}\n[fetch] SAGE rev {REV[:10]} ({N_SHARDS} shards) role={ROLE} "
      f"max_side={MAX_SIDE} budget={BUDGET_H} h\n{'=' * 72}", flush=True)
subprocess.run([sys.executable, "-u", str(S / "sage_data.py"),
                "--role", ROLE, "--max-side", str(MAX_SIDE),
                "--budget-h", str(BUDGET_H), "--shard-timeout", str(SHARD_TIMEOUT)])

# --------------------------------------------------------------- verify
per_crop, per_class, n_images = Counter(), {}, 0
for d in sorted(DATA.iterdir()):
    if d.is_dir() and "___" in d.name:
        k = len(list(d.glob("*.jpg")))
        per_crop[d.name.split("___", 1)[0]] += k
        per_class[d.name] = k
        n_images += k

done = []
mk = DATA / ".shards_done.json"
if mk.exists():
    done = json.loads(mk.read_text())

print(f"\n{'=' * 72}\n[data] {n_images:,} images | {len(per_crop)} crops | {len(per_class)} classes"
      f" | shards {len(done)}/{N_SHARDS} | t+{elapsed_h():.1f} h\n{'=' * 72}")
for crop, k in sorted(per_crop.items(), key=lambda kv: -kv[1]):
    ncls = sum(1 for c in per_class if c.startswith(crop + "___"))
    print(f"   {crop:12s} {k:7,d} imgs  {ncls:3d} classes")

sizes = {"exp_data GB": sum(f.stat().st_size for f in DATA.rglob("*.jpg")) / 1e9}
print(f"\n   on disk: {sizes['exp_data GB']:.2f} GB | free {free_gb():.1f} GB")

ok = n_images >= MIN_IMAGES and len(per_crop) >= MIN_CROPS and len(done) >= N_SHARDS
if not ok:
    print(f"\n[INCOMPLETE] {n_images:,} images / {len(per_crop)} crops / {len(done)} of 48 shards.")
    print("             Nothing is lost. Commit this version, publish the output as a dataset,")
    print("             attach it to a fresh copy of this notebook and run again -- the fetch")
    print("             resumes at the first shard not in .shards_done.json.")
else:
    # The manifest stores ABSOLUTE paths, so it is always rebuilt where it will be used, never
    # carried between runs.
    subprocess.run([sys.executable, "-u", str(S / "build_manifest.py"), "--min-images", "25"])
    print("\n[OK] dataset complete.")

# Keep the resume marker and a machine-readable summary inside the output folder.
(WORK / "fetch_summary.json").write_text(json.dumps({
    "images": n_images, "crops": len(per_crop), "classes": len(per_class),
    "shards_done": len(done), "shards_total": N_SHARDS, "complete": bool(ok),
    "sage_revision": REV,
    "per_crop": dict(per_crop), "per_class": per_class,
    "max_side": MAX_SIDE, "hours": round(elapsed_h(), 2)}, indent=2))

print(f"""
{'=' * 72}
NEXT
{'=' * 72}
  1. Save Version -> Save & Run All (Commit)   (if you have not already)
  2. Output tab -> "Create Dataset" -> name it exactly:  pde-sage-data
  3. {'Run 2 (kaggle/run2_experiments.py): attach pde-sage-data, GPU T4 x2.'
       if ok else 'Re-run THIS notebook with pde-sage-data attached to finish the fetch.'}

  wall {elapsed_h():.2f} h | free disk {free_gb():.1f} GB
""")
