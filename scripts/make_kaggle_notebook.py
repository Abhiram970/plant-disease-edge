"""
Generate kaggle/run_everything.ipynb — one notebook that reproduces the whole study on Kaggle.

Writing the notebook from a script (rather than committing hand-edited JSON) keeps the cell contents
reviewable in a normal diff and lets the paths and architecture list stay in one place.

    python scripts/make_kaggle_notebook.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

OUT = C.REPO_ROOT / "kaggle" / "run_everything.ipynb"

MD_INTRO = """# plant-disease-edge — full reproduction on Kaggle

Runs every experiment in the paper end to end. **Set these before you start:**

| Setting | Value |
|---|---|
| Accelerator | **GPU T4 x2** (or P100) |
| Internet | **ON** — the clone and the dataset fetch both need it |
| Persistence | Files only |

**Add-ons → Secrets:** add `GH_TOKEN`, a GitHub fine-grained token with read-only Contents.
The repository is private, so an anonymous clone returns 404.

**Save yourself 40 minutes:** if you already published the image subset as a Kaggle Dataset, attach
it with *Add data*. Cell 3 finds any attached folder containing `exp_data` and skips the fetch.

Then use **Save Version → Save & Run All (Commit)** and close the tab. Nothing here needs babysitting.

### Runtime
Roughly 9–11 h for everything. The stage flags in Cell 2 let you run a subset; every stage skips
work whose result file already exists, so re-running after a timeout resumes rather than restarts."""

CELL_CONFIG = '''# ============================== WHAT TO RUN ==============================
# Each stage is skipped if its result files already exist, so a re-run resumes.
RUN_FETCH      = True   # build the image subset from SAGE          (~40 min, or 0 if attached)
RUN_ZEROSHOT   = True   # cross-crop zero-shot, scales A/B/C        (~1.5 h)
RUN_ABSTAIN    = True   # top-5 + risk-coverage, scales A/B/C       (~1 h)
RUN_PROBE      = True   # seen-crop linear probe, scales A/B/C      (~1 h)
RUN_LOCO       = True   # leave-one-crop-out + bootstrap CIs        (~20 min)
RUN_CNNS       = True   # 14 supervised baselines                   (~8 h)  <- the long one
RUN_CLEAN_EVAL = True   # label-corrected sensitivity run           (~30 min)

EPOCHS, BATCH, WORKERS = 8, 128, 4
MIN_BATCH  = 16     # CNN batch is halved on CUDA OOM down to this
MAX_SIDE   = 288    # store images at 288 px; models train at 224
BUDGET_H   = 8.5    # stop starting new architectures after this many hours
# =========================================================================

import json, os, shutil, subprocess, sys, time
from pathlib import Path

T0 = time.time()
WORK = Path("/kaggle/working")
os.chdir(WORK)

def sh(cmd, **kw):
    return subprocess.run(cmd, **kw).returncode

def free_gb(p="/kaggle/working"):
    return shutil.disk_usage(p).free / 1e9

def elapsed_h():
    return (time.time() - T0) / 3600

print(f"free disk: {free_gb():.1f} GB")'''

CELL_REPO = '''# --- clone the repo (private -> needs the GH_TOKEN secret) ---------------------
tok = None
try:
    from kaggle_secrets import UserSecretsClient
    tok = UserSecretsClient().get_secret("GH_TOKEN")
except Exception as e:
    print(f"[warn] no GH_TOKEN secret ({type(e).__name__}); an anonymous clone only works "
          f"if the repository is public.")

REPO = WORK / "pde"
if not REPO.exists():
    url = (f"https://{tok}@github.com/Abhiram970/plant-disease-edge.git" if tok
           else "https://github.com/Abhiram970/plant-disease-edge.git")
    if sh(["git", "clone", "--depth", "1", url, str(REPO)]) != 0:
        raise SystemExit("[fatal] clone failed -> check Internet=ON and the GH_TOKEN secret")
print(f"[ok] repo at {REPO}")

# --no-deps keeps pip from pulling a 2 GB torch wheel that can break Kaggle's CUDA build
sh([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "timm", "open_clip_torch"])
sh([sys.executable, "-m", "pip", "install", "-q",
    "safetensors", "pyyaml", "huggingface_hub", "pyarrow", "tqdm", "regex", "ftfy"])

import torch
print(f"cuda={torch.cuda.is_available()} "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")'''

CELL_DATA = '''# --- data: attach if available, otherwise fetch from HuggingFace ---------------
# HF blobs go to /kaggle/temp (73 GB scratch), NOT the ~20 GB working volume. Without this the
# fetch fills the disk partway through and dies.
os.environ["HF_HOME"] = "/kaggle/temp/hf"
os.environ["HF_HUB_CACHE"] = "/kaggle/temp/hf/hub"
Path("/kaggle/temp/hf/hub").mkdir(parents=True, exist_ok=True)
os.environ["PDE_DATA_ROOT"] = str(WORK)

attached = None
inp = Path("/kaggle/input")
if inp.exists():
    for cand in sorted(inp.iterdir()):
        if (cand / "exp_data").is_dir():
            attached = cand / "exp_data"; break
        if cand.is_dir() and any(cand.glob("*___*")):
            attached = cand; break

if attached:
    os.environ["PDE_DATASET_DIR"] = str(attached)
    print(f"[ok] using attached images at {attached} "
          f"({len(list(attached.glob('*___*')))} class folders) -- skipping fetch")
else:
    os.environ["PDE_DATASET_DIR"] = str(WORK / "exp_data")
    if RUN_FETCH:
        print("[data] fetching SAGE -- expect 40-90 min", flush=True)
        sh([sys.executable, "-u", str(REPO / "scripts" / "sage_data.py"),
            "--role", "all", "--max-side", str(MAX_SIDE)])

# Always rebuild: the manifest stores ABSOLUTE paths, so one built elsewhere points at files that
# are not here.
sh([sys.executable, "-u", str(REPO / "scripts" / "build_manifest.py"), "--min-images", "25"])
if not (WORK / "manifest.csv").exists():
    raise SystemExit("[fatal] no manifest.csv -- check the fetch log above")

RESULTS = WORK / "results"; RESULTS.mkdir(parents=True, exist_ok=True)
# seed any results carried in from a previous session's output
for cand in (inp.iterdir() if inp.exists() else []):
    src = cand / "results"
    if src.is_dir():
        for f in src.glob("*.json"):
            if not (RESULTS / f.name).exists():
                shutil.copy(f, RESULTS / f.name)
        print(f"[ok] seeded prior results from {src}")

data_gb = sum(f.stat().st_size for f in Path(os.environ["PDE_DATASET_DIR"]).rglob("*")
              if f.is_file()) / 1e9 if Path(os.environ["PDE_DATASET_DIR"]).exists() else 0
print(f"[data] images {data_gb:.2f} GB | free {free_gb():.1f} GB | t+{elapsed_h():.1f} h")'''

CELL_ZS = '''# --- zero-shot scale study + abstention (scales A, B, C) -----------------------
S = REPO / "scripts"
for exp in ["A", "B", "C"]:
    if RUN_ZEROSHOT and not (RESULTS / f"zeroshot_eval_{exp}.json").exists():
        print(f"\\n=== zero-shot {exp} === t+{elapsed_h():.1f} h", flush=True)
        sh([sys.executable, "-u", str(S / "evaluate.py"), "--exp", exp,
            "--strategies", "bare", "crude", "rich", "grounded", "--heavy", "--teachers"])
    if RUN_ABSTAIN and not (RESULTS / f"metrics_abstain_{exp}.json").exists():
        print(f"\\n=== abstain {exp} === t+{elapsed_h():.1f} h", flush=True)
        sh([sys.executable, "-u", str(S / "metrics.py"), "--exp", exp,
            "--strategies", "rich", "grounded", "--reference"])

# Label-corrected sensitivity run: merges SAGE's duplicate disease labels and drops the
# non-disease ones. Writes a SEPARATE *_clean.json so the as-published numbers stay intact.
if RUN_CLEAN_EVAL and not (RESULTS / "zeroshot_eval_C_clean.json").exists():
    print(f"\\n=== zero-shot C (label-corrected) === t+{elapsed_h():.1f} h", flush=True)
    sh([sys.executable, "-u", str(S / "evaluate.py"), "--exp", "C", "--clean",
        "--strategies", "bare", "crude", "rich", "grounded", "--heavy", "--teachers"])'''

CELL_PROBE = '''# --- seen-crop probe (all encoders, all scales) and leave-one-crop-out ---------
if RUN_PROBE:
    print(f"\\n=== seen-crop probe === t+{elapsed_h():.1f} h", flush=True)
    # probe_seen_all embeds the config-C pool ONCE per encoder and derives A and B by subsetting,
    # instead of re-encoding the same images three times. It also caches embeddings, so an
    # interrupted run resumes.
    sh([sys.executable, "-u", str(S / "probe_seen_all.py"), "--workers", str(WORKERS)])

if RUN_LOCO and not (RESULTS / "loco_s0_rich.json").exists():
    print(f"\\n=== leave-one-crop-out === t+{elapsed_h():.1f} h", flush=True)
    sh([sys.executable, "-u", str(S / "loco.py"), "--model", "s0",
        "--strategy", "rich", "--bootstrap", "2000"])'''

CELL_CNN = '''# --- 14 supervised CNN baselines, one protocol --------------------------------
# Ordered so a truncated run still yields the informative ones. FastViT is near the front because
# it is the same architecture family as the MobileCLIP image encoder, which makes it the fairest
# supervised-vs-VLM comparison in the paper.
ARCHS = [
    "tf_efficientnetv2_s", "convnextv2_tiny", "resnet101", "densenet121",
    "fastvit_t8", "fastvit_sa12", "convnextv2_nano",
    "resnet50", "mobilenetv3_small_100", "mobilenetv4_conv_small",
    "mobilenetv3_large_100", "mobilenetv4_conv_medium", "efficientnet_b0", "regnety_040",
]

if RUN_CNNS:
    script = str(S / "supervised_baseline.py")
    for arch in ARCHS:
        out = RESULTS / f"supervised_{arch}.json"
        if out.exists():
            print(f"[skip] {arch}"); continue
        if elapsed_h() > BUDGET_H:
            print(f"[budget] {elapsed_h():.1f} h elapsed -- stopping. Re-run to continue.")
            break
        # A T4 has 14.56 GB and batch 128 does not fit the heavier models; halve and retry
        # instead of losing the architecture to an OOM.
        rc, batch = 1, BATCH
        while rc != 0 and batch >= MIN_BATCH:
            print(f"\\n=== {arch} (batch {batch}) === t+{elapsed_h():.1f} h", flush=True)
            rc = sh([sys.executable, "-u", script, "--arch", arch, "--epochs", str(EPOCHS),
                     "--batch", str(batch), "--workers", str(WORKERS), "--resume"])
            if rc != 0:
                batch //= 2
                if batch >= MIN_BATCH:
                    print(f"  [retry] likely CUDA OOM -> batch {batch}", flush=True)
        if rc == 0:
            ck = WORK / "checkpoints" / f"{arch}_ckpt.pt"
            if ck.exists():
                ck.unlink()   # frees several GB across 14 architectures'''

CELL_SUMMARY = '''# --- summary -------------------------------------------------------------------
print("=" * 74)
print("RESULT FILES")
print("=" * 74)
for f in sorted(RESULTS.glob("*.json")):
    print(f"  {f.name:44s} {f.stat().st_size/1024:6.1f} KB")

sup = []
for f in sorted(RESULTS.glob("supervised_*.json")):
    try:
        d = json.loads(f.read_text())
        sup.append((d["arch"], d.get("params_M"), d.get("seen_top1")))
    except Exception:
        pass
if sup:
    print("\\n" + "=" * 74)
    print(f"SUPERVISED BASELINES ({len(sup)}/14)")
    print("=" * 74)
    for a, p, acc in sorted(sup, key=lambda r: -(r[2] or 0)):
        print(f"  {a:26s} {('--' if p is None else f'{p:6.2f}M')}  {acc:.1%}")

for e in "ABC":
    p = RESULTS / f"zeroshot_eval_{e}.json"
    if p.exists():
        d = json.loads(p.read_text())
        ms = [v for k, v in d["models"].items() if "SigLIP" not in k]
        r = sum(m["rich"]["acc"] for m in ms) / len(ms)
        g = sum(m["grounded"]["acc"] for m in ms) / len(ms)
        print(f"\\n  scale {e}: {d['n_classes']} unseen classes, chance {d['chance']:.1%}  "
              f"rich {r:.1%}  grounded {g:.1%}  delta {(g-r)*100:+.1f}pp")

print(f"\\nwall time: {elapsed_h():.2f} h | free disk {free_gb():.1f} GB")'''

MD_AFTER = """## When it finishes

1. **Output → Download** `results/*.json` into `docs/paper/` locally.
2. Regenerate everything — nothing is typed by hand:

```
python docs/paper/make_tables.py --write
python docs/paper/make_tex_tables.py
python docs/paper/make_figures.py
python scripts/collect_results.py
python scripts/build_submission.py
```

3. **Output → Create Dataset**, name it `pde-sage-data`. Attach it next time and the 40-minute
   fetch disappears.

### If it stops early
Re-run the same notebook. Every stage skips work whose result file already exists, the embedding
cache resumes mid-encoder, and CNNs resume from their last epoch checkpoint.

### Multi-seed (the one experiment the paper still lacks)
The headline claim — source-grounded descriptors beat hand-curated ones by 4.3 points at 51 unseen
classes — rests on a single seed. Neither `evaluate.py` nor `metrics.py` exposes a `--seed` flag yet;
adding one and running scales 0/1/2 would turn that point estimate into a mean with an error bar,
which is the most valuable few hours left in the project."""


def cell(src, kind="code"):
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}


def main():
    nb = {
        "cells": [
            cell(MD_INTRO, "markdown"),
            cell(CELL_CONFIG), cell(CELL_REPO), cell(CELL_DATA),
            cell(CELL_ZS), cell(CELL_PROBE), cell(CELL_CNN), cell(CELL_SUMMARY),
            cell(MD_AFTER, "markdown"),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"[nb] wrote {OUT} ({len(nb['cells'])} cells, {OUT.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
