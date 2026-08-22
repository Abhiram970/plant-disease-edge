"""
KAGGLE — reproduce the entire study in one run (paste this whole file as ONE cell).
====================================================================================

Runs: SAGE fetch -> cross-crop zero-shot at three scales -> abstention/top-5 -> seen-crop probe ->
leave-one-crop-out -> label-corrected sensitivity run -> all 14 supervised CNN baselines.

NOTEBOOK SETTINGS: Accelerator = GPU T4 x2 (or P100) · Internet = ON · Persistence = Files only
SECRET (Add-ons -> Secrets): GH_TOKEN = GitHub fine-grained PAT, read-only Contents.
Then: Save Version -> "Save & Run All (Commit)" and close the tab.

SAVE 40 MINUTES — ATTACH THE DATA (optional):
  Open a previous run -> Output -> "Create Dataset" (e.g. pde-sage-data), then here use Add data to
  attach it. Any attached folder containing exp_data/ is found automatically and the fetch is
  skipped. manifest.csv is ALWAYS rebuilt rather than reused, because it stores absolute image
  paths and an attached dataset mounts at a different path than it was built at.

RESUMING: every stage skips work whose result file already exists, the embedding cache resumes
mid-encoder, and CNNs resume from their last epoch checkpoint. If the 9 h session wall or the
budget stops the run, just run it again — nothing is redone.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

# ============================== WHAT TO RUN ==============================
RUN_FETCH      = True   # build the image subset from SAGE        (~40-90 min, 0 if attached)
RUN_ZEROSHOT   = True   # cross-crop zero-shot, scales A/B/C      (~1.5 h)
RUN_ABSTAIN    = True   # top-5 + risk-coverage, scales A/B/C     (~1 h)
RUN_PROBE      = True   # seen-crop linear probe, scales A/B/C    (~1 h)
RUN_LOCO       = True   # leave-one-crop-out + bootstrap CIs      (~20 min)
RUN_CLEAN_EVAL = True   # label-corrected sensitivity run         (~30 min)
RUN_UNGROUNDED = True   # ungrounded-LLM control arm, 3 seeds     (~1 h, needs the descriptors)
UNGROUNDED_SEEDS = [0, 1, 2]
RUN_CNNS       = True   # 14 supervised baselines                 (~8 h)  <- the long one

EPOCHS, BATCH, WORKERS = 8, 128, 4
MIN_BATCH = 16          # CNN batch is halved on CUDA OOM down to this
MAX_SIDE  = 288         # store images at 288 px; every model trains at 224
BUDGET_H  = 8.5         # stop starting new architectures past this
ROLE      = "all"       # fetch seen + held-out crops

# Sanity floor for the fetch. The reference build is 86,199 images over 18 crops; anything far
# below that means shards were missed and the downstream class counts will not match the paper.
MIN_EXPECTED_IMAGES = 60_000
MIN_EXPECTED_CROPS = 16
# =========================================================================

T0 = time.time()
WORK = Path("/kaggle/working")
os.chdir(WORK)


def sh(cmd):
    return subprocess.run(cmd).returncode


def free_gb(p="/kaggle/working"):
    return shutil.disk_usage(p).free / 1e9


def elapsed_h():
    return (time.time() - T0) / 3600


print(f"[start] free disk {free_gb():.1f} GB")

# ----------------------------------------------------------------- repo
tok = None
try:
    from kaggle_secrets import UserSecretsClient
    tok = UserSecretsClient().get_secret("GH_TOKEN")
except Exception as e:
    print(f"[warn] no GH_TOKEN secret ({type(e).__name__}); an anonymous clone only works if the "
          f"repository is public.")

REPO = WORK / "pde"
if not REPO.exists():
    url = (f"https://{tok}@github.com/Abhiram970/plant-disease-edge.git" if tok
           else "https://github.com/Abhiram970/plant-disease-edge.git")
    if sh(["git", "clone", "--depth", "1", url, str(REPO)]) != 0:
        sys.exit("[fatal] clone failed -> check Internet=ON and the GH_TOKEN secret.")
print(f"[ok] repo at {REPO}")

S = REPO / "scripts"

# --no-deps keeps pip from pulling a 2 GB torch wheel that can break Kaggle's CUDA build.
sh([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "timm", "open_clip_torch"])
sh([sys.executable, "-m", "pip", "install", "-q",
    "safetensors", "pyyaml", "huggingface_hub", "pyarrow", "tqdm", "regex", "ftfy"])

# ----------------------------------------------------------------- data
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
            attached = cand / "exp_data"
            break
        if cand.is_dir() and any(cand.glob("*___*")):   # published as the class folders themselves
            attached = cand
            break

if attached:
    os.environ["PDE_DATASET_DIR"] = str(attached)
    print(f"[ok] using attached images at {attached} "
          f"({len(list(attached.glob('*___*')))} class folders) -- skipping fetch")
else:
    os.environ["PDE_DATASET_DIR"] = str(WORK / "exp_data")
    if RUN_FETCH:
        print(f"[data] fetching SAGE (role={ROLE}, max_side={MAX_SIDE}). Expect 40-90 min.",
              flush=True)
        sh([sys.executable, "-u", str(S / "sage_data.py"),
            "--role", ROLE, "--max-side", str(MAX_SIDE)])

# --- verify the fetch actually delivered before spending hours on top of it -------------------
DATA = Path(os.environ["PDE_DATASET_DIR"])
if not DATA.exists():
    sys.exit(f"[fatal] {DATA} does not exist -- the fetch produced nothing.")

per_crop = Counter()
n_images = 0
for d in DATA.iterdir():
    if d.is_dir() and "___" in d.name:
        k = len(list(d.glob("*.jpg")))
        per_crop[d.name.split("___", 1)[0]] += k
        n_images += k

print(f"\n[data] {n_images:,} images | {len(per_crop)} crops | "
      f"{sum(1 for d in DATA.iterdir() if d.is_dir() and '___' in d.name)} class folders")
for crop, k in sorted(per_crop.items(), key=lambda kv: -kv[1]):
    print(f"       {crop:12s} {k:6,d}")

if n_images < MIN_EXPECTED_IMAGES or len(per_crop) < MIN_EXPECTED_CROPS:
    sys.exit(f"[fatal] fetch is short: {n_images:,} images over {len(per_crop)} crops, expected at "
             f"least {MIN_EXPECTED_IMAGES:,} over {MIN_EXPECTED_CROPS}. Re-running continues from "
             f"the shard marker, so just run this cell again -- do NOT train on a partial pull, the "
             f"class counts will not match the paper.")

# Always rebuild: the manifest stores ABSOLUTE paths, so one built elsewhere points at files that
# are not here.
sh([sys.executable, "-u", str(S / "build_manifest.py"), "--min-images", "25"])
if not (WORK / "manifest.csv").exists():
    sys.exit("[fatal] no manifest.csv -- check the fetch/attach step above.")

RESULTS = WORK / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# Seed results carried in from a previous session's output so finished work is skipped.
if inp.exists():
    for cand in inp.iterdir():
        src = cand / "results"
        if src.is_dir():
            for f in src.glob("*.json"):
                if not (RESULTS / f.name).exists():
                    shutil.copy(f, RESULTS / f.name)
            print(f"[ok] seeded prior results from {src}")

import torch
print(f"\n[env] cuda={torch.cuda.is_available()} "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''} "
      f"| free {free_gb():.1f} GB | t+{elapsed_h():.1f} h")

# ----------------------------------------------------------------- zero-shot + abstention
# NOTE ON RE-RUNNING: the rich keyword matcher was normalising `base` but not the match key, so
# 13 of the 32 multi-word bank entries were unreachable and classes with a correct distinct entry
# fell through to a coarser key. That is fixed, which means every previously published `rich` number
# was produced by different code and must be re-measured. Delete stale zeroshot_eval_*.json before
# re-running, or the guard below will skip the very stages that need redoing.
STALE = [p for p in RESULTS.glob("zeroshot_eval_*.json")
         if not json.loads(p.read_text()).get("matcher_normalised")]
if STALE and RUN_ZEROSHOT:
    for p in STALE:
        p.rename(p.with_suffix(".json.pre_matcher_fix"))
    print(f"[rich-fix] set aside {len(STALE)} evaluation(s) produced by the old matcher; "
          f"they will be recomputed")

for exp in ["A", "B", "C"]:
    if RUN_ZEROSHOT and not (RESULTS / f"zeroshot_eval_{exp}.json").exists():
        print(f"\n{'=' * 72}\n[zero-shot {exp}]  t+{elapsed_h():.1f} h\n{'=' * 72}", flush=True)
        sh([sys.executable, "-u", str(S / "evaluate.py"), "--exp", exp,
            "--strategies", "bare", "crude", "rich", "grounded", "--heavy", "--teachers"])
    if RUN_ABSTAIN and not (RESULTS / f"metrics_abstain_{exp}.json").exists():
        print(f"\n{'=' * 72}\n[abstain {exp}]  t+{elapsed_h():.1f} h\n{'=' * 72}", flush=True)
        sh([sys.executable, "-u", str(S / "metrics.py"), "--exp", exp,
            "--strategies", "rich", "grounded", "--reference"])

# Merges SAGE's duplicate disease labels and drops the non-disease ones. Writes a SEPARATE
# *_clean.json so the as-published numbers are never overwritten.
if RUN_CLEAN_EVAL and not (RESULTS / "zeroshot_eval_C_clean.json").exists():
    print(f"\n{'=' * 72}\n[zero-shot C, label-corrected]  t+{elapsed_h():.1f} h\n{'=' * 72}",
          flush=True)
    sh([sys.executable, "-u", str(S / "evaluate.py"), "--exp", "C", "--clean",
        "--strategies", "bare", "crude", "rich", "grounded", "--heavy", "--teachers"])

# --- ungrounded control arm: same LLM, same schema, no citation constraint ---------------------
# This is the arm that decides whether the contribution is about GROUNDING or merely about per-class
# coverage. Several seeds, because one sample of LLM text cannot separate "grounding helps" from
# "that generation was lucky". Needs descriptors_ungrounded/<seed>/ in the repo -- generate it
# offline first with scripts/build_ungrounded.py, which requires an API key.
if RUN_UNGROUNDED:
    ung_root = REPO / "descriptors_ungrounded"
    have = [s for s in UNGROUNDED_SEEDS if (ung_root / str(s)).is_dir()]
    if not have:
        print("[ungrounded] SKIPPED -- no descriptors_ungrounded/<seed>/ in the repo. Generate it "
              "locally (needs an LLM key) and push before this arm can run:")
        print("             python scripts/build_ungrounded.py --seed 0 --which heldout")
    for s in have:
        out = RESULTS / f"zeroshot_eval_C_ung{s}.json"
        if out.exists():
            print(f"[skip] ungrounded seed {s}")
            continue
        print(f"\n{'=' * 72}\n[ungrounded seed {s}]  t+{elapsed_h():.1f} h\n{'=' * 72}", flush=True)
        sh([sys.executable, "-u", str(S / "evaluate.py"), "--exp", "C",
            "--strategies", "rich", "grounded", "ungrounded",
            "--ungrounded-seed", str(s), "--heavy"])

# ----------------------------------------------------------------- seen probe + LOCO
if RUN_PROBE:
    print(f"\n{'=' * 72}\n[seen-crop probe]  t+{elapsed_h():.1f} h\n{'=' * 72}", flush=True)
    # Embeds the config-C pool ONCE per encoder and derives A and B by subsetting, instead of
    # re-encoding the same images three times. Caches embeddings, so an interruption resumes.
    sh([sys.executable, "-u", str(S / "probe_seen_all.py"), "--workers", str(WORKERS)])

if RUN_LOCO and not (RESULTS / "loco_s0_rich.json").exists():
    print(f"\n{'=' * 72}\n[leave-one-crop-out]  t+{elapsed_h():.1f} h\n{'=' * 72}", flush=True)
    sh([sys.executable, "-u", str(S / "loco.py"), "--model", "s0",
        "--strategy", "rich", "--bootstrap", "2000"])

# ----------------------------------------------------------------- supervised CNN baselines
# Ordered so a truncated run still yields the informative ones. FastViT is near the front because
# it is the same architecture family as the MobileCLIP image encoder, which makes it the fairest
# supervised-versus-VLM comparison in the paper.
ARCHS = [
    "tf_efficientnetv2_s", "convnextv2_tiny", "resnet101", "densenet121",
    "fastvit_t8", "fastvit_sa12", "convnextv2_nano",
    "resnet50", "mobilenetv3_small_100", "mobilenetv4_conv_small",
    "mobilenetv3_large_100", "mobilenetv4_conv_medium", "efficientnet_b0", "regnety_040",
]

done, failed, skipped = [], [], []
if RUN_CNNS:
    script = str(S / "supervised_baseline.py")
    for i, arch in enumerate(ARCHS):
        out = RESULTS / f"supervised_{arch}.json"
        if out.exists():
            print(f"[skip] {arch} (already complete)")
            continue
        if elapsed_h() > BUDGET_H:
            skipped = ARCHS[i:]
            print(f"[budget] {elapsed_h():.1f} h elapsed -- stopping cleanly. "
                  f"Not started: {skipped}")
            break

        # A T4 has 14.56 GB and batch 128 does not fit the heavier models. Halve and retry rather
        # than lose the architecture to an OOM, which is what happened on the first sweep.
        rc, batch = 1, BATCH
        while rc != 0 and batch >= MIN_BATCH:
            print(f"\n{'=' * 72}\n[cnn] {arch}  (epochs={EPOCHS} batch={batch})  "
                  f"t+{elapsed_h():.1f} h  free {free_gb():.1f} GB\n{'=' * 72}", flush=True)
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
                ck.unlink()   # 14 checkpoints would otherwise eat several GB of the output quota

# ----------------------------------------------------------------- summary
print(f"\n{'=' * 72}\nRESULT FILES\n{'=' * 72}")
for f in sorted(RESULTS.glob("*.json")):
    print(f"  {f.name:44s} {f.stat().st_size/1024:7.1f} KB")

sup = []
for f in sorted(RESULTS.glob("supervised_*.json")):
    try:
        d = json.loads(f.read_text())
        sup.append((d.get("arch"), d.get("params_M"), d.get("seen_top1"),
                    d.get("seen_classes"), len(d.get("epoch_log") or [])))
    except Exception:
        pass
if sup:
    print(f"\n{'=' * 72}\nSUPERVISED BASELINES ({len(sup)}/14)\n{'=' * 72}")
    for a, p, acc, ncls, neps in sorted(sup, key=lambda r: -(r[2] or 0)):
        print(f"  {a:26s} {'  --  ' if p is None else f'{p:6.2f}M'}  {acc:.1%}"
              f"   classes={ncls} epochs={neps}")

for e in "ABC":
    p = RESULTS / f"zeroshot_eval_{e}.json"
    if p.exists():
        d = json.loads(p.read_text())
        ms = [v for k, v in d["models"].items() if "SigLIP" not in k]
        r = sum(m["rich"]["acc"] for m in ms) / len(ms)
        g = sum(m["grounded"]["acc"] for m in ms) / len(ms)
        print(f"\n  scale {e}: {d['n_classes']:3d} unseen classes  chance {d['chance']:.1%}  "
              f"rich {r:.1%}  grounded {g:.1%}  delta {(g - r) * 100:+.1f} pp")

ung = sorted(RESULTS.glob("zeroshot_eval_C_ung*.json"))
if ung:
    print(f"\n  UNGROUNDED CONTROL ARM ({len(ung)} seed(s)) -- the comparison that decides whether")
    print("  the contribution is grounding or per-class coverage:")
    for p in ung:
        d = json.loads(p.read_text())
        ms = [v for k, v in d["models"].items() if "SigLIP" not in k]
        for arm in ("rich", "grounded", "ungrounded"):
            if arm in ms[0]:
                v = sum(m[arm]["acc"] for m in ms) / len(ms)
                print(f"    {p.stem.split('_ung')[-1]:>4s} {arm:11s} {v:.1%}")

if done:
    print(f"\n  CNNs this session: {done}")
if failed:
    print(f"  STILL FAILING at batch {MIN_BATCH}: {failed}  (read the traceback above)")
if skipped:
    print(f"  NOT STARTED (budget): {skipped}  -- re-run this cell to continue")
print(f"\n  wall {elapsed_h():.2f} h | free disk {free_gb():.1f} GB")

print("""
NEXT
  1. Output -> download results/*.json into docs/paper/
  2. Locally, regenerate everything (nothing is typed by hand):
       python docs/paper/make_tables.py --write
       python docs/paper/make_tex_tables.py
       python docs/paper/make_figures.py
       python scripts/collect_results.py
       python scripts/build_submission.py
  3. Output -> Create Dataset ("pde-sage-data") so no future run refetches.
""")
