"""
KAGGLE — THE WHOLE STUDY IN ONE FILE.  Paste this entire file as ONE cell.
=========================================================================

Run it, let it commit, publish its Output as a Dataset, attach that to a fresh copy, run the SAME
file again. Two runs if you attach images; three if you let it fetch. Nothing is ever redone.

    NOTEBOOK SETTINGS
      Accelerator = GPU T4 x2      (see "CPU FIRST RUN" below if you are fetching)
      Internet    = ON
      Persistence = Files only

    SECRETS  (Add-ons -> Secrets)
      GH_TOKEN           GitHub fine-grained PAT, read-only Contents      (required, private repo)
      HF_TOKEN           HuggingFace READ token                           (required if fetching)
      LAVA_API_KEY       your Lava spend key            \\ either one; needed ONLY by the descriptor
      ANTHROPIC_API_KEY  sk-ant-...                     /  arms. Everything else runs without a key.

      Both descriptor arms are generated with claude-sonnet-5 (override with PDE_LLM_MODEL). The key
      is checked with one cheap call BEFORE the long stages, so a bad key or an unavailable model
      costs seconds rather than being discovered six GPU-hours in.

    ADD DATA
      Attach the previous run's output every time after the first. Images, results, ungrounded
      descriptors and CNN checkpoints are all picked up automatically from any attached dataset.

    Then: Save Version -> "Save & Run All (Commit)" -> close the tab.
    After it finishes: Output -> "Create Dataset". Attach that next time.

WHY IT IS SPLIT ACROSS RUNS AND NOT ONE LONG ONE
------------------------------------------------
The whole study is ~15 h of compute and a Kaggle cell is killed at 12 h. A killed cell commits
NOTHING -- including every stage that had already finished. The previous single-cell attempt lost a
full session that way: three shards arrived at 28-150 MB/s, then shard 0002 was requested at t+548 s
and had produced no bytes when the notebook was terminated 11.8 hours later. Nothing bounded that
request and nothing bounded the cell, so a single stalled socket cost everything. This file stops
itself at BUDGET_H, prints exactly what is left, and exits cleanly. Every stage writes its own result
file and is skipped next time; the probe embedding cache resumes mid-encoder; CNNs resume from their
last epoch checkpoint; the fetch resumes per shard, and a stalled shard is killed and retried.

CPU FIRST RUN (saves your GPU quota)
------------------------------------
If you have no images yet, run this file ONCE with Accelerator = **None (CPU)**. It will fetch the
data, skip everything that needs a GPU, and exit. Kaggle CPU sessions do not spend the 30 h/week GPU
quota that the experiments need. Then publish the output and re-run on GPU.
Better still: skip the fetch entirely by building the images locally, which takes ~25 min instead of
downloading 114 GB --  python scripts/prepare_upload.py  -- and uploading the 1.7 GB result.

THE DATASET WAS REWRITTEN UNDER THIS PAPER -- READ BEFORE CHANGING THE PIN
--------------------------------------------------------------------------
SAGE shipped two incompatible releases:

    2026-05-07  bc9bd2899f   13 shards, 114 GB   8 held-out crops   51 classes at scale C
    2026-08-24  dde0de8633   48 shards,  21 GB   7 held-out crops   48 classes at scale C

The August release is NOT a superset: its own canonical_mapping.json marks all 14 Cotton entries
"how": "no-canonical-crop", and the crop column of all 48 August shards contains zero Cotton rows.
Every published number here was measured with Cotton held out, so config.py pins the May commit SHA.
`refs/convert/parquet` is a FLOATING branch and now resolves to the August data, so pinning is what
keeps this reproducible. (The 12 h failure was not caused by that mismatch -- that run read May
throughout; see config.py. The pin prevents a future run from silently losing Cotton.)

WHAT THIS RUN IS ACTUALLY FOR
-----------------------------
The paper's headline -- "only source-grounded descriptors scale" -- is retracted and not yet
rewritten. `rich` is a keyword-retrieved bank in which, at scale C, only 8 of 51 held-out classes
get a unique descriptor: 17 fall back to the bare class name and 26 share text with another class.
Beating that measures per-class DISTINCTNESS, not grounding.

The ungrounded arm removes the confound -- same model, same schema, same fall-through, temperature
1.0, three seeds; the only difference is that the "cite a retrievable source" constraint is dropped.

    ungrounded ~= grounded  -> grounding is free and buys auditability. The cleaner paper.
    ungrounded <  grounded  -> the sourcing constraint itself helps. A real finding.

This run also re-measures `rich`. descriptors.text_for normalised underscores for the prompt but not
for the match key, so all 13 multi-word bank entries ("powdery mildew", "citrus canker", ...) were
unreachable for every label in the dataset. Every published `rich` accuracy came from that broken
matcher. Results are now stamped "matcher_normalised"; anything unstamped is set aside and
recomputed rather than skipped.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

# ============================== SETTINGS ==============================
BUDGET_H         = 8.5      # do not START a new stage past this. Must stay under the 12 h cell wall.
FETCH_BUDGET_H   = 7.0      # sub-budget for the fetch, so a slow pull cannot eat the whole session
SHARD_TIMEOUT    = 240      # kill + retry a stalled shard. Short by design: a shard that
                            # transfers lands in 7-48 s and one that stalls sends no bytes,
                            # so 900 s only burned 45 min per bad shard learning nothing.
STAGE_TIMEOUT_H  = 3.0      # no single stage may exceed this
ARCH_MAX_H       = 1.5      # no single CNN architecture may exceed this

MAX_SIDE         = 288      # store at 288 px; everything trains at 224, so more is bytes for nothing
UNGROUNDED_SEEDS = [0, 1, 2]
LLM_MODEL        = "claude-sonnet-5"   # both descriptor arms; see MATCHED_GROUNDED below
MATCHED_GROUNDED = True
# The shipped grounded registry records no generating model (217 records: crop, disease,
# symptom_text, fields, status -- no provenance). Comparing a freshly generated ungrounded set
# against text of unknown origin would confound GROUNDING with MODEL VERSION, which is the same
# class of confound that invalidated the original claim. So we regenerate a grounded set with the
# same model, in the same run, and evaluate it as `grounded_matched`. The shipped `grounded` arm is
# still evaluated, so the published numbers remain comparable; `grounded_matched` vs `ungrounded` is
# the clean test.
EPOCHS, BATCH, WORKERS = 8, 128, 4
MIN_BATCH        = 16       # CNN batch is halved on CUDA OOM down to this

ALLOW_FETCH      = False    # fetching is OPT-IN. With no images attached the run stops and tells
                            # you what /kaggle/input actually contains, instead of falling through
                            # to a 114 GB pull that cannot finish in a session.
MIN_IMAGES       = 60_000   # sanity floor; a complete build is 84,123 images
MIN_CROPS        = 18       # all 18; Cotton exists only in the pinned May release
REPO_URL         = "github.com/Abhiram970/plant-disease-edge.git"

ARCHS = [   # ordered so a truncated sweep still yields the informative ones. FastViT is early
            # because it is the MobileCLIP image encoder's architecture family, which makes it the
            # fairest supervised-versus-VLM comparison in the paper.
    "tf_efficientnetv2_s", "convnextv2_tiny", "resnet101", "densenet121",
    "fastvit_t8", "fastvit_sa12", "convnextv2_nano",
    "resnet50", "mobilenetv3_small_100", "mobilenetv4_conv_small",
    "mobilenetv3_large_100", "mobilenetv4_conv_medium", "efficientnet_b0", "regnety_040",
]
# ======================================================================

T0 = time.time()
WORK = Path("/kaggle/working")
os.chdir(WORK)
elapsed_h = lambda: (time.time() - T0) / 3600
free_gb = lambda p="/kaggle/working": shutil.disk_usage(p).free / 1e9
LEFT = []            # stages not done this session, reported at the end


def secret(name):
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return None


def sh(cmd, timeout_h=STAGE_TIMEOUT_H):
    """Run a stage under a hard deadline. A hung stage must cost one stage, not the session."""
    try:
        return subprocess.run(cmd, timeout=timeout_h * 3600).returncode
    except subprocess.TimeoutExpired:
        print(f"\n[TIMEOUT] stage exceeded {timeout_h} h and was killed\n", flush=True)
        return 124


def ok_to_start(stage, rest=()):
    """True if there is budget left to begin `stage`.

    `rest` is the remaining work in the same loop. Recording it matters: a bare `break` would report
    "zero-shot A" as the only thing left while silently omitting B and C, and the whole point of the
    end-of-run report is that you can trust it to say what is actually outstanding."""
    if elapsed_h() > BUDGET_H:
        LEFT.append(stage)
        LEFT.extend(rest)
        return False
    return True


def banner(s):
    print(f"\n{'=' * 74}\n[{s}]  t+{elapsed_h():.1f} h  free {free_gb():.1f} GB\n{'=' * 74}",
          flush=True)


print(f"[start] free disk {free_gb():.1f} GB")

# ---------------------------------------------------------------- secrets + repo
for k in ("GH_TOKEN", "HF_TOKEN", "LAVA_API_KEY", "ANTHROPIC_API_KEY",
          "LAVA_BASE_URL", "PDE_LLM_MODEL"):
    v = secret(k)
    if v:
        os.environ[k] = v
        print(f"[ok] secret {k}")
if os.environ.get("HF_TOKEN"):
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
os.environ.setdefault("PDE_LLM_MODEL", LLM_MODEL)
HAVE_LLM_KEY = bool(os.environ.get("LAVA_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))

REPO = WORK / "pde"
if not REPO.exists():
    gh = os.environ.get("GH_TOKEN")
    url = f"https://{gh}@{REPO_URL}" if gh else f"https://{REPO_URL}"
    if subprocess.run(["git", "clone", "--depth", "1", url, str(REPO)]).returncode != 0:
        sys.exit("[fatal] clone failed -> check Internet=ON and the GH_TOKEN secret.")
print(f"[ok] repo at {REPO}")
S = REPO / "scripts"

# --no-deps keeps pip from pulling a 2 GB torch wheel over Kaggle's working CUDA build.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "timm", "open_clip_torch"])
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "safetensors", "pyyaml",
                "huggingface_hub", "hf_transfer", "pyarrow", "tqdm", "regex", "ftfy", "pillow",
                "openai", "anthropic"])

# HF blobs go to /kaggle/temp (73 GB scratch), NOT the ~20 GB working volume. Without this a fetch
# fills the disk partway through and dies.
os.environ["HF_HOME"] = "/kaggle/temp/hf"
os.environ["HF_HUB_CACHE"] = "/kaggle/temp/hf/hub"
Path("/kaggle/temp/hf/hub").mkdir(parents=True, exist_ok=True)
os.environ["PDE_DATA_ROOT"] = str(WORK)

sys.path.insert(0, str(S))
# Load OUR config by file path rather than `import config`. Kaggle's image preloads packages that
# already occupy the bare name `config` in sys.modules (google-adk among them), and a plain import
# returns that cached module -- which produced
#     AttributeError: module 'config' has no attribute 'N_SHARDS'
# on a file that defines N_SHARDS perfectly well. sys.path.insert cannot help: it only affects
# modules not already imported. Evicting the name first, then loading from an explicit spec, makes
# the import unambiguous. The child scripts are unaffected -- they run in their own processes with
# scripts/ first on sys.path.
import importlib.util                                            # noqa: E402
sys.modules.pop("config", None)
_spec = importlib.util.spec_from_file_location("pde_config", S / "config.py")
CFG = importlib.util.module_from_spec(_spec)
sys.modules["pde_config"] = sys.modules["config"] = CFG          # so child imports resolve to ours
_spec.loader.exec_module(CFG)
print(f"[ok] SAGE pinned to {CFG.SHARD_REVISION[:12]} ({CFG.N_SHARDS} shards) "
      f"| {len(CFG.WANT_CROPS)} crops")

try:
    import torch
    HAS_GPU = torch.cuda.is_available()
    print(f"[env] cuda={HAS_GPU} {torch.cuda.get_device_name(0) if HAS_GPU else '(CPU session)'}")
except Exception:
    HAS_GPU = False
    print("[env] torch unavailable -> CPU-only stages")

# ---------------------------------------------------------------- carry work forward
RESULTS = WORK / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
DATA = WORK / "exp_data"
inp = Path("/kaggle/input")

def find_dirs(root, name, max_depth=5):
    """Every directory called `name` at or below `root`, searched breadth-first.

    Kaggle does NOT mount a dataset at a fixed depth. Attaching one can produce
    /kaggle/input/<dataset>/ or, as seen in practice, /kaggle/input/datasets/<owner>/<dataset>/.
    The old code only looked one level down, so a perfectly good dataset was invisible and the run
    fell through to a 114 GB fetch. Search instead of assuming."""
    found, frontier = [], [(root, 0)]
    while frontier:
        d, depth = frontier.pop(0)
        if depth > max_depth:
            continue
        try:
            kids = sorted(p for p in d.iterdir() if p.is_dir())
        except Exception:
            continue
        for k in kids:
            if k.name == name:
                found.append(k)
            else:
                frontier.append((k, depth + 1))
    return found


def find_images(root, max_depth=5):
    """First directory at or below `root` holding <Crop>___<Disease> class folders."""
    frontier = [(root, 0)]
    while frontier:
        d, depth = frontier.pop(0)
        if depth > max_depth:
            continue
        try:
            if any(d.glob("*___*")):
                return d
            frontier.extend((p, depth + 1) for p in sorted(d.iterdir()) if p.is_dir())
        except Exception:
            continue
    return None


attached_imgs = find_images(inp) if inp.exists() else None
if attached_imgs is not None:
    print(f"[found] image dataset at {attached_imgs}")

if inp.exists():
    # Carried-forward artefacts are searched at any depth too, for the same reason.
    for _n, _dst in (("descriptors_ungrounded", REPO), ("descriptors_grounded_matched", REPO)):
        for u in find_dirs(inp, _n):
            shutil.copytree(u, _dst / _n, dirs_exist_ok=True)
            print(f"[carry] {_n} from {u}")
    for ck in find_dirs(inp, "checkpoints"):
        shutil.copytree(ck, WORK / "checkpoints", dirs_exist_ok=True)
        print(f"[carry] CNN checkpoints from {ck} -- interrupted archs resume")
    for r in find_dirs(inp, "results"):
        n = 0
        for f in r.glob("*.json"):
            if not (RESULTS / f.name).exists():
                shutil.copy(f, RESULTS / f.name)
                n += 1
        print(f"[carry] {n} result file(s) from {r}")

# ================================ STAGE 1 — DATA ================================
if attached_imgs is not None:
    os.environ["PDE_DATASET_DIR"] = str(attached_imgs)
    print(f"[data] using attached images at {attached_imgs} -- no fetch")
else:
    # Refuse, rather than warn-and-continue. Warning was not enough: a GPU session with the
    # dataset simply not attached silently fell through to a 114 GB fetch that cannot finish
    # inside a Kaggle cell, and spent an hour of GPU quota proving it. Fetching is now opt-in.
    print(f"\n{'=' * 74}\nNO IMAGES ATTACHED -- stopping before anything expensive.\n{'=' * 74}")
    seen_inputs = sorted(p.name for p in inp.iterdir()) if inp.exists() else []
    print(f"  /kaggle/input contains: {seen_inputs or '(nothing)'}")
    for cand in (p for p in (inp.iterdir() if inp.exists() else []) if p.is_dir()):
        inner = sorted(q.name for q in cand.iterdir())[:6]
        print(f"    {cand.name}/ -> {inner}{' ...' if len(inner) == 6 else ''}")
    print("""
  This run needs the image dataset ATTACHED, not fetched:
    right panel -> Add Data -> Your Datasets -> pde-sage-data -> Add

  The runner looks for <dataset>/exp_data/<Crop>___<Disease>/ or <dataset>/<Crop>___<Disease>/.
  If your dataset nests deeper than that (for example because a .zip was uploaded alongside
  the folder and Kaggle expanded it), the listing above shows the real layout -- send it over
  rather than reshuffling the dataset.

  To fetch from HuggingFace anyway, set ALLOW_FETCH = True at the top of this file. Be aware
  the pinned May release is 114 GB in ~10.7 GB shards and has never completed inside one
  session; the supported path is prepare_upload.py + upload.""")
    if not ALLOW_FETCH:
        sys.exit(0)
    os.environ["PDE_DATASET_DIR"] = str(DATA)
    DATA.mkdir(parents=True, exist_ok=True)
    print("\n[data] ALLOW_FETCH is set -- fetching anyway.")
    if HAS_GPU:
        print("[data] WARNING: this spends GPU quota on a network-bound job. A CPU session is "
              "cheaper for a fetch.")
    if not os.environ.get("HF_TOKEN"):
        print("[data] WARNING: no HF_TOKEN. Anonymous pulls are throttled, and a throttled HF "
              "connection stalls rather than failing. This is what hung the previous run.")
    banner(f"fetch SAGE rev {CFG.SHARD_REVISION[:10]} ({CFG.N_SHARDS} shards)")
    sh([sys.executable, "-u", str(S / "sage_data.py"), "--role", "all",
        "--max-side", str(MAX_SIDE), "--budget-h", str(min(FETCH_BUDGET_H, BUDGET_H)),
        "--shard-timeout", str(SHARD_TIMEOUT)], min(FETCH_BUDGET_H, BUDGET_H) + 0.5)

# --- verify before spending hours on top of it ---------------------------------------------
DDIR = Path(os.environ["PDE_DATASET_DIR"])
per_crop, n_images = Counter(), 0
for d in DDIR.iterdir():
    if d.is_dir() and "___" in d.name:
        k = sum(1 for _ in d.glob("*.jpg"))
        per_crop[d.name.split("___", 1)[0]] += k
        n_images += k
n_classes = sum(1 for d in DDIR.iterdir() if d.is_dir() and "___" in d.name)
print(f"\n[data] {n_images:,} images | {len(per_crop)} crops | {n_classes} classes")
for crop, k in sorted(per_crop.items(), key=lambda kv: -kv[1]):
    role = "HELD" if crop in CFG.HELDOUT_CROPS else "seen"
    print(f"       {crop:12s} {k:7,d}  {role}")

DATA_OK = n_images >= MIN_IMAGES and len(per_crop) >= MIN_CROPS
if not DATA_OK:
    missing = [c for c in CFG.WANT_CROPS if per_crop.get(c, 0) < 25]
    print(f"""
{'=' * 74}
DATA INCOMPLETE -- stopping here on purpose.
{'=' * 74}
  have {n_images:,} images over {len(per_crop)} crops; need >= {MIN_IMAGES:,} over {MIN_CROPS}.
  missing / too small: {missing}

  Nothing is lost. The fetch is checkpointed per shard in .shards_done.json.
    1. Save Version -> Save & Run All (Commit)   (already running if you used that)
    2. Output -> "Create Dataset"
    3. Attach it to a fresh copy of this notebook and run this same file again.

  Training on a partial pull would silently change every class count in the paper, which is why
  this stops rather than continuing.
  wall {elapsed_h():.2f} h
""")
    sys.exit(0)

# The manifest stores ABSOLUTE image paths, so it is ALWAYS rebuilt where it will be used -- an
# attached dataset mounts at a different path than it was built at.
sh([sys.executable, "-u", str(S / "build_manifest.py"), "--min-images", "25"], 0.5)
if not (WORK / "manifest.csv").exists():
    sys.exit("[fatal] no manifest.csv")
print(f"[ok] data verified: {n_images:,} images, {len(per_crop)} crops, {n_classes} classes")

if not HAS_GPU:
    print(f"""
{'=' * 74}
CPU SESSION -- data is ready, experiments need a GPU.
{'=' * 74}
  1. Output -> "Create Dataset"  (call it pde-sage-data)
  2. New notebook, Accelerator = GPU T4 x2, attach it, paste this same file, run.
  wall {elapsed_h():.2f} h
""")
    sys.exit(0)

# ---------------------------------------------------------------- stale-result guard
# Anything produced by the pre-fix rich matcher must be RECOMPUTED, not skipped.
# Every one of these reports a `rich` arm, and LOCO runs on `rich` end to end, so all three families
# are invalid if produced before the fix -- not just the zero-shot evaluations. Guarding only
# zeroshot_eval_* would let a carried-forward abstain or LOCO file be skipped on "it already
# exists" and keep contaminated numbers in the final results.
stale = []
for pat in ("zeroshot_eval_*.json", "metrics_abstain_*.json", "loco_*.json"):
    for p in RESULTS.glob(pat):
        try:
            if not json.loads(p.read_text()).get("matcher_normalised"):
                stale.append(p)
        except Exception:
            stale.append(p)
for p in stale:
    p.rename(p.with_suffix(".json.pre_matcher_fix"))
if stale:
    print(f"[rich-fix] set aside {len(stale)} evaluation(s) from the old matcher; recomputing")

# ================================ STAGE 2 — UNGROUNDED DESCRIPTORS ================================
# Text-only and cheap, and it is what decides the paper, so it runs before the long GPU stages.
ARMS = [("ungrounded", REPO / "descriptors_ungrounded")]
if MATCHED_GROUNDED:
    ARMS.append(("grounded", REPO / "descriptors_grounded_matched"))


def llm_reachable():
    """One cheap call before spending the run. Discovering a bad key or an unavailable model after
    six hours of GPU time is the expensive way to learn it."""
    try:
        if os.environ.get("LAVA_API_KEY"):
            from openai import OpenAI
            c = OpenAI(api_key=os.environ["LAVA_API_KEY"],
                       base_url=os.environ.get("LAVA_BASE_URL", "https://api.lava.so/v1"))
            c.chat.completions.create(model=os.environ["PDE_LLM_MODEL"], max_tokens=4,
                                      messages=[{"role": "user", "content": "ping"}])
        else:
            import anthropic
            anthropic.Anthropic().messages.create(
                model=os.environ["PDE_LLM_MODEL"], max_tokens=4,
                messages=[{"role": "user", "content": "ping"}])
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:200]}"


if not HAVE_LLM_KEY:
    print("\n[descriptors] no LAVA_API_KEY / ANTHROPIC_API_KEY secret -> generation SKIPPED. "
          "Every other stage still runs; add the secret to get the control arm.")
else:
    okc, err = llm_reachable()
    print(f"\n[llm] model {os.environ['PDE_LLM_MODEL']} via "
          f"{'Lava' if os.environ.get('LAVA_API_KEY') else 'Anthropic'}: "
          f"{'reachable' if okc else 'UNREACHABLE -- ' + err}")
    if not okc:
        HAVE_LLM_KEY = False
        print("[llm] skipping both descriptor arms rather than filling the registry with stubs. "
              "Fix the secret or PDE_LLM_MODEL and re-run; nothing else is affected.")

if HAVE_LLM_KEY:
    for arm, root in ARMS:
        for s in UNGROUNDED_SEEDS:
            d = root / str(s)
            filled = 0
            if d.is_dir():
                for p in d.glob("*.json"):
                    try:
                        filled += sum(1 for r in json.loads(p.read_text())
                                      if r.get("status") == "filled")
                    except Exception:
                        pass
            if filled >= 40:
                print(f"[skip] {arm} descriptors seed {s} ({filled} filled)")
                continue
            if not ok_to_start(f"{arm} descriptors seed {s}",
                               [f"{arm} descriptors seed {x}"
                                for x in UNGROUNDED_SEEDS[UNGROUNDED_SEEDS.index(s) + 1:]]):
                break
            banner(f"generate {arm} descriptors, seed {s}, model {os.environ['PDE_LLM_MODEL']}")
            sh([sys.executable, "-u", str(S / "build_ungrounded.py"),
                "--seed", str(s), "--which", "heldout", "--arm", arm], 1.0)

UNG_ROOT = REPO / "descriptors_ungrounded"

# Always re-save, not just after generating: descriptors carried in from an attached dataset must
# also land in THIS run's output, or a later run that attaches only this output loses the arm.
for _name in ("descriptors_ungrounded", "descriptors_grounded_matched"):
    if (REPO / _name).is_dir():
        shutil.copytree(REPO / _name, WORK / _name, dirs_exist_ok=True)
        print(f"[ok] {_name} saved into this run's output")

# ================================ STAGE 3 — ZERO-SHOT ================================
for exp in ["A", "B", "C"]:
    if (RESULTS / f"zeroshot_eval_{exp}.json").exists():
        print(f"[skip] zero-shot {exp}")
        continue
    if not ok_to_start(f"zero-shot {exp}",
                       [f"zero-shot {x}" for x in "ABC"[("ABC".index(exp) + 1):]
                        if not (RESULTS / f"zeroshot_eval_{x}.json").exists()]):
        break
    banner(f"zero-shot, scale {exp}")
    sh([sys.executable, "-u", str(S / "evaluate.py"), "--exp", exp,
        "--strategies", "bare", "crude", "rich", "grounded", "--heavy", "--teachers"])

# ================================ STAGE 4 — THE CONTROL ARM ================================
MIN_FILLED = 40      # of the 51 held-out classes at configuration C


def filled_count(root, seed):
    d = root / str(seed)
    if not d.is_dir():
        return 0
    n = 0
    for p in d.glob("*.json"):
        try:
            n += sum(1 for r in json.loads(p.read_text()) if r.get("status") == "filled")
        except Exception:
            pass
    return n


# A seed directory existing is NOT enough. Every arm falls through to `rich` for a class it has no
# record for -- deliberately, so coverage gaps are handled identically everywhere -- which means a
# directory of stubs would make the "ungrounded" arm silently BE `rich`, and the run would report a
# confident delta between an arm and itself. Require most classes to be genuinely filled.
have_seeds = []
for s in UNGROUNDED_SEEDS:
    n = filled_count(UNG_ROOT, s)
    if n >= MIN_FILLED:
        have_seeds.append(s)
    elif (UNG_ROOT / str(s)).is_dir():
        print(f"[ungrounded] seed {s} has only {n} filled records (need {MIN_FILLED}) -> NOT "
              f"evaluated; it would fall through to `rich` and compare rich against itself.")
if not have_seeds:
    print("\n[ungrounded] no usable descriptor seeds -> control arm SKIPPED")

for s in have_seeds:
    if (RESULTS / f"zeroshot_eval_C_ung{s}.json").exists():
        print(f"[skip] control arm eval seed {s}")
        continue
    if not ok_to_start(f"control arm eval seed {s}",
                       [f"control arm eval seed {x}"
                        for x in have_seeds[have_seeds.index(s) + 1:]
                        if not (RESULTS / f"zeroshot_eval_C_ung{x}.json").exists()]):
        break
    banner(f"control arm, seed {s}")
    arms = ["rich", "grounded", "ungrounded"]
    gm = filled_count(REPO / "descriptors_grounded_matched", s)
    if MATCHED_GROUNDED and gm >= MIN_FILLED:
        arms.append("grounded_matched")   # same model as `ungrounded`; this is the clean comparison
    elif MATCHED_GROUNDED:
        print(f"  [note] grounded_matched seed {s} has {gm} filled records (need {MIN_FILLED}); "
              f"evaluating without it, so this seed compares against the shipped registry only.")
    sh([sys.executable, "-u", str(S / "evaluate.py"), "--exp", "C",
        "--strategies", *arms, "--ungrounded-seed", str(s), "--heavy"])

# ================================ STAGE 5 — ABSTENTION ================================
for exp in ["A", "B", "C"]:
    if (RESULTS / f"metrics_abstain_{exp}.json").exists():
        print(f"[skip] abstain {exp}")
        continue
    if not ok_to_start(f"abstain {exp}",
                       [f"abstain {x}" for x in "ABC"[("ABC".index(exp) + 1):]
                        if not (RESULTS / f"metrics_abstain_{x}.json").exists()]):
        break
    banner(f"abstention + top-5, scale {exp}")
    sh([sys.executable, "-u", str(S / "metrics.py"), "--exp", exp,
        "--strategies", "rich", "grounded", "--reference"])

# ================================ STAGE 6 — LABEL-CORRECTED SENSITIVITY ================================
# Merges SAGE's duplicate disease labels and drops the non-disease ones, into a SEPARATE
# *_clean.json so the as-published numbers are never overwritten.
if (RESULTS / "zeroshot_eval_C_clean.json").exists():
    print("[skip] clean eval")
elif ok_to_start("label-corrected eval"):
    banner("zero-shot scale C, label-corrected")
    sh([sys.executable, "-u", str(S / "evaluate.py"), "--exp", "C", "--clean",
        "--strategies", "bare", "crude", "rich", "grounded", "--heavy", "--teachers"])

# ================================ STAGE 7 — SEEN PROBE + LOCO ================================
if ok_to_start("seen-crop probe"):
    banner("seen-crop linear probe, scales A/B/C")
    # Embeds the config-C pool ONCE per encoder and derives A and B by subsetting instead of
    # re-encoding the same images three times. The cache resumes mid-encoder.
    sh([sys.executable, "-u", str(S / "probe_seen_all.py"), "--workers", str(WORKERS)], 3.5)

if (RESULTS / "loco_s0_rich.json").exists():
    print("[skip] loco")
elif ok_to_start("leave-one-crop-out"):
    banner("leave-one-crop-out + bootstrap CIs")
    sh([sys.executable, "-u", str(S / "loco.py"), "--model", "s0",
        "--strategy", "rich", "--bootstrap", "2000"], 1.0)

# ================================ STAGE 8 — CNN BASELINES ================================
# Last on purpose: this is ~8 h, and everything above decides the paper.
cnn_done, cnn_failed = [], []
for i, arch in enumerate(ARCHS):
    if (RESULTS / f"supervised_{arch}.json").exists():
        print(f"[skip] cnn {arch}")
        continue
    if not ok_to_start(f"cnn {arch}",
                       [f"cnn {a}" for a in ARCHS[i + 1:]
                        if not (RESULTS / f"supervised_{a}.json").exists()]):
        break
    # A T4 has 14.56 GB and batch 128 does not fit the heavier models. Halve and retry rather than
    # lose the architecture -- three were lost this way on an earlier sweep and the cause was
    # misdiagnosed as a missing timm arch; all three were torch.OutOfMemoryError.
    rc, batch = 1, BATCH
    while rc != 0 and batch >= MIN_BATCH:
        banner(f"cnn {i + 1}/{len(ARCHS)}: {arch} (epochs={EPOCHS} batch={batch})")
        rc = sh([sys.executable, "-u", str(S / "supervised_baseline.py"), "--arch", arch,
                 "--epochs", str(EPOCHS), "--batch", str(batch),
                 "--workers", str(WORKERS), "--resume"], ARCH_MAX_H)
        if rc != 0:
            batch //= 2
            if batch >= MIN_BATCH:
                print(f"  [retry] {arch} failed (likely CUDA OOM) -> batch {batch}", flush=True)
    (cnn_done if rc == 0 else cnn_failed).append(arch)
    # Only delete the checkpoint of an architecture that COMPLETED -- a failed one needs it to
    # resume next run. 14 checkpoints would otherwise eat several GB of the output quota.
    if rc == 0:
        ck = WORK / "checkpoints" / f"{arch}_ckpt.pt"
        if ck.exists():
            ck.unlink()

# ================================ SUMMARY ================================
# Regenerated from the fixed matcher so the collision counts can never be hand-typed into the paper.
sh([sys.executable, "-u", str(S / "descriptor_coverage.py"), "--write"], 0.2)

print(f"\n{'=' * 74}\nRESULT FILES\n{'=' * 74}")
for f in sorted(RESULTS.glob("*.json")):
    print(f"  {f.name:46s} {f.stat().st_size / 1024:8.1f} KB")

for e in "ABC":
    p = RESULTS / f"zeroshot_eval_{e}.json"
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    ms = [v for k, v in d["models"].items() if "SigLIP" not in k]
    if not ms:
        continue
    r = sum(m["rich"]["acc"] for m in ms) / len(ms)
    g = sum(m["grounded"]["acc"] for m in ms) / len(ms)
    print(f"\n  scale {e}: {d['n_classes']:3d} unseen classes  chance {d['chance']:.1%}  "
          f"rich {r:.1%}  grounded {g:.1%}  delta {(g - r) * 100:+.1f} pp"
          f"   [matcher_normalised={d.get('matcher_normalised')}]")

ung = sorted(RESULTS.glob("zeroshot_eval_C_ung*.json"))
if ung:
    print(f"\n{'=' * 74}\nUNGROUNDED CONTROL ARM ({len(ung)} seed(s))\n"
          f"decides whether the contribution is grounding or per-class coverage\n{'=' * 74}")
    acc = {}
    for p in ung:
        d = json.loads(p.read_text())
        ms = [v for k, v in d["models"].items() if "SigLIP" not in k]
        for arm in ("rich", "grounded", "grounded_matched", "ungrounded"):
            if ms and arm in ms[0]:
                v = sum(m[arm]["acc"] for m in ms) / len(ms)
                acc.setdefault(arm, []).append(v)
                print(f"    seed {p.stem.split('_ung')[-1]}  {arm:16s} {v:.1%}")
    # grounded_matched is generated by the SAME model as ungrounded, so it is the comparison that
    # isolates sourcing. The shipped `grounded` registry records no model, so grounded-vs-ungrounded
    # cannot separate the sourcing constraint from a change of model version.
    ref = "grounded_matched" if "grounded_matched" in acc else "grounded"
    if "ungrounded" in acc and ref in acc:
        u = sum(acc["ungrounded"]) / len(acc["ungrounded"])
        g = sum(acc[ref]) / len(acc[ref])
        spread = max(acc["ungrounded"]) - min(acc["ungrounded"])
        n = len(acc["ungrounded"])
        print(f"\n    {ref} {g:.1%} | ungrounded {u:.1%} (n={n} seeds) | "
              f"delta {(g - u) * 100:+.1f} pp | seed spread {spread * 100:.1f} pp")
        if ref == "grounded":
            print("    NOTE: compared against the SHIPPED registry, whose generating model is not")
            print("    recorded, so this delta confounds sourcing with model version.")
        if (g - u) * 100 > spread:
            print("    -> the gap exceeds seed noise: SOURCING itself is doing work.")
        else:
            print("    -> the gap is within seed noise: grounding buys AUDITABILITY, not accuracy.")
        if n < 3:
            print(f"    Only {n} seed(s). Do not put a delta in the manuscript from fewer than 3.")

sup = []
for f in sorted(RESULTS.glob("supervised_*.json")):
    try:
        d = json.loads(f.read_text())
        sup.append((d.get("arch"), d.get("params_M"), d.get("seen_top1")))
    except Exception:
        pass
if sup:
    print(f"\n{'=' * 74}\nSUPERVISED BASELINES ({len(sup)}/{len(ARCHS)})\n{'=' * 74}")
    for a, p, acc_ in sorted(sup, key=lambda r: -(r[2] or 0)):
        print(f"  {a:26s} {'  --  ' if p is None else f'{p:6.2f}M'}  {(acc_ or 0):.1%}")
    if len(sup) > 1:
        best = max(sup, key=lambda r: r[2] or 0)
        big = max(sup, key=lambda r: r[1] or 0)
        if best[1] and big[1]:
            print(f"\n  best {best[0]} at {best[1]:.1f}M -> {best[2]:.1%};  "
                  f"largest {big[0]} at {big[1]:.1f}M -> {big[2]:.1%}  "
                  f"({(best[2] - big[2]) * 100:+.1f} pp for {big[1] / best[1]:.1f}x the parameters)")

if cnn_done:
    print(f"\n  CNNs trained this session: {cnn_done}")
if cnn_failed:
    print(f"  STILL FAILING at batch {MIN_BATCH}: {cnn_failed}  (read the traceback above)")

if LEFT:
    print(f"""
{'=' * 74}
NOT DONE YET ({len(LEFT)} stage(s)) -- stopped at the {BUDGET_H} h budget, on purpose
{'=' * 74}
  {chr(10).join('    ' + s for s in LEFT[:20])}
  {'    ... and %d more' % (len(LEFT) - 20) if len(LEFT) > 20 else ''}

  1. Output -> "Create Dataset"
  2. Attach it to a fresh copy of this notebook and run this SAME file again.
     Everything above is skipped; it picks up exactly here.
""")
else:
    print(f"""
{'=' * 74}
EVERYTHING COMPLETE
{'=' * 74}
  1. Output -> download results/*.json into docs/paper/
  2. Locally, regenerate -- nothing in the paper is typed by hand:
       python docs/paper/make_tables.py --write
       python docs/paper/make_tex_tables.py
       python docs/paper/make_figures.py
       python scripts/collect_results.py
       python scripts/build_submission.py
  3. Rewrite section 5.3, the abstract and the title around what the ungrounded arm showed.

  Still open and independent of this run:
    * PENDING-ZENODO-DOI in main.tex -- COMPAG Option C needs the data deposited and linked.
      Cite SAGE at {CFG.SHARD_REVISION[:12]}, not "latest".
    * The 181-URL pass in docs/paper/SOURCE_CHECKLIST.md. Auditability is now the load-bearing
      claim, so this is critical path; only 16 of 217 records are page-verified.
""")

# ---------------------------------------------------------------- downloadable bundle
# /kaggle/working also holds exp_data/ (1.7 GB), the cloned repo and CNN checkpoints, so grabbing
# the whole Output to get the results means pulling gigabytes for well under a megabyte of JSON.
# Everything the paper needs is bundled into one small zip instead, with a direct download link.
def build_bundle():
    import zipfile
    out = WORK / "pde_results.zip"
    picked, skipped = [], []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in sorted(RESULTS.glob("*.json")):
            z.write(f, f"results/{f.name}")
            picked.append(f"results/{f.name}")
        # set-aside pre-fix evaluations: carried so a reviewer can diff old against new
        for f in sorted(RESULTS.glob("*.pre_matcher_fix")):
            z.write(f, f"results/superseded/{f.name}")
            picked.append(f"results/superseded/{f.name}")
        for arm in ("descriptors_ungrounded", "descriptors_grounded_matched"):
            root = WORK / arm
            if root.is_dir():
                for f in sorted(root.rglob("*.json")):
                    z.write(f, f"{arm}/{f.relative_to(root)}")
                    picked.append(f"{arm}/{f.relative_to(root)}")
        cov = REPO / "docs" / "paper" / "descriptor_coverage.json"
        if cov.exists():
            z.write(cov, "descriptor_coverage.json")
            picked.append("descriptor_coverage.json")
        for log in sorted((WORK / "logs").glob("*.log")) if (WORK / "logs").is_dir() else []:
            z.write(log, f"logs/{log.name}")
            picked.append(f"logs/{log.name}")
        # a receipt, so the bundle can be checked against the run that produced it
        z.writestr("BUNDLE.json", json.dumps({
            "files": picked,
            "images": n_images, "classes": n_classes, "crops": len(per_crop),
            "sage_revision": CFG.SHARD_REVISION,
            "llm_model": os.environ.get("PDE_LLM_MODEL"),
            "wall_hours": round(elapsed_h(), 2),
            "stages_outstanding": LEFT,
        }, indent=2))
    for big in ("exp_data", "checkpoints", "pde"):     # named so it is clear they are EXCLUDED
        if (WORK / big).exists():
            skipped.append(big)
    return out, picked, skipped


try:
    BUNDLE, picked, skipped = build_bundle()
    mb = BUNDLE.stat().st_size / 1e6
    print(f"\n{'=' * 74}\nDOWNLOAD THE RESULTS\n{'=' * 74}")
    print(f"  {BUNDLE.name}  --  {mb:.2f} MB, {len(picked)} file(s)")
    print(f"  excluded (would make this gigabytes): {', '.join(skipped) or 'nothing'}")
    print("\n  Get it WITHOUT downloading the whole Output:")
    print("    * Notebook -> Output tab -> click pde_results.zip -> Download")
    print("    * or run this in a cell after the run finishes:")
    print("        from IPython.display import FileLink; FileLink('pde_results.zip')")
    try:                                    # renders a real clickable link inside the notebook
        from IPython.display import FileLink, display
        display(FileLink(str(BUNDLE.relative_to(WORK))))
    except Exception:
        pass
    print("\n  Then locally:")
    print("    unzip -o pde_results.zip -d /tmp/pde && cp /tmp/pde/results/*.json docs/paper/")
    print("    cp -r /tmp/pde/descriptors_* .        # only if the control arm ran")
except Exception as e:
    print(f"\n[warn] could not build the results bundle: {type(e).__name__}: {e}")
    print("       The individual files are still in the Output tab under results/.")


print(f"  wall {elapsed_h():.2f} h | free disk {free_gb():.1f} GB")
