"""
KAGGLE RUN 2 of 3 — ALL VLM EXPERIMENTS.  Paste this whole file as ONE cell.
===========================================================================

Zero-shot at three scales, abstention/top-5, the label-corrected sensitivity run, the UNGROUNDED
control arm at three seeds, the seen-crop probe, and leave-one-crop-out.

NOTEBOOK SETTINGS
  Accelerator = GPU T4 x2   ·   Internet = ON   ·   Persistence = Files only
  Add data    -> **pde-sage-data**  (the output of run 1). Without it this run refetches 21 GB.
  Secrets (Add-ons -> Secrets):
      GH_TOKEN       = GitHub fine-grained PAT, read-only Contents
      LAVA_API_KEY   = your Lava spend key           <- either this ...
      ANTHROPIC_API_KEY = sk-ant-...                 <- ... or this. Needed ONLY by the ungrounded
                                                        arm; every other stage runs without a key.
Then: Save Version -> "Save & Run All (Commit)" -> close the tab.
When it finishes: Output -> "Create Dataset" -> name it **pde-results-2**.

THE ARM THAT MATTERS
--------------------
The paper's headline -- "only source-grounded descriptors scale" -- is retracted and not yet
rewritten. `rich` is a keyword-retrieved bank in which, at scale C, only 8 of 51 held-out classes
get a unique descriptor: 17 fall back to the bare class name and 26 share text with another class.
Beating that measures per-class DISTINCTNESS, not grounding.

The ungrounded arm removes the confound: same model, same schema, same fall-through, sampled at
temperature 1.0 -- the only difference is that the "cite a retrievable source" constraint is
dropped. Three seeds, because one sample of LLM text cannot separate "grounding helps" from "that
generation was lucky".

    ungrounded ~= grounded  -> grounding is free and buys auditability. The cleaner paper.
    ungrounded <  grounded  -> the sourcing constraint itself helps. A real finding.

Until this lands, no delta number goes in the manuscript.

A SECOND THING THIS RUN FIXES
-----------------------------
descriptors.text_for normalised underscores for the prompt but not for the match key, so all 13
multi-word bank entries ("powdery mildew", "citrus canker", ...) were unreachable for every label in
the dataset. Every published `rich` accuracy came from that broken matcher and is invalid. New
results are stamped "matcher_normalised"; anything unstamped is set aside below and recomputed
rather than skipped.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ================================ SETTINGS ================================
BUDGET_H         = 8.5          # stop starting new stages past this (session limit is 12 h)
STAGE_TIMEOUT_H  = 3.0          # no single stage may exceed this
UNGROUNDED_SEEDS = [0, 1, 2]
REPO_URL         = "github.com/Abhiram970/plant-disease-edge.git"

RUN_UNGROUNDED_GEN = True   # generate the ungrounded descriptors here (needs an LLM key)
RUN_ZEROSHOT       = True   # cross-crop zero-shot, scales A/B/C      (~1.5 h)
RUN_ABSTAIN        = True   # top-5 + risk-coverage, scales A/B/C     (~1 h)
RUN_CLEAN_EVAL     = True   # label-corrected sensitivity run         (~30 min)
RUN_UNGROUNDED     = True   # the control arm, 3 seeds                (~1 h)
RUN_PROBE          = True   # seen-crop linear probe, scales A/B/C    (~1 h)
RUN_LOCO           = True   # leave-one-crop-out + bootstrap CIs      (~20 min)
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


def sh(cmd, timeout_h=STAGE_TIMEOUT_H):
    """Run a stage under a hard deadline. A stage that hangs must cost one stage, not the session."""
    try:
        return subprocess.run(cmd, timeout=timeout_h * 3600).returncode
    except subprocess.TimeoutExpired:
        print(f"\n[TIMEOUT] stage exceeded {timeout_h} h and was killed: {' '.join(map(str, cmd))}\n",
              flush=True)
        return 124


def budget_left(stage):
    if elapsed_h() > BUDGET_H:
        print(f"\n[budget] {elapsed_h():.1f} h elapsed -- skipping '{stage}' and everything after it "
              f"so this session commits. Re-run with this output attached to continue.\n", flush=True)
        return False
    return True


print(f"[start] free disk {free_gb():.1f} GB")

# --------------------------------------------------------------- repo + keys
gh_tok = secret("GH_TOKEN")
REPO = WORK / "pde"
if not REPO.exists():
    url = f"https://{gh_tok}@{REPO_URL}" if gh_tok else f"https://{REPO_URL}"
    if subprocess.run(["git", "clone", "--depth", "1", url, str(REPO)]).returncode != 0:
        sys.exit("[fatal] clone failed -> check Internet=ON and the GH_TOKEN secret.")
print(f"[ok] repo at {REPO}")
S = REPO / "scripts"

for k in ("LAVA_API_KEY", "ANTHROPIC_API_KEY", "LAVA_BASE_URL", "PDE_LLM_MODEL", "HF_TOKEN"):
    v = secret(k)
    if v:
        os.environ[k] = v
        print(f"[ok] {k} loaded")
have_key = bool(os.environ.get("LAVA_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "timm", "open_clip_torch"])
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "safetensors", "pyyaml",
                "huggingface_hub", "hf_transfer", "pyarrow", "tqdm", "regex", "ftfy",
                "openai", "anthropic"])

# --------------------------------------------------------------- data
os.environ["HF_HOME"] = "/kaggle/temp/hf"
os.environ["HF_HUB_CACHE"] = "/kaggle/temp/hf/hub"
Path("/kaggle/temp/hf/hub").mkdir(parents=True, exist_ok=True)
os.environ["PDE_DATA_ROOT"] = str(WORK)

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
    sys.exit("[fatal] no image dataset attached. Add data -> pde-sage-data (the output of run 1). "
             "This run will not refetch 21 GB on your GPU quota.")
os.environ["PDE_DATASET_DIR"] = str(attached)
n_cls = len(list(attached.glob("*___*")))
n_img = sum(1 for _ in attached.rglob("*.jpg"))
print(f"[ok] images: {n_img:,} in {n_cls} class folders at {attached}")
if n_img < 60_000:
    sys.exit(f"[fatal] only {n_img:,} images -- run 1 did not finish. Complete the fetch first; "
             f"the class counts will not match the paper otherwise.")

RESULTS = WORK / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
UNG_OUT = WORK / "descriptors_ungrounded"

# Carry forward anything a previous session already produced.
if inp.exists():
    for cand in inp.iterdir():
        src = cand / "results"
        if src.is_dir():
            for f in src.glob("*.json"):
                if not (RESULTS / f.name).exists():
                    shutil.copy(f, RESULTS / f.name)
            print(f"[ok] seeded prior results from {src}")
        ung = cand / "descriptors_ungrounded"
        if ung.is_dir():
            shutil.copytree(ung, REPO / "descriptors_ungrounded", dirs_exist_ok=True)
            print(f"[ok] restored ungrounded descriptors from {ung}")

# The manifest stores ABSOLUTE paths, so it is rebuilt here rather than reused from run 1.
sh([sys.executable, "-u", str(S / "build_manifest.py"), "--min-images", "25"], 0.5)
if not (WORK / "manifest.csv").exists():
    sys.exit("[fatal] no manifest.csv")

import torch
print(f"\n[env] cuda={torch.cuda.is_available()} "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''} "
      f"| free {free_gb():.1f} GB | t+{elapsed_h():.1f} h")

# --------------------------------------------------------------- ungrounded descriptors
# Generated HERE rather than locally, so the whole study runs on Kaggle. Text only, no GPU.
if RUN_UNGROUNDED_GEN and budget_left("ungrounded generation"):
    if not have_key:
        print("[ungrounded] SKIPPED generation -- no LAVA_API_KEY / ANTHROPIC_API_KEY secret. "
              "Every other stage still runs; add the secret and re-run for the control arm.")
    else:
        for s in UNGROUNDED_SEEDS:
            d = REPO / "descriptors_ungrounded" / str(s)
            filled = sum(1 for p in d.glob("*.json")
                         for r in json.loads(p.read_text()) if r.get("status") == "filled") \
                if d.is_dir() else 0
            if filled >= 40:
                print(f"[skip] ungrounded descriptors seed {s} ({filled} filled records)")
                continue
            print(f"\n{'=' * 72}\n[ungrounded gen] seed {s}  t+{elapsed_h():.1f} h\n{'=' * 72}",
                  flush=True)
            sh([sys.executable, "-u", str(S / "build_ungrounded.py"),
                "--seed", str(s), "--which", "heldout"], 1.0)
        if (REPO / "descriptors_ungrounded").is_dir():
            shutil.copytree(REPO / "descriptors_ungrounded", UNG_OUT, dirs_exist_ok=True)
            print(f"[ok] ungrounded descriptors copied to {UNG_OUT} (they persist in this output)")

# --------------------------------------------------------------- stale-result guard
# Anything produced by the pre-fix matcher must be recomputed, not skipped.
stale = []
for p in RESULTS.glob("zeroshot_eval_*.json"):
    try:
        if not json.loads(p.read_text()).get("matcher_normalised"):
            stale.append(p)
    except Exception:
        stale.append(p)
for p in stale:
    p.rename(p.with_suffix(".json.pre_matcher_fix"))
if stale:
    print(f"[rich-fix] set aside {len(stale)} evaluation(s) from the old matcher; recomputing them")

# --------------------------------------------------------------- zero-shot + abstention
for exp in ["A", "B", "C"]:
    if RUN_ZEROSHOT and not (RESULTS / f"zeroshot_eval_{exp}.json").exists():
        if not budget_left(f"zero-shot {exp}"):
            break
        print(f"\n{'=' * 72}\n[zero-shot {exp}]  t+{elapsed_h():.1f} h\n{'=' * 72}", flush=True)
        sh([sys.executable, "-u", str(S / "evaluate.py"), "--exp", exp,
            "--strategies", "bare", "crude", "rich", "grounded", "--heavy", "--teachers"])
    if RUN_ABSTAIN and not (RESULTS / f"metrics_abstain_{exp}.json").exists():
        if not budget_left(f"abstain {exp}"):
            break
        print(f"\n{'=' * 72}\n[abstain {exp}]  t+{elapsed_h():.1f} h\n{'=' * 72}", flush=True)
        sh([sys.executable, "-u", str(S / "metrics.py"), "--exp", exp,
            "--strategies", "rich", "grounded", "--reference"])

# Merges SAGE's duplicate disease labels and drops the non-disease ones. Writes a SEPARATE
# *_clean.json so the as-published numbers are never overwritten.
if RUN_CLEAN_EVAL and not (RESULTS / "zeroshot_eval_C_clean.json").exists() \
        and budget_left("clean eval"):
    print(f"\n{'=' * 72}\n[zero-shot C, label-corrected]  t+{elapsed_h():.1f} h\n{'=' * 72}",
          flush=True)
    sh([sys.executable, "-u", str(S / "evaluate.py"), "--exp", "C", "--clean",
        "--strategies", "bare", "crude", "rich", "grounded", "--heavy", "--teachers"])

# --------------------------------------------------------------- the control arm
if RUN_UNGROUNDED:
    root = REPO / "descriptors_ungrounded"
    have = [s for s in UNGROUNDED_SEEDS if (root / str(s)).is_dir()]
    if not have:
        print("\n[ungrounded] SKIPPED -- no descriptors_ungrounded/<seed>/ available.")
    for s in have:
        if (RESULTS / f"zeroshot_eval_C_ung{s}.json").exists():
            print(f"[skip] ungrounded eval seed {s}")
            continue
        if not budget_left(f"ungrounded eval seed {s}"):
            break
        print(f"\n{'=' * 72}\n[ungrounded seed {s}]  t+{elapsed_h():.1f} h\n{'=' * 72}", flush=True)
        sh([sys.executable, "-u", str(S / "evaluate.py"), "--exp", "C",
            "--strategies", "rich", "grounded", "ungrounded",
            "--ungrounded-seed", str(s), "--heavy"])

# --------------------------------------------------------------- probe + LOCO
if RUN_PROBE and budget_left("seen probe"):
    print(f"\n{'=' * 72}\n[seen-crop probe]  t+{elapsed_h():.1f} h\n{'=' * 72}", flush=True)
    # Embeds the config-C pool ONCE per encoder and derives A and B by subsetting instead of
    # re-encoding the same images three times. The cache resumes mid-encoder.
    sh([sys.executable, "-u", str(S / "probe_seen_all.py"), "--workers", "4"], 3.5)

if RUN_LOCO and not (RESULTS / "loco_s0_rich.json").exists() and budget_left("loco"):
    print(f"\n{'=' * 72}\n[leave-one-crop-out]  t+{elapsed_h():.1f} h\n{'=' * 72}", flush=True)
    sh([sys.executable, "-u", str(S / "loco.py"), "--model", "s0",
        "--strategy", "rich", "--bootstrap", "2000"], 1.0)

# --------------------------------------------------------------- coverage table
# Regenerated from the fixed matcher so the collision counts in the paper can never be hand-typed.
sh([sys.executable, "-u", str(S / "descriptor_coverage.py"), "--write"], 0.2)

# --------------------------------------------------------------- summary
print(f"\n{'=' * 72}\nRESULT FILES\n{'=' * 72}")
for f in sorted(RESULTS.glob("*.json")):
    print(f"  {f.name:44s} {f.stat().st_size / 1024:7.1f} KB")

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
          f"rich {r:.1%}  grounded {g:.1%}  delta {(g - r) * 100:+.1f} pp   "
          f"[matcher_normalised={d.get('matcher_normalised')}]")

ung = sorted(RESULTS.glob("zeroshot_eval_C_ung*.json"))
if ung:
    print(f"\n{'=' * 72}\nUNGROUNDED CONTROL ARM ({len(ung)} seed(s))\n"
          f"the comparison that decides whether the contribution is grounding or coverage\n"
          f"{'=' * 72}")
    acc = {}
    for p in ung:
        d = json.loads(p.read_text())
        ms = [v for k, v in d["models"].items() if "SigLIP" not in k]
        seed = p.stem.split("_ung")[-1]
        for arm in ("rich", "grounded", "ungrounded"):
            if ms and arm in ms[0]:
                v = sum(m[arm]["acc"] for m in ms) / len(ms)
                acc.setdefault(arm, []).append(v)
                print(f"    seed {seed}  {arm:11s} {v:.1%}")
    if "ungrounded" in acc and "grounded" in acc:
        u = sum(acc["ungrounded"]) / len(acc["ungrounded"])
        g = sum(acc["grounded"]) / len(acc["grounded"])
        sp = max(acc["ungrounded"]) - min(acc["ungrounded"])
        print(f"\n    mean grounded {g:.1%} | mean ungrounded {u:.1%} | "
              f"delta {(g - u) * 100:+.1f} pp | ungrounded seed spread {sp * 100:.1f} pp")
        print(f"    -> {'grounding is doing the work' if (g - u) * 100 > sp else 'the delta is within seed noise: grounding buys auditability, not accuracy'}")

print(f"""
{'=' * 72}
NEXT
{'=' * 72}
  1. Save Version -> Save & Run All (Commit)
  2. Output -> "Create Dataset" -> name it  pde-results-2
  3. Run 3 (kaggle/run3_cnns.py): attach BOTH pde-sage-data and pde-results-2.
  4. Download results/*.json into docs/paper/ and regenerate (nothing is typed by hand):
       python docs/paper/make_tables.py --write
       python docs/paper/make_tex_tables.py
       python docs/paper/make_figures.py
       python scripts/collect_results.py && python scripts/build_submission.py

  wall {elapsed_h():.2f} h | free disk {free_gb():.1f} GB
""")
