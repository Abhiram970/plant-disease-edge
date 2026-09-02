"""
=====================================================================================
 PDE PART 2 of 3  --  PROBE, LEAVE-ONE-CROP-OUT, WiSE-FT
=====================================================================================
Run this SECOND. Needs no API key.

SETUP
  1. Add Data -> `pde-sage-data`.
  2. Add Data -> the OUTPUT of PART 1 (so its results carry forward). Optional but
     recommended: the bundle is imported automatically if present.
  3. GPU on, Internet on. Paste into ONE cell and run.

WHAT IT DOES                                                        est. time
  1  seen-crop linear probe A/B/C                                    ~0.5 h
  2  leave-one-crop-out + bootstrap CIs                              ~0.3 h
  3  WiSE-FT alpha sweep (0 / 0.5 / 1), re-measured                  ~0.8 h
                                                              TOTAL  ~1.6 h

WHY WiSE-FT IS RE-MEASURED. The stranded file it replaced recorded unseen_classes=17
(the pilot protocol) against seen_classes=166 (nested C) -- one table reporting two
different experiments. scripts/wiseft.py measures both sides under configuration C and
records the protocol in its output.

READ THE WARNINGS. WiSE-FT only behaves when the fine-tuned model is genuinely
fine-tuned from the same initialisation. Two smoke tests looked like a learning-rate
problem -- loss 5.08 at lr 1e-5, then 4.80 at 1e-4, against a random-guess loss of 5.11 --
but the real cause was that zeroshot.load_model() sets requires_grad=False on every
parameter, so the visual tower was never training and the sweep was interpolating a model
with itself. That is fixed; the learning rate here is the standard CLIP value.

The script now verifies, and records in wiseft.json: that the visual tower has trainable
parameters, that the fine-tuned weights actually moved (relative L2 distance, exits if
zero), that fine-tuning converged below 0.9x random-guess loss, that seen accuracy does
not dip below both endpoints, and that alpha=0 reproduces the frozen probe. Each failure
prints a [WARNING]. If you see any, raise --epochs or --lr before using the table.
=====================================================================================
"""

# ---------------------------------------------------------------- settings
BUDGET_H    = 11.0
WISE_EPOCHS = 3
WISE_LR     = "1e-5"     # standard CLIP fine-tuning range; see the note below
WISE_ALPHAS = ["0.0", "0.5", "1.0"]
REPO_URL = "https://github.com/Abhiram970/plant-disease-edge.git"
REPO_REF = "paper/draft-audit-2026-09-01"


import os, sys, json, time, shutil, subprocess, glob
from pathlib import Path

T0 = time.time()
def elapsed_h(): return (time.time() - T0) / 3600.0
def left_h():    return BUDGET_H - elapsed_h()

def banner(msg):
    print("\n" + "=" * 78, flush=True)
    print(f"[{msg}]  t+{elapsed_h():.1f} h  ({left_h():.1f} h left)", flush=True)
    print("=" * 78, flush=True)

def gpu_free_gb():
    """VRAM, not system RAM. The old runner printed psutil RAM ("free 20.3 GB") while the T4
    has 14.56 GiB of VRAM, so its OOM guard never once fired."""
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return free / 1e9, total / 1e9
    except Exception:
        pass
    return 0.0, 0.0

def ok_to_start(name, remaining, need_h):
    if left_h() < need_h:
        print(f"\n[budget] {left_h():.1f} h left, '{name}' needs ~{need_h:.1f} h -> STOP.", flush=True)
        if remaining:
            print(f"[budget] not run: {remaining}", flush=True)
        print("[budget] Re-run this cell to resume; finished stages are skipped.", flush=True)
        return False
    return True

# Kaggle attaches a second stream handler, which doubled every log line last run.
try:
    import logging
    _root = logging.getLogger()
    for _h in _root.handlers[1:]:
        _root.removeHandler(_h)
except Exception:
    pass

WORK = Path("/kaggle/working")
REPO = WORK / "pde"
S    = REPO / "scripts"
RESULTS = WORK / "results"; RESULTS.mkdir(exist_ok=True, parents=True)
CKPT    = WORK / "checkpoints"; CKPT.mkdir(exist_ok=True, parents=True)

banner("bootstrap")
if REPO.exists():
    shutil.rmtree(REPO, ignore_errors=True)
_rc = subprocess.run(["git", "clone", "--depth", "1", "--branch", REPO_REF, REPO_URL, str(REPO)],
                     capture_output=True, text=True)
if _rc.returncode != 0:
    _rc = subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO)],
                         capture_output=True, text=True)
    if _rc.returncode != 0:
        sys.exit(f"[fatal] clone failed:\n{_rc.stderr}")
    subprocess.run(["git", "-C", str(REPO), "fetch", "origin", REPO_REF],
                   capture_output=True, text=True)
    subprocess.run(["git", "-C", str(REPO), "checkout", "FETCH_HEAD"],
                   capture_output=True, text=True)
HEAD = subprocess.run(["git", "-C", str(REPO), "log", "--oneline", "-1"],
                      capture_output=True, text=True).stdout.strip()
print(f"[repo] HEAD = {HEAD}", flush=True)
if not (S / "evaluate.py").exists():
    sys.exit("[fatal] clone incomplete: scripts/evaluate.py missing")

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "open_clip_torch", "timm", "anthropic", "openai"],
               check=False, capture_output=True)

def find_dir(name, roots, max_depth=5):
    for root in roots:
        r = Path(root)
        if not r.exists():
            continue
        stack = [(r, 0)]
        while stack:
            d, depth = stack.pop(0)
            if d.name == name:
                return d
            if depth < max_depth:
                try:
                    stack += [(c, depth + 1) for c in d.iterdir() if c.is_dir()]
                except Exception:
                    pass
    return None

DATA = find_dir("exp_data", ["/kaggle/input", "/kaggle/working"])
if DATA is None:
    sys.exit("[fatal] exp_data not found. Attach the pde-sage-data dataset.")
os.environ["PDE_DATASET_DIR"] = str(DATA)
os.environ["PDE_DATA_ROOT"]   = str(WORK)
os.environ["PDE_NO_FETCH"]    = "1"

N_IMGS  = sum(1 for _ in DATA.rglob("*.jpg")) + sum(1 for _ in DATA.rglob("*.png"))
CLASSES = sorted({p.name for p in DATA.iterdir() if p.is_dir()})
CROPS   = sorted({c.split("___")[0] for c in CLASSES})
print(f"[data] {N_IMGS:,} images  {len(CLASSES)} classes  {len(CROPS)} crops", flush=True)
if N_IMGS < 60_000 or len(CROPS) < 18:
    sys.exit(f"[fatal] dataset too small ({N_IMGS:,} imgs / {len(CROPS)} crops).")
_fg, _tg = gpu_free_gb()
print(f"[gpu] {_fg:.1f} GB free of {_tg:.1f} GB VRAM", flush=True)

# Results carried forward from an earlier PART (attach that part's output as a dataset).
PRIOR = find_dir("results", ["/kaggle/input"])
if PRIOR and PRIOR.exists():
    _n = 0
    for _f in PRIOR.glob("*.json"):
        if not (RESULTS / _f.name).exists():
            shutil.copy2(_f, RESULTS / _f.name); _n += 1
    if _n:
        print(f"[prior] imported {_n} result file(s) from a previous part", flush=True)
for _arm in ("descriptors_ungrounded", "descriptors_grounded_matched",
             "descriptors_ungrounded_short", "descriptors_grounded_matched_short"):
    _src = find_dir(_arm, ["/kaggle/input"])
    if _src and _src.exists():
        _dst = REPO / _arm
        if not _dst.exists():
            shutil.copytree(_src, _dst)
            print(f"[prior] imported {_arm}", flush=True)

def sh(cmd, need_h, tag=""):
    """Run a child process under a wall-clock cap. Returns (returncode, combined output)."""
    to = int(min(need_h, max(left_h(), 0.05)) * 3600)
    try:
        p = subprocess.run(cmd, timeout=to, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if p.stdout:
            print(p.stdout, flush=True)
        return p.returncode, (p.stdout or "")
    except subprocess.TimeoutExpired as e:
        out = e.stdout or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        if out:
            print(out, flush=True)
        print(f"\n[TIMEOUT] {tag or 'stage'} exceeded {to/3600:.2f} h and was killed", flush=True)
        return 124, out

def filled_count(root, seed):
    """Records that are genuinely usable: status filled AND real, non-placeholder text.

    Counting on the flag alone reported an arm as complete when 33 of its records had
    status="filled" with EMPTY symptom_text, and let 4 literal "TODO:" strings reach CLIP."""
    d = Path(root) / str(seed)
    if not d.exists():
        return 0
    n = 0
    for f in d.glob("*.json"):
        try:
            for r in json.load(open(f, encoding="utf-8")):
                t = (r.get("symptom_text") or "").strip()
                if r.get("status") == "filled" and t and not t.upper().startswith("TODO"):
                    n += 1
        except Exception:
            pass
    return n

def bundle(part, extra_receipt=None):
    """Zip the JSON + descriptor text (no images, no checkpoints) and show a download link."""
    stage = WORK / "_bundle"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    files = []
    for f in sorted(RESULTS.glob("*.json")):
        d = stage / "results"; d.mkdir(exist_ok=True)
        shutil.copy2(f, d / f.name); files.append(f"results/{f.name}")
    for arm_dir in ("descriptors_ungrounded", "descriptors_grounded_matched",
                    "descriptors_ungrounded_short", "descriptors_grounded_matched_short"):
        src = REPO / arm_dir
        if not src.exists():
            continue
        for f in sorted(src.rglob("*.json")):
            rel = f.relative_to(REPO)
            (stage / rel.parent).mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, stage / rel); files.append(str(rel).replace("\\", "/"))
    cov = REPO / "docs" / "paper" / "descriptor_coverage.json"
    if cov.exists():
        shutil.copy2(cov, stage / "descriptor_coverage.json"); files.append("descriptor_coverage.json")
    receipt = {"part": part, "files": files, "images": N_IMGS, "classes": len(CLASSES),
               "crops": len(CROPS), "repo_head": HEAD,
               "sage_revision": "bc9bd2899f19379be29c7a99d37d2e89bf8e430d",
               "wall_hours": round(elapsed_h(), 2)}
    if extra_receipt:
        receipt.update(extra_receipt)
    json.dump(receipt, open(stage / "BUNDLE.json", "w"), indent=1)
    zp = WORK / f"pde_part{part}.zip"
    if zp.exists():
        zp.unlink()
    shutil.make_archive(str(WORK / f"pde_part{part}"), "zip", stage)
    print(f"\n[bundle] {zp}  ({zp.stat().st_size/1e6:.2f} MB, {len(files)} files)", flush=True)
    print(f"[bundle] wall time {receipt['wall_hours']} h", flush=True)
    try:
        from IPython.display import FileLink, display
        print("\nDownload:", flush=True)
        display(FileLink(str(zp.relative_to(WORK))))
    except Exception:
        print(f"\nDownload from the Output tab: {zp}", flush=True)
    return zp

# ================================================================ 1  linear probe
if (RESULTS / "probe_seen_C.json").exists():
    print("[skip] probe (already have probe_seen_C.json)", flush=True)
elif ok_to_start("probe", [], 0.8):
    banner("seen-crop linear probe A/B/C")
    sh([sys.executable, "-u", str(S / "probe_seen_all.py"), "--workers", "2"], 2.0, "probe")

# ================================================================ 2  leave-one-crop-out
if (RESULTS / "loco_s0_rich.json").exists():
    print("[skip] loco", flush=True)
elif ok_to_start("loco", [], 0.5):
    banner("leave-one-crop-out + bootstrap CIs")
    sh([sys.executable, "-u", str(S / "loco.py"), "--model", "s0", "--strategy", "rich"],
       1.0, "loco")

# ================================================================ 3  WiSE-FT
# workers=0 on purpose: a CUDA context and a loaded model exist before the loader is
# built, and spawning workers around that killed them outright.
if (RESULTS / "wiseft.json").exists():
    print("[skip] wiseft", flush=True)
elif not (S / "wiseft.py").exists():
    print("\n[wiseft] scripts/wiseft.py missing -> SKIPPED; numbers stay OLD-BUILD.", flush=True)
elif ok_to_start("wiseft", [], 1.0):
    banner("WiSE-FT alpha sweep (both sides under one protocol)")
    # Fine-tuning the whole visual tower needs far more memory than the frozen passes that
    # precede it, so WiSE-FT is the one stage here that can OOM. Try progressively smaller
    # batches rather than losing the stage; if none fit, carry on -- probe and LOCO are
    # already saved and are unaffected.
    _wise_ok = False
    for _bs in (64, 32, 16):
        _rc, _out = sh([sys.executable, "-u", str(S / "wiseft.py"), "--model", "s0", "--exp", "C",
                        "--epochs", str(WISE_EPOCHS), "--lr", WISE_LR, "--batch", str(_bs),
                        "--workers", "0", "--extract-workers", "2",
                        "--max-per-class", "200",
                        "--alphas", *WISE_ALPHAS], 2.5, f"wiseft@{_bs}")
        if (RESULTS / "wiseft.json").exists():
            _wise_ok = True
            break
        if "OutOfMemoryError" in _out or _rc not in (0, 124):
            print(f"[wiseft] batch {_bs} failed -> retrying smaller", flush=True)
            continue
        break
    if not _wise_ok:
        print("[wiseft] no result. The WiSE-FT table stays OLD-BUILD and must be labelled", flush=True)
        print("[wiseft] as such. Everything else in this part is unaffected.", flush=True)
    _w = RESULTS / "wiseft.json"
    if _w.exists():
        try:
            _d = json.load(open(_w, encoding="utf-8"))
            if _d.get("warnings"):
                print("\n[wiseft] THIS RESULT HAS WARNINGS -- do not use the table until fixed:",
                      flush=True)
                for _m in _d["warnings"]:
                    print(f"   - {_m}", flush=True)
            else:
                print("\n[wiseft] all sanity checks passed.", flush=True)
        except Exception:
            pass

# ================================================================ bundle
#__P2_TAIL__
bundle(2, {"wise_epochs": WISE_EPOCHS, "wise_lr": WISE_LR})
banner("PART 2 DONE")
print("NEXT: PART 3 runs the 14 supervised CNNs. Attach this output to it as well.", flush=True)
