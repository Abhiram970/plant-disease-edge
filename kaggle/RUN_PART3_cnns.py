"""
=====================================================================================
 PDE PART 3 of 3  --  THE 14 SUPERVISED CNN BASELINES
=====================================================================================
Run this LAST. Needs no API key. This is the long one.

SETUP
  1. Add Data -> `pde-sage-data`.
  2. Add Data -> the outputs of PART 1 and PART 2 (imported automatically).
  3. GPU on. Paste into ONE cell and run.

WHAT IT DOES                                                        est. time
  14 architectures x 4 epochs, one SUBPROCESS each                   ~4.8 h

WHY THE LAST RUN NEVER FINISHED THESE. Epochs took ~1400 s (23 min), so 8 epochs x 14
architectures = 43 h in a 12 h session. The cause was the DataLoader plus no mixed
precision at all -- not the GPU. Fixed: AMP (bf16, fp16 fallback), channels_last,
persistent workers with prefetch and pinned memory, and 4 epochs instead of 8 (the logged
curves were already flat after epoch 3: 83.7 / 85.4 / 85.5 / 85.3 / 85.9).

MEMORY IS FULLY RELEASED BETWEEN ARCHITECTURES. Each model trains in its own subprocess,
so when it exits the OS reclaims every byte of VRAM and host RAM -- torch.cuda.empty_cache()
inside one long-lived process does not do that, which is what produced the epoch-4 OOM.
The JSON is written and verified readable BEFORE checkpoints are deleted and the next
architecture starts.

THREE MORE FIXES FROM THE LAST LOG
  * The "free 20.3 GB" guard was reading system RAM while the T4 has 14.56 GiB of VRAM,
    so it never prevented an OOM. It now reads torch.cuda.mem_get_info().
  * An OOM was discovered minutes into an epoch, throwing that epoch away, twice. The
    batch size is now probed with one forward+backward on random data first (seconds).
  * A 1.5 h TIMEOUT was misread as an OOM and the batch halved, making the retry slower.
    Only a real torch.OutOfMemoryError now triggers a reduction.

IF IT STOPS EARLY: re-run the same cell. Completed architectures are skipped, and the
receipt lists exactly which ones still need to run.
=====================================================================================
"""

# ---------------------------------------------------------------- settings
BUDGET_H    = 11.0
CNN_EPOCHS  = 4
CNN_BATCH   = 96
CNN_WORKERS = 2      # 4 vCPUs: 2 workers + prefetch beats 4
CNN_AMP     = True
CNN_MAX_H   = 0.75   # per architecture
ARCHS = ["mobilenetv3_small_100", "mobilenetv4_conv_small", "fastvit_t8", "efficientnet_b0",
         "mobilenetv3_large_100", "densenet121", "mobilenetv4_conv_medium", "fastvit_sa12",
         "convnextv2_nano", "regnety_040", "resnet50", "tf_efficientnetv2_s",
         "convnextv2_tiny", "resnet101"]
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
# A stale clone must actually be gone before cloning. rmtree(ignore_errors=True) can fail
# silently -- a read-only .git object, a file still held open -- and the clone then aborts
# with "destination path already exists", killing the whole run at t+0. Retry, then fall
# back to cloning into a fresh directory rather than dying.
if REPO.exists():
    shutil.rmtree(REPO, ignore_errors=True)
if REPO.exists():
    def _force_rm(func, path, exc):
        import stat
        try:
            os.chmod(path, stat.S_IWRITE); func(path)
        except Exception:
            pass
    shutil.rmtree(REPO, onerror=_force_rm)
if REPO.exists():
    _alt = WORK / "pde_fresh"
    _n = 1
    while _alt.exists():
        _n += 1; _alt = WORK / f"pde_fresh{_n}"
    print(f"[bootstrap] could not remove the stale clone at {REPO}; using {_alt}", flush=True)
    REPO = _alt
    S = REPO / "scripts"
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

def _ensure_manifest():
    """Build manifest.csv if it is missing.

    build_ungrounded.py, wiseft.py and supervised_baseline.py all read it -- it is the
    class list and the seen/held split. The old single-file runner built it; splitting that
    runner into parts dropped the step, so descriptor generation exited immediately with
    "manifest not found" for every seed and every arm, the integrity gate then rejected all
    of them, and the control arms silently had nothing to evaluate. Zero-shot still ran
    because it reads the dataset directly, which is why the failure looked survivable in the
    log when it was not.
    """
    mf = WORK / "manifest.csv"
    if mf.exists() and mf.stat().st_size > 0:
        print(f"[manifest] present ({mf})", flush=True)
        return True
    print("[manifest] building (needed by descriptors, WiSE-FT and the CNNs) ...", flush=True)
    r = subprocess.run([sys.executable, "-u", str(S / "build_manifest.py"),
                        "--min-images", "25"], text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if r.stdout:
        print(r.stdout, flush=True)
    if not (mf.exists() and mf.stat().st_size > 0):
        print("[manifest] FAILED -- descriptor generation and WiSE-FT cannot run.", flush=True)
        return False
    return True


_HAVE_MANIFEST = _ensure_manifest()


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

# A single forward+backward on random data tells us in seconds whether a batch fits,
# instead of discovering it minutes into a real epoch and losing that epoch.
_PROBE_SRC = """
import sys, torch, timm
arch, bs, amp = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1"
try:
    m = timm.create_model(arch, pretrained=False, num_classes=166).cuda()
    m = m.to(memory_format=torch.channels_last)
    o = torch.optim.AdamW(m.parameters(), lr=1e-4)
    dt = torch.bfloat16 if (amp and torch.cuda.is_bf16_supported()) else torch.float16
    x = torch.randn(bs, 3, 224, 224, device="cuda").to(memory_format=torch.channels_last)
    y = torch.randint(0, 166, (bs,), device="cuda")
    with torch.autocast("cuda", dtype=dt, enabled=amp):
        loss = torch.nn.functional.cross_entropy(m(x), y)
    loss.backward(); o.step(); torch.cuda.synchronize()
    print("FIT")
except torch.OutOfMemoryError:
    print("OOM")
except Exception as e:
    print("ERR", type(e).__name__, e)
"""
_probe = WORK / "_probe_batch.py"
_probe.write_text(_PROBE_SRC)

def largest_fitting_batch(arch, start):
    bs = start
    while bs >= 16:
        rc, out = sh([sys.executable, str(_probe), arch, str(bs), "1" if CNN_AMP else "0"],
                     0.08, f"probe {arch}@{bs}")
        if "FIT" in out:
            return bs
        if "OOM" not in out:
            return bs   # unrelated failure: let the real run surface it properly
        print(f"    [probe] {arch} @ batch {bs}: OOM -> trying {bs // 2}", flush=True)
        bs //= 2
    return 16

banner(f"supervised CNNs ({len(ARCHS)} architectures x {CNN_EPOCHS} epochs)")
done, skipped = [], []
for i, arch in enumerate(ARCHS, 1):
    out_json = RESULTS / f"supervised_{arch}.json"
    if out_json.exists():
        print(f"[skip] {arch}", flush=True); done.append(arch); continue

    remaining = [a for a in ARCHS[i - 1:] if not (RESULTS / f"supervised_{a}.json").exists()]
    if not ok_to_start(f"cnn {arch}", remaining, CNN_MAX_H * 0.55):
        skipped = remaining
        break

    fg, tg = gpu_free_gb()
    print(f"\n--- cnn {i}/{len(ARCHS)}: {arch} --- t+{elapsed_h():.1f} h | "
          f"VRAM {fg:.1f}/{tg:.1f} GB free", flush=True)

    bs = largest_fitting_batch(arch, CNN_BATCH)
    if bs != CNN_BATCH:
        print(f"    [probe] using batch {bs}", flush=True)

    cmd = [sys.executable, "-u", str(S / "supervised_baseline.py"),
           "--arch", arch, "--epochs", str(CNN_EPOCHS), "--batch", str(bs),
           "--workers", str(CNN_WORKERS), "--resume"]
    if CNN_AMP:
        cmd.append("--amp")
    rc, out = sh(cmd, CNN_MAX_H, f"cnn {arch}")

    # Only a REAL OOM justifies halving the batch. A timeout does not -- last run that
    # confusion made convnextv2_tiny restart at a smaller, slower batch.
    if rc not in (0, 124) and "OutOfMemoryError" in out and bs > 16:
        print(f"    [retry] genuine OOM -> batch {bs // 2}", flush=True)
        cmd[cmd.index("--batch") + 1] = str(bs // 2)
        rc, out = sh(cmd, min(CNN_MAX_H, left_h()), f"cnn {arch} retry")

    # Verify the JSON before clearing anything.
    if out_json.exists():
        try:
            d = json.load(open(out_json, encoding="utf-8"))
            print(f"    [ok] {arch}: seen_top1={d.get('seen_top1', 0) * 100:.2f}% "
                  f"({d.get('params_M')}M)", flush=True)
            done.append(arch)
        except Exception as e:
            print(f"    [warn] {arch}: JSON unreadable ({e})", flush=True)
    else:
        print(f"    [miss] {arch}: no JSON (rc={rc}) -- recorded as NOT RUN", flush=True)

    for ck in CKPT.glob(f"{arch}*"):
        try:
            ck.unlink()
        except Exception:
            pass
    try:
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.ipc_collect()
    except Exception:
        pass
    fg, tg = gpu_free_gb()
    print(f"    [clear] checkpoints removed; VRAM now {fg:.1f}/{tg:.1f} GB free", flush=True)

print(f"\n[cnn] completed {len(done)}/{len(ARCHS)}: {done}", flush=True)
if skipped:
    print(f"[cnn] NOT RUN (budget): {skipped}", flush=True)
    print("[cnn] Re-run this cell to continue.", flush=True)

#__P3_TAIL__
bundle(3, {"cnn_epochs": CNN_EPOCHS, "cnn_batch_requested": CNN_BATCH,
           "cnn_completed": done, "cnn_not_run": skipped})
banner("PART 3 DONE" if not skipped else "PART 3 INCOMPLETE -- re-run to finish")
