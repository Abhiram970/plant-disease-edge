"""
=====================================================================================
 PDE FIX-UP RUN  --  THE 14 CNN BASELINES AND THE WiSE-FT SWEEP
=====================================================================================
Everything else is already done and is carried forward from the attached dataset, not
recomputed. This run exists to repair the two stages the 2026-09-06 morning run lost.

SETUP
  1. Add Data -> `pde-sage-data`
  2. Add Data -> the output of the morning run (results and descriptors carry forward)
  3. GPU T4 x2, Internet ON. No API key needed: no descriptors are generated here.

WHAT FAILED, AND WHAT CHANGED

  CNNs -- 0 of 14 completed.
    Ten architectures (MobileNetV3/V4, EfficientNet, FastViT, ConvNeXt-V2, EfficientNetV2)
    died in seconds with "GET was unable to find an engine to execute this computation"
    inside their depthwise convolutions. torch.cuda.is_bf16_supported() returns True on the
    T4, but Turing has no native bf16, and cuDNN has no depthwise engine for the emulated
    path. Precision is now chosen from the compute capability -- fp16 on the T4, bf16 only
    on Ampere and later -- with an automatic step-down ladder if a kernel is still missing.
    The batch probe walks the same ladder, so it can no longer report a failure and then
    hand back the batch anyway, which is what let all ten walk into an identical error.

    The other four (densenet121, regnety_040, resnet50, resnet101) trained correctly and
    were killed by the 0.75 h per-architecture cap partway through epoch 2 -- densenet121
    had already reached 81.6% and resnet50 79.8% on epoch 1. The cap is now 1.6 h, and a
    checkpoint is no longer deleted unless the architecture actually produced its JSON, so
    an overrun resumes instead of starting over.

  WiSE-FT -- completed, but the sweep was not usable.
    seen went 58.7 -> 45.1 -> 63.8 while unseen fell 21.6 -> 7.8 -> 1.8. A midpoint below
    both endpoints means the frozen and fine-tuned weights are not linearly connected, so
    interpolating between them is meaningless. The head was randomly initialised and trained
    jointly with the unfrozen encoder, so its early gradients pushed the encoder out of the
    pretrained basin. The head is now fitted on frozen features first and used to warm-start
    fine-tuning (that fit doubles as the alpha=0 reference, so it costs no extra pass), the
    encoder is excluded from weight decay, and the sweep runs 5 alphas instead of 3.

IF IT STOPS EARLY: re-run the same cell. Finished architectures are skipped and partially
trained ones resume from their checkpoint.
=====================================================================================
"""

# ---------------------------------------------------------------- settings
BUDGET_H    = 11.0
WISE_EPOCHS = 3
WISE_LR     = "1e-5"     # standard CLIP fine-tuning range; see the note below
WISE_ALPHAS = ["0.0", "0.25", "0.5", "0.75", "1.0"]
CNN_EPOCHS  = 4
CNN_BATCH   = 96
CNN_WORKERS = 2      # 4 vCPUs: 2 workers + prefetch beats 4
CNN_AMP     = True
CNN_MAX_H   = 1.6
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
# Checkpoints of architectures that ran out of time last session. Without this the
# retain-on-timeout rule is inert ACROSS sessions: the .pt sits in /kaggle/input, nothing
# copies it to CKPT, --resume finds no file and the architecture restarts from epoch 0 --
# which is the outcome the retention fix exists to prevent. Only checkpoints without a
# matching result JSON are worth importing; a finished architecture needs none, and each
# file is 30-350 MB, so importing indiscriminately would waste minutes and disk.
_PCK = find_dir("checkpoints", ["/kaggle/input"])
if _PCK and _PCK.exists():
    _n = _skipped = 0
    for _f in sorted(_PCK.glob("*_ckpt.pt")):
        _arch = _f.name[:-len("_ckpt.pt")]
        if (RESULTS / f"supervised_{_arch}.json").exists():
            _skipped += 1
            continue
        if not (CKPT / _f.name).exists():
            shutil.copy2(_f, CKPT / _f.name); _n += 1
    if _n or _skipped:
        print(f"[prior] imported {_n} checkpoint(s) for --resume"
              + (f"; skipped {_skipped} already finished" if _skipped else ""), flush=True)

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

def arm_ceiling(root, seeds):
    """The most records any seed of this arm actually achieved.

    A fixed MIN_FILLED cannot work here. Some held-out labels are not diseases at all --
    config.EXCLUDE_LABELS already names the three Wheat Resistance_Phenotype entries -- and a
    grounding prompt that forbids uncited claims correctly refuses them. In the 2026-09-04 run
    seven classes failed in EVERY grounded_matched seed (the three phenotypes plus
    Coffee/Berry_Blotch, Coffee/Phoma, Orange/Whisker_Mold, Wheat/Fusarium_Wilts), putting the
    real ceiling at 44 of 51. MIN_FILLED=48 therefore demanded more than the arm could ever
    produce, and every seed was rejected -- discarding 2.9 h of generation and leaving the
    matched arm, the one that removes the model-version confound, unevaluated.

    Judging each seed against the best its own arm managed keeps the gate meaningful (a seed
    that fell short of its peers is still excluded) without demanding the impossible.
    """
    return max((filled_count(root, s) for s in seeds), default=0)


def usable_seeds(root, seeds, min_filled, tolerance=2):
    """Seeds good enough to evaluate: at least min_filled, or within `tolerance` of the
    arm's own ceiling when that ceiling is itself below min_filled."""
    ceiling = arm_ceiling(root, seeds)
    floor = min_filled if ceiling >= min_filled else max(ceiling - tolerance, 1)
    out = []
    for s in seeds:
        n = filled_count(root, s)
        if n == 0:
            continue
        if n >= floor:
            out.append(s)
        elif (Path(root) / str(s)).exists():
            print(f"  [reject] {Path(root).name} seed {s}: {n} filled, below {floor}", flush=True)
    if 0 < ceiling < min_filled and out:
        print(f"  [note] {Path(root).name}: ceiling is {ceiling}/{min_filled} because some "
              f"held-out labels are not diseases and the grounding prompt correctly refuses "
              f"them; judging seeds against that ceiling instead.", flush=True)
    return out


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

def _wiseft_is_current():
    """True when results/wiseft.json came from the repaired protocol (see scripts/wiseft.py)."""
    _p = RESULTS / "wiseft.json"
    if not _p.exists():
        return False
    try:
        return int(json.loads(_p.read_text(encoding="utf-8")).get("protocol_version", 0)) >= 2
    except Exception:
        return False

# ================================================================ 3  WiSE-FT
# workers=0 on purpose: a CUDA context and a loaded model exist before the loader is
# built, and spawning workers around that killed them outright.
if not globals().get("RUN_WISEFT", True):
    print("[skip] wiseft (RUN_WISEFT=False)", flush=True)
elif not (S / "wiseft.py").exists():
    print("\n[wiseft] scripts/wiseft.py missing -> SKIPPED; numbers stay OLD-BUILD.", flush=True)
elif _wiseft_is_current():
    print("[skip] wiseft (results/wiseft.json already uses protocol_version 2)", flush=True)
elif ok_to_start("wiseft", [], 1.0):
    banner("WiSE-FT alpha sweep (both sides under one protocol)")
    _stale = RESULTS / "wiseft.json"
    if _stale.exists():
        _keep = RESULTS / "wiseft_SUPERSEDED_2026-09-06.json"
        if not _keep.exists():
            _stale.replace(_keep)
            print(f"[wiseft] quarantined the superseded sweep -> {_keep.name}", flush=True)
            print("[wiseft] (alpha=0.5 fell below both endpoints; head warmup is the fix)",
                  flush=True)
        else:
            _stale.unlink()
            print("[wiseft] discarded a carried-forward pre-fix sweep "
                  f"({_keep.name} already holds the record)", flush=True)
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


# A single forward+backward on random data tells us in seconds whether a batch fits,
# instead of discovering it minutes into a real epoch and losing that epoch.
_PROBE_SRC = """
# Walks the same precision ladder as supervised_baseline.py and prints the first rung that
# actually executes, so the probe can never bless a configuration the real run will reject.
import sys, torch, timm
arch, bs, amp = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1"
cc = torch.cuda.get_device_capability()
plan = []
if amp:
    if cc[0] >= 8:                       # bf16 is native only on Ampere and later
        plan.append(("bf16", torch.bfloat16, True))
    plan += [("fp16", torch.float16, True), ("fp16-contig", torch.float16, False)]
plan.append(("fp32", torch.float32, False))
ENGINE = "unable to find an engine"
last = ""
for name, dt, cl in plan:
    try:
        m = timm.create_model(arch, pretrained=False, num_classes=166).cuda()
        m = m.to(memory_format=torch.channels_last if cl else torch.contiguous_format)
        o = torch.optim.AdamW(m.parameters(), lr=1e-4)
        x = torch.randn(bs, 3, 224, 224, device="cuda")
        x = x.to(memory_format=torch.channels_last if cl else torch.contiguous_format)
        y = torch.randint(0, 166, (bs,), device="cuda")
        with torch.autocast("cuda", dtype=dt, enabled=(dt is not torch.float32)):
            loss = torch.nn.functional.cross_entropy(m(x), y)
        loss.backward(); o.step(); torch.cuda.synchronize()
        print("FIT", name); break
    except torch.OutOfMemoryError:
        print("OOM"); break            # a smaller batch is the answer, not a lower precision
    except RuntimeError as e:
        last = f"{type(e).__name__} {e}"
        del m, o
        torch.cuda.empty_cache()
        if ENGINE in str(e):
            continue                   # try the next rung
        print("ERR", last); break
    except Exception as e:
        print("ERR", type(e).__name__, e); break
else:
    print("ERR no precision worked:", last)
"""
_probe = WORK / "_probe_batch.py"
_probe.write_text(_PROBE_SRC)

def largest_fitting_batch(arch, start):
    """Largest batch that completes a real step, or None if no configuration works.

    Returning the requested batch on a non-OOM error -- as this did before -- meant the probe
    printed the cuDNN engine failure and then handed back the batch anyway, so all ten
    depthwise architectures walked into an identical failure and the sweep finished 0/14.
    A probe that cannot find any working precision now says so, and the caller skips the
    architecture instead of spending its whole time budget failing.
    """
    bs = start
    while bs >= 16:
        rc, out = sh([sys.executable, str(_probe), arch, str(bs), "1" if CNN_AMP else "0"],
                     0.12, f"probe {arch}@{bs}")
        if "FIT" in out:
            prec = out.split("FIT", 1)[1].strip().split()[0:1]
            print(f"    [probe] {arch} @ batch {bs}: fits"
                  f"{' (' + prec[0] + ')' if prec else ''}", flush=True)
            return bs
        if "OOM" in out:
            print(f"    [probe] {arch} @ batch {bs}: OOM -> trying {bs // 2}", flush=True)
            bs //= 2
            continue
        print(f"    [probe] {arch}: no working configuration -- {out.strip()[-200:]}",
              flush=True)
        return None
    return 16

banner(f"supervised CNNs ({len(ARCHS)} architectures x {CNN_EPOCHS} epochs)")
done, skipped, unsupported = [], [], []
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
    if bs is None:
        print(f"    [skip] {arch}: probe found no workable precision on this GPU", flush=True)
        unsupported.append(arch)
        continue

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

    # Delete the checkpoint ONLY once the JSON proves the architecture finished. Clearing it
    # unconditionally is what made the 2026-09-06 timeouts total losses: densenet121 and
    # resnet50 had each completed a full epoch and checkpointed it, and the cleanup threw that
    # away, so the re-run restarted them from scratch. A kept checkpoint makes --resume real.
    if arch in done:
        for ck in CKPT.glob(f"{arch}*"):
            try:
                ck.unlink()
            except Exception:
                pass
    else:
        _keep = [c.name for c in CKPT.glob(f"{arch}*")]
        if _keep:
            print(f"    [keep] {arch}: checkpoint retained for --resume ({_keep[0]})",
                  flush=True)
    try:
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.ipc_collect()
    except Exception:
        pass
    fg, tg = gpu_free_gb()
    print(f"    [clear] VRAM now {fg:.1f}/{tg:.1f} GB free", flush=True)

print(f"\n[cnn] completed {len(done)}/{len(ARCHS)}: {done}", flush=True)
if skipped:
    print(f"[cnn] NOT RUN (budget): {skipped}", flush=True)
if unsupported:
    print(f"[cnn] NOT RUN (no workable precision on this GPU): {unsupported}", flush=True)
    print("[cnn] Re-run this cell to continue.", flush=True)


# ================================================================ regenerate + bundle
# The generators read the JSONs sitting NEXT TO THEM in docs/paper, not results/, so this
# run's outputs have to be staged across first -- otherwise the tables regenerate from the
# previous build's numbers, which is exactly the build mixture the audit flagged. `CODE` does
# not exist in this bootstrap; the clone is `REPO`.
banner("regenerating tables and figures from the repaired results")
_docs = REPO / "docs" / "paper"
for _g in ("make_tex_tables.py", "make_figures.py"):
    if (_docs / _g).exists():
        for _f in RESULTS.glob("*.json"):
            shutil.copy2(_f, _docs / _f.name)
        _r = subprocess.run([sys.executable, "-u", str(_docs / _g)], text=True, cwd=str(_docs),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(_r.stdout or f"[{_g}] no output", flush=True)

bundle("fixup", {"cnn_epochs": CNN_EPOCHS, "cnn_max_h": CNN_MAX_H,
                 "wise_epochs": WISE_EPOCHS, "wise_lr": WISE_LR,
                 "wise_alphas": WISE_ALPHAS, "cnn_completed": done,
                 "cnn_not_run": skipped, "cnn_unsupported": unsupported})
banner("FIX-UP RUN DONE" if (len(done) == len(ARCHS) and not unsupported)
       else "FIX-UP RUN INCOMPLETE -- re-run to finish")
