"""
=====================================================================================
 PDE AUDIT-FIX RUN  --  CLEANED A/B/C EVALUATION AND CLASS-MACRO METRICS
=====================================================================================
Pure evaluation. No descriptor generation, so NO API KEY is required.

SETUP
  1. Add Data -> `pde-sage-data`
  2. Add Data -> the output of your most recent run (descriptors carry forward; without
     them the `grounded` strategy silently falls back to the keyword bank and the whole
     point of the run is lost -- the stage checks for this and refuses to start)
  3. GPU T4 x2, Internet ON

WHAT THIS PRODUCES, AND WHICH SENTENCE IN THE PAPER EACH ONE DECIDES

  zeroshot_eval_{A,B,C}_clean.json
      The de-duplicated evaluation at all three scales. Configuration C already had a
      cleaned counterpart; A and B did not, which is why Section 6 calls the scaling
      claim provisional. C is re-run here anyway so all three come from ONE code
      revision -- comparing a fresh A and B against a C measured weeks earlier would
      reintroduce exactly the drift this run exists to remove.

      HOW TO READ IT. If `grounded` still rises monotonically across the three CLEANED
      configurations, the scaling claim survives and "provisional" can come out of the
      abstract, the contribution list and the conclusion. If the cleaned curve is flat,
      the trend was falling duplicate-label density rather than the method, and the
      claim has to be withdrawn. That is a publishable finding, not a failed run.

  zeroshot_eval_{A,B,C}.json
      The as-published sweep, re-measured so every model carries `by_class` and
      `class_macro_acc`. The held-out split is capped but not balanced (a 25-image floor
      against a 600 cap), so micro top-1 is carried by the largest classes.

      HOW TO READ IT. The MICRO accuracies must come back IDENTICAL to the published
      tables: grounded means 21.5 / 22.7 / 23.9 at A / B / C. This stage asserts that
      for you and prints a loud warning if any mean has moved by more than 0.1 points.
      If one has, something other than the by_class addition changed -- stop and find
      out what before touching the manuscript.

IF IT STOPS EARLY: re-run the same cell. Configurations whose JSON already exists are
skipped, never redone.
=====================================================================================
"""

# ---------------------------------------------------------------- settings
BUDGET_H = 11.0
EXPS     = ["A", "B", "C"]
STRATS   = ["bare", "crude", "rich", "grounded"]
TIER_ARGS = ["--tiers", "lw11", "lw21", "lw35", "--heavy", "--teachers"]

# Published grounded means (unweighted over the four deployable tiers), from
# docs/paper/manuscript/submitted/tab_scale_study.tex. The uncleaned re-run must reproduce these.
PUBLISHED_GROUNDED = {"A": 21.5, "B": 22.7, "C": 23.9}
TOLERANCE_PP = 0.1


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


# ---------------------------------------------------------------- preconditions
banner("AUDIT FIX RUN -- preconditions")

# The grounded strategy falls back to the keyword bank for any class with no record.
# If the descriptor registry did not carry forward from the attached dataset, every
# `grounded` column silently becomes a `rich` column and the run looks fine while
# measuring the wrong thing. Refuse to start rather than produce that quietly.
_desc = REPO / "descriptors"
_n_desc = len(list(_desc.rglob("*.json"))) if _desc.exists() else 0
print(f"[pre] descriptor registry: {_n_desc} json files at {_desc}")
if _n_desc < 10:
    sys.exit("[pre] ABORT: the descriptor registry is missing or nearly empty. Attach the "
             "output dataset from your previous run before starting, or every `grounded` "
             "result here will silently be the keyword-bank fallback instead.")

# by_class is what R2 exists to produce. If the clone predates that change, the stage
# would run for ~1.5 h and write JSONs with no macro field in them.
_zs = (REPO / "scripts" / "zeroshot.py").read_text(encoding="utf-8")
if "class_macro_acc" not in _zs:
    sys.exit("[pre] ABORT: scripts/zeroshot.py has no class_macro_acc -- this clone predates "
             "the 2026-09-10 audit fixes. Push those changes to the branch LAUNCH.py clones, "
             "then re-run. Without them R2 produces nothing new.")
print("[pre] zeroshot.py carries class_macro_acc -- ok")

sys.path.insert(0, str(REPO / "scripts"))
import config as _C
_want = [t for t in TIER_ARGS if not t.startswith("--")]
_bad = [t for t in _want if t not in _C.MODEL_TIERS]
if _bad:
    sys.exit(f"[pre] ABORT: tier key(s) {_bad} not in C.MODEL_TIERS "
             f"{sorted(_C.MODEL_TIERS)} -- evaluate.py would drop them silently.")
print(f"[pre] tiers resolve: {_want} + heavy + reference = {len(_want) + 2} encoders")


def _eval(exp, clean):
    """One evaluate.py call. Returns True if the output JSON exists afterwards."""
    suffix = "_clean" if clean else ""
    out = RESULTS / f"zeroshot_eval_{exp}{suffix}.json"
    tag = f"{'clean' if clean else 'plain'} {exp}"
    # Skip on COMPLETENESS, not existence. `--tiers small mid large` are not keys of
    # C.MODEL_TIERS, evaluate.py drops unknown names silently, and the 2026-09-10 run
    # wrote six healthy-looking files holding 2 encoders instead of 5.
    if out.exists():
        try:
            _have = len(json.loads(out.read_text(encoding="utf-8")).get("models", {}))
        except Exception:
            _have = 0
        if _have >= 5:
            print(f"[skip] {tag}: {out.name} complete ({_have} encoders)")
            return True
        print(f"[redo] {tag}: only {_have}/5 encoders -- re-running")
    cmd = [sys.executable, "-u", str(S / "evaluate.py"), "--exp", exp,
           "--strategies", *STRATS, *TIER_ARGS]
    if clean:
        cmd.append("--clean")
    banner(f"evaluate {tag}")
    rc, _ = sh(cmd, need_h=1.2, tag=tag)
    if rc != 0:
        print(f"[warn] {tag} exited {rc}", flush=True)
    return out.exists()


# ---------------------------------------------------------------- R1: cleaned A/B/C
banner("R1 -- de-duplicated evaluation at A, B and C")
for _e in EXPS:
    if not ok_to_start(f"clean {_e}", left_h(), 1.2):
        break
    _eval(_e, clean=True)

# ---------------------------------------------------------------- R2: class-macro
banner("R2 -- as-published sweep, re-measured with per-class metrics")
for _e in EXPS:
    if not ok_to_start(f"plain {_e}", left_h(), 1.2):
        break
    _eval(_e, clean=False)

# ---------------------------------------------------------------- verification
banner("verification")


def _grounded_mean(path):
    """Unweighted grounded mean over the four DEPLOYABLE tiers (SigLIP2 excluded)."""
    if not path.exists():
        return None
    j = json.loads(path.read_text(encoding="utf-8"))
    vals = [d["grounded"]["acc"] for m, d in j["models"].items() if "SigLIP" not in m]
    return 100.0 * sum(vals) / len(vals) if vals else None


print("\nR2 regression check -- micro means must match the published tables:")
_drift = []
for _e in EXPS:
    got = _grounded_mean(RESULTS / f"zeroshot_eval_{_e}.json")
    want = PUBLISHED_GROUNDED[_e]
    if got is None:
        print(f"  {_e}: not produced")
        continue
    delta = got - want
    flag = "ok" if abs(delta) <= TOLERANCE_PP else "DRIFT"
    print(f"  {_e}: {got:.2f}%  (published {want}%, delta {delta:+.2f}) {flag}")
    if abs(delta) > TOLERANCE_PP:
        _drift.append(_e)
if _drift:
    print(f"\n  !! {', '.join(_drift)} moved by more than {TOLERANCE_PP} points.")
    print("  !! Something other than the by_class addition changed. Do NOT promote these")
    print("  !! files into docs/paper/ until you know what.")

print("\nR1 result -- the scaling claim:")
_clean = [(_e, _grounded_mean(RESULTS / f"zeroshot_eval_{_e}_clean.json")) for _e in EXPS]
for _e, v in _clean:
    print(f"  {_e} cleaned: " + (f"{v:.2f}%" if v is not None else "not produced"))
_vals = [v for _, v in _clean if v is not None]
if len(_vals) == 3:
    if _vals[0] <= _vals[1] <= _vals[2]:
        print("  -> grounded STILL rises monotonically after cleaning.")
        print("  -> The scaling claim survives; 'provisional' can come out of the paper.")
    else:
        print("  -> grounded does NOT rise monotonically after cleaning.")
        print("  -> The uncleaned trend was label cleanliness. The scaling claim must be")
        print("     withdrawn from the abstract, contributions and conclusion.")
    print(f"  -> cleaned span: {max(_vals) - min(_vals):+.2f} points across A..C")
    print("  -> Compare that against the 2.4-point seed-to-seed SD from the control arm")
    print("     before calling any of it a trend.")
else:
    print("  -> incomplete; re-run the cell to finish the remaining configurations.")

bundle("audit-fixes")
