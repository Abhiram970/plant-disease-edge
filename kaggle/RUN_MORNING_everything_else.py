"""
=====================================================================================
 PDE MORNING  --  everything tonight's 4 h run did not cover
=====================================================================================
Run this after the Kaggle quota resets. Needs the API key only for the extra seeds.

  descriptor seeds 4-7 (2 arms)                                    ~0.7 h
  control arms re-evaluated with all 8 seeds                       ~0.7 h
  short arms (77-token truncation control)                         ~0.7 h
  label-corrected C eval                                           ~0.3 h
  leave-one-crop-out + bootstrap CIs                               ~0.3 h
  WiSE-FT alpha sweep                                              ~0.3 h
  14 supervised CNN baselines                                      ~4.8 h
                                                           TOTAL   ~7.8 h

SETUP
  1. Add Data -> your `pde-sage-data` dataset.
  2. Add Data -> the OUTPUT of tonight's notebook. This carries forward the descriptor
     text and results, so seeds 0-3 are NOT regenerated and the run resumes cleanly.
  3. GPU T4 x2, Internet ON, LAVA_API_KEY in Secrets.
  4. Save Version -> "Save & Run All".

ORDER. The CNNs run LAST despite being the largest block, because everything above them
is short and completes the manuscript's remaining tables. If the session is cut short it
should be cut inside the CNN sweep, where each finished architecture is already saved and
a later run skips it -- not inside a table that would then be half-measured.

WHY THE CONTROL ARMS ARE RE-RUN. Seed evaluations share one image-embedding pass, so
adding seeds 4-7 means re-evaluating the arm rather than appending to it. That costs one
embedding pass (~21 min per arm) and overwrites zeroshot_eval_C_*seeds.json with the full
8-seed result. Going from 4 to 8 seeds tightens the 95% interval from about +/-3.1 pp to
+/-1.6 pp; no seed count resolves the ~0.7 pp gap under test, so the goal is a tight null.

IF IT STOPS EARLY: re-run the same cell. Every stage is resumable and finished work is
skipped.
=====================================================================================
"""

# ---------------------------------------------------------------- settings
BUDGET_H         = 11.0
UNGROUNDED_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]   # 0-3 already exist and are skipped
SHORT_WORDS      = 50
LLM_MODEL        = "claude-sonnet-5"
MAX_TOKENS       = 4000
MIN_FILLED       = 48

EVAL_SHORT_ARMS  = True    # the truncation control, deferred from tonight
RUN_CLEAN_EVAL   = True
RUN_LOCO         = True
RUN_WISEFT       = True

WISE_EPOCHS      = 3
WISE_LR          = "1e-5"
WISE_ALPHAS      = ["0.0", "0.5", "1.0"]

CNN_EPOCHS  = 4
CNN_BATCH   = 96
CNN_WORKERS = 2
CNN_AMP     = True
CNN_MAX_H   = 0.75
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

os.environ["PDE_LLM_MODEL"]  = LLM_MODEL
os.environ["PDE_MAX_TOKENS"] = str(MAX_TOKENS)

try:
    from kaggle_secrets import UserSecretsClient
    _sec = UserSecretsClient()
    for _k in ("LAVA_API_KEY", "ANTHROPIC_API_KEY", "LAVA_BASE_URL", "LAVA_SHAPE"):
        try:
            _v = _sec.get_secret(_k)
            if _v:
                os.environ[_k] = _v
        except Exception:
            pass
except Exception:
    pass
HAVE_KEY = bool(os.environ.get("LAVA_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))
print(f"[llm] api key present: {HAVE_KEY}", flush=True)
if not HAVE_KEY:
    print("[llm] WITHOUT A KEY THIS PART CANNOT PRODUCE THE CONTROL ARMS.", flush=True)
    print("[llm] Add LAVA_API_KEY under Add-ons -> Secrets and re-run.", flush=True)

# ================================================================ 1  descriptors
# build_ungrounded.py takes --which (not --exp); --arm "grounded" writes the
# descriptors_grounded_matched directory, which the evaluator loads as grounded_matched.
if HAVE_KEY:
    banner("descriptors")
    for arm, root in (("ungrounded", REPO / "descriptors_ungrounded"),
                      ("grounded",   REPO / "descriptors_grounded_matched")):
        for s in UNGROUNDED_SEEDS:
            have = filled_count(root, s)
            if have >= MIN_FILLED:
                print(f"[skip] {arm} seed {s} ({have} filled)", flush=True)
                continue
            # Reserve what the DEPENDENT stages still need (zero-shot + control arms).
            # Guarding only on this seed's own cost let generation consume the whole quota
            # and stop immediately before the control arms -- spending the budget and
            # producing nothing for Section 5.3, which is the entire point of the run.
            _reserve = 1.2 + 0.06 * len(UNGROUNDED_SEEDS) * 4
            if not ok_to_start(f"{arm} seed {s}", [], 0.3 + _reserve):
                print(f"[budget] stopping descriptor generation to protect the {_reserve:.1f} h "
                      f"the zero-shot and control-arm stages still need.", flush=True)
                print(f"[budget] seeds completed so far are kept and will be used.", flush=True)
                break
            print(f"\n--- {arm} seed {s} (have {have}, need {MIN_FILLED}) ---", flush=True)
            rc, _ = sh([sys.executable, "-u", str(S / "build_ungrounded.py"),
                        "--arm", arm, "--seed", str(s), "--which", "heldout"],
                       0.6, f"{arm} seed {s}")
            _got = filled_count(root, s)
            print(f"    -> {_got} filled", flush=True)
            if _got == 0:
                print(f"    [ERROR] {arm} seed {s} produced NOTHING. The control arms cannot", flush=True)
                print(f"    [ERROR] run without descriptors -- Section 5.3 will be empty.", flush=True)
                print(f"    [ERROR] Check the message above (missing manifest, bad API key,", flush=True)
                print(f"    [ERROR] exhausted credit) before letting this session continue.", flush=True)
            if rc == 3:
                print("[llm] endpoint reported no credit -> stopping descriptor generation.",
                      flush=True)
                break

# ================================================================ 2  short arms
# Every generated prototype exceeded CLIP's 77-token text window (51/51 ungrounded,
# 35/40 matched), so the full arms are compared on their leading sentences only. These
# arms hold the same text compressed to fit, which removes truncation as a confound.
# It is a deterministic text transform, so it costs nothing and needs no API calls.
banner(f"short arms (<= {SHORT_WORDS} words)")
def shorten(text, n=SHORT_WORDS):
    t = " ".join((text or "").split())
    if not t:
        return t
    w = t.split()
    if len(w) <= n:
        return t
    cut = " ".join(w[:n])
    for sep in (". ", "; ", ", "):
        i = cut.rfind(sep)
        if i > len(cut) * 0.6:
            return cut[:i + 1].rstrip(" ;,")
    return cut

_made = 0
for _src_name, _dst_name in (("descriptors_ungrounded", "descriptors_ungrounded_short"),
                             ("descriptors_grounded_matched", "descriptors_grounded_matched_short")):
    _src = REPO / _src_name
    if not _src.exists():
        continue
    for _sd in sorted(_src.glob("*")):
        if not _sd.is_dir():
            continue
        _dst = REPO / _dst_name / _sd.name
        _dst.mkdir(parents=True, exist_ok=True)
        for _f in _sd.glob("*.json"):
            try:
                _recs = json.load(open(_f, encoding="utf-8"))
            except Exception:
                continue
            for _r in _recs:
                _r["symptom_text"] = shorten(_r.get("symptom_text"))
            json.dump(_recs, open(_dst / _f.name, "w", encoding="utf-8"), indent=1)
            _made += 1
print(f"[short] wrote {_made} files", flush=True)

# ================================================================ 3  integrity gate
banner("descriptor integrity")
USABLE = {}
for _arm, _root in (("ungrounded", REPO / "descriptors_ungrounded"),
                    ("grounded_matched", REPO / "descriptors_grounded_matched"),
                    ("ungrounded_short", REPO / "descriptors_ungrounded_short"),
                    ("grounded_matched_short", REPO / "descriptors_grounded_matched_short")):
    _seeds = []
    for _s in UNGROUNDED_SEEDS:
        _n = filled_count(_root, _s)
        if _n >= MIN_FILLED:
            _seeds.append(_s)
        elif (Path(_root) / str(_s)).exists():
            print(f"  [reject] {_arm} seed {_s}: {_n}/{MIN_FILLED} usable -> excluded", flush=True)
    USABLE[_arm] = _seeds
    print(f"  {_arm:24} usable seeds: {_seeds}", flush=True)
if not any(USABLE.values()):
    print("", flush=True)
    print("  ####################################################################", flush=True)
    print("  #  NO DESCRIPTOR ARM IS USABLE. The control arms will not run and   #", flush=True)
    print("  #  Section 5.3 gets no result -- the main reason for this session.  #", flush=True)
    print("  #  Zero-shot/probe/LOCO below still work, so the run continues, but #", flush=True)
    print("  #  fix the cause above and re-run before spending more quota.       #", flush=True)
    print("  ####################################################################", flush=True)
    print("", flush=True)
json.dump(USABLE, open(RESULTS / "descriptor_arm_integrity.json", "w"), indent=1)

# ================================================================ 4  zero-shot A/B/C
STRATS = ["bare", "crude", "rich", "grounded"]
for _e in ("A", "B", "C"):
    if (RESULTS / f"zeroshot_eval_{_e}.json").exists():
        print(f"[skip] zeroshot {_e}", flush=True); continue
    if not ok_to_start(f"zeroshot {_e}", [], 0.4):
        break
    banner(f"zero-shot {_e}")
    sh([sys.executable, "-u", str(S / "evaluate.py"), "--exp", _e, "--strategies", *STRATS,
        "--tiers", "lw11", "lw21", "lw35", "--heavy", "--teachers"], 1.5, f"zeroshot {_e}")

if not globals().get("RUN_CLEAN_EVAL", True):
    print("[skip] label-corrected C eval (RUN_CLEAN_EVAL=False)", flush=True)
elif not (RESULTS / "zeroshot_eval_C_clean.json").exists() and ok_to_start("clean", [], 0.4):
    banner("zero-shot C, label-corrected")
    sh([sys.executable, "-u", str(S / "evaluate.py"), "--exp", "C", "--clean",
        "--strategies", *STRATS, "--tiers", "lw11", "lw21", "lw35", "--heavy"], 1.5, "clean")

# ================================================================ 5  control arms
# evaluate.py names the output per ARM as well as per seed. It previously wrote
# "_ung{seed}" for every arm, so ungrounded and grounded_matched at the same seed
# silently overwrote each other -- and the matched arm is the one that removes the
# model-version confound, so losing it defeated the whole experiment.
_SUF = {"ungrounded": "ung", "grounded_matched": "gm",
        "ungrounded_short": "ungs", "grounded_matched_short": "gms"}
# ALL seeds of an arm in ONE process. evaluate.py caches image embeddings per model and
# reuses them across strategies, but that cache dies with the process: running 16
# seed-evaluations as 16 subprocesses would re-embed 14,204 images 16 times (~5.5 h,
# far past the budget). Sharing one process per arm amortises the embedding pass over
# every seed and brings the whole block to roughly 25 minutes.
# EVAL_SHORT_ARMS defers the truncation-control arms. They ask whether CLIP's 77-token
# window confounds the comparison -- a real question, but a SECONDARY one that only matters
# once the primary null is established, and each costs a full ~21 min embedding pass. The
# descriptor text is written either way, so they can be evaluated in a later session at no
# extra generation cost by re-running with EVAL_SHORT_ARMS = True.
_CONTROL_ARMS = ["ungrounded", "grounded_matched"]
if globals().get("EVAL_SHORT_ARMS", True):
    _CONTROL_ARMS += ["ungrounded_short", "grounded_matched_short"]
else:
    print("[control] short arms deferred (EVAL_SHORT_ARMS=False); descriptor text is still",
          flush=True)
    print("[control] written, so a later run can evaluate them without regenerating.", flush=True)
for _arm in _CONTROL_ARMS:
    _seeds = USABLE.get(_arm, [])
    if not _seeds:
        continue
    _tag = f"C_{_SUF[_arm]}seeds"
    _out = RESULTS / f"zeroshot_eval_{_tag}.json"
    # Skip only if the existing file already covers EVERY usable seed. A previous session may
    # have written this arm with fewer seeds (tonight runs 4, the morning run extends to 8);
    # a plain exists() check would keep the smaller file and silently discard the new seeds,
    # leaving the interval wider than the run was meant to make it.
    if _out.exists():
        try:
            _have = set(json.load(open(_out, encoding="utf-8")).get("seeds") or [])
        except Exception:
            _have = set()
        if _have >= set(_seeds):
            print(f"[skip] {_tag} (already has seeds {sorted(_have)})", flush=True); continue
        print(f"[redo] {_tag}: file has {sorted(_have)}, need {_seeds} -> re-evaluating",
              flush=True)
    if not ok_to_start(_tag, [], 0.35):
        break
    banner(f"control arm {_arm}: seeds {_seeds}")
    sh([sys.executable, "-u", str(S / "evaluate.py"), "--exp", "C",
        "--strategies", _arm, "--ungrounded-seeds", *[str(x) for x in _seeds],
        "--tiers", "lw11", "lw21", "lw35", "--heavy"], 1.0, _tag)

# ================================================================ bundle
banner("descriptor coverage")
sh([sys.executable, "-u", str(S / "descriptor_coverage.py"), "--write"], 0.2, "coverage")

banner("descriptors + control arms COMPLETE")

# ================================================================ 1  linear probe
if (RESULTS / "probe_seen_C.json").exists():
    print("[skip] probe (already have probe_seen_C.json)", flush=True)
elif ok_to_start("probe", [], 0.8):
    banner("seen-crop linear probe A/B/C")
    sh([sys.executable, "-u", str(S / "probe_seen_all.py"), "--workers", "2"], 2.0, "probe")

# ================================================================ 2  leave-one-crop-out
if not globals().get("RUN_LOCO", True):
    print("[skip] loco (RUN_LOCO=False)", flush=True)
elif (RESULTS / "loco_s0_rich.json").exists():
    print("[skip] loco", flush=True)
elif ok_to_start("loco", [], 0.5):
    banner("leave-one-crop-out + bootstrap CIs")
    sh([sys.executable, "-u", str(S / "loco.py"), "--model", "s0", "--strategy", "rich"],
       1.0, "loco")

# ================================================================ 2b  abstention + top-5
# metrics.py writes metrics_abstain_{A,B,C}.json, which feed tab_abstain and the
# risk-coverage figure. Without this stage those stay on the previous build while every
# neighbouring table regenerates -- the exact mixture the audit flagged.
for _e in ("A", "B", "C"):
    if (RESULTS / f"metrics_abstain_{_e}.json").exists():
        print(f"[skip] metrics {_e}", flush=True); continue
    if not ok_to_start(f"metrics {_e}", [], 0.3):
        break
    banner(f"abstention + top-5, config {_e}")
    sh([sys.executable, "-u", str(S / "metrics.py"), "--exp", _e,
        "--strategies", "rich", "grounded", "--reference"], 1.0, f"metrics {_e}")

# ================================================================ 3  WiSE-FT
# workers=0 on purpose: a CUDA context and a loaded model exist before the loader is
# built, and spawning workers around that killed them outright.
if not globals().get("RUN_WISEFT", True):
    print("[skip] wiseft (RUN_WISEFT=False)", flush=True)
elif (RESULTS / "wiseft.json").exists():
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

banner("remaining tables COMPLETE -- starting the CNN sweep")

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


bundle("morning", {"llm_model": LLM_MODEL, "max_tokens": MAX_TOKENS,
                   "seeds_requested": UNGROUNDED_SEEDS, "usable_arms": USABLE,
                   "short_arm_words": SHORT_WORDS,
                   "wise_epochs": WISE_EPOCHS, "wise_lr": WISE_LR,
                   "cnn_epochs": CNN_EPOCHS, "cnn_completed": done, "cnn_not_run": skipped})
banner("MORNING DONE" if not skipped else "MORNING INCOMPLETE -- re-run to finish the CNNs")
print("", flush=True)
print("Download pde_partmorning.zip, then regenerate the paper tables:", flush=True)
print("  python docs/paper/make_tex_tables.py && python docs/paper/make_figures.py", flush=True)
