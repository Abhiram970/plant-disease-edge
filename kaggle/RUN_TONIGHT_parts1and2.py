"""
=====================================================================================
 PDE TONIGHT  --  PART 1 + PART 2 in one session
=====================================================================================
Run this tonight. Run RUN_PART3_cnns.py in the morning.

  descriptors (4 seeds x 2 arms) + short arms + integrity gate      ~1.1 h
  zero-shot A/B/C + label-corrected C                               ~1.2 h
  control arms at C  (the numbers Section 5.3 needs)                ~0.9 h
  seen-crop linear probe A/B/C                                      ~0.5 h
  leave-one-crop-out + bootstrap CIs                                ~0.3 h
  abstention + top-5 metrics A/B/C                                  ~0.6 h
  WiSE-FT alpha sweep                                               ~0.3 h
                                                            TOTAL   ~5.0 h

  Sized for a ~5.5 h remaining quota. BUDGET_H below is the guard -- set it to what
  Kaggle says you have left. This produces every table the manuscript needs EXCEPT
  tab_supervised, which is PART 3 (the 14 CNNs, ~4.8 h) after the quota resets.

SETUP
  1. Add Data -> your `pde-sage-data` dataset (the 288 px exp_data build).
  2. Settings -> Accelerator: GPU T4 x2 (or P100).  Internet: ON.
  3. Add-ons -> Secrets: LAVA_API_KEY (or ANTHROPIC_API_KEY). REQUIRED for the control arms.
  4. Paste this whole file into ONE cell, then use "Save Version -> Save & Run All" so the
     run survives you closing the browser.
  5. In the morning: download pde_tonight.zip, then run RUN_PART3_cnns.py.

ORDER MATTERS. Descriptors run first because everything downstream needs them, and they are
the only stage whose duration depends on an external API. The budget guard sits between the
halves: if generation runs long, the probe/LOCO/WiSE-FT block is skipped with a receipt
naming what is left, rather than started and abandoned half-done.

IF IT STOPS EARLY: re-run the same cell. Every stage is resumable and finished work is
skipped, so a second run continues exactly where this one stopped.
=====================================================================================
"""

# ---------------------------------------------------------------- settings
# BUDGET_H is a WALL-CLOCK guard, not a target: the run stops cleanly when it would
# otherwise overrun, and re-running resumes. Set it to the compute you actually have left
# (Kaggle shows this on the session page), NOT to 12.
BUDGET_H         = 5.2

# 4 seeds, not 8. The previous run had 3 (95% CI +/-4.8 pp); 4 nearly halves that to
# +/-3.1 pp, and 8 would reach +/-1.6 pp -- but 8 seeds costs ~4.0 h of generation plus
# control-arm evaluation, which on a 5.5 h quota would consume everything and stop right
# BEFORE the control arms, i.e. spend the whole budget and produce nothing for Section 5.3.
# The gap under test (~0.7 pp) is not resolvable at ANY seed count (that needs ~54); the
# goal is a tight, honest null. Add seeds later by re-running with more -- finished seeds
# are skipped, so seeds accumulate across sessions.
UNGROUNDED_SEEDS = [0, 1, 2, 3]
SHORT_WORDS      = 50
LLM_MODEL        = "claude-sonnet-5"
MAX_TOKENS       = 4000    # 2000 still truncated the grounded schema -> unparseable JSON
MIN_FILLED       = 48      # of 51; 3 held-out labels are not real diseases
WISE_EPOCHS      = 3
WISE_LR          = "1e-5"  # standard CLIP fine-tuning range
WISE_ALPHAS      = ["0.0", "0.5", "1.0"]

# False tonight: the two truncation-control arms cost a ~21 min embedding pass each and are
# a secondary check. Their descriptor text is still generated, so set this True in a later
# session to evaluate them without regenerating anything.
EVAL_SHORT_ARMS  = False
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

if not (RESULTS / "zeroshot_eval_C_clean.json").exists() and ok_to_start("clean", [], 0.4):
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
    if (RESULTS / f"zeroshot_eval_{_tag}.json").exists():
        print(f"[skip] {_tag}", flush=True); continue
    if not ok_to_start(_tag, [], 0.35):
        break
    banner(f"control arm {_arm}: seeds {_seeds}")
    sh([sys.executable, "-u", str(S / "evaluate.py"), "--exp", "C",
        "--strategies", _arm, "--ungrounded-seeds", *[str(x) for x in _seeds],
        "--tiers", "lw11", "lw21", "lw35", "--heavy"], 1.0, _tag)

# ================================================================ bundle
banner("descriptor coverage")
sh([sys.executable, "-u", str(S / "descriptor_coverage.py"), "--write"], 0.2, "coverage")

banner("HALFWAY: descriptors + zero-shot + control arms COMPLETE")

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

bundle("tonight", {"llm_model": LLM_MODEL, "max_tokens": MAX_TOKENS,
                   "seeds_requested": UNGROUNDED_SEEDS, "usable_arms": USABLE,
                   "short_arm_words": SHORT_WORDS,
                   "wise_epochs": WISE_EPOCHS, "wise_lr": WISE_LR})
banner("TONIGHT DONE")
print("", flush=True)
print("MORNING: run kaggle/RUN_PART3_cnns.py in a NEW notebook, and attach THIS", flush=True)
print("         notebook's output as a dataset so these results carry forward.", flush=True)
