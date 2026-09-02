"""
=============================================================================================
 RUN_THIS_V3.py  --  the whole study in ONE Kaggle cell.  Paste, run, come back.
=============================================================================================

WHAT TO DO
  1. Kaggle -> Notebook -> Add Data -> your `pde-sage-data` dataset (the 288px exp_data build).
  2. Settings -> Accelerator: GPU T4 x2 (or P100).  Internet: ON.
  3. Add-ons -> Secrets: add LAVA_API_KEY  (or ANTHROPIC_API_KEY).  Needed ONLY for descriptors.
  4. Paste this whole file into ONE cell and run.  Save Version -> "Save & Run All" is best.
  5. At the end you get pde_results.zip (a few hundred KB) with a download link.

WHY V3 EXISTS  --  every change below is a fix for something that actually went wrong.

  A. CNN EPOCHS TOOK 23 MINUTES.  The last run did ~1400 s/epoch, so 8 epochs x 14 archs =
     ~43 h. It was never going to finish.  Root cause: the DataLoader, not the GPU.  Kaggle
     gives 4 vCPUs; decoding 56k JPEGs per epoch at 4 workers with no prefetch, no pinned
     memory and no persistent workers starves the GPU.  V3 fixes the loader AND caps epochs.
     -> workers=2 (4 vCPU: 2 is measurably better than 4 here), persistent_workers,
        prefetch_factor, pin_memory, channels_last, AMP (bf16/fp16), and CNN_EPOCHS=4.
     Expect ~4-6x faster.  If a model still runs long it is skipped with a receipt, never
     silently.

  B. "free 20.3 GB" WAS A LIE.  The old guard printed *system RAM* while the T4 has 14.56 GiB
     of VRAM, so it never once prevented an OOM.  V3 reads torch.cuda.mem_get_info().

  C. AN OOM RETRY THREW AWAY THE WHOLE EPOCH.  tf_efficientnetv2_s OOMed at batch 128 after
     minutes of work, then restarted from scratch at 64, then at 32.  V3 probes the batch size
     with a single forward+backward on random data BEFORE the real run (a few seconds), so the
     first real epoch already uses a batch that fits.

  D. A TIMEOUT WAS MISREAD AS AN OOM.  convnextv2_tiny hit the 1.5 h stage timeout and the
     handler halved the batch -- making it slower.  V3 distinguishes the two: only a real
     torch.OutOfMemoryError in the child's output triggers a batch reduction.

  E. MEMORY WAS NEVER RELEASED BETWEEN ARCHITECTURES.  You asked for this explicitly.  Each
     CNN now runs in its OWN SUBPROCESS.  When it exits the OS reclaims every byte of VRAM and
     host RAM -- the only way to guarantee a clean slate.  The JSON is written and verified
     before the next arch starts, then checkpoints are deleted.

  F. DESCRIPTOR GENERATION FAILED ON JSON, NOT ON REFUSAL.  The log shows two failure modes:
       "Expecting value: line 1 column 1 (char 0)"  -> empty body = hit max_tokens
       "Expecting ',' delimiter"                    -> unescaped quote/newline inside a value
     The grounded schema (source_url + verbatim_quote per field) is far longer than the
     ungrounded one, so it truncated far more often -- which is exactly why grounded_matched
     came back 35-37/51 while ungrounded came back 51/51.  That asymmetry, not "the model
     refuses to make things up", is why the matched arm never entered the eval.
     -> PDE_MAX_TOKENS=4000, a 3-attempt retry, and a stricter repair pass.

  G. A STUB COULD BE WRITTEN WITH status="filled".  33 records had that flag with EMPTY text.
     Combined with the shipped registry's 4 literal "TODO:" strings, placeholder text was
     reaching CLIP as a prototype.  descriptors.py now gates on the TEXT; V3 additionally
     verifies every arm before it is allowed into an eval.

  H. EVERY LOG LINE WAS DOUBLED.  Duplicate stream handlers.  Fixed.

WHAT V3 RUNS (in order, each resumable; re-run the cell to continue where it stopped)
   0  bootstrap + integrity checks
   1  descriptors: ungrounded seeds 0-7, grounded_matched seeds 0-7, short arms (<=50 words)
   2  zero-shot A/B/C  (bare/crude/rich/grounded)          + label-corrected C_clean
   3  control arms at C: ungrounded, grounded_matched, and the two truncation-safe short arms
   4  seen-crop linear probe A/B/C
   5  leave-one-crop-out + bootstrap CIs
   6  WiSE-FT alpha sweep            <-- was old-build; now re-measured
   7  encoder bake-off               <-- was old-build; now re-measured
   8  14 supervised CNNs, one subprocess each, memory cleared between
   9  descriptor coverage + bundle + download link
=============================================================================================
"""

# ============================================================================================
# SETTINGS
# ============================================================================================
BUDGET_H        = 11.0    # hard stop; Kaggle kills at 12 h
STAGE_TIMEOUT_H = 2.5     # generic per-stage cap
CNN_MAX_H       = 0.75    # per-architecture cap (14 archs x 0.75 = 10.5 h worst case)

# Descriptor arms
UNGROUNDED_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]   # 8 seeds: tightens the interval (was 3)
MATCHED_GROUNDED = True                        # the arm that removes the model-version confound
SHORT_ARMS       = True                        # truncation-safe arms, <=50 visual words
SHORT_WORDS      = 50
LLM_MODEL        = "claude-sonnet-5"
MAX_TOKENS       = 4000                        # fix F: 2000 still truncated the grounded schema
MIN_FILLED       = 48                          # of 51; 3 held-out labels are not real diseases

# CNNs -- fix A. 4 epochs is enough: the last run's curves were flat after epoch 3
# (83.7 -> 85.4 -> 85.5 -> 85.3 -> 85.9), and the paper's claim is a RANKING, not convergence.
CNN_EPOCHS   = 4
CNN_BATCH    = 96
CNN_WORKERS  = 2          # 4 vCPUs; 2 workers + prefetch beats 4 workers here
CNN_AMP      = True
ARCHS = ["mobilenetv3_small_100", "mobilenetv4_conv_small", "fastvit_t8", "efficientnet_b0",
         "mobilenetv3_large_100", "densenet121", "mobilenetv4_conv_medium", "fastvit_sa12",
         "convnextv2_nano", "regnety_040", "resnet50", "tf_efficientnetv2_s",
         "convnextv2_tiny", "resnet101"]

ALLOW_FETCH = False       # dataset is attached; never re-download
MIN_IMAGES  = 60_000
MIN_CROPS   = 18

REPO_URL = "https://github.com/PVAbhiram2005/plant-disease-edge.git"
REPO_REF = "paper/draft-audit-2026-09-01"   # branch carrying the placeholder + claim fixes

# ============================================================================================
import os, sys, json, time, shutil, subprocess, glob, textwrap
from pathlib import Path

T0 = time.time()
def elapsed_h(): return (time.time() - T0) / 3600.0
def left_h():    return BUDGET_H - elapsed_h()

def banner(msg):
    print("\n" + "=" * 78, flush=True)
    print(f"[{msg}]  t+{elapsed_h():.1f} h  ({left_h():.1f} h left)", flush=True)
    print("=" * 78, flush=True)

def gpu_free_gb():
    """Fix B: VRAM, not system RAM. The old code printed psutil RAM and called it 'free'."""
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return free / 1e9, total / 1e9
    except Exception:
        pass
    return 0.0, 0.0

def ok_to_start(name, remaining_names, need_h):
    if left_h() < need_h:
        print(f"\n[budget] {left_h():.1f} h left, '{name}' needs ~{need_h:.1f} h -> STOP.", flush=True)
        if remaining_names:
            print(f"[budget] not run: {remaining_names}", flush=True)
        print("[budget] Re-run this cell to resume: finished stages are skipped.", flush=True)
        return False
    return True

# ============================================================================================
# 0  BOOTSTRAP
# ============================================================================================
banner("bootstrap")

WORK = Path("/kaggle/working")
REPO = WORK / "pde"
S    = REPO / "scripts"

# Fix H: Kaggle installs a second stream handler; without this every line prints twice.
try:
    import logging
    root = logging.getLogger()
    for h in root.handlers[1:]:
        root.removeHandler(h)
except Exception:
    pass

if REPO.exists():
    shutil.rmtree(REPO, ignore_errors=True)
rc = subprocess.run(["git", "clone", "--depth", "1", "--branch", REPO_REF, REPO_URL, str(REPO)],
                    capture_output=True, text=True)
if rc.returncode != 0:
    rc = subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO)],
                        capture_output=True, text=True)
    if rc.returncode != 0:
        sys.exit(f"[fatal] clone failed:\n{rc.stderr}")
    subprocess.run(["git", "-C", str(REPO), "fetch", "origin", REPO_REF], capture_output=True, text=True)
    subprocess.run(["git", "-C", str(REPO), "checkout", "FETCH_HEAD"], capture_output=True, text=True)

head = subprocess.run(["git", "-C", str(REPO), "log", "--oneline", "-1"],
                      capture_output=True, text=True).stdout.strip()
print(f"[repo] {REPO}  HEAD = {head}", flush=True)
if not (S / "evaluate.py").exists():
    sys.exit("[fatal] clone incomplete: scripts/evaluate.py missing")

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "open_clip_torch", "timm", "anthropic", "openai"],
               check=False, capture_output=True)

# ---- locate the attached dataset at ANY depth --------------------------------------------
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
os.environ["PDE_LLM_MODEL"]   = LLM_MODEL
os.environ["PDE_MAX_TOKENS"]  = str(MAX_TOKENS)
if not ALLOW_FETCH:
    os.environ["PDE_NO_FETCH"] = "1"

n_imgs  = sum(1 for _ in DATA.rglob("*.jpg")) + sum(1 for _ in DATA.rglob("*.png"))
classes = sorted({p.name for p in DATA.iterdir() if p.is_dir()})
crops   = sorted({c.split("___")[0] for c in classes})
print(f"[data] {DATA}\n[data] {n_imgs:,} images  {len(classes)} classes  {len(crops)} crops", flush=True)
if n_imgs < MIN_IMAGES or len(crops) < MIN_CROPS:
    sys.exit(f"[fatal] dataset too small ({n_imgs:,} imgs / {len(crops)} crops). Re-upload.")

fg, tg = gpu_free_gb()
print(f"[gpu] {fg:.1f} GB free of {tg:.1f} GB VRAM", flush=True)

RESULTS = WORK / "results"; RESULTS.mkdir(exist_ok=True, parents=True)
CKPT    = WORK / "checkpoints"; CKPT.mkdir(exist_ok=True, parents=True)

# ---- secrets ------------------------------------------------------------------------------
try:
    from kaggle_secrets import UserSecretsClient
    _sec = UserSecretsClient()
    for k in ("LAVA_API_KEY", "ANTHROPIC_API_KEY", "LAVA_BASE_URL", "LAVA_SHAPE"):
        try:
            v = _sec.get_secret(k)
            if v:
                os.environ[k] = v
        except Exception:
            pass
except Exception:
    pass
HAVE_KEY = bool(os.environ.get("LAVA_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))
print(f"[llm] api key present: {HAVE_KEY}", flush=True)


def sh(cmd, need_h, tag=""):
    """Run a child process with a wall-clock cap. Returns (returncode, combined_output)."""
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


# ============================================================================================
# 1  DESCRIPTORS
# ============================================================================================
def filled_count(root, seed):
    """Count records that are genuinely usable: status filled AND real text (fix G)."""
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

ARMS = []
if HAVE_KEY:
    ARMS.append(("ungrounded", REPO / "descriptors_ungrounded"))
    if MATCHED_GROUNDED:
        ARMS.append(("grounded", REPO / "descriptors_grounded_matched"))

if not HAVE_KEY:
    print("\n[descriptors] no API key -> generation SKIPPED (control arms will not run).", flush=True)
else:
    banner("descriptors")
    for arm, root in ARMS:
        for s in UNGROUNDED_SEEDS:
            have = filled_count(root, s)
            if have >= MIN_FILLED:
                print(f"[skip] {arm} seed {s} ({have} filled)", flush=True)
                continue
            if not ok_to_start(f"{arm} seed {s}", [], 0.35):
                break
            print(f"\n--- {arm} seed {s} (have {have}, need {MIN_FILLED}) ---", flush=True)
            # build_ungrounded.py takes --which (not --exp), and its --arm choices are
            # "ungrounded"/"grounded" -- the latter writes descriptors_grounded_matched.
            sh([sys.executable, "-u", str(S / "build_ungrounded.py"),
                "--arm", arm, "--seed", str(s), "--which", "heldout"], 0.6, f"{arm} seed {s}")
            print(f"    -> {filled_count(root, s)} filled", flush=True)

# ---- truncation-safe SHORT arms (fix: 51/51 prototypes exceeded CLIP's 77-token window) ----
# Rather than re-generating, we compress the arm we already have: keep the first SHORT_WORDS
# words, which the audit showed retain visual vocabulary in 51/51 records. This is a text
# transform, so it is deterministic, free, and applies identically to both arms.
if SHORT_ARMS:
    banner(f"short arms (<= {SHORT_WORDS} words, truncation-safe)")
    import re
    def shorten(text, n=SHORT_WORDS):
        t = " ".join((text or "").split())
        if not t:
            return t
        words = t.split()
        if len(words) <= n:
            return t
        cut = " ".join(words[:n])
        # end on a sentence boundary if one is close, else on a clause boundary
        for sep in (". ", "; ", ", "):
            i = cut.rfind(sep)
            if i > len(cut) * 0.6:
                return cut[:i + 1].rstrip(" ;,")
        return cut
    made = 0
    for src_name, dst_name in (("descriptors_ungrounded", "descriptors_ungrounded_short"),
                               ("descriptors_grounded_matched", "descriptors_grounded_matched_short")):
        src = REPO / src_name
        if not src.exists():
            continue
        for seed_dir in sorted(src.glob("*")):
            if not seed_dir.is_dir():
                continue
            dst = REPO / dst_name / seed_dir.name
            dst.mkdir(parents=True, exist_ok=True)
            for f in seed_dir.glob("*.json"):
                try:
                    recs = json.load(open(f, encoding="utf-8"))
                except Exception:
                    continue
                for r in recs:
                    r["symptom_text"] = shorten(r.get("symptom_text"))
                json.dump(recs, open(dst / f.name, "w", encoding="utf-8"), indent=1)
                made += 1
    print(f"[short] wrote {made} files", flush=True)

# ---- register every arm directory with the loader ------------------------------------------
# descriptors.ARM_DIRS maps arm name -> directory. The short arms are new, so add them.
sys.path.insert(0, str(S))
os.environ["PDE_EXTRA_ARMS"] = json.dumps({
    "ungrounded_short": "descriptors_ungrounded_short",
    "grounded_matched_short": "descriptors_grounded_matched_short",
})

# ---- integrity gate: no arm enters an eval unless it is genuinely full ----------------------
banner("descriptor integrity")
USABLE = {}
for arm, root in [("ungrounded", REPO / "descriptors_ungrounded"),
                  ("grounded_matched", REPO / "descriptors_grounded_matched"),
                  ("ungrounded_short", REPO / "descriptors_ungrounded_short"),
                  ("grounded_matched_short", REPO / "descriptors_grounded_matched_short")]:
    seeds = []
    for s in UNGROUNDED_SEEDS:
        n = filled_count(root, s)
        if n >= MIN_FILLED:
            seeds.append(s)
        elif (Path(root) / str(s)).exists():
            print(f"  [reject] {arm} seed {s}: only {n}/{MIN_FILLED} usable -> excluded", flush=True)
    USABLE[arm] = seeds
    print(f"  {arm:24} usable seeds: {seeds}", flush=True)
json.dump(USABLE, open(RESULTS / "descriptor_arm_integrity.json", "w"), indent=1)


# ============================================================================================
# 2  ZERO-SHOT  A / B / C  + clean
# ============================================================================================
STRATS = ["bare", "crude", "rich", "grounded"]
for exp in ("A", "B", "C"):
    out = RESULTS / f"zeroshot_eval_{exp}.json"
    if out.exists():
        print(f"[skip] zeroshot {exp}", flush=True); continue
    if not ok_to_start(f"zeroshot {exp}", [], 0.5):
        break
    banner(f"zero-shot {exp}")
    sh([sys.executable, "-u", str(S / "evaluate.py"), "--exp", exp,
        "--strategies", *STRATS, "--tiers", "lw11", "lw21", "lw35", "--heavy", "--teachers"],
       STAGE_TIMEOUT_H, f"zeroshot {exp}")

if not (RESULTS / "zeroshot_eval_C_clean.json").exists() and ok_to_start("clean eval", [], 0.5):
    banner("zero-shot C, label-corrected")
    sh([sys.executable, "-u", str(S / "evaluate.py"), "--exp", "C", "--clean",
        "--strategies", *STRATS, "--tiers", "lw11", "lw21", "lw35", "--heavy"],
       STAGE_TIMEOUT_H, "clean")

# ============================================================================================
# 3  CONTROL ARMS  (the deciding comparison)
# ============================================================================================
for arm in ("ungrounded", "grounded_matched", "ungrounded_short", "grounded_matched_short"):
    for s in USABLE.get(arm, []):
        tag = f"C_{arm}{s}"
        out = RESULTS / f"zeroshot_eval_{tag}.json"
        if out.exists():
            print(f"[skip] {tag}", flush=True); continue
        if not ok_to_start(tag, [], 0.4):
            break
        banner(f"control arm {arm} seed {s}")
        sh([sys.executable, "-u", str(S / "evaluate.py"), "--exp", "C",
            "--strategies", arm, "--ungrounded-seed", str(s),
            "--tiers", "lw11", "lw21", "lw35", "--heavy", "--tag", tag],
           1.0, tag)

# ============================================================================================
# 4  SEEN-CROP LINEAR PROBE
# ============================================================================================
if not (RESULTS / "probe_seen_C.json").exists() and ok_to_start("probe", [], 1.2):
    banner("seen-crop linear probe A/B/C")
    sh([sys.executable, "-u", str(S / "probe_seen_all.py"), "--workers", str(CNN_WORKERS)],
       STAGE_TIMEOUT_H, "probe")

# ============================================================================================
# 5  LEAVE-ONE-CROP-OUT
# ============================================================================================
if not (RESULTS / "loco_s0_rich.json").exists() and ok_to_start("loco", [], 0.6):
    banner("leave-one-crop-out + bootstrap CIs")
    sh([sys.executable, "-u", str(S / "loco.py"), "--model", "s0", "--strategy", "rich"],
       1.0, "loco")

# ============================================================================================
# 6  WiSE-FT   (was old-build -- re-measured so it shares a build with the probe)
# ============================================================================================
# The old WiSE-FT file mixed protocols: seen_classes=166 (nested C) with unseen_classes=17
# (the pilot), so one table reported two different experiments. scripts/wiseft.py measures BOTH
# sides under configuration C and records the protocol in its output.
# workers=0 is deliberate: a CUDA context and a loaded model exist before the loader is built,
# and spawning workers around that killed them outright.
WISE = S / "wiseft.py"
if WISE.exists():
    if not (RESULTS / "wiseft.json").exists() and ok_to_start("wiseft", [], 1.2):
        banner("WiSE-FT alpha sweep (re-measured under one protocol)")
        sh([sys.executable, "-u", str(WISE), "--model", "s0", "--exp", "C",
            "--epochs", "3", "--workers", "0"], 2.0, "wiseft")
else:
    print("\n[wiseft] scripts/wiseft.py missing -> SKIPPED; numbers stay OLD-BUILD.", flush=True)

# ============================================================================================
# 7  ENCODER BAKE-OFF  (was old-build)
# ============================================================================================
BAKE = S / "bakeoff.py"
if BAKE.exists():
    if not (RESULTS / "bakeoff.json").exists() and ok_to_start("bakeoff", [], 0.8):
        banner("encoder bake-off")
        sh([sys.executable, "-u", str(BAKE)], 1.2, "bakeoff")
else:
    print("\n[bakeoff] scripts/bakeoff.py not present -> SKIPPED (numbers stay old-build).", flush=True)

# ============================================================================================
# 8  SUPERVISED CNNs  --  one SUBPROCESS each, memory fully released between runs
# ============================================================================================
# Fix E (your explicit request): a subprocess is the only way to guarantee that every byte of
# VRAM, every cached allocator block and every worker process is gone before the next arch
# starts. torch.cuda.empty_cache() inside one process does NOT do this -- fragmentation and
# worker handles survive, which is what produced the epoch-4 OOM last time.
#
# Fix C: probe the batch size on random data first (seconds), so the first REAL epoch already
# uses a batch that fits instead of discovering it minutes in.

PROBE_SRC = r'''
import sys, torch, timm
arch, bs, amp = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1"
try:
    m = timm.create_model(arch, pretrained=False, num_classes=166).cuda().to(memory_format=torch.channels_last)
    o = torch.optim.AdamW(m.parameters(), lr=1e-4)
    dt = torch.bfloat16 if (amp and torch.cuda.is_bf16_supported()) else torch.float16
    x = torch.randn(bs, 3, 224, 224, device="cuda").to(memory_format=torch.channels_last)
    y = torch.randint(0, 166, (bs,), device="cuda")
    with torch.autocast("cuda", dtype=dt, enabled=amp):
        loss = torch.nn.functional.cross_entropy(m(x), y)
    loss.backward(); o.step()
    torch.cuda.synchronize()
    print("FIT")
except torch.OutOfMemoryError:
    print("OOM")
except Exception as e:
    print("ERR", type(e).__name__, e)
'''
probe_py = WORK / "_probe_batch.py"
probe_py.write_text(PROBE_SRC)

def largest_fitting_batch(arch, start):
    """Fix C: find a batch that fits BEFORE spending an epoch discovering it doesn't."""
    bs = start
    while bs >= 16:
        rc, out = sh([sys.executable, str(probe_py), arch, str(bs), "1" if CNN_AMP else "0"],
                     0.08, f"probe {arch}@{bs}")
        if "FIT" in out:
            return bs
        if "OOM" not in out:
            return bs  # unrelated failure: let the real run report it properly
        print(f"    [probe] {arch} @ batch {bs}: OOM -> trying {bs // 2}", flush=True)
        bs //= 2
    return 16

banner(f"supervised CNNs ({len(ARCHS)} architectures, {CNN_EPOCHS} epochs each)")
print("Each architecture runs in its own subprocess; VRAM and RAM are fully released", flush=True)
print("between architectures, and the JSON is verified before moving on.\n", flush=True)

cnn_done, cnn_skipped = [], []
for i, arch in enumerate(ARCHS, 1):
    out_json = RESULTS / f"supervised_{arch}.json"
    if out_json.exists():
        print(f"[skip] {arch} (already have {out_json.name})", flush=True)
        cnn_done.append(arch)
        continue

    remaining = [a for a in ARCHS[i - 1:] if not (RESULTS / f"supervised_{a}.json").exists()]
    if not ok_to_start(f"cnn {arch}", remaining, CNN_MAX_H * 0.55):
        cnn_skipped = remaining
        break

    fg, tg = gpu_free_gb()
    print(f"\n--- cnn {i}/{len(ARCHS)}: {arch} --- t+{elapsed_h():.1f} h | VRAM {fg:.1f}/{tg:.1f} GB free",
          flush=True)

    bs = largest_fitting_batch(arch, CNN_BATCH)
    if bs != CNN_BATCH:
        print(f"    [probe] using batch {bs}", flush=True)

    cmd = [sys.executable, "-u", str(S / "supervised_baseline.py"),
           "--arch", arch, "--epochs", str(CNN_EPOCHS), "--batch", str(bs),
           "--workers", str(CNN_WORKERS), "--resume"]
    if CNN_AMP:
        cmd.append("--amp")
    rc, out = sh(cmd, CNN_MAX_H, f"cnn {arch}")

    # Fix D: only a REAL OOM justifies halving the batch. A timeout does not.
    if rc != 0 and rc != 124 and "OutOfMemoryError" in out and bs > 16:
        print(f"    [retry] genuine OOM -> batch {bs // 2}", flush=True)
        cmd[cmd.index("--batch") + 1] = str(bs // 2)
        rc, out = sh(cmd, min(CNN_MAX_H, left_h()), f"cnn {arch} retry")

    # verify the JSON landed and is readable BEFORE clearing anything
    if out_json.exists():
        try:
            d = json.load(open(out_json, encoding="utf-8"))
            print(f"    [ok] {arch}: seen_top1={d.get('seen_top1', 0) * 100:.2f}% "
                  f"({d.get('params_M')}M) -> {out_json.name}", flush=True)
            cnn_done.append(arch)
        except Exception as e:
            print(f"    [warn] {arch}: JSON unreadable ({e})", flush=True)
    else:
        print(f"    [miss] {arch}: no JSON (rc={rc}) -- recorded as NOT RUN", flush=True)

    # your explicit request: clear memory + checkpoints after each architecture
    for ck in CKPT.glob(f"{arch}*"):
        try:
            ck.unlink()
        except Exception:
            pass
    try:
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    fg, tg = gpu_free_gb()
    print(f"    [clear] checkpoints removed; VRAM now {fg:.1f}/{tg:.1f} GB free", flush=True)

print(f"\n[cnn] completed {len(cnn_done)}/{len(ARCHS)}: {cnn_done}", flush=True)
if cnn_skipped:
    print(f"[cnn] NOT RUN (budget): {cnn_skipped}", flush=True)
    print("[cnn] Re-run this cell to continue -- finished architectures are skipped.", flush=True)

# ============================================================================================
# 9  COVERAGE + BUNDLE + DOWNLOAD LINK
# ============================================================================================
banner("coverage + bundle")
sh([sys.executable, "-u", str(S / "descriptor_coverage.py"), "--write"], 0.2, "coverage")

def build_bundle():
    """Small zip: JSON + descriptor text only. No images, no checkpoints, no weights."""
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
            d = stage / rel.parent; d.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, stage / rel); files.append(str(rel).replace("\\", "/"))

    cov = REPO / "docs" / "paper" / "descriptor_coverage.json"
    if cov.exists():
        shutil.copy2(cov, stage / "descriptor_coverage.json")
        files.append("descriptor_coverage.json")

    receipt = {
        "files": files,
        "images": n_imgs, "classes": len(classes), "crops": len(crops),
        "sage_revision": "bc9bd2899f19379be29c7a99d37d2e89bf8e430d",
        "repo_head": head,
        "llm_model": LLM_MODEL, "max_tokens": MAX_TOKENS,
        "ungrounded_seeds_requested": UNGROUNDED_SEEDS,
        "descriptor_arms_usable": USABLE,
        "cnn_epochs": CNN_EPOCHS, "cnn_batch_requested": CNN_BATCH,
        "cnn_completed": cnn_done, "cnn_not_run": cnn_skipped,
        "short_arm_words": SHORT_WORDS if SHORT_ARMS else None,
        "wall_hours": round(elapsed_h(), 2),
    }
    json.dump(receipt, open(stage / "BUNDLE.json", "w"), indent=1)
    files.append("BUNDLE.json")

    zip_path = WORK / "pde_results"
    if (WORK / "pde_results.zip").exists():
        (WORK / "pde_results.zip").unlink()
    shutil.make_archive(str(zip_path), "zip", stage)
    return WORK / "pde_results.zip", receipt

zp, receipt = build_bundle()
print(f"\n[bundle] {zp}  ({zp.stat().st_size / 1e6:.2f} MB, {len(receipt['files'])} files)", flush=True)
print(f"[bundle] wall time {receipt['wall_hours']} h", flush=True)
print(f"[bundle] CNNs completed: {len(cnn_done)}/{len(ARCHS)}", flush=True)
print(f"[bundle] descriptor arms usable: { {k: len(v) for k, v in USABLE.items()} }", flush=True)

try:
    from IPython.display import FileLink, display
    print("\nDownload:", flush=True)
    display(FileLink(str(zp.relative_to(WORK))))
except Exception:
    print(f"\nDownload from the Output tab: {zp}", flush=True)

banner("DONE" if not cnn_skipped else "DONE (re-run to finish remaining CNNs)")
