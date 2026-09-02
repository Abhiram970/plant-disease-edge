"""
Generate the three standalone Kaggle runners from the shared bootstrap.

Each PART file must be a single paste-and-go cell, so the bootstrap is INLINED rather than
imported. Keeping the bootstrap in one place (_pde_common.py) and generating the parts from it
means a fix lands in all three at once -- the three files cannot drift.

    python kaggle/build_parts.py
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _pde_common import BOOTSTRAP

COMMON_SETTINGS = '''REPO_URL = "https://github.com/PVAbhiram2005/plant-disease-edge.git"
REPO_REF = "paper/draft-audit-2026-09-01"
'''

# =====================================================================================
PART1 = '''"""
=====================================================================================
 PDE PART 1 of 3  --  DESCRIPTORS + ZERO-SHOT + THE DECIDING CONTROL ARMS
=====================================================================================
Run this FIRST. It produces the numbers Section 5.3 is waiting on.

SETUP
  1. Add Data -> your `pde-sage-data` dataset (the 288 px exp_data build).
  2. Settings -> Accelerator: GPU T4 x2 (or P100).  Internet: ON.
  3. Add-ons -> Secrets: LAVA_API_KEY (or ANTHROPIC_API_KEY). REQUIRED for this part.
  4. Paste this whole file into ONE cell, run, then "Save Version -> Save & Run All".
  5. Download pde_part1.zip at the end.

WHAT IT DOES                                                        est. time
  1  descriptors: ungrounded + grounded_matched, 8 seeds each         ~2.2 h
  2  truncation-safe SHORT arms (<=50 words, free text transform)     ~0.0 h
  3  integrity gate: no arm enters an eval unless genuinely full        --
  4  zero-shot A/B/C (bare/crude/rich/grounded) + label-corrected C   ~1.2 h
  5  control arms at C: both arms x 8 seeds, plus both short arms     ~1.8 h
                                                              TOTAL  ~5.2 h

WHY 8 SEEDS. The previous 3-seed run gave a 95% CI of +/-4.8 pp on the ungrounded mean,
which is far wider than the ~0.7 pp gap being examined. 8 seeds tightens that to +/-1.6 pp.
No seed count can RESOLVE a 0.7 pp gap (that would need ~54); the goal is a tight, honest
NULL, not a resolved difference.

IF IT STOPS EARLY: re-run the same cell. Every stage is resumable and finished work is
skipped, so a second run continues exactly where this one stopped.
=====================================================================================
"""

# ---------------------------------------------------------------- settings
BUDGET_H         = 11.0
UNGROUNDED_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
SHORT_WORDS      = 50
LLM_MODEL        = "claude-sonnet-5"
MAX_TOKENS       = 4000    # 2000 still truncated the grounded schema -> unparseable JSON
MIN_FILLED       = 48      # of 51; 3 held-out labels are not real diseases
''' + COMMON_SETTINGS + '''
''' + BOOTSTRAP + '''
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
            if not ok_to_start(f"{arm} seed {s}", [], 0.3):
                break
            print(f"\\n--- {arm} seed {s} (have {have}, need {MIN_FILLED}) ---", flush=True)
            rc, _ = sh([sys.executable, "-u", str(S / "build_ungrounded.py"),
                        "--arm", arm, "--seed", str(s), "--which", "heldout"],
                       0.6, f"{arm} seed {s}")
            print(f"    -> {filled_count(root, s)} filled", flush=True)
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
for _arm in ("ungrounded", "grounded_matched", "ungrounded_short", "grounded_matched_short"):
    for _s in USABLE.get(_arm, []):
        _tag = f"C_{_SUF[_arm]}{_s}"
        if (RESULTS / f"zeroshot_eval_{_tag}.json").exists():
            print(f"[skip] {_tag}", flush=True); continue
        if not ok_to_start(_tag, [], 0.25):
            break
        banner(f"control arm {_arm} seed {_s}")
        sh([sys.executable, "-u", str(S / "evaluate.py"), "--exp", "C",
            "--strategies", _arm, "--ungrounded-seed", str(_s),
            "--tiers", "lw11", "lw21", "lw35", "--heavy"], 0.8, _tag)

# ================================================================ bundle
banner("coverage + bundle")
sh([sys.executable, "-u", str(S / "descriptor_coverage.py"), "--write"], 0.2, "coverage")
bundle(1, {"llm_model": LLM_MODEL, "max_tokens": MAX_TOKENS,
           "seeds_requested": UNGROUNDED_SEEDS, "usable_arms": USABLE,
           "short_arm_words": SHORT_WORDS})
banner("PART 1 DONE")
print("NEXT: attach this notebook's output as a dataset to PART 2, so the probe and", flush=True)
print("      WiSE-FT runs can reuse these results instead of recomputing them.", flush=True)
'''

# =====================================================================================
PART2 = '''"""
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
fine-tuned from the same initialisation. A smoke test at lr 1e-5 left the encoder
untrained (loss 5.08 against a random-guess loss of 5.11) and the midpoint interpolation
then landed in a degenerate region, dipping BELOW both endpoints. The script now checks
convergence, monotonicity and the alpha=0 identity, prints [WARNING] for each failure and
records them in wiseft.json. If you see warnings, raise --epochs or --lr before using the
table.
=====================================================================================
"""

# ---------------------------------------------------------------- settings
BUDGET_H    = 11.0
WISE_EPOCHS = 3
WISE_LR     = "1e-4"     # 1e-5 did not train the encoder at all in a smoke test
WISE_ALPHAS = ["0.0", "0.5", "1.0"]
''' + COMMON_SETTINGS + '''
''' + BOOTSTRAP + '''
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
    print("\\n[wiseft] scripts/wiseft.py missing -> SKIPPED; numbers stay OLD-BUILD.", flush=True)
elif ok_to_start("wiseft", [], 1.0):
    banner("WiSE-FT alpha sweep (both sides under one protocol)")
    sh([sys.executable, "-u", str(S / "wiseft.py"), "--model", "s0", "--exp", "C",
        "--epochs", str(WISE_EPOCHS), "--lr", WISE_LR, "--workers", "0",
        "--alphas", *WISE_ALPHAS], 2.5, "wiseft")
    _w = RESULTS / "wiseft.json"
    if _w.exists():
        try:
            _d = json.load(open(_w, encoding="utf-8"))
            if _d.get("warnings"):
                print("\\n[wiseft] THIS RESULT HAS WARNINGS -- do not use the table until fixed:",
                      flush=True)
                for _m in _d["warnings"]:
                    print(f"   - {_m}", flush=True)
            else:
                print("\\n[wiseft] all sanity checks passed.", flush=True)
        except Exception:
            pass

# ================================================================ bundle
bundle(2, {"wise_epochs": WISE_EPOCHS, "wise_lr": WISE_LR})
banner("PART 2 DONE")
print("NEXT: PART 3 runs the 14 supervised CNNs. Attach this output to it as well.", flush=True)
'''

# =====================================================================================
PART3 = '''"""
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
''' + COMMON_SETTINGS + '''
''' + BOOTSTRAP + '''
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
    print(f"\\n--- cnn {i}/{len(ARCHS)}: {arch} --- t+{elapsed_h():.1f} h | "
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

print(f"\\n[cnn] completed {len(done)}/{len(ARCHS)}: {done}", flush=True)
if skipped:
    print(f"[cnn] NOT RUN (budget): {skipped}", flush=True)
    print("[cnn] Re-run this cell to continue.", flush=True)

bundle(3, {"cnn_epochs": CNN_EPOCHS, "cnn_batch_requested": CNN_BATCH,
           "cnn_completed": done, "cnn_not_run": skipped})
banner("PART 3 DONE" if not skipped else "PART 3 INCOMPLETE -- re-run to finish")
'''

for name, body in (("RUN_PART1_descriptors.py", PART1),
                   ("RUN_PART2_probe_loco_wiseft.py", PART2),
                   ("RUN_PART3_cnns.py", PART3)):
    p = HERE / name
    p.write_text(body, encoding="utf-8")
    compile(body, str(p), "exec")          # fail loudly rather than shipping broken code
    print(f"wrote {p}  ({len(body.splitlines())} lines)")
