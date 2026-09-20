"""
=====================================================================================
 PDE PART 4  --  THE TWO MISSING BASELINE RUNS   (scoring only, no API key, no credits)
=====================================================================================
The 2026-09-12 audit named two runs as the only things that can settle the paper's
descriptor-authoring claim. Both are TEXT-ONLY: no new images, no training, no descriptor
generation at run time. The DCLIP and CuPL registries are already committed to the repo
(descriptors_dclip/0/, descriptors_cupl/0/), so this cell just clones and scores.

  RUN 1  bare80 -- the class name ensembled over CLIP's 80 OpenAI ImageNet templates.
         Section 6 concedes the published `bare` row uses only 3 templates, so the paper's
         largest effect (+8.8 points at configuration A) is measured against a baseline
         weaker than the standard one. `bare` is re-scored here in the SAME process on the
         SAME image-embedding cache, so the bare80-vs-bare delta cannot be contaminated by
         any environment difference.
         READ THIS ROW FIRST. If bare80 gains more than ~4 points over bare, the +2.4 gain
         at configuration B is gone and the +4.8 at C is in doubt -- and Contribution 2,
         the Discussion and the Conclusion all quote those numbers.

  RUN 2  dclip + cupl -- the established per-class descriptor-generation methods
         (Menon and Vondrick, ICLR 2023; Pratt et al., ICCV 2023). The paper is positioned
         against both and implements neither, so it currently cannot claim that SOURCE-
         grounding adds anything over ordinary LLM-written per-class text. These two rows
         decide that. Both emit text per class and cannot collide, so they also serve as a
         non-colliding comparator at A and B, not only at C where the ungrounded arm exists.

SETUP -- there is nothing to configure
  1. Add Data -> your `pde-sage-data` dataset (the 288 px exp_data build).
  2. Settings -> Accelerator: GPU T4 x2 (or P100).  Internet: ON.
  3. NO Secrets and NO API key are needed. Nothing is generated and nothing is billed.
  4. Paste this whole file into ONE cell, run, then "Save Version -> Save & Run All".
  5. Download pde_part4.zip at the end.

STAGES                                                              est. time
  1  clone + descriptor coverage gate (51/51 per method)              ~0.05 h
  2  score 6 strategies at A, B and C on all 5 encoders               ~1.2 h
  3  de-duplicated repeat of stage 2 (--clean)                        ~1.2 h
                                                              TOTAL  ~2.5 h

WHAT THIS RUN IS AND IS NOT. The registries are ONE draw, hand-generated in a chat rather
than sampled over seeds. Section 4.3 measures a 2.4-point standard deviation between
descriptor registries, so any margin below about 2.4 points here is not an effect and must
not be reported as one. To turn a direction into a measurement, regenerate the registries
into descriptors_{dclip,cupl}/1/ and /2/ and re-run with SEEDS = [0, 1, 2].

IF IT STOPS EARLY: re-run the same cell. Finished configurations are skipped.
=====================================================================================
"""

# ---------------------------------------------------------------- settings
BUDGET_H   = 11.0
SEEDS      = [0]                 # one registry per method today; add 1, 2 when they exist
MIN_FILLED = 44                  # of 51; 5 held-out labels are not real diseases
RUN_CLEAN  = True                # also score the de-duplicated label set (Section 4.4)
REPO_URL   = "https://github.com/Abhiram970/plant-disease-edge.git"
REPO_REF   = "baselines/dclip-cupl-manual"

# `bare` and `rich` and `grounded` are re-scored alongside the new rows so every comparison
# in this run shares one image-embedding cache. Comparing a new row against the PUBLISHED
# table instead would let any environment difference masquerade as a descriptor effect.
STRATS = ["bare", "bare80", "rich", "grounded", "grounded_split", "dclip", "cupl"]

import os, sys, json, time, shutil, subprocess
from pathlib import Path

# ---------------------------------------------------------------- bootstrap
T0 = time.time()
def elapsed_h(): return (time.time() - T0) / 3600.0
def left_h():    return BUDGET_H - elapsed_h()

def banner(msg):
    print("\n" + "=" * 78, flush=True)
    print(f"[{msg}]  t+{elapsed_h():.1f} h  ({left_h():.1f} h left)", flush=True)
    print("=" * 78, flush=True)

def ok_to_start(name, remaining, need_h):
    if left_h() < need_h:
        print(f"\n[budget] {left_h():.1f} h left, '{name}' needs ~{need_h:.1f} h -> STOP.", flush=True)
        if remaining:
            print(f"[budget] not run: {remaining}", flush=True)
        print("[budget] Re-run this cell to resume; finished work is skipped.", flush=True)
        return False
    return True

try:                                    # Kaggle attaches a second handler; it doubles every line
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

banner("bootstrap")
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
    _alt, _n = WORK / "pde_fresh", 1
    while _alt.exists():
        _n += 1; _alt = WORK / f"pde_fresh{_n}"
    print(f"[bootstrap] stale clone survived removal; using {_alt}", flush=True)
    REPO, S = _alt, _alt / "scripts"

_rc = subprocess.run(["git", "clone", "--depth", "1", "--branch", REPO_REF, REPO_URL, str(REPO)],
                     capture_output=True, text=True)
if _rc.returncode != 0:
    sys.exit(f"[fatal] clone of branch {REPO_REF} failed:\n{_rc.stderr}\n"
             f"Push the descriptor-baseline branch, or set REPO_REF to the branch that has it.")
print("[repo] HEAD = " + subprocess.run(["git", "-C", str(REPO), "log", "--oneline", "-1"],
                                        capture_output=True, text=True).stdout.strip(), flush=True)
for _need in ("evaluate.py", "descriptors.py"):
    if not (S / _need).exists():
        sys.exit(f"[fatal] clone incomplete: scripts/{_need} missing")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "open_clip_torch", "timm"],
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
CROPS   = sorted({p.name.split("___")[0] for p in DATA.iterdir() if p.is_dir()})
print(f"[data] {N_IMGS:,} images over {len(CROPS)} crops", flush=True)
if N_IMGS < 60_000 or len(CROPS) < 18:
    sys.exit(f"[fatal] dataset too small ({N_IMGS:,} imgs / {len(CROPS)} crops).")
try:
    import torch
    if torch.cuda.is_available():
        _f, _t = torch.cuda.mem_get_info()
        print(f"[gpu] {_f/1e9:.1f} GB free of {_t/1e9:.1f} GB VRAM", flush=True)
except Exception:
    pass


def sh(cmd):
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    rc = subprocess.run([str(c) for c in cmd], cwd=str(REPO)).returncode
    if rc != 0:
        print(f"  [warn] exit {rc}", flush=True)
    return rc


def collect():
    n = 0
    for src in (REPO / "docs" / "paper").glob("zeroshot_eval_*.json"):
        dst = RESULTS / src.name
        if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
            shutil.copy2(src, dst); n += 1
    return n


# ---------------------------------------------------------------- 1. coverage gate
# A class with no record falls through to the keyword bank, which would silently make a
# DCLIP row part-bank and produce a number that looks like a measurement of DCLIP but is
# not. That is the defect that invalidated the original `rich` column, so it is gated here
# rather than discovered afterwards.
banner("descriptor coverage gate")
sys.path.insert(0, str(S))
import config as _C            # noqa: E402
import descriptors as _D       # noqa: E402

RUN_STRATS = ["bare", "bare80", "rich", "grounded", "grounded_split"]
for method, field in (("dclip", "descriptors"), ("cupl", "sentences")):
    d = _C.REPO_ROOT / _D.BASELINE_DIRS[method] / str(SEEDS[0])
    filled = 0
    for p in (sorted(d.glob("*.json")) if d.exists() else []):
        try:
            for r in json.loads(p.read_text(encoding="utf-8")):
                if r.get("status") == "filled" and [x for x in (r.get(field) or []) if str(x).strip()]:
                    filled += 1
        except Exception as e:
            print(f"  [warn] unreadable {p.name}: {e}", flush=True)
    mark = "OK " if filled >= MIN_FILLED else "LOW"
    print(f"  [{mark}] {method:5s} seed {SEEDS[0]}: {filled} filled records (need {MIN_FILLED})", flush=True)
    if filled >= MIN_FILLED:
        RUN_STRATS.append(method)
    else:
        print(f"  [gate] {method} EXCLUDED -- it would score as part keyword-bank, not as {method}.",
              flush=True)

print(f"\n[gate] scoring: {RUN_STRATS}", flush=True)
if "dclip" not in RUN_STRATS and "cupl" not in RUN_STRATS:
    print("[gate] neither generated baseline passed; only RUN 1 (bare80) will be produced.", flush=True)

# ---------------------------------------------------------------- 2. score A / B / C
for exp in ("A", "B", "C"):
    out = REPO / "docs" / "paper" / f"zeroshot_eval_{exp}.json"
    if not ok_to_start(f"zero-shot {exp}", "later configurations", 0.5):
        break
    banner(f"zero-shot {exp}: {' + '.join(RUN_STRATS)}")
    sh([sys.executable, "-u", str(S / "evaluate.py"), "--exp", exp,
        "--strategies", *RUN_STRATS, "--heavy", "--teachers"])
    print(f"[collect] {collect()} result file(s) mirrored", flush=True)

# ---------------------------------------------------------------- 3. de-duplicated repeat
# Section 4.4: the uncleaned label set carries 5 alias pairs and 4 non-disease labels. A new
# baseline has to be read on both label sets or it is not comparable with Table 2.
if RUN_CLEAN:
    for exp in ("A", "B", "C"):
        if not ok_to_start(f"clean {exp}", "later configurations", 0.5):
            break
        banner(f"zero-shot {exp}, de-duplicated labels")
        sh([sys.executable, "-u", str(S / "evaluate.py"), "--exp", exp, "--clean",
            "--strategies", *RUN_STRATS, "--heavy", "--teachers"])
        print(f"[collect] {collect()} result file(s) mirrored", flush=True)

# ---------------------------------------------------------------- 4. summary + package
banner("summary")


def dep_mean(js, strat):
    """Mean over the four DEPLOYABLE tiers. The 92.9 M reference is excluded everywhere the
    manuscript quotes a mean, so it must be excluded here too."""
    vals = [v[strat]["acc"] * 100 for m, v in js.get("models", {}).items()
            if "SigLIP2" not in m and strat in v]
    return sum(vals) / len(vals) if vals else None


rows = []
for exp in ("A", "B", "C"):
    p = RESULTS / f"zeroshot_eval_{exp}.json"
    if not p.exists():
        continue
    js = json.loads(p.read_text(encoding="utf-8"))
    rows.append((exp, js.get("n_classes"), {s: dep_mean(js, s) for s in RUN_STRATS}))

if rows:
    hdr = "  cfg  cls " + "".join(f"{s:>10s}" for s in RUN_STRATS)
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for exp, n, d in rows:
        print(f"  {exp:3s} {str(n):>4s} " +
              "".join(f"{(f'{d[s]:.1f}' if d.get(s) is not None else '--'):>10s}" for s in RUN_STRATS))
    print("\n  THE NUMBER THAT MATTERS FIRST -- bare80 minus bare:")
    for exp, n, d in rows:
        if d.get("bare") is not None and d.get("bare80") is not None:
            delta = d["bare80"] - d["bare"]
            note = ("  <-- larger than the authoring gain at this scale; Contribution 2 needs revising"
                    if delta > 4 else "")
            print(f"    {exp}: {delta:+.2f} points{note}")
    print("\n  Any margin under ~2.4 points is inside the registry-to-registry spread of\n"
          "  Section 4.3 and must not be reported as an effect.")

for method in ("dclip", "cupl"):
    src = _C.REPO_ROOT / _D.BASELINE_DIRS[method]
    if src.exists():
        shutil.copytree(src, RESULTS / _D.BASELINE_DIRS[method], dirs_exist_ok=True)

OUT = WORK / "pde_part4"
if OUT.exists():
    shutil.rmtree(OUT, ignore_errors=True)
shutil.copytree(RESULTS, OUT)
shutil.make_archive(str(WORK / "pde_part4"), "zip", root_dir=str(OUT))
print(f"\n[done] {WORK / 'pde_part4.zip'}", flush=True)
for f in sorted(RESULTS.glob("zeroshot_eval_*.json")):
    print(f"   {f.name}  ({f.stat().st_size/1024:.0f} KB)", flush=True)

print("""
NEXT, LOCALLY
  1. unzip pde_part4.zip over docs/paper/
  2. python docs/paper/make_tables.py --write && python docs/paper/make_figures.py
  3. python scripts/verify_tables_figures.py
  4. read bare80 first: it sets the honest size of the authoring effect that Contribution 2,
     the Discussion and the Conclusion all quote.
  5. then dclip / cupl against grounded at C. If either matches grounded within 2.4 points,
     the paper cannot claim sourcing is the better authoring route and Section 5 must say so.
""", flush=True)
