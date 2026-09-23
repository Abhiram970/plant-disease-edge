"""
=====================================================================================
 PDE AUDIT RUN -- paste this whole file into ONE Kaggle cell and run it.
=====================================================================================
Self-contained. It needs NOTHING pushed to the repo: it clones the pinned branch as it
stands today and applies the one source change it needs in memory, on the Kaggle box.

WHAT IT RUNS
  R1  evaluate.py --clean at configurations A, B and C.
      --clean already exists on the branch, but no Kaggle stage has ever called it for
      A or B (all four call sites are --exp C). Section 6 of the paper says these runs
      "do not yet exist" and calls the scaling claim provisional until they do.

  R2  evaluate.py at A, B and C without --clean, after patching scripts/zeroshot.py to
      record per-class hits, so every model carries class_macro_acc. The macro figure
      cannot be recovered from any stored JSON -- it has to be measured.

SETUP
  Accelerator : GPU T4 x2          Internet : ON          Persistence : Files only
  Add Data    : pde-sage-data
  Add Data    : the output of your most recent run  (descriptor registry carries forward)
  Secrets     : none. No descriptors are generated here, so no API key is needed.

RESUMABLE. A configuration whose JSON already exists is skipped. If the session runs out
of time, re-run the same cell and it continues.

RESULTS land in /kaggle/working/results/. They do NOT overwrite anything in docs/paper/.
=====================================================================================
"""

# ------------------------------------------------------------------ settings
BUDGET_H = 11.0
REPO_URL = "https://github.com/Abhiram970/plant-disease-edge.git"
REPO_REF = "paper/draft-audit-2026-09-01"

EXPS      = ["A", "B", "C"]
STRATS    = ["bare", "crude", "rich", "grounded"]
TIER_ARGS = ["--tiers", "lw11", "lw21", "lw35", "--heavy", "--teachers"]
PER_EVAL_H = 1.2

# Published grounded means, unweighted over the four deployable tiers, from
# docs/paper/tex/tab_scale_study.tex. The UNCLEANED re-run must reproduce these; if it
# does not, something other than the by_class patch changed and nothing should be
# promoted into the manuscript until you know what.
PUBLISHED_GROUNDED = {"A": 21.5, "B": 22.7, "C": 23.9}
TOLERANCE_PP = 0.15

# ------------------------------------------------------------------ bootstrap
# Pull bootstrap from a throwaway shallow clone and exec its BOOTSTRAP, so this cell
# reuses the same tested logic every other stage uses: clone, locate exp_data, set
# PDE_* env vars, import prior results and descriptor arms, install deps.
import subprocess, sys, shutil
from pathlib import Path

_CODE = Path("/kaggle/working/_pde_audit_code")
if _CODE.exists():
    shutil.rmtree(_CODE, ignore_errors=True)
_rc = subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", REPO_REF,
                      REPO_URL, str(_CODE)], text=True, capture_output=True)
if _rc.returncode != 0:
    sys.exit(f"[fatal] could not clone {REPO_REF}:\n{_rc.stderr}")
# Tolerate BOTH layouts. The repository moved kaggle/ under scripts/kaggle/ and renamed
# _pde_common.py to bootstrap.py, but this cell clones a PINNED branch that may still
# carry the old tree until the move is pushed. Try the new location first, fall back to
# the old one, and say which was used so a surprising result is traceable.
_CANDIDATES = [
    (_CODE / "scripts" / "kaggle", "bootstrap"),   # current layout
    (_CODE / "kaggle", "_pde_common"),             # layout before 2026-09-11
]
BOOTSTRAP = None
for _dir, _mod in _CANDIDATES:
    if (_dir / f"{_mod}.py").exists():
        sys.path.insert(0, str(_dir))
        BOOTSTRAP = __import__(_mod).BOOTSTRAP
        print(f"[bootstrap] loaded {_mod}.py from {_dir.relative_to(_CODE)}")
        break
if BOOTSTRAP is None:
    sys.exit("[fatal] no bootstrap found in the clone: looked for "
             + " and ".join(str((d / f'{m}.py').relative_to(_CODE)) for d, m in _CANDIDATES))

exec(BOOTSTRAP)     # defines WORK, REPO, S, RESULTS, sh(), banner(), ok_to_start(), bundle()

# ------------------------------------------------------------------ preconditions
banner("AUDIT RUN -- preconditions")

# `grounded` falls back to the keyword bank for any class with no descriptor record. If
# the registry did not carry forward, every grounded column silently becomes a rich
# column and the run measures the wrong thing while looking perfectly healthy.
_desc = REPO / "descriptors"
_n_desc = len(list(_desc.rglob("*.json"))) if _desc.exists() else 0
print(f"[pre] descriptor registry: {_n_desc} json files")
if _n_desc < 10:
    sys.exit("[pre] ABORT: descriptor registry missing or nearly empty. Attach the output "
             "dataset from your previous run, or every `grounded` result here would "
             "silently be the keyword-bank fallback.")

# Validate the tier keys against config.py BEFORE spending an hour of GPU. evaluate.py
# builds its model list with `[... for t in args.tiers if t in C.MODEL_TIERS]`, so a
# misspelled tier is silently dropped: the 2026-09-10 run passed "small mid large",
# matched zero tiers, and evaluated only --heavy and --teachers. Every JSON was written
# and looked healthy while holding 2 encoders instead of 5.
sys.path.insert(0, str(S))
import config as _C                                          # noqa: E402
_want = [t for t in TIER_ARGS if not t.startswith("--")]
_bad = [t for t in _want if t not in _C.MODEL_TIERS]
if _bad:
    sys.exit(f"[pre] ABORT: tier key(s) {_bad} are not in C.MODEL_TIERS "
             f"{sorted(_C.MODEL_TIERS)}. evaluate.py would silently drop them and "
             f"evaluate only --heavy/--teachers.")
print(f"[pre] tiers resolve: {_want} + heavy + reference = {len(_want) + 2} encoders")

# ------------------------------------------------------------------ in-memory patch
# Adds per-class hits + macro accuracies to the clone on this box. Exact-string, asserted:
# if the branch has moved, this stops rather than half-patching. (Once the local change is
# pushed, this block reports "already present" and does nothing.)
banner("patch zeroshot.py for per-class metrics")
_zs_path = S / "zeroshot.py"
_zs = _zs_path.read_text(encoding="utf-8")

_OLD_ACC = '''    pred = (img_emb.to(device) @ protos.T).argmax(1).cpu().tolist()
    per = defaultdict(lambda: [0, 0]); ok = tot = 0
    for p, gt in zip(pred, labels):
        hit = classes[p] == gt
        ok += hit; tot += 1
        cr = gt.split("|")[0]
        per[cr][0] += hit; per[cr][1] += 1
    return ok / tot, {c: a / n for c, (a, n) in per.items()}'''

_NEW_ACC = '''    pred = (img_emb.to(device) @ protos.T).argmax(1).cpu().tolist()
    per = defaultdict(lambda: [0, 0])
    per_class = defaultdict(lambda: [0, 0])
    ok = tot = 0
    for p, gt in zip(pred, labels):
        hit = classes[p] == gt
        ok += hit; tot += 1
        cr = gt.split("|")[0]
        per[cr][0] += hit; per[cr][1] += 1
        per_class[gt][0] += hit; per_class[gt][1] += 1
    by_crop = {c: a / n for c, (a, n) in per.items()}
    by_class = {c: a / n for c, (a, n) in per_class.items()}
    return ok / tot, by_crop, by_class'''

_OLD_EVAL = '''    acc, by_crop = zeroshot_accuracy(img_emb, protos, labels, classes, device)
    return {"img_params_M": round(img_params_m, 2), "acc": acc, "by_crop": by_crop}, (img_emb, labels)'''

_NEW_EVAL = '''    acc, by_crop, by_class = zeroshot_accuracy(img_emb, protos, labels, classes, device)
    crop_macro = sum(by_crop.values()) / len(by_crop) if by_crop else 0.0
    class_macro = sum(by_class.values()) / len(by_class) if by_class else 0.0
    return {
        "img_params_M": round(img_params_m, 2),
        "acc": acc,
        "crop_macro_acc": crop_macro,
        "class_macro_acc": class_macro,
        "by_crop": by_crop,
        "by_class": by_class,
    }, (img_emb, labels)'''

if "class_macro_acc" in _zs:
    print("[patch] clone already carries class_macro_acc -- nothing to do")
else:
    for _old, _new, _tag in ((_OLD_ACC, _NEW_ACC, "zeroshot_accuracy"),
                             (_OLD_EVAL, _NEW_EVAL, "evaluate")):
        if _zs.count(_old) != 1:
            sys.exit(f"[patch] ABORT: could not find exactly one {_tag} anchor "
                     f"(found {_zs.count(_old)}). The branch has moved; re-derive the patch "
                     f"instead of running a half-patched file.")
        _zs = _zs.replace(_old, _new, 1)
    _zs_path.write_text(_zs, encoding="utf-8")
    print("[patch] zeroshot.py now records by_class / class_macro_acc")

# ------------------------------------------------------------------ the runs
import json
EXPECTED_MODELS = 5     # 4 deployable tiers + the SigLIP2 reference ceiling


def _n_models(path):
    """How many encoders a result file actually contains, 0 if unreadable."""
    try:
        return len(json.loads(path.read_text(encoding="utf-8")).get("models", {}))
    except Exception:
        return 0


def _eval(exp, clean):
    suffix = "_clean" if clean else ""
    out = RESULTS / f"zeroshot_eval_{exp}{suffix}.json"
    tag = f"{'clean' if clean else 'plain'} {exp}"
    # Skip on COMPLETENESS, not mere existence. The 2026-09-10 run passed
    # `--tiers small mid large`, which are not keys of C.MODEL_TIERS ("lw11"/"lw21"/
    # "lw35"); evaluate.py filters unknown names with `if t in C.MODEL_TIERS` and
    # silently evaluated nothing but --heavy and --teachers. Every file was written,
    # looked fine, and held 2 encoders instead of 5. An existence check would now skip
    # those forever, so a short file is re-run instead.
    if out.exists():
        have = _n_models(out)
        if have >= EXPECTED_MODELS:
            print(f"[skip] {tag}: {out.name} complete ({have} encoders)")
            return
        print(f"[redo] {tag}: {out.name} has only {have}/{EXPECTED_MODELS} encoders "
              f"-- re-running")
    cmd = [sys.executable, "-u", str(S / "evaluate.py"), "--exp", exp,
           "--strategies", *STRATS, *TIER_ARGS]
    if clean:
        cmd.append("--clean")
    banner(f"evaluate {tag}")
    rc, _ = sh(cmd, need_h=PER_EVAL_H, tag=tag)
    if rc != 0:
        print(f"[warn] {tag} exited {rc}", flush=True)


banner("R1 -- de-duplicated evaluation at A, B and C")
for _e in EXPS:
    if not ok_to_start(f"clean {_e}", left_h(), PER_EVAL_H):
        break
    _eval(_e, clean=True)

banner("R2 -- as-published sweep, re-measured with per-class metrics")
for _e in EXPS:
    if not ok_to_start(f"plain {_e}", left_h(), PER_EVAL_H):
        break
    _eval(_e, clean=False)

# ------------------------------------------------------------------ verdict
banner("verdict")
import json


def _gm(path):
    """Unweighted grounded mean over the four DEPLOYABLE tiers (SigLIP2 excluded)."""
    if not path.exists():
        return None
    j = json.loads(path.read_text(encoding="utf-8"))
    v = [d["grounded"]["acc"] for m, d in j["models"].items() if "SigLIP" not in m]
    return 100.0 * sum(v) / len(v) if v else None


print("\nR2 regression check -- micro means must still match the published tables:")
_drift = []
for _e in EXPS:
    got, want = _gm(RESULTS / f"zeroshot_eval_{_e}.json"), PUBLISHED_GROUNDED[_e]
    if got is None:
        print(f"  {_e}: not produced")
        continue
    d = got - want
    print(f"  {_e}: {got:.2f}%   published {want}%   delta {d:+.2f}   "
          + ("ok" if abs(d) <= TOLERANCE_PP else "DRIFT"))
    if abs(d) > TOLERANCE_PP:
        _drift.append(_e)
if _drift:
    print(f"\n  !! {', '.join(_drift)} moved by more than {TOLERANCE_PP} points.")
    print("  !! Do NOT promote these files into docs/paper/ until you know why.")

print("\nR1 -- does the scaling claim survive de-duplication?")
_c = [(e, _gm(RESULTS / f"zeroshot_eval_{e}_clean.json")) for e in EXPS]
for e, v in _c:
    print(f"  {e} cleaned: " + (f"{v:.2f}%" if v is not None else "not produced"))
_v = [v for _, v in _c if v is not None]
if len(_v) == 3:
    if _v[0] <= _v[1] <= _v[2]:
        print("  -> grounded STILL rises monotonically after cleaning.")
        print("  -> Claim survives: 'provisional' can come out of the abstract,")
        print("     the contribution list and the conclusion.")
    else:
        print("  -> grounded does NOT rise monotonically after cleaning.")
        print("  -> The uncleaned trend was falling duplicate-label density, not the")
        print("     method. Withdraw the scaling claim. That is a real finding.")
    print(f"  -> cleaned span across A..C: {max(_v) - min(_v):+.2f} points")
    print("  -> Weigh that against the 2.4-point seed-to-seed SD from the control arm")
    print("     before calling any of it a trend.")
else:
    print("  -> incomplete; re-run this cell to finish the remaining configurations.")

bundle("audit")
print("\nDone. Output -> Create Dataset, then attach it to your next run.")
