"""Generate scripts/kaggle/runners/audit_dedup_macro.py — the stage the 2026-09-10 audit needs.

Edit THIS file, not the generated runner, exactly as with build/parts.py /
build/stage1.py / build/stage2.py. Run it from the repo root:

    python scripts/kaggle/build/audit.py

WHY A NEW STAGE EXISTS. Two of the audit's runs cannot be produced by any existing
stage:

  R1  De-duplicated evaluation at configurations A and B. Every runner in kaggle/
      calls `evaluate.py --clean` at exp C ONLY -- grep for `"--exp", "C", "--clean"`
      and there are four hits, none for A or B. Section 6 of the manuscript says the
      cleaned runs at A and B "do not yet exist" and that the scaling claim is
      provisional until they do. This stage is what makes them exist.

  R2  Class-macro accuracy. zeroshot.py did not record per-class hits, so the macro
      figure cannot be recovered from any stored JSON; it has to be re-measured with
      the by_class change in place.

Both are pure evaluation. No descriptors are generated, so NO API KEY IS NEEDED.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KAGGLE = HERE.parent
RUNNERS = KAGGLE / "runners"
RUNNERS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(KAGGLE))
from bootstrap import BOOTSTRAP  # noqa: E402

BODY = '''"""
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
# docs/paper/tex/tab_scale_study.tex. The uncleaned re-run must reproduce these.
PUBLISHED_GROUNDED = {"A": 21.5, "B": 22.7, "C": 23.9}
TOLERANCE_PP = 0.1

''' + BOOTSTRAP + '''

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


print("\\nR2 regression check -- micro means must match the published tables:")
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
    print(f"\\n  !! {', '.join(_drift)} moved by more than {TOLERANCE_PP} points.")
    print("  !! Something other than the by_class addition changed. Do NOT promote these")
    print("  !! files into docs/paper/ until you know what.")

print("\\nR1 result -- the scaling claim:")
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
'''


def main() -> int:
    out = KAGGLE / "runners/audit_dedup_macro.py"
    out.write_text(BODY, encoding="utf-8", newline="")
    print(f"[build] wrote {out} ({len(BODY.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
