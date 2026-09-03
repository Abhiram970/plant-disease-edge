"""
Generate RUN_MORNING_everything_else.py -- the second-session runner.

Run AFTER build_parts.py:
    python kaggle/build_parts.py && python kaggle/build_morning.py

Tonight's run is sized for a 4 h quota and defers whatever is not Section 5.3. This runner
picks up everything that was left, in the order that matters if the session is cut short:

    1  descriptor seeds 4-7          extends the control arm from 4 seeds to 8
    2  control arms, all 8 seeds     re-evaluated in ONE process (the file is overwritten)
    3  short arms                    the 77-token truncation control
    4  label-corrected C eval        label-noise robustness
    5  leave-one-crop-out            tab_loco
    6  WiSE-FT                       tab_wiseft
    7  the 14 supervised CNNs        tab_supervised -- last because it is the longest

Assembled from the same PART1/PART2/PART3 source strings as every other runner, so a fix to a
stage lands here too.
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _pde_common import BOOTSTRAP
import build_parts as BP

HEADER = '''"""
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
'''

TAIL = '''
# ---- regenerate the paper's tables and figures inside the run -------------------------
# make_figures.py reads zeroshot_eval_{A,B,C}.json and probe_seen_{A,B,C}.json directly, so
# it must run AFTER those land or the ten figures stay on the previous build while every
# table regenerates -- the build mixture the audit flagged. Both generators are pure
# matplotlib/json (Agg backend), so they run headless here in seconds, and the rendered
# .tex and .png are carried out in the bundle.
banner("regenerate paper tables + figures")
_docs = REPO / "docs" / "paper"
for _g in ("make_tex_tables.py", "make_figures.py"):
    if (_docs / _g).exists():
        # The generators read the JSONs sitting next to them, so stage this run's results
        # into docs/paper first.
        for _f in RESULTS.glob("*.json"):
            shutil.copy2(_f, _docs / _f.name)
        _r = subprocess.run([sys.executable, "-u", str(_docs / _g)], text=True, cwd=str(_docs),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(_r.stdout or f"[{_g}] no output", flush=True)
    else:
        print(f"[warn] {_g} not found -- regenerate locally after downloading", flush=True)

# Carry the regenerated tex/figures out with the results.
for _sub in ("tex", "figures"):
    _src = _docs / _sub
    if _src.exists():
        _dst = RESULTS / _sub
        if _dst.exists():
            shutil.rmtree(_dst, ignore_errors=True)
        shutil.copytree(_src, _dst)
        print(f"[bundle] staged docs/paper/{_sub}", flush=True)

bundle("morning", {"llm_model": LLM_MODEL, "max_tokens": MAX_TOKENS,
                   "seeds_requested": UNGROUNDED_SEEDS, "usable_arms": USABLE,
                   "short_arm_words": SHORT_WORDS,
                   "wise_epochs": WISE_EPOCHS, "wise_lr": WISE_LR,
                   "cnn_epochs": CNN_EPOCHS, "cnn_completed": done, "cnn_not_run": skipped})
banner("MORNING DONE" if not skipped else "MORNING INCOMPLETE -- re-run to finish the CNNs")
print("", flush=True)
print("Download pde_partmorning.zip, then regenerate the paper tables:", flush=True)
print("  python docs/paper/make_tex_tables.py && python docs/paper/make_figures.py", flush=True)
'''


def body_of(part_src):
    after = part_src.split(BOOTSTRAP, 1)[1]
    for marker in ("#__P1_TAIL__", "#__P2_TAIL__", "#__P3_TAIL__"):
        if marker in after:
            after = after.split(marker, 1)[0]
    return after


def main():
    p1 = body_of(BP.PART1)
    p2 = body_of(BP.PART2)
    p3 = body_of(BP.PART3)

    # PART1's own coverage banner is misleading mid-run; the bundle happens once at the end.
    p1 = p1.replace('banner("coverage + bundle")', 'banner("descriptor coverage")')

    src = (HEADER + BP.COMMON_SETTINGS + "\n" + BOOTSTRAP
           + p1
           + '\nbanner("descriptors + control arms COMPLETE")\n'
           + p2
           + '\nbanner("remaining tables COMPLETE -- starting the CNN sweep")\n'
           + p3
           + TAIL)

    out = HERE / "RUN_MORNING_everything_else.py"
    out.write_text(src, encoding="utf-8")
    compile(src, str(out), "exec")
    print(f"wrote {out}  ({len(src.splitlines())} lines)")

    for name in ("ARCHS", "CNN_EPOCHS", "WISE_LR", "USABLE", "done", "skipped"):
        if name not in src:
            print(f"  WARNING: {name} referenced but not defined")
    print("  checks: compiled, single bootstrap, CNNs last")


if __name__ == "__main__":
    main()
