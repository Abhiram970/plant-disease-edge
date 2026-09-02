"""
Generate RUN_TONIGHT_parts1and2.py by concatenating the PART1 and PART2 bodies.

Run AFTER build_parts.py:
    python kaggle/build_parts.py && python kaggle/build_tonight.py

The combined runner is assembled from the SAME PART1/PART2 source strings, split on the
`#__P1_TAIL__` / `#__P2_TAIL__` markers, so it cannot drift from the individual parts: fix a
stage once and every runner that contains it is regenerated.

Why a combined file at all: PART1 (~5.2 h) and PART2 (~1.6 h) together are ~6.9 h, comfortably
inside one 12 h Kaggle session, and PART2 depends on nothing PART1 does not already produce.
Running them as one session avoids re-attaching PART1's output as a dataset and re-cloning.
PART3 (the 14 CNNs, ~4.8 h) stays separate and runs in a fresh session.
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _pde_common import BOOTSTRAP
import build_parts as BP

HEADER = '''"""
=====================================================================================
 PDE TONIGHT  --  PART 1 + PART 2 in one session
=====================================================================================
Run this tonight. Run RUN_PART3_cnns.py in the morning.

  descriptors (8 seeds x 2 arms) + short arms + integrity gate      ~2.2 h
  zero-shot A/B/C + label-corrected C                               ~1.2 h
  control arms at C  (the numbers Section 5.3 needs)                ~1.8 h
  seen-crop linear probe A/B/C                                      ~0.5 h
  leave-one-crop-out + bootstrap CIs                                ~0.3 h
  WiSE-FT alpha sweep                                               ~0.8 h
                                                            TOTAL   ~6.9 h
  (12 h Kaggle limit, 11 h internal budget -> ~4 h headroom)

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
BUDGET_H         = 11.0
UNGROUNDED_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
SHORT_WORDS      = 50
LLM_MODEL        = "claude-sonnet-5"
MAX_TOKENS       = 4000    # 2000 still truncated the grounded schema -> unparseable JSON
MIN_FILLED       = 48      # of 51; 3 held-out labels are not real diseases
WISE_EPOCHS      = 3
WISE_LR          = "1e-5"  # standard CLIP fine-tuning range
WISE_ALPHAS      = ["0.0", "0.5", "1.0"]
'''

TAIL = '''
bundle("tonight", {"llm_model": LLM_MODEL, "max_tokens": MAX_TOKENS,
                   "seeds_requested": UNGROUNDED_SEEDS, "usable_arms": USABLE,
                   "short_arm_words": SHORT_WORDS,
                   "wise_epochs": WISE_EPOCHS, "wise_lr": WISE_LR})
banner("TONIGHT DONE")
print("", flush=True)
print("MORNING: run kaggle/RUN_PART3_cnns.py in a NEW notebook, and attach THIS", flush=True)
print("         notebook's output as a dataset so these results carry forward.", flush=True)
'''


def body_of(part_src):
    """The executable body of a part: everything after the inlined bootstrap, minus its own
    bundle()/banner() tail (which is replaced by the combined tail)."""
    after = part_src.split(BOOTSTRAP, 1)[1]
    for marker in ("#__P1_TAIL__", "#__P2_TAIL__"):
        if marker in after:
            after = after.split(marker, 1)[0]
    return after


def main():
    p1 = body_of(BP.PART1)
    p2 = body_of(BP.PART2)

    # PART2's settings block is folded into the header above; its body references only names
    # the header or the bootstrap already define.
    # PART1's tail banner reads "coverage + bundle", which is misleading here: the bundle
    # happens once at the very end of the combined run, not at this point.
    p1 = p1.replace('banner("coverage + bundle")', 'banner("descriptor coverage")')

    src = HEADER + BP.COMMON_SETTINGS + "\n" + BOOTSTRAP + p1 + \
        '\nbanner("HALFWAY: descriptors + zero-shot + control arms COMPLETE")\n' + p2 + TAIL

    out = HERE / "RUN_TONIGHT_parts1and2.py"
    out.write_text(src, encoding="utf-8")
    compile(src, str(out), "exec")          # fail loudly rather than shipping broken code
    print(f"wrote {out}  ({len(src.splitlines())} lines)")

    # Names PART2's body uses must be defined by the header or by PART1's body.
    import re
    for name in ("WISE_EPOCHS", "WISE_LR", "WISE_ALPHAS", "USABLE", "RESULTS", "S", "sh"):
        if name not in src:
            print(f"  WARNING: {name} referenced but not defined")
    print("  checks: compiled, settings folded in, single bootstrap")


if __name__ == "__main__":
    main()
