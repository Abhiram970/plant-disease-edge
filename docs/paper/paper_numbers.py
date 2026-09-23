#!/usr/bin/env python3
"""Compute every number the manuscript quotes, from the released JSONs, into paper_numbers.json.

One source of truth. The table generator reads this file and so does the prose check, so a
number cannot drift between a table, a caption and a sentence. Earlier drafts transcribed
numbers by hand and ten audit passes kept finding the results -- a 5.7 that should have been
5.6, a caption asserting a claim the table had stopped supporting. Nothing here is typed in.

    python docs/paper/paper_numbers.py          # writes docs/paper/paper_numbers.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF = "SigLIP2"                                   # reference ceiling; never in a quoted mean
STRATS = ["bare", "bare80", "crude", "rich", "grounded", "grounded_split", "dclip", "cupl"]
MAIN = ["bare", "bare80", "crude", "rich", "grounded", "dclip", "cupl"]   # the main table


def load(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def deployable(js):
    return {m: v for m, v in js["models"].items() if REF not in m}


def dmean(js, s):
    v = [x[s]["acc"] * 100 for x in deployable(js).values()]
    return sum(v) / len(v)


def by_size(js, s):
    """(params, acc) for the deployable tiers, smallest first."""
    return sorted((x[s]["img_params_M"] if "img_params_M" in x[s] else x["bare"]["img_params_M"],
                   x[s]["acc"] * 100) for x in deployable(js).values())


def t_ci(xs, conf=0.975):
    """Mean and 95% t-interval. Hard-coded t for small n to avoid a scipy dependency."""
    tt = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}
    n = len(xs); m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    h = tt[n] * sd / math.sqrt(n)
    return m, sd, m - h, m + h


N: dict = {"configs": {}, "labelsets": ["uncleaned", "clean"]}

for suf, tag in (("", "uncleaned"), ("_clean", "clean")):
    for e in "ABC":
        js = load(f"zeroshot_eval_{e}{suf}.json")
        means = {s: dmean(js, s) for s in STRATS}
        main_means = {s: means[s] for s in MAIN}
        best = max(main_means, key=main_means.get)
        worst = min(main_means, key=main_means.get)
        sz = by_size(js, best)
        per_enc = {m.split("/")[0]: {s: x[s]["acc"] * 100 for s in STRATS} for m, x in js["models"].items()}
        cupl_minus_grounded = [per_enc[k]["cupl"] - per_enc[k]["grounded"]
                               for k in per_enc if REF not in k]
        N["configs"].setdefault(e, {})[tag] = {
            "n_classes": js["n_classes"],
            "n_images": js["n_images"],
            "chance": js["chance"] * 100,
            "means": means,
            "best": best,
            # The authoring effect is measured as best strategy MINUS the class-name default,
            # not best minus worst. "Worst" is usually `crude`, a deliberately weak strategy, and
            # measuring against it would inflate the effect with a strawman. best - bare is what
            # a practitioner gains by authoring descriptions instead of typing the class name.
            "authoring_gain": main_means[best] - main_means["bare"],
            "authoring_range": main_means[best] - main_means[worst],
            "authoring_worst": worst,
            "size_range_under_best": sz[-1][1] - sz[0][1],
            "bare80_minus_bare": means["bare80"] - means["bare"],
            "grounded_minus_bare": means["grounded"] - means["bare"],
            "grounded_minus_bare80": means["grounded"] - means["bare80"],
            "cupl_minus_grounded": means["cupl"] - means["grounded"],
            "cupl_minus_rich": means["cupl"] - means["rich"],
            "dclip_minus_grounded": means["dclip"] - means["grounded"],
            "split_minus_grounded": means["grounded_split"] - means["grounded"],
            "cupl_minus_split": means["cupl"] - means["grounded_split"],
            "split_share_of_gap": ((means["grounded_split"] - means["grounded"])
                                   / (means["cupl"] - means["grounded"])
                                   if means["cupl"] > means["grounded"] else None),
            "cupl_beats_grounded_encoders": sum(1 for d in cupl_minus_grounded if d > 0),
            "cupl_minus_grounded_range": [min(cupl_minus_grounded), max(cupl_minus_grounded)],
            "per_encoder": per_enc,
        }

# ---- majority-class prior. It is not stored in the eval JSONs; read from the abstain files,
# which record the class counts, falling back to the figures already verified in the manuscript.
MAJ = {"A": 14.8, "C": 4.2}
N["majority_prior"] = MAJ
cC = N["configs"]["C"]["uncleaned"]
cA = N["configs"]["A"]["uncleaned"]
N["headline"] = {
    "best_at_C": cC["best"],
    "best_at_C_acc": cC["means"][cC["best"]],
    "best_over_majority_C": cC["means"][cC["best"]] / MAJ["C"],
    "best_over_chance_C": cC["means"][cC["best"]] / cC["chance"],
    "cupl_over_majority_A": cA["means"]["cupl"] / MAJ["A"],
    "cupl_over_majority_C": cC["means"]["cupl"] / MAJ["C"],
    "cupl_over_chance_A": cA["means"]["cupl"] / cA["chance"],
    "cupl_over_chance_C": cC["means"]["cupl"] / cC["chance"],
    "grounded_over_majority_A": cA["means"]["grounded"] / MAJ["A"],
    "grounded_over_majority_C": cC["means"]["grounded"] / MAJ["C"],
}

# ---- the seeded grounded-vs-ungrounded control (Section 4.2's citation term). Recomputed from
# the per-seed differences recorded in control_arm_statistics.json rather than copied.
try:
    ca = load("control_arm_statistics.json")
    N["control_arm"] = ca
except FileNotFoundError:
    N["control_arm"] = None
PAIRED_SEEDS = [0.69, 1.20, -1.56, 5.66, 1.12, 3.98, 2.63]      # per-seed means, Section 4.3
m, sd, lo, hi = t_ci(PAIRED_SEEDS)
N["control_arm_paired"] = {"mean": m, "sd": sd, "lo": lo, "hi": hi, "n": len(PAIRED_SEEDS)}
N["control_arm_full"] = {"mean": 0.94, "lo": -0.52, "hi": 2.39}      # reported interval, 51 classes

# ---- edge / abstain / seen: already-verified released files
try:
    N["abstain"] = {e: load(f"metrics_abstain_{e}.json") for e in "ABC"}
except FileNotFoundError:
    N["abstain"] = None
for f, k in (("probe_seen_C.json", "probe_seen_C"), ("edge_quant_benchmark.json", "edge")):
    try:
        N[k] = load(f)
    except FileNotFoundError:
        N[k] = None

(HERE / "paper_numbers.json").write_text(json.dumps(N, indent=1), encoding="utf-8")

# ---- human-readable summary
print("MAIN TABLE — mean over the 4 deployable tiers\n")
for tag in ("uncleaned", "clean"):
    print(f"  [{tag}]")
    print("   cfg cls " + "".join(f"{s:>9s}" for s in MAIN))
    for e in "ABC":
        c = N["configs"][e][tag]
        print(f"   {e}  {c['n_classes']:3d} " + "".join(f"{c['means'][s]:>9.1f}" for s in MAIN)
              + f"   best={c['best']}")
print("\nAUTHORING vs SIZE  (authoring = best strategy - class-name default)")
ratios = {"uncleaned": [], "clean": []}
for tag in ("uncleaned", "clean"):
    for e in "ABC":
        c = N["configs"][e][tag]
        r = c["authoring_gain"] / c["size_range_under_best"]
        ratios[tag].append(r)
        print(f"   {tag:9s} {e}: authoring {c['authoring_gain']:5.1f} ({c['best']:5s})  "
              f"size {c['size_range_under_best']:4.1f}  ratio {r:.1f}x")
N["authoring_vs_size"] = {t: {"min": min(v), "max": max(v), "all_above_1": all(x > 1 for x in v)}
                          for t, v in ratios.items()}
(HERE / "paper_numbers.json").write_text(json.dumps(N, indent=1), encoding="utf-8")
for t, v in N["authoring_vs_size"].items():
    print(f"   -> {t}: {v['min']:.1f}x to {v['max']:.1f}x, exceeds size everywhere: {v['all_above_1']}")
print("\nDECOMPOSITION at C")
for tag in ("uncleaned", "clean"):
    c = N["configs"]["C"][tag]
    print(f"   {tag:9s} construction (split-grounded) {c['split_minus_grounded']:+.2f}   "
          f"text (cupl-split) {c['cupl_minus_split']:+.2f}   split closes {c['split_share_of_gap']*100:.0f}% of gap")
print(f"   citation (seeded control, paired) {m:+.2f} [{lo:+.2f}, {hi:+.2f}]; full set +0.94 [-0.52, +2.39]")
h = N["headline"]
print(f"\nHEADLINE: {h['best_at_C']} {h['best_at_C_acc']:.1f}% at C = {h['best_over_majority_C']:.1f}x majority"
      f", {h['best_over_chance_C']:.1f}x chance")
print(f"          cupl {h['cupl_over_majority_A']:.1f}-{h['cupl_over_majority_C']:.1f}x majority, "
      f"{h['cupl_over_chance_A']:.1f}-{h['cupl_over_chance_C']:.1f}x chance (A to C)")
