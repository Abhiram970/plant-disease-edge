#!/usr/bin/env python3
"""Run the experiments the 2026-09-10 audit found missing, in dependency order.

Each entry names the claim it unblocks, so a half-finished sweep still tells you which
sentences in the manuscript are supported and which are not. Nothing here is a new
method: every run is an existing script under flags that were built but never exercised.

Ordering is deliberate. R1 is the only run that can change a headline claim, so it goes
first and the rest are optional hardening.

    python scripts/run_missing_experiments.py --list          # show the plan, run nothing
    python scripts/run_missing_experiments.py --only R1       # one run
    python scripts/run_missing_experiments.py --through R3    # R1..R3
    python scripts/run_missing_experiments.py --all --dry-run # print every command

On Kaggle, set PDE_DATA_ROOT=/kaggle/working first. Every run writes a JSON under
RESULTS_DIR; none overwrites an as-published file (the --clean runs use a _clean suffix).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable


@dataclass
class Run:
    ident: str
    title: str
    unblocks: str
    commands: list[list[str]]
    outputs: list[str]
    minutes: int
    needs_gpu: bool = True
    notes: str = ""
    depends_on: list[str] = field(default_factory=list)


# Four deployable encoders + the reference ceiling, matching the published tables.
TIERS = ["--tiers", "lw11", "lw21", "lw35", "--heavy", "--teachers"]
STRATS = ["--strategies", "bare", "crude", "rich", "grounded"]

RUNS: list[Run] = [
    Run(
        ident="R1",
        title="De-duplicated evaluation at configurations A and B",
        unblocks=(
            "The headline scaling claim. Section 6 concedes duplicate-label density falls "
            "across A/B/C (37.5%, 23.5%, 19.6%) in the same direction as the grounded rise, "
            "and that the runs which would separate the two do not exist. Only configuration C "
            "has a cleaned counterpart. Until A and B do, 'source-grounded descriptors keep "
            "improving as the label space grows' is confounded and the paper says so."
        ),
        commands=[
            [PY, "scripts/evaluate.py", "--exp", "A", "--clean", *STRATS, *TIERS],
            [PY, "scripts/evaluate.py", "--exp", "B", "--clean", *STRATS, *TIERS],
            # C already exists as zeroshot_eval_C_clean.json, but rerun it here so all three
            # cleaned configurations come from one code revision. A cleaned A and B compared
            # against a C measured months earlier would reintroduce the drift this run exists
            # to remove.
            [PY, "scripts/evaluate.py", "--exp", "C", "--clean", *STRATS, *TIERS],
        ],
        outputs=["zeroshot_eval_A_clean.json", "zeroshot_eval_B_clean.json",
                 "zeroshot_eval_C_clean.json"],
        minutes=95,
        notes=(
            "Read the result as follows. If grounded still rises monotonically across the "
            "three CLEANED configurations, the scaling claim survives and 'provisional' can "
            "come out of the abstract, contributions and conclusion. If the cleaned curve is "
            "flat, the trend was label cleanliness and the claim must be withdrawn -- which is "
            "a publishable finding in its own right, not a failure."
        ),
    ),
    Run(
        ident="R2",
        title="Class-macro accuracy on every configuration",
        unblocks=(
            "The averaging-convention objection. The held-out split is capped but not "
            "balanced -- 25 images at the floor against a 600 cap -- so micro top-1 is carried "
            "by the largest classes. The paper reports crop-macro as a robustness check but "
            "never class-macro, which is the one reviewers ask for on imbalanced problems."
        ),
        commands=[
            [PY, "scripts/evaluate.py", "--exp", "A", *STRATS, *TIERS],
            [PY, "scripts/evaluate.py", "--exp", "B", *STRATS, *TIERS],
            [PY, "scripts/evaluate.py", "--exp", "C", *STRATS, *TIERS],
        ],
        outputs=["zeroshot_eval_A.json", "zeroshot_eval_B.json", "zeroshot_eval_C.json"],
        minutes=95,
        notes=(
            "This re-runs the AS-PUBLISHED sweep. It exists only because zeroshot.py did not "
            "record per-class hits, so class-macro cannot be recovered from the stored JSONs -- "
            "the fix (by_class / class_macro_acc) is in place but the numbers must be "
            "regenerated. Micro accuracies MUST come back identical to the published tables; "
            "if any moves, stop and find out why before touching the manuscript."
        ),
    ),
    Run(
        ident="R3",
        title="Control arm at eleven generation seeds",
        unblocks=(
            "The central null result. Seven seeds give a paired interval of [-0.23, +4.14] "
            "that contains zero; Section 4.3 computes that roughly eleven seeds would resolve "
            "an effect of this size at 80% power and names that as the experiment which would "
            "settle it. This is that experiment."
        ),
        commands=[
            # --seed is singular and --arm generates one side per invocation, so four new
            # seeds is eight generation calls. Both arms must exist at every seed or the
            # paired subset silently shrinks instead of growing.
            *[[PY, "scripts/build_ungrounded.py", "--seed", str(sd), "--which", "heldout",
               "--arm", arm]
              for sd in (8, 9, 10, 11) for arm in ("ungrounded", "grounded")],
            [PY, "scripts/evaluate.py", "--exp", "C",
             "--paired-arms", "grounded_matched", "ungrounded",
             "--ungrounded-seeds", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11",
             *TIERS],
            [PY, "scripts/analyse_control_arms.py"],
        ],
        outputs=["zeroshot_eval_C_paired_ungseeds.json", "control_arm_statistics.json"],
        minutes=150,
        notes=(
            "The seed is the replication unit, so all eleven must be scored by every encoder "
            "on the same images -- run them in ONE process (that is what --ungrounded-seeds "
            "is for) or the image-embedding cache dies between seeds and the sweep takes hours "
            "longer. Do not stop early and report the best subset: the stopping rule has to be "
            "fixed in advance or the interval means nothing. Generation needs ANTHROPIC_API_KEY "
            "or LAVA_API_KEY set; the evaluation half does not."
        ),
        depends_on=["R2"],
    ),
    Run(
        ident="R4",
        title="WiSE-FT on the full seen split",
        unblocks=(
            "Section 4.5's own caveat. The sweep subsamples the seen split to 200 images per "
            "class, so its frozen reference is 58.8% against the 82.2% Table 4 reports for the "
            "same encoder, and full fine-tuning reaches 81.5% -- still below the full-split "
            "frozen probe. The 22.5 points the dial appears to buy may be recovering the "
            "subsample deficit rather than adding accuracy."
        ),
        commands=[
            [PY, "scripts/wiseft.py", "--exp", "C", "--model", "s0", "--max-per-class", "0"],
        ],
        outputs=["wiseft_fullsplit.json"],
        minutes=70,
        notes=(
            "--max-per-class 0 disables the cap. Expect alpha=0 to land near 82%, not 59%. "
            "If the dial then buys far less than 22.5 points, Section 4.5's hedge becomes the "
            "finding and the sweep can be reported as a measurement rather than a "
            "demonstration that the interpolation behaves."
        ),
    ),
    Run(
        ident="R5",
        title="DCLIP / CuPL per-class descriptor baseline",
        unblocks=(
            "The missing comparator. The paper argues the keyword bank is collision-crippled "
            "and that per-class generation methods 'cannot collide', then never runs one. The "
            "ungrounded control arm is per-class-generated and stands in for it, but it was "
            "not designed as a reimplementation of either published method."
        ),
        commands=[
            [PY, "scripts/build_ungrounded.py", "--arm", "dclip", "--seeds", "1"],
            [PY, "scripts/evaluate.py", "--exp", "C", "--strategies", "dclip", *TIERS],
        ],
        outputs=["zeroshot_eval_C_dclip.json"],
        minutes=60,
        notes=(
            "NEEDS WORK BEFORE IT WILL RUN: build_ungrounded.py has no --arm flag yet and "
            "descriptors.py has no 'dclip' strategy. Both are small (a prompt that asks for "
            "'useful visual features for distinguishing X in a photograph', one entry in "
            "ARM_DIRS), but they are real edits, not flags. Cheapest honest alternative: leave "
            "the limitation paragraph as written and cite the control arm as the stand-in."
        ),
    ),
    Run(
        ident="R6",
        title="Arm-class edge measurement",
        unblocks=(
            "The title. 'At the Edge' is currently supported by an unnamed laptop x86 CPU. "
            "Section 4.8's INT8 conclusion is explicitly x86-specific and the paper already "
            "flags that Arm vector extensions may reverse it for the hybrid tiers."
        ),
        commands=[
            [PY, "scripts/benchmark_edge.py", "--models", "s0", "s1", "s2", "b",
             "--runs", "50"],
        ],
        outputs=["edge_benchmark.json  (RENAME IT IMMEDIATELY -- see note)"],
        minutes=40,
        needs_gpu=False,
        notes=(
            "Must run ON the target device (Raspberry Pi 5, Jetson Orin Nano, or an Apple "
            "silicon laptop), not cross-compiled. benchmark_edge.py has no --tag flag and "
            "writes results/edge/edge_benchmark.json unconditionally, so copy the x86 file "
            "aside first and rename the Arm output afterwards, or the laptop numbers behind "
            "Table 8 are gone. One board is enough to turn a flagged assumption into a "
            "measurement, and it is the cheapest fix to the venue-fit objection a Computers "
            "and Electronics in Agriculture reviewer will raise about smallholder deployment."
        ),
    ),
]

BY_ID = {r.ident: r for r in RUNS}


def show_plan(runs: list[Run]) -> None:
    total = sum(r.minutes for r in runs)
    print(f"\n{len(runs)} run(s), ~{total} min ({total / 60:.1f} h) of compute\n")
    for r in runs:
        gpu = "GPU" if r.needs_gpu else "target device"
        dep = f"  [after {', '.join(r.depends_on)}]" if r.depends_on else ""
        print(f"{r.ident}  {r.title}   (~{r.minutes} min, {gpu}){dep}")
        print(f"      unblocks: {r.unblocks}")
        if r.notes:
            print(f"      note:     {r.notes}")
        for c in r.commands:
            print(f"      $ {' '.join(c[1:]) if c[0] == PY else ' '.join(c)}")
        print(f"      writes:   {', '.join(r.outputs)}\n")


def execute(runs: list[Run], dry_run: bool) -> int:
    failures = 0
    for r in runs:
        print(f"\n{'=' * 78}\n{r.ident}  {r.title}\n{'=' * 78}")
        for cmd in r.commands:
            printable = " ".join(cmd[1:] if cmd[0] == PY else cmd)
            print(f"$ {printable}")
            if dry_run:
                continue
            t0 = time.time()
            proc = subprocess.run(cmd, cwd=REPO)
            dt = (time.time() - t0) / 60
            if proc.returncode != 0:
                print(f"  FAILED (exit {proc.returncode}) after {dt:.1f} min -- stopping {r.ident}")
                failures += 1
                break
            print(f"  ok ({dt:.1f} min)")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--only", metavar="ID", help="run exactly one (e.g. R1)")
    g.add_argument("--through", metavar="ID", help="run R1 up to and including ID")
    g.add_argument("--all", action="store_true", help="run everything")
    ap.add_argument("--list", action="store_true", help="print the plan and exit")
    ap.add_argument("--dry-run", action="store_true", help="print commands without running")
    args = ap.parse_args()

    if args.only:
        if args.only not in BY_ID:
            raise SystemExit(f"unknown run {args.only}; choose from {', '.join(BY_ID)}")
        selected = [BY_ID[args.only]]
    elif args.through:
        if args.through not in BY_ID:
            raise SystemExit(f"unknown run {args.through}; choose from {', '.join(BY_ID)}")
        selected = RUNS[: [r.ident for r in RUNS].index(args.through) + 1]
    elif args.all:
        selected = RUNS
    else:
        selected = RUNS
        args.list = True

    if args.list:
        show_plan(selected)
        print("Pick one with --only R1, a prefix with --through R3, or everything with --all.")
        return 0

    failures = execute(selected, args.dry_run)
    print(f"\n{len(selected) - failures}/{len(selected)} run(s) completed")
    return failures


if __name__ == "__main__":
    sys.exit(main())
