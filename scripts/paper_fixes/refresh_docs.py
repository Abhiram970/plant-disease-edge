#!/usr/bin/env python3
"""Bring README.md, scripts/README.md and scripts/kaggle/README.md in step with the move.

kaggle/ became scripts/kaggle/, the runners were renamed away from time-of-day labels,
and the audit material moved out of the repository. The docs still describe the old tree.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Applied to every doc listed in TARGETS. Longest patterns first.
COMMON = [
    ("[`kaggle/RUNBOOK.md`](kaggle/RUNBOOK.md)",
     "[`scripts/kaggle/README.md`](scripts/kaggle/README.md)"),
    ("kaggle/LAUNCH.py", "scripts/kaggle/launch.py"),
    ("kaggle/RUNBOOK.md", "scripts/kaggle/README.md"),
    ("kaggle/build_parts.py", "scripts/kaggle/build/parts.py"),
    ("kaggle/build_tonight.py", "scripts/kaggle/build/stage1.py"),
    ("kaggle/build_morning.py", "scripts/kaggle/build/stage2.py"),
    ("kaggle/build_fixup.py", "scripts/kaggle/build/stage3.py"),
    ("kaggle/RUN_*.py", "scripts/kaggle/runners/*.py"),
    ("kaggle/build_*.py", "scripts/kaggle/build/*.py"),
    ('`"tonight"`', '`"stage1"`'),
    ('`"morning"`', '`"stage2"`'),
    ('`"fixup"`', '`"stage3"`'),
    ('"tonight"', '"stage1"'),
    ('"morning"', '"stage2"'),
    ('"fixup"', '"stage3"'),
]

TARGETS = ["README.md", "scripts/README.md", "scripts/kaggle/README.md", "PROJECT_GUIDE.md",
           "CONTRIBUTING.md"]

# Whole-block replacements, per file.
BLOCKS = {
    "README.md": [
        ("""```
scripts/kaggle/launch.py        paste this; clones and runs the chosen stage
scripts/kaggle/runners/*.py         the stages themselves (generated, do not hand-edit)
scripts/kaggle/build/*.py       generators for the above, from one shared bootstrap
scripts/kaggle/README.md       how to run it, and what went wrong before
scripts/                config, data, descriptors, evaluation, checkers
descriptors/            the source-grounded symptom registry (committed)
docs/paper/tex/         main.tex + generated tab_*.tex  <- the submission
docs/paper/*.json       result files; every table and figure is generated from these
results/, logs/         per-run outputs and per-epoch training logs
```""",
         """```
scripts/                    config, data, descriptors, evaluation, analysis, checkers
scripts/kaggle/launch.py    paste into a Kaggle cell; clones and runs the chosen stage
scripts/kaggle/bootstrap.py the shared setup every stage execs
scripts/kaggle/build/       generators
scripts/kaggle/runners/     the stages themselves (generated -- do not hand-edit)
scripts/kaggle/cell_audit.py self-contained cell for the de-duplication re-run
scripts/paper_fixes/        manuscript patches and the pre-submission gates
descriptors/                the source-grounded symptom registry (committed)
docs/paper/tex/             main.tex + generated tab_*.tex  <- the submission
docs/paper/*.json           result files; every table and figure is generated from these
docs/paper/rerun_2026-09-11/ the de-duplicated evaluation at all three scales
results/                    per-run outputs and per-epoch training logs
```

Superseded material -- earlier review passes, backups, build outputs -- lives outside the
repository in `../plant-disease-edge-archive/`; see its `MANIFEST.json`."""),
    ],
    "scripts/kaggle/README.md": [
        ("""**Run `"stage3"` next.** The 2026-09-06 morning session (10.12 h, exit 0) completed everything
except those two stages, and its results carry forward from the attached output rather than being
recomputed. Attach `pde-sage-data` **and** that session's output, then run with `PART = "stage3"`.""",
         """**Run `"audit"` next.** It is the only stage that produces the de-duplicated evaluation at
configurations A and B -- every other stage calls `--clean` at C alone -- and it needs no API key.
Attach `pde-sage-data` **and** the previous session's output, then run with `PART = "audit"`."""),
    ],
}


def main() -> int:
    changed = 0
    for rel in TARGETS:
        p = REPO / rel
        if not p.exists():
            print(f"  [absent] {rel}")
            continue
        with open(p, encoding="utf-8", newline="") as fh:
            text = fh.read()
        before = text
        for old, new in BLOCKS.get(rel, []):
            if old in text:
                text = text.replace(old, new)
        for old, new in COMMON:
            text = text.replace(old, new)
        if text != before:
            with open(p, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
            print(f"  updated {rel}")
            changed += 1
        else:
            print(f"  unchanged {rel}")
    print(f"\n{changed} doc(s) refreshed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
