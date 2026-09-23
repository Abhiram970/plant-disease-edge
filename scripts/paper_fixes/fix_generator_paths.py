#!/usr/bin/env python3
"""Repoint the Kaggle generators after the move into scripts/kaggle/build/.

Each generator used to sit next to bootstrap.py and next to the runners it emits, so
`HERE` served for both. It now sits one level down in build/, so the bootstrap import
and the output directory both need HERE.parent. Left unfixed, the generators would
import nothing and write runners into scripts/kaggle/build/runners/.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BUILD = REPO / "scripts" / "kaggle" / "build"

PREAMBLE = (
    "HERE = Path(__file__).resolve().parent\n"
    "KAGGLE = HERE.parent                 # scripts/kaggle -- holds bootstrap.py\n"
    "RUNNERS = KAGGLE / \"runners\"         # generated stage files land here\n"
    "RUNNERS.mkdir(parents=True, exist_ok=True)\n"
    "sys.path.insert(0, str(KAGGLE))\n"
)

FIXES = [
    # bootstrap now lives one level up
    ("HERE = Path(__file__).resolve().parent\nsys.path.insert(0, str(HERE))\n", PREAMBLE),
    ("sys.path.insert(0, str(Path(__file__).resolve().parent))\n",
     "HERE = Path(__file__).resolve().parent\n"
     "KAGGLE = HERE.parent\n"
     "RUNNERS = KAGGLE / \"runners\"\n"
     "RUNNERS.mkdir(parents=True, exist_ok=True)\n"
     "sys.path.insert(0, str(KAGGLE))\n"),
    # runners are emitted one level up, not into build/
    ('    p = HERE / name\n', '    p = KAGGLE / name\n'),
    ('p = HERE / name\n', 'p = KAGGLE / name\n'),
    ('    out = HERE / "runners/', '    out = KAGGLE / "runners/'),
    ('out = HERE / "runners/', 'out = KAGGLE / "runners/'),
    ('out = Path(__file__).resolve().parent / "runners/',
     'out = KAGGLE / "runners/'),
]


def main() -> int:
    changed = 0
    for f in sorted(BUILD.glob("*.py")):
        with open(f, encoding="utf-8", newline="") as fh:
            text = fh.read()
        before = text
        for old, new in FIXES:
            if old in text:
                text = text.replace(old, new)
        # Any generator that emits a runner must know where RUNNERS is.
        if ("KAGGLE /" in text or "RUNNERS" in text) and "KAGGLE = " not in text:
            text = text.replace(
                "HERE = Path(__file__).resolve().parent\n",
                "HERE = Path(__file__).resolve().parent\n"
                "KAGGLE = HERE.parent\n"
                "RUNNERS = KAGGLE / \"runners\"\n"
                "RUNNERS.mkdir(parents=True, exist_ok=True)\n", 1)
        if text != before:
            with open(f, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
            print(f"  repointed {f.relative_to(REPO)}")
            changed += 1
        else:
            print(f"  unchanged {f.relative_to(REPO)}")
    print(f"\n{changed} generator(s) repointed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
