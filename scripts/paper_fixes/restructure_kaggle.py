#!/usr/bin/env python3
"""Fold kaggle/ into scripts/kaggle/ with names that say what each stage does.

The old layout named runners after when they happened to be launched -- TONIGHT,
MORNING, FIXUP -- which tells a reader nothing and dates the repository. The new names
say what runs. Generated runners are not moved; they are regenerated at the new path
from their generators, which is the only way to keep the two in step.

    kaggle/_pde_common.py                 -> scripts/kaggle/bootstrap.py
    kaggle/LAUNCH.py                      -> scripts/kaggle/launch.py
    kaggle/CELL_AUDIT.py                  -> scripts/kaggle/cell_audit.py
    kaggle/RUNBOOK.md                     -> scripts/kaggle/README.md
    kaggle/build_parts.py                 -> scripts/kaggle/build/parts.py
    kaggle/build_tonight.py               -> scripts/kaggle/build/stage1.py
    kaggle/build_morning.py               -> scripts/kaggle/build/stage2.py
    kaggle/build_fixup.py                 -> scripts/kaggle/build/stage3.py
    kaggle/build_audit_fixes.py           -> scripts/kaggle/build/audit.py
    RUN_PART1_descriptors.py              -> runners/part1_descriptors.py
    RUN_PART2_probe_loco_wiseft.py        -> runners/part2_probe_loco_wiseft.py
    RUN_PART3_cnns.py                     -> runners/part3_cnns.py
    RUN_TONIGHT_parts1and2.py             -> runners/stage1_descriptors_zeroshot.py
    RUN_MORNING_everything_else.py        -> runners/stage2_seeds_cnns.py
    RUN_FIXUP_cnns_wiseft.py              -> runners/stage3_cnns_wiseft.py
    RUN_AUDIT_FIXES.py                    -> runners/audit_dedup_macro.py

    python scripts/paper_fixes/restructure_kaggle.py --dry-run
    python scripts/paper_fixes/restructure_kaggle.py
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OLD = REPO / "kaggle"
NEW = REPO / "scripts" / "kaggle"

# source -> destination, relative to REPO. Generated RUN_*.py are omitted on purpose.
MOVES = {
    "kaggle/_pde_common.py": "scripts/kaggle/bootstrap.py",
    "kaggle/LAUNCH.py": "scripts/kaggle/launch.py",
    "kaggle/CELL_AUDIT.py": "scripts/kaggle/cell_audit.py",
    "kaggle/RUNBOOK.md": "scripts/kaggle/README.md",
    "kaggle/build_parts.py": "scripts/kaggle/build/parts.py",
    "kaggle/build_tonight.py": "scripts/kaggle/build/stage1.py",
    "kaggle/build_morning.py": "scripts/kaggle/build/stage2.py",
    "kaggle/build_fixup.py": "scripts/kaggle/build/stage3.py",
    "kaggle/build_audit_fixes.py": "scripts/kaggle/build/audit.py",
}

RUNNER_RENAMES = {
    "RUN_PART1_descriptors.py": "part1_descriptors.py",
    "RUN_PART2_probe_loco_wiseft.py": "part2_probe_loco_wiseft.py",
    "RUN_PART3_cnns.py": "part3_cnns.py",
    "RUN_TONIGHT_parts1and2.py": "stage1_descriptors_zeroshot.py",
    "RUN_MORNING_everything_else.py": "stage2_seeds_cnns.py",
    "RUN_FIXUP_cnns_wiseft.py": "stage3_cnns_wiseft.py",
    "RUN_AUDIT_FIXES.py": "audit_dedup_macro.py",
}

# Text rewrites applied to every moved .py/.md. Order matters: longest first, so that
# "build_audit_fixes.py" is not partly rewritten by the "build_" rules.
REWRITES = [
    # module + path references
    ('from _pde_common import BOOTSTRAP', 'from bootstrap import BOOTSTRAP'),
    ('_pde_common.py', 'bootstrap.py'),
    ('_pde_common', 'bootstrap'),
    ('kaggle/LAUNCH.py', 'scripts/kaggle/launch.py'),
    ('kaggle/RUNBOOK.md', 'scripts/kaggle/README.md'),
    ('kaggle/CELL_AUDIT.py', 'scripts/kaggle/cell_audit.py'),
    ('kaggle/build_audit_fixes.py', 'scripts/kaggle/build/audit.py'),
    ('kaggle/build_parts.py', 'scripts/kaggle/build/parts.py'),
    ('kaggle/build_tonight.py', 'scripts/kaggle/build/stage1.py'),
    ('kaggle/build_morning.py', 'scripts/kaggle/build/stage2.py'),
    ('kaggle/build_fixup.py', 'scripts/kaggle/build/stage3.py'),
    ('build_parts.py', 'build/parts.py'),
    ('build_tonight.py', 'build/stage1.py'),
    ('build_morning.py', 'build/stage2.py'),
    ('build_fixup.py', 'build/stage3.py'),
    ('build_audit_fixes.py', 'build/audit.py'),
    ('kaggle/RUN_AUDIT_FIXES.py', 'scripts/kaggle/runners/audit_dedup_macro.py'),
]
REWRITES += [(old, f"runners/{new}") for old, new in RUNNER_RENAMES.items()]


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=REPO, check=False, capture_output=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not OLD.exists():
        print("kaggle/ already folded in; nothing to do")
        return 0

    print("MOVE")
    for src_rel, dst_rel in MOVES.items():
        src, dst = REPO / src_rel, REPO / dst_rel
        if not src.exists():
            print(f"  [absent] {src_rel}")
            continue
        print(f"  {src_rel:34s} -> {dst_rel}")
        if not args.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            run(["git", "mv", src_rel, dst_rel])
            if src.exists():                      # not tracked, or git mv declined
                shutil.move(str(src), str(dst))

    print("\nREWRITE references inside the moved files")
    if not args.dry_run:
        for dst_rel in MOVES.values():
            p = REPO / dst_rel
            if not p.exists():
                continue
            with open(p, encoding="utf-8", newline="") as fh:
                text = fh.read()
            before = text
            for old, new in REWRITES:
                text = text.replace(old, new)
            if text != before:
                with open(p, "w", encoding="utf-8", newline="\n") as fh:
                    fh.write(text)
                print(f"  updated {dst_rel}")

    print("\nDROP generated runners (regenerated below at the new path)")
    for name in RUNNER_RENAMES:
        p = OLD / name
        if p.exists():
            print(f"  {name}")
            if not args.dry_run:
                run(["git", "rm", "-q", "-f", f"kaggle/{name}"])
                if p.exists():
                    p.unlink()

    if not args.dry_run:
        for leftover in list(OLD.glob("*")):
            if leftover.name == "__pycache__":
                shutil.rmtree(leftover, ignore_errors=True)
        if OLD.exists() and not any(OLD.iterdir()):
            OLD.rmdir()
            print("\nremoved empty kaggle/")
        elif OLD.exists():
            print(f"\nkaggle/ still holds: {[p.name for p in OLD.iterdir()]}")

    print("\nNext: run each generator to emit runners/ at the new path.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
