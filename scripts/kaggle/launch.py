"""
=====================================================================================
 PDE LAUNCHER  --  paste THIS one cell. It clones and RUNS the chosen stage.
=====================================================================================
Nothing else to copy. Change PART below and run the cell.

    PART = "stage1"    -> descriptors (4 seeds) + zero-shot A/B/C + control arms
                          + probe + abstention metrics             (~3.9 h, NEEDS API KEY)
    PART = "stage2"    -> seeds 4-7, control arms at 8 seeds, short arms, clean eval,
                          LOCO, WiSE-FT, and the 14 CNNs           (~7.8 h, NEEDS API KEY)

    PART = "stage3"    -> ONLY the two stages the 2026-09-06 run lost: the 14 CNN
                          baselines and the WiSE-FT sweep. Everything else is carried
                          forward from the attached output, not recomputed.
                                                                   (~6.5 h, NO API KEY)

    PART = "1" / "2" / "3"  -> the original single-purpose parts, if you want one stage

SETUP (all parts)
  Add Data      -> your `pde-sage-data` dataset (the 288 px exp_data build)
  Accelerator   -> GPU T4 x2 (or P100)
  Internet      -> ON
  Save Version  -> "Save & Run All", so the run survives closing the browser

SETUP (tonight / part 1 only)
  Add-ons -> Secrets -> LAVA_API_KEY   (or ANTHROPIC_API_KEY)

FOR "stage3" AND PART 3
  Also Add Data -> the output of the previous notebook, so its results carry forward.
  "stage3" needs no API key: it generates no descriptors.

WHY A LAUNCHER. The previous attempt pasted a helper cell that PRINTS the runner
(`print(open(...).read())`) instead of executing it, so the whole file was echoed to the log
and nothing ran. This cell removes that failure mode: there is only one thing to paste, and
it executes the file rather than displaying it.
=====================================================================================
"""

PART = "audit"       # "audit" | "stage1" | "stage2" | "stage3" | "1" | "2" | "3"

REPO_URL = "https://github.com/Abhiram970/plant-disease-edge.git"
REPO_REF = "paper/draft-audit-2026-09-01"

import subprocess, sys, shutil, os
from pathlib import Path

SRC = {
    "stage1":  "runners/stage1_descriptors_zeroshot.py",
    "stage2":  "runners/stage2_seeds_cnns.py",
    "1":       "runners/part1_descriptors.py",
    "2":       "runners/part2_probe_loco_wiseft.py",
    "3":       "runners/part3_cnns.py",
    "stage3":  "runners/stage3_cnns_wiseft.py",
    # 2026-09-10 audit. Cleaned evaluation at A and B (no other stage runs --clean
    # anywhere but exp C) plus the class-macro re-measurement. Pure evaluation, no
    # API key. Needs a clone that carries the by_class change in scripts/zeroshot.py --
    # the stage checks and refuses to start otherwise.
    "audit":   "runners/audit_dedup_macro.py",
}
if PART not in SRC:
    sys.exit(f"[launcher] PART must be one of {list(SRC)}, got {PART!r}")

CODE = Path("/kaggle/working/_pde_code")
if CODE.exists():
    shutil.rmtree(CODE, ignore_errors=True)

print(f"[launcher] cloning {REPO_REF} ...", flush=True)
rc = subprocess.run(["git", "clone", "--depth", "1", "--branch", REPO_REF, REPO_URL, str(CODE)],
                    capture_output=True, text=True)
if rc.returncode != 0:
    sys.exit(f"[launcher] clone failed:\n{rc.stderr}")

# The tree moved under scripts/kaggle/ on 2026-09-11. A clone of a branch older than
# that still has the flat kaggle/ layout, so resolve against both rather than failing
# with a bare "runner not found".
_roots = [CODE / "scripts" / "kaggle", CODE / "kaggle"]
runner = next((r / SRC[PART] for r in _roots if (r / SRC[PART]).exists()), None)
if runner is None:
    _legacy = {"runners/stage1_descriptors_zeroshot.py": "RUN_TONIGHT_parts1and2.py",
               "runners/stage2_seeds_cnns.py": "RUN_MORNING_everything_else.py",
               "runners/part1_descriptors.py": "RUN_PART1_descriptors.py",
               "runners/part2_probe_loco_wiseft.py": "RUN_PART2_probe_loco_wiseft.py",
               "runners/part3_cnns.py": "RUN_PART3_cnns.py",
               "runners/stage3_cnns_wiseft.py": "RUN_FIXUP_cnns_wiseft.py",
               "runners/audit_dedup_macro.py": "RUN_AUDIT_FIXES.py"}
    _old = CODE / "kaggle" / _legacy.get(SRC[PART], "")
    runner = _old if _old.name and _old.exists() else None
if runner is None:
    sys.exit(f"[launcher] {SRC[PART]} not found in the clone under either layout")
print(f"[launcher] runner: {runner.relative_to(CODE)}", flush=True)
if not runner.exists():
    sys.exit(f"[launcher] {runner} not found in the clone")

head = subprocess.run(["git", "-C", str(CODE), "log", "--oneline", "-1"],
                      capture_output=True, text=True).stdout.strip()
print(f"[launcher] HEAD = {head}", flush=True)
print(f"[launcher] running {SRC[PART]}\n", flush=True)

# Stream the child's output live so Kaggle's log shows progress as it happens rather than
# buffering everything until the end. -u keeps the child unbuffered.
proc = subprocess.Popen([sys.executable, "-u", str(runner)],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                        bufsize=1, cwd=str(CODE))
for line in proc.stdout:
    print(line, end="", flush=True)
proc.wait()

print(f"\n[launcher] {SRC[PART]} exited with code {proc.returncode}", flush=True)
if proc.returncode != 0:
    print("[launcher] Non-zero exit. Re-running this cell resumes: every stage is", flush=True)
    print("[launcher] resumable and finished work is skipped.", flush=True)

# Surface the bundle regardless of exit code -- partial results are still worth downloading.
for z in sorted(Path("/kaggle/working").glob("pde_*.zip")):
    print(f"[launcher] bundle: {z}  ({z.stat().st_size/1e6:.2f} MB)", flush=True)
    try:
        from IPython.display import FileLink, display
        display(FileLink(str(z.relative_to("/kaggle/working"))))
    except Exception:
        pass
