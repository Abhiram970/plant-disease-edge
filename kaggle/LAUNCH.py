"""
=====================================================================================
 PDE LAUNCHER  --  paste THIS one cell. It clones and RUNS the chosen stage.
=====================================================================================
Nothing else to copy. Change PART below and run the cell.

    PART = "tonight"   -> descriptors (4 seeds) + zero-shot A/B/C + control arms
                          + probe + abstention metrics             (~3.9 h, NEEDS API KEY)
    PART = "morning"   -> seeds 4-7, control arms at 8 seeds, short arms, clean eval,
                          LOCO, WiSE-FT, and the 14 CNNs           (~7.8 h, NEEDS API KEY)

    PART = "1" / "2" / "3"  -> the original single-purpose parts, if you want one stage

SETUP (all parts)
  Add Data      -> your `pde-sage-data` dataset (the 288 px exp_data build)
  Accelerator   -> GPU T4 x2 (or P100)
  Internet      -> ON
  Save Version  -> "Save & Run All", so the run survives closing the browser

SETUP (tonight / part 1 only)
  Add-ons -> Secrets -> LAVA_API_KEY   (or ANTHROPIC_API_KEY)

FOR PART 3 IN THE MORNING
  Also Add Data -> the output of tonight's notebook, so its results carry forward.

WHY A LAUNCHER. The previous attempt pasted a helper cell that PRINTS the runner
(`print(open(...).read())`) instead of executing it, so the whole file was echoed to the log
and nothing ran. This cell removes that failure mode: there is only one thing to paste, and
it executes the file rather than displaying it.
=====================================================================================
"""

PART = "tonight"     # "tonight" | "1" | "2" | "3"

REPO_URL = "https://github.com/Abhiram970/plant-disease-edge.git"
REPO_REF = "paper/draft-audit-2026-09-01"

import subprocess, sys, shutil, os
from pathlib import Path

SRC = {
    "tonight": "RUN_TONIGHT_parts1and2.py",
    "morning": "RUN_MORNING_everything_else.py",
    "1":       "RUN_PART1_descriptors.py",
    "2":       "RUN_PART2_probe_loco_wiseft.py",
    "3":       "RUN_PART3_cnns.py",
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

runner = CODE / "kaggle" / SRC[PART]
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
