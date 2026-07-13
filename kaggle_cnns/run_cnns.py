"""Plant-Disease-Edge CNN Baselines — Self-contained Kaggle script
Downloads SAGE data, builds manifest, trains 4 CNNs, saves results.
"""
import subprocess, sys, os
from pathlib import Path

# ====== Install deps ======
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "timm", "torchvision", "pillow", "tqdm", "python-dotenv", "huggingface_hub", "pyarrow"])

# ====== Setup ======
os.environ["PDE_DATA_ROOT"] = "/kaggle/working"
os.environ["PDE_DATASET_DIR"] = "/kaggle/working/exp_data"
WORK = Path("/kaggle/working")
os.chdir(str(WORK))

# ====== Clone or download the repo scripts ======
import urllib.request, io, zipfile

REPO_URL = "https://github.com/Abhiram970/plant-disease-edge/archive/refs/heads/main.zip"
print(">>> Downloading repo scripts...")
with urllib.request.urlopen(REPO_URL) as resp:
    zf = zipfile.ZipFile(io.BytesIO(resp.read()))
    for f in zf.namelist():                       # any */scripts/*.py (branch-agnostic)
        if "/scripts/" in f and f.endswith(".py"):
            (WORK / Path(f).name).write_bytes(zf.read(f))
            print(f"  extracted: {Path(f).name}")

# ====== Download SAGE data ======
print(">>> Downloading SAGE data...")
subprocess.check_call([sys.executable, str(WORK / "sage_data.py"), "--role", "all"])

# ====== Build manifest ======
print(">>> Building manifest...")
subprocess.check_call([sys.executable, str(WORK / "build_manifest.py"), "--min-images", "25"])

# ====== Run CNN baselines ======
print(">>> Running CNN baselines (4 epochs, batch 128, workers 4)...")

ARCHS = [
    "convnextv2_nano",
    "fastvit_t8",
    "mobilenetv4_conv_small",
    "tf_efficientnetv2_s",
]

for arch in ARCHS:
    out = WORK / "results" / f"supervised_{arch}.json"
    if out.exists():
        print(f"  [skip] {arch} — already done")
    else:
        print(f"\n  === Training {arch} ===")
        rc = subprocess.run([
            sys.executable, "-u", str(WORK / "supervised_baseline.py"),
            "--arch", arch, "--epochs", "4", "--batch", "128", "--workers", "4"
        ]).returncode
        if rc != 0:
            print(f"  [FAIL] {arch} (exit {rc})")

# ====== Summary ======
import json
print("\n" + "="*60)
print("CNN BASELINE SUMMARY")
print("="*60)
for arch in ARCHS:
    p = WORK / "results" / f"supervised_{arch}.json"
    if p.exists():
        r = json.loads(p.read_text())
        print(f"  {arch:30s}  {r['seen_top1']:.1%}")
    else:
        print(f"  {arch:30s}  MISSING")
print("\nDone!")