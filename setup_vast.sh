#!/usr/bin/env bash
# One-shot environment bootstrap for a rented GPU (Vast/RunPod/Lambda).
# Finds the python that ALREADY has torch (never reinstalls the 2GB torch), then installs only the
# small deps timm needs. Writes the chosen interpreter to /tmp/PDEPY so later commands reuse it.
#
#   bash setup_vast.sh
#   PY=$(cat /tmp/PDEPY)          # then run everything with "$PY scripts/..."
set -u

echo "[setup] locating a python with torch already installed ..."
CANDS="python python3 /opt/conda/bin/python /venv/main/bin/python /venv/bin/python /root/venv/bin/python"
CANDS="$CANDS $(ls /opt/conda/envs/*/bin/python 2>/dev/null) $(ls /venv*/bin/python 2>/dev/null) $(ls /workspace/venv/bin/python 2>/dev/null)"

PY=""
for c in $CANDS; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c "import torch" >/dev/null 2>&1; then PY="$c"; break; fi
done

if [ -z "$PY" ]; then
  echo "[setup] no pre-installed torch found via common paths; scanning filesystem ..."
  T=$(find / -maxdepth 9 -path "*/site-packages/torch/__init__.py" 2>/dev/null | head -1)
  if [ -n "$T" ]; then
    # site-packages/.../pythonX.Y/site-packages/torch -> the env's bin/python is two levels up from lib
    ENV=$(echo "$T" | sed -E 's#/lib/python[0-9.]+/site-packages/torch/__init__.py##')
    [ -x "$ENV/bin/python" ] && "$ENV/bin/python" -c "import torch" >/dev/null 2>&1 && PY="$ENV/bin/python"
  fi
fi

if [ -z "$PY" ]; then
  echo "[setup] !! torch NOT found anywhere. This image has no torch."
  echo "[setup]    Either pick a real PyTorch template, or install it (slow):"
  echo "           python3 -m pip install torch torchvision"
  exit 1
fi

echo "[setup] using PY=$PY"
"$PY" -c "import torch; print('[setup] torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "[setup] installing light deps (timm + open_clip WITHOUT touching torch) ..."
# --no-deps on the two big ones so pip never tries to "upgrade" the pre-installed torch (2GB re-download)
"$PY" -m pip install -q --no-deps timm open_clip_torch
"$PY" -m pip install -q safetensors pyyaml huggingface_hub pyarrow tqdm \
                       regex ftfy pandas numpy pillow

"$PY" - <<'PYCHK'
import torch, timm, pyarrow, huggingface_hub, tqdm
import open_clip                      # required by every model in the pipeline
print("[setup] ENV OK — torch", torch.__version__,
      "| cuda", torch.cuda.is_available(),
      "| open_clip", open_clip.__version__ if hasattr(open_clip, "__version__") else "ok")
if torch.cuda.is_available():
    print("[setup] GPU:", torch.cuda.get_device_name(0),
          f"({torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB)")
PYCHK
echo "$PY" > /tmp/PDEPY
echo ""
echo "=========================================================="
echo "  READY. Interpreter saved to /tmp/PDEPY"
echo "  Run everything with:   PY=\$(cat /tmp/PDEPY)"
echo "                         \$PY scripts/sage_data.py --role all"
echo "=========================================================="
