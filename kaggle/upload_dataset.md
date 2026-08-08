# Getting the images onto Kaggle without re-downloading SAGE

**Do not use Git LFS for this.** The image set is **19.81 GB across 92,602 files**. GitHub's free LFS
allowance is 1 GB storage and 1 GB/month bandwidth; data packs are \$5/month per 50 GB, and *every*
Kaggle clone would re-pull ~20 GB, so two clones a month exhausts the quota. LFS is also slow with
tens of thousands of small objects, and a 20 GB clone would burn 20--40 minutes of every Kaggle
session — the exact cost you are trying to avoid.

A **Kaggle Dataset** is the right tool: free, private, and mounted read-only at `/kaggle/input` with
**zero** download time inside the notebook.

There are two ways to create it. Pick based on your upload speed.

---

## Option A — upload the data you already have (recommended)

You have the built, deduplicated subset on disk at `C:\kaggle\working\exp_data`. Uploading it once is
simpler and more reproducible than re-deriving it on Kaggle, and it is the *same* data every result so
far was computed from.

**Cost:** one 19.8 GB upload. At 20 Mbps up that is ~2.5 h; at 50 Mbps ~1 h. Start it before bed.

### 1. Get a Kaggle API token

kaggle.com → your avatar → **Settings** → **API** → *Create New Token*. That downloads `kaggle.json`.
Put it where the CLI looks:

```powershell
mkdir "$env:USERPROFILE\.kaggle" -Force
move "$env:USERPROFILE\Downloads\kaggle.json" "$env:USERPROFILE\.kaggle\kaggle.json"
```

(You do not currently have this file — the CLI is installed but unauthenticated.)

### 2. Write the dataset metadata

```powershell
@'
{
  "title": "PDE SAGE subset",
  "id": "YOUR_KAGGLE_USERNAME/pde-sage-data",
  "licenses": [{"name": "MIT"}]
}
'@ | Out-File -Encoding utf8 C:\kaggle\working\exp_data\dataset-metadata.json
```

Replace `YOUR_KAGGLE_USERNAME` with your actual Kaggle username (lowercase, as it appears in your
profile URL).

### 3. Copy the manifest in alongside the images

The training scripts need it, and keeping it in the same dataset means one thing to attach:

```powershell
copy C:\kaggle\working\manifest.csv C:\kaggle\working\exp_data\manifest.csv
```

### 4. Upload

```powershell
kaggle datasets create -p C:\kaggle\working\exp_data --dir-mode zip
```

`--dir-mode zip` matters: it zips each class folder rather than uploading 92,602 files individually,
which would take far longer and often fails partway. Kaggle unpacks them on their side.

If the connection drops, re-run the same command with `version` instead of `create`:

```powershell
kaggle datasets version -p C:\kaggle\working\exp_data -m "resume" --dir-mode zip
```

### 5. Use it

In the notebook: **Add data** → *Your Datasets* → `pde-sage-data`. It mounts at
`/kaggle/input/pde-sage-data`. `kaggle/cnn_baselines_notebook.py` already looks there first and skips
the HuggingFace fetch entirely.

---

## Option B — build it once on Kaggle

If your upload is slow, let Kaggle's datacenter bandwidth do the download instead, then snapshot the
result as a Dataset. This is `RUNBOOK_KAGGLE.md` Session 1.

**This previously got stuck, and we found why.** `hf_hub_download` returns a *symlink* into the
HuggingFace cache; the code deleted the symlink but not the ~10 GB blob behind it, so each shard
permanently consumed disk and the fetch died once `/kaggle/working` filled. Fixed in `sage_data.py`
(the blob is now resolved and deleted), but **also point the cache at scratch space**, because the
default cache still lands on the small disk:

```python
import os
os.environ["HF_HOME"] = "/kaggle/temp/hf"        # 73 GB scratch, wiped at session end
os.environ["HF_HUB_CACHE"] = "/kaggle/temp/hf/hub"
os.environ["PDE_DATA_ROOT"] = "/kaggle/working"
os.environ["PDE_DATASET_DIR"] = "/kaggle/working/exp_data"
!python scripts/sage_data.py --role all
!python scripts/build_manifest.py --min-images 25
!du -sh /kaggle/working/exp_data
```

Then right pane → **Output → Create Dataset** → name it `pde-sage-data`.

**Watch the ceiling.** Notebook output is capped around 20 GB and the full subset is 19.81 GB — you are
right at the edge. If it fails, either fetch seen crops only (`--role train`, which is all the CNN
baselines need, ~15 GB) or lower the per-class cap in `scripts/config.py`.

---

## Which to choose

| | Option A (upload) | Option B (build on Kaggle) |
|---|---|---|
| Your time | one long upload, unattended | ~2--4 h of Kaggle session |
| Reproducibility | **identical to every result so far** | re-derived; dedup/caps should match, but not guaranteed byte-identical |
| Risk | connection drops (resumable) | 20 GB output ceiling |
| GPU quota | none | none (use CPU) |

Option A is the safer choice, mainly because it guarantees the Kaggle runs use *exactly* the data the
local results came from. Given the pool-drift problem we already hit once — where class counts changed
because the on-disk snapshot changed — that guarantee is worth the upload time.
