"""
Phase 0 — DE-RISK SPIKE  (self-contained Kaggle script)
=======================================================
ONE question this answers, before we build the full pipeline:

    Does a ~5M-param edge student (EMOv2-class, our tier-1), distilled from a frozen
    CLIP teacher on a handful of TRAIN crops, still do ZERO-SHOT diagnosis on crops it
    NEVER saw — via text prototypes — meaningfully above chance, and how much of the
    teacher's zero-shot does it RETAIN?

This is the load-bearing claim of the whole paper. If the answer is "yes, clearly,"
we proceed to the full Phase A-E build. If "weak," we apply the Gate-1 pivots
(bigger/more-diverse distill set, 5M student, SigLIP2 teacher) and re-test. If "no,"
we pivot the headline (few-shot / retention story) BEFORE sinking 8 more weeks.

It is intentionally SMALL (a few thousand images, a single Kaggle GPU session, <2h).
It is a spike, not the final experiment — it uses LIGHT keyword-based symptom descriptors
as a cheap stand-in (bare class names leave even the teacher at ~chance); the full
source-grounded descriptors come in Phase A/C.

-------------------------------------------------------------------------------
HOW TO RUN ON KAGGLE
-------------------------------------------------------------------------------
1. New Notebook -> Settings:  Accelerator = GPU T4 x1 (or P100),  Internet = ON.
2. Upload this file (or paste cells). Then in a cell:  %run phase0_spike.py
   (or just run it as a script).
3. First real run uses the defaults below. To go faster/cheaper, lower ROW_LIMIT
   and the per-class caps.
4. Read the RESULT BLOCK printed at the end + the saved /kaggle/working/phase0_result.json.

Internet must be ON: we download a few SAGE parquet SHARDS from HuggingFace (never the
full ~133GB) and filter them locally with pyarrow -- row-by-row streaming is ~2.5s/row
(139h ETA) and unusable. We also pull CLIP / timm weights.
"""
from __future__ import annotations
import io, json, time, hashlib, random, sys
from collections import defaultdict, Counter
from pathlib import Path

# --------------------------------------------------------------------------- #
# CONFIG  (edit these for the spike; everything else flows from here)
# --------------------------------------------------------------------------- #
SPIKE_TRAIN_CROPS   = ["Tomato", "Apple", "Corn", "Grape", "Potato"]  # distilled on
SPIKE_HELDOUT_CROPS = ["Coffee", "Orange", "Peach"]  # NEVER trained -> zero-shot test (need >=2 to survive)
MIN_HELDOUT_CROPS   = 2                               # HARD requirement: a valid spike tests >=2 unseen crops
MIN_TRAIN_CROPS     = 4                               # need >=4 train crops for a fair cross-crop test
#                                                       (SAGE shards are crop-clustered -> pull until covered)

ROW_LIMIT            = 200_000   # max SAGE rows to stream (raise if held-out crops come up short)
CAP_TRAIN_PER_CLASS  = 250       # distill images per (train crop, disease)
CAP_HELD_PER_CLASS   = 120       # eval images per (held-out crop, disease)
MIN_CLASS_IMAGES     = 15        # ignore ultra-rare classes (lowered so a 2nd held-out crop survives)

# Data access: row-by-row HF streaming is ~2.5s/row (139h ETA) -> UNUSABLE. Instead we
# download a few auto-converted parquet SHARDS locally and filter with pyarrow. SAGE has
# 13 shards (0000..0012); 0000 is smallest (~264MB). We pull shards in this order and
# auto-stop as soon as every crop is covered, so we rarely fetch more than 1-2.
SHARD_ORDER          = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # all 13; incremental, auto-stops
MAX_SHARDS           = 13                    # hard cap on shards to download
MIN_KEPT_TO_STOP     = 1500                  # stop once this many imgs + >=4 train & >=2 held crops present

EPOCHS               = 12
BATCH                = 64
LR                   = 1e-3
MIMIC_W              = 1.0       # weight: student image-emb mimics teacher image-emb (cosine)
ANCHOR_W             = 1.0       # weight: student image-emb -> correct TRAIN text prototype (zero-shot driver)
TAU                  = 0.07      # temperature for the text-anchoring contrastive loss
# Frozen teacher: try SigLIP2 (stronger) first, fall back to SigLIP, then CLIP (always available).
TEACHER_CANDIDATES = [
    ("ViT-B-16-SigLIP2", "webli"),
    ("ViT-B-16-SigLIP2-256", "webli"),
    ("ViT-B-16-SigLIP", "webli"),
    ("ViT-B-16", "openai"),
]
SEED                 = 42

OUT_DIR  = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("./phase0_out")
DATA_DIR = OUT_DIR / "spike_data"
RESULT_JSON = OUT_DIR / "phase0_result.json"

# SAGE crop strings vary in case/spelling -> canonical name.
CROP_ALIASES = {
    "tomato": "Tomato", "apple": "Apple",
    "corn": "Corn", "maize": "Corn", "corn (maize)": "Corn",
    "grape": "Grape", "grapevine": "Grape", "potato": "Potato",
    "coffee": "Coffee", "peach": "Peach",
    "orange": "Orange", "citrus": "Orange", "pumpkin": "Pumpkin",
    "soybean": "Soybean", "rice": "Rice",
}
WANT = set(SPIKE_TRAIN_CROPS) | set(SPIKE_HELDOUT_CROPS)


def canonical_crop(raw):
    if not raw:
        return None
    k = str(raw).strip().lower()
    if k in CROP_ALIASES:
        c = CROP_ALIASES[k]
    else:
        c = next((v for a, v in CROP_ALIASES.items() if a in k), None)
    return c if c in WANT else None


def safe(s):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(s).strip())


# --------------------------------------------------------------------------- #
# STAGE 1 — stream + filter SAGE (no 114GB download)
# --------------------------------------------------------------------------- #
def _load_existing():
    """Reconstruct rows from JPEGs already on disk (resume after a late crash)."""
    if not DATA_DIR.exists():
        return []
    rows = []
    for cls_dir in DATA_DIR.iterdir():
        if not cls_dir.is_dir() or "___" not in cls_dir.name:
            continue
        crop, disease = cls_dir.name.split("___", 1)
        role = "heldout" if crop in SPIKE_HELDOUT_CROPS else "train"
        for jpg in cls_dir.glob("*.jpg"):
            rows.append({"path": str(jpg), "crop": crop, "disease": disease, "role": role})
    return rows


def _load_done():
    p = DATA_DIR / ".shards_done.json"
    if p.exists():
        try:
            return set(json.loads(p.read_text()))
        except Exception:
            return set()
    return set()


def _save_done(done):
    (DATA_DIR / ".shards_done.json").write_text(json.dumps(sorted(done)))


def _finalize(rows):
    """Drop ultra-rare classes, print the per-crop summary, and HARD-GUARD on >=2 held crops."""
    cc = Counter((r["crop"], r["disease"]) for r in rows)
    rows = [r for r in rows if cc[(r["crop"], r["disease"])] >= MIN_CLASS_IMAGES]
    by_crop = Counter(r["crop"] for r in rows)
    print(f"    final kept={len(rows):,}")
    for c in sorted(WANT):
        tag = "HELDOUT" if c in SPIKE_HELDOUT_CROPS else "train"
        print(f"      {c:<9} [{tag:<7}] {by_crop.get(c,0):,}")
    held_present = [c for c in SPIKE_HELDOUT_CROPS if by_crop.get(c, 0) > 0]
    if len(held_present) < MIN_HELDOUT_CROPS:
        sys.exit(
            f"\nINVALID SPIKE: need >= {MIN_HELDOUT_CROPS} held-out crops with surviving classes; "
            f"got {held_present}.\n  per-crop found: {dict(by_crop)}\n"
            f"  FIX: raise MAX_SHARDS / extend SHARD_ORDER to later shards, lower MIN_CLASS_IMAGES, "
            f"or swap held-out crops to denser ones.")
    return rows


def build_subset():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from tqdm.auto import tqdm

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # RESUME + INCREMENTAL: seed state from disk and only download shards not done yet, so
    # adding more train crops never re-pulls shards we already processed.
    rows = _load_existing()
    done = _load_done()
    kept = Counter((r["crop"], r["disease"]) for r in rows)      # per-class counts so far
    hashes = {Path(r["path"]).stem for r in rows}                # 16-char stems for dedup
    stats = Counter()

    def covered():
        by = Counter(r["crop"] for r in rows)
        ntrain = sum(1 for c in SPIKE_TRAIN_CROPS if by.get(c, 0) >= MIN_CLASS_IMAGES)
        nheld = sum(1 for c in SPIKE_HELDOUT_CROPS if by.get(c, 0) >= MIN_CLASS_IMAGES)
        return ntrain >= MIN_TRAIN_CROPS and nheld >= MIN_HELDOUT_CROPS and len(rows) >= MIN_KEPT_TO_STOP

    if rows:
        by = Counter(r["crop"] for r in rows)
        print(f"[1] resume: {len(rows):,} imgs on disk, shards done={sorted(done)}  "
              + ", ".join(f"{c}={by.get(c,0)}" for c in sorted(WANT)))
    if covered():
        print(f"[1] on-disk data already covers >= {MIN_TRAIN_CROPS} train + {MIN_HELDOUT_CROPS} held -> skip download.")
        return _finalize(rows)

    print(f"[1] SAGE via parquet shards (train {SPIKE_TRAIN_CROPS}, held-out {SPIKE_HELDOUT_CROPS}) ...")
    for si in SHARD_ORDER[:MAX_SHARDS]:
        if si in done:
            continue
        fn = f"default/train/{si:04d}.parquet"
        print(f"    downloading shard {si:04d} ...")
        try:
            path = hf_hub_download(repo_id="tirtho149/SAGE", repo_type="dataset",
                                   filename=fn, revision="refs/convert/parquet")
        except Exception as e:
            print(f"    !! could not fetch shard {si:04d}: {e}")
            continue
        pf = pq.ParquetFile(path)
        try:
            names = set(pf.schema_arrow.names)
            cols = [c for c in ("image", "crop", "disease") if c in names]
        except Exception:
            cols = None  # read all columns
        nrows = pf.metadata.num_rows
        print(f"    reading {nrows:,} rows ...")
        for batch in tqdm(pf.iter_batches(batch_size=512, columns=cols),
                          total=nrows // 512 + 1, desc=f"shard{si:04d}"):
            d = batch.to_pydict()
            imgs = d.get("image", [])
            crops = d.get("crop", [])
            diss = d.get("disease", [None] * len(crops))
            for img_obj, craw, draw in zip(imgs, crops, diss):
                crop = canonical_crop(craw)
                if crop is None:
                    continue
                held = crop in SPIKE_HELDOUT_CROPS
                cap = CAP_HELD_PER_CLASS if held else CAP_TRAIN_PER_CLASS
                disease = str(draw if draw is not None else "Unknown")
                key = (crop, disease)
                if kept[key] >= cap:
                    continue
                try:
                    raw = img_obj["bytes"] if isinstance(img_obj, dict) else img_obj
                    img = Image.open(io.BytesIO(raw)).convert("RGB")
                    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=92)
                    jpg = buf.getvalue()
                except Exception:
                    stats["unreadable"] += 1
                    continue
                h16 = hashlib.sha1(jpg).hexdigest()[:16]
                if h16 in hashes:
                    stats["dup"] += 1
                    continue
                hashes.add(h16)
                cls = DATA_DIR / f"{safe(crop)}___{safe(disease)}"
                cls.mkdir(parents=True, exist_ok=True)
                fp = cls / f"{h16}.jpg"
                fp.write_bytes(jpg)
                kept[key] += 1
                stats["kept"] += 1
                rows.append({"path": str(fp), "crop": crop, "disease": disease,
                             "role": "heldout" if held else "train"})
        # free disk: drop the parquet we just consumed (best effort)
        try:
            Path(path).unlink()
        except Exception:
            pass
        done.add(si); _save_done(done)
        by = Counter(r["crop"] for r in rows)
        print(f"    after shard {si:04d}: total={len(rows):,}  "
              + ", ".join(f"{c}={by.get(c,0)}" for c in sorted(WANT)))
        if covered():
            print(f"    >= {MIN_TRAIN_CROPS} train + {MIN_HELDOUT_CROPS} held crops covered -> stop.")
            break

    print(f"    new this run: kept={stats['kept']:,}  dup={stats['dup']:,}  unreadable={stats['unreadable']:,}")
    return _finalize(rows)


# --------------------------------------------------------------------------- #
# STAGE 2 — distill student, then ZERO-SHOT eval vs teacher upper-bound vs chance
# --------------------------------------------------------------------------- #
def run_experiment(rows):
    import torch, torch.nn as nn, torch.nn.functional as Fn
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    import timm, open_clip

    random.seed(SEED); torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[2] device={dev}"
          + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else " (CPU)"))

    # frozen CLIP teacher
    teacher = preprocess = tok = teacher_name = None
    for nm, pre in TEACHER_CANDIDATES:
        try:
            teacher, _, preprocess = open_clip.create_model_and_transforms(nm, pretrained=pre)
            tok = open_clip.get_tokenizer(nm)
            teacher_name = f"{nm}/{pre}"
            break
        except Exception as e:
            print(f"    teacher {nm}/{pre} unavailable ({type(e).__name__}) -> next")
    assert teacher is not None, "no teacher loaded; check open_clip version / pretrained tags"
    teacher.eval().to(dev)
    for p in teacher.parameters():
        p.requires_grad = False
    with torch.no_grad():
        tdim = teacher.encode_text(tok(["a leaf"]).to(dev)).shape[-1]
    print(f"    teacher: {teacher_name}  (embed dim {tdim})")

    # student: a ~5M-tier edge backbone (matches the new tier-1) + projection head into
    # teacher space. Try the paper's tier-1 (EMOv2-5M) first, then reliable ~5M timm
    # fallbacks, then the old 1.3M floor — so this RUNS regardless of Kaggle's timm build.
    STUDENT_CANDIDATES = [
        "emov2_5m",                  # paper tier-1, if this timm build registers it
        "edgenext_small.usi_in1k",   # ~5.6M, reliably in timm
        "edgenext_small",
        "tiny_vit_5m_224.in1k",      # ~5.4M
        "efficientvit_b1.r224_in1k", # ~9M, hybrid
        "edgenext_xx_small.in1k",    # ~1.3M last-resort (old floor)
        "edgenext_xx_small",
    ]
    backbone = student_name = None
    for n in STUDENT_CANDIDATES:
        try:
            backbone = timm.create_model(n, pretrained=True, num_classes=0)
            student_name = n
            break
        except Exception:
            continue
    assert backbone is not None, "no candidate student backbone loaded; upgrade timm (>=1.0.3)"
    student = nn.Sequential(backbone, nn.Linear(backbone.num_features, tdim)).to(dev)
    nparams = sum(p.numel() for p in student.parameters())
    print(f"    student: {student_name}  ({nparams/1e6:.2f}M params)  -> teacher dim {tdim}")

    class DS(Dataset):
        def __init__(self, rs): self.rs = rs
        def __len__(self): return len(self.rs)
        def __getitem__(self, i):
            r = self.rs[i]
            return preprocess(Image.open(r["path"]).convert("RGB")), \
                   f'{r["crop"]}|{r["disease"]}'

    train_rows = [r for r in rows if r["role"] == "train"]
    held_rows  = [r for r in rows if r["role"] == "heldout"]
    assert train_rows and held_rows, "need both train and held-out images"
    tdl = DataLoader(DS(train_rows), batch_size=BATCH, shuffle=True, num_workers=2)
    hdl = DataLoader(DS(held_rows),  batch_size=BATCH, num_workers=2)

    # ---- descriptor text prototypes (LIGHT keyword proxy for Phase-A source-grounded ones) ----
    SYMPTOM_HINTS = {
        "rust": "orange to brown powdery pustules on the underside of the leaf",
        "blight": "rapidly spreading brown necrotic lesions and dead leaf tissue",
        "spot": "small dark circular spots with concentric rings on the leaf",
        "mildew": "a white or grey powdery fungal coating on the leaf surface",
        "canker": "sunken corky lesions with yellow halos on the leaf",
        "greening": "blotchy asymmetric yellow mottling of the leaf",
        "huanglongbing": "blotchy asymmetric yellow mottling of the leaf",
        "curl": "puckered, thickened, distorted and reddened curled leaves",
        "brown rot": "brown spreading rot with tan fungal spore masses",
        "scab": "olive-green to black velvety scabby lesions on the leaf",
        "mosaic": "a mottled light-and-dark green mosaic pattern on the leaf",
        "cercospora": "brown spots with grey centers and yellow halos on the leaf",
        "deficiency": "interveinal yellowing of the leaf from nutrient deficiency",
        "healthy": "a healthy green leaf with no disease symptoms",
    }
    templates = ["a photo of {}", "a close-up leaf photo: {}", "a leaf with {}"]
    def descriptor_text(lbl):
        crop, dis = lbl.split("|")
        k = dis.lower()
        hint = next((v for kw, v in SYMPTOM_HINTS.items() if kw in k), "")
        base = f"{dis} on {crop} leaf".replace("_", " ")
        return f"{base}: {hint}" if hint else base
    def build_protos(classes):
        out = []
        with torch.no_grad():
            for lbl in classes:
                toks = tok([t.format(descriptor_text(lbl)) for t in templates]).to(dev)
                emb = Fn.normalize(teacher.encode_text(toks), dim=-1).mean(0)
                out.append(Fn.normalize(emb, dim=-1))
        return torch.stack(out).to(dev)

    # TRAIN-class text prototypes -> ANCHOR the student in the text space. This is the half
    # the previous spike was missing: plain image-mimicry on 5 crops does NOT transfer zero-shot.
    train_classes = sorted({f'{r["crop"]}|{r["disease"]}' for r in train_rows})
    train_idx = {c: i for i, c in enumerate(train_classes)}
    train_protos = build_protos(train_classes)      # [Ctrain, dim]

    # ---- distill: image-mimic (cosine to teacher) + text-anchor (contrastive to train protos) ----
    opt = torch.optim.AdamW(student.parameters(), lr=LR)
    print(f"[3] distilling {len(train_rows):,} imgs x {EPOCHS} epochs "
          f"(mimic_w={MIMIC_W}, anchor_w={ANCHOR_W}, tau={TAU}) ...")
    for ep in range(EPOCHS):
        student.train(); run = run_m = run_a = 0.0; t0 = time.time()
        for imgs, labels in tdl:
            imgs = imgs.to(dev)
            with torch.no_grad():
                t_emb = Fn.normalize(teacher.encode_image(imgs), dim=-1)
            s_emb = Fn.normalize(student(imgs), dim=-1)
            mimic = (1 - (s_emb * t_emb).sum(-1)).mean()
            tgt = torch.tensor([train_idx[l] for l in labels], device=dev)
            anchor = Fn.cross_entropy((s_emb @ train_protos.T) / TAU, tgt)
            loss = MIMIC_W * mimic + ANCHOR_W * anchor
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item(); run_m += mimic.item(); run_a += anchor.item()
        n = len(tdl)
        print(f"    epoch {ep+1:2d}/{EPOCHS}  loss={run/n:.4f}  mimic={run_m/n:.4f}  "
              f"anchor={run_a/n:.4f}  ({time.time()-t0:.0f}s)")

    # ---- text prototypes for HELD-OUT classes (zero-shot eval targets) ----
    held_classes = sorted({f'{r["crop"]}|{r["disease"]}' for r in held_rows})
    protos = build_protos(held_classes)             # [C, dim]
    chance = 1.0 / len(held_classes)

    # ---- evaluate: teacher (upper bound) vs student (the real question) ----
    def zero_shot(encode):
        correct = total = 0
        per_crop = defaultdict(lambda: [0, 0])
        with torch.no_grad():
            for imgs, labels in hdl:
                emb = Fn.normalize(encode(imgs.to(dev)), dim=-1)
                pred = (emb @ protos.T).argmax(1).cpu().tolist()
                for p, gt in zip(pred, labels):
                    ok = held_classes[p] == gt
                    correct += ok; total += 1
                    crop = gt.split("|")[0]
                    per_crop[crop][0] += ok; per_crop[crop][1] += 1
        return correct / total, {c: a / n for c, (a, n) in per_crop.items()}

    print("[4] zero-shot eval on held-out crops ...")
    student.eval()
    t_acc, t_by = zero_shot(teacher.encode_image)
    s_acc, s_by = zero_shot(lambda x: student(x))
    retention = s_acc / t_acc if t_acc > 0 else 0.0

    result = {
        "spike_train_crops": SPIKE_TRAIN_CROPS,
        "spike_heldout_crops": SPIKE_HELDOUT_CROPS,
        "n_train_imgs": len(train_rows),
        "n_heldout_imgs": len(held_rows),
        "n_heldout_classes": len(held_classes),
        "n_heldout_crops": len({c.split("|")[0] for c in held_classes}),
        "student_backbone": student_name,
        "student_params_M": round(nparams / 1e6, 3),
        "chance_acc": chance,
        "teacher_zeroshot_acc": t_acc,
        "student_zeroshot_acc": s_acc,
        "retention_student_over_teacher": retention,
        "teacher_by_crop": t_by,
        "student_by_crop": s_by,
        "config": {"epochs": EPOCHS, "batch": BATCH, "lr": LR, "teacher": teacher_name,
                   "mimic_w": MIMIC_W, "anchor_w": ANCHOR_W, "tau": TAU,
                   "cap_train": CAP_TRAIN_PER_CLASS, "cap_held": CAP_HELD_PER_CLASS,
                   "row_limit": ROW_LIMIT},
    }
    RESULT_JSON.write_text(json.dumps(result, indent=2))
    return result


def verdict(r):
    chance = r["chance_acc"]; s = r["student_zeroshot_acc"]; t = r["teacher_zeroshot_acc"]
    ret = r["retention_student_over_teacher"]
    teacher_signal = (t - chance) >= 0.08     # is there ANY zero-shot in the TEACHER to retain?
    student_signal = (s - chance) >= 0.08
    if not teacher_signal:
        return "INVALID", ("the TEACHER itself is ~chance even with descriptor text -> there is no "
                           "zero-shot signal to distill. This is NOT a method NO-GO. Use a stronger/"
                           "domain teacher (SigLIP2 / BioCLIP2 / SCOLD) or richer source-grounded "
                           "descriptors, then re-spike.")
    if student_signal and ret >= 0.70:
        return "GO", "student preserves the teacher's descriptor zero-shot at 5M -> proceed to Phase A-E."
    if student_signal and ret >= 0.45:
        return "WEAK", ("beats chance but loses too much vs a teacher that HAS signal -> Gate-1 pivots "
                        "(stronger teacher, more/better descriptors) and re-spike.")
    return "NO-GO", ("student collapses despite a teacher with real signal -> the distillation recipe "
                     "needs work (loss, curriculum, longer training) before the full build.")


def ensure_deps():
    """Self-contained: install the few deps not always on the Kaggle image, BEFORE the
    expensive data download, so a missing module never crashes us mid-run."""
    import importlib, subprocess
    need = []
    for mod, pkg in [("datasets", "datasets>=2.19"), ("open_clip", "open_clip_torch>=2.24"),
                     ("timm", "timm>=1.0.3"), ("pyarrow", "pyarrow"),
                     ("huggingface_hub", "huggingface_hub")]:
        try:
            importlib.import_module(mod)
        except Exception:
            need.append(pkg)
    if need:
        print(f"[0] installing missing deps: {need}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *need], check=True)
        print("[0] deps installed.")


def main():
    ensure_deps()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = build_subset()
    res = run_experiment(rows)
    tag, msg = verdict(res)

    print("\n" + "=" * 64)
    print("PHASE 0 SPIKE — RESULT")
    print("=" * 64)
    print(f"  held-out crops      : {res['spike_heldout_crops']}  "
          f"({res['n_heldout_crops']} crops, {res['n_heldout_classes']} classes, {res['n_heldout_imgs']:,} imgs)")
    print(f"  student params      : {res['student_params_M']}M")
    print(f"  chance accuracy     : {res['chance_acc']:.1%}")
    print(f"  TEACHER zero-shot   : {res['teacher_zeroshot_acc']:.1%}   (upper bound)")
    print(f"  STUDENT zero-shot   : {res['student_zeroshot_acc']:.1%}   <-- the answer")
    print(f"  retention (s/t)     : {res['retention_student_over_teacher']:.1%}")
    print(f"  per-crop (student)  : " +
          ", ".join(f"{c}={a:.0%}" for c, a in res["student_by_crop"].items()))
    print("-" * 64)
    print(f"  VERDICT: {tag}")
    print(f"  {msg}")
    print(f"  ({time.time()-t0:.0f}s total; result saved to {RESULT_JSON})")
    print("=" * 64)


if __name__ == "__main__":
    main()
