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
It is a spike, not the final experiment — naive class-name text prototypes are fine
here; source-grounded descriptors come in Phase A/C.

-------------------------------------------------------------------------------
HOW TO RUN ON KAGGLE
-------------------------------------------------------------------------------
1. New Notebook -> Settings:  Accelerator = GPU T4 x1 (or P100),  Internet = ON.
2. Upload this file (or paste cells). Then in a cell:  %run phase0_spike.py
   (or just run it as a script).
3. First real run uses the defaults below. To go faster/cheaper, lower ROW_LIMIT
   and the per-class caps.
4. Read the RESULT BLOCK printed at the end + the saved /kaggle/working/phase0_result.json.

Internet must be ON: we STREAM SAGE from HuggingFace (never download the 114GB) and
pull CLIP / EdgeNeXt weights.
"""
from __future__ import annotations
import io, json, time, hashlib, random
from collections import defaultdict, Counter
from pathlib import Path

# --------------------------------------------------------------------------- #
# CONFIG  (edit these for the spike; everything else flows from here)
# --------------------------------------------------------------------------- #
SPIKE_TRAIN_CROPS   = ["Tomato", "Apple", "Corn", "Grape", "Potato"]  # distilled on
SPIKE_HELDOUT_CROPS = ["Coffee", "Peach"]                             # NEVER trained -> zero-shot test

ROW_LIMIT            = 200_000   # max SAGE rows to stream (raise if held-out crops come up short)
CAP_TRAIN_PER_CLASS  = 250       # distill images per (train crop, disease)
CAP_HELD_PER_CLASS   = 120       # eval images per (held-out crop, disease)
MIN_CLASS_IMAGES     = 25        # ignore ultra-rare classes in the spike

EPOCHS               = 12
BATCH                = 64
LR                   = 1e-3
TEACHER              = ("ViT-B-16", "openai")   # frozen CLIP teacher (image+text aligned)
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
def build_subset():
    from datasets import load_dataset
    from PIL import Image
    from tqdm.auto import tqdm

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    kept = defaultdict(int)
    hashes = set()
    rows = []
    stats = Counter()

    print(f"[1] streaming {ROW_LIMIT:,} rows max from tirtho149/SAGE (train {SPIKE_TRAIN_CROPS}, "
          f"held-out {SPIKE_HELDOUT_CROPS}) ...")
    ds = load_dataset("tirtho149/SAGE", split="train", streaming=True)
    for i, row in enumerate(tqdm(ds, total=ROW_LIMIT, desc="scan")):
        if i >= ROW_LIMIT:
            break
        crop = canonical_crop(row.get("crop"))
        if crop is None:
            continue
        held = crop in SPIKE_HELDOUT_CROPS
        cap = CAP_HELD_PER_CLASS if held else CAP_TRAIN_PER_CLASS
        disease = str(row.get("disease", "Unknown"))
        key = (crop, disease)
        if kept[key] >= cap:
            continue
        try:
            img = row.get("image")
            if isinstance(img, dict) and img.get("bytes"):
                img = Image.open(io.BytesIO(img["bytes"]))
            img = img.convert("RGB")
            buf = io.BytesIO(); img.save(buf, format="JPEG", quality=92)
            jpg = buf.getvalue()
        except Exception:
            stats["unreadable"] += 1
            continue
        h = hashlib.sha1(jpg).hexdigest()
        if h in hashes:
            stats["dup"] += 1
            continue
        hashes.add(h)
        cls = DATA_DIR / f"{safe(crop)}___{safe(disease)}"
        cls.mkdir(parents=True, exist_ok=True)
        fn = f"{h[:16]}.jpg"
        (cls / fn).write_bytes(jpg)
        kept[key] += 1
        stats["kept"] += 1
        rows.append({"path": str(cls / fn), "crop": crop, "disease": disease,
                     "role": "heldout" if held else "train"})

    # drop ultra-rare classes
    cc = Counter((r["crop"], r["disease"]) for r in rows)
    rows = [r for r in rows if cc[(r["crop"], r["disease"])] >= MIN_CLASS_IMAGES]
    print(f"    kept={stats['kept']:,}  dup={stats['dup']:,}  unreadable={stats['unreadable']:,}")
    by_crop = Counter(r["crop"] for r in rows)
    for c in sorted(WANT):
        tag = "HELDOUT" if c in SPIKE_HELDOUT_CROPS else "train"
        print(f"      {c:<9} [{tag:<7}] {by_crop.get(c,0):,}")
    missing = [c for c in WANT if by_crop.get(c, 0) == 0]
    if missing:
        print(f"    WARNING: 0 images for {missing} -> raise ROW_LIMIT or check CROP_ALIASES.")
    return rows


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
    name, pre = TEACHER
    teacher, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pre)
    tok = open_clip.get_tokenizer(name)
    teacher.eval().to(dev)
    for p in teacher.parameters():
        p.requires_grad = False
    tdim = teacher.text_projection.shape[1] if hasattr(teacher, "text_projection") else 512

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

    # ---- distill: student image emb -> teacher image emb (cosine) ----
    opt = torch.optim.AdamW(student.parameters(), lr=LR)
    print(f"[3] distilling {len(train_rows):,} imgs x {EPOCHS} epochs ...")
    for ep in range(EPOCHS):
        student.train(); run = 0.0; t0 = time.time()
        for imgs, _ in tdl:
            imgs = imgs.to(dev)
            with torch.no_grad():
                t_emb = Fn.normalize(teacher.encode_image(imgs), dim=-1)
            s_emb = Fn.normalize(student(imgs), dim=-1)
            loss = (1 - (s_emb * t_emb).sum(-1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item()
        print(f"    epoch {ep+1:2d}/{EPOCHS}  distill_loss={run/len(tdl):.4f}  ({time.time()-t0:.0f}s)")

    # ---- text prototypes for HELD-OUT classes (naive class-name ensemble) ----
    held_classes = sorted({f'{r["crop"]}|{r["disease"]}' for r in held_rows})
    templates = ["a photo of {} leaf", "a leaf showing {}", "{} symptoms on a leaf",
                 "a close-up photo of {}"]
    def class_phrase(lbl):
        crop, dis = lbl.split("|")
        return f"{dis} on {crop}".replace("_", " ")
    protos = []
    with torch.no_grad():
        for lbl in held_classes:
            phrase = class_phrase(lbl)
            toks = tok([t.format(phrase) for t in templates]).to(dev)
            emb = Fn.normalize(teacher.encode_text(toks), dim=-1).mean(0)
            protos.append(Fn.normalize(emb, dim=-1))
    protos = torch.stack(protos).to(dev)            # [C, dim]
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
        "student_backbone": student_name,
        "student_params_M": round(nparams / 1e6, 3),
        "chance_acc": chance,
        "teacher_zeroshot_acc": t_acc,
        "student_zeroshot_acc": s_acc,
        "retention_student_over_teacher": retention,
        "teacher_by_crop": t_by,
        "student_by_crop": s_by,
        "config": {"epochs": EPOCHS, "batch": BATCH, "lr": LR, "teacher": TEACHER,
                   "cap_train": CAP_TRAIN_PER_CLASS, "cap_held": CAP_HELD_PER_CLASS,
                   "row_limit": ROW_LIMIT},
    }
    RESULT_JSON.write_text(json.dumps(result, indent=2))
    return result


def verdict(r):
    chance, s, t = r["chance_acc"], r["student_zeroshot_acc"], r["teacher_zeroshot_acc"]
    ret = r["retention_student_over_teacher"]
    above_chance = s >= 3 * chance
    good_retention = ret >= 0.70
    if above_chance and good_retention:
        return "GO", "student preserves teacher zero-shot at edge scale -> proceed to full Phase A-E."
    if above_chance and ret >= 0.45:
        return "WEAK", ("clearly beats chance but loses too much vs teacher -> apply Gate-1 pivots "
                        "(bigger/more-diverse distill set, 5M student, SigLIP2) and re-spike.")
    return "NO-GO", ("mechanism does not survive distillation here -> pivot the headline "
                     "(few-shot cross-crop, or retention/efficiency story) before the full build.")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = build_subset()
    res = run_experiment(rows)
    tag, msg = verdict(res)

    print("\n" + "=" * 64)
    print("PHASE 0 SPIKE — RESULT")
    print("=" * 64)
    print(f"  held-out crops      : {res['spike_heldout_crops']}  "
          f"({res['n_heldout_classes']} classes, {res['n_heldout_imgs']:,} imgs)")
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
