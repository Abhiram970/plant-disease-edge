"""
=====================================================================================
 PDE REVISION RE-SCORE  --  paste THIS one cell into Kaggle, set nothing, run.
=====================================================================================
Phase 2 of the 2026-09-22 deep review. It embeds the held-out images ONCE per encoder
and reuses that pass for every configuration, label set and strategy, then answers:

  1. shortlist and abstention for CuPL+ (top-5, AURC, acc@cov80) -- measured for the
     registry the paper recommends, not only for grounded and rich;
  2. the majority-class prior and class-macro accuracy at EVERY configuration;
  3. the taxonomy test: grounded_split vs grounded_visual_split, same source text and
     same construction, with the pathogen/taxonomy prose removed;
  4. the organ-rule bound: the CuPL+ minus grounded gap on foliar classes only, where
     the image-informed plant-part instruction is inert.

SETUP
  Add Data     -> pde-sage-data   (the 288 px exp_data build -- NOT the 448 px one)
  Accelerator  -> GPU T4 x2 or P100
  Internet     -> ON  (for the clone and the pip install; no HF token needed, the
                       images come from the attached dataset)
  Save Version -> "Save & Run All" so the run survives closing the tab

  No API key. No descriptor generation. Nothing is trained. ~30-60 min on a T4.

OUTPUT
  /kaggle/working/rescore_out/revision_numbers.json
  /kaggle/working/rescore_out/metrics_abstain_{A,B,C}.json
  /kaggle/working/pde_descriptor_rescore.zip        <- download this, unzip into docs/paper/

THEN, LOCALLY
  copy revision_numbers.json          -> docs/paper/manuscript/revision_numbers.json
  copy metrics_abstain_*.json     -> docs/paper/manuscript/revision_results/
  python docs/paper/make_tex_tables_revision.py
  python docs/paper/make_figures_revision.py
  python scripts/paper_fixes/apply_revision_results.py
  cd docs/paper/manuscript/revision && latexmk -pdf main.tex
=====================================================================================
"""

# ---- settings ---------------------------------------------------------------------
REPO_URL = "https://github.com/Abhiram970/plant-disease-edge.git"
REPO_REF = "baselines/dclip-cupl-manual"   # carries descriptors_cupl/, descriptors_dclip/
MODELS = ["s0", "s1", "s2", "b"]           # the four deployable encoders
REFERENCE = False                          # True also scores the SigLIP2 ceiling
EXPS = ["A", "B", "C"]
STRATEGIES = ["bare", "bare80", "crude", "rich", "grounded", "grounded_split",
              "grounded_visual", "grounded_visual_split", "dclip", "cupl"]

import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

T0 = time.time()
# Env overrides, for a quick smoke test before committing a 12 h session to it:
#   PDE_REVISION_MODELS=s0 PDE_REVISION_EXPS=A python scripts/kaggle/runners/rescore_descriptors.py
MODELS = os.environ.get("PDE_REVISION_MODELS", " ".join(MODELS)).split()
EXPS = os.environ.get("PDE_REVISION_EXPS", " ".join(EXPS)).split()
# Kaggle sets these; a bare Path("/kaggle").exists() is true on any Windows box that happens to
# have C:\kaggle, which made a local test clone the repo and write into C:\kaggle\working.
ON_KAGGLE = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE")) \
    and Path("/kaggle/working").exists()
WORK = Path("/kaggle/working") if ON_KAGGLE else Path.cwd()
OUT = WORK / "rescore_out"
OUT.mkdir(parents=True, exist_ok=True)


def banner(msg):
    print(f"\n{'=' * 84}\n[rescore t+{(time.time() - T0) / 60:5.1f}m] {msg}\n{'=' * 84}", flush=True)


# ---- 1. code ----------------------------------------------------------------------
if ON_KAGGLE:
    CODE = WORK / "_pde_code"
    if CODE.exists():
        shutil.rmtree(CODE, ignore_errors=True)
    banner(f"cloning {REPO_REF}")
    rc = subprocess.run(["git", "clone", "--depth", "1", "--branch", REPO_REF, REPO_URL,
                         str(CODE)], capture_output=True, text=True)
    if rc.returncode != 0:
        sys.exit(f"[rescore] clone failed:\n{rc.stderr}")
    head = subprocess.run(["git", "-C", str(CODE), "log", "--oneline", "-1"],
                          capture_output=True, text=True).stdout.strip()
    print(f"[rescore] HEAD = {head}", flush=True)
else:
    # Pasted into a notebook there is no __file__, so find the repo by walking up from
    # wherever we are until a checkout appears.
    here = Path(__file__).resolve() if "__file__" in globals() else Path.cwd().resolve()
    CODE = next((p for p in [here, *here.parents] if (p / "descriptors_cupl").is_dir()), here)
    print(f"[rescore] local run, repo = {CODE}", flush=True)

for need in ("descriptors_cupl", "descriptors_dclip", "descriptors"):
    if not (CODE / need).exists():
        sys.exit(f"[rescore] the clone has no {need}/ -- wrong branch? (REPO_REF={REPO_REF})")

sys.path.insert(0, str(CODE / "scripts"))

# ---- 2. deps ----------------------------------------------------------------------
try:
    import open_clip  # noqa: F401
    import timm  # noqa: F401
except Exception:
    banner("installing open_clip_torch / timm")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "open_clip_torch>=2.24", "timm>=1.0.3"], check=True)

# ---- 3. data ----------------------------------------------------------------------
# config.py reads these at import time, so they are set before the import below.
def find_exp_data():
    env = os.environ.get("PDE_DATASET_DIR")
    if env and Path(env).exists():
        return Path(env)
    for root in ("/kaggle/input", "/kaggle/working"):
        r = Path(root)
        if not r.exists():
            continue
        for depth in range(1, 5):
            for p in r.glob("/".join(["*"] * depth) + "/exp_data"):
                if p.is_dir():
                    return p
        if (r / "exp_data").is_dir():
            return r / "exp_data"
    return None


DATA = find_exp_data()
if DATA is None:
    sys.exit("[rescore] no exp_data/ found -- attach the pde-sage-data dataset")
os.environ["PDE_DATASET_DIR"] = str(DATA)
os.environ["PDE_DATA_ROOT"] = str(WORK)
print(f"[rescore] images: {DATA}", flush=True)

import torch  # noqa: E402
import config as C  # noqa: E402
import sage_data  # noqa: E402
import zeroshot  # noqa: E402
import descriptors as D  # noqa: E402
from metrics import topk_and_riskcoverage  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[rescore] device = {device}"
      + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""), flush=True)

# ---- 4. the taxonomy-test strategy ------------------------------------------------
# grounded_visual_split keeps ONLY the visual_symptoms field and ensembles it exactly as
# grounded_split ensembles the whole paragraph, so the pair differs in nothing but the
# pathogen/taxonomy prose. Defined here rather than in the clone so this cell runs against
# the published branch unchanged; if the branch already carries it, this is a no-op.
if "grounded_visual_split" not in D.STRATEGY_TEMPLATES:
    _orig_texts_for = D.texts_for

    def _texts_for(label, strategy="rich", coverage=None):
        if strategy == "grounded_visual_split":
            crop, dis = label.split("|", 1)
            base = f"{dis} on {crop} leaf".replace("_", " ")
            gv = D._grounded_visual(crop, dis)
            sents = D._split_sentences(gv) if gv else []
            if sents:
                if coverage is not None:
                    coverage[label] = "grounded_visual_split"
                return [f"{base}. {s}" for s in sents]
            return [D.text_for(label, "grounded_visual", coverage)]
        return _orig_texts_for(label, strategy, coverage)

    D.texts_for = _texts_for          # build_prototypes resolves this at call time
    D.STRATEGY_TEMPLATES["grounded_visual_split"] = ["{}"]
    print("[rescore] grounded_visual_split defined in-cell", flush=True)

NON_FOLIAR_PATTERNS = ["rot", "mould", "mold", "bunt", "smut", "scab", "blotch"]


def is_non_foliar(label):
    dis = label.split("|", 1)[1].lower()
    return any(p in dis for p in NON_FOLIAR_PATTERNS)


# ---- 5. rows, and the six (configuration, label set) views ------------------------
banner("loading held-out images")
rows_all = sage_data.fetch(C.HELDOUT_CROPS, sage_data.full_caps(),
                           min_held_crops=C.MIN_HELD_CROPS)
held = set().union(*(set(C.EXPERIMENTS[e]["held"]) for e in EXPS))
rows_all = [r for r in rows_all if r["crop"] in held]
if not rows_all:
    sys.exit("[rescore] no held-out images -- is the attached dataset the exp_data build?")

from PIL import Image  # noqa: E402

# The published cells were measured on the 288 px build. The 448 px build moves
# MobileCLIP2-S0 at configuration A by up to +0.4 points, which would silently contradict
# Tables 1 and 2. Downscaling only shrinks, so smaller originals are fine: this is an
# upper bound, not equality.
sizes = {max(Image.open(r["path"]).size) for r in rows_all[:25]}
if max(sizes) > 288:
    sys.exit(f"[rescore] WRONG IMAGE BUILD: sampled longest edges up to {max(sizes)}, expected "
             f"<= 288. Attach the 288 px pde-sage-data build, not the full-resolution one.")
print(f"[rescore] {len(rows_all):,} images, longest edge <= {max(sizes)}", flush=True)

idx_of_path = {r["path"]: i for i, r in enumerate(rows_all)}
views = {}
for exp in EXPS:
    base = [r for r in rows_all if r["crop"] in set(C.EXPERIMENTS[exp]["held"])]
    for labelset in ("uncleaned", "clean"):
        rows = base if labelset == "uncleaned" else C.clean_rows(base)[0]
        labels = [r["label"] for r in rows]
        classes = sorted(set(labels))
        counts = Counter(labels)
        views[(exp, labelset)] = {
            "idx": [idx_of_path[r["path"]] for r in rows],
            "labels": labels, "classes": classes,
            "n_classes": len(classes), "n_images": len(rows),
            "chance": 100.0 / len(classes),
            "majority_prior": 100.0 * max(counts.values()) / len(rows),
            "foliar_idx": [i for i, l in enumerate(labels) if not is_non_foliar(l)],
        }
        v = views[(exp, labelset)]
        print(f"[rescore] {exp}/{labelset:9s} {v['n_images']:6,} imgs  {v['n_classes']:3d} classes  "
              f"majority prior {v['majority_prior']:5.1f}%  foliar {len(v['foliar_idx']):,}",
              flush=True)

non_foliar = sorted({l for l in views[(EXPS[-1], "uncleaned")]["labels"] if is_non_foliar(l)})
print(f"[rescore] organ rule ACTIVE on {len(non_foliar)} classes: "
      + ", ".join(n.replace("|", "/") for n in non_foliar), flush=True)

# ---- 6. score ----------------------------------------------------------------------
models = C.resolve_models(MODELS, include_reference=REFERENCE)
out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "device": device,
       "matcher_normalised": True, "strategies": STRATEGIES,
       "non_foliar_classes": non_foliar,
       "majority_prior": {ls: {e: views[(e, ls)]["majority_prior"] for e in EXPS}
                          for ls in ("uncleaned", "clean")},
       "configs": {}, "per_model": {}}
abstain = {}

for name, pretrained in models:
    key = f"{name}/{pretrained}"
    banner(f"{name} ({pretrained})")
    t = time.time()
    model, preprocess, tok, params_m = zeroshot.load_model(name, pretrained, device)
    img_emb, emb_labels = zeroshot.embed_images(model, preprocess, rows_all, device)
    assert emb_labels == [r["label"] for r in rows_all], "embedding order drifted from rows"
    print(f"[rescore] {params_m:.1f}M  embedded {len(rows_all):,} images in {time.time() - t:.0f}s",
          flush=True)

    for (exp, labelset), v in views.items():
        emb = img_emb[torch.tensor(v["idx"])]
        labels, classes = v["labels"], v["classes"]
        foliar = torch.tensor(v["foliar_idx"]) if v["foliar_idx"] else None
        for strat in STRATEGIES:
            protos = D.build_prototypes(model, tok, classes, strat, device)
            sims = (emb.to(device) @ protos.T).cpu()
            gt = torch.tensor([classes.index(l) for l in labels])
            hit = (sims.argmax(1) == gt)
            per_class = {}
            for h, lab in zip(hit.tolist(), labels):
                a = per_class.setdefault(lab, [0, 0])
                a[0] += h
                a[1] += 1
            rec = {"micro": 100.0 * hit.float().mean().item(),
                   "macro": 100.0 * sum(a / n for a, n in per_class.values()) / len(per_class)}
            if foliar is not None:
                rec["micro_foliar"] = 100.0 * hit[foliar].float().mean().item()
            out["per_model"].setdefault(key, {}).setdefault(f"{exp}/{labelset}", {})[strat] = rec
            if labelset == "uncleaned" and strat in ("cupl", "grounded", "rich"):
                a = abstain.setdefault(exp, {"matcher_normalised": True,
                                             "n_classes": v["n_classes"],
                                             "chance": v["chance"] / 100.0,
                                             "n_images": v["n_images"], "models": {}})
                a["models"].setdefault(key, {"img_params_M": round(params_m, 2)})[strat] = \
                    topk_and_riskcoverage(sims, labels, classes)
        got = out["per_model"][key][f"{exp}/{labelset}"]
        print(f"[rescore]   {exp}/{labelset:9s} "
              + "  ".join(f"{s}={got[s]['micro']:.1f}" for s in ("grounded", "cupl")), flush=True)

    # Partial save after every encoder: a session that dies still leaves usable rows.
    (OUT / "revision_numbers.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

# ---- 7. means over the four deployable encoders -----------------------------------
deploy = [f"{n}/{p}" for n, p in (C.DEPLOY_MODELS[k] for k in C.DEPLOY_MODELS)
          if f"{n}/{p}" in out["per_model"]]
for exp in EXPS:
    for labelset in ("uncleaned", "clean"):
        v = views[(exp, labelset)]
        vk = f"{exp}/{labelset}"
        b = {"n_classes": v["n_classes"], "n_images": v["n_images"], "chance": v["chance"],
             "majority_prior": v["majority_prior"], "n_deployable": len(deploy),
             "means": {}, "macro_means": {}, "means_foliar": {}}
        for strat in STRATEGIES:
            vals = [out["per_model"][m][vk][strat] for m in deploy]
            if not vals:
                continue
            b["means"][strat] = sum(x["micro"] for x in vals) / len(vals)
            b["macro_means"][strat] = sum(x["macro"] for x in vals) / len(vals)
            if "micro_foliar" in vals[0]:
                b["means_foliar"][strat] = sum(x["micro_foliar"] for x in vals) / len(vals)
        m, mf = b["means"], b["means_foliar"]
        if {"grounded_split", "grounded_visual_split"} <= set(m):
            b["taxonomy_test"] = {
                "grounded_split": m["grounded_split"],
                "grounded_visual_split": m["grounded_visual_split"],
                "delta_taxonomy_removed": m["grounded_visual_split"] - m["grounded_split"],
                "cupl_minus_grounded_visual_split": m.get("cupl", 0) - m["grounded_visual_split"]}
        if {"cupl", "grounded"} <= set(m):
            b["organ_rule_bound"] = {
                "gap_all_images": m["cupl"] - m["grounded"],
                "gap_foliar_only": (mf.get("cupl", 0) - mf.get("grounded", 0)) if mf else None,
                "n_foliar_images": len(v["foliar_idx"])}
        out["configs"].setdefault(exp, {})[labelset] = b

(OUT / "revision_numbers.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
for exp, payload in abstain.items():
    (OUT / f"metrics_abstain_{exp}.json").write_text(json.dumps(payload, indent=2),
                                                     encoding="utf-8")
print(f"\n[rescore] wrote {len(list(OUT.glob('*.json')))} files to {OUT}", flush=True)

# ---- 8. what the run settles -------------------------------------------------------
banner("what this run settles")
for exp in EXPS:
    for labelset in ("uncleaned", "clean"):
        b = out["configs"][exp][labelset]
        t, o = b.get("taxonomy_test"), b.get("organ_rule_bound")
        print(f"  {exp}/{labelset:9s} majority prior {b['majority_prior']:5.1f}%   "
              f"CuPL+ micro {b['means'].get('cupl', float('nan')):5.1f}  "
              f"macro {b['macro_means'].get('cupl', float('nan')):5.1f}")
        if t:
            print(f"      taxonomy removed: grounded_split {t['grounded_split']:.1f} -> "
                  f"visual_split {t['grounded_visual_split']:.1f} "
                  f"({t['delta_taxonomy_removed']:+.1f} points)")
        if o and o["gap_foliar_only"] is not None:
            print(f"      CuPL+ - grounded: all {o['gap_all_images']:+.1f}, "
                  f"foliar only {o['gap_foliar_only']:+.1f} ({o['n_foliar_images']:,} images)")

zip_path = WORK / "pde_descriptor_rescore"
shutil.make_archive(str(zip_path), "zip", OUT)
print(f"\n[rescore] bundle: {zip_path}.zip "
      f"({Path(str(zip_path) + '.zip').stat().st_size / 1e6:.2f} MB)", flush=True)
print(f"[rescore] done in {(time.time() - T0) / 60:.1f} min", flush=True)
try:
    from IPython.display import FileLink, display
    display(FileLink("pde_descriptor_rescore.zip"))
except Exception:
    pass
