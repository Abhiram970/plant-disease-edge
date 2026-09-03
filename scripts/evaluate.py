"""
Evaluate — the end-to-end Phase-0/A driver (frozen model family x descriptor strategy).

Recreates the validated process in one run:
  1. fetch held-out crop images from SAGE (skipped if a Kaggle Dataset is attached);
  2. for each model (MODEL_TIERS + TEACHERS) and each descriptor strategy, measure cross-crop
     zero-shot accuracy;
  3. save results to <RESULTS_DIR>/zeroshot_eval.json  (feeds docs/paper/make_figures.py).

Run on Kaggle (GPU, Internet ON) after cloning the repo:
    PDE_DATA_ROOT=/kaggle/working python scripts/evaluate.py
    python scripts/evaluate.py --strategies bare crude rich --tiers small mid large --teachers
"""
from __future__ import annotations
import argparse
import importlib
import os
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import sage_data
import zeroshot


def ensure_deps():
    need = []
    for mod, pkg in [("open_clip", "open_clip_torch>=2.24"), ("timm", "timm>=1.0.3"),
                     ("huggingface_hub", "huggingface_hub"), ("pyarrow", "pyarrow"), ("tqdm", "tqdm")]:
        try:
            importlib.import_module(mod)
        except Exception:
            need.append(pkg)
    if need:
        print(f"[deps] installing {need}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *need], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", nargs="+", default=["bare", "crude", "rich"],
                    help="bare | crude | rich | grounded | grounded_visual | ungrounded | "
                         "grounded_matched (same model as ungrounded -- the clean sourcing test)")
    ap.add_argument("--ungrounded-seeds", nargs="+", type=int, default=None,
                    help="Evaluate SEVERAL seeds of a seeded arm in ONE process. Image "
                         "embeddings are cached per model and reused across strategies, but "
                         "that cache dies with the process -- running 16 seed-evaluations as 16 "
                         "subprocesses re-embedded 14,204 images 16 times (~5.5 h). Sharing one "
                         "process amortises the embedding pass over every seed (~25 min).")
    ap.add_argument("--ungrounded-seed", type=int, default=None,
                    help="which descriptors_ungrounded/<seed>/ set to use; also tags the "
                         "output filename so seeds do not overwrite each other")
    ap.add_argument("--tiers", nargs="+", default=list(C.MODEL_TIERS))
    ap.add_argument("--heavy", action="store_true", help="also evaluate the heavyweight (~86M) tier")
    ap.add_argument("--teachers", action="store_true", help="also evaluate the reference/teacher VLMs")
    ap.add_argument("--exp", choices=list(C.EXPERIMENTS), default="C",
                    help="scale-study subset: A (3 held) / B (6) / C (8, all)")
    ap.add_argument("--clean", action="store_true",
                    help="merge SAGE's duplicate disease labels and drop non-disease labels "
                         "(see config.LABEL_ALIASES / EXCLUDE_LABELS). Writes a separate "
                         "*_clean.json so the as-published result is never overwritten.")
    args = ap.parse_args()

    if args.ungrounded_seed is not None:
        os.environ["PDE_UNGROUNDED_SEED"] = str(args.ungrounded_seed)

    ensure_deps()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    held_crops = C.EXPERIMENTS[args.exp]["held"]
    rows = sage_data.fetch(C.HELDOUT_CROPS, sage_data.full_caps(), min_held_crops=C.MIN_HELD_CROPS)
    rows = [r for r in rows if r["crop"] in set(held_crops)]     # subset to this experiment's held crops
    assert rows, f"no held-out images for experiment {args.exp} ({held_crops})"
    clean_stats = None
    if args.clean:
        n_before = len({r["label"] for r in rows})
        rows, clean_stats = C.clean_rows(rows)
        n_after = len({r["label"] for r in rows})
        print(f"[clean] {n_before} -> {n_after} classes  "
              f"({clean_stats['merged_images']:,} imgs relabelled, "
              f"{clean_stats['dropped_images']:,} dropped)")
    classes = sorted({r["label"] for r in rows})
    print(f"[eval] experiment {args.exp}: held crops = {held_crops}")
    chance = 1.0 / len(classes)
    crops = sorted({c.split("|")[0] for c in classes})
    print(f"[eval] held={len(rows):,} imgs  {len(classes)} classes  crops={crops}  chance={chance:.1%}\n")

    models = [(C.MODEL_TIERS[t][0], C.MODEL_TIERS[t][1]) for t in args.tiers if t in C.MODEL_TIERS]
    if args.heavy:
        models.append(C.HEAVYWEIGHT)
    if args.teachers:
        models += [m for m in C.TEACHERS if m not in models]

    import descriptors as _D
    _SEEDED_ARMS = set(_D.ARM_DIRS)

    results, coverage = {}, {}
    for name, pretrained in models:
        try:
            cache = None
            row = {}
            for strat in args.strategies:
                if args.ungrounded_seeds and strat in _SEEDED_ARMS:
                    # One entry per seed, all sharing this model's image embeddings.
                    for _sd in args.ungrounded_seeds:
                        # descriptors._seed() reads this env var at CALL time and its arm cache
                        # is keyed on (arm, crop, seed), so setting it here is sufficient -- no
                        # cache invalidation needed. (The import-time read of this variable is
                        # exactly the bug that made all three seeds identical in an earlier run.)
                        os.environ["PDE_UNGROUNDED_SEED"] = str(_sd)
                        res, cache = zeroshot.evaluate(name, pretrained, rows, classes, strat,
                                                       device, reuse_img_emb=cache)
                        row[f"{strat}__seed{_sd}"] = res
                else:
                    res, cache = zeroshot.evaluate(name, pretrained, rows, classes, strat, device,
                                                   reuse_img_emb=cache)
                    row[strat] = res
            results[f"{name}/{pretrained}"] = row
            # Iterate the keys actually written: a multi-seed run stores "<arm>__seed<N>",
            # not the bare strategy name, so indexing by args.strategies raised KeyError.
            cols = "  ".join(f"{k}={v['acc']:.1%}" for k, v in row.items())
            _first = next(iter(row.values()))
            print(f"  {name:20s} img={_first['img_params_M']:6.1f}M  {cols}")
        except Exception as e:
            print(f"  {name:20s} skipped ({type(e).__name__}: {str(e)[:60]})")

    # record rich-descriptor coverage (which classes matched a rich descriptor)
    import descriptors as D
    for c in classes:
        D.text_for(c, "rich", coverage)

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_clean" if args.clean else ""
    if args.ungrounded_seed is not None:
        # Key the filename on the ARM as well as the seed. With "_ung{seed}" alone, the
        # ungrounded and grounded_matched arms at the same seed wrote to the SAME file, so
        # whichever ran second silently overwrote the first -- and the matched arm is the
        # one that removes the model-version confound, so losing it defeats the experiment.
        _seeded = [a for a in args.strategies
                   if a in ("ungrounded", "grounded_matched",
                            "ungrounded_short", "grounded_matched_short")]
        _arm = _seeded[0] if _seeded else "ung"
        _short = {"ungrounded": "ung", "grounded_matched": "gm",
                  "ungrounded_short": "ungs", "grounded_matched_short": "gms"}.get(_arm, _arm)
        suffix += f"_{_short}{args.ungrounded_seed}"
    elif args.ungrounded_seeds:
        # Multi-seed run: one file per ARM holding every seed, keyed inside as
        # "<arm>__seed<N>". Analysis reads the seeds out of the one file.
        _seeded = [a for a in args.strategies if a in _SEEDED_ARMS]
        _arm = _seeded[0] if _seeded else "ung"
        _short = {"ungrounded": "ung", "grounded_matched": "gm",
                  "ungrounded_short": "ungs", "grounded_matched_short": "gms"}.get(_arm, _arm)
        suffix += f"_{_short}seeds"
    out = C.RESULTS_DIR / f"zeroshot_eval_{args.exp}{suffix}.json"
    out.write_text(json.dumps({"matcher_normalised": True,
                               "seeds": args.ungrounded_seeds, "chance": chance, "n_classes": len(classes), "crops": crops,
                               "n_images": len(rows), "clean": bool(args.clean),
                               "clean_stats": clean_stats,
                               "coverage": coverage, "models": results}, indent=2))
    print(f"\n[eval] saved {out}")


if __name__ == "__main__":
    main()
