"""
Phase 2 of the 2026-09-22 deep review: the four re-scores that turn open caveats into numbers.

Nothing here needs a new image, a new registry or any training. It embeds the held-out images
ONCE per encoder and reuses that pass for every configuration, label set and strategy, because
zeroshot.evaluate's cache dies with the process and re-embedding per strategy is what made the
earlier sweeps take hours.

It answers, for every configuration and both label sets:

  1. SHORTLIST AND ABSTENTION FOR CuPL+ -- top-5, AURC and acc@cov80 for the registry the paper
     actually recommends. Table 3 measured them for grounded and rich only, while the Discussion
     framed the system as "a shortlist with the ability to abstain".
  2. MAJORITY-CLASS PRIOR -- at every configuration, not just the flattering one. At A the prior
     is 14.8%, above four of the seven strategies, and the manuscript quoted only C's 4.2%.
     Class-macro accuracy is recorded next to micro for the same reason.
  3. THE TAXONOMY TEST -- grounded_split against grounded_visual_split. Same construction, same
     source text, with the pathogen/taxonomy prose removed. The Discussion used to advise
     "leave taxonomy out of the prototype" with no experiment behind it; this is the experiment.
  4. THE ORGAN-RULE BOUND -- the CuPL+ minus grounded gap on foliar classes only, where the
     image-informed plant-part instruction is inert. If the gap survives there, that instruction
     is not what produces it.

POINT IT AT THE 288-PIXEL BUILD, NOT THE FULL-RESOLUTION ONE. Every published number was measured
on C:\\kaggle\\upload\\exp_data (longest edge 288, the subset uploaded to Kaggle). Re-scoring the
same images from C:\\kaggle\\working\\exp_data, which is the 448-pixel build, moves MobileCLIP2-S0
at configuration A by up to +0.4 points -- enough to contradict the manuscript's own tables. On the
288-pixel build this script reproduces the published cells to within 0.02. The size check below
refuses to run on anything else unless you pass --allow-any-resolution.

USAGE (Windows, local GPU)
    set PDE_DATASET_DIR=C:\\kaggle\\upload\\exp_data
    set PDE_DATA_ROOT=C:\\kaggle\\upload
    python scripts/rescore_descriptors.py

    python scripts/rescore_descriptors.py --models s0 --exps C      # quick smoke test, one encoder
    python scripts/rescore_descriptors.py --reference               # add the SigLIP2 ceiling rows

Writes:
    docs/paper/manuscript/revision_numbers.json                 (read by make_tex_tables_revision / make_figures_revision)
    docs/paper/manuscript/revision_results/metrics_abstain_*.json  (same shape as scripts/metrics.py writes)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

# Every strategy the manuscript reports, plus the two that answer the taxonomy question.
STRATEGIES = ["bare", "bare80", "crude", "rich", "grounded", "grounded_split",
              "grounded_visual", "grounded_visual_split", "dclip", "cupl"]

# Classes the plant-part instruction was written for: the prompt names fruit and post-harvest
# rots (cigar end rot, green/whisker mould, belly rot, brown rot, berry blotch) and cereal head
# diseases (head scab, karnal bunt, loose smut). Everything else was to be described as a leaf,
# which is what the class name already says, so the instruction is inert there.
NON_FOLIAR_PATTERNS = ["rot", "mould", "mold", "bunt", "smut", "scab", "blotch"]


def is_non_foliar(label: str) -> bool:
    dis = label.split("|", 1)[1].lower()
    return any(p in dis for p in NON_FOLIAR_PATTERNS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["s0", "s1", "s2", "b"])
    ap.add_argument("--reference", action="store_true", help="also score the SigLIP2 ceiling")
    ap.add_argument("--exps", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--strategies", nargs="+", default=STRATEGIES)
    ap.add_argument("--allow-any-resolution", action="store_true",
                    help="skip the 288-pixel check (results will not match the published cells)")
    ap.add_argument("--dataset-dir", default=None, help="sets PDE_DATASET_DIR before import")
    ap.add_argument("--data-root", default=None, help="sets PDE_DATA_ROOT before import")
    ap.add_argument("--out", default=str(REPO / "docs" / "paper" / "revision_numbers.json"))
    args = ap.parse_args()

    # config.py reads these at import time, so they must be set first.
    if args.dataset_dir:
        os.environ["PDE_DATASET_DIR"] = args.dataset_dir
    if args.data_root:
        os.environ["PDE_DATA_ROOT"] = args.data_root

    import torch
    import config as C
    import sage_data
    import zeroshot
    import descriptors as D
    from metrics import topk_and_riskcoverage

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[rescore] device={device}  images={C.DATASET_DIR}")

    # ---- one fetch, the widest configuration ------------------------------------------------
    rows_all = sage_data.fetch(C.HELDOUT_CROPS, sage_data.full_caps(),
                               min_held_crops=C.MIN_HELD_CROPS)
    # The configurations are nested, so the widest one requested is the union: embedding it once
    # covers the others. Asking for A alone embeds 4,050 images instead of 14,204.
    held_c = set().union(*(set(C.EXPERIMENTS[e]["held"]) for e in args.exps))
    rows_all = [r for r in rows_all if r["crop"] in held_c]
    assert rows_all, "no held-out images -- check PDE_DATASET_DIR"
    print(f"[rescore] {len(rows_all):,} held-out images over {len(held_c)} crops")

    # ---- anti-drift: the published cells were measured on the 288-pixel build ----------------
    from PIL import Image
    # The build downscales to a longest edge of 288 and leaves smaller originals alone, so the
    # test is an upper bound, not equality: 220 and 244 are normal, 448 and 1300 are the wrong build.
    sizes = {max(Image.open(r["path"]).size) for r in rows_all[:25]}
    if max(sizes) > 288:
        msg = (f"[rescore] IMAGE BUILD MISMATCH: sampled longest edges up to {max(sizes)}, "
               f"expected at most 288.\n"
               f"[rescore]   {C.DATASET_DIR}\n"
               f"[rescore] The manuscript's numbers come from the 288-pixel subset "
               f"(C:\\kaggle\\upload\\exp_data). Re-scoring from the 448-pixel build shifts cells by "
               f"up to 0.4 points, so the new rows would silently disagree with Tables 1 and 2.")
        if not args.allow_any_resolution:
            print(msg + "\n[rescore] refusing to run; pass --allow-any-resolution to override.")
            return 2
        print(msg + "\n[rescore] --allow-any-resolution given, continuing anyway.")

    # Row order is the embedding order, so an index into rows_all indexes the embeddings too.
    idx_of_path = {r["path"]: i for i, r in enumerate(rows_all)}

    # ---- the six (configuration, label set) views, as index lists into rows_all --------------
    views = {}
    for exp in args.exps:
        held = set(C.EXPERIMENTS[exp]["held"])
        base = [r for r in rows_all if r["crop"] in held]
        for labelset in ("uncleaned", "clean"):
            rows = base if labelset == "uncleaned" else C.clean_rows(base)[0]
            labels = [r["label"] for r in rows]
            classes = sorted(set(labels))
            counts = Counter(labels)
            views[(exp, labelset)] = {
                "idx": [idx_of_path[r["path"]] for r in rows],
                "labels": labels,
                "classes": classes,
                "n_classes": len(classes),
                "n_images": len(rows),
                "chance": 100.0 / len(classes),
                # The baseline a constant predictor achieves: what "×chance" quietly ignores.
                "majority_prior": 100.0 * max(counts.values()) / len(rows),
                "foliar_idx": [i for i, lab in enumerate(labels) if not is_non_foliar(lab)],
            }
            print(f"[rescore] {exp}/{labelset:9s} {len(rows):6,} imgs  {len(classes):3d} classes  "
                  f"majority prior {views[(exp, labelset)]['majority_prior']:.1f}%  "
                  f"foliar images {len(views[(exp, labelset)]['foliar_idx']):,}")

    non_foliar = sorted({lab for lab in views[(args.exps[-1], "uncleaned")]["labels"]
                         if is_non_foliar(lab)})
    print(f"[rescore] organ rule treated as ACTIVE on {len(non_foliar)} classes: "
          + ", ".join(n.replace('|', '/') for n in non_foliar))

    models = C.resolve_models(args.models, include_reference=args.reference)
    out: dict = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": device,
        "matcher_normalised": True,
        "strategies": args.strategies,
        "non_foliar_classes": non_foliar,
        "majority_prior": {ls: {e: views[(e, ls)]["majority_prior"] for e in args.exps}
                           for ls in ("uncleaned", "clean")},
        "configs": {},
        "per_model": {},
    }
    abstain: dict = {}

    for name, pretrained in models:
        key = f"{name}/{pretrained}"
        t0 = time.time()
        model, preprocess, tok, params_m = zeroshot.load_model(name, pretrained, device)
        img_emb, emb_labels = zeroshot.embed_images(model, preprocess, rows_all, device)
        assert emb_labels == [r["label"] for r in rows_all], "embedding order drifted from rows"
        print(f"[rescore] {name:18s} {params_m:6.1f}M  embedded {len(rows_all):,} images "
              f"in {time.time() - t0:.0f}s")

        for (exp, labelset), v in views.items():
            sel = torch.tensor(v["idx"])
            emb = img_emb[sel]
            labels, classes = v["labels"], v["classes"]
            foliar = torch.tensor(v["foliar_idx"]) if v["foliar_idx"] else None
            for strat in args.strategies:
                protos = D.build_prototypes(model, tok, classes, strat, device)
                sims = (emb.to(device) @ protos.T).cpu()
                pred = sims.argmax(1)
                gt = torch.tensor([classes.index(l) for l in labels])
                hit = (pred == gt)
                micro = 100.0 * hit.float().mean().item()
                # Class-macro: the average over classes, not images. Held-out classes range from
                # the 25-image floor to the 600-image cap, so micro is carried by the big ones.
                per_class = {}
                for h, lab in zip(hit.tolist(), labels):
                    a = per_class.setdefault(lab, [0, 0])
                    a[0] += h
                    a[1] += 1
                macro = 100.0 * sum(a / n for a, n in per_class.values()) / len(per_class)
                rec = {"micro": micro, "macro": macro}
                if foliar is not None:
                    rec["micro_foliar"] = 100.0 * hit[foliar].float().mean().item()
                out["per_model"].setdefault(key, {}).setdefault(
                    f"{exp}/{labelset}", {})[strat] = rec

                # Shortlist and abstention, in the shape scripts/metrics.py writes.
                if labelset == "uncleaned" and strat in ("cupl", "grounded", "rich"):
                    a = abstain.setdefault(exp, {
                        "matcher_normalised": True, "n_classes": v["n_classes"],
                        "chance": v["chance"] / 100.0, "n_images": v["n_images"], "models": {}})
                    a["models"].setdefault(key, {"img_params_M": round(params_m, 2)})[strat] = \
                        topk_and_riskcoverage(sims, labels, classes)
            print(f"[rescore]   {exp}/{labelset:9s} "
                  + "  ".join(f"{s}={out['per_model'][key][f'{exp}/{labelset}'][s]['micro']:.1f}"
                              for s in ("grounded", "cupl") if s in args.strategies))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # ---- means over the four deployable encoders --------------------------------------------
    deploy = [f"{n}/{p}" for n, p in (C.DEPLOY_MODELS[k] for k in C.DEPLOY_MODELS)
              if f"{n}/{p}" in out["per_model"]]
    for exp in args.exps:
        for labelset in ("uncleaned", "clean"):
            v = views[(exp, labelset)]
            view_key = f"{exp}/{labelset}"
            block = {"n_classes": v["n_classes"], "n_images": v["n_images"],
                     "chance": v["chance"], "majority_prior": v["majority_prior"],
                     "n_deployable": len(deploy), "means": {}, "macro_means": {},
                     "means_foliar": {}}
            for strat in args.strategies:
                vals = [out["per_model"][m][view_key][strat] for m in deploy]
                if not vals:
                    continue
                block["means"][strat] = sum(x["micro"] for x in vals) / len(vals)
                block["macro_means"][strat] = sum(x["macro"] for x in vals) / len(vals)
                if "micro_foliar" in vals[0]:
                    block["means_foliar"][strat] = sum(x["micro_foliar"] for x in vals) / len(vals)
            m = block["means"]
            mf = block["means_foliar"]
            # The two headline deltas this run exists to measure.
            if {"grounded_split", "grounded_visual_split"} <= set(m):
                block["taxonomy_test"] = {
                    "grounded_split": m["grounded_split"],
                    "grounded_visual_split": m["grounded_visual_split"],
                    "delta_taxonomy_removed": m["grounded_visual_split"] - m["grounded_split"],
                    "cupl_minus_grounded_visual_split": m.get("cupl", 0) - m["grounded_visual_split"],
                }
            if {"cupl", "grounded"} <= set(m):
                block["organ_rule_bound"] = {
                    "gap_all_images": m["cupl"] - m["grounded"],
                    "gap_foliar_only": (mf.get("cupl", 0) - mf.get("grounded", 0)) if mf else None,
                    "n_foliar_images": len(v["foliar_idx"]),
                }
            out["configs"].setdefault(exp, {})[labelset] = block

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[rescore] wrote {args.out}")

    res_dir = REPO / "docs" / "paper" / "revision_results"
    res_dir.mkdir(parents=True, exist_ok=True)
    for exp, payload in abstain.items():
        p = res_dir / f"metrics_abstain_{exp}.json"
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[rescore] wrote {p}")

    # ---- the four answers, printed so the run reports itself --------------------------------
    print("\n[rescore] ====== what this run settles ======")
    for exp in args.exps:
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
                print(f"      CuPL+ - grounded: all images {o['gap_all_images']:+.1f}, "
                      f"foliar only {o['gap_foliar_only']:+.1f} "
                      f"({o['n_foliar_images']:,} images)")
    print("\n[rescore] next: python docs/paper/make_tex_tables_revision.py")
    print("[rescore]       python docs/paper/make_figures_revision.py")
    print("[rescore]       python scripts/paper_fixes/apply_revision_results.py   (writes the prose)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
