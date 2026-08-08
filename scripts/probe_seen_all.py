"""
Seen-crop linear probe for ALL THREE nested configs (A/B/C) against ONE data pool, resumably.

WHY THIS EXISTS
---------------
1. COMPARABILITY. `probe_seen.py --exp X` derives its class list from whatever SAGE shards happen to
   be on disk. Running A today and C three weeks ago produces configs that cannot be compared:
   `probe_seen_C.json` (14 Jul 2026) had 166 classes, while exp A on the grown pool reports 97 —
   the nested seen pools are only nested if all three are computed from the same snapshot.
2. COST. The seen pools are nested (A subset B subset C), so embedding per-config re-encodes the same
   images three times (~190k encodes instead of ~86k). We embed the config-C pool ONCE per model and
   derive A and B by subsetting on crop.
3. RESUMABILITY. Embeddings are cached to `<RESULTS_DIR>/emb_cache/`, so an interrupted run picks up
   where it left off instead of starting over.

Writes probe_seen_{A,B,C}.json — same schema as probe_seen.py, plus seen_images / n_seen_crops.

    python scripts/probe_seen_all.py                 # all models, all configs
    python scripts/probe_seen_all.py --models s0     # one tier
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import sage_data
import zeroshot


class _ImageDS:
    """Module-level (therefore picklable) dataset so DataLoader workers can spawn on Windows.

    zeroshot.embed_images defines its Dataset inside the function, which forces num_workers=0 under
    Windows' spawn start method -- making JPEG decode single-threaded and the embed pass I/O-bound
    (~8k images per 10 min). Defining it here lets us use real workers.
    """

    def __init__(self, rows, preprocess):
        self.rows, self.preprocess = rows, preprocess

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from PIL import Image
        r = self.rows[i]
        return self.preprocess(Image.open(r["path"]).convert("RGB")), r["label"]


def _save(cache, parts, labels, params_m, n_rows):
    import torch
    torch.save({"emb": torch.cat(parts), "labels": labels,
                "img_params_M": params_m, "n_rows": n_rows}, cache)


def embed_cached(name, pretrained, rows, cache, device, chunk=8192, workers=8):
    """Embed `rows` with checkpointing every `chunk` images, resuming from a partial cache.

    A full pass over the ~70k-image seen pool can exceed the wall-clock budget of whatever is
    invoking us. Saving only at the end means a kill loses everything, so we checkpoint
    incrementally and resume. Row order is preserved (embed_images uses shuffle=False,
    num_workers=0 on Windows), so `n_done` is a valid resume offset.
    """
    import torch

    done_emb, done_labels, params_m = None, [], None
    if cache.exists():
        blob = torch.load(cache, weights_only=False)
        # n_rows was added with chunked checkpointing; a legacy cache without it is only trustworthy
        # if it is a COMPLETE pass over exactly this many rows.
        n_rows = blob.get("n_rows", len(blob.get("labels", [])))
        if n_rows == len(rows):                      # same pool -> the partial is usable
            done_emb = blob["emb"]
            done_labels = list(blob["labels"])
            params_m = blob.get("img_params_M")
            if len(done_labels) >= len(rows):
                print(f"  [{name}] cache complete ({len(done_labels):,})")
                return done_emb, done_labels, params_m
            print(f"  [{name}] resuming at {len(done_labels):,}/{len(rows):,}", flush=True)
        else:
            print(f"  [{name}] cache is for a different pool "
                  f"({blob.get('n_rows')} vs {len(rows)}) -> re-embedding")

    import time
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    model, preprocess, _tok, params_m = zeroshot.load_model(name, pretrained, device)
    parts = [done_emb] if done_emb is not None else []
    labels = list(done_labels)
    todo = rows[len(labels):]

    # ONE DataLoader over everything remaining, rather than one per chunk (which respawns `workers`
    # Windows processes each time under the spawn start method).
    # NOTE ON THROUGHPUT: an apparent hard "stall" during development turned out to be GPU/CPU
    # contention with a second training job on the same machine, NOT a loader bug. If throughput
    # collapses, check for other python processes before rewriting this.
    dl = DataLoader(_ImageDS(todo, preprocess), batch_size=128, num_workers=workers,
                    shuffle=False, pin_memory=(device == "cuda"),
                    persistent_workers=bool(workers), prefetch_factor=4 if workers else None)
    t0, since_save = time.time(), 0
    try:
        with torch.no_grad():
            for imgs, lab in dl:
                parts.append(F.normalize(model.encode_image(imgs.to(device, non_blocking=True)),
                                         dim=-1).cpu())
                labels += list(lab)
                since_save += len(lab)
                if since_save >= chunk:
                    _save(cache, parts, labels, params_m, len(rows))
                    since_save = 0
                    rate = (len(labels) - len(done_labels)) / max(1e-9, time.time() - t0)
                    eta = (len(rows) - len(labels)) / max(1e-9, rate) / 60
                    print(f"  [{name}] {len(labels):,}/{len(rows):,}  "
                          f"{rate:.0f} img/s  eta {eta:.1f} min", flush=True)
        _save(cache, parts, labels, params_m, len(rows))
        print(f"  [{name}] embedded {len(labels):,}", flush=True)
    finally:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    return torch.cat(parts), labels, params_m


def fit_probe(emb, labels, classes, epochs, device, seed=None):
    """Train a linear probe on frozen features; return top-1 on a per-class 20% holdout."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    cidx = {c: i for i, c in enumerate(classes)}
    # Seed BOTH RNGs. `random` controls the train/test split; torch controls the minibatch order in
    # the loop below. Seeding only `random` (as probe_seen.py does) leaves probe training stochastic
    # and produces ~0.2 pp of run-to-run drift on identical cached embeddings -- which is exactly the
    # size of the discrepancies we spent time chasing between reruns.
    s = C.RANDOM_SEED if seed is None else seed
    random.seed(s)
    torch.manual_seed(s)
    by = defaultdict(list)
    for i, l in enumerate(labels):
        by[l].append(i)
    tr, te = [], []
    for l, idxs in by.items():
        random.shuffle(idxs)
        k = max(1, int(0.2 * len(idxs)))
        te += idxs[:k]
        tr += idxs[k:]

    Xtr = emb[tr].to(device)
    ytr = torch.tensor([cidx[labels[i]] for i in tr], device=device)
    Xte = emb[te].to(device)
    yte = [labels[i] for i in te]

    clf = nn.Linear(emb.shape[1], len(classes)).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(len(tr), device=device)
        for s in range(0, len(perm), 256):
            b = perm[s:s + 256]
            loss = F.cross_entropy(clf(Xtr[b]), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = clf(Xte).argmax(1).cpu().tolist()
    return sum(classes[p] == gt for p, gt in zip(pred, yte)) / len(yte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(C.DEPLOY_MODELS))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--exps", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"])
    ap.add_argument("--no-cache", action="store_true", help="delete any cache and re-embed")
    ap.add_argument("--chunk", type=int, default=8192,
                    help="checkpoint the embedding cache every N images (resume granularity)")
    ap.add_argument("--workers", type=int, default=8,
                    help="DataLoader workers; JPEG decode is the bottleneck, not the GPU")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- one fetch, one pool: everything below is a subset of the config-C seen crops ----
    pool_crops = C.EXPERIMENTS["C"]["seen"]
    rows = sage_data.fetch(C.TRAIN_CROPS, sage_data.full_caps(), min_held_crops=C.MIN_HELD_CROPS)
    rows = [r for r in rows if r["crop"] in set(pool_crops)]
    assert rows, f"no seen images on disk for {pool_crops}"
    print(f"[pool] {len(rows):,} seen images over {len(pool_crops)} crops, "
          f"{len({r['label'] for r in rows})} classes  (device={device})")

    cache_dir = C.RESULTS_DIR / "emb_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = {e: {"exp": e, "models": {}} for e in args.exps}

    for name, pretrained in C.resolve_models(args.models):
        cache = cache_dir / f"{C.safe_name(name)}_seenpool.pt"
        if args.no_cache and cache.exists():
            cache.unlink()
        emb, labels, params_m = embed_cached(name, pretrained, rows, cache, device,
                                             chunk=args.chunk, workers=args.workers)

        crop_of = [l.split("|", 1)[0] for l in labels]
        for e in args.exps:
            seen = set(C.EXPERIMENTS[e]["seen"])
            keep = [i for i, cr in enumerate(crop_of) if cr in seen]
            sub_labels = [labels[i] for i in keep]
            classes = sorted(set(sub_labels))
            acc = fit_probe(emb[keep], sub_labels, classes, args.epochs, device)
            o = out[e]
            o.update(seen_classes=len(classes), seen_images=len(keep),
                     n_seen_crops=len(C.EXPERIMENTS[e]["seen"]),
                     seen_crops=C.EXPERIMENTS[e]["seen"])
            o["models"][name] = {"img_params_M": round(params_m, 2),
                                 "seen_probe_top1": round(acc, 4)}
            print(f"    exp {e}: {len(classes):3d} cls / {len(keep):6,d} imgs  "
                  f"{name:16s} {params_m:6.1f}M  top1 = {acc:.1%}", flush=True)

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for e in args.exps:
        p = C.RESULTS_DIR / f"probe_seen_{e}.json"
        # MERGE, don't clobber: invoking with --models s1 must not delete s0's entry from a previous
        # invocation. Prior entries are kept only if they describe the SAME pool (same class/image
        # counts) -- otherwise they came from a different data snapshot and are not comparable.
        if p.exists():
            try:
                old = json.loads(p.read_text(encoding="utf-8"))
                same_pool = (old.get("seen_classes") == out[e]["seen_classes"]
                             and old.get("seen_images") == out[e]["seen_images"])
                if same_pool:
                    merged = dict(old.get("models", {}))
                    merged.update(out[e]["models"])
                    out[e]["models"] = merged
                elif old.get("models"):
                    print(f"  [{e}] discarding {len(old['models'])} stale model entr"
                          f"{'y' if len(old['models']) == 1 else 'ies'} from a different pool "
                          f"({old.get('seen_classes')} cls / {old.get('seen_images')} imgs)")
            except Exception as ex:
                print(f"  [{e}] could not merge existing {p.name}: {type(ex).__name__}: {ex}")
        # keep the tier order stable regardless of invocation order
        order = {n: i for i, n in enumerate(n for n, _ in C.resolve_models(list(C.DEPLOY_MODELS)))}
        out[e]["models"] = dict(sorted(out[e]["models"].items(),
                                       key=lambda kv: order.get(kv[0], 99)))
        p.write_text(json.dumps(out[e], indent=2))
        print(f"[probe] saved {p}  ({len(out[e]['models'])} models)")
    print("[probe] the VLM tiers do seen (probe) AND unseen (zero-shot); the CNNs do seen only.")


if __name__ == "__main__":
    main()
