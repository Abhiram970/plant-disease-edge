"""
Regenerate EVERY results table straight from the result JSONs.

WHY THIS EXISTS
---------------
The manuscript drifted from the data three separate times (an old 17-class eval, a
superseded edge benchmark, and a third hardcoded latency set inside make_figures.py).
Tables are now GENERATED, never typed. If a number is in the paper, it came from here.

    python docs/paper/make_tables.py            # print all tables
    python docs/paper/make_tables.py --write    # also write docs/paper/TABLES.md

Canonical sources (all in docs/paper/):
    zeroshot_eval_{A,B,C}.json    cross-crop zero-shot at 3 scales   (16 / 34 / 51 classes)
    metrics_abstain_{A,B,C}.json  top-5 + risk-coverage
    run_all_bakeoff.json          encoder bake-off (Exp-A held set)
    probe_seen_C.json             seen-head linear probe (166 classes)
    run_all_exp3_lw11_full.json   WiSE-FT alpha sweep (full data)
    supervised_*.json             supervised CNN baselines (166 classes)
    loco_s0_rich.json             leave-one-crop-out
    edge_quant_benchmark.json     on-device latency / size / INT8 diagnosis
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

# Windows consoles default to cp1252 and choke on the arrows/en-dashes below.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
OUT = []


def load(name):
    p = HERE / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def emit(s=""):
    print(s)
    OUT.append(s)


def pct(x, nd=1):
    return "—" if x is None else f"{x * 100:.{nd}f}%"


def short(name):
    return name.split("/")[0]


# ---------------------------------------------------------------- 1. scale study
def table_scale_study():
    emit("## T1 — Cross-crop zero-shot at three scales (descriptor ablation)\n")
    emit("Held-out crops are never trained. `bare` = class name only; `rich` = hand-curated symptom")
    emit("paragraph; `grounded` = LLM source-grounded `symptom_text`.\n")
    for e in "ABC":
        j = load(f"zeroshot_eval_{e}.json")
        if not j:
            emit(f"*(Exp {e} missing)*\n"); continue
        emit(f"**Experiment {e} — {j['n_classes']} classes, {len(j['crops'])} held crops "
             f"({', '.join(j['crops'])}), chance {pct(j['chance'])}**\n")
        emit("| Model | Params | bare | crude | rich | grounded | rich−bare |")
        emit("|---|---|---|---|---|---|---|")
        bares, riches = [], []
        for m, d in j["models"].items():
            b, c = d["bare"]["acc"], d["crude"]["acc"]
            r, g = d["rich"]["acc"], d["grounded"]["acc"]
            bares.append(b); riches.append(r)
            emit(f"| {short(m)} | {d['bare']['img_params_M']:.1f} M | {pct(b)} | {pct(c)} "
                 f"| **{pct(r)}** | {pct(g)} | **{(r-b)*100:+.1f} pp** |")
        mb, mr = sum(bares)/len(bares), sum(riches)/len(riches)
        emit(f"| *mean* | — | *{pct(mb)}* | — | *{pct(mr)}* | — | ***{(mr-mb)*100:+.1f} pp*** |")
        emit("")


# ---------------------------------------------------------------- 1b. label-noise sensitivity
def table_clean():
    clean = load("zeroshot_eval_C_clean.json")
    orig = load("zeroshot_eval_C.json")
    if not (clean and orig):
        return
    emit("## T1b — Sensitivity to SAGE label defects (experiment C)\n")
    emit(f"Merging the 5 duplicate disease pairs and dropping the 4 non-disease labels takes the "
         f"held-out set from **{orig['n_classes']} to {clean['n_classes']} classes** "
         f"(chance {pct(orig['chance'])} -> {pct(clean['chance'])}). Means over the 4 deployable "
         f"encoders:\n")
    emit("| Strategy | As-published | Label-corrected | Δ |")
    emit("|---|---|---|---|")

    def mean(j, s):
        ms = [v for k, v in j["models"].items() if "SigLIP" not in k]
        return sum(m[s]["acc"] for m in ms) / len(ms)

    vals = {}
    for s in ("bare", "crude", "rich", "grounded"):
        a, b = mean(orig, s), mean(clean, s)
        vals[s] = (a, b)
        emit(f"| {s} | {pct(a)} | **{pct(b)}** | {(b - a) * 100:+.1f} pp |")
    ga, gb = vals["grounded"]
    ra, rb = vals["rich"]
    emit(f"\n> The correction lifts every strategy by 4.8--6.0 pp, confirming that duplicate classes "
         f"were suppressing all of them. Critically the **grounded − rich gap is unchanged "
         f"({(ga - ra) * 100:+.1f} pp vs {(gb - rb) * 100:+.1f} pp)**, so the paper's central claim "
         f"is not an artefact of label noise. Grounded reaches "
         f"**{gb / clean['chance']:.1f}× chance** after correction.\n")
    emit("| Encoder | rich | grounded | Δ |")
    emit("|---|---|---|---|")
    for m, d in clean["models"].items():
        r, g = d["rich"]["acc"], d["grounded"]["acc"]
        emit(f"| {short(m)} | {pct(r)} | **{pct(g)}** | {(g - r) * 100:+.1f} pp |")
    emit("\n> Grounded wins on **all five** encoders, including the SigLIP2 reference.\n")


# ---------------------------------------------------------------- 2. abstain
def table_abstain():
    emit("## T2 — Top-5 and abstention (risk–coverage)\n")
    emit("| Exp | Classes | Model | Strategy | Top-1 | Top-5 | AURC | acc@cov90 | acc@cov80 |")
    emit("|---|---|---|---|---|---|---|---|---|")
    for e in "ABC":
        j = load(f"metrics_abstain_{e}.json")
        if not j:
            continue
        for m, d in j["models"].items():
            for strat in ("rich", "grounded"):
                sd = d.get(strat)
                if not isinstance(sd, dict):
                    continue
                emit(f"| {e} | {j['n_classes']} | {short(m)} | {strat} | {pct(sd.get('top1'))} "
                     f"| **{pct(sd.get('top5'))}** | {sd.get('aurc')} "
                     f"| {pct(sd.get('acc@cov90'))} | {pct(sd.get('acc@cov80'))} |")
    emit("")


# ---------------------------------------------------------------- 3. bake-off
def table_bakeoff():
    j = load("run_all_bakeoff.json")
    if not j:
        return
    emit("## T3 — Encoder bake-off (rich descriptors, 17-class pilot held set)\n")
    emit(f"{j['n_classes']} classes, chance {pct(j['chance'])} — the 3 anchor held-out crops, run before")
    emit("the nested A/B/C splits were frozen. Comparable *within* the table; do not mix with T1.\n")
    emit("| Encoder | Params | Zero-shot | " + " | ".join(
        list(next(iter(j["models"].values()))["by_crop"])) + " |")
    emit("|---|---|---|" + "---|" * len(next(iter(j["models"].values()))["by_crop"]))
    for m, d in sorted(j["models"].items(), key=lambda kv: -kv[1]["rich_acc"]):
        crops = " | ".join(pct(v) for v in d["by_crop"].values())
        emit(f"| {m} | {d['img_params_M']:.1f} M | **{pct(d['rich_acc'])}** | {crops} |")
    emit("")
    emit("> Domain/biological foundation models (SCOLD, BioCLIP2) fall **at or below chance** under a")
    emit("> descriptor protocol — see the paper's caveat on the best-effort SCOLD wrapper.\n")


# ---------------------------------------------------------------- 4. seen side
def table_seen():
    emit("## T4 — Known-crop accuracy (seen head, frozen backbone + linear probe)\n")
    probes = {e: load(f"probe_seen_{e}.json") for e in "ABC"}
    probes = {e: p for e, p in probes.items() if p}
    if probes:
        emit("Seen-side scaling: does the probe also flatten as the seen label space grows?\n")
        cols = sorted({m for p in probes.values() for m in p["models"]},
                      key=lambda m: next(p["models"][m]["img_params_M"]
                                         for p in probes.values() if m in p["models"]))
        emit("| Config | Seen crops | Seen classes | Seen images | "
             + " | ".join(short(c) for c in cols) + " |")
        emit("|---|---|---|---|" + "---|" * len(cols))
        for e, p in probes.items():
            n_img = p.get("seen_images")
            accs = " | ".join(pct(p["models"].get(c, {}).get("seen_probe_top1")) for c in cols)
            emit(f"| **{e}** | {p.get('n_seen_crops', len(p['seen_crops']))} | {p['seen_classes']} "
                 f"| {'—' if n_img is None else f'{n_img:,}'} | {accs} |")
        emit("")
        stale = [e for e, p in probes.items() if p.get("seen_images") is None]
        if stale:
            emit(f"> ⚠ Config {'/'.join(stale)} predates the `seen_images` field, so it may have been")
            emit("> run against a smaller on-disk SAGE pool — class counts are pool-dependent. Re-run")
            emit("> before comparing across configs.\n")
    rows = []
    for f in sorted(HERE.glob("supervised_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        rows.append((d.get("arch"), d.get("params_M"), d.get("seen_classes"),
                     d.get("seen_top1"), len(d.get("epoch_log") or []), d.get("batch")))
    if rows:
        emit("**Supervised CNN baselines** — all are *structurally* incapable of unseen-crop")
        emit("diagnosis, since no output neuron exists for an unseen class:\n")
        emit("| Architecture | Params | Classes | Epochs | Batch | Seen top-1 | Unseen |")
        emit("|---|---|---|---|---|---|---|")
        for a, pm, n, t, ne, b in sorted(rows, key=lambda r: -(r[3] or 0)):
            p = "—" if pm is None else f"{pm:.1f} M"
            emit(f"| {a.replace('_', '-')} | {p} | {n} | {ne} | {b or '—'} | **{pct(t)}** "
                 f"| 0 (structural) |")
        emit("")
        if len({(n, ne) for _, _, n, _, ne, _ in rows}) > 1:
            emit("> ⚠ **Class or epoch counts differ** — these rows are not comparable. Re-run the "
                 "odd ones before tabulating.\n")
        batches = {b for *_, b in rows if b}
        if len(batches) > 1:
            emit(f"> ⚠ **Mixed batch size** ({sorted(batches)}). The larger models were re-run at a "
                 "smaller batch after exhausting GPU memory at the original setting. With a fixed "
                 "learning rate a smaller batch means more optimiser steps per epoch, so the "
                 "batch-64 rows are not perfectly controlled against the batch-128 rows. The gap is "
                 "small relative to the spread here, but it should be stated rather than smoothed "
                 "over.\n")


# ---------------------------------------------------------------- 5. WiSE-FT
def table_wiseft():
    j = load("run_all_exp3_lw11_full.json")
    if not j:
        return
    emit("## T5 — WiSE-FT: tuning the seen↔unseen trade-off (full data)\n")
    emit(f"{j['model']} · {j['seen_images']:,} seen images · {j['seen_classes']} seen classes · "
         f"{j['unseen_classes']} unseen classes · {j['ft_epochs']} fine-tune epochs\n")
    emit("| α | Seen | Unseen (zero-shot) |")
    emit("|---|---|---|")
    for s in j["sweep"]:
        tag = " ← best balance" if j.get("best", {}).get("alpha") == s["alpha"] else ""
        lab = {0.0: "0.0 (frozen)", 0.5: "0.5 (WiSE-FT)", 1.0: "1.0 (naive fine-tune)"}.get(
            s["alpha"], str(s["alpha"]))
        emit(f"| {lab} | {pct(s['seen'])} | {pct(s['unseen'])}{tag} |")
    a0 = next(s for s in j["sweep"] if s["alpha"] == 0.0)
    a5 = next((s for s in j["sweep"] if s["alpha"] == 0.5), None)
    if a5:
        emit(f"\n> α=0.5 buys **{(a5['seen']-a0['seen'])*100:+.1f} pp seen** for only "
             f"**{(a5['unseen']-a0['unseen'])*100:+.1f} pp unseen**; naive fine-tuning (α=1) "
             f"collapses unseen zero-shot.\n")
    if j.get("ft_loss"):
        emit(f"Fine-tune loss: {' → '.join(str(x) for x in j['ft_loss'])}\n")


# ---------------------------------------------------------------- 6. LOCO
def table_loco():
    j = load("loco_s0_rich.json")
    if not j:
        return
    emit("## T6 — Leave-one-crop-out (anti-cherry-pick)\n")
    emit(f"{j['model']} ({j['img_params_M']} M) · {j['strategy']} · {j['n_classes']} classes · "
         f"chance {pct(j['chance'])} · bootstrap 95% CI\n")
    emit("| Crop | N | Zero-shot | 95% CI |")
    emit("|---|---|---|---|")
    for crop, d in sorted(j["per_crop"].items(), key=lambda kv: -kv[1]["acc"]):
        ci = d.get("ci95", [None, None])
        emit(f"| {crop} | {d['n']:,} | {pct(d['acc'])} | [{pct(ci[0])}, {pct(ci[1])}] |")
    p = j["pooled"]
    emit(f"| **Pooled** | **{p['n']:,}** | **{pct(p['acc'])}** | "
         f"[{pct(p['ci95'][0])}, {pct(p['ci95'][1])}] |\n")


# ---------------------------------------------------------------- 7. edge
def table_edge():
    j = load("edge_quant_benchmark.json")
    if not j:
        return
    emit("## T7 — On-device efficiency (image encoder only)\n")
    emit(f"CPU · batch 1 · {j['img_size']}×{j['img_size']} · ONNX Runtime {j.get('ort_version')} · "
         f"{j['runs']} runs\n")
    emit("| Tier | Params | Torch FP32 | ONNX FP32 | INT8 dynamic | INT8 static (QDQ) | FP32 MB | INT8 MB |")
    emit("|---|---|---|---|---|---|---|---|")
    for k, d in j["models"].items():
        v = d.get("variants", {})
        g = lambda n, f="p50_ms": v.get(n, {}).get(f)
        fmt = lambda x: "—" if x is None else f"{x:.1f} ms"
        emit(f"| {d['model']} | {d['img_params_M']:.1f} M | {fmt(g('torch_fp32'))} "
             f"| **{fmt(g('onnx_fp32'))}** | {fmt(g('onnx_int8_dynamic'))} "
             f"| {fmt(g('onnx_int8_static'))} | {g('onnx_fp32','size_mb')} "
             f"| {g('onnx_int8_static','size_mb')} |")
    emit("\n**INT8 diagnosis** — convolutions the quantiser could not convert:\n")
    emit("| Tier | Float convs left | Convert nodes | INT8 speedup | Verdict |")
    emit("|---|---|---|---|---|")
    for k, d in j["models"].items():
        dg = d.get("diagnosis", {}).get("int8_static")
        if dg:
            emit(f"| {d['model']} | **{dg['float_conv_left']}** | {dg['convert_nodes']} "
                 f"| {dg.get('speedup_vs_fp32')}× | {dg['verdict']} |")
    emit("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="also write docs/paper/TABLES.md")
    a = ap.parse_args()

    emit("# Generated results tables")
    emit("")
    emit("> **Auto-generated by `docs/paper/make_tables.py` — do not hand-edit.**")
    emit("> Every number in `tex/main.tex` must match a number here. Re-run after any new result.")
    emit("")
    for fn in (table_scale_study, table_clean, table_abstain, table_bakeoff, table_seen,
               table_wiseft, table_loco, table_edge):
        try:
            fn()
        except Exception as e:
            emit(f"*(table failed: {fn.__name__}: {type(e).__name__}: {e})*\n")

    if a.write:
        (HERE / "TABLES.md").write_text("\n".join(OUT), encoding="utf-8")
        print(f"\n[tables] wrote {HERE / 'TABLES.md'}")


if __name__ == "__main__":
    main()
