"""
Generate paper figures from the Phase-0 results.

Data from the de-risk runs is embedded below (so figures regenerate anywhere); where a
Kaggle result JSON is present it is read to refresh/extend a figure (e.g. the descriptor
test). Outputs PNGs into ./figures next to this script.

    python docs/paper/make_figures.py
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)
CHANCE = 0.0588  # 1/17 held-out classes

# ---- Phase-0 probe: frozen pretrained image-text models, held-out zero-shot (17 classes) ----
PROBE = [  # (name, image-encoder params M, zero-shot acc)
    ("MobileCLIP2-S0", 11.41, 0.199),
    ("MobileCLIP-S1", 21.54, 0.208),
    ("MobileCLIP-S2", 35.82, 0.177),
    ("MobileCLIP2-S2", 35.82, 0.203),
    ("MobileCLIP-B", 86.35, 0.219),
    ("MobileCLIP2-B", 86.35, 0.171),
    ("MobileCLIP2-S3", 125.24, 0.195),
    ("ViT-B-16-SigLIP2", 92.88, 0.256),
    ("MobileCLIP2-L-14", 303.97, 0.216),
    ("MobileCLIP2-S4", 321.78, 0.210),
]
FROM_SCRATCH = (5.5, 0.110)   # edgenext_small distilled from scratch (4 runs, best)

# ---- specialization (MobileCLIP2-S0 + residual adapter) held-out zero-shot per epoch ----
SPEC_EPOCH = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
SPEC_HELD = [.121, .176, .122, .142, .140, .138, .150, .138, .150, .148, .154]
SPEC_BASE = 0.199

# ---- EXP1 bake-off: encoder rich-descriptor zero-shot (held 8 classes, chance 12.5%) ----
BAKEOFF = [  # (label, img params M, rich zero-shot) -- 17-class held (Coffee+Orange+Peach), chance 5.9%
    ("MobileCLIP2-S0", 11.4, 0.270),   # exact from run_all_bakeoff.json (Jul 1, Coffee-inclusive)
    ("MobileCLIP-S1", 21.5, 0.224),
    ("MobileCLIP2-S2", 35.8, 0.287),
    ("SigLIP2", 92.9, 0.315),
    ("BioCLIP2", 304.0, 0.096),
    ("SCOLD", 237.5, 0.044),           # below chance = broken wrapper (RoBERTa base fallback); footnote/drop
]
BAKEOFF_CHANCE = 0.059
# ---- EXP2 hybrid (MobileCLIP2-S0, 11M): trained seen vs zero-shot seen vs zero-shot unseen ----
HYBRID = {"seen_trained": 0.672, "seen_zeroshot": 0.093, "unseen_zeroshot": 0.270}
# ---- EXP3 WiSE-FT sweep (MobileCLIP2-S0): (alpha, seen, unseen) from run_all_exp3_lw11.json ----
WISEFT = [(0.0, 0.670, 0.270), (0.5, 0.776, 0.154), (1.0, 0.822, 0.104)]
# ---- descriptor ablation (held 17-class): bare/crude/rich/grounded per model (results/zeroshot_eval.json) ----
DESC_ABLATION = [  # (model, params, bare, crude, rich, grounded)
    ("MC2-S0", 11.4, 0.183, 0.198, 0.270, 0.242),
    ("MC-S1",  21.5, 0.189, 0.211, 0.224, 0.281),
    ("MC2-S2", 35.8, 0.185, 0.203, 0.287, 0.214),
    ("MC-B",   86.3, 0.216, 0.226, 0.268, 0.282),
]
# ---- edge benchmark (laptop CPU 16 threads, batch 1, 224px, ORT 1.26, 50 runs) ----
# SOURCE: docs/paper/edge_quant_benchmark.json (27 Jul 2026). The int8 column is the STATIC QDQ
# path (per-channel + calibrated + shape-inference pre-pass) — the *best* INT8 recipe. The older
# `edge_benchmark.json` used quantize_dynamic() defaults and is SUPERSEDED; do not use it.
EDGE = [  # (model, params, macs_G, onnx_fp32_ms, fp32_mb, int8_static_ms, int8_mb, rich_zeroshot_acc)
    ("MC2-S0", 11.41, 1.839,  17.35,  45.81,  62.18, 12.92, 0.270),
    ("MC-S1",  21.54, 3.587,  33.74,  86.51,  79.51, 24.29, 0.224),
    ("MC2-S2", 35.82, 6.026,  49.19, 143.62,  98.69, 39.16, 0.287),
    ("MC-B",   86.35, 16.993, 100.78, 345.55, 46.39, 87.40, 0.268),
]


def fig_efficiency_curve():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = [p for _, p, _ in PROBE]
    ys = [a for _, _, a in PROBE]
    ax.scatter(xs, ys, s=60, color="#1f77b4", label="pretrained CLIP (frozen, zero-shot)", zorder=3)
    for name, p, a in PROBE:
        ax.annotate(name.replace("MobileCLIP", "MC"), (p, a), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    # highlights
    ax.scatter([11.41], [0.199], s=160, facecolors="none", edgecolors="#d62728", linewidths=2,
               label="MobileCLIP2-S0 (~11M, deploy tier)", zorder=4)
    ax.scatter([FROM_SCRATCH[0]], [FROM_SCRATCH[1]], s=90, color="#7f7f7f", marker="x",
               label="from-scratch 5M student (failed)", zorder=4)
    ax.axhline(CHANCE, ls="--", color="grey", lw=1, label=f"chance ({CHANCE:.1%})")
    ax.set_xscale("log")
    ax.set_xlabel("image-encoder parameters (M, log scale)")
    ax.set_ylabel("cross-crop zero-shot accuracy (17 classes)")
    ax.set_title("Accuracy is nearly flat from 11M to 300M parameters")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "fig_efficiency_curve.png", dpi=160); plt.close(fig)


def fig_arch_comparison():
    fig, ax = plt.subplots(figsize=(6, 4))
    names = ["chance", "from-scratch\n5M student", "MobileCLIP2-S0\n11M (frozen)", "SigLIP2\n93M (teacher)"]
    vals = [CHANCE, FROM_SCRATCH[1], 0.199, 0.256]
    colors = ["#bbbbbb", "#7f7f7f", "#d62728", "#2ca02c"]
    ax.bar(names, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f"{v:.1%}", ha="center", fontsize=9)
    ax.set_ylabel("cross-crop zero-shot accuracy")
    ax.set_title("Pretrained alignment, not model size, is what matters")
    ax.set_ylim(0, 0.30)
    fig.tight_layout(); fig.savefig(FIG / "fig_arch_comparison.png", dpi=160); plt.close(fig)


def fig_specialization_forgetting():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(SPEC_EPOCH, SPEC_HELD, "-o", color="#d62728", label="specialized (adapter)")
    ax.axhline(SPEC_BASE, ls="--", color="#1f77b4", label=f"frozen base ({SPEC_BASE:.1%})")
    ax.axhline(CHANCE, ls=":", color="grey", label=f"chance ({CHANCE:.1%})")
    ax.set_xlabel("epoch"); ax.set_ylabel("held-out zero-shot accuracy")
    ax.set_title("Specializing on seen crops degrades unseen-crop zero-shot")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "fig_specialization_forgetting.png", dpi=160); plt.close(fig)


def fig_descriptors():
    """Built only if the descriptor-test result JSON is available (run in the morning)."""
    for cand in [Path("/kaggle/working/phase0_descriptors_result.json"),
                 HERE / "phase0_descriptors_result.json"]:
        if cand.exists():
            data = json.loads(cand.read_text())
            break
    else:
        print("  (descriptors fig skipped — run phase0_descriptors.py, drop its JSON next to this script)")
        return
    models = list(data["models"].keys())
    strategies = ["bare", "crude", "rich"]
    colors = {"bare": "#bbbbbb", "crude": "#7f7f7f", "rich": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.25
    xs = range(len(models))
    for j, s in enumerate(strategies):
        vals = [data["models"][m][s]["acc"] for m in models]
        ax.bar([x + (j - 1) * w for x in xs], vals, width=w, label=s, color=colors[s])
    ax.axhline(data.get("chance", CHANCE), ls="--", color="grey", lw=1, label="chance")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([m.split("/")[0] for m in models], fontsize=8)
    ax.set_ylabel("cross-crop zero-shot accuracy")
    ax.set_title("Descriptor quality: bare vs crude vs source-grounded-style")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig_descriptors.png", dpi=160); plt.close(fig)


def fig_bakeoff():
    fig, ax = plt.subplots(figsize=(6.5, 4))
    items = sorted(BAKEOFF, key=lambda x: x[2])
    names = [f"{n}\n{p:.0f}M" for n, p, _ in items]
    vals = [a for _, _, a in items]
    colors = ["#2ca02c" if "SigLIP" in n else ("#bbbbbb" if "BioCLIP" in n else "#1f77b4")
              for n, _, _ in items]
    ax.barh(names, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(v + 0.005, i, f"{v:.1%}", va="center", fontsize=8)
    ax.axvline(BAKEOFF_CHANCE, ls="--", color="grey", lw=1, label=f"chance ({BAKEOFF_CHANCE:.1%})")
    ax.set_xlabel("rich-descriptor zero-shot accuracy (held-out crops)")
    ax.set_title("Encoder bake-off (17 cls): SigLIP2 best; lightweight 22-29%; BioCLIP2 poor")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig_bakeoff.png", dpi=160); plt.close(fig)


def fig_hybrid():
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["SEEN\ntrained head", "SEEN\nzero-shot", "UNSEEN\nzero-shot"]
    vals = [HYBRID["seen_trained"], HYBRID["seen_zeroshot"], HYBRID["unseen_zeroshot"]]
    colors = ["#2ca02c", "#bbbbbb", "#1f77b4"]
    ax.bar(labels, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.1%}", ha="center", fontsize=10)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 0.8)
    ax.set_title("Hybrid (11M MobileCLIP2-S0): train for seen, zero-shot for unseen")
    fig.tight_layout(); fig.savefig(FIG / "fig_hybrid.png", dpi=160); plt.close(fig)


def fig_wiseft():
    """EXP3: the seen<->unseen tradeoff as WiSE-FT alpha sweeps 0 (frozen) -> 1 (full fine-tune)."""
    alphas = [a for a, _, _ in WISEFT]
    seen = [s for _, s, _ in WISEFT]
    unseen = [u for _, _, u in WISEFT]
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.plot(alphas, seen, "-o", color="#2ca02c", label="SEEN (trained-crop) accuracy")
    ax.plot(alphas, unseen, "-o", color="#1f77b4", label="UNSEEN (zero-shot) accuracy")
    for a, s, u in WISEFT:
        ax.text(a, s + 0.015, f"{s:.0%}", ha="center", fontsize=8, color="#2ca02c")
        ax.text(a, u - 0.03, f"{u:.0%}", ha="center", fontsize=8, color="#1f77b4")
    ax.axvline(0.5, ls=":", color="grey", lw=1)
    ax.text(0.5, 0.02, "WiSE-FT\nsweet spot", ha="center", fontsize=8, color="grey")
    ax.set_xlabel(r"WiSE-FT $\alpha$  (0 = frozen, 1 = full fine-tune)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 0.9)
    ax.set_title("Fine-tuning trades unseen zero-shot for seen accuracy (catastrophic forgetting)")
    ax.legend(fontsize=8, loc="center right"); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "fig_wiseft.png", dpi=160); plt.close(fig)


def fig_descriptor_ablation():
    """Held-out zero-shot per model across bare/crude/rich/grounded — the descriptor-detail lever."""
    strategies = ["bare", "crude", "rich", "grounded"]
    colors = ["#bbbbbb", "#9ecae1", "#2ca02c", "#1f77b4"]
    n = len(DESC_ABLATION); w = 0.2
    xs = range(n)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for j, s in enumerate(strategies):
        vals = [row[2 + j] for row in DESC_ABLATION]
        ax.bar([x + (j - 1.5) * w for x in xs], vals, width=w, label=s, color=colors[j])
    ax.axhline(CHANCE, ls="--", color="grey", lw=1, label=f"chance ({CHANCE:.1%})")
    ax.set_xticks(list(xs)); ax.set_xticklabels([r[0] for r in DESC_ABLATION])
    ax.set_ylabel("held-out zero-shot accuracy (17 cls)")
    ax.set_ylim(0, 0.34)
    ax.set_title("Descriptor detail is the lever: any full description >> class-name (bare)")
    ax.legend(fontsize=8, ncol=5, loc="upper center")
    fig.tight_layout(); fig.savefig(FIG / "fig_descriptor_ablation.png", dpi=160); plt.close(fig)


def fig_edge_pareto():
    """Accuracy vs on-device latency; point size = params, label = INT8 size. S0 is the sweet spot."""
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    for name, p, macs, fp32, fp32mb, int8ms, int8mb, acc in EDGE:
        ax.scatter(fp32, acc, s=40 + p * 3, color="#1f77b4", zorder=3)
        ax.annotate(f"{name}\n{p:.0f}M · {int8mb:.0f}MB int8", (fp32, acc), fontsize=7,
                    xytext=(7, -3), textcoords="offset points")
    ax.scatter([EDGE[0][3]], [EDGE[0][7]], s=240, facecolors="none", edgecolors="#d62728",
               linewidths=2, label="S0 deploy tier (fastest & smallest)", zorder=4)
    ax.set_xlabel("ONNX FP32 latency, laptop CPU (ms/image, batch 1)")
    ax.set_ylabel("held-out zero-shot accuracy")
    ax.set_title("Real-time Pareto: 11M S0 = 15.8 ms/img (~63 img/s) at ~equal accuracy")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "fig_edge_pareto.png", dpi=160); plt.close(fig)


def fig_scaling():
    """Zero-shot accuracy vs #held-out classes across Experiments A/B/C (the scale study)."""
    pts = []
    for e in ("A", "B", "C"):
        cands = [HERE / f"zeroshot_eval_{e}.json", Path(f"C:/kaggle/working/results/zeroshot_eval_{e}.json"),
                 Path(f"results/zeroshot_eval_{e}.json")]
        p = next((c for c in cands if c.exists()), None)
        if not p:
            continue
        d = json.loads(p.read_text())
        key = next((k for k in d["models"] if "S0" in k), None)
        if not key:
            continue
        row = {"exp": e, "n": d["n_classes"], "chance": d["chance"]}
        for s in ("rich", "grounded"):
            if s in d["models"][key]:
                row[s] = d["models"][key][s]["acc"]
        pts.append(row)
    if len(pts) < 2:
        print("  (scaling fig skipped — need >=2 of zeroshot_eval_{A,B,C}.json)")
        return
    fig, ax = plt.subplots(figsize=(6.2, 4))
    xs = [p["n"] for p in pts]
    for s, col in (("rich", "#2ca02c"), ("grounded", "#1f77b4")):
        ys = [p.get(s) for p in pts]
        if all(y is not None for y in ys):
            ax.plot(xs, ys, "-o", color=col, label=f"S0 {s}")
    ax.plot(xs, [p["chance"] for p in pts], "--", color="grey", label="chance")
    for p in pts:
        ax.annotate(f"Exp {p['exp']}", (p["n"], max(p.get("rich", 0), p.get("grounded", 0))),
                    fontsize=8, xytext=(4, 5), textcoords="offset points")
    ax.set_xlabel("# held-out disease classes")
    ax.set_ylabel("cross-crop zero-shot accuracy (S0, 11M)")
    ax.set_title("Scaling: zero-shot degrades gracefully as held-out classes grow (A→B→C)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "fig_scaling.png", dpi=160); plt.close(fig)


def fig_riskcoverage():
    """Selective accuracy vs coverage (abstain gate) for S0, from metrics_abstain.json if present."""
    cands = [HERE / "metrics_abstain.json", Path("C:/kaggle/working/results/metrics_abstain.json"),
             Path("results/metrics_abstain.json")]
    path = next((c for c in cands if c.exists()), None)
    if path is None:
        print("  (risk-coverage fig skipped — no metrics_abstain.json)")
        return
    data = json.loads(path.read_text())
    key = next((k for k in data["models"] if "S0" in k), None)
    if not key:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4))
    colors = {"rich": "#2ca02c", "grounded": "#1f77b4"}
    for strat, col in colors.items():
        cur = data["models"][key].get(strat, {}).get("risk_coverage_curve")
        if cur:
            xs = [p[0] for p in cur]; ys = [p[1] for p in cur]
            ax.plot(xs, ys, "-", color=col, label=f"{strat} (top-1 {data['models'][key][strat]['top1']:.0%})")
    ax.axhline(data.get("chance", CHANCE), ls="--", color="grey", lw=1, label="chance")
    ax.set_xlabel("coverage (fraction of images answered)")
    ax.set_ylabel("selective accuracy")
    ax.set_title("Abstain gate (S0, margin confidence): accuracy rises as coverage drops")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    fig.tight_layout(); fig.savefig(FIG / "fig_riskcoverage.png", dpi=160); plt.close(fig)


def main():
    fig_efficiency_curve()
    fig_arch_comparison()
    fig_specialization_forgetting()
    fig_bakeoff()
    fig_hybrid()
    fig_wiseft()
    fig_descriptor_ablation()
    fig_edge_pareto()
    try:
        fig_riskcoverage()
    except Exception as e:
        print(f"  (risk-coverage fig skipped: {e})")
    try:
        fig_scaling()
    except Exception as e:
        print(f"  (scaling fig skipped: {e})")
    try:
        fig_descriptors()
    except Exception as e:
        print(f"  (descriptors fig skipped: {e})")
    print(f"figures written to {FIG}")


if __name__ == "__main__":
    main()
