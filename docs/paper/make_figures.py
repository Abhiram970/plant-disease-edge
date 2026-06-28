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
BAKEOFF = [  # (label, img params M, rich zero-shot)
    ("MobileCLIP2-S0", 11.4, 0.371),
    ("MobileCLIP-S1", 21.5, 0.402),
    ("MobileCLIP2-S2", 35.8, 0.346),
    ("BioCLIP2", 304.0, 0.254),
    ("SigLIP2", 92.9, 0.498),
]
BAKEOFF_CHANCE = 0.125
# ---- EXP2 hybrid (MobileCLIP2-S0, 11M): trained seen vs zero-shot seen vs zero-shot unseen ----
HYBRID = {"seen_trained": 0.671, "seen_zeroshot": 0.085, "unseen_zeroshot": 0.371}


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
    ax.set_title("Encoder bake-off: SigLIP2 best; MobileCLIP-S1 best lightweight; BioCLIP2 poor")
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


def main():
    fig_efficiency_curve()
    fig_arch_comparison()
    fig_specialization_forgetting()
    fig_bakeoff()
    fig_hybrid()
    try:
        fig_descriptors()
    except Exception as e:
        print(f"  (descriptors fig skipped: {e})")
    print(f"figures written to {FIG}")


if __name__ == "__main__":
    main()
