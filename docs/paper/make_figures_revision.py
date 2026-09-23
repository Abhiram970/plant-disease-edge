"""
Revision figures (docs/paper/revision/figures/), for the manuscript built by
scripts/paper_fixes/apply_revision_text.py.

Reuses make_figures.py and only redirects its output directory, so docs/paper/figures/ -- the
figures of the frozen 2026-09-22 build -- is never written to.

What differs:
  * the legend labels the arms CuPL+ / DCLIP+ (our prompts add three constraints to the
    published question forms);
  * the scaling figure draws the majority-class prior as well as uniform chance. At 16 classes
    that prior is 14.8%, above four of the seven strategies, and plotting only uniform chance
    (6.25%) flattered every one of them;
  * the risk-coverage figure adds CuPL+ once the Phase-2 re-score has produced
    revision_results/metrics_abstain_C.json.

    python docs/paper/make_figures_revision.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import make_figures as MF  # noqa: E402

REVISION_FIG = HERE / "revision" / "figures"
REVISION_RESULTS = HERE / "revision_results"
REVISION_FIG.mkdir(parents=True, exist_ok=True)

MF.FIG = REVISION_FIG
MF.FINAL_SERIES = [
    ("cupl",     "CuPL+ (photo-descriptive sentences)", "#2a78d6", "o", "-"),
    ("grounded", "grounded (source-cited paragraph)",   "#eb6834", "s", "-"),
    ("dclip",    "DCLIP+ (visual-feature phrases)",     "#1baf7a", "^", "-"),
    ("rich",     "rich (keyword bank)",                 "#4a3aa7", "D", "-"),
    ("bare",     "bare (class name)",                   "#8a8984", "",  "--"),
]

INK, INK2 = MF.INK, MF.INK2
DPI = MF.DPI


def _revision_numbers():
    p = HERE / "revision_numbers.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fig_scaling():
    """make_figures.fig_final_scaling plus the majority-class prior."""
    N = MF._numbers()
    if not N:
        print("  (scaling skipped - run paper_numbers.py)")
        return
    R = _revision_numbers()
    cfg = [N["configs"][e]["uncleaned"] for e in "ABC"]
    x = [c["n_classes"] for c in cfg]
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    for key, label, col, mk, ls in MF.FINAL_SERIES:
        y = [c["means"][key] for c in cfg]
        ax.plot(x, y, ls=ls, marker=mk or None, ms=7, color=col,
                lw=2.0 if key != "bare" else 1.4, label=label,
                zorder=3 if key == "cupl" else 2, mec="white" if mk else None, mew=1.0)
    ax.plot(x, [c["chance"] for c in cfg], ":", color=INK2, lw=1.1, label="uniform chance")
    # The honest floor. Uniform chance alone made configuration A look far better than it is:
    # the largest class there is 14.8% of the images, above bare, bare80, crude and DCLIP+.
    if R and "majority_prior" in R:
        prior = [R["majority_prior"]["uncleaned"][e] for e in "ABC"]
        ax.plot(x, prior, "-.", color=INK2, lw=1.3, label="majority-class prior")
    else:
        # paper_numbers.json carries the prior at A and C only. Joining two of three points
        # would draw a line through a value at B that nobody has computed, so the series waits
        # for the Phase-2 re-score rather than being interpolated.
        print("  (majority-prior line omitted: run scripts/rescore_descriptors.py, which computes it "
              "at all three configurations and on both label sets)")
    cc = cfg[-1]["means"]["cupl"]
    ax.annotate(f"{cc:.1f}%", (x[-1], cc), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=8.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n({e})" for n, e in zip(x, "ABC")])
    ax.set_xlim(x[0] - 4, x[-1] + 7)
    ax.set_ylim(0, 35)
    ax.set_xlabel("unseen classes (nested configuration)", color=INK)
    ax.set_ylabel("top-1 accuracy (%), mean of 4 encoders", color=INK)
    MF._style(ax)
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.27),
              ncol=3, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(REVISION_FIG / "fig_descriptor_scaling.png", dpi=DPI)
    plt.close(fig)
    print(f"[fig] wrote {REVISION_FIG / 'fig_descriptor_scaling.png'}")


def fig_pareto():
    """make_figures.fig_final_pareto with the CuPL+ axis label."""
    N = MF._numbers()
    if not N:
        return
    pe = N["configs"]["C"]["uncleaned"]["per_encoder"]
    col = next(s[2] for s in MF.FINAL_SERIES if s[0] == "cupl")
    full = {"MC2-S0": "MobileCLIP2-S0", "MC-S1": "MobileCLIP-S1",
            "MC2-S2": "MobileCLIP2-S2", "MC-B": "MobileCLIP-B"}
    label_at = {"MobileCLIP2-S0": (0, 13, "center", "bottom"),
                "MobileCLIP-S1": (0, -13, "center", "top"),
                "MobileCLIP2-S2": (0, 13, "center", "bottom"),
                "MobileCLIP-B": (0, 14, "center", "bottom")}
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    plotted = 0
    for short_name, p, macs, fp32, fp32mb, int8ms, int8mb, _pilot in MF.EDGE:
        name = full.get(short_name, short_name)
        if name not in pe:
            raise KeyError(f"edge model {short_name!r} -> {name!r} has no accuracy")
        acc = pe[name]["cupl"]
        plotted += 1
        ax.scatter(fp32, acc, s=30 + p * 2.6, color=col, edgecolor="white", linewidth=1.2, zorder=3)
        dx, dy, ha, va = label_at[name]
        ax.annotate(f"{name}\n{p:.1f}M, {int8mb:.1f}MB INT8", (fp32, acc), xytext=(dx, dy),
                    textcoords="offset points", fontsize=7.5, color=INK, ha=ha, va=va)
    assert plotted == 4, f"expected 4 edge points, plotted {plotted}"
    ax.set_xlabel("FP32 latency on an x86 laptop CPU (ms per image, batch 1)", color=INK)
    ax.set_ylabel("top-1 accuracy (%), CuPL+, 51 classes", color=INK)
    ax.set_xlim(0, 135)
    ax.set_ylim(20, 38)
    MF._style(ax)
    fig.tight_layout()
    fig.savefig(REVISION_FIG / "fig_edge_pareto.png", dpi=DPI)
    plt.close(fig)
    print(f"[fig] wrote {REVISION_FIG / 'fig_edge_pareto.png'}")


def fig_riskcoverage():
    """Risk-coverage at C, adding CuPL+ when the Phase-2 re-score has produced it."""
    path = REVISION_RESULTS / "metrics_abstain_C.json"
    strategies = ("cupl", "grounded", "rich")
    if not path.exists():
        path = HERE / "metrics_abstain_C.json"
        strategies = ("grounded", "rich")
        print("  (risk-coverage: no revision re-score yet, drawing grounded and rich only)")
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    key = next((k for k in data["models"] if "S0" in k), None)
    if not key:
        return
    style = {s[0]: s for s in MF.FINAL_SERIES}
    COV_FLOOR = 0.5
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    ys_all = []
    for strat in strategies:
        cur = data["models"][key].get(strat, {}).get("risk_coverage_curve")
        if not cur:
            continue
        pts = [p for p in cur if p[0] >= COV_FLOOR]
        xs, ys = [p[0] for p in pts], [p[1] * 100 for p in pts]
        ys_all += ys
        _, label, col, mk, ls = style[strat]
        top1 = data["models"][key][strat]["top1"] * 100
        ax.plot(xs, ys, ls=ls, color=col, lw=2.0, label=f"{label}, top-1 {top1:.1f}%",
                marker=mk, markevery=10, ms=6, mec="white", mew=1.0)
    if not ys_all:
        plt.close(fig)
        return
    ax.set_ylim(min(ys_all) - 1.5, max(ys_all) + 1.5)
    ax.set_xlim(1.0, COV_FLOOR)
    ax.set_xlabel("coverage (fraction of images answered)", color=INK)
    ax.set_ylabel("selective accuracy (%)", color=INK)
    MF._style(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(REVISION_FIG / "fig_riskcoverage.png", dpi=DPI)
    plt.close(fig)
    print(f"[fig] wrote {REVISION_FIG / 'fig_riskcoverage.png'}")


def main() -> int:
    fig_scaling()
    MF.fig_final_perencoder()          # labels come from FINAL_SERIES, so this needs no copy
    print(f"[fig] wrote {REVISION_FIG / 'fig_descriptor_ablation.png'}")
    fig_pareto()
    fig_riskcoverage()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
