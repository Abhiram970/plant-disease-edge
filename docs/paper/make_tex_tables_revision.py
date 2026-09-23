"""
Revision tables (docs/paper/revision/tab_*.tex), for the manuscript built by
scripts/paper_fixes/apply_revision_text.py.

It reuses make_tex_tables.py rather than copying it: the unchanged tables are produced by the
same code that produced the 2026-09-22 ones, only written to a different directory. docs/paper/tex
is never written to, so the submitted build stays frozen.

What differs from the 2026-09-22 tables:
  * arms are labelled DCLIP+ / CuPL+ (our prompts add three constraints to the published forms);
  * Table 1 gains an encoder-spread row (max - min over the four encoders under the best
    strategy), because the endpoint difference alone understates the encoder effect, and a
    majority-prior row once revision_numbers.json exists;
  * Table 2 reports the decomposition at A and B as well as C -- those numbers were already in
    paper_numbers.json, they were simply not shown;
  * a CuPL+ shortlist/abstention table is written when the Phase-2 re-score has produced
    revision_results/metrics_abstain_*.json.

    python docs/paper/make_tex_tables_revision.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import make_tex_tables as M  # noqa: E402

REVISION = HERE / "revision"
REVISION_RESULTS = HERE / "revision_results"
REVISION.mkdir(parents=True, exist_ok=True)

M.TEX = REVISION
M.AUTH_ROWS = [("bare", "bare"), ("bare80", "bare80"), ("crude", "crude"), ("rich", "rich"),
               ("grounded", "grounded"), ("dclip", "DCLIP+"), ("cupl", "CuPL+")]

DEPLOY = ["MobileCLIP2-S0", "MobileCLIP-S1", "MobileCLIP2-S2", "MobileCLIP-B"]


def _revision_numbers():
    """Phase-2 output, or None when the re-score has not been run yet."""
    p = HERE / "revision_numbers.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _small(lines):
    """Insert \\small before the tabular. Table 2 carries three more numeric columns than the
    2026-09-22 version and overflows the cas-dc table* measure by 28.7pt at normal size."""
    out = list(lines)
    out.insert(out.index("\\centering") + 1, "\\small")
    return out


def tab_authoring():
    """Table 1 plus the two rows the audit asked for."""
    N = M._numbers()
    R = _revision_numbers()
    cols = [(e, t) for t in ("uncleaned", "clean") for e in "ABC"]
    best = {(e, t): N["configs"][e][t]["best"] for e, t in cols}
    body = []
    for key, name in M.AUTH_ROWS:
        cells = []
        for e, t in cols:
            v = N["configs"][e][t]["means"][key]
            s = f"{v:.1f}"
            cells.append(f"\\textbf{{{s}}}" if best[(e, t)] == key else s)
        body.append(f"{name} & " + " & ".join(cells) + " \\\\")
    body.append("\\midrule")

    ag = [f"{N['configs'][e][t]['authoring_gain']:.1f}" for e, t in cols]
    sg = [f"{N['configs'][e][t]['size_range_under_best']:.1f}" for e, t in cols]
    # Spread across the four deployable encoders under the same (best) strategy. The endpoint
    # difference in the row above is one particular pair; this is the range a deployer chooses
    # from, and at configuration A it is the larger of the two.
    spread = []
    for e, t in cols:
        pe = N["configs"][e][t]["per_encoder"]
        vals = [pe[m][best[(e, t)]] for m in DEPLOY]
        spread.append(f"{max(vals) - min(vals):.1f}")
    body.append("Authoring gain (best $-$ bare) & " + " & ".join(ag) + " \\\\")
    body.append("Encoder gain (86.3 $-$ 11.4\\,M, best strategy) & " + " & ".join(sg) + " \\\\")
    body.append("Encoder spread (max $-$ min of four, best strategy) & "
                + " & ".join(spread) + " \\\\")
    if R and "majority_prior" in R:
        mp = R["majority_prior"]
        cells = [f"{mp[t][e]:.1f}" for e, t in cols]
        body.append("\\midrule")
        body.append("Majority-class prior & " + " & ".join(cells) + " \\\\")

    n_cls = [str(N["configs"][e][t]["n_classes"]) for e, t in cols]
    header = ("& \\multicolumn{3}{c}{Full label set} & \\multicolumn{3}{c}{De-duplicated} \\\\\n"
              "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
              "Strategy & " + " & ".join(f"{e} ({n})" for (e, _), n in zip(cols, n_cls)) + " \\\\")
    cap = ("Unseen-crop top-1 accuracy (\\%) by descriptor strategy: mean over the four deployable "
           "encoders, at three nested label spaces (classes in parentheses), on the full and the "
           "de-duplicated label sets. The best strategy in each column is in bold, and is chosen "
           "per column, so the authoring gain below it is a best-of-seven. The encoder gain is the "
           "11.4\\,M to 86.3\\,M endpoint difference under that same strategy; the encoder spread "
           "is the range over all four encoders, which is the wider quantity at configuration A.")
    M.write("tab_authoring.tex", M.wrap(cap, "tab:authoring", "lrrrrrr", header, body, wide=True))


def tab_decomp():
    """Table 2 at all three configurations. The A and B values were already computed."""
    N = M._numbers()
    cols = [(e, t) for t in ("uncleaned", "clean") for e in "ABC"]

    def row(label, comparison, held, key):
        cells = [f"{N['configs'][e][t][key]:+.1f}" for e, t in cols]
        return f"{label} & {comparison} & {held} & " + " & ".join(cells) + " \\\\"

    body = [
        row("Construction", "grounded $\\rightarrow$ split into sentences", "text",
            "split_minus_grounded"),
        row("Text", "split $\\rightarrow$ CuPL+", "construction", "cupl_minus_split"),
        "\\midrule",
        row("Total gap", "grounded $\\rightarrow$ CuPL+", "---", "cupl_minus_grounded"),
    ]
    n_cls = [str(N["configs"][e][t]["n_classes"]) for e, t in cols]
    header = ("& & & \\multicolumn{3}{c}{Full label set} & \\multicolumn{3}{c}{De-duplicated} \\\\\n"
              "\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}\n"
              "Factor & Comparison & Held fixed & "
              + " & ".join(f"{e} ({n})" for (e, _), n in zip(cols, n_cls)) + " \\\\")
    cap = ("Decomposition of the accuracy gap between source-grounded text and CuPL+, in points of "
           "mean top-1 over the four deployable encoders, at all three configurations. Each row "
           "changes one factor and holds the other fixed; construction and text sum to the total "
           "gap before rounding, so the rounded entries may differ from it by 0.1. The pattern is "
           "not uniform: at 51 classes construction explains almost none of the gap, while at 16 "
           "classes splitting the paragraph costs 3.0 points, more than the whole gap there. The "
           "citation requirement is tested separately by a seeded control "
           "(Section~\\ref{sec:decomp}): removing it changes accuracy by an amount whose 95\\% "
           "interval includes zero.")
    M.write("tab_decomp.tex",
            _small(M.wrap(cap, "tab:decomp", "lllrrrrrr", header, body, wide=True)))


def tab_abstain_cupl():
    """Shortlist and abstention for CuPL+, from the Phase-2 re-score. Skipped until it is run."""
    body = []
    for e in "ABC":
        p = REVISION_RESULTS / f"metrics_abstain_{e}.json"
        if not p.exists():
            continue
        j = json.loads(p.read_text(encoding="utf-8"))
        zs = (M.load(f"zeroshot_eval_{e}.json") or {}).get("models", {})
        for m, d in j["models"].items():
            sd = d.get("cupl")
            if not isinstance(sd, dict):
                continue
            # Top-1 comes from the full-precision zero-shot file, as in the grounded/rich table.
            top1 = zs.get(m, {}).get("cupl", {}).get("acc", sd.get("top1"))
            aurc = sd.get("aurc")
            body.append(" & ".join([e, str(j["n_classes"]), M.short(m), M.pct(top1),
                                    M.pct(sd.get("top5")),
                                    "---" if aurc is None else f"{aurc:.4f}",
                                    M.pct(sd.get("acc@cov80"))]) + " \\\\")
    if not body:
        # Remove a table left over from an earlier partial run. main.tex \input{}s this file once
        # the Phase-2 prose patch has run, and a stale one would be a table of old numbers under
        # a caption describing new ones.
        stale = REVISION / "tab_abstain_cupl.tex"
        if stale.exists():
            stale.unlink()
            print(f"[tex] removed stale {stale.name} (no revision_results to regenerate it from)")
        print("[tex] tab_abstain_cupl skipped - run scripts/rescore_descriptors.py first")
        return
    M.write("tab_abstain_cupl.tex", M.wrap(
        "Top-5 and selective prediction for CuPL+, the recommended registry, on the same "
        "coverage grid as Table~\\ref{tab:abstain}. acc@cov80 = accuracy when the 80\\% most "
        "confident predictions are kept; AURC = area under the risk--coverage curve, lower is "
        "better. The reference encoder is omitted here, as it is from every quoted mean.",
        "tab:abstaincupl", "lllrrrr",
        "Config & Classes & Model & Top-1 & Top-5 & AURC $\\downarrow$ & acc@cov80 \\\\",
        body,
        "Confidence is the top-1 minus top-2 similarity margin, as in Table~\\ref{tab:abstain}.",
        wide=True))


def main() -> int:
    for fn in (tab_authoring, tab_decomp, tab_abstain_cupl,
               M.tab_abstain, M.tab_edge, M.tab_scale_study, M.tab_seen, M.tab_supervised,
               M.tab_wiseft, M.tab_loco):
        try:
            fn()
        except Exception as e:  # same tolerance as the 2026-09-22 generator
            print(f"[tex] FAILED {fn.__name__}: {type(e).__name__}: {e}")
    stamp_old = "% GENERATED by docs/paper/make_tex_tables.py -- do not edit by hand."
    stamp_new = "% GENERATED by docs/paper/make_tex_tables_revision.py -- do not edit by hand."
    for f in REVISION.glob("tab_*.tex"):
        s = f.read_text(encoding="utf-8").replace(stamp_old, stamp_new, 1)
        with open(f, "w", encoding="utf-8", newline="") as fh:
            fh.write(s)

    # The monotonicity footnote counts the cells that were measured. CuPL+ adds 12 of them
    # (4 deployable encoders x 3 configurations), all of which rise, so the count is 42, not 30.
    # Guarded: if the upstream wording changes, this fails loudly instead of silently missing.
    ab = REVISION / "tab_abstain.tex"
    if ab.exists() and (REVISION / "tab_abstain_cupl.tex").exists():
        s = ab.read_text(encoding="utf-8")
        old = "in all 30 cells"
        new = "in all 42 cells of Tables~\\ref{tab:abstain} and~\\ref{tab:abstaincupl}"
        if old not in s:
            print(f"[tex] WARNING: {ab.name} no longer contains {old!r}; check the footnote")
        else:
            with open(ab, "w", encoding="utf-8", newline="") as fh:
                fh.write(s.replace(old, new, 1))
            print(f"[tex] tab_abstain footnote: 30 -> 42 cells")
    R = _revision_numbers()
    if not R:
        print("[tex] note: revision_numbers.json absent, so Table 1 has no majority-prior row yet "
              "(run scripts/rescore_descriptors.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
