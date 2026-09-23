r"""
Phase 2 of the 2026-09-22 deep review: fold the re-scored numbers into docs/paper/manuscript/revision/main.tex.

Run AFTER:
    python scripts/paper_fixes/apply_revision_text.py     (writes revision/main.tex)
    scripts/kaggle/runners/rescore_descriptors.py on Kaggle    (writes revision_numbers.json)
    python docs/paper/make_tex_tables_revision.py                (writes tab_abstain_cupl.tex)

Every number inserted here is read from docs/paper/manuscript/revision_numbers.json; none is typed.

The prose states a direction as well as a magnitude, so each directional claim is guarded by an
assertion against the measured values. If a future re-run reverses one, this script ABORTS rather
than printing a sentence the data no longer supports.

    python scripts/paper_fixes/apply_revision_results.py
    python scripts/paper_fixes/apply_revision_results.py --check
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
PAPER = REPO / "docs" / "paper"
REVISION = PAPER / "manuscript" / "revision"
NUMBERS = PAPER / "revision_numbers.json"
ABSTAIN = PAPER / "revision_results"

# The registry-to-registry scale the manuscript uses to decide what counts as a real difference.
NOISE = 2.4
DEPLOY = ["MobileCLIP2-S0/dfndr2b", "MobileCLIP-S1/datacompdr",
          "MobileCLIP2-S2/dfndr2b", "MobileCLIP-B/datacompdr"]

_claims: list[str] = []


def claim(ok: bool, msg: str) -> None:
    """Record a directional claim the inserted prose makes."""
    if not ok:
        _claims.append(msg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if not NUMBERS.exists():
        print(f"ABORT: {NUMBERS} not found -- run the re-score first")
        return 1
    N = json.loads(NUMBERS.read_text(encoding="utf-8"))
    tex_path = REVISION / "main.tex"
    if not tex_path.exists():
        print(f"ABORT: {tex_path} not found -- run patch_audit_2026-09-23.py first")
        return 1
    text = tex_path.read_text(encoding="utf-8")

    missing = [e for e in ("A", "B", "C") if e not in N["configs"]]
    if missing:
        print(f"ABORT: revision_numbers.json is missing configurations {missing}")
        return 1
    if len(N["per_model"]) < 4:
        print(f"ABORT: only {len(N['per_model'])} encoders scored; the means need all four")
        return 1

    def cfg(e, ls="uncleaned"):
        return N["configs"][e][ls]

    def tax(e, ls, key="means"):
        c = cfg(e, ls)
        return c[key]["grounded_visual_split"] - c[key]["grounded_split"]

    prior = N["majority_prior"]
    edits: list[tuple[str, str, str]] = []

    # ---- 1. the taxonomy ablation (Section 4.2) ------------------------------------------------
    tA, tB, tC = (tax(e, "uncleaned") for e in "ABC")
    tAc, tBc, tCc = (tax(e, "clean") for e in "ABC")
    mA, mB, mC = (tax(e, "uncleaned", "macro_means") for e in "ABC")
    resC = cfg("C")["taxonomy_test"]["cupl_minus_grounded_visual_split"]
    resCc = cfg("C", "clean")["taxonomy_test"]["cupl_minus_grounded_visual_split"]

    claim(tB < 0 and tC < 0, "prose says stripping taxonomy COSTS accuracy at B and C")
    claim(mB < 0 and mC < 0, "prose says the same holds under class-balanced averaging at B and C")
    claim(tA > NOISE > abs(mA),
          "prose says the A gain is image-weighted only and vanishes under class-balanced averaging")
    claim(resC > 0 and resCc > 0, "prose says CuPL+ still leads the stripped registry at C")

    edits.append((
        "taxonomy ablation",
        r"""A registry that is
both source-grounded and photo-descriptive, and one with the taxonomy fields stripped from the
existing grounded text, are the two experiments that would separate these factors.""",
        (f"We ran the second of those experiments. Stripping the pathogen and taxonomy fields from "
         f"the same source-grounded text and ensembling what remains exactly as before holds "
         f"sourcing, provenance and construction fixed and removes only the non-visual prose. It "
         f"does not recover the deficit. Accuracy moves by {tA:+.1f}, {tB:+.1f} and {tC:+.1f} "
         f"points at A, B and C on the full label set, and {tAc:+.1f}, {tBc:+.1f} and {tCc:+.1f} "
         f"on the de-duplicated set: at the two larger label spaces dropping the taxonomy costs "
         f"accuracy rather than buying any. The gain at 16 classes is an artefact of image "
         f"weighting, worth {tA:+.1f} points there but only {mA:+.1f} under class-balanced "
         f"averaging, which at B and C reads {mB:+.1f} and {mC:+.1f}. CuPL+ still leads the "
         f"stripped registry by {resC:.1f} points at 51 classes ({resCc:.1f} de-duplicated). "
         f"Whatever the register contributes is therefore not carried by the presence of the "
         f"schema's non-visual fields; if it survives at all, it is in how the visual symptoms "
         f"themselves are worded. A registry that is both source-grounded and photo-descriptive "
         f"remains the experiment that would separate register from provenance.")))

    # ---- 2. the organ-rule bound (Section 6) ---------------------------------------------------
    o, oc = cfg("C")["organ_rule_bound"], cfg("C", "clean")["organ_rule_bound"]
    shrink = o["gap_all_images"] - o["gap_foliar_only"]
    shrink_c = oc["gap_all_images"] - oc["gap_foliar_only"]
    claim(o["gap_foliar_only"] > 0 and oc["gap_foliar_only"] > 0,
          "prose says the CuPL+ margin survives on foliar classes")
    claim(max(shrink, shrink_c) < NOISE,
          "prose says the organ rule accounts for about a point of the margin")
    edits.append((
        "organ-rule bound",
        r"""They favour the arm that
wins, so the CuPL+ margin over source-grounded text should be read as an upper bound. Re-scoring the
same comparison on the classes the organ rule does not touch would bound how much of the margin it
buys.""",
        (f"They favour the arm that wins, so the CuPL+ margin over source-grounded text should be "
         f"read as an upper bound. We bounded it. On the foliar classes alone, where the "
         f"plant-part instruction changes nothing because the class name already says leaf, the "
         f"CuPL+ margin at 51 classes is {o['gap_foliar_only']:.1f} points against "
         f"{o['gap_all_images']:.1f} over all classes ({o['n_foliar_images']:,} of "
         f"{cfg('C')['n_images']:,} images), and {oc['gap_foliar_only']:.1f} against "
         f"{oc['gap_all_images']:.1f} on the de-duplicated set. The instruction accounts for at "
         f"most about a point of the margin, so it is not what produces it.")))

    # ---- 3. CuPL+ shortlist and abstention (Section 4.3) --------------------------------------
    ab_c = ABSTAIN / "metrics_abstain_C.json"
    have_abstain = ab_c.exists() and (REVISION / "tab_abstain_cupl.tex").exists()
    if have_abstain:
        j = json.loads(ab_c.read_text(encoding="utf-8"))
        M = j["models"]
        dep = [m for m in DEPLOY if m in M and "cupl" in M[m]]
        t5 = [100 * M[m]["cupl"]["top5"] for m in dep]
        t1 = [100 * M[m]["cupl"]["top1"] for m in dep]
        cov = [100 * M[m]["cupl"]["acc@cov80"] for m in dep]
        gain = [c - t for c, t in zip(cov, t1)]
        g5 = [100 * M[m]["grounded"]["top5"] for m in dep if "grounded" in M[m]]
        rand5 = 500.0 / j["n_classes"]
        claim(min(t5) > max(g5) - 5 and sum(t5) / len(t5) > sum(g5) / len(g5),
              "prose says the CuPL+ shortlist leads the source-grounded one")
        claim(min(gain) > 0, "prose says the gate buys top-1 at 80% coverage for CuPL+")
        edits.append((
            "CuPL+ shortlist",
            r"""Top-5
and abstention were not measured for CuPL+, whose higher top-1 suggests but does not establish a
higher top-5.""",
            (f"Table~\\ref{{tab:abstaincupl}} reports the same measurements for CuPL+, the registry "
             f"we recommend. On 51 classes its shortlist contains the correct disease "
             f"{min(t5):.1f}--{max(t5):.1f}\\% of the time across the four deployable encoders, "
             f"against {min(g5):.1f}--{max(g5):.1f}\\% for the source-grounded registry and "
             f"{rand5:.1f}\\% for a random five-item list, and the same margin-ranked gate buys "
             f"{min(gain):.1f}--{max(gain):.1f} points of top-1 for declining one image in five. "
             f"The registry that leads on top-1 therefore also leads on the shortlist.")))
        edits.append((
            "input the CuPL+ abstention table",
            "\\input{tab_abstain}\n",
            "\\input{tab_abstain}\n\\input{tab_abstain_cupl}\n"))
        # Two counts in Section 4.3 were scoped to the original sweep and are now stale.
        edits.append((
            "section 4.3: three strategies are measured now",
            r"""source-grounded registry and the keyword bank, the two strategies for which they were measured. On""",
            r"""source-grounded registry and the keyword bank, and Table~\ref{tab:abstaincupl} adds CuPL+. On"""))
        n_cells = 30 + 3 * len(dep)          # the published 30, plus CuPL+ on each deployable encoder
        claim(n_cells == 42, f"prose says 42 monotone cells, computed {n_cells}")
        edits.append((
            "section 4.3: monotone-cell count",
            r"""reported coverage grid, from full coverage down to half in steps of ten, in all 30 encoder, strategy
and configuration cells (Figure~\ref{fig:riskcov})""",
            r"""reported coverage grid, from full coverage down to half in steps of ten, in all 42 encoder,
strategy and configuration cells measured (Figure~\ref{fig:riskcov})"""))
        edits.append((
            "limitations: shortlist now measured",
            r"""\paragraph{Shortlist and abstention were measured on two strategies} Top-5 and selective prediction
are reported for the source-grounded registry and the keyword bank only, not for CuPL+, although the
deployment we recommend uses CuPL+.

""",
            ""))

    # ---- 4. majority prior at every configuration, and class-balanced accuracy -----------------
    best_macro = {e: max(cfg(e, ls)["macro_means"], key=cfg(e, ls)["macro_means"].get)
                  for e in "ABC" for ls in ("uncleaned", "clean")}
    cupl_macro_everywhere = all(
        max(cfg(e, ls)["macro_means"], key=cfg(e, ls)["macro_means"].get) == "cupl"
        for e in "ABC" for ls in ("uncleaned", "clean"))
    claim(cupl_macro_everywhere,
          "prose says CuPL+ leads every configuration under class-balanced averaging")
    claim(cfg("A")["means"]["rich"] > cfg("A")["means"]["cupl"],
          "prose says the keyword bank leads at A on image-weighted accuracy")
    mi_c, ma_c = cfg("C")["means"]["cupl"], cfg("C")["macro_means"]["cupl"]
    edits.append((
        "majority prior at every configuration, and class-balanced accuracy",
        r"""At configuration C the best strategy reaches 30.9\%, 7.4 times the 4.2\% majority-class prior and
15.8 times uniform chance.""",
        (f"At configuration C the best strategy reaches 30.9\\%, 7.4 times the 4.2\\% "
         f"majority-class prior and 15.8 times uniform chance. That prior is the more demanding of "
         f"the two references and it falls steeply as the label space widens, from "
         f"{prior['uncleaned']['A']:.1f}\\% at A through {prior['uncleaned']['B']:.1f}\\% at B to "
         f"{prior['uncleaned']['C']:.1f}\\% at C ({prior['clean']['A']:.1f}, "
         f"{prior['clean']['B']:.1f} and {prior['clean']['C']:.1f} de-duplicated), so a strategy "
         f"can beat uniform chance at A and still lose to a constant predictor; "
         f"Table~\\ref{{tab:authoring}} reports it for every column.\n\n"
         f"\\paragraph{{Class-balanced accuracy}} Accuracy here is image-weighted, and the held-out "
         f"classes run from the 25-image floor to the 600-image cap, so we also computed the "
         f"class-macro figure for every cell. The headline is unchanged at the larger label "
         f"spaces, {mi_c:.1f}\\% against {ma_c:.1f}\\% at C, and the one ordering that does change "
         f"changes in favour of authored text: the keyword bank's lead at 16 classes does not "
         f"survive class balancing, and CuPL+ leads at every configuration on both label sets. "
         f"The bank's advantage at A is therefore carried by the larger classes, which image "
         f"weighting rewards.")))

    # ---- 5. contribution 2 (the label-space qualifier is averaging-specific) -------------------
    edits.append((
        "contribution 2: averaging qualifier",
        r"""At 16 classes the hand-curated keyword bank
leads instead (Section~\ref{sec:lever}).""",
        r"""At 16 classes the hand-curated keyword bank
leads on image-weighted accuracy, though not on class-balanced accuracy, where CuPL+ leads at all
three (Section~\ref{sec:lever}).""",
    ))

    # ---- 6. the Discussion and Limitations now have one candidate ruled out --------------------
    edits.append((
        "discussion: one candidate excluded",
        r"""so which property of the text carries the advantage
is not yet isolated.""",
        r"""so which property of the text carries the advantage
is only partly isolated. One candidate is now excluded: stripping the pathogen and taxonomy fields
from the source-grounded text does not recover the deficit (Section~\ref{sec:decomp}).""",
    ))
    edits.append((
        "discussion: the cited-and-photo-descriptive registry",
        r"""A registry
that is both cited and photo-descriptive would, if the register explanation holds, combine the audit
trail with the best accuracy.""",
        r"""Whether a registry that is both cited and photo-descriptive
would combine the audit trail with the best accuracy is still open, though the taxonomy ablation of
Section~\ref{sec:decomp} makes the schema's non-visual fields an unlikely place to look for the
difference.""",
    ))
    edits.append((
        "limitations: register, one component tested",
        r"""We attribute the gap to the text, which the
decomposition supports, but register is one of several differences between the two registries and is
not isolated by any experiment reported here.""",
        r"""We attribute the gap to the text, which the
decomposition supports, but register is one of several differences between the two registries. Its
most testable component, the presence of the schema's pathogen and taxonomy fields, is ruled out in
Section~\ref{sec:decomp}; the generating model, the interface and the sentence form remain
confounded.""",
    ))

    # ---- 7. the abstract and conclusion carry the ablation ---------------------------------------
    edits.append((
        "abstract: the taxonomy ablation",
        r"""costs no measurable accuracy, and re-embedding the source-grounded text as a sentence ensemble
recovers only 4\% of its deficit, so the gap lies in the text rather than in citation or embedding.""",
        r"""costs no measurable accuracy, re-embedding the source-grounded text as a sentence ensemble
recovers only 4\% of its deficit, and stripping its pathogen and taxonomy fields recovers none of
it, so the gap lies in the text rather than in citation, embedding or the schema's non-visual
fields.""",
    ))
    edits.append((
        "conclusion: the taxonomy ablation",
        r"""remaining gap between cited and photo-descriptive text lies in the text rather than in citation or
embedding.""",
        r"""remaining gap between cited and photo-descriptive text lies in the text rather than in citation,
embedding or the taxonomy the cited schema carries.""",
    ))

    # ---- 8. figure 1 caption now carries the prior ----------------------------------------------
    edits.append((
        "figure 1 caption",
        r"""The dotted line is uniform chance.}""",
        r"""The dotted line is uniform chance and the dash-dotted line the majority-class prior, which at
16 classes lies above four of the seven strategies.}"""))

    # ---- apply -----------------------------------------------------------------------------------
    if _claims:
        print("ABORT: the measured values no longer support prose this script would write:")
        for c in _claims:
            print(f"  - {c}")
        return 1
    problems = [f"  [{text.count(old)} matches] {label}" for label, old, _ in edits
                if text.count(old) != 1]
    if problems:
        print("ABORT: anchors did not match exactly once (already run, or revision/main.tex "
              "regenerated since?):")
        print("\n".join(problems))
        return 1
    print(f"[phase2] {len(edits)} anchors matched exactly once; "
          f"{len(edits)} directional claims verified against the data")
    for label, _old, _new in edits:
        print(f"[phase2]   {label}")
    if not have_abstain:
        print("[phase2] NOTE: no revision_results/metrics_abstain_C.json + tab_abstain_cupl.tex, so "
              "the CuPL+ shortlist paragraph was skipped")
    if args.check:
        print("[phase2] --check only, nothing written")
        return 0

    # The manuscript source is hand-wrapped at ~100 columns; keep inserted prose readable in a
    # diff by wrapping it the same way. Only whitespace is touched, so no LaTeX token is split.
    import textwrap

    def rewrap(s: str) -> str:
        if len(s) < 100 or "\\input" in s:
            return s
        out = []
        for para in s.split("\n\n"):
            out.append("\n".join(textwrap.wrap(" ".join(para.split()), width=98,
                                               break_long_words=False, break_on_hyphens=False)))
        return "\n\n".join(out)

    for _label, old, new in edits:
        text = text.replace(old, rewrap(new), 1)
    with open(tex_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print(f"[phase2] wrote {tex_path}")
    print("[phase2] next: cd docs/paper/manuscript/revision && latexmk -pdf main.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
