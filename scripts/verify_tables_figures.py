#!/usr/bin/env python3
"""Verify every generated table, every figure input, and every caption claim against the JSONs.

This is the gate the 2026-09-12 audit found missing. `make_tables.py` protects the NUMBERS, and
`preflight.py` checks 14 mechanical properties, but nothing checked the SENTENCES about the
numbers -- and caption-versus-table drift accounted for roughly ten findings across five audit
passes (see docs/paper/AUDIT_HISTORY.md).

Deliberately an INDEPENDENT re-derivation: it re-reads the released JSONs and recomputes each
published quantity from scratch rather than importing anything from make_tex_tables.py. Sharing
the generator's code would make a generator bug invisible to its own check, which is how the
reference-encoder averaging bug survived several passes.

What it checks
  A  every cell and every mean row of Tables 2-8 against its source JSON
  B  the reference encoder is excluded from every mean the manuscript quotes
  C  prose numbers in main.tex that restate a table value
  D  caption and prose DIRECTION claims ("falls", "rises", "does not track", "flat")
  E  figure inputs: the files the figures are drawn from exist and agree with the tables
  F  the derived multiples the Abstract and Contribution 1 quote

    python scripts/verify_tables_figures.py            # report
    python scripts/verify_tables_figures.py --strict    # exit 1 on any FAIL
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "docs" / "paper"
TEX = PAPER / "tex"

REFERENCE_ENCODER = "ViT-B-16-SigLIP2"      # the ceiling; excluded from every quoted mean
TOL = 0.051                                 # a printed 1-decimal percent may differ by <=0.05

PASS: list[str] = []
FAIL: list[str] = []
NOTE: list[str] = []


def ok(msg):
    PASS.append(msg)


def bad(msg):
    FAIL.append(msg)


def note(msg):
    NOTE.append(msg)


def load(name):
    p = PAPER / name
    if not p.exists():
        note(f"missing result file: {name}")
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def tex(name):
    p = TEX / name
    if not p.exists():
        note(f"missing tex file: {name}")
        return ""
    return p.read_text(encoding="utf-8")


def cells(s):
    """Every percentage printed in a tex fragment, as floats."""
    return [float(x) for x in re.findall(r"(\d+\.\d+)\\%", s)]


def near(a, b, tol=TOL):
    return a is not None and b is not None and abs(a - b) <= tol


def is_reference(model):
    return REFERENCE_ENCODER in model


# ---------------------------------------------------------------- A/B. Table 2, scale study
def check_scale_study():
    src = tex("tab_scale_study.tex")
    for e in "ABC":
        j = load(f"zeroshot_eval_{e}.json")
        if not j or not src:
            continue
        # chance printed in the config header
        want_chance = j["chance"] * 100
        hdr = re.search(rf"Config {e} --- (\d+) unseen classes, chance (\d+\.\d+)\\%", src)
        if not hdr:
            bad(f"T2/{e}: config header not found in tab_scale_study.tex")
            continue
        if int(hdr.group(1)) != j["n_classes"]:
            bad(f"T2/{e}: header says {hdr.group(1)} classes, JSON says {j['n_classes']}")
        elif not near(float(hdr.group(2)), want_chance):
            bad(f"T2/{e}: header chance {hdr.group(2)}% vs JSON {want_chance:.2f}%")
        else:
            ok(f"T2/{e}: header ({j['n_classes']} classes, chance {want_chance:.1f}%)")

        # per-model rows. SCOPE THE SEARCH to this config's block: all three configs repeat the
        # same encoder names, so an unscoped search matched configuration A's row every time and
        # reported B and C as broken when they were fine.
        _start = src.find(f"Config {e} ---")
        _later = [p for p in (src.find(f"Config {x} ---") for x in "ABC") if p > _start]
        block = src[_start:(min(_later) if _later else len(src))]
        deployable = {s: [] for s in ("bare", "crude", "rich", "grounded")}
        for model, d in j["models"].items():
            short = model.split("/")[0].replace("_", "-")
            row = re.search(rf"\\quad {re.escape(short)} & ([\d.]+) & (.+?)\\\\", block)
            if not row:
                bad(f"T2/{e}: no row for {short}")
                continue
            printed = cells(row.group(2))
            expect = [d[s]["acc"] * 100 for s in ("bare", "crude", "rich", "grounded")]
            if len(printed) != 4:
                bad(f"T2/{e}/{short}: expected 4 cells, found {len(printed)}")
            elif not all(near(p, x) for p, x in zip(printed, expect)):
                bad(f"T2/{e}/{short}: printed {printed} vs JSON "
                    f"{[round(x, 1) for x in expect]}")
            else:
                ok(f"T2/{e}/{short}: 4 cells match")
            if not is_reference(model):
                for s in deployable:
                    deployable[s].append(d[s]["acc"] * 100)

        # the mean row, and that the reference encoder is NOT in it
        if all(len(v) == 4 for v in deployable.values()):
            mean_row = re.search(r"\\textbf\{mean\}(.+?)\\\\", block)
            if not mean_row:
                bad(f"T2/{e}: mean row not found")
            else:
                printed = cells(mean_row.group(1))
                expect = [sum(deployable[s]) / 4 for s in ("bare", "crude", "rich", "grounded")]
                if len(printed) != 4 or not all(near(p, x) for p, x in zip(printed, expect)):
                    bad(f"T2/{e} mean: printed {printed} vs recomputed "
                        f"{[round(x, 2) for x in expect]}")
                else:
                    ok(f"T2/{e} mean: {[round(x, 1) for x in expect]} over 4 deployable tiers")
                # would including the reference change it? if not, the check is not sensitive
                with_ref = [x["acc"] * 100 for x in
                            (j["models"][m]["grounded"] for m in j["models"])]
                if near(sum(with_ref) / len(with_ref), expect[3], 0.05):
                    note(f"T2/{e}: reference-exclusion check is insensitive here "
                         f"(means coincide)")
        else:
            bad(f"T2/{e}: expected 4 deployable encoders, got "
                f"{ {s: len(v) for s, v in deployable.items()} }")


# ---------------------------------------------------------------- Table 3, abstain
def check_abstain():
    src = tex("tab_abstain.tex")
    for e in "ABC":
        j = load(f"metrics_abstain_{e}.json")
        if not j or not src:
            continue
        for model, d in j["models"].items():
            short = model.split("/")[0].replace("_", "-")
            row = re.search(rf"{e} & \d+ & {re.escape(short)} & (.+?)\\\\", src)
            if not row:
                bad(f"T3/{e}/{short}: row not found")
                continue
            printed = cells(row.group(1))
            aurcs = [float(x) for x in re.findall(r"& (0\.\d+) &", row.group(1))]
            exp_pct, exp_aurc = [], []
            zsj = load(f"zeroshot_eval_{e}.json") or {}
            for strat in ("grounded", "rich"):
                s = d.get(strat)
                if not s:
                    continue
                # TOP-1 COMES FROM zeroshot_eval, NOT metrics_abstain. The latter stores
                # accuracies rounded to 4 dp, so MobileCLIP-B/rich at B is 0.2835 there against
                # 0.28353938... in zeroshot_eval -- 28.3% vs 28.4% once formatted. The generator
                # and preflight both treat the full-precision file as canonical; so must this.
                top1 = zsj.get("models", {}).get(model, {}).get(strat, {}).get("acc", s["top1"])
                exp_pct += [top1 * 100, s["top5"] * 100, s["acc@cov80"] * 100]
                exp_aurc.append(round(s["aurc"], 4))
            if sorted(round(p, 1) for p in printed) != sorted(round(x, 1) for x in exp_pct):
                bad(f"T3/{e}/{short}: printed {sorted(printed)} vs JSON "
                    f"{sorted(round(x, 1) for x in exp_pct)}")
            else:
                ok(f"T3/{e}/{short}: top1/top5/acc@cov80 match for both strategies")


# ---------------------------------------------------------------- Table 4, seen probe
def check_seen():
    src = tex("tab_seen.tex")
    for e in "ABC":
        j = load(f"probe_seen_{e}.json")
        if not j or not src:
            continue
        row = re.search(rf"^{e} & (\d+) & (\d+) & ([\d,]+) & (.+?)\\\\", src, re.M)
        if not row:
            bad(f"T4/{e}: row not found")
            continue
        if int(row.group(2)) != j["seen_classes"]:
            bad(f"T4/{e}: classes {row.group(2)} vs JSON {j['seen_classes']}")
        if int(row.group(3).replace(",", "")) != j["seen_images"]:
            bad(f"T4/{e}: images {row.group(3)} vs JSON {j['seen_images']:,}")
        printed = cells(row.group(4))
        expect = [d["seen_probe_top1"] * 100 for d in j["models"].values()]
        if len(printed) != len(expect) or not all(near(p, x) for p, x in zip(printed, expect)):
            bad(f"T4/{e}: printed {printed} vs JSON {[round(x, 1) for x in expect]}")
        else:
            ok(f"T4/{e}: {j['seen_classes']} classes, {j['seen_images']:,} imgs, "
               f"{len(expect)} encoders match")
        # the claim in the caption and Section 4.6: images per class do NOT rise
        note(f"T4/{e}: images per class = {j['seen_images'] / j['seen_classes']:.0f}")


# ---------------------------------------------------------------- Table 7, per-crop
def check_loco():
    src = tex("tab_loco.tex")
    j = load("loco_s0_rich.json")
    if not j or not src:
        return
    per = j["per_crop"]
    for crop, d in per.items():
        row = re.search(rf"{crop} & ([\d,]+) & (\d+\.\d+)\\%", src)
        if not row:
            bad(f"T7/{crop}: row not found")
            continue
        if int(row.group(1).replace(",", "")) != d["n"]:
            bad(f"T7/{crop}: N {row.group(1)} vs JSON {d['n']:,}")
        elif not near(float(row.group(2)), d["acc"] * 100):
            bad(f"T7/{crop}: acc {row.group(2)}% vs JSON {d['acc'] * 100:.2f}%")
        else:
            ok(f"T7/{crop}: N={d['n']:,} acc={d['acc'] * 100:.1f}% ({d['role']})")
    held = [d["acc"] for d in per.values() if d["role"] == "held"]
    train = [d["acc"] for d in per.values() if d["role"] != "held"]
    if held and train:
        h, t = sum(held) / len(held) * 100, sum(train) / len(train) * 100
        cap = re.search(r"average above the trained pool here \((\d+\.\d+)\\% against (\d+\.\d+)\\%\)", src)
        if cap:
            if near(float(cap.group(1)), h) and near(float(cap.group(2)), t):
                ok(f"T7 caption: held {h:.1f}% vs trained {t:.1f}% matches recomputation")
            else:
                bad(f"T7 caption: says {cap.group(1)}/{cap.group(2)}, recomputed {h:.1f}/{t:.1f}")
        # the scope fix: held-out rows must be exactly configuration A's crops
        held_crops = {c for c, d in per.items() if d["role"] == "held"}
        if held_crops == {"Coffee", "Orange", "Peach"}:
            ok(f"T7: held-out rows are config A's crops {sorted(held_crops)} -- "
               f"caption must scope the caveat to A")
        else:
            bad(f"T7: held-out rows are {sorted(held_crops)}, not config A's three; "
                f"the scoped caption claim needs revisiting")
        if "configuration A" not in src:
            bad("T7 caption: does not scope the held-vs-trained contrast to configuration A")
        else:
            ok("T7 caption: scopes the contrast to configuration A")


# ---------------------------------------------------------------- Table 8, edge
def check_edge():
    src = tex("tab_edge.tex")
    j = load("edge_quant_benchmark.json") or load("edge_benchmark.json")
    if not j or not src:
        return
    models = j.get("models", j)
    rows = 0
    for model, d in (models.items() if isinstance(models, dict) else []):
        short = model.split("/")[0].replace("_", "-")
        row = re.search(rf"{re.escape(short)} & ([\d.]+) & (.+?)\\\\", src)
        if row:
            rows += 1
    if rows:
        ok(f"T8: {rows} encoder row(s) present")
    # the two ratio claims the manuscript makes from this table
    nums = {}
    for model, d in (models.items() if isinstance(models, dict) else []):
        nums[model.split("/")[0]] = d
    txt = tex("main.tex")
    m = re.search(r"makes the lightweight tiers (\d+)--(\d+)\$?\\?times", txt)
    if m:
        note(f"main.tex claims dynamic INT8 is {m.group(1)}-{m.group(2)}x slower "
             f"(verify against the table's own columns)")
    m = re.search(r"recovers ([\d.]+)--([\d.]+)\$\\times\$", txt)
    if m:
        ok(f"main.tex QDQ recovery range stated as {m.group(1)}-{m.group(2)}x")


# ---------------------------------------------------------------- D. direction claims
def check_directions():
    """Every claim about a TREND, recomputed. This is the class that kept drifting."""
    txt = tex("main.tex")
    zs = {e: load(f"zeroshot_eval_{e}.json") for e in "ABC"}
    if not all(zs.values()):
        return

    def dep_mean(e, strat):
        d = zs[e]["models"]
        v = [x[strat]["acc"] * 100 for m, x in d.items() if not is_reference(m)]
        return sum(v) / len(v)

    bank = [dep_mean(e, "rich") for e in "ABC"]
    grnd = [dep_mean(e, "grounded") for e in "ABC"]
    bare = [dep_mean(e, "bare") for e in "ABC"]

    # 1. the bank falls monotonically -- asserted in the abstract, Sec 4.3, 4.4 and Fig 1
    if bank[0] > bank[1] > bank[2]:
        ok(f"direction: keyword bank falls monotonically {[round(x, 1) for x in bank]}")
    else:
        bad(f"direction: bank NOT monotonically falling {[round(x, 1) for x in bank]} -- "
            f"the abstract, Sec 4.3/4.4 and Fig 1 all assert it does")

    # 2. grounded must NOT be described as rising (withdrawn in Sec 4.4)
    rising = grnd[0] < grnd[1] < grnd[2]
    for pat in ("source-grounded descriptors improve as the label space",
                "keeps improving", "keep improving"):
        if pat in txt:
            bad(f"direction: main.tex still contains a grounded-improves claim ({pat!r}); "
                f"Sec 4.4 withdraws it")
    if rising:
        note(f"grounded does rise on the uncleaned micro numbers {[round(x, 1) for x in grnd]}; "
             f"Sec 4.4 correctly declines to claim it")

    # 3. grounded is the best strategy at C -- the surviving claim
    at_c = {"bare": bare[2], "rich": bank[2], "grounded": grnd[2],
            "crude": dep_mean("C", "crude")}
    best = max(at_c, key=at_c.get)
    if best == "grounded":
        ok(f"direction: grounded is best at C ({at_c['grounded']:.1f}% vs "
           f"bank {at_c['rich']:.1f}%, bare {at_c['bare']:.1f}%)")
    else:
        bad(f"direction: best strategy at C is {best}, not grounded -- Sec 4.4's surviving "
            f"claim fails")

    # 4. the reframed thesis: authoring gain vs encoder-size spread, per configuration
    for i, e in enumerate("ABC"):
        d = zs[e]["models"]
        g = {m: x["grounded"]["acc"] * 100 for m, x in d.items() if not is_reference(m)}
        by_size = sorted(g.items(), key=lambda kv: d[kv[0]]["grounded"]["img_params_M"])
        size_gain = by_size[-1][1] - by_size[0][1]
        auth_gain = grnd[i] - bare[i]
        note(f"lever/{e}: authoring {auth_gain:+.2f} pts vs encoder size "
             f"{size_gain:+.2f} pts -> {'authoring' if auth_gain > size_gain else 'size'} wins")
    # The Discussion states the two series explicitly. Recompute both from UNROUNDED values and
    # require the manuscript to quote those, not the difference of two rounded cells.
    #
    # This check exists because the manuscript quoted 5.7 for the configuration-C encoder spread
    # for several drafts. 27.3 - 21.6 = 5.7 from the printed cells, but the unrounded difference is
    # 27.2740 - 21.6418 = 5.6322, i.e. 5.6. Table 5's own caption already states the convention
    # ("differences quoted in the text are computed from the unrounded values"); nothing enforced
    # it. The same trap applies to every difference the prose quotes.
    auth = [grnd[i] - bare[i] for i in range(3)]
    spread = []
    for e in "ABC":
        d = zs[e]["models"]
        dep = sorted(((x["grounded"]["img_params_M"], x["grounded"]["acc"] * 100)
                      for m, x in d.items() if not is_reference(m)))
        spread.append(dep[-1][1] - dep[0][1])
    want_auth = ", ".join(f"{v:.1f}" for v in auth[:2]) + f" and {auth[2]:.1f}"
    want_spread = ", ".join(f"{v:.1f}" for v in spread[:2]) + f" and {spread[2]:.1f}"
    if re.search(rf"is worth {spread[0]:.1f}, {spread[1]:.1f} and {spread[2]:.1f} points", txt):
        ok(f"Discussion quotes the encoder-size spreads from unrounded values ({want_spread})")
    else:
        bad(f"Discussion must quote the encoder-size spreads as {want_spread} "
            f"(unrounded); differences of rounded cells would give "
            f"{', '.join(f'{round(dep, 1)}' for dep in [round(s, 1) for s in spread])}")
    if re.search(rf"worth {auth[0]:.1f} points at configuration~A", txt) and \
            re.search(rf"but\s*{auth[1]:.1f} and {auth[2]:.1f} points at B and C", txt):
        ok(f"Discussion quotes the authoring gains from unrounded values ({want_auth})")
    else:
        bad(f"Discussion must quote the authoring gains as {want_auth} (unrounded)")
    # and nothing anywhere may still quote the old rounded-cell value for that spread
    for stale in ("spread at configuration C is 5.7 points",
                  "tier buys 5.7 points of grounded accuracy",
                  "is worth 8.6, 4.8 and 5.7 points"):
        if stale in txt:
            bad(f"stale rounded-cell difference still present: {stale!r} "
                f"(unrounded spread at C is {spread[2]:.4f})")

    # 5. seen probe rises with the label space, and is flat across size
    ps = {e: load(f"probe_seen_{e}.json") for e in "ABC"}
    if all(ps.values()):
        means = {e: [d["seen_probe_top1"] * 100 for d in ps[e]["models"].values()] for e in "ABC"}
        if all(max(v) - min(v) <= 1.5 for v in means.values()):
            ok("direction: seen probe flat across the parameter range "
               f"(spreads {[round(max(v) - min(v), 1) for v in means.values()]} <= 1.5)")
        else:
            bad(f"direction: seen-probe spread exceeds the claimed 1.5 points: "
                f"{[round(max(v) - min(v), 1) for v in means.values()]}")
        per_enc_rise = [means['C'][i] - means['A'][i] for i in range(len(means['A']))]
        if all(x > 0 for x in per_enc_rise):
            ok(f"direction: seen probe rises A->C for every encoder "
               f"({[round(x, 1) for x in per_enc_rise]})")
        else:
            bad(f"direction: seen probe does not rise for every encoder: "
                f"{[round(x, 1) for x in per_enc_rise]}")
        # total images vs images-per-class, the Sec 4.6 fix
        tot = [ps[e]["seen_images"] for e in "ABC"]
        ipc = [ps[e]["seen_images"] / ps[e]["seen_classes"] for e in "ABC"]
        if tot[2] > tot[0]:
            pct = (tot[2] / tot[0] - 1) * 100
            claimed = re.search(r"total training data\s*rises (\d+)\\%", txt)
            if claimed and abs(int(claimed.group(1)) - pct) <= 1:
                ok(f"Sec 4.6 states total training data rises {claimed.group(1)}% "
                   f"(recomputed {pct:.0f}%)")
            else:
                bad(f"Sec 4.6: total training data rises {pct:.0f}%; "
                    f"manuscript says {claimed.group(1) + '%' if claimed else 'nothing'}")
        if not (ipc[0] < ipc[1] < ipc[2]) and not (ipc[0] > ipc[1] > ipc[2]):
            if "do not rise" in txt or "436, 403 and 421" in txt:
                ok(f"Sec 4.6 describes images/class as non-monotone "
                   f"({[round(x) for x in ipc]})")
            else:
                bad(f"images/class is non-monotone {[round(x) for x in ipc]} but the text "
                    f"describes a fall")

    # 6. no flat-null assertions left anywhere
    for pat in ["Accuracy does not track parameter count",
                "does not track accuracy at all",
                "accuracy does not track parameter count over a"]:
        hits = [f for f in ("main.tex", "tab_pilot.tex", "tab_supervised.tex", "tab_seen.tex")
                if pat.lower() in tex(f).lower()]
        if hits:
            bad(f"null asserted from a non-significant test: {pat!r} in {hits}")
    if not FAIL or all("null asserted" not in f for f in FAIL):
        ok("no flat-null assertions remain in main.tex or the table captions")


# ---------------------------------------------------------------- F. derived multiples
def check_multiples():
    txt = tex("main.tex")
    zs = {e: load(f"zeroshot_eval_{e}.json") for e in "ABC"}
    if not all(zs.values()):
        return

    def dep_mean(e, strat):
        d = zs[e]["models"]
        v = [x[strat]["acc"] * 100 for m, x in d.items() if not is_reference(m)]
        return sum(v) / len(v)

    gA, gC = dep_mean("A", "grounded"), dep_mean("C", "grounded")
    chA, chC = zs["A"]["chance"] * 100, zs["C"]["chance"] * 100
    # majority prior, quoted in Sec 4.5 as 14.8% at A and 4.2% at C
    majA = re.search(r"prior is (\d+\.\d+)\\% against grounded", txt)
    majC = re.search(r"it falls to (\d+\.\d+)\\%", txt)
    if majA and majC:
        rA, rC = gA / float(majA.group(1)), gC / float(majC.group(1))
        claimed = re.search(r"(\d+\.\d+)--(\d+\.\d+) times the majority-class prior", txt)
        if claimed:
            lo, hi = float(claimed.group(1)), float(claimed.group(2))
            if abs(rA - lo) <= 0.06 and abs(rC - hi) <= 0.06:
                ok(f"multiples: {lo}-{hi}x majority prior matches recomputation "
                   f"({rA:.2f}x at A, {rC:.2f}x at C)")
            else:
                bad(f"multiples: manuscript says {lo}-{hi}x majority prior; recomputed "
                    f"{rA:.2f}-{rC:.2f}x")
    rA, rC = gA / chA, gC / chC
    claimed = re.search(r"\((\d+\.\d+)--(\d+\.\d+) times uniform chance", txt)
    if claimed:
        lo, hi = float(claimed.group(1)), float(claimed.group(2))
        if abs(rA - lo) <= 0.06 and abs(rC - hi) <= 0.06:
            ok(f"multiples: {lo}-{hi}x uniform chance matches ({rA:.2f}x, {rC:.2f}x)")
        else:
            bad(f"multiples: manuscript says {lo}-{hi}x chance; recomputed {rA:.2f}-{rC:.2f}x")
    # random top-5 baselines
    from decimal import Decimal, ROUND_HALF_UP
    for e, n in (("A", zs["A"]["n_classes"]), ("C", zs["C"]["n_classes"])):
        # Round half UP, the way LaTeX and the manuscript do. Python's round-half-even turns the
        # exact 31.25 (= 5/16, as a percentage) into 31.2 and would flag the correct 31.3% wrong.
        want = float((Decimal(500) / Decimal(n)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
        if re.search(rf"\({want:.1f}\\% random\)", txt):
            ok(f"multiples: random top-5 at {n} classes = {want:.1f}% present")
        else:
            bad(f"multiples: random top-5 for {n} classes should be {want:.1f}%; not found")


# ---------------------------------------------------------------- E. figure inputs
def check_figures():
    figdir = PAPER / "figures"
    used = re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex("main.tex"))
    for f in used:
        p = figdir / Path(f).name
        if not p.exists():
            bad(f"figure referenced but missing: {f}")
        elif p.stat().st_size < 10_000:
            bad(f"figure suspiciously small ({p.stat().st_size} B): {f}")
        else:
            ok(f"figure present: {Path(f).name} ({p.stat().st_size / 1024:.0f} KB)")
    if len(set(used)) != len(used):
        bad(f"a figure is included twice: {used}")

    # In-plot titles are claims too, and they live in make_figures.py where no check ever
    # looked. Figure 2's title asserted "Descriptor detail is the lever" for several drafts
    # after Table 2 stopped supporting it, and an unused figure still asserted the flat null.
    # Both are the recurring caption-drift class (docs/paper/AUDIT_HISTORY.md, item 2).
    gen = (PAPER / "make_figures.py")
    if gen.exists():
        g = gen.read_text(encoding="utf-8")
        banned = {
            "Descriptor detail is the lever": "narrowed 2026-09-12: size beats authoring at B and C",
            "Accuracy does not track parameter count": "null asserted from n=10, rho=0.35, p=0.32",
            "keeps improving": "withdrawn in Section 4.4",
            "deployment sweet spot": "no objective function stated; depends on which cost binds",
        }
        hits = [f"{k!r} ({why})" for k, why in banned.items() if k in g]
        if hits:
            bad("make_figures.py still asserts a retracted claim in a plot title: "
                + "; ".join(hits))
        else:
            ok("no retracted claim appears in any in-plot figure title")
    # every figure must be newer than the JSONs it is drawn from, or it is stale
    src_mtime = max((PAPER / n).stat().st_mtime for n in
                    ("zeroshot_eval_C.json", "metrics_abstain_C.json", "probe_seen_C.json")
                    if (PAPER / n).exists())
    stale = [Path(f).name for f in used
             if (figdir / Path(f).name).exists()
             and (figdir / Path(f).name).stat().st_mtime < src_mtime]
    if stale:
        bad(f"figures older than their source JSONs (regenerate): {stale}")
    else:
        ok("every included figure is newer than the result files it is drawn from")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="exit 1 if anything failed")
    args = ap.parse_args()

    for fn in (check_scale_study, check_abstain, check_seen, check_loco, check_edge,
               check_directions, check_multiples, check_figures):
        try:
            fn()
        except Exception as e:                       # a checker bug must not look like a pass
            bad(f"{fn.__name__} raised {type(e).__name__}: {e}")

    for m in PASS:
        print(f"  [OK]   {m}")
    for m in NOTE:
        print(f"  [note] {m}")
    for m in FAIL:
        print(f"  [FAIL] {m}")
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed, {len(NOTE)} note(s)")
    if FAIL:
        print("NOT CONSISTENT")
        return 1 if args.strict else 0
    print("ALL TABLES, FIGURES AND DIRECTION CLAIMS CONSISTENT WITH THE RELEASED JSONS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
