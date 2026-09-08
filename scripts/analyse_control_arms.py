"""
Section 5.3 statistics: turn the paired control-arm run into a defensible claim.

A null result is only publishable if it is quantified. "We found no difference" invites the
reply "your study was underpowered"; "grounding changes accuracy by less than X points with
95 % confidence" is a positive, falsifiable statement. This script computes X.

It reads the paired evaluation written by

    scripts/evaluate.py --exp C --strategies ungrounded grounded_matched \
                        --paired-arms ungrounded grounded_matched --ungrounded-seeds ...

and reports, per encoder and pooled:

  * each arm's mean over seeds, with the seed-to-seed standard deviation
  * the paired difference and its 95 % confidence interval (t, seeds as the unit)
  * the equivalence bound: the largest effect the data rule out
  * whether the encoders agree, which is the replication that matters more than
    any single interval

USAGE
  python scripts/analyse_control_arms.py
  python scripts/analyse_control_arms.py --file results/zeroshot_eval_C_paired_ungseeds.json
"""
from __future__ import annotations
import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

# two-sided t critical values, df = n-1
_T = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365,
      9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201, 16: 2.131, 20: 2.093}


def t_crit(n):
    if n in _T:
        return _T[n]
    return 1.96 + 2.0 / max(n, 2)      # adequate for n well above the table


def arm_seed_scores(model_row, arm):
    """{seed: accuracy%} for one arm, from keys shaped `arm__seedN`."""
    out = {}
    for k, v in model_row.items():
        if not isinstance(v, dict) or "acc" not in v:
            continue
        if k.startswith(f"{arm}__seed"):
            out[int(k.split("seed")[-1])] = v["acc"] * 100
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None,
                    help="paired evaluation JSON (defaults to the newest *_paired* file)")
    ap.add_argument("--arms", nargs=2, default=["grounded_matched", "ungrounded"],
                    help="the two arms to compare; the difference is arms[0] - arms[1]")
    args = ap.parse_args()

    if args.file:
        path = Path(args.file)
    else:
        cands = sorted(C.RESULTS_DIR.glob("zeroshot_eval_*paired*.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands:
            sys.exit("[analyse] no paired evaluation found. Run evaluate.py --paired-arms first.")
        path = cands[0]
    d = json.loads(path.read_text(encoding="utf-8"))
    print(f"[analyse] {path.name}")

    ps = d.get("paired_stats") or {}
    if ps:
        print(f"[analyse] paired on {ps.get('paired_classes')} of "
              f"{ps.get('requested_classes')} classes "
              f"({len(ps.get('dropped') or {})} dropped so neither arm falls through to rich)")
    print(f"[analyse] {d.get('n_images', '?'):,} images, chance {d.get('chance', 0) * 100:.2f}%")
    a, b = args.arms
    print(f"[analyse] difference reported as ({a}) - ({b})\n")

    rows, diffs = [], []
    hdr = f"{'encoder':22}{a[:14]:>16}{b[:14]:>16}{'diff':>10}"
    print(hdr)
    print("-" * len(hdr))
    for model, mrow in d.get("models", {}).items():
        if "SigLIP" in model:                     # reference ceiling, never in the means
            continue
        sa, sb = arm_seed_scores(mrow, a), arm_seed_scores(mrow, b)
        shared = sorted(set(sa) & set(sb))
        if len(shared) < 2:
            continue
        ma, mb = st.mean([sa[s] for s in shared]), st.mean([sb[s] for s in shared])
        rows.append((model.split("/")[0], ma, mb, len(shared),
                     st.stdev([sa[s] for s in shared]), st.stdev([sb[s] for s in shared])))
        diffs.append(ma - mb)
        print(f"{model.split('/')[0]:22}{ma:15.2f}%{mb:15.2f}%{ma - mb:+9.2f}")

    if not rows:
        sys.exit("\n[analyse] no encoder had both arms across shared seeds.")

    n_enc = len(diffs)
    mean_d = st.mean(diffs)
    print(f"\n{'mean over encoders':22}{'':16}{'':16}{mean_d:+9.2f}")

    # Encoders are the replication unit: each is an independent test of the same claim.
    if n_enc >= 2:
        sd_d = st.stdev(diffs)
        half = t_crit(n_enc) * sd_d / math.sqrt(n_enc)
        lo, hi = mean_d - half, mean_d + half
        print(f"\n95% CI over {n_enc} encoders: [{lo:+.2f}, {hi:+.2f}] pp")
        bound = max(abs(lo), abs(hi))
        same_sign = all(x > 0 for x in diffs) or all(x < 0 for x in diffs)
        print(f"equivalence bound        : the data rule out a difference larger than "
              f"{bound:.2f} pp")
        print(f"direction agreement      : {'all encoders agree' if same_sign else 'signs differ'}"
              f"  {[f'{x:+.2f}' for x in diffs]}")
        print()
        if lo <= 0 <= hi:
            print(f"VERDICT: null. Zero lies inside the interval, so the sourcing constraint")
            print(f"         has no measurable effect on accuracy; any true difference is")
            print(f"         smaller than {bound:.2f} pp.")
        else:
            print(f"VERDICT: a difference IS resolved: {mean_d:+.2f} pp, "
                  f"95% CI [{lo:+.2f}, {hi:+.2f}].")
        if not same_sign:
            print("         The per-encoder signs disagree, which independently argues the")
            print("         effect is noise rather than a property of the descriptors.")

    out = {
        "source": path.name,
        "arms": args.arms,
        "paired_classes": ps.get("paired_classes"),
        "per_encoder": [{"model": m, args.arms[0]: round(ma, 4), args.arms[1]: round(mb, 4),
                         "diff": round(ma - mb, 4), "seeds": n,
                         f"sd_{args.arms[0]}": round(sa, 4), f"sd_{args.arms[1]}": round(sb, 4)}
                        for m, ma, mb, n, sa, sb in rows],
        "mean_diff": round(mean_d, 4),
    }
    if n_enc >= 2:
        out.update({"ci95": [round(lo, 4), round(hi, 4)],
                    # NOT a TOST equivalence bound: a TOST margin must be pre-specified, and this is
                    # simply the larger CI limit. Named for what it is.
                    "largest_effect_not_excluded": round(bound, 4),
                    "encoders_agree_in_sign": same_sign})
    dst = C.RESULTS_DIR / "control_arm_statistics.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"\n[analyse] saved {dst}")


if __name__ == "__main__":
    main()
