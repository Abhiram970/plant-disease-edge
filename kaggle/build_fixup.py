"""Generate RUN_FIXUP_cnns_wiseft.py -- the two stages the 2026-09-06 morning run lost.

That run finished everything else cleanly (8 ungrounded + 8 grounded_matched seeds, zero-shot
A/B/C, the paired Section 5.3 comparison, control arms, coverage, probe, LOCO, abstention) and
those results are carried forward from the attached dataset, not recomputed. Two stages did
not survive:

  CNNs      0 of 14. Ten depthwise architectures hit a cuDNN "unable to find an engine"
            failure under emulated bf16 on the T4; the other four each completed an epoch and
            were then killed by a 0.75 h per-architecture cap.
  WiSE-FT   produced a sweep whose alpha=0.5 point fell below both endpoints, because the
            randomly initialised head kicked the encoder out of the pretrained basin before
            it had learned anything.

Both causes are fixed at the source (scripts/supervised_baseline.py, scripts/wiseft.py,
and the probe in build_parts.py). This runner re-runs only those two stages.

Like the other generators, this one composes its body from build_parts.py rather than
restating it, so a fix there lands here automatically.
"""
import io
import re
from pathlib import Path

Q = chr(39) * 3          # a triple quote, written this way so it can appear in source
HERE = Path(__file__).resolve().parent
src = io.open(HERE / "build_parts.py", encoding="utf-8").read()


def _block(name):
    """Return the source of one PARTn as written in build_parts.py.

    Regex is the wrong tool here. A PARTn is not one literal -- it is a concatenation
    (`<q>...<q> + COMMON_SETTINGS + <q>...<q>`) whose body also contains lines that look like
    top-level assignments (`CNN_EPOCHS = 4`), so both "match one quoted run" and "stop at the
    next assignment" truncate it. Slice on the `PARTn = ` lines instead, which are the real
    boundaries, and evaluate the result so the `+ NAME +` joins resolve to their values.
    """
    lines = src.splitlines(keepends=True)
    starts = {}
    for i, ln in enumerate(lines):
        for n in ("PART1", "PART2", "PART3"):
            if ln.startswith(f"{n} = "):
                starts[n] = i
        if ln.startswith("for name, body in"):
            starts["_END"] = i
    if name not in starts:
        raise SystemExit(f"[build_fixup] could not find {name} in build_parts.py")
    order = sorted(starts.items(), key=lambda kv: kv[1])
    idx = [i for i, (n, _) in enumerate(order) if n == name][0]
    lo, hi = order[idx][1], order[idx + 1][1]
    expr = "".join(lines[lo:hi]).split("=", 1)[1].strip()
    ns = dict(_PREAMBLE)
    return eval(expr, ns)


# The preamble defines COMMON_SETTINGS and imports BOOTSTRAP from _pde_common.py. Executing
# it once gives both the values the PARTn expressions need and the values this file needs.
_PREAMBLE = {"__file__": str(HERE / "build_parts.py")}
exec(src[:src.index("PART1 = ")], _PREAMBLE)

P2, P3 = _block("PART2"), _block("PART3")

# --- the WiSE-FT stage, lifted verbatim from PART 2 -------------------------------------
_w0 = P2.index("# ================================================================ 3  WiSE-FT")
_w1 = P2.index("# ================================================================ bundle")
WISE = P2[_w0:_w1]

# This runner exists BECAUSE wiseft.json is present and wrong, so the "already have it" guard
# has to go. Nothing else in the stage changes.
WISE = WISE.replace(
    '''elif (RESULTS / "wiseft.json").exists():
    print("[skip] wiseft", flush=True)
''', "")

# Dropping the guard is not enough on its own. The bootstrap copies every *.json forward from
# the attached dataset, so the superseded wiseft.json lands in results/ before this stage runs.
# If the re-run then failed for any reason, make_tex_tables.py would read that file and quietly
# reprint the broken sweep as though it were fresh. Move it aside first: the table generator
# cannot find it, and the old numbers are still on disk under a name that says what they are.
# Dropping the guard is not enough on its own. The bootstrap copies every *.json forward from
# the attached dataset, so the superseded wiseft.json lands in results/ before this stage runs.
# If the re-run then failed, make_tex_tables.py would read that file and quietly reprint the
# broken sweep as though it were fresh.
#
# The move has to happen INSIDE the branch that is actually about to re-run WiSE-FT, not at
# module level: quarantining first and skipping afterwards (RUN_WISEFT off, wiseft.py absent,
# or the budget gate declining) would leave no wiseft.json at all, and tab_wiseft would fall
# back to the legacy mixed-protocol file without the log ever saying so.
WISE = WISE.replace(
    '''    banner("WiSE-FT alpha sweep (both sides under one protocol)")''',
    '''    banner("WiSE-FT alpha sweep (both sides under one protocol)")
    _stale = RESULTS / "wiseft.json"
    if _stale.exists():
        _keep = RESULTS / "wiseft_SUPERSEDED_2026-09-06.json"
        if not _keep.exists():
            _stale.replace(_keep)
            print(f"[wiseft] quarantined the superseded sweep -> {_keep.name}", flush=True)
            print("[wiseft] (alpha=0.5 fell below both endpoints; head warmup is the fix)",
                  flush=True)
        else:
            _stale.unlink()
            print("[wiseft] discarded a carried-forward pre-fix sweep "
                  f"({_keep.name} already holds the record)", flush=True)''')

# A resume must not re-run a sweep that is already correct. The guard cannot simply test for
# wiseft.json -- that is the file this runner exists to replace -- so it tests the protocol
# version the script stamps into its own output: 2 means the head is warm-started from the
# frozen-feature fit, which is the fix. Anything older, or unstamped, is re-run.
WISE = WISE.replace(
    '''elif ok_to_start("wiseft", [], 1.0):''',
    '''elif _wiseft_is_current():
    print("[skip] wiseft (results/wiseft.json already uses protocol_version 2)", flush=True)
elif ok_to_start("wiseft", [], 1.0):''')

# Defined before the stage so the guard above can call it.
WISE = '''def _wiseft_is_current():
    """True when results/wiseft.json came from the repaired protocol (see scripts/wiseft.py)."""
    _p = RESULTS / "wiseft.json"
    if not _p.exists():
        return False
    try:
        return int(json.loads(_p.read_text(encoding="utf-8")).get("protocol_version", 0)) >= 2
    except Exception:
        return False

''' + WISE

# --- the CNN stage, lifted verbatim from PART 3 -----------------------------------------
_c0 = P3.index("# A single forward+backward on random data")
_c1 = P3.index("#__P3_TAIL__")
CNN = P3[_c0:_c1]

# --- settings, taken from the two parts so they cannot diverge ---------------------------
def _settings(block, keys):
    out = []
    for line in block.splitlines():
        if any(re.match(rf"^{k}\s*=", line) for k in keys) or line.startswith("ARCHS"):
            out.append(line)
        elif out and (line.startswith(" ") or line.startswith('"')) and "ARCHS" in out[-1][:6]:
            out.append(line)
        elif out and out[-1].rstrip().endswith((",", "[")) and line.strip().startswith('"'):
            out.append(line)
    return "\n".join(out)


SET2 = _settings(P2, ["WISE_EPOCHS", "WISE_LR", "WISE_ALPHAS"])
SET3 = _settings(P3, ["CNN_EPOCHS", "CNN_BATCH", "CNN_WORKERS", "CNN_AMP", "CNN_MAX_H"])

COMMON = _PREAMBLE["COMMON_SETTINGS"]
BOOT = _PREAMBLE["BOOTSTRAP"]

HEAD = '''"""
=====================================================================================
 PDE FIX-UP RUN  --  THE 14 CNN BASELINES AND THE WiSE-FT SWEEP
=====================================================================================
Everything else is already done and is carried forward from the attached dataset, not
recomputed. This run exists to repair the two stages the 2026-09-06 morning run lost.

SETUP
  1. Add Data -> `pde-sage-data`
  2. Add Data -> the output of the morning run (results and descriptors carry forward)
  3. GPU T4 x2, Internet ON. No API key needed: no descriptors are generated here.

WHAT FAILED, AND WHAT CHANGED

  CNNs -- 0 of 14 completed.
    Ten architectures (MobileNetV3/V4, EfficientNet, FastViT, ConvNeXt-V2, EfficientNetV2)
    died in seconds with "GET was unable to find an engine to execute this computation"
    inside their depthwise convolutions. torch.cuda.is_bf16_supported() returns True on the
    T4, but Turing has no native bf16, and cuDNN has no depthwise engine for the emulated
    path. Precision is now chosen from the compute capability -- fp16 on the T4, bf16 only
    on Ampere and later -- with an automatic step-down ladder if a kernel is still missing.
    The batch probe walks the same ladder, so it can no longer report a failure and then
    hand back the batch anyway, which is what let all ten walk into an identical error.

    The other four (densenet121, regnety_040, resnet50, resnet101) trained correctly and
    were killed by the 0.75 h per-architecture cap partway through epoch 2 -- densenet121
    had already reached 81.6% and resnet50 79.8% on epoch 1. The cap is now 1.6 h, and a
    checkpoint is no longer deleted unless the architecture actually produced its JSON, so
    an overrun resumes instead of starting over.

  WiSE-FT -- completed, but the sweep was not usable.
    seen went 58.7 -> 45.1 -> 63.8 while unseen fell 21.6 -> 7.8 -> 1.8. A midpoint below
    both endpoints means the frozen and fine-tuned weights are not linearly connected, so
    interpolating between them is meaningless. The head was randomly initialised and trained
    jointly with the unfrozen encoder, so its early gradients pushed the encoder out of the
    pretrained basin. The head is now fitted on frozen features first and used to warm-start
    fine-tuning (that fit doubles as the alpha=0 reference, so it costs no extra pass), the
    encoder is excluded from weight decay, and the sweep runs 5 alphas instead of 3.

IF IT STOPS EARLY: re-run the same cell. Finished architectures are skipped and partially
trained ones resume from their checkpoint.
=====================================================================================
"""

# ---------------------------------------------------------------- settings
BUDGET_H    = 11.0
''' + SET2 + '''
''' + SET3 + '''
''' + COMMON + '''
''' + BOOT + '''
'''

TAIL = '''
# ================================================================ regenerate + bundle
# The generators read the JSONs sitting NEXT TO THEM in docs/paper, not results/, so this
# run's outputs have to be staged across first -- otherwise the tables regenerate from the
# previous build's numbers, which is exactly the build mixture the audit flagged. `CODE` does
# not exist in this bootstrap; the clone is `REPO`.
banner("regenerating tables and figures from the repaired results")
_docs = REPO / "docs" / "paper"
for _g in ("make_tex_tables.py", "make_figures.py"):
    if (_docs / _g).exists():
        for _f in RESULTS.glob("*.json"):
            shutil.copy2(_f, _docs / _f.name)
        _r = subprocess.run([sys.executable, "-u", str(_docs / _g)], text=True, cwd=str(_docs),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(_r.stdout or f"[{_g}] no output", flush=True)

bundle("fixup", {"cnn_epochs": CNN_EPOCHS, "cnn_max_h": CNN_MAX_H,
                 "wise_epochs": WISE_EPOCHS, "wise_lr": WISE_LR,
                 "wise_alphas": WISE_ALPHAS, "cnn_completed": done,
                 "cnn_not_run": skipped, "cnn_unsupported": unsupported})
banner("FIX-UP RUN DONE" if (len(done) == len(ARCHS) and not unsupported)
       else "FIX-UP RUN INCOMPLETE -- re-run to finish")
'''

# WiSE-FT first: it is the shorter stage and the one the manuscript is waiting on, so a
# truncated session still returns it.
BODY = HEAD + WISE + "\n" + CNN + TAIL

out = HERE / "RUN_FIXUP_cnns_wiseft.py"
out.write_text(BODY, encoding="utf-8")
compile(BODY, str(out), "exec")
print(f"wrote {out}  ({len(BODY.splitlines())} lines)")
