"""Rebuild logs/README.md from the files actually present, so the index cannot drift."""
import io
import os
import re
from datetime import datetime, timezone

LOGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")

# What each log family records, and which paper artefact it backs.
DESC = [
    (re.compile(r"^exp1_output"), "EXP1 encoder bake-off",
     "`run_all_bakeoff.json`, `fig_bakeoff.png`"),
    (re.compile(r"^exp2_output"), "EXP2 seen-crop training",
     "`run_all_train_seen_lw11.json`"),
    (re.compile(r"^exp3_output"), "EXP3 fine-tune + WiSE-FT",
     "`run_all_exp3_lw11*.json`, `fig_wiseft.png`"),
    (re.compile(r"^eval_exp[ABC]_clean"), "label-corrected zero-shot at scale C",
     "`zeroshot_eval_C_clean.json`"),
    (re.compile(r"^eval_exp([ABC])$"), "descriptor ablation, scale {g1}",
     "`zeroshot_eval_{g1}.json`, `fig_descriptor_ablation.png`"),
    (re.compile(r"^metrics_exp([ABC])$"), "abstention and top-5, scale {g1}",
     "`metrics_abstain_{g1}.json`, `fig_riskcoverage.png`"),
    (re.compile(r"^probe_all_b$"), "seen-crop probe, heavyweight tier",
     "`probe_seen_*.json`"),
    (re.compile(r"^probe_all$"), "seen-crop probe, all tiers",
     "`probe_seen_A/B/C.json`, `fig_seen_scaling.png`"),
    (re.compile(r"^probe_s1$"), "seen-crop probe, MobileCLIP-S1",
     "`probe_seen_*.json`"),
    (re.compile(r"^supervised_output$"), "supervised CNN sweep, combined output",
     "`supervised_*.json`, `tab_supervised`"),
    (re.compile(r"^cnn_(.+)$"), "supervised CNN baseline: {g1}",
     "`supervised_{g1}.json`, `fig_cnn_training.png`"),
    (re.compile(r"^descriptors_fill_all"), "source-grounded descriptor generation",
     "`descriptors/*.json`"),
    (re.compile(r"^descriptor_audit"), "descriptor spot-check before use",
     "audit trail only"),
    (re.compile(r"^edge_quant_bench"), "INT8 quantisation sweep",
     "`edge_quant_benchmark.json`, §5.7"),
    (re.compile(r"^edge_bench"), "on-device latency and size",
     "`edge_benchmark.json`, `fig_edge_pareto.png`"),
    (re.compile(r"^loco"), "leave-one-crop-out with bootstrap CIs",
     "`loco_s0_rich.json`, `tab_loco`"),
    (re.compile(r"^morning_run_(\d{4}-\d{2}-\d{2})$"),
     "full session {g1}: paired Section 5.3 comparison, both control arms, "
     "probe, LOCO, abstention; CNN and WiSE-FT failures diagnosed here",
     "`control_arm_statistics.json`, `zeroshot_eval_C_paired_ungseeds.json`"),
]


def describe(stem):
    for pat, what, feeds in DESC:
        m = pat.match(stem)
        if m:
            g1 = m.group(1) if m.groups() else ""
            return what.format(g1=g1), feeds.format(g1=g1)
    return "—", "—"


files = sorted(f for f in os.listdir(LOGS) if f.endswith(".log"))
total = sum(os.path.getsize(os.path.join(LOGS, f)) for f in files)

rows = []
for f in files:
    what, feeds = describe(f[:-4])
    kb = os.path.getsize(os.path.join(LOGS, f)) / 1024
    rows.append(f"| `{f}` | {what} | {feeds} | {kb:.0f} KB |")

body = f"""# Run logs — the reproducibility evidence trail

Raw stdout from every experiment, committed so that any number in `docs/paper/` can be traced
back to the run that produced it. {len(files)} files, {total / 1024:.0f} KB total.

`*.log` is gitignored everywhere **except this folder** (`!logs/*.log` in `.gitignore`).

This index is generated — run `python scripts/index_logs.py` after adding a log, rather than
editing the table by hand.

| Log | Records | Feeds | Size |
|---|---|---|---|
{chr(10).join(rows)}

## Reading a log

Each begins with the resolved configuration (dataset revision, class counts, chance level) and
ends with the path of the JSON it wrote. When a manuscript number is questioned, the sequence is:
find the claim in `main.tex`, find the JSON the generator read, find the log that wrote it.

Logs from superseded builds are kept rather than deleted: a result that changed between builds is
evidence about the pipeline, and `docs/paper/_oldbuild_backup/` holds the corresponding JSONs.
"""

io.open(os.path.join(LOGS, "README.md"), "w", encoding="utf-8", newline="").write(body)
print(f"logs/README.md rebuilt: {len(files)} logs indexed, {total/1024:.0f} KB")
undoc = [f for f in files if describe(f[:-4])[0] == "—"]
if undoc:
    print(f"  still undescribed: {undoc}")
