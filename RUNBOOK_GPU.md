# GPU runbook — what is actually left

**Status as of 8 Aug 2026: every experiment the paper's claims depend on is DONE.**
Nothing in this file is required to draft or submit. Everything here is a *strengthener*.

Verified complete (result JSONs live in `docs/paper/`, tables regenerate from them via
`python docs/paper/make_tables.py --write`):

| Experiment | Artefact | Status |
|---|---|---|
| Zero-shot scale study A/B/C (4 encoders × 4 descriptor strategies) | `zeroshot_eval_{A,B,C}.json` | done |
| Top-5 + risk–coverage / AURC, A/B/C (5 encoders × 2 strategies) | `metrics_abstain_{A,B,C}.json` | done |
| Encoder bake-off incl. BioCLIP2 + SCOLD | `run_all_bakeoff.json` | done |
| Seen-head linear probe, 166 classes, 4 encoders | `probe_seen_C.json` | done |
| WiSE-FT α sweep, full data (166 seen cls, 55,981 imgs) | `run_all_exp3_lw11_full.json` | done |
| Supervised CNN baselines ×3 (ResNet-50, MNv3-S, MNv4-conv-S) | `supervised_*.json` | done |
| Leave-one-crop-out + bootstrap CIs | `loco_s0_rich.json` | done |
| Edge latency/size ×5 variants + INT8 graph diagnosis | `edge_quant_benchmark.json` | done |

---

## Priority order if you have GPU time

### 1. `probe_seen` for scales A and B  (~25 min, highest value)
The only *asymmetry* in the paper: zero-shot is measured at three scales, the seen head at one.
Filling this gives a seen-side scaling curve to sit beside Fig. `fig_descriptor_scaling.png`, and
answers a reviewer question that is very likely to be asked ("does the probe also flatten?").

```bash
python scripts/probe_seen.py --exp A
python scripts/probe_seen.py --exp B
```

(Defaults match the run that produced `probe_seen_C.json` — all four deployable encoders. Add
`--models ...` / `--epochs N` only if you want to deviate.)

Then copy `probe_seen_{A,B}.json` into `docs/paper/` and re-run `make_tables.py --write`. Note that
`table_seen()` in `make_tables.py` currently reads **only** `probe_seen_C.json`; extend it to loop
over A/B/C once those files exist.

### 2. Multi-seed for the headline table  (~2–3 h)
Currently a single seed with no variance estimate anywhere except the LOCO bootstrap. Three seeds on
the scale-C zero-shot eval turns "grounded beats rich by +4.3 pp" from a point estimate into a claim
with an error bar. Given that this is now the paper's headline finding, it is the difference between
a reviewer accepting it and asking for it.

```bash
python scripts/evaluate.py --exp C --heavy          # writes zeroshot_eval_C.json
python scripts/metrics.py  --exp C --reference      # writes metrics_abstain_C.json
```

Neither script currently exposes a `--seed` flag, so this needs a small edit first: thread a seed
through the eval subset sampling, run 0/1/2, and report mean ± sd on the `rich` vs `grounded` gap.

### 3. Faithful SCOLD loader  (~1 h, optional)
Our wrapper puts SCOLD *below chance*, which is a strong negative claim resting on a best-effort load.
Either get the authors' inference pipeline working or keep the current footnote. Do **not** strengthen
the claim without this.

### 4. Raspberry Pi / ARM latency row  (needs hardware, not GPU)
`python kaggle/benchmark_quantization.py` on-device. ARM NEON has well-optimised INT8 depthwise
kernels and **may reverse the §5.10 S-tier result** — which would make that section stronger, not
weaker, because the deployment rule becomes runtime-specific rather than universal.

---

## Environment

Local interpreter that has the full stack (torch 2.5.1+cu121, open_clip 3.3.0, onnx 1.22, ORT 1.26):

```
C:\Users\PV Abhiram\.pyenv\pyenv-win\versions\3.11.9\python.exe
```

The bare `python` on PATH is an unrelated venv with nothing installed — using it is the single most
common way these runs fail.

For Kaggle, follow `kaggle/RUNBOOK_KAGGLE.md`: build the data subset **once**, save it as the Kaggle
Dataset `pde-sage-data`, then attach it read-only. `/kaggle/working` is ~20 GB and a single SAGE shard
is ~10 GB, so re-downloading per session will run you out of disk.

## The one rule

Never type a number into `paper.md`. Every table comes from `docs/paper/make_tables.py`, and
`make_figures.py` now reads the same JSONs. Three separate number-drift incidents came from
hand-copied values; both scripts are the fix. After any new result:

```bash
python docs/paper/make_tables.py --write
python docs/paper/make_figures.py
```
