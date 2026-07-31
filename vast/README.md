# Vast.ai run plan — finish the paper on a ~$4.60 budget

Everything GPU-bound that's still blocking the paper, ordered so the **paper-blocking work finishes
first**. If the credit runs out after stage 3 you still have everything needed to write the draft.

---

## 0. What to rent

| Pick | GPU | ~$/hr | Why |
|---|---|---|---|
| ✅ **Best value** | **RTX 3090 / A5000 (24 GB)** | **$0.20–0.25** | ~18–22 h of runtime on $4.60. Plenty for this plan. |
| Faster, still fine | RTX 4090 (24 GB) | $0.35–0.45 | ~10–13 h. ~1.5–2× faster on the CNN stage. |
| ❌ Avoid | A100 / H100 | $1.00–2.50 | You'd get 2–4 h. Wasted on 11–86 M models. |

**Filters when renting:** template **PyTorch (cuda 12.x)** · **≥ 100 GB disk** (the SAGE shards are
~10 GB each) · **≥ 500 Mbps down** (bandwidth is the real cost driver here, not FLOPs) ·
reliability ≥ 99 %.

> The whole plan is ~**6–7 h ≈ $1.50–2.00 on a 3090**. The remaining credit is your safety margin —
> don't rent a bigger card just because you can.

---

## 1. Connect and bootstrap (~5 min)

The repo is **private**, so cloning needs a token. Create a fine-grained PAT with *read-only Contents*
access at <https://github.com/settings/tokens>, then on the box:

```bash
cd /workspace
git clone https://<YOUR_TOKEN>@github.com/Abhiram970/plant-disease-edge.git
cd plant-disease-edge

bash setup_vast.sh          # finds the pre-installed torch, adds timm + open_clip (never re-downloads torch)
PY=$(cat /tmp/PDEPY)        # the interpreter every later command should use
```

`setup_vast.sh` prints the GPU name and confirms `open_clip` imports. **If it fails, stop** — don't
burn credit on a broken env.

> Rotate/delete that token when you're done. Never commit it.

---

## 2. Run the plan

```bash
export PDE_DATA_ROOT=/workspace/working
export PDE_DATASET_DIR=/workspace/working/exp_data

bash vast/run_plan.sh                 # all stages, in priority order
# or, to control spend precisely:
bash vast/run_plan.sh 1 2             # just data + the paper blocker
bash vast/run_plan.sh 3               # then the scale study, etc.
```

Every stage is **resumable** and skips work whose result JSON already exists, so re-running after a
crash (or after topping up credit) costs nothing extra.

Run it under `tmux` so a dropped SSH session doesn't kill the job:
```bash
tmux new -s run
# ... start the plan ...
# detach: Ctrl-B then D      reattach: tmux attach -t run
```

---

## 3. The stages, in value order

| # | Stage | Time | ~Cost | Priority | Why it matters |
|---|---|---|---|---|---|
| 1 | SAGE data + manifest | 30–45 m | $0.15 | **mandatory** | Everything else needs it. Incremental — keeps `.shards_done.json`, only pulls missing shards. |
| 2 | **EXP3 WiSE-FT full sweep** | 1–1.5 h | $0.35 | **P0 — blocks paper** | The α-table in the paper is from the old small config; the full-data run died at epoch 1/5. |
| 3 | **Zero-shot scale study A/B/C** | 45–60 m | $0.20 | **P0 — fixes the numbers** | Produces all three configs from ONE consistent sweep → kills the 27 %-vs-17 % inconsistency. |
| 4 | Seen-head probe A/B/C | 45–60 m | $0.25 | P1 | The "seen accuracy vs #classes" half of the scale study. |
| 5 | Supervised CNN baselines (6 archs) | 2–2.5 h | $0.55 | P1 | Only 1 of the family is done. This is the "CNNs are strong on seen, structurally 0 on unseen" table. |
| 6 | LOCO + abstain metrics | 30–40 m | $0.15 | P1 | Rebuttal-proofing (anti-cherry-pick + risk-coverage). |
| | **total** | **~6–7 h** | **~$1.65** | | leaves ~$3 margin |

**If credit gets tight:** stages **1 → 2 → 3** are the ones that unblock writing. 4–6 are strengthening.

---

## 4. Get the results off the box (do this BEFORE destroying it)

The plan writes everything to `vast_results/` and tars it:

```bash
ls -la vast_results/            # sanity-check the JSONs are there
```

From **your laptop**:
```bash
scp -P <PORT> root@<HOST>:/workspace/plant-disease-edge/vast_results.tar.gz .
```
(Vast shows the exact `ssh -p PORT root@HOST` string on the instance card.)

Then locally:
```bash
tar -xzf vast_results.tar.gz
cp vast_results/*.json docs/paper/
"$PY" docs/paper/make_figures.py
git add docs/paper/ logs/ && git commit -m "Vast.ai run: EXP3 sweep, A/B/C scale study, CNN baselines"
```

**Destroy the instance** as soon as the tarball is on your laptop — storage bills even when stopped.

---

## 5. Do NOT spend GPU money on these

These are still open but need **no GPU** — do them locally for free while the box runs:

| Item | Where | Cost |
|---|---|---|
| Fill the 66 stub descriptors (esp. **Coffee: only 1 of 5 filled**) | `scripts/build_descriptors.py --fill` | ~$1–2 Lava API |
| Replace the 5 dead source URLs | `scripts/apply_verified_citations.py` | free |
| Update paper §5.9 with the corrected edge numbers | edit `docs/paper/paper.md` | free |
| Elsevier LaTeX conversion, authors, funding | writing | free |
| **Raspberry Pi latency row** | needs a **real Pi**, not a rented GPU | free |

> The Pi row is the one gap a rented GPU can never fill — and it's the most likely reviewer attack on
> an "edge" paper. Plan a Pi (or an Android phone via TFLite) run separately.

---

## 6. Troubleshooting

- **`open_clip` missing** → re-run `bash setup_vast.sh`; it installs it `--no-deps` so torch is untouched.
- **CUDA OOM in stage 2/5** → lower the batch: `--batch 32` (stage 2) / `--batch 64` (stage 5).
- **Data fetch looks stuck** → it's downloading a ~10 GB shard; watch `logs/vast/01_sage_data.log`.
  It resumes, so a Ctrl-C and restart is safe.
- **A stage fails** → the plan *continues to the next stage on purpose* so one failure can't waste the
  box. Check `logs/vast/<stage>.log`, fix, and re-run just that stage number.
- **Slow download** → you rented a low-bandwidth host. Bandwidth matters more than FLOPs for stage 1;
  destroy and re-rent if it's crawling.
