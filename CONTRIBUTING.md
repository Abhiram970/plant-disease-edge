# CONTRIBUTING — How we work on Plant-Disease-Edge

Read `PROJECT_GUIDE.md` first (the plan). This file is **how the team operates**: roles, Kaggle
coordination, environment, and git workflow. Goal: anyone can clone and start a phase without asking.

---

## 1. Roles (fill in names)

| Role | Owns | Phases | Member |
|---|---|---|---|
| **LEAD** | direction, paper, final calls | all / E | _Abhiram_ |
| **DATA** | SAGE subset, descriptors, splits | A | _TBD_ |
| **MODEL** | embedding cache, distillation, experiments | B, C | _TBD_ |
| **EDGE** | INT8, ONNX/TFLite, phone+Pi benchmark | D | _TBD_ |
| **WRITE** | figures, tables, paper draft, citations | E | _TBD_ |

One person can hold multiple roles. The role tags (**DATA/MODEL/EDGE/WRITE/LEAD**) appear on every
task in the guide so you know what's yours.

---

## 2. Kaggle accounts — coordination (IMPORTANT)

We have **3 Kaggle accounts** (~30 GPU-hr/week each ≈ 90 total). Don't waste them.

| Account | Holder | Primary use |
|---|---|---|
| Kaggle-1 | _TBD_ | data build (Phase A) + publishing the shared datasets |
| Kaggle-2 | _TBD_ | embedding cache (Phase B) — heaviest GPU job |
| Kaggle-3 | _TBD_ | student training / ablations (Phase C) |

**Rules:**
- **Never run the same job on two accounts** — coordinate in the team chat / a pinned issue before a
  long run. State: which notebook, which account, expected hours.
- Always launch long jobs with **Save Version → Run All (commit)** (background, survives tab close,
  9-hr cap). Never rely on the interactive session staying open.
- **Every notebook must checkpoint + resume** so a 9-hr cutoff doesn't lose work.
- **Shared artifacts are Kaggle Datasets, not git.** The subset (Phase A) and the embedding cache
  (Phase B) are published as Kaggle Datasets; all accounts read them. Put the dataset slug in the
  README/issue so others can attach it.
- Add `kaggle.json` locally if you use the CLI — it is **git-ignored**, never commit it.

---

## 3. Environment setup

**Python 3.10+.** Two ways to run:

- **Kaggle (preferred for GPU jobs):** most deps preinstalled. In a cell:
  `!pip install -q open_clip_torch timm datasets` (add `transformers` for SigLIP2). Attach the shared
  Kaggle Datasets as inputs instead of downloading.
- **Local (RTX 4060 / smoke tests):**
  ```bash
  python -m venv .venv && . .venv/Scripts/activate    # Windows: .venv\Scripts\activate
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  pip install open_clip_torch timm datasets transformers huggingface_hub pandas numpy pillow tqdm onnx onnxruntime
  ```
  (A pinned `requirements.txt` will be added with the first real script.)

**Credentials:** HF token for streaming SAGE (`huggingface-cli login` or `HF_TOKEN` env var);
`kaggle.json` for the Kaggle CLI; a Claude/Anthropic key for descriptor generation (Phase A2).
**All of these are git-ignored — never commit secrets.**

---

## 4. Git workflow

- Default branch is **`main`** (protected by convention — don't force-push it).
- **Branch per task:** `phaseA/build-subset`, `phaseC/student-train`, `docs/paper`, etc.
- Small, focused commits. Message says **why**, not just what.
- Open a **PR** into `main`; another member skims it before merge (even a quick LGTM).
- **Never commit:** weights, datasets, `*.parquet`, embedding caches, `kaggle.json`, `.env`,
  notebook checkpoints. (`.gitignore` already blocks these — don't `-f` past it.)
- Large outputs → Kaggle/HF; **paste the link** in the PR or an issue.

```bash
git checkout -b phaseA/build-subset
# ...work...
git add scripts/build_sage_subset.py
git commit -m "Phase A1: SAGE stream-filter subset builder (11 crops, per-class cap, manifest)"
git push -u origin phaseA/build-subset
# open a PR on GitHub
```

---

## 5. Definition of done (per task)

A task is done when its **Deliverable** exists and its **"Done when"** criteria in `PROJECT_GUIDE.md §6`
are met — not just "the code runs." For experiments, "done" means **the result is recorded in the
results matrix (§7)**, with the numbers, so the paper can use it.

---

## 6. Communication

- Decisions and blockers → a GitHub **issue** (so they're searchable later), not just chat.
- Each phase has an owner (§1); the owner posts a one-line status when a sub-task lands.
- Anything that changes scope/architecture/data → ping **LEAD** before doing it (the guide is "locked"
  for a reason; changing it mid-stream costs the timeline).
