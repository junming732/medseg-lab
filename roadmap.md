# MedSeg Lab — Roadmap & Playbook (SOTA Medical Image Segmentation)

_A compact, actionable plan to turn your GitHub into a living portfolio of state‑of‑the‑art medical image segmentation, with structure, milestones, and CI so continuous learning is visible._


---

## TL;DR — Your Next 3 Steps

- [ ] **Create a branch**: `git checkout -b feat/unetr-btcv`
- [ ] **Run the synthetic smoke**: `python tools/train.py --config configs/unetr/btcv.yaml --synthetic true --max_epochs 1`
- [ ] **Open a PR** with the results + a short progress note in `docs/progress/`

---

## 1) Repo Strategy (unify your work)

**Goal:** A single, clean lab where new models/datasets plug in quickly.

**Do this**
- Evolve your current project into **`medseg-lab`** (or create a new repo).
- Use a consistent API and folders so models/datasets are swappable.

**Suggested tree**
```
medseg-lab/
  configs/          # Hydra/YAML configs per model+dataset
  datasets/         # Download/convert scripts (MSD/BraTS/KiTS/BTCV)
  engines/          # Train/val loops (Lightning or raw PyTorch)
  models/           # UNet/ResUNet, MONAI nets, wrappers (nnUNetv2, SAM)
  tools/            # train.py, eval.py, visualize.py, export.py
  notebooks/        # EDA & “learning journal” demos
  tests/            # CPU smoke & unit tests
  docs/             # mkdocs site: progress, dataset cards, results
  .github/workflows # CI jobs (PR + nightly mini-bench)
  docker/           # CPU and CUDA images
  README.md LICENSE CONTRIBUTING.md CODE_OF_CONDUCT.md CITATION.cff
```

**Deliverable:** repo scaffold committed; minimal README + CI badge.

---

## 2) Model Roadmap (baseline → SOTA → foundation)

**Goal:** Climb from baselines to current SOTA with clear comparisons.

**Do this**
- Baselines: **UNet / ResUNet** (2D & 3D), strong aug, mixed precision.
- Transformers via **MONAI**: **UNETR**, **Swin‑UNETR**.
- Auto‑config gold standard: **nnU‑Net v2** as an external wrapper.
- Promptable/foundation: **SAM 2 / MedSAM** (interactive demo).

**Deliverable:** `models/*` + `configs/*`; a results table in README/docs.

---

## 3) Datasets & Benchmarks (curate + standardize)

**Goal:** Reproducible evaluations across common medical tasks.

**Do this**
- Start small: **BTCV**, **ACDC**, **BraTS**, **KiTS**; add **MSD** tasks later.
- Ship **dataset cards** in `docs/datasets/*.md` (modality, size, license, metrics).
- Provide converters and integrity checks (hashes). Keep raw data out of Git.

**Deliverable:** `datasets/` scripts + dataset cards; config toggles per task.

---

## 4) Training & Evaluation Standards

**Goal:** Consistent training loops and metrics you can trust.

**Do this**
- Common engine (Lightning or hooks): seed control, AMP, checkpoint naming.
- Augmentations with MONAI transforms per modality.
- Metrics: **Dice**, **Jaccard**, **HD95**, **ASSD**. Export CSV under `results/`.
- Save `config.yaml` next to each checkpoint for reproducibility.

**Deliverable:** `tools/train.py`, `tools/eval.py`, metrics CSVs per run.

---

## 5) Clear, Visible Learning Path (make it obvious on GitHub)

**Goal:** Every improvement is a public “story” with code + results.

**Do this**
- Add **/docs/progress/** dated notes:
  `docs/progress/YYYY-MM-DD-topic.md` with Goal/Setup/Results/Learnings/Next.
- Keep a **Roadmap** section in README linking to open PRs/milestones.
- Screenshots/plots attached to PRs for quick scanning.

**Deliverable:** at least one progress note per PR; README Roadmap updated.

---

## 6) Tooling & Developer Experience (fast to run, easy to review)

**Goal:** Anyone can reproduce a run locally—no SaaS required.

**Do this**
- Dependency mgmt via `pyproject.toml` (extras: `[cpu]`, `[cuda]`, `[test]`).
- `Makefile` targets: `setup`, `test`, `train`.
- Local **MLflow** (later) or CSV logs; `visualize.py` for overlays & 3D slices.
- Optional: small **Streamlit** app for interactive review & SAM prompts.

**Deliverable:** Make targets + simple logging; a tiny viz script or app stub.

---

## 7) CI/CD that Proves Quality (GPU‑free)

**Goal:** Confidence before merging; nightly guardrails.

**Do this**
- **PR CI**: lint/type (optional), unit + **CPU smoke tests** on synthetic 3D.
- **Nightly mini‑bench**: eval on 2–4 cases per dataset subset; publish HTML to GitHub Pages; keep runtimes < 15 min.

**Deliverable:** `.github/workflows/ci.yml` + `nightly-bench.yml` and a small report artifact/page.

---

## 8) Documentation (publish it)

**Goal:** A browsable site for users and recruiters.

**Do this**
- Use **mkdocs‑material**.
- Pages: *Quickstart*, *Models*, *Datasets*, *Training recipes*, *Results*, *Progress*, *Changelog*, *Reproducibility*.
- Badges in README: build, docs, license, last update.

**Deliverable:** `docs/` with mkdocs config; GitHub Pages enabled.

---

## 9) Concrete First Milestones (copy as PR titles)

1. **UNETR baseline (BTCV)** — config, dataloader (or synthetic for smoke), metrics CSV, progress note.
2. **Swin‑UNETR recipe** — compare vs. UNet/UNETR.
3. **nnU‑Net v2 wrapper** — converter + training script; result card.
4. **SAM/MedSAM interactive demo** — point/box prompts; notes on strengths/limits.
5. **Nightly CPU mini‑bench** — cron CI + GitHub Pages leaderboard snapshot.
6. **Docs site + “progress” posts** — publish the learning journal; link PRs.

**Deliverable:** Each item merged via PR with a short results table and a progress note.

---

## 10) Ready‑to‑Paste Snippets (quick reuse)

**Quickstart**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .[cpu]
pytest -q -k "cpu_smoke"
```

**Train (example)**
```bash
python tools/train.py --config configs/unetr/btcv.yaml
```

**Evaluate (example)**
```bash
python tools/eval.py --checkpoint runs/.../best.ckpt --split val
```

**Results table (README/docs)**
```markdown
| Model       | Dataset | Dice ↑ | HD95 ↓ | Notes          |
|-------------|---------|--------|--------|----------------|
| UNet (2D)   | MSD-XX  | 0.XX   |  X.X   | baseline       |
| UNETR (3D)  | BTCV    | 0.XX   |  X.X   | transformer    |
| nnU‑Net v2  | BTCV    | 0.XX   |  X.X   | auto‑configure |
```

---

# PR Workflow — Quick Guide

**Branch → experiment → note → PR → merge.**

1. Create a branch per experiment:
   ```bash
   git checkout -b feat/unetr-btcv
   ```
2. Run & record metrics (console, CSV, or MLflow).
3. Write a progress note:
   `docs/progress/{YYYY-MM-DD}-{topic}.md`.
4. Commit everything and push:
   ```bash
   git add .
   git commit -m "feat: UNETR baseline on BTCV; progress note YYYY-MM-DD"
   git push -u origin feat/unetr-btcv
   ```
5. Open a PR (auto-uses your PR template). Include **How to reproduce**, **Results**, **Next steps**.
6. Merge when CI is green.

**Labels you can use:** `experiment`, `baseline`, `docs`, `infra`, `demo`.

---

## Checklists (keep at top of your repo README)

### On every PR
- [ ] Branch named clearly (e.g., `feat/…`, `exp/…`, `demo/…`)
- [ ] Synthetic CPU smoke test passes locally (`pytest -k cpu_smoke`)
- [ ] PR includes results (table/screenshot) + progress note
- [ ] “How to reproduce” command is copy‑paste ready

### For model additions
- [ ] Config added under `configs/<model>/<dataset>.yaml`
- [ ] Model code under `models/` or wrapper `tools/*`
- [ ] Metrics exported to `results/*.csv`
- [ ] README results table updated

---

_Last updated: 2025-10-09_
