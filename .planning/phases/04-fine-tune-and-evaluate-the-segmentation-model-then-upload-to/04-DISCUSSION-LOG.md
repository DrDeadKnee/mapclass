# Phase 4: Fine-tune and evaluate the segmentation model - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-16
**Phase:** 4-Fine-tune and evaluate the segmentation model
**Areas discussed:** Phase 4/5 re-scope, Training matrix scope, GPU execution & handoff
**Areas deferred to Phase 5 by user:** Deliverable & model selection, HuggingFace publish shape

---

## Phase 4/5 re-scope (user-initiated)

User declined to discuss "Deliverable & model selection" and "HuggingFace
publish shape" in Phase 4, stating they need to test/play with the models and
dataset by hand before delivering or publishing.

| Option | Description | Selected |
|--------|-------------|----------|
| Numbers + checkpoints + report | Trained checkpoints + joint-NLL eval on held-out test + written A/B × backbone comparison report; no selection/packaging | ✓ |
| Checkpoints only, eval deferred too | Pure training in P4; eval also moves to P5 | |
| Numbers + checkpoints, no formal report | Raw metrics dumped, no written narrative | |

**User's choice:** Numbers + checkpoints + report.
**Notes:** DELIV-01, DELIV-02, and EVAL-02's "reported-with-published-model"
clause move to a new Phase 5 (hands-on exploration → select → publish). EVAL-02
*computation* and EVAL-03 stay in Phase 4. Requires a ROADMAP edit.

---

## Training matrix scope

| Option | Description | Selected |
|--------|-------------|----------|
| Full 6-config grid | Train all A/B × SigLIP/DINOv2/Swin to completion | |
| SigLIP A+B full, others single-variant | Fully train SigLIP A+B; one variant for DINOv2/Swin | |
| SigLIP A+B full, others light/short | SigLIP A+B full; short benchmark runs for DINOv2/Swin | |
| Decide after a cost probe | Short calibration run, then gated breadth decision from measured cost | ✓ |

**User's choice:** Decide after a cost probe.

### Probe & gate follow-up

| Option | Description | Selected |
|--------|-------------|----------|
| SigLIP Variant A; you decide | Cheapest probe; manual checkpoint:decision gate | |
| SigLIP Variant B; you decide | Probe upper-bounds SigLIP cost; manual gate | ✓ |
| SigLIP A; auto budget rule | Probe A; auto-select breadth under a GPU-hour ceiling | |

**User's choice:** SigLIP Variant B; you decide.
**Notes:** Gate mirrors the Phase-2 blocking checkpoint:decision pattern.

---

## GPU execution & handoff

### Artifact handoff

| Option | Description | Selected |
|--------|-------------|----------|
| Checkpoints stay on GPU box | Weights local to GPU box; only metrics+report to git | |
| Push to private HF Hub repos | Per-config private HF repos as transfer | |
| Shared storage / scp | Shared volume or scp + path manifest | ✓ (GCS variant) |

**User's choice:** Checkpoints to GCP — `gs://mapclass-training-northeast1/models/`
(per-config prefix). Git gets URI manifest + joint-NLL metrics + report.
**Notes:** No GCS path convention exists in this repo today (data/models are
local `data/`); Phase 4 introduces it. User initially recalled an in-repo
"narrative-campaign" gs:// convention; grep confirmed none exists — surfaced to
user, who then supplied the exact bucket path.

### Resumption

| Option | Description | Selected |
|--------|-------------|----------|
| Periodic GCS checkpoints + auto-resume | {weights,optimizer,step} every N steps; auto-detect latest on restart | ✓ |
| Best+last only, manual resume | Persist best+last; manual re-launch | |
| Epoch-boundary only | Checkpoint at epoch boundaries, auto-resume last epoch | |

**User's choice:** Periodic GCS checkpoints + auto-resume (preemption-safe).

---

## Claude's Discretion

- Training hyperparameters, optimizer, LR schedule, epoch budget, weighted-loss
  formulation consuming `sample_weights.json`.
- Validation-signal methodology — constrained: val slice from `train/` ONLY,
  never `test/` (EVAL-01). Researcher recommends; not re-asked.
- Exact checkpoint cadence N, GCS I/O + resume-detection logic.
- Recursive c2f invocation at eval time; comparison-report format.
- Pyramid-aware batch sampling / dataloader internals.

## Deferred Ideas

- Deliverable / model selection → Phase 5.
- HuggingFace publish shape (repo/org/license/card/benchmark numbers) → Phase 5.
- OCR / place-name reading → future v2 / GEOREF-V2 (carried from Phase 3).
