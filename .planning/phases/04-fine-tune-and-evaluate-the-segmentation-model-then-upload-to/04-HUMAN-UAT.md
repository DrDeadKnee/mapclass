---
status: DEFERRED — GPU-host gate, not run on the planning VM
phase: 04-fine-tune-and-evaluate-the-segmentation-model-then-upload-to
type: deferred-gpu-gate
analogous_to: 02-HUMAN-UAT.md
---

# Phase 4 GPU-Host Gate: Fine-tune + Evaluate Segmentation Model

**Status: DEFERRED — GPU-host gate, not run on the planning VM.**

This artifact lists the exact hand-commands for running the Phase 4 GPU training
and evaluation workload on the separate GPU box. It mirrors the structure of
`02-HUMAN-UAT.md` (networked-host gate). Do NOT run these commands on the
planning VM — it has no GPU, no live GCS credentials, and insufficient RAM for
real backbone forward passes.

Push planning-VM commits first (see Section 0), then pull on the GPU host and
follow Sections 1–8 in order.

---

## ⛔ BLOCKING PRECONDITION — Phase 6 must complete first

**Do NOT run any GPU training in this gate until Phase 6 (historical
page-border void detection) is complete AND the 4 historical samples have
been rebuilt to GCS.**

Rationale: Phase 6 changes the historical training labels (emits the `255`
void sentinel for non-terrain page regions — title pages, cartouches,
bindings). The Phase-4 training matrix here consumes
`gs://mapclass-training-northeast1/data/historical/dataset/`. Training the
grid against the *pre-Phase-6* historical labels would burn GPU on a model
fit partly to scanned paper, requiring a full retrain. The
`ignore_index=255` loss fix (commit `e65bd8c`) is already in place and
expects those void labels to exist.

Resume checklist before proceeding past this point:

- [ ] Phase 6 executed and verified (`spec → discuss → plan → execute`)
- [ ] `build_historical_dataset.py build` re-run; the 4
      `gs://.../data/historical/dataset/RUMSEY_*__plate0/` dirs rebuilt with
      fresh `_BUILD_COMPLETE` sentinels (post-Phase-6 labels)
- [ ] STATE.md Blockers/Concerns gate cleared

The SigLIP Variant B **cost probe** (a few steps, GPU-minutes, not the
deliverable model) MAY run before Phase 6 — it only measures GPU-hrs/epoch
and rebuilding 4 of ~203 samples will not move that number. The **full
04-05 training grid** MUST NOT.

---

## Section 0: Push-Before-Pull Discipline

On the **planning VM** (before switching to the GPU host):

```bash
# Ensure all plan commits are pushed to origin
git push origin phase4
```

On the **GPU host** (before any other step):

```bash
git pull origin phase4
# Confirm you have the finetune_seg.py, evaluate_seg.py, seg/gcs_checkpoint.py
# and the rest of Phase 4 plan outputs:
git log --oneline -10
```

---

## Section 1: GPU-Host Preconditions

### 1.1 Python packages (run once on GPU host)

```bash
pip install gcsfs google-cloud-storage
```

These are not installed on the planning VM (`gcsfs` is runtime-only; see
04-02-SUMMARY.md decision note). Both are required before any GCS checkpoint
write or resume.

### 1.2 GCS authentication (IAM / ADC — no key files in git)

```bash
gcloud auth application-default login
```

This establishes V4 IAM credentials for the `narrative-campaign` GCP project.
The bucket `gs://mapclass-training-northeast1/models/` uses V4 access control;
no service-account key files are committed to git (D-08, T-04-11).

Verify auth is working:

```bash
gcloud storage ls gs://mapclass-training-northeast1/
```

Expected: an empty listing (or existing model prefixes if prior runs exist).
If you get `AccessDenied`, re-run `gcloud auth application-default login`.

### 1.3 Verify the training data is present

```bash
ls data/synthetic/train/      # should contain map subdirs from Phase 2
ls data/synthetic/test/       # held-out test set — DO NOT pass to finetune_seg.py
```

If `data/synthetic/` is absent, re-run the Phase 2 pipeline (`build_dataset.py`)
to regenerate the synthetic dataset on the GPU host or rsync it from the
planning VM.

---

## Section 2: timm Tag Verification (DINOv2 + Swin)

Before training DINOv2 or Swin backbones, confirm that the timm model tags used
by `Dinov2Backbone` and `SwinBackbone` still resolve (RESEARCH Assumptions A5/A6).

```bash
# DINOv2 — confirm the tag used by Dinov2Backbone resolves
python -c "
import timm
models = timm.list_models('*dinov2*', pretrained=True)
print('DINOv2 models available:')
for m in models:
    print(' ', m)
"

# Swin — confirm the tag used by SwinBackbone resolves
python -c "
import timm
models = timm.list_models('*swin*', pretrained=True)
print('Swin models available:')
for m in models:
    print(' ', m)
"
```

Expected: each list is non-empty and contains the specific tag that
`scripts/seg/backbones.py` passes to `timm.create_model(...)`.

If a tag has been renamed upstream (timm occasionally renames models), open
`scripts/seg/backbones.py` → `Dinov2Backbone` / `SwinBackbone` and update the
`_MODEL_ID` or equivalent constant before proceeding to training.

---

## Section 3: Cost Probe (D-03/D-04)

Run the SigLIP Variant B cost probe — this is the mandatory first step before
any full training run. It measures real GPU-hrs/epoch and projects the full-grid
cost. The output feeds the plan-05 `checkpoint:decision` gate.

```bash
python scripts/finetune_seg.py \
  --backbone siglip \
  --variant B \
  --probe \
  --epochs 10 \
  --train-root data/synthetic/train \
  --ckpt-every 50
```

The command will:
1. Construct the train/val split via `carve_train_val` (train/ only, EVAL-01).
2. Run exactly **one epoch** over the training split.
3. Print a structured `PROBE RESULT` block, e.g.:

```
PROBE RESULT | measured_epoch_hrs=0.4321 (GPU-hrs) | projected_full_grid_hrs=25.93 | configs=6 | epochs=10
```

**Record the printed `PROBE RESULT` line in full** — this is the required input
to the plan-05 `checkpoint:decision` gate. The gate will present options based on
the projected cost (full 6-config grid vs. reduced breadth).

Note: the probe does NOT write a GCS checkpoint (it exits before completing an
optimizer step on epoch 2). No GCS state is created by a probe-only run.

---

## Section 4: DINOv2/Swin Pretrained Flag Callout

**IMPORTANT:** For any DINOv2 or Swin training run, you MUST pass `--pretrained`
(which is the default). DO NOT pass `--no-pretrained` on the GPU host.

| Flag | Effect | When to use |
|------|--------|-------------|
| `--pretrained` (default) | Loads timm pretrained weights | Always on GPU host |
| `--no-pretrained` | Random backbone weights | Offline CI / planning VM only |

Passing `--no-pretrained` for DINOv2/Swin on the GPU host yields a random-weight
baseline, making the EVAL-03 comparison meaningless (RESEARCH Pitfall 6). The
SigLIP backbone always uses the Phase-1 LoRA-adapted weights regardless of this
flag (the stub path is only active when `--offline-stub` is also passed).

---

## Section 5: Full Grid Training Run (Post Cost-Probe Gate)

Run these commands after the plan-05 `checkpoint:decision` gate has been resolved
and the grid breadth chosen. Replace `--epochs N` with the epoch count decided at
the gate.

Each command writes checkpoints to
`gs://mapclass-training-northeast1/models/<backbone>-<variant>/step_NNNNNNN.pt`.

### 5.1 SigLIP Variant A

```bash
python scripts/finetune_seg.py \
  --backbone siglip \
  --variant A \
  --epochs N \
  --lr 1e-4 \
  --ckpt-every 50 \
  --train-root data/synthetic/train \
  --amp
```

### 5.2 SigLIP Variant B

```bash
python scripts/finetune_seg.py \
  --backbone siglip \
  --variant B \
  --epochs N \
  --lr 1e-4 \
  --ckpt-every 50 \
  --train-root data/synthetic/train \
  --amp
```

### 5.3 DINOv2 Variant A

```bash
python scripts/finetune_seg.py \
  --backbone dinov2 \
  --variant A \
  --pretrained \
  --epochs N \
  --lr 1e-4 \
  --ckpt-every 50 \
  --train-root data/synthetic/train \
  --amp
```

### 5.4 DINOv2 Variant B

```bash
python scripts/finetune_seg.py \
  --backbone dinov2 \
  --variant B \
  --pretrained \
  --epochs N \
  --lr 1e-4 \
  --ckpt-every 50 \
  --train-root data/synthetic/train \
  --amp
```

### 5.5 Swin Variant A

```bash
python scripts/finetune_seg.py \
  --backbone swin \
  --variant A \
  --pretrained \
  --epochs N \
  --lr 1e-4 \
  --ckpt-every 50 \
  --train-root data/synthetic/train \
  --amp
```

### 5.6 Swin Variant B

```bash
python scripts/finetune_seg.py \
  --backbone swin \
  --variant B \
  --pretrained \
  --epochs N \
  --lr 1e-4 \
  --ckpt-every 50 \
  --train-root data/synthetic/train \
  --amp
```

If the gate selected a reduced grid (e.g. SigLIP A+B only, or benchmarks with
fewer epochs), run only the relevant subset of commands above.

---

## Section 6: Preemption/Resume Verification (D-09)

To verify auto-resume works correctly before committing to a full training run:

```bash
# Start a run and interrupt it mid-training with Ctrl-C (or let VM preempt)
python scripts/finetune_seg.py \
  --backbone siglip \
  --variant B \
  --epochs N \
  --ckpt-every 10 \
  --train-root data/synthetic/train

# Verify a checkpoint was written to GCS
gcloud storage ls "gs://mapclass-training-northeast1/models/siglip-b/"
# Expected: one or more step_NNNNNNN.pt files

# Relaunch the same command — it should auto-resume from the latest GCS step
python scripts/finetune_seg.py \
  --backbone siglip \
  --variant B \
  --epochs N \
  --ckpt-every 10 \
  --train-root data/synthetic/train
```

Expected output on relaunch:
```
Resuming from GCS step NNNN (config=siglip-b)
```

If you see `Starting fresh (config=siglip-b)`, the checkpoint was not found.
Check GCS access (Section 1.2) and that `--ckpt-every` was small enough to
write at least one checkpoint before the interruption.

---

## Section 7: Evaluation Run (EVAL-02/EVAL-03)

Run `evaluate_seg.py` per config over `data/synthetic/test/` via `split.json`.
This produces `metrics/<config>-nll.json` for each backbone/variant combination.

**IMPORTANT:** `evaluate_seg.py` reads `data/synthetic/test/` only (from
`split.json["test"]`). The training script NEVER touches `test/` — EVAL-01
zero-leakage is preserved end-to-end.

First, confirm `split.json` points at the held-out synthetic test set:

```bash
python -c "
import json
split = json.loads(open('data/synthetic/split.json').read())
print(f'Test maps: {len(split[\"test\"])} | Train maps: {len(split[\"train\"])}')
"
```

Then run evaluation for each trained config. Adjust `--checkpoint` to the path
(local or `gs://`) of the final trained checkpoint for each config:

```bash
mkdir -p metrics

# SigLIP Variant A
python scripts/evaluate_seg.py \
  --backbone siglip --variant A \
  --split-json data/synthetic/split.json \
  --data-root data/synthetic \
  --metrics-dir metrics \
  --checkpoint gs://mapclass-training-northeast1/models/siglip-a/step_NNNNNNN.pt

# SigLIP Variant B
python scripts/evaluate_seg.py \
  --backbone siglip --variant B \
  --split-json data/synthetic/split.json \
  --data-root data/synthetic \
  --metrics-dir metrics \
  --checkpoint gs://mapclass-training-northeast1/models/siglip-b/step_NNNNNNN.pt

# (Repeat for dinov2-a, dinov2-b, swin-a, swin-b as applicable)
```

Each command produces `metrics/<config>-nll.json` with keys:
`joint_nll`, `backbone`, `variant`, `eval_date`.

Verify the outputs:

```bash
ls metrics/
cat metrics/siglip-b-nll.json
```

---

## Section 8: Hand-Back (D-08 — Weights NEVER Committed)

After training and evaluation are complete, commit only the lightweight artifacts:

```bash
# Stage metrics JSON (joint-NLL per config)
git add metrics/*.json

# Stage the GCS URI manifest (text file listing gs:// checkpoint URIs)
# Create this manually or adapt report_comparison.py output:
echo "gs://mapclass-training-northeast1/models/siglip-a/step_NNNNNNN.pt" > artifacts/checkpoint_uris.txt
echo "gs://mapclass-training-northeast1/models/siglip-b/step_NNNNNNN.pt" >> artifacts/checkpoint_uris.txt
# (add other configs as trained)
git add artifacts/checkpoint_uris.txt

# Stage comparison report (produced by scripts/report_comparison.py in plan 04-05)
git add docs/04-comparison-report.md

git commit -m "results(04): joint-NLL metrics + checkpoint URI manifest + comparison report"
git push origin phase4
```

**NEVER commit model weights to git.** The `.pt` files must remain in GCS only.
Add `*.pt` to `.gitignore` if not already present:

```bash
grep -q '*.pt' .gitignore || echo '*.pt' >> .gitignore
```

---

## Checklist

- [ ] Section 0: Planning VM push + GPU host pull complete
- [ ] Section 1: gcsfs + google-cloud-storage installed; `gcloud auth ADC` working
- [ ] Section 2: timm DINOv2 + Swin tags verified (or updated if renamed)
- [ ] Section 3: Cost probe run; PROBE RESULT line recorded
- [ ] Section 4: DINOv2/Swin runs use `--pretrained` (not `--no-pretrained`)
- [ ] Section 5: Full grid training complete (breadth per plan-05 gate)
- [ ] Section 6: Preemption/resume verified for at least one config
- [ ] Section 7: evaluate_seg.py run per trained config; metrics/*.json produced
- [ ] Section 8: Only metrics JSON + URI manifest + report committed; no *.pt in git
