# EVAL-03 Protocol — Small-Backbone vs PaliGemma NLL Bellwether

**Pre-registered:** Phase 1 (walking skeleton)
**Date:** 2026-05-09
**Authority:** ROADMAP.md Phase 1 success criterion 6 + CONTEXT.md D-05/D-06/D-07/D-08
**Pitfall mitigated:** PITFALL 2 (CRITICAL) — pre-registration of the comparison structure before any real training run starts.

This document is the audit trail of the EVAL-03 contract. Phase 5's `eval_03_compare.py` reads this file at startup and refuses to run if the `DECIDE_AT_PHASE_5` literal is still present (D-08 mechanical guard). The structure of the comparison is locked in Phase 1; only the numeric NLL-gap threshold is deferred to Phase 5 (D-06 calibrated-after-v0-results compromise).

## 1. Held-out split hash

The committed deterministic split manifest at `mapclass/configs/splits.json` is the canonical held-out set. PaliGemma and the small backbone are evaluated against the byte-identical file.

- **Path:** `mapclass/configs/splits.json`
- **sha256:** `a7b61846833381501db6b3d42a80575cd28471b62ccc9c21135051a70211d421`
- **Bucket policy** (PITFALL 5 prevention #1): sample-id sha256 first 32 bits → bucket 0–9; buckets 0–7 = train, 8 = val, 9 = test (80/10/10).

## 2. Loss-weights file hash

The per-source × per-class loss weights table at `mapclass/configs/loss_weights.yaml` is byte-identical between the two backbones. This hash equals `LossWeights.load(...).source_class_weights_hash`.

- **Path:** `mapclass/configs/loss_weights.yaml`
- **sha256:** `10f7e4dd4a3e7759152019721644939646f8b20f38dbde2385632b206c9c09ab`

## 3. Augmentation policy

Phase 1 ships **no augmentations beyond per-channel normalization** via `Backbone.preprocess()` (PITFALL 4 prevention #1: backbone owns the processor). PaliGemma in Phase 5 inherits the same augmentation policy — i.e. none. If Phase 2/3 introduces augmentations (rotation/jitter/color/parchment), this section MUST be amended in a separate commit before EVAL-03 runs (D-07 strict-fairness corollary).

- **Phase 1 normalize:** `mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]` (matches `MockBackbone` preprocess).
- **Augmentation:** none.

## 4. Fine-tune budget shape

The training budget that Phase 5 PaliGemma runs must respect (D-07 strict fairness — only the `backbone:` field differs between configs):

- **Epochs:** matches the small-backbone v0/v1 config epoch count (locked at Phase 3 / Phase 4 commit time); Phase-1 placeholder = 2 (`mapclass/configs/v0.yaml n_epochs`).
- **Batch size:** matches the small-backbone config (`mapclass/configs/v0.yaml batch_size`); Phase-1 placeholder = 4.
- **LR schedule:** linear-warmup + cosine-decay (default per `mapclass.train` Phase-3 plan); Phase-1 placeholder = constant `lr=1e-3` (Phase 5 inherits the v0/v1 schedule, NOT the Mock placeholder).
- **Image size:** 384 (`mapclass/configs/v0.yaml image_size`); PaliGemma evaluated at 224 — the ONE deviation noted in CONTEXT.md D-07 KNOWN TENSION (Phase 5 must verify a one-batch dry-run before committing to strict fairness).
- **Taxonomy hash:** `052624d2022aa6650b4ef09a519d86c3d4c8eb4a0fe8d7cbc683f3947a489e22` (the 9-class land-cover + 3-class topography contract; embedded in both checkpoints' safetensors metadata).

## 5. Evaluation seed

A single integer reproducibility seed for the held-out evaluation pass. Same seed for both backbones. Matches the value in `mapclass/configs/v0.yaml`:

- **seed:** `42`

## 6. Kill-switch criterion

Per CONTEXT.md D-06: the comparison rule structure is committed in Phase 1; only the numeric threshold is deferred to Phase 5.

> The small backbone ships if the NLL gap to PaliGemma is within `DECIDE_AT_PHASE_5`% relative on the held-out splits (per-source mean and overall mean both within threshold). Otherwise, MODEL-01 is reconsidered before ship — recovery options noted in ROADMAP.md Phase 5 success criterion 5.

## Phase 5 enforcement (D-08 two-commit policy)

The literal placeholder string `DECIDE_AT_PHASE_5` above MUST be replaced with a numeric threshold (e.g. `5`) in a SEPARATE COMMIT before `eval_03_compare.py` runs against any PaliGemma checkpoint.

- Phase 5's `eval_03_compare.py` reads this file at startup and `grep`s for the literal `DECIDE_AT_PHASE_5`. If present, it raises `RuntimeError` and refuses to run.
- The threshold MUST be set BEFORE either PaliGemma checkpoint is evaluated, so the gap-acceptability decision cannot be retroactively shaped to fit the observed numbers.
- The replacement commit message MUST reference Phase 3 v0 NLL stability (the calibration anchor per D-06).

## Strict fairness statement (D-07)

There is no "Permitted Differences" table for EVAL-03. The two configs differ in exactly the `backbone:` field, the `dtype:` field (NF4 + LoRA for PaliGemma per STACK.md), and the input resolution (224 vs 384 — the ONE acknowledged deviation under D-07's KNOWN TENSION). All other fields — augmentation, splits, loss weights, evaluation seed — are byte-identical (enforced via the hashes in §1 and §2 above).

If the Phase 5 one-batch dry-run on the cloud VM (PITFALL 6 prevention) reveals strict fairness is infeasible (e.g. 384-input PaliGemma OOMs), this protocol is RE-OPENED in Phase 5, NOT silently amended. The re-opening triggers a planner discuss-phase round; this is the only legitimate path to relax strict fairness.
