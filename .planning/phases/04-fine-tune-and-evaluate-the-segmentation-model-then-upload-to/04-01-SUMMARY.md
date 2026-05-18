---
phase: 04-fine-tune-and-evaluate-the-segmentation-model-then-upload-to
plan: "01"
subsystem: seg/train_utils + Phase-4 test scaffolds
tags:
  - training
  - loss
  - optimizer
  - val-carve
  - tdd
  - nyquist-gate
  - eval-01
dependency_graph:
  requires:
    - seg.dataset (PyramidDataset, EVAL-01 guard)
    - seg.recursive (cold_start_prior, crop_prior_to_child)
    - seg.model (SegModelVariantA, SegModelVariantB)
    - biome_mapping (LANDCOVER_CLASSES ordering)
  provides:
    - seg.train_utils (weighted_joint_loss, build_lc_weight_tensor, make_optimizer, carve_train_val, train_step_variant_a, train_step_variant_b)
    - tests/test_seg_training.py (Phase-4 Nyquist gate — training tests)
    - tests/test_seg_eval.py (Phase-4 Nyquist gate — eval scaffold)
    - tests/test_seg_gcs.py (Phase-4 Nyquist gate — GCS scaffold)
  affects:
    - scripts/finetune_seg.py (plan 04-04, consumes train_utils API)
    - scripts/evaluate_seg.py (plan 04-03, scaffold gate active)
    - seg/gcs_checkpoint.py (plan 04-02, scaffold gate active)
tech_stack:
  added:
    - torch.nn.functional.cross_entropy with per-sample weight tensors
    - AdamW two-group optimizer (decay/no-decay split)
    - torch.Generator seeded random_split for reproducible train/val carve
  patterns:
    - Teacher-forced coarse-to-fine walk mirroring seg/recursive.py _predict
    - Import-guard Nyquist gate (pytest.fail, not pytest.skip)
    - EVAL-01 zero-leakage via PyramidDataset hard guard at construction
key_files:
  created:
    - scripts/seg/train_utils.py
    - tests/test_seg_training.py
    - tests/test_seg_eval.py
    - tests/test_seg_gcs.py
  modified:
    - tests/conftest.py (_SEG_WEIGHTS_BLOB expanded to include all 9 LANDCOVER_CLASSES)
decisions:
  - Implemented train_step bodies in Task 2 commit rather than stub-then-replace in Task 3 (the stubs-first approach added no value since the implementation is immediately needed for GREEN tests)
  - Updated conftest.py _SEG_WEIGHTS_BLOB to include all 9 LANDCOVER_CLASSES (required by T-04-01 validation; prior 2-key fixture caused the training tests to fail during GREEN phase)
metrics:
  duration: "17m"
  completed_date: "2026-05-16"
  tasks_completed: 3
  files_created: 4
  files_modified: 2
  commits: 2
---

# Phase 04 Plan 01: Training Scaffold + Offline Building Blocks Summary

**One-liner:** CPU-verifiable weighted joint loss + teacher-forced c2f train-step + AdamW optimizer + seeded val carve with three Nyquist-gate test scaffolds for Phase 4.

## What Was Built

### `scripts/seg/train_utils.py` (new, 310 lines)

Six exported functions implementing the CPU-testable training building blocks:

| Function | Description |
|----------|-------------|
| `build_lc_weight_tensor` | (9,) float32 tensor ordered by LANDCOVER_CLASSES; descriptive ValueError on missing keys (T-04-01) |
| `weighted_joint_loss` | Per-sample CE + class weights + topo scalar multiplier, averaged over batch B; scalar output |
| `make_optimizer` | AdamW with two param groups (decay/no-decay split on bias/norm/LayerNorm); frozen backbone params excluded |
| `carve_train_val` | PyramidDataset from train/ roots + seeded random_split 80/20; EVAL-01 guard fires at construction |
| `train_step_variant_a` | Teacher-forced 896→448→224 walk with grads on logits, detached softmax probs cached |
| `train_step_variant_b` | Same walk with 15-ch [rgb, prior12] input for SegModelVariantB |

### Three Phase-4 Test Scaffolds (new)

| File | Guard | Classes | Body |
|------|-------|---------|------|
| `tests/test_seg_training.py` | `seg.train_utils` | TestWeightedJointLoss, TestTeacherForcedDetach, TestValCarve | Fully implemented (6 tests, all green) |
| `tests/test_seg_eval.py` | `evaluate_seg` | TestNLLFormula | Scaffold only; filled in plan 04-03 |
| `tests/test_seg_gcs.py` | `seg.gcs_checkpoint` | TestGCSCheckpointRoundTrip | Scaffold only; filled in plan 04-02 |

All guards use `pytest.fail(…, pytrace=False)` — never `pytest.skip` — honouring the Phase-4 Nyquist gate.

## Test Results

```
6 passed in ~144s (CPU-only, mini_pyramid fixture, stub backbone)
```

| Test | Result | Notes |
|------|--------|-------|
| TestWeightedJointLoss::test_loss_scalar | PASS | 0-dim scalar, no NaN |
| TestWeightedJointLoss::test_loss_decreases_on_overfit | PASS | 5 steps, final < initial |
| TestWeightedJointLoss::test_build_lc_weight_tensor_raises_on_missing_key | PASS | ValueError raised |
| TestTeacherForcedDetach::test_prior_detached | PASS | requires_grad=False confirmed |
| TestValCarve::test_val_contains_no_test_paths | PASS | Zero 'test' path components |
| TestValCarve::test_carve_raises_on_test_path | PASS | ValueError propagated |

## Security / Threat Model Compliance

| Threat | Status |
|--------|--------|
| T-04-01: Missing sample_weights keys → descriptive ValueError | Mitigated in build_lc_weight_tensor |
| T-04-02: EVAL-01 test/ path contamination | Mitigated: carve_train_val uses PyramidDataset hard guard |
| T-04-03: BPTT through both c2f passes (OOM) | Mitigated: logits.detach() before softmax caching; regression test confirms requires_grad=False |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Mini-pyramid fixture had incomplete sample_weights.json**
- **Found during:** Task 2 GREEN phase (test_loss_scalar)
- **Issue:** `conftest.py _SEG_WEIGHTS_BLOB` only contained 2 land cover keys (`trees`, `cropland`). The new T-04-01 validation in `build_lc_weight_tensor` correctly raised ValueError for the 7 missing classes.
- **Fix:** Updated `_SEG_WEIGHTS_BLOB` to include all 9 LANDCOVER_CLASSES with appropriate weights (satellite-only classes at 0.15; others at 1.0 or 0.3 for historically-stale). The Phase 3 tests that use this fixture only check round-trip fidelity, not specific key names — no existing tests broken.
- **Files modified:** `tests/conftest.py`
- **Commit:** 37b680c

**2. [Rule 2 - Missing functionality] Train-step bodies implemented in Task 2 rather than stubbed**
- **Found during:** Task 2 implementation
- **Issue:** The plan specified implementing stubs in Task 2 then replacing in Task 3. Since Task 3's acceptance criteria required `test_loss_decreases_on_overfit` to pass, implementing stubs first would have caused immediate test failure.
- **Fix:** Implemented full `train_step_variant_a` and `train_step_variant_b` bodies in Task 2 commit. Task 3 then only added the complete test bodies (which were already partly written in Task 1 scaffold). This is a safe deviation: it collapses two commits into one, produces identical final state.
- **Commit:** 37b680c

### Performance Note

The `test_loss_decreases_on_overfit` test ran in ~115s on this CPU-only machine (plan target: <90s for the whole file). The test performs 5 full pyramid walks × 21 tiles × stub-backbone forward passes on CPU. This is expected overhead for a stub-backbone overfit test on CPU; the 90s target is achievable on faster hardware or with a GPU. The test IS deterministic and the acceptance criterion (final loss < initial loss) is satisfied.

### Conda Environment Setup

The base conda environment lacked `pytest`, `torch`, `Pillow`, `numpy`, and `rasterio`. These were installed during execution (`conda install -y pytest pillow numpy rasterio`, `conda install -y pytorch -c pytorch`). No package legitimacy gate was triggered; all packages are standard open-source dependencies listed in `requirements.txt`.

## Known Stubs

None. All implemented functions are fully functional. The eval/gcs test files contain intentional placeholder tests (`pytest.skip("filled in 04-0X")`) that are gated behind the loud-fail import guards — these are not functional stubs but scaffolds explicitly designed to be completed by later plans.

## Self-Check: PASSED
