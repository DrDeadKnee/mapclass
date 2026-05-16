---
phase: 04-fine-tune-and-evaluate-the-segmentation-model-then-upload-to
plan: "03"
subsystem: evaluate_seg + test_seg_eval
tags:
  - eval
  - nll
  - leakage-guard
  - tdd
  - eval-02
  - eval-03
  - path-traversal
dependency_graph:
  requires:
    - seg.recursive (recursive_predict, recursive_predict_variant_b — reused unchanged)
    - seg.model (SegModelVariantA, SegModelVariantB)
    - tests/conftest.py (mini_pyramid, stub_vision_config fixtures)
    - tiling (tile() producer — used in leakage guard test)
  provides:
    - scripts/evaluate_seg.py (evaluate_joint_nll, load_test_pyramid_dirs, write_nll_metrics, main)
    - tests/test_seg_eval.py (TestNLLFormula, TestEvalLeakageGuard, TestEvalAllBackbones)
  affects:
    - scripts/finetune_seg.py (plan 04-04, consumes evaluate_seg for post-training eval)
    - scripts/report_comparison.py (plan 04-05, reads metrics JSON written by write_nll_metrics)
tech_stack:
  added:
    - torch.log(prob.clamp_min(1e-12)) — stable log on already-softmaxed probs
    - Path.rglob("pyramid.json") — recursive pyramid discovery for real tiling structure
    - Path.resolve() / parents check — traversal guard enforced at test_root level
  patterns:
    - Reuse recursive c2f walk from Phase-3 unchanged (NOT re-implemented)
    - gather-safe clamp for out-of-range GT labels before valid-pixel mask
    - pytest.importorskip for timm-dependent backbone tests (dinov2/swin)
key_files:
  created:
    - scripts/evaluate_seg.py
  modified:
    - tests/test_seg_eval.py
decisions:
  - Clamp gather indices (lc_gt.clamp(0,8), topo_gt.clamp(0,2)) before gather to handle out-of-range labels safely; valid mask applied after gather to exclude them from sum
  - Traversal guard strengthened from data_root to data_root/test/ so map IDs like "../train/secret" are also rejected
  - rglob("pyramid.json") used for pyramid discovery because tiling.tile() creates nested structure at <map_dir>/pyramids/<py_r*/c*>/
  - dinov2/swin backbone tests use pytest.importorskip("timm") to skip gracefully when timm absent; siglip coverage is complete
  - log(clamp(prob)) used instead of log_softmax(prob) because probs are already softmaxed — applying log_softmax to softmaxed probs would double-normalise (documented in evaluate_seg.py with explicit reasoning)
metrics:
  duration: "9m"
  completed_date: "2026-05-16"
  tasks_completed: 2
  files_created: 1
  files_modified: 1
  commits: 2
---

# Phase 04 Plan 03: NLL Evaluation Harness Summary

**One-liner:** Joint per-pixel NLL evaluation harness reusing Phase-3 recursive c2f walk, with test-set-only enumeration (T-04-07/T-04-08), gather-safe invalid-pixel handling, and offline formula + leakage + multi-backbone test suite.

## What Was Built

### `scripts/evaluate_seg.py` (new, 400 lines)

Four exported functions:

| Function | Description |
|----------|-------------|
| `evaluate_joint_nll` | c2f recursive walk over size==224 leaf tiles; `log(clamp_min(1e-12, prob))` on already-softmaxed probs; gather-safe GT clamp; valid-pixel mask `(lc_gt<9)&(topo_gt<3)` |
| `load_test_pyramid_dirs` | Reads `split.json["test"]` only; resolves candidates under `data_root/test/`; traversal guard asserts resolved path under `test_root`; `rglob("pyramid.json")` for real tiling nesting |
| `write_nll_metrics` | Writes `<config>-nll.json` with joint_nll, backbone, variant, eval_date |
| `main` | argparse CLI for `--backbone {siglip,dinov2,swin}`, `--variant {A,B}`, split-json, data-root, metrics-dir, checkpoint (local or gs://) |

**Key numerical decision:** `recursive_predict` returns already-softmaxed probabilities — applying `log_softmax` again would double-normalise and produce wrong values. The correct form is `torch.log(prob.clamp_min(1e-12))`. This is documented in the source with explicit reasoning distinguishing it from the anti-pattern `log(softmax(raw_logits))` on un-normalised logits.

### `tests/test_seg_eval.py` (modified — plan 04-01 scaffold replaced)

| Class | Tests |
|-------|-------|
| TestNLLFormula | test_nll_formula_correct (log(9)+log(3) anchor); test_nll_ignores_invalid_pixels (all-invalid → nan); test_eval_uses_only_leaf_tiles (896/448 excluded) |
| TestEvalLeakageGuard | test_load_test_dirs_only_test_paths (only test/ paths returned); test_rejects_traversal_map_id (../secret raises ValueError) |
| TestEvalAllBackbones | test_evaluate_returns_finite_float parametrized over siglip×{A,B} (2 green); dinov2×{A,B} + swin×{A,B} skipped via importorskip when timm absent |

## Test Results

```
7 passed, 4 skipped in 32.82s (CPU-only, mini_pyramid fixture, stub backbone)
```

| Test | Result | Notes |
|------|--------|-------|
| TestNLLFormula::test_nll_formula_correct | PASS | NLL = log(9)+log(3) within 1e-5 |
| TestNLLFormula::test_nll_ignores_invalid_pixels | PASS | all-invalid pixels → nan |
| TestNLLFormula::test_eval_uses_only_leaf_tiles | PASS | 896/448 probs excluded |
| TestEvalLeakageGuard::test_load_test_dirs_only_test_paths | PASS | all returned paths have "test" in parts |
| TestEvalLeakageGuard::test_rejects_traversal_map_id | PASS | ValueError raised for ../secret |
| TestEvalAllBackbones::siglip-A | PASS | finite float |
| TestEvalAllBackbones::siglip-B | PASS | finite float |
| TestEvalAllBackbones::dinov2-A | SKIP | timm not installed |
| TestEvalAllBackbones::dinov2-B | SKIP | timm not installed |
| TestEvalAllBackbones::swin-A | SKIP | timm not installed |
| TestEvalAllBackbones::swin-B | SKIP | timm not installed |

## Security / Threat Model Compliance

| Threat | Status |
|--------|--------|
| T-04-07: split.json map ID path traversal | Mitigated — traversal guard asserts `test_root` (not just `data_root`) is ancestor; regression test confirms `../secret` raises ValueError |
| T-04-08: EVAL-01 zero-leakage | Mitigated — only `split.json["test"]` read; only `data_root/test/...` paths constructed; never touches train/ |
| T-04-09: log(0) = -inf from underflowed probs | Mitigated — `prob.clamp_min(1e-12)` before `torch.log`; formula tested with exact value anchor |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `gather` crash on out-of-range ground truth labels**
- **Found during:** Task 2, test_nll_ignores_invalid_pixels
- **Issue:** `evaluate_joint_nll` called `lc_gt.unsqueeze(0)` as gather indices without clamping. When test labels contain value 9 (out of range for 9-class LC), `RuntimeError: index 9 is out of bounds for dimension 0 with size 9` was raised, even though the valid mask would have excluded those pixels.
- **Fix:** Clamp gather indices to valid range (`lc_gt.clamp(0, 8)`, `topo_gt.clamp(0, 2)`) before gather. The valid mask is then applied to the NLL values to exclude the clamped pixels from the sum — their clamped NLL values are never accumulated.
- **Files modified:** `scripts/evaluate_seg.py`
- **Commit:** 42c9a98

**2. [Rule 1 - Bug] Path traversal guard checked `data_root` instead of `data_root/test/`**
- **Found during:** Task 2, test_rejects_traversal_map_id
- **Issue:** The traversal guard in `load_test_pyramid_dirs` verified the resolved path was under `data_root`, but the map ID `"../secret"` resolves `data_root/test/../secret` = `data_root/secret` which IS under `data_root`. A map ID like `"../train/secret"` would also silently pass, violating T-04-07 and EVAL-01.
- **Fix:** Changed guard to check `test_root_resolved` (`data_root/test/`) as the required ancestor — any map ID traversing above `data_root/test/` is rejected.
- **Files modified:** `scripts/evaluate_seg.py`
- **Commit:** 42c9a98

**3. [Rule 1 - Bug] `load_test_pyramid_dirs` looked only one level deep for pyramid.json**
- **Found during:** Task 2, test_load_test_dirs_only_test_paths
- **Issue:** The implementation looked for dirs at `candidate/*.pyramid.json` but the tiling producer creates `data_root/test/<map_id>/pyramids/py_r*/pyramid.json` (two levels deeper).
- **Fix:** Changed to `candidate.rglob("pyramid.json")` to discover pyramid dirs at any depth under the candidate.
- **Files modified:** `scripts/evaluate_seg.py`
- **Commit:** 42c9a98

**4. [Rule 2 - Missing functionality] dinov2/swin tests need `pytest.importorskip("timm")`**
- **Found during:** Task 2, TestEvalAllBackbones[dinov2-A]
- **Issue:** `timm` is not installed in this environment. DINOv2 and Swin backbone construction requires timm; tests failed with `ModuleNotFoundError` (not a clean skip).
- **Fix:** Added `pytest.importorskip("timm")` for backbone in `("dinov2", "swin")` so these tests skip cleanly when timm is absent. SigLIP uses `stub_config` and runs offline without timm.
- **Files modified:** `tests/test_seg_eval.py`
- **Commit:** 42c9a98

## Known Stubs

None. All implemented functions are fully functional. The test_nll_formula_correct test anchors the formula to a specific value (log(9)+log(3)), and the leakage/traversal guards are fully wired.

## Threat Flags

None. All surface introduced by this plan (split.json enumeration, NLL computation) was in the pre-existing threat model and is mitigated per T-04-07, T-04-08, T-04-09.

## Self-Check: PASSED

- scripts/evaluate_seg.py: FOUND (400 lines, >110 min_lines)
- tests/test_seg_eval.py: FOUND
- 04-03-SUMMARY.md: FOUND
- Commit b25e1f0: FOUND (Task 1 — evaluate_seg.py implementation)
- Commit 42c9a98: FOUND (Task 2 — test_seg_eval.py + evaluate_seg bug fixes)
- All 4 exports (evaluate_joint_nll, load_test_pyramid_dirs, write_nll_metrics, main): OK
