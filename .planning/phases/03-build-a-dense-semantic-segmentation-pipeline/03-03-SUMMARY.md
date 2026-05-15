---
phase: 03-build-a-dense-semantic-segmentation-pipeline
plan: 03
subsystem: dataset
tags: [pytorch, torch.utils.data, pillow, numpy, pyramid, segmentation, dataloader]

# Dependency graph
requires:
  - phase: 02-build-a-dataset-of-pixel-label-pairs
    provides: "Phase-2 nested-pyramid tile tree: pyramid.json, image/land_cover/topography PNGs, sample_weights.json per pyramid"

provides:
  - "PyramidDataset class in scripts/seg/dataset.py — torch.utils.data.Dataset over Phase-2 nested-pyramid directories"
  - "tile_path(i) accessor for split-safety testing"
  - "sample_weights.json surfaced byte-identical as raw dict in each sample"
  - "EVAL-01 runtime guard: ValueError on any indexed path containing 'test/' component"

affects:
  - 03-build-a-dense-semantic-segmentation-pipeline
  - 04-training (Phase-4 DataLoader wraps PyramidDataset)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Manifest-driven indexing (D-04): tile paths always come from pyramid.json keys, never re-derived"
    - "ToonDataset idiom for ValueError on empty index (finetune_paligemma.py:93-94)"
    - "EVAL-01 runtime guard pattern: constructor-time path component check mirroring test_tiling.py:213-215"
    - "numpy.array() (not np.asarray()) for writable tensors to avoid torch non-writable warning"

key-files:
  created:
    - scripts/seg/dataset.py
  modified: []

key-decisions:
  - "PyramidDataset accepts either a single pyramid dir (has pyramid.json) or a parent dir of pyramid dirs — single-call convenience without glob risk"
  - "sample_weights surfaced as raw dict (not tensor/string); test accepts dict or JSON string — Phase-4 owns the weighting strategy"
  - "EVAL-01 guard is both a constructor-time assertion AND a design constraint: caller must point at train/ subtree only"
  - "tile_path(i) public method added to support the split-safety test contract defined in test_seg_dataset.py"

patterns-established:
  - "Manifest-driven dataset: PyramidDataset is the canonical consumer of tiling.py's pyramid.json contract"
  - "Constructor ValueError on empty dataset: project-wide idiom from ToonDataset"
  - "Runtime leakage guard at dataset construction: assert 'test' not in path.parts for every indexed sample"

requirements-completed: [PHASE-03]

# Metrics
duration: 15min
completed: 2026-05-15
---

# Phase 3 Plan 03: PyramidDataset (D-06) Summary

**torch.utils.data.Dataset over Phase-2 nested-pyramid tiles — reads pyramid.json manifests, surfaces sample_weights.json byte-unchanged, enforces EVAL-01 train/-only split-safety at constructor time**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-05-15T23:00:00Z
- **Completed:** 2026-05-15T23:09:07Z
- **Tasks:** 1 (1 auto TDD task)
- **Files modified:** 1 (created)

## Accomplishments

- Implemented `PyramidDataset` in `scripts/seg/dataset.py` — reads Phase-2 pyramid directories via `pyramid.json`, indexes all 21 tiles per pyramid, loads image/land_cover/topography PNGs as typed tensors
- `sample_weights.json` surfaced byte-identical (raw dict, no re-normalisation) — Phase-4 loss plumbing consumes it unchanged
- EVAL-01 zero-leakage enforced: constructor raises `ValueError` if any indexed path contains a `test/` component; never uses `glob("**/pyramid.json")` across a synthetic root (Pitfall 6 guard)
- All 8 `test_seg_dataset.py` tests pass; 72 offline tests pass; no regressions

## Task Commits

1. **Task 1: PyramidDataset (Phase-2 tree reader, train/-only, weight surfacing)** - `0ffa2ce` (feat)

**Plan metadata:** (committed as part of SUMMARY commit)

## Files Created/Modified

- `scripts/seg/dataset.py` — PyramidDataset class: `__init__`, `__len__`, `__getitem__`, `tile_path(i)` with EVAL-01 guard and manifest-driven indexing

## Decisions Made

- **Single-dir or parent-dir input:** `PyramidDataset(root)` auto-detects whether `root` is a single pyramid dir (contains `pyramid.json` directly) or a parent dir of pyramid dirs. Avoids requiring callers to enumerate dirs manually, and avoids the forbidden `glob("**/pyramid.json")` pattern.
- **`tile_path(i)` public method:** Added as a first-class API method (not a private attribute) because `test_seg_dataset.py` calls it directly to assert the split-safety invariant.
- **`numpy.array()` over `numpy.asarray()`:** `asarray` returns a read-only view for certain PIL internal buffers; `array` always returns a writable copy, eliminating the `torch.from_numpy` UserWarning about non-writable tensors.

## Deviations from Plan

None - plan executed exactly as written.

The implementation follows the ToonDataset skeleton (finetune_paligemma.py:67-156), the manifest contract from tiling.py, and the EVAL-01 leakage guard pattern from test_tiling.py:213-215.

## Issues Encountered

None. `torch` reported `No module named 'torch'` on the first invocation but imported correctly on retry — transient environment state, not a structural issue.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `PyramidDataset` is ready for Phase-4 training loops to wrap with `torch.utils.data.DataLoader`
- `sample_weights` dict is surfaced per sample; Phase-4 loss function should read `entry["sample_weights"]` directly
- Split-safety guarantee is runtime-enforced; Phase-4 need only pass `train/` subtree pyramid dirs
- No blockers; this plan ran parallel to 03-02 (backbones) with no file conflicts

## Known Stubs

None - all data paths wired from real Phase-2 pyramid.json manifests.

---
*Phase: 03-build-a-dense-semantic-segmentation-pipeline*
*Completed: 2026-05-15*
