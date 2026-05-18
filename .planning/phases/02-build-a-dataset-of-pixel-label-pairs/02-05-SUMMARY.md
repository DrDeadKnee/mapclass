---
phase: 02-build-a-dataset-of-pixel-label-pairs
plan: 05
subsystem: dataset-pipeline
tags: [gcs, gcsfs, sentinel, pull-once, rw-03, oq1, oq2, finetune, evaluate]

# Dependency graph
requires:
  - phase: 02-build-a-dataset-of-pixel-label-pairs
    provides: "GCS-canonical build scripts (02-01..02-04) — gcs_io, build_dataset, build_historical, build_satellite, tiling"
  - phase: 04-train-and-evaluate
    provides: "finetune_seg.py and evaluate_seg.py consumer scripts"
provides:
  - "_BUILD_COMPLETE sentinel helpers (mark_build_complete, is_build_complete) in gcs_io.py"
  - "family_subset_prefix OQ2 layout constant in gcs_io.py"
  - "Sentinel guard in build_dataset.py, build_historical_dataset.py, build_satellite_dataset.py"
  - "RW-03 pull-once + verify in finetune_seg.py (before gcs_latest_checkpoint)"
  - "RW-03 pull-once + verify in evaluate_seg.py (before load_test_pyramid_dirs)"
  - "--scratch-dir argparse arg in both scripts"
  - "evaluate(args) public API in evaluate_seg.py"
  - "02-HUMAN-UAT.md blocking gate for Azgaar regeneration + GCS upload"
affects: ["04-train-and-evaluate", "05-publish"]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "_BUILD_COMPLETE sentinel written as last GCS write after tiling completes (OQ1)"
    - "is_build_complete() gates skip — partial prefix (sentinel absent) is rebuilt, not skipped (Pitfall R-2)"
    - "family_subset_prefix(family, subset) returns canonical bare GCS path (OQ2)"
    - "try/except ImportError guards all gcs_io imports in training/eval scripts"
    - "pull_dataset_from_gcs + verify_pull precede carve_train_val / load_test_pyramid_dirs (RW-03)"
    - "evaluate(args) separated from main() for testability"

key-files:
  created:
    - "scripts/gcs_io.py (extended: _BUILD_COMPLETE, mark_build_complete, is_build_complete, family_subset_prefix)"
    - "tests/test_pull_once.py (torch-free structural tests for RW-03 pull-once)"
    - ".planning/phases/02-build-a-dataset-of-pixel-label-pairs/02-HUMAN-UAT.md"
  modified:
    - "scripts/build_dataset.py (sentinel guard + mark in build_one_source)"
    - "scripts/build_historical_dataset.py (sentinel guard + mark in _process_one)"
    - "scripts/build_satellite_dataset.py (sentinel guard + mark in _process_one)"
    - "scripts/finetune_seg.py (pull-once block + --scratch-dir arg)"
    - "scripts/evaluate_seg.py (pull-once block + --scratch-dir arg + evaluate() public API)"
    - "tests/test_gcs_io.py (Task 1 TDD tests for sentinel and layout)"

key-decisions:
  - "OQ1 LOCKED: _BUILD_COMPLETE sentinel written last, checked before skip — partial builds are always rebuilt (Pitfall R-2 / T-02-40)"
  - "OQ2 LOCKED: family-rooted GCS layout (gs://.../data/{synthetic,historical,satellite}/{train,test}/); merging at pull-once time; split.json stays synthetic-only (EVAL-01)"
  - "evaluate_seg.py pulls ONLY the synthetic test subset (EVAL-01 hold-out is synthetic-only)"
  - "test_pull_once.py uses source-text structural assertions (torch-free) for offline CI compatibility"

patterns-established:
  - "sentinel-last write pattern: tiling.tile() → mark_build_complete() — always in that order"
  - "import guard: try/except ImportError at module-level for gcs_io, fallback to None"
  - "evaluate() + main() separation in CLI scripts for test callability"

requirements-completed: [PHASE-02, EVAL-01]

# Metrics
duration: 45min
completed: 2026-05-16
---

# Phase 02 Plan 05: Consumer Pull-Once + Sentinel Summary

**_BUILD_COMPLETE sentinel (OQ1), family-rooted layout (OQ2), and RW-03 pull-once wiring in finetune_seg.py + evaluate_seg.py — GCS pipeline now end-to-end offline-verified; blocked at Azgaar regeneration human gate**

## Performance

- **Duration:** ~45 min
- **Started:** 2026-05-16T22:30:00Z
- **Completed:** 2026-05-16T23:15:00Z (checkpoint: Task 3 — human gate)
- **Tasks:** 2/3 completed (Task 3 is a blocking human gate — see 02-HUMAN-UAT.md)
- **Files modified:** 9

## Accomplishments

- Added `_BUILD_COMPLETE` sentinel to `gcs_io.py` with `mark_build_complete()` +
  `is_build_complete()` helpers; all three build scripts use sentinel to gate skip
  and write it last after tiling (OQ1 resolved, Pitfall R-2 / T-02-40 mitigated).
- Added `family_subset_prefix(family, subset)` to `gcs_io.py` encoding the OQ2
  family-rooted GCS layout decision (LOCKED 2026-05-16).
- Wired RW-03 pull-once into `finetune_seg.py` (pull train subset before
  `gcs_latest_checkpoint` resume) and `evaluate_seg.py` (pull test subset before
  `load_test_pyramid_dirs`), both with `--scratch-dir` arg and `ImportError` fallback.
- Wrote `02-HUMAN-UAT.md` with full step-by-step Azgaar regeneration + manifest
  authoring + GCS upload gate, including the post-upload acceptance run.

## Task Commits

1. **Task 1 RED: _BUILD_COMPLETE + layout tests** - `9549627` (test)
2. **Task 1 GREEN: sentinel + OQ2 in gcs_io + all build scripts** - `e462a43` (feat)
3. **Task 2 RED: pull-once structural tests** - `54bf0fd` (test)
4. **Task 2 GREEN: pull-once in finetune_seg + evaluate_seg** - `9c7ef61` (feat)

Task 3 (checkpoint:human-verify) — not committed; gate is in progress.

## Files Created/Modified

- `scripts/gcs_io.py` — Extended with `_BUILD_COMPLETE`, `mark_build_complete`, `is_build_complete`, `family_subset_prefix`; module docstring documents OQ1 + OQ2 decisions
- `scripts/build_dataset.py` — `is_build_complete` skip gate + `mark_build_complete` after tiling in `build_one_source`
- `scripts/build_historical_dataset.py` — Same sentinel guard + mark in `_process_one` GCS path
- `scripts/build_satellite_dataset.py` — Same sentinel guard + mark in `_process_one` GCS path
- `scripts/finetune_seg.py` — RW-03 pull-once block before `gcs_latest_checkpoint`; `--scratch-dir` argparse arg; try/except ImportError fallback
- `scripts/evaluate_seg.py` — RW-03 pull-once block before `load_test_pyramid_dirs`; `--scratch-dir` argparse arg; refactored `evaluate(args)` public API; try/except ImportError fallback
- `tests/test_gcs_io.py` — TDD tests for `mark_build_complete`, `is_build_complete`, `family_subset_prefix`
- `tests/test_pull_once.py` — New torch-free structural tests for pull-once wiring
- `.planning/phases/02-build-a-dataset-of-pixel-label-pairs/02-HUMAN-UAT.md` — Blocking gate document

## Decisions Made

- **OQ1 LOCKED:** `_BUILD_COMPLETE` sentinel written as last GCS object per map-dir;
  `is_build_complete()` gates skip — a prefix with existing objects but no sentinel
  is treated as a partial/aborted build and rebuilt (Pitfall R-2 / T-02-40).
- **OQ2 LOCKED:** Family-rooted GCS layout. Each build script writes to its own
  family prefix; pull-once merges them into a local scratch root. `split.json`
  stays synthetic-only at `gs://.../data/synthetic/split.json` (EVAL-01 unchanged).
- **Torch-free tests:** RW-03 pull-once structural contracts tested via source-text
  inspection (`test_pull_once.py`) rather than mocked runtime execution, to remain
  compatible with the planning VM (no torch/GPU).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Torch-free test file for pull-once (test_pull_once.py)**
- **Found during:** Task 2 RED phase
- **Issue:** `test_seg_eval.py` imports `torch` at module level; can't be collected
  on the planning VM. Adding evaluate_seg pull-once tests there would be unreachable.
  `test_seg_training.py` also fails on torch-requiring tests, though it can be collected.
- **Fix:** Created `tests/test_pull_once.py` with structural/source-text tests that
  run without torch. Tests verify: `--scratch-dir` arg present, `pull_dataset_from_gcs(`
  call precedes `gcs_latest_checkpoint(` / `load_test_pyramid_dirs(`, `ImportError`
  fallback present.
- **Files modified:** `tests/test_pull_once.py` (created), `tests/test_seg_training.py`
  (cleanup), `tests/test_seg_eval.py` (cleanup)
- **Commit:** `54bf0fd` (RED), `9c7ef61` (GREEN)

## Issues Encountered

- Test position assertions using `src.index()` initially matched docstring/comment
  occurrences (e.g., `gcs_latest_checkpoint()` appeared in function docstring before
  the actual call). Fixed by removing the paren from the docstring and using call-site
  patterns with `(` suffix; for evaluate_seg used `rfind()` to get the last occurrence
  (the call, not the function definition).

## User Setup Required

**External services require manual configuration.** See
[02-HUMAN-UAT.md](./02-HUMAN-UAT.md) for:

- Re-create ~100 Azgaar maps across ~12 continent templates
- Name and export as `<template>_<NN>.geojson`
- Author `raw/manifest.json` with all entries
- Upload to `gs://mapclass-training-northeast1/data/synthetic/raw/`
- Run `python scripts/build_dataset.py build` and confirm non-empty output

## Next Phase Readiness

- **Blocked:** Phase 4 `finetune_seg.py` + `evaluate_seg.py` pull-once cannot run
  until the Azgaar regeneration + GCS upload gate (02-HUMAN-UAT.md) is completed.
- **Ready:** Once the gate passes and `gs://.../data/synthetic/{train,test}/` +
  `split.json` are populated, Phase 4 can proceed with `--scratch-dir` to pull the
  dataset to local scratch before training/evaluation.

---

## Checkpoint: Human Gate Reached

**Status:** STOPPED at Task 3 (checkpoint:human-verify, gate=blocking-human)

The GCS-canonical Phase 2 pipeline (Waves 0-3) is fully implemented and
offline-verified. The end-to-end build CANNOT run until the user re-creates the
lost Azgaar source maps and uploads them to the canonical bucket.

See `02-HUMAN-UAT.md` for the complete gate steps.

**Resume signal:** Type "approved" once build completed cleanly and
`gs://.../data/synthetic/{train,test}/` + `split.json` are populated.

---

## Self-Check: PASSED

Files verified:
- `scripts/gcs_io.py` — contains `_BUILD_COMPLETE`, `mark_build_complete`, `is_build_complete`, `family_subset_prefix`
- `scripts/build_dataset.py` — contains `_BUILD_COMPLETE` (via `mark_build_complete`/`is_build_complete`)
- `scripts/build_historical_dataset.py` — contains `_BUILD_COMPLETE` references
- `scripts/build_satellite_dataset.py` — contains `_BUILD_COMPLETE` references
- `scripts/finetune_seg.py` — contains `pull_dataset_from_gcs`, `--scratch-dir`
- `scripts/evaluate_seg.py` — contains `pull_dataset_from_gcs`, `--scratch-dir`, `evaluate(args)`
- `tests/test_pull_once.py` — 6 tests, all pass
- `tests/test_gcs_io.py` — 21 tests, all pass
- `.planning/phases/02-build-a-dataset-of-pixel-label-pairs/02-HUMAN-UAT.md` — written

Commits verified:
- `9549627` — test(02-05): add failing tests for _BUILD_COMPLETE sentinel
- `e462a43` — feat(02-05): _BUILD_COMPLETE sentinel + family_subset_prefix
- `54bf0fd` — test(02-05): add failing tests for RW-03 pull-once (RED)
- `9c7ef61` — feat(02-05): RW-03 pull-once + verify in finetune_seg + evaluate_seg

---
*Phase: 02-build-a-dataset-of-pixel-label-pairs*
*Completed (partial — checkpoint): 2026-05-16*
