---
phase: 02-build-a-dataset-of-pixel-label-pairs
plan: "01"
subsystem: gcs-io
tags: [gcs, fsspec, testing, wave-0, dataset-pipeline]
dependency_graph:
  requires: []
  provides:
    - scripts/gcs_io._GCSWriter
    - scripts/gcs_io.validate_manifest
    - scripts/gcs_io.pull_dataset_from_gcs
    - scripts/gcs_io.verify_pull
    - tests/conftest.local_fs
    - tests/conftest.mock_gcs_module
  affects:
    - scripts/build_dataset.py (imports _GCSWriter, validate_manifest)
    - scripts/tiling.py (consumes _GCSWriter as out_root)
    - scripts/finetune_seg.py (imports pull_dataset_from_gcs, verify_pull)
    - scripts/evaluate_seg.py (imports pull_dataset_from_gcs, verify_pull)
tech_stack:
  added: []
  patterns:
    - lazy-gcsfs-import (try/except ModuleNotFoundError mirrored from gcs_checkpoint.py)
    - _GCSWriter shim (thin fsspec-backed writer; one instance per pyramid in pool)
    - validate_manifest hard-fail (sys.exit(1) only; 4-way cross-check; RW-02)
    - pull-once + spot-check verification (RW-03)
    - conftest mock_gcs_module (in-memory _store; mirrors test_seg_gcs._MockGCSModule)
key_files:
  created:
    - scripts/gcs_io.py
    - tests/test_gcs_io.py
    - tests/test_manifest.py
  modified:
    - tests/conftest.py
decisions:
  - "_GCSWriter accepts any fsspec AbstractFileSystem (not just GCSFileSystem) — enables local-fs offline tests without mock patching (cleaner than the test_seg_gcs.py approach)"
  - "validate_manifest replicates _sanitize_stem from build_dataset (not imported) to avoid circular dependency"
  - "verify_pull uses raise RuntimeError (not sys.exit) so callers can handle failures; contrast with validate_manifest which sys.exit(1) (RW-02 vs RW-03 distinction)"
  - "concurrent test calls root.mkdir() before writing — mirrors actual tiling.py call order where pdir.mkdir() precedes tile writes"
metrics:
  duration: "7m 15s"
  completed: "2026-05-16T21:37:06Z"
  tasks_completed: 2
  files_created: 3
  files_modified: 1
---

# Phase 2 Plan 01: GCS I/O Foundation (Wave 0) Summary

**One-liner:** Lazy-import gcsfs shim (_GCSWriter), manifest hard-fail (sys.exit(1)), pull-once helpers (pull_dataset_from_gcs + verify_pull), and full offline Wave 0 test suite (23 tests green).

## What Was Built

### scripts/gcs_io.py (NEW)

The single GCS I/O abstraction shared by all five Phase-2 plans.

- **`_GCSWriter`**: thin write-only shim over any `fsspec.AbstractFileSystem`. Accepts `fs` injected by caller (never instantiates `GCSFileSystem` internally). Methods: `__truediv__`, `mkdir`, `name`, `write_bytes`, `write_text`, `open`. Offline-testable via `fsspec.filesystem("file")`.
- **`validate_manifest(gcs_filenames, manifest) -> dict[str, str]`**: 4-way cross-check (regex, unlisted, phantom, template mismatch). Hard-fails via `sys.exit(1)` — no warnings, no `ValueError`. Returns `{sanitized_id: template}` on clean input. The only gate before `split.json` is frozen (RW-02 + T-02-02/03).
- **`pull_dataset_from_gcs(subset, local_scratch, gcs_prefix) -> Path`**: idempotent bulk-fetch via `gcsfs.GCSFileSystem.get(..., recursive=True)`. Safe after preemption.
- **`verify_pull(local_root, split_json, subset) -> None`**: membership check + 10% pyramid spot-check (>= 60 PNGs + pyramid.json). Raises `RuntimeError` (not `sys.exit`) so callers handle.
- **Constants**: `GCS_PROJECT = "narrative-campaign"` (verbatim from gcs_checkpoint.py), `BUCKET = "mapclass-training-northeast1"`, `DATA_PREFIX = "mapclass-training-northeast1/data"`.
- **Lazy import**: `try: import gcsfs / except ModuleNotFoundError: gcsfs = None` — module imports offline.

### tests/test_gcs_io.py (NEW)

23 tests covering the full Wave 0 behavior:
- `_GCSWriter`: write_bytes, write_text, truediv prefix/name, Pillow compat (PNG round-trip), mkdir no-error, offline importable
- `pull_dataset_from_gcs`: idempotency (second call overwrites, no raise)
- `verify_pull`: raises on missing map dir, raises on truncated pyramid (< 60 PNGs), passes on complete pyramid
- `test_concurrent_tile_writes`: 32-thread pool writing 64 distinct tiles via `_GCSWriter` to local fs — zero key loss

### tests/test_manifest.py (NEW)

8 tests covering `validate_manifest`:
- Hard-fail on regex mismatch (uppercase filename)
- Hard-fail on unlisted file
- Hard-fail on phantom manifest entry (in manifest, missing from GCS list)
- Hard-fail on template mismatch
- Hard-fail raises `SystemExit` (not `ValueError`)
- Clean pass returns `{id: template}` mapping
- All filenames present as keys
- `east_asia_01.geojson` sanitized correctly

### tests/conftest.py (MODIFIED)

Added two shared fixtures without touching existing fixtures:
- **`local_fs`**: `fsspec.filesystem("file")` for offline `_GCSWriter` tests
- **`mock_gcs_module`**: module stub exposing `_MockGCSFileSystem` (in-memory `_store: dict[str, bytes]`) with `open`, `ls`, `pipe_file`, `cat`, `exists`, `get`, `mkdirs`. Resets store at fixture setup.

## TDD Gate Compliance

| Gate | Commit | Status |
|------|--------|--------|
| RED — failing tests | `0d4bb1b` | PASS — gcs_io not yet implemented; tests fail loudly |
| GREEN — implementation | `757758c` + `8f74ad6` | PASS — 23/23 tests green |
| REFACTOR | N/A | No refactor needed |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] concurrent_tile_writes needed mkdir() before writes**
- **Found during:** Task 2 GREEN verification
- **Issue:** `_GCSWriter.write_bytes` via local fs `pipe_file` fails if parent directory doesn't exist. The test wrote tiles directly to `root / f"tile_{i}.png"` without creating `root` first.
- **Fix:** Added `root.mkdir()` call before the concurrent writes in the test — mirrors the actual `tiling.py` call order where `pdir.mkdir()` precedes tile writes.
- **Files modified:** `tests/test_gcs_io.py`
- **Commit:** `8f74ad6`

## Verification Results

All plan acceptance criteria satisfied:

| Criterion | Result |
|-----------|--------|
| `gcs_io` importable with `gcsfs=None` | PASS |
| `_GCSWriter` has all required methods | PASS |
| `validate_manifest` body has `sys.exit(1)`, no `raise ValueError`, no `WARNING` | PASS |
| `GCS_PROJECT == "narrative-campaign"` | PASS |
| `BUCKET == "mapclass-training-northeast1"` | PASS |
| `DATA_PREFIX == "mapclass-training-northeast1/data"` | PASS |
| `pytest tests/test_gcs_io.py -k "writer or importable" -x` exits 0 | PASS |
| `pytest tests/test_manifest.py::TestValidateManifestHardFails -x` exits 0 | PASS |
| `pull_dataset_from_gcs` and `verify_pull` in gcs_io.py | PASS |
| `verify_pull` body has `raise RuntimeError`, no `sys.exit` | PASS |
| `conftest.py` exposes `local_fs` and `mock_gcs_module` | PASS |
| `pytest tests/test_gcs_io.py tests/test_manifest.py -x` exits 0 (23 passed) | PASS |
| Collection: 141 tests collected, 0 errors (excl. pre-existing torch-dep tests) | PASS |

**Pre-existing collection errors (not caused by this plan):**
- `tests/test_seg_eval.py` — imports torch (not installed in this env)
- `tests/test_seg_gcs.py` — imports torch (not installed in this env)

These errors exist on the base commit (`3729f1f`) before this plan's changes.

## Threat Model Compliance

| Threat ID | Status |
|-----------|--------|
| T-02-01 (Spoofing — ADC only, no key literals) | MITIGATED — lazy-gcsfs import, no token/key in source; ADC-only |
| T-02-02 (Tampering — filename injection) | MITIGATED — `_FILENAME_RE.fullmatch` + 4-way cross-check in `validate_manifest` |
| T-02-03 (Repudiation — hard-fail bypass) | MITIGATED — `sys.exit(1)` only; no warn-and-continue path; covered by `TestValidateManifestHardFails` |
| T-02-SC (Tampering — pip installs) | ACCEPTED — no new packages installed |

## Known Stubs

None. All methods are fully implemented with real behavior; no placeholders or TODO markers.

## Threat Flags

None. No new network endpoints, auth paths, or trust boundaries introduced beyond those explicitly modeled in the plan's `<threat_model>`.

## Self-Check

### Checking created files exist:
- scripts/gcs_io.py: FOUND
- tests/test_gcs_io.py: FOUND
- tests/test_manifest.py: FOUND
- tests/conftest.py (modified): FOUND

### Checking commits exist:
- 0d4bb1b (RED): test(02-01): add failing tests
- 757758c (GREEN Task 1): feat(02-01): implement scripts/gcs_io.py
- 8f74ad6 (GREEN Task 2): feat(02-01): add conftest GCS fixtures
