---
phase: 04-fine-tune-and-evaluate-the-segmentation-model-then-upload-to
plan: "02"
subsystem: seg/gcs_checkpoint + offline GCS test suite
tags:
  - gcs
  - checkpoint
  - preemption-safe
  - tdd
  - offline
  - traversal-guard
  - corrupt-skip
dependency_graph:
  requires:
    - torch (for serialisation)
    - seg package marker (scripts/seg/__init__.py)
  provides:
    - seg.gcs_checkpoint (gcs_save_checkpoint, gcs_latest_checkpoint, validate_config_name, config_prefix)
    - tests/test_seg_gcs.py (TestGCSCheckpointRoundTrip, TestCorruptCheckpointSkip, TestConfigNaming — 10 tests)
  affects:
    - scripts/finetune_seg.py (plan 04-04, consumes gcs_save_checkpoint / gcs_latest_checkpoint)
    - scripts/evaluate_seg.py (no direct dependency, but shares the config naming scheme)
tech_stack:
  added:
    - gcsfs (runtime-only, not installed on planning VM — module-level try/except import)
    - io.BytesIO atomic GCS object write (preemption-safe — Pitfall 4)
  patterns:
    - Module-level try/except gcsfs import: offline-importable + mock.patch-able
    - _MockGCSFileSystem: in-memory dict store with BytesIO-backed context manager
    - _MockGCSModule: module stub so mock.patch replaces gcsfs module attribute
    - Corrupt-skip fallback loop: descending step order, try/except per blob
key_files:
  created:
    - scripts/seg/gcs_checkpoint.py
    - (tests/test_seg_gcs.py body — scaffold created in 04-01)
  modified:
    - tests/test_seg_gcs.py (scaffold replaced with full suite)
decisions:
  - config_prefix lowercases both backbone and variant (e.g. config_prefix('SigLIP','B')=='siglip-b') so the traversal-safe ^[a-z0-9][a-z0-9\-]*$ regex always passes; plan-05 comparison reports must use the same lowercased form
  - gcsfs imported at module level via try/except (not inside each function) so mock.patch("seg.gcs_checkpoint.gcsfs", ...) works — functions reference the module-level name which is replaced by the patcher
  - Tests patch the entire gcsfs module reference with _MockGCSModule rather than patching gcsfs.GCSFileSystem directly (required because gcsfs=None when not installed; None has no GCSFileSystem attribute to patch)
metrics:
  duration: "8m"
  completed_date: "2026-05-16"
  tasks_completed: 2
  files_created: 1
  files_modified: 1
  commits: 2
---

# Phase 04 Plan 02: GCS Checkpoint Module + Offline Test Suite Summary

**One-liner:** Preemption-safe GCS checkpoint write/auto-resume with traversal-guard + corrupt-blob fallback, proven offline by a 10-test fully-mocked suite in 1.1s.

## What Was Built

### `scripts/seg/gcs_checkpoint.py` (new, 201 lines)

Four exported symbols implementing the D-07/D-09 checkpoint contract:

| Symbol | Description |
|--------|-------------|
| `GCS_PROJECT` | `"narrative-campaign"` — GCS project constant |
| `BUCKET_PREFIX` | `"gs://mapclass-training-northeast1/models"` — root prefix |
| `validate_config_name(name)` | Validates against `^[a-z0-9][a-z0-9\-]*$`; raises ValueError otherwise (T-04-04) |
| `config_prefix(backbone, variant)` | Lowercases both args, concatenates, validates; e.g. `config_prefix('SigLIP','B')=='siglip-b'` |
| `gcs_save_checkpoint(config_name, step, state)` | BytesIO atomic write to `gs://.../step_{step:07d}.pt`; returns path string |
| `gcs_latest_checkpoint(config_name)` | Lists prefix, picks highest step, corrupt-skip loop; returns `(step, dict)` or `(0, None)` |

**Key design decisions:**
- Module-level `try: import gcsfs except: gcsfs = None` — importable offline; patchable by tests
- BytesIO buffer ensures object-atomic GCS upload (no partial writes on preemption)
- Descending step sort + per-blob try/except covers truncated/corrupt blobs (T-04-05)
- `weights_only=False` documented with trust assumption comment (T-04-06 / D-08)

### `tests/test_seg_gcs.py` (scaffold replaced, 239 lines)

10 tests across 3 test classes:

| Class | Tests | Coverage |
|-------|-------|---------|
| `TestGCSCheckpointRoundTrip` | 3 | save+resume, empty prefix, highest-step selection |
| `TestCorruptCheckpointSkip` | 1 | corrupt blob at higher step skipped, valid lower returned |
| `TestConfigNaming` | 6 | 3 parametrized format cases + traversal + uppercase + leading-hyphen rejection |

All tests use `_MockGCSFileSystem` (in-memory dict store) patched via `_MockGCSModule` stub.
Runtime: 1.1s (limit: 30s). Fully offline — no gcsfs, no network.

## Test Results

```
10 passed in 1.12s
```

| Test | Result |
|------|--------|
| TestGCSCheckpointRoundTrip::test_save_and_resume | PASS |
| TestGCSCheckpointRoundTrip::test_resume_returns_zero_on_empty_prefix | PASS |
| TestGCSCheckpointRoundTrip::test_resume_picks_highest_step | PASS |
| TestCorruptCheckpointSkip::test_corrupt_latest_skipped | PASS |
| TestConfigNaming::test_config_name_format[siglip-B-siglip-b] | PASS |
| TestConfigNaming::test_config_name_format[dinov2-A-dinov2-a] | PASS |
| TestConfigNaming::test_config_name_format[swin-B-swin-b] | PASS |
| TestConfigNaming::test_rejects_traversal | PASS |
| TestConfigNaming::test_rejects_uppercase | PASS |
| TestConfigNaming::test_rejects_leading_hyphen | PASS |

## Security / Threat Model Compliance

| Threat | Status |
|--------|--------|
| T-04-04: Config-name path traversal | Mitigated: `validate_config_name` enforces `^[a-z0-9][a-z0-9\-]*$`; regression tests cover `../../x`, `a/b`, uppercase, leading hyphen |
| T-04-05: Corrupt/truncated GCS blob DoS | Mitigated: descending step loop with per-blob try/except (UnpicklingError, EOFError, RuntimeError, Exception); `test_corrupt_latest_skipped` regression test |
| T-04-06: `weights_only=False` deserialization | Accepted: own-produced bucket artifacts only; trust assumption documented in source comment |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Mock patch target required module-level gcsfs attribute**
- **Found during:** Task 2 GREEN phase (first test run)
- **Issue:** Plan specified `mock.patch("seg.gcs_checkpoint.gcsfs.GCSFileSystem", _MockGCSFileSystem)`. The initial implementation imported gcsfs lazily inside each function body, so `seg.gcs_checkpoint` had no `gcsfs` attribute for mock to resolve — `AttributeError: module 'seg.gcs_checkpoint' has no attribute 'gcsfs'`.
- **Fix (attempt 1):** Added module-level `try: import gcsfs except: gcsfs = None`. This created the attribute, but then `mock.patch("seg.gcs_checkpoint.gcsfs.GCSFileSystem", ...)` failed with `AttributeError: None does not have the attribute 'GCSFileSystem'` (since gcsfs is None on the planning VM).
- **Fix (attempt 2):** Changed mock strategy: patch the entire `gcsfs` module reference with `_MockGCSModule` (a stub class with `GCSFileSystem = _MockGCSFileSystem`). This works correctly whether gcsfs is installed or not.
- **Files modified:** `scripts/seg/gcs_checkpoint.py`, `tests/test_seg_gcs.py`
- **Commit:** 41a258b

## Known Stubs

None. All implemented functions are fully functional. Real GCS I/O is deferred to the GPU-host gate (plan 04-04) as intended by the plan.

## Threat Flags

None. No new network endpoints, auth paths, file access patterns, or schema changes beyond what the plan's threat model covers.

## Self-Check: PASSED
