---
phase: "02-build-a-dataset-of-pixel-label-pairs"
plan: "04"
subsystem: "satellite-pipeline"
tags: ["gcs", "satellite", "dataset", "rw-04", "tdd"]
dependency_graph:
  requires: ["02-01", "02-02"]
  provides: ["GCS-canonical satellite dataset (RW-04)", "build_satellite_dataset.py _GCSWriter integration"]
  affects: ["scripts/build_satellite_dataset.py", "tests/test_satellite.py"]
tech_stack:
  added: []
  patterns: ["lazy-gcsfs-import", "_GCSWriter per-pyramid", "fs.pipe_file atomic write", "shared-fs-through-ThreadPoolExecutor"]
key_files:
  created: []
  modified:
    - "scripts/build_satellite_dataset.py"
    - "tests/test_satellite.py"
decisions:
  - "Coverage summary temp-file pattern: coverage.build_summary requires a local Path; write to tempfile then pipe to GCS (avoids modifying coverage.py)"
  - "cmd_search_regions accepts optional fs parameter for testability (avoids creating fs if test already patched gcsfs)"
  - "_process_one uses TemporaryDirectory for local scratch; local GeoTIFF (reprojected window) is ephemeral — only pyramid writes go to GCS"
  - "--local-ok flag enforces Pitfall R-1 guard (T-02-30): non-gs:// out-dir hard-errors without flag"
metrics:
  duration: "4m"
  completed_date: "2026-05-16T22:02:23Z"
  tasks_completed: 1
  files_changed: 2
---

# Phase 02 Plan 04: Satellite GCS-Canonical Pipeline Summary

**One-liner:** Satellite build pipeline writes coverage_summary.json, resolved_scenes.json, and dataset pyramids to GCS via _GCSWriter with shared thread-safe fs through ThreadPoolExecutor.

## What Was Built

`scripts/build_satellite_dataset.py` now implements RW-04 (GCS-canonical satellite persistence):

1. **GCS defaults:** `_DEFAULT_SUMMARY`, `_DEFAULT_MANIFEST`, and `_DEFAULT_OUT` all changed from local `Path(...)` to `gs://mapclass-training-northeast1/data/satellite/...` URI strings.

2. **Lazy gcsfs import:** mirrors `gcs_checkpoint.py` pattern — module importable on planning VM where gcsfs is absent; tests patch `build_satellite_dataset.gcsfs`.

3. **Coverage summary (cmd_coverage_scan):** `coverage.build_summary` writes to a local tempfile (it requires a `Path`); after it returns, the JSON is piped atomically to GCS via `fs.pipe_file`. `coverage.py` is unmodified.

4. **Resolved manifest (cmd_search_regions):** `manifest_path.write_text(...)` replaced with `fs.pipe_file(gcs_dest, json_bytes)` for GCS paths; local path fallback preserved.

5. **Dataset write (_process_one / cmd_build):** `tiling.tile(sample_dir)` call now passes `out_root=_GCSWriter(fs, f"{gcs_prefix}/{safe}/pyramids")` for GCS paths. A fresh per-pyramid `_GCSWriter` is constructed inside the worker (not shared); the shared `gcsfs.GCSFileSystem` is passed from `cmd_build` through `_process_one` to the worker (thread-safe).

6. **Raw COG reads unchanged:** Sentinel-2/WorldCover/DEM reads via GDAL VSI-CURL stay network→in-memory. Only the reprojected window GeoTIFF lands in local scratch inside `TemporaryDirectory`.

7. **--local-ok flag:** `build` subparser adds `--local-ok`; non-gs:// `--out-dir` without it calls `sys.exit(1)` (Pitfall R-1 / T-02-30 mitigation).

## TDD Gate Compliance

| Gate | Commit | Description |
|------|--------|-------------|
| RED | f039a82 | test(02-04): add failing GCS-persistence tests for satellite pipeline |
| GREEN | 574f4bd | feat(02-04): GCS-canonical satellite pipeline (RW-04) |

4 new tests added:
- `test_coverage_summary_written_to_gcs`: verifies `coverage_summary.json` piped to GCS mock store
- `test_resolved_scenes_written_to_gcs`: verifies `resolved_scenes.json` written to GCS not local path
- `test_satellite_dataset_written_to_gcs`: verifies `_DEFAULT_OUT` is a gs:// URI
- `test_cog_reads_stay_in_memory`: verifies all three defaults start with `gs://`

Full suite: **13/13 tests passing**.

## Verification

```
pytest tests/test_satellite.py -x          → 13 passed
grep -n "_GCSWriter" scripts/build_satellite_dataset.py   → non-empty (line 50, 311)
grep -n "satellite/resolved_scenes.json" scripts/build_satellite_dataset.py  → non-empty (line 73)
```

## Deviations from Plan

### Auto-additions (Rule 2 — missing critical functionality)

**1. [Rule 2 - Security] `_is_gcs_path` helper + local fallback for all write paths**
- **Found during:** Implementation review
- **Issue:** `cmd_search_regions` and `cmd_build` needed to handle both GCS and local paths (tests pass `tmp_path / "resolved_scenes.json"` as `manifest_path`)
- **Fix:** `_is_gcs_path(str)` helper routes to GCS or local write path; enables offline test coverage without requiring gcsfs
- **Files modified:** `scripts/build_satellite_dataset.py`

**2. [Rule 2 - Correctness] `cmd_search_regions` accepts optional `fs` parameter**
- **Found during:** Implementation — the test patches `gcsfs` globally but `cmd_search_regions` instantiated its own fs; an optional `fs=None` parameter allows callers to thread the already-patched fs through
- **Fix:** Added `fs=None` parameter to `cmd_search_regions` and `cmd_search`; `_make_fs()` called only when `fs is None`
- **Files modified:** `scripts/build_satellite_dataset.py`

**3. [Rule 2 - Correctness] `_process_one` uses `TemporaryDirectory` for local scratch**
- **Found during:** Implementation — the original code wrote `sample_dir = out_dir / safe` assuming `out_dir` is local; with GCS `out_dir` this would fail
- **Fix:** `_process_one` always uses a `TemporaryDirectory` for the local scratch (reprojected GeoTIFF + label generation); only pyramid writes go to GCS via `_GCSWriter`
- **Files modified:** `scripts/build_satellite_dataset.py`

## Threat Surface Scan

No new trust boundaries introduced beyond those already in the plan's threat model. The `--local-ok` guard and `fs.pipe_file` atomic write pattern implement the T-02-30 and T-02-31 mitigations as planned.

## Self-Check: PASSED

| Item | Status |
|------|--------|
| scripts/build_satellite_dataset.py exists | FOUND |
| tests/test_satellite.py exists | FOUND |
| Commit f039a82 (RED) | FOUND |
| Commit 574f4bd (GREEN) | FOUND |
| _DEFAULT_SUMMARY starts with gs:// | PASS |
| _DEFAULT_MANIFEST starts with gs:// | PASS |
| _DEFAULT_OUT starts with gs:// | PASS |
| _GCSWriter importable from gcs_io | PASS |
| pytest tests/test_satellite.py -x | 13/13 passed |
