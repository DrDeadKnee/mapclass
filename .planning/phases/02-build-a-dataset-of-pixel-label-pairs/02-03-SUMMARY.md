---
phase: 02-build-a-dataset-of-pixel-label-pairs
plan: "03"
subsystem: historical-pipeline
tags: [gcs-io, historical, rw-04, tiling, dataset-build]
dependency_graph:
  requires: ["02-01", "02-02"]
  provides: ["GCS-canonical historical dataset write path", "unregistered_manifest.json in GCS"]
  affects: ["02-05"]
tech_stack:
  added: []
  patterns:
    - "lazy gcsfs import (try/except at module level)"
    - "fs.pipe_file for atomic GCS blob writes"
    - "_GCSWriter passed as out_root to tiling.tile (per-pyramid, not shared)"
    - "shared gcsfs.GCSFileSystem across ThreadPoolExecutor workers"
key_files:
  created:
    - tests/test_historical.py
  modified:
    - scripts/build_historical_dataset.py
decisions:
  - "option (b) for manifest: emit_manifest writes to local scratch, then fs.pipe_file uploads bytes to GCS — avoids modifying rumsey.py"
  - "raw Rumsey GeoTIFFs stay local scratch (re-downloadable from LUNA per RW-04 A-R5)"
  - "_process_one receives fs + gcs_out_prefix; constructs per-sample _GCSWriter inside the worker"
metrics:
  duration: "~12 minutes"
  completed: "2026-05-16"
  tasks_completed: 1
  files_changed: 2
---

# Phase 02 Plan 03: Historical GCS-Canonical Pipeline Summary

**One-liner:** GCS-canonical historical pipeline: `cmd_search` pipes `unregistered_manifest.json` to `mapclass-training-northeast1/data/historical/raw/` via `fs.pipe_file`; `cmd_build` streams pyramid output through `_GCSWriter` to `data/historical/dataset/<sample>/pyramids`; raw Rumsey GeoTIFFs stay ephemeral local scratch (RW-04).

## What Was Built

### Task 1: build_historical_dataset.py — GCS manifest + GCS dataset write (TDD)

**RED commit:** `2339afd` — failing tests covering three behaviours (manifest to GCS, dataset to GCS via _GCSWriter, raw .tif stays local).

**GREEN commit:** `e3fcf0c` — implementation satisfying all tests.

Changes to `scripts/build_historical_dataset.py`:

1. **Lazy gcsfs import** at module level (try/except pattern from `gcs_checkpoint.py`). The module can be imported offline where gcsfs is absent; tests patch `build_historical_dataset.gcsfs`.

2. **Import `_GCSWriter`, `GCS_PROJECT`, `DATA_PREFIX` from `gcs_io`** — pins to the single authoritative constants source.

3. **`cmd_search` — manifest to GCS (RW-04):**
   - `rumsey.emit_manifest` writes to local scratch (`raw_dir/unregistered_manifest.json`) — no modification to `rumsey.py` (PATTERNS.md option b).
   - Immediately after: `fs.pipe_file(_HISTORICAL_MANIFEST_GCS_KEY, manifest_bytes)` uploads the bytes to `mapclass-training-northeast1/data/historical/raw/unregistered_manifest.json`.
   - `raw_dir / "georeferenced"` is created before download begins (ensures emit_manifest parent exists).

4. **`_process_one` — pyramid output to GCS:**
   - Receives `fs` (shared `gcsfs.GCSFileSystem`, thread-safe) and `gcs_out_prefix` (bare GCS path).
   - When `gcs_out_prefix` is set: creates `sample_dir` in `tempfile.mkdtemp` scratch; calls `tiling.tile(sample_dir, out_root=_GCSWriter(fs, gcs_prefix))` where `gcs_prefix = f"{gcs_out_prefix}/{sample_name}/pyramids"`.
   - Each `_GCSWriter` is constructed inside the worker, not shared across threads (_GCSWriter is not thread-safe; `gcsfs.GCSFileSystem` is).

5. **`cmd_build` — detect GCS out_dir:**
   - `is_gcs = out_dir_str.startswith("gs://")` — branches on local vs. GCS.
   - GCS path: instantiates shared `fs`, strips `gs://` for bare prefix, skips `out_dir.mkdir`.
   - Local path: keeps `Path(out_dir).mkdir(parents=True, exist_ok=True)` for tests/offline use.
   - Passes `fs` and `gcs_out_prefix` through the `ThreadPoolExecutor` submit call.

6. **Argparse (Pitfall R-1 guard):**
   - `--out-dir` default changed to `"gs://mapclass-training-northeast1/data/historical/dataset"`.
   - `--local-ok` flag added to both `build` and `full` subparsers.
   - Startup guard: non-`gs://` `--out-dir` without `--local-ok` calls `parser.error(...)` (hard exit).

New file `tests/test_historical.py`:

- `TestManifestWrittenToGCS.test_manifest_written_to_gcs` — verifies `_MockGCSFileSystem._store` contains the manifest at the canonical GCS key; confirms no raw `.tif` in GCS.
- `TestHistoricalDatasetWrittenToGCS.test_historical_dataset_written_to_gcs` — intercepts `tiling.tile` calls; asserts `out_root` is a `_GCSWriter` with prefix under `historical/dataset/.../pyramids`.
- `TestRawGeoTIFFStaysLocal.test_raw_geotiff_stays_local` — tracks `pipe_file` calls; asserts zero `.tif` uploads.

## Verification Results

```
pytest tests/test_historical.py -x -v
3 passed in 0.47s
```

Acceptance criteria verified:
- `grep -n "_GCSWriter" scripts/build_historical_dataset.py` — non-empty (lines 13, 50, 169, 172)
- `grep -n "historical/raw/unregistered_manifest.json" scripts/build_historical_dataset.py` — non-empty (line 59)
- `pytest tests/test_historical.py::TestManifestWrittenToGCS::test_manifest_written_to_gcs -x` — exits 0
- `pytest tests/test_historical.py -x` — exits 0

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] geo_dir not created before emit_manifest**
- **Found during:** GREEN phase (first test run)
- **Issue:** `rumsey.emit_manifest` writes to `raw_dir/unregistered_manifest.json`; the parent `raw_dir` was not created before the call, causing `FileNotFoundError` when the test mock tried to write the manifest.
- **Fix:** Added `geo_dir.mkdir(parents=True, exist_ok=True)` before `rumsey.search_maps()` call in `cmd_search` — ensures both `raw_dir` and `raw_dir/georeferenced` exist on local scratch prior to any writes.
- **Files modified:** `scripts/build_historical_dataset.py`
- **Commit:** `e3fcf0c`

None others — plan executed as specified.

## TDD Gate Compliance

| Gate | Commit | Status |
|------|--------|--------|
| RED (test commit) | 2339afd | PASS — 3 tests fail as expected before implementation |
| GREEN (feat commit) | e3fcf0c | PASS — 3 tests pass after implementation |
| REFACTOR | N/A | No structural cleanup needed |

## Threat Surface Scan

No new trust boundaries beyond those declared in the plan's `<threat_model>`. The implementation matches the declared mitigations:
- T-02-20: `--out-dir` defaults `gs://`, non-`gs://` requires `--local-ok` (Pitfall R-1 guard).
- T-02-21: `pipe_file` is atomic per object; missing-file SKIP guard precedes `tile()`.
- T-02-22: ADC-only lazy-gcsfs pattern; no secrets in source.
- T-02-23: raw TIFs stay local (re-downloadable from LUNA).

## Known Stubs

None — the implementation is complete. `cmd_search` and `cmd_build` both write to GCS as specified. The plan's goal (RW-04: historical outputs survive ephemeral compute) is achieved.

## Self-Check: PASSED

- `tests/test_historical.py` exists: FOUND
- `scripts/build_historical_dataset.py` modified: FOUND
- Commit `2339afd` (RED): FOUND
- Commit `e3fcf0c` (GREEN): FOUND
