---
phase: 02-build-a-dataset-of-pixel-label-pairs
plan: 02
subsystem: historical-pipeline
tags: [iiif, allmaps, georeferencing, rasterio, geotiff, drop-accounting]
requires:
  - "02-01: allmaps.lookup(manifest_url) -> list[dict]"
provides:
  - "scripts/historical/iiif.py: build_iiif_url / fetch_iiif_image / scale_gcps"
  - "scripts/historical/georef.py: gcps_to_affine / write_georeferenced_geotiff"
  - "rumsey.download_georeferenced -> (list[Path], status) Allmaps path"
  - "rumsey.emit_manifest v2 hand-off (per-reason status)"
  - "build_historical_dataset.cmd_search per-reason drop counter (D-05)"
affects:
  - scripts/historical/rumsey.py
  - scripts/build_historical_dataset.py
tech-stack:
  added: []
  patterns:
    - "exponential-backoff HTTP retry (copied from rumsey._get_json)"
    - "rasterio.transform.from_gcps least-squares affine (no hand-roll)"
    - "GroundControlPoint(row=py,col=px,x=lng,y=lat) named-kwarg convention"
    - "per-reason drop-counter dict (D-05)"
key-files:
  created:
    - scripts/historical/iiif.py
    - scripts/historical/georef.py
    - tests/integration/test_iiif_online.py
    - tests/integration/test_rumsey_online.py
  modified:
    - scripts/historical/rumsey.py
    - scripts/build_historical_dataset.py
    - tests/test_iiif.py
    - tests/test_georef.py
    - tests/test_rumsey.py
    - tests/test_label.py
decisions:
  - "emit_manifest signature changed to list[(item, status)] — status now comes from the Allmaps lookup result, not derivable from the LUNA item alone"
  - "IIIF JPEG cached to <plate>/_iiif.jpg then unlinked after GeoTIFF write (no persistent cache; keeps raw dir to source.tif only)"
metrics:
  duration: ~20m
  completed: 2026-05-15
  tasks: 3
  files: 10
---

# Phase 2 Plan 02: Historical Pipeline (Allmaps→IIIF→GeoTIFF) Summary

Replaced the dead WMS download path with the locked LUNA → Allmaps → IIIF →
georeferenced-GeoTIFF flow: per-annotation IIIF best-fit fetch, GCP scaling by
actual fetched/original ratio, GDAL least-squares affine fit, EPSG:4326 GeoTIFF
write, and per-reason drop accounting with a loud >50% out-of-scale signal.

## What Was Built

- **`scripts/historical/iiif.py` (NEW)** — `build_iiif_url` emits
  `<service>/full/!4096,4096/0/default.jpg`; `fetch_iiif_image` streams the JPEG
  with exp-backoff retry, a 200 MB Content-Length + stream guard (threat
  T-02-04), then reopens the file to return the ACTUAL `(w, h)`; `scale_gcps`
  scales by the actual fetched/original ratio (Pitfall 2 — sx/sy computed
  independently, not assumed equal).
- **`scripts/historical/georef.py` (NEW)** — `gcps_to_affine` builds
  `GroundControlPoint(row=py, col=px, x=lng, y=lat)` with named kwargs
  (Pitfall 1) and delegates the fit to `rasterio.transform.from_gcps` (GDAL
  LSQ, not hand-rolled); `write_georeferenced_geotiff` writes a 3-band uint8
  EPSG:4326 GeoTIFF directly consumable by `label.make_labels` (D-03).
- **`scripts/historical/rumsey.py`** — deleted `_wms_url`, `_parse_bbox`,
  `_download_wms_geotiff` (D-01). Added `_sanitize_id` (non-`[\w-]` → `_`,
  threat T-02-02). Rewrote `download_georeferenced` to iterate
  `allmaps.lookup` results, drop out-of-scale annotations silently, fetch +
  scale + affine + write each surviving annotation to
  `<sanitized_id>__plate<i>/source.tif`, and return `(paths, status)` with the
  five D-04 reasons; one bad plate cannot abort the batch (T-02-05). Rewrote
  `emit_manifest` to carry the v2 hand-off fields + per-reason status.
  `search_maps`, `_richness_score`, `_field`, `_get_json`, geo helpers kept
  verbatim.
- **`scripts/build_historical_dataset.py`** — `cmd_search` now drives a
  five-reason `drops` dict from `download_georeferenced` status, routes only
  `not_in_allmaps`/`gcps_insufficient` to the manifest, prints a per-reason
  "Search summary", and emits a loud >50% out-of-scale warning (D-05).
  `_process_one`/`cmd_build`/`main` byte-unchanged.

## Tasks

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 (RED) | failing iiif/georef tests | fe26b9c | tests/test_georef.py, tests/test_iiif.py, tests/integration/test_iiif_online.py |
| 1 (GREEN) | iiif.py + georef.py | 1ebb1cb | scripts/historical/iiif.py, scripts/historical/georef.py |
| 2 | rumsey Allmaps refactor | ed2931f | scripts/historical/rumsey.py, tests/test_rumsey.py, tests/integration/test_rumsey_online.py |
| 3 | cmd_search drop counter | 7507cf0 | scripts/build_historical_dataset.py, tests/test_label.py |

## Verification

- `pytest tests/test_georef.py tests/test_iiif.py tests/test_rumsey.py tests/test_label.py -q --ignore=tests/integration` → 9 passed
- Full offline suite → 13 passed, 8 skipped (skips are other-wave stubs)
- `grep -E '_wms_url|_parse_bbox|_download_wms_geotiff' rumsey.py build_historical_dataset.py` → no matches
- Acceptance greps: `from_gcps`, `GroundControlPoint(row=`, `/full/!4096,4096/0/default.jpg`, `allmaps.lookup`, `re.sub(r"[^\w-]"`, `out_of_scale`, `Search summary`, `0.5` all present
- Online integration tests (`test_iiif_online`, `test_rumsey_online`) added, marked `@pytest.mark.integration` (run by the phase gate, not per-commit)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Installed missing `pyproj` dependency**
- **Found during:** Task 3 (test_label.py collection)
- **Issue:** `historical/label.py` imports `pyproj` at module top; `pyproj` was
  declared in `requirements.txt` but absent from the active environment, causing
  a `ModuleNotFoundError` at test collection.
- **Fix:** `pip install pyproj` (3.7.2) — already a declared dependency, env was
  just missing it. No code change.
- **Files modified:** none (environment only)
- **Commit:** n/a (environment fix, not committed)

### Plan-Interpretation Notes (not deviations)

- `emit_manifest` signature changed from `list[dict]` to `list[(item, status)]`.
  The plan said replace the hardcoded `"status": "needs_gcps"` with the D-04
  per-reason status; that status is only knowable from the Allmaps lookup result
  (not from the LUNA item alone), so the caller now passes `(item, status)`
  pairs. `cmd_search` was updated to match in the same plan.
- The plan's pre-existing test stubs used different function names
  (`test_max_edge_size_syntax`, `test_emit_manifest_per_reason_status`, etc.)
  than the plan `<behavior>` spec. The `<behavior>` spec is authoritative, so
  the test bodies use the spec names (`test_build_iiif_url`,
  `test_scale_factor_best_fit`, `test_affine_roundtrip`, `test_gcp_convention`).
  The pre-existing skip stubs were overwritten by the real bodies.

## Authentication Gates

None.

## Known Stubs

None. The IIIF JPEG is fetched to a temporary `<plate>/_iiif.jpg` and unlinked
after the GeoTIFF write — `source.tif` is the only persisted artifact, as the
plan specifies.

## TDD Gate Compliance

Task 1 (`tdd="true"`) followed RED → GREEN: `test(02-02)` commit fe26b9c (tests
failing — modules absent) precedes `feat(02-02)` commit 1ebb1cb (implementation,
tests green). No REFACTOR commit (implementation was minimal and clean). Tasks 2
and 3 are `type="auto"` (not TDD-gated).

## Self-Check: PASSED

- scripts/historical/iiif.py — FOUND
- scripts/historical/georef.py — FOUND
- scripts/historical/rumsey.py — FOUND (WMS removed)
- scripts/build_historical_dataset.py — FOUND (cmd_search rewired)
- commits fe26b9c, 1ebb1cb, ed2931f, 7507cf0 — all FOUND in git log
