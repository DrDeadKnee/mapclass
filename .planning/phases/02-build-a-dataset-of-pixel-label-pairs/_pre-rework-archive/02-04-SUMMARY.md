---
phase: 02-build-a-dataset-of-pixel-label-pairs
plan: 04
subsystem: satellite-pipeline
status: complete
tags: [satellite, sentinel-2, stac, cog, coverage-scan, worldcover]
requires:
  - 02-01
  - 02-02 (scripts/historical/georef.py::write_georeferenced_geotiff — reused verbatim)
provides:
  - scripts/satellite/ package (stac, fetch, coverage, weights)
  - scripts/build_satellite_dataset.py (coverage-scan / search / build)
affects:
  - requirements.txt (pystac-client>=0.9 added)
tech-stack:
  added: [pystac-client>=0.9]
  patterns:
    - anonymous public-S3 via AWS_NO_SIGN_REQUEST + GDAL VSI-CURL (no boto3)
    - COG byte-range windowed read (rasterio.windows.Window) — never full-scene
    - typed lookup error + exponential backoff (StacLookupError, mirrors AllmapsLookupError)
    - cached one-shot coarse-WorldCover summary (mandatory sidecar cache)
    - per-reason drop counter (D-13, parallels D-05)
key-files:
  created:
    - scripts/satellite/__init__.py
    - scripts/satellite/weights.py
    - scripts/satellite/stac.py
    - scripts/satellite/fetch.py
    - scripts/satellite/coverage.py
    - scripts/build_satellite_dataset.py
    - tests/test_coverage.py
    - tests/integration/test_stac_online.py
    - tests/integration/test_satellite_online.py
  modified:
    - requirements.txt
    - tests/test_stac.py
    - tests/test_satellite.py
decisions:
  - "Satellite loss weights: balance-tilt (checkpoint:decision RESOLVED) — water/trees 0.6, cropland/built_up/flooded_wetland 1.3, others 1.0, SATELLITE_TOPO_WEIGHT 1.0"
  - "Rule 1 fix: reproject the windowed UTM read to EPSG:4326 before the shared georef writer (which hard-codes EPSG:4326) so make_labels' WorldCover/DEM alignment stays geographically correct"
metrics:
  duration: ~1 session
  completed: 2026-05-15
---

# Phase 02 Plan 04: Satellite Pipeline Summary

One-liner: Sentinel-2 L2A satellite source family — a cached class-diversity
coarse-WorldCover region picker (D-14) feeds a cloud-filtered STAC search whose
lowest-cloud scene is fetched as a 4096-px RGB COG byte-range window, then
labelled with the reused historical pipeline and the approved balance-tilt
satellite loss weights.

## Execution Status

All implementation tasks complete. The plan's first task (Task 0) was a
BLOCKING `checkpoint:decision` for the per-source loss-weight values; it was
presented to the user and **RESOLVED as `balance-tilt`**. This continuation
agent resumed at Task 1 and executed Tasks 1-3 to completion.

## Tasks Completed

| Task | Name | Status | Commit |
| ---- | ---- | ------ | ------ |
| 0 | Checkpoint: approve per-source loss weights | RESOLVED — balance-tilt | (decision, pre-resume) |
| 1 | satellite/ package — stac.py, fetch.py, weights.py | done | `8387e65` |
| 2 | coverage.py — class-diversity region picker (D-14) | done | `e6ccaba` |
| 3 | build_satellite_dataset.py — sub-commands (D-13) | done | `aa2b54f` |

## Approved Checkpoint Decision

**balance-tilt** (locked — hard-coded verbatim into `scripts/satellite/weights.py`):

| class | weight | rationale |
|-------|--------|-----------|
| water | 0.6 | globally over-represented — discount |
| trees | 0.6 | globally over-represented — discount |
| shrubland | 1.0 | neutral |
| grassland | 1.0 | neutral |
| cropland | 1.3 | synthetic-absent, PROJECT.md up-weighted, satellite is primary source |
| built_up | 1.3 | synthetic-absent, PROJECT.md up-weighted, satellite is primary source |
| bare_sparse | 1.0 | neutral |
| flooded_wetland | 1.3 | synthetic-absent, PROJECT.md up-weighted, satellite is primary source |
| snow_ice | 1.0 | neutral |

`SATELLITE_TOPO_WEIGHT = 1.0`. WorldCover labels are contemporaneous with the
imagery (no temporal-drift discount, unlike the historical source). The
`sample_weights.json` shape is the locked 4-key contract
(`land_cover_weights` / `topography_weight` / `source` / `map_file`) with
`source == "satellite"`.

## What Was Built

- **`scripts/satellite/weights.py`** — the locked-shape satellite weights dict
  + `write_sample_weights(output_dir, map_file)`. Hard-codes the approved
  balance-tilt floats; the rationale is recorded in the module docstring.
- **`scripts/satellite/stac.py`** — `find_lowest_cloud_scene(bbox,
  datetime_range, max_cloud=10)` over the Element84 Earth Search v1 API:
  `query={"eo:cloud_cover": {"lt": max_cloud}}` on `sentinel-2-l2a`, returns
  the min-cloud item or `None`; typed `StacLookupError`; retry/backoff copied
  from `allmaps.py`; `AWS_NO_SIGN_REQUEST` set module-top.
- **`scripts/satellite/fetch.py`** — `fetch_visual_window(item, dst_path,
  dst_window_px=4096)` reads a centred `rasterio.windows.Window` of the scene's
  `visual` TCI asset via COG byte-range (never the full ~600 MB scene, threat
  T-02-10), reprojects to EPSG:4326, and writes via the **single shared**
  `historical.georef.write_georeferenced_geotiff` (no fork — W-1). Missing /
  unreadable asset → `None` (threat T-02-11).
- **`scripts/satellite/coverage.py`** — `build_summary()` builds a one-shot
  per-1° WorldCover class-count summary (mandatory JSON sidecar cache;
  reloaded if present), reusing `historical.worldcover._tile_origins` and
  `WC_REMAP` verbatim. `pick_regions(n, seed)` ranks cells by Shannon entropy
  with a multiplicative up-weight on cropland/built_up/flooded_wetland (D-14)
  and a seeded deterministic tie-break; each region carries a
  latitude-appropriate season (`season_for_latitude`, Pitfall 4).
- **`scripts/build_satellite_dataset.py`** — `coverage-scan` / `search` /
  `build` on the `build_historical_dataset.py` argparse + threadpool scaffold.
  `search` drop-counts `no_qualifying_scene` / `stac_search_failed`; `build`
  drop-counts `fetch_failed`; both print the D-13 summary with a loud <50%
  warning. The `build` worker reuses `historical.label.make_labels` **verbatim**
  then writes the satellite `sample_weights.json`. Region ids are sanitized
  (non-`[\w-]` → `_`) before the path join (threat T-02-12).

## Reuse (verbatim, no fork)

- `historical.georef.write_georeferenced_geotiff` — imported directly by
  `fetch.py` (exactly one import, no local writer def — W-1 verified).
- `historical.label.make_labels` — called verbatim by the build worker; zero
  new label or GeoTIFF-writer code.
- `historical.worldcover._tile_origins` / `WC_REMAP` / `_tile_name` / base-URL
  constants — reused by `coverage.py`.
- `git diff scripts/historical/` is empty for this plan — the historical
  package was not modified.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Reproject the windowed UTM read to EPSG:4326 before the shared writer**
- **Found during:** Task 1 (fetch.py)
- **Issue:** Sentinel-2 scenes are in a UTM CRS, but the shared Plan-02
  `write_georeferenced_geotiff` hard-codes `crs=EPSG:4326`. Passing the raw
  UTM `window_transform` to it would label UTM coordinates as WGS84, corrupting
  `make_labels`' downstream WorldCover/DEM bbox alignment (D-03).
- **Fix:** `fetch.py` now reprojects the 4096-px window (and only the window —
  never the full scene) from the scene CRS to EPSG:4326 via
  `rasterio.warp.reproject` + `calculate_default_transform`, then passes the
  WGS84 transform/array to the unmodified shared writer. The writer is still
  reused verbatim (no fork); the fix lives entirely in `fetch.py`.
- **Files modified:** scripts/satellite/fetch.py
- **Commit:** `8387e65`

## Threat Mitigations Applied

- **T-02-10** (full-scene download DoS): `fetch.py` reads only a centred
  `Window`; offline test asserts the centred-window math and clamping.
- **T-02-11** (malformed item crashes batch): missing/unreadable `visual`
  asset → `None`; the threadpool worker returns status instead of raising.
- **T-02-12** (region-id path traversal): `_sanitize` replaces non-`[\w-]`
  with `_` before the `out_dir / safe` join.

## Verification

- `pytest tests/test_stac.py tests/test_satellite.py tests/test_coverage.py -q
  --ignore=tests/integration` → 17 passed.
- Full quick suite `pytest tests/ -q --ignore=tests/integration` → 43 passed,
  2 skipped (the 2 skips are unrelated Wave-0 skeletons — `test_tiling.py` and
  another future-plan skeleton — not introduced or owned by this plan).
- `python -c "import ast; ast.parse(open('scripts/build_satellite_dataset.py').read())"`
  succeeds; `--help` parses.
- W-1: exactly one `from historical.georef import write_georeferenced_geotiff`
  in `scripts/satellite/`, no local writer def.
- Integration online tests added (`tests/integration/test_stac_online.py`,
  `tests/integration/test_satellite_online.py`) — `@pytest.mark.integration`,
  run as the phase gate against the live STAC API / S3.

## Known Stubs

None. All modules are wired end-to-end; the only placeholder-shaped value is
the empty `_SceneItem.properties = {}` in the build orchestrator, which is an
intentional minimal STAC-item stand-in (the manifest already carries the
resolved `visual_href`; `properties` is unused on the build path).

## Self-Check: PASSED

All 9 created files exist on disk; all 3 task commits (`8387e65`, `e6ccaba`,
`aa2b54f`) are present in git history. No missing items.
