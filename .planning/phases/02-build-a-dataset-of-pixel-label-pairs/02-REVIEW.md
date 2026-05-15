---
phase: 02-build-a-dataset-of-pixel-label-pairs
reviewed: 2026-05-15T00:00:00Z
depth: standard
files_reviewed: 38
files_reviewed_list:
  - scripts/biome_mapping.py
  - scripts/build_dataset.py
  - scripts/build_historical_dataset.py
  - scripts/build_satellite_dataset.py
  - scripts/historical/allmaps.py
  - scripts/historical/georef.py
  - scripts/historical/iiif.py
  - scripts/historical/rumsey.py
  - scripts/label.py
  - scripts/render.py
  - scripts/satellite/__init__.py
  - scripts/satellite/coverage.py
  - scripts/satellite/fetch.py
  - scripts/satellite/stac.py
  - scripts/satellite/weights.py
  - scripts/synthetic_weights.py
  - scripts/tiling.py
  - tests/conftest.py
  - tests/integration/__init__.py
  - tests/integration/test_allmaps_online.py
  - tests/integration/test_iiif_online.py
  - tests/integration/test_rumsey_online.py
  - tests/integration/test_satellite_online.py
  - tests/integration/test_stac_online.py
  - tests/test_allmaps.py
  - tests/test_coverage.py
  - tests/test_georef.py
  - tests/test_iiif.py
  - tests/test_label.py
  - tests/test_render.py
  - tests/test_rumsey.py
  - tests/test_satellite.py
  - tests/test_split.py
  - tests/test_stac.py
  - tests/test_synthetic_weights.py
  - tests/test_tiling.py
findings:
  critical: 3
  warning: 9
  info: 5
  total: 17
status: issues_found
---

# Phase 2: Code Review Report

**Reviewed:** 2026-05-15T00:00:00Z
**Depth:** standard
**Files Reviewed:** 38 (17 source modules; the remainder are tests read for cross-reference, not separately findings-scoped except where they mask a defect)
**Status:** issues_found

## Summary

Phase 2 implements a three-source pixel-label dataset pipeline (historical Rumsey/Allmaps/IIIF, synthetic Azgaar, satellite Sentinel-2/WorldCover/DEM) plus a shared nested-pyramid tiler. Overall the HTTP retry idioms, decompression-bomb guards, path sanitization, and the frozen-split design are present and largely sound, and the test suite is thorough for the happy paths.

However the adversarial pass surfaced three correctness-class BLOCKERs that the existing tests do not catch because they only assert relative/structural properties:

1. The satellite seasonal datetime windows are wrong: the tropics window is 24 months (not the documented "trailing 12"), and the southern-hemisphere window spans two different austral summers rather than one contiguous season. This silently degrades scene quality for ~half the globe and is not caught because `test_season_varies_by_hemisphere` only asserts the *prefix* and *inequality*, never the actual span.
2. `build_dataset` collapses distinct Azgaar source files that sanitize/collide to the same ID, causing silent data loss and cross-split contamination of the EVAL-01 frozen split.
3. `label.make_label_arrays` (and `render`) crash with an unhandled `KeyError`/`TypeError` on Azgaar features missing `biome`/`height` or carrying non-Polygon/MultiPolygon geometry, and `_bbox` raises `ValueError` on an empty/all-non-polygon feature set — these abort a whole source instead of the documented per-source resilience.

Warnings cluster around silent error swallowing that defeats the loudly-advertised per-reason drop accounting (D-05/D-13), an unsafe `assert` used as a runtime data-integrity check, GCP/CRS edge cases in the georeferencing path, and a resume path that returns a non-existent file.

## Critical Issues

### CR-01: Satellite seasonal datetime windows are wrong (tropics = 24 months, southern = split season)

**File:** `scripts/satellite/coverage.py:67-80`
**Issue:** `season_for_latitude` builds three windows:

- Southern temperate (`lat < -23.5`): `f"{ref_year - 1}-11-01/{ref_year}-03-31"` → `2022-11-01/2023-03-31`. This is a 5-month range that includes November 2022 (start of the 2022–23 austral summer) and January–March 2023 (end of the same summer) but is described/intended as one austral summer. It is acceptable as a single contiguous span, but it is inconsistent with the northern window which is 5 months *within one calendar year* — the asymmetry is undocumented and the southern range is 2 months longer (Nov, Dec, Jan, Feb, Mar vs May–Sep). More seriously:
- Tropics (`|lat| <= 23.5`): `f"{ref_year - 1}-{1:02d}-01/{ref_year}-12-31"` → `2022-01-01/2023-12-31`. The docstring states "the trailing 12 months", but this is a **24-month** window. STAC `find_lowest_cloud_scene` will then min-by-cloud over two full years of scenes, defeating the seasonal-consistency intent (Pitfall 4) and pulling scenes from arbitrary years. The `{1:02d}` formatting of a literal `1` is also dead/confused code — it was clearly meant to be a month variable.

`test_season_varies_by_hemisphere` does not catch this: it asserts only `north.startswith("2023-05-01")`, `south.startswith("2022-11-01")`, and `tropic != north and tropic != south` — never the end date or span length.

**Fix:**
```python
def season_for_latitude(lat: float, ref_year: int = 2023) -> str:
    if lat > 23.5:
        return f"{ref_year}-05-01/{ref_year}-09-30"
    if lat < -23.5:
        # one contiguous austral summer (Dec–Feb), no double-season
        return f"{ref_year - 1}-12-01/{ref_year}-02-28"
    # tropics: exactly the trailing 12 months
    return f"{ref_year - 1}-12-31/{ref_year}-12-31"
```
Then add a test asserting the exact full range string for a northern, southern, and tropical latitude.

### CR-02: Source-ID sanitization collisions silently drop maps and corrupt the frozen EVAL-01 split

**File:** `scripts/build_dataset.py:80-83, 215-224`
**Issue:** `build()` computes `source_ids = [_sanitize_stem(g.stem) for g in geojsons]` and routes each source by `sid in test_ids`. `_sanitize_stem` maps any non-`[\w-]` char to `_`. Two distinct raw files — e.g. `europe (1).geojson` and `europe-1.geojson`, or `map.v2.geojson` and `map_v2.geojson` — collapse to the same sanitized `src_id`. Consequences:

1. `build_one_source` writes both into the same `<src_id>__<style>/` directory; the second silently overwrites the first → data loss. The phase explicitly calls out "A clean pipeline that quietly throws away most of the data is not job done."
2. The split is keyed by the sanitized ID. If one of a colliding pair is in `test_ids` and the other is not, the loop at line 222–224 routes them to *different* roots (`train/` vs `test/`) under the *same* directory name on the same `output_dir` — depending on iteration order one overwrites the other across the train/test boundary. This is exactly the EVAL-01 leakage the frozen split is designed to prevent, and it is invisible because `_sanitize_stem` is also applied when computing `test_ids`, so the disjointness test in `test_split.py` (which uses already-distinct IDs) never exercises the collision.

**Fix:** Detect collisions and fail loudly before any build/split:
```python
source_ids = [_sanitize_stem(g.stem) for g in geojsons]
dupes = {s for s in source_ids if source_ids.count(s) > 1}
if dupes:
    print(f"FATAL: source stems collide after sanitization: {sorted(dupes)} "
          f"— rename the raw .geojson files; aborting to protect the "
          f"frozen split (EVAL-01).")
    sys.exit(1)
```
Better long-term: derive `src_id` from a content hash or the full original stem with a reversible encoding, so distinct sources never alias.

### CR-03: Malformed Azgaar GeoJSON crashes the whole source instead of per-source resilience

**File:** `scripts/label.py:33-47, 77-92`; `scripts/render.py:86-99, 116-127`
**Issue:** Several unguarded accesses on untrusted (downloaded/user-supplied) GeoJSON:

- `_bbox` at `label.py:47` does `return min(xs), min(ys), ...` — if `features` is empty, or every feature is a non-Polygon/non-MultiPolygon geometry (e.g. `Point`, `LineString`, `GeometryCollection`, or `null`), `xs`/`ys` are empty and `min()` raises `ValueError: min() arg is an empty sequence`.
- `_rings`/`_bbox` assume `geom["type"]` is exactly `"Polygon"` else treats it as MultiPolygon and indexes `geom["coordinates"]` two levels deep — a `Point` geometry then raises `TypeError`/`ValueError` while unpacking `for x, y in ring`.
- `make_label_arrays:79-80` does `int(props["biome"])` / `int(props["height"])` with no guard — a feature missing either key raises `KeyError`; a non-numeric value raises `ValueError`.
- `h_to_landcover` (`biome_mapping.py:94`) does `BIOME_TO_LANDCOVER_NAME[biome]` with no default — a biome ID outside 0–21 (Azgaar has added biomes before) raises `KeyError`, aborting the source.

`build_dataset.build_one_source` is wrapped in a per-source `except Exception` (line 228), so the *process* survives, but the documented contract (T-02-09 "per-source resilience") becomes "any one malformed feature drops the entire map" rather than skipping the bad feature. `width = int(max_x - min_x) + 1` can also produce a degenerate or enormous canvas if coordinates are pathological (no bound on image size — decompression-bomb-analogue for the synthetic path; threat parity with T-02-04 is absent here).

**Fix:** Validate and skip per-feature, and bound the canvas:
```python
def _bbox(features):
    xs, ys = [], []
    for feat in features:
        geom = (feat or {}).get("geometry") or {}
        if geom.get("type") not in ("Polygon", "MultiPolygon"):
            continue
        for ring in _rings(geom):
            for pt in ring:
                if len(pt) >= 2:
                    xs.append(pt[0]); ys.append(pt[1])
    if not xs:
        raise ValueError("GeoJSON has no usable Polygon/MultiPolygon geometry")
    return min(xs), min(ys), max(xs), max(ys)
```
Guard the property reads (`props.get("biome")`, `props.get("height")`, skip feature on missing/non-int), give `h_to_landcover` a NODATA fallback for unknown biome IDs, and assert a sane max canvas dimension before allocating the PIL image.

## Warnings

### WR-01: `assert` used as a runtime data-integrity check (stripped under `python -O`)

**File:** `scripts/tiling.py:190`
**Issue:** `assert (pdir / _WEIGHTS_FILE).read_bytes() == weights_blob` is the only verification that the per-source loss weights propagated byte-identically to every pyramid (D-claude-discretion / locked weight invariant). Running under `python -O` strips all `assert` statements, silently disabling this integrity check in exactly the production batch-build scenario where it matters. An `assert` is also the wrong tool for I/O verification.
**Fix:**
```python
if (pdir / _WEIGHTS_FILE).read_bytes() != weights_blob:
    raise RuntimeError(f"weight propagation corrupted for pyramid {pid}")
```

### WR-02: `download_georeferenced` resume path returns a path to a possibly partial/zero file without re-validation

**File:** `scripts/historical/rumsey.py:344-347`
**Issue:** When `out_path.exists()` the code appends it to `written` and `continue`s, treating it as a completed plate. If a prior run crashed mid-`write_georeferenced_geotiff` (or the JPEG decode failed after the `.tif` was opened for write), a truncated/invalid `source.tif` is now permanently accepted as "ok" — the build will hand a corrupt GeoTIFF to `make_labels`. There is no size/openability check.
**Fix:** Before trusting an existing `source.tif`, open it with `rasterio` in a try/except (or check it has a CRS + nonzero dimensions); if invalid, delete and re-fetch.

### WR-03: Silent `except (KeyError, AttributeError, TypeError)` in STAC `visual` resolution miscounts the drop reason

**File:** `scripts/build_satellite_dataset.py:131-136`
**Issue:** A resolved scene with no `visual` asset is counted as `no_qualifying_scene`. That bucket already means "no scene under the cloud threshold." Conflating "found a scene but it lacks the TCI asset" with "no scene at all" corrupts the D-13 per-reason accounting that the phase explicitly requires to be loud and accurate ("A clean pipeline that quietly throws away most of the data is not job done"). The malformed-asset case is invisible.
**Fix:** Add a distinct `"missing_visual_asset"` (or `"resolved_no_visual"`) key to the `drops` dict and count it separately so the summary surfaces it.

### WR-04: Broad `except Exception` around STAC client masks programming errors and retries non-transient failures

**File:** `scripts/satellite/stac.py:77-84`
**Issue:** The `try` wraps `pystac_client.Client.open`, `client.search`, and `list(search.items())` in a single `except Exception`. A deterministic failure (bad bbox shape, invalid datetime string from CR-01, an `AttributeError` from a pystac API change) is treated as transient and retried `_MAX_RETRIES` times with exponential backoff, then re-raised as `StacLookupError` — wasting ~6s+ per region across potentially hundreds of regions and obscuring the real cause. Compare `allmaps.lookup`, which narrowly catches `requests.RequestException`.
**Fix:** Catch `pystac_client` / network exception types specifically; let `TypeError`/`ValueError`/`AttributeError` propagate immediately without retry.

### WR-05: GCP scaling divides by `orig_w`/`orig_h` with no zero/None guard

**File:** `scripts/historical/iiif.py:129-130`; `scripts/historical/allmaps.py:142`
**Issue:** `_parse_annotation` builds `image_size = (int(width), int(height)) if width and height else None`. If Allmaps returns `width: 0` or a stringy `"0"`, `image_size` may pass the truthiness guard (e.g. non-empty string `"0"` is truthy → `int("0")` = 0) or be `None`. `rumsey.download_georeferenced:331` guards `not image_size` but `scale_gcps` then computes `sx = fetched_w / orig_w` → `ZeroDivisionError`, caught by the per-plate `except Exception` and miscounted as `download_failed` rather than a data-quality drop. The georeferenced output for a near-zero dimension would also be garbage.
**Fix:** In `_parse_annotation`, require `int(width) > 0 and int(height) > 0` (inside a try/except) before setting `image_size`; otherwise `None` so it is correctly classified `gcps_insufficient`.

### WR-06: `_read_rgb` global min/max normalization corrupts multi-band radiometry

**File:** `scripts/historical/label.py:76-92`
**Issue:** For non-uint8 GeoTIFFs, normalization uses a single global `lo, hi = bands.min(), bands.max()` across all three bands jointly. Per-band dynamic ranges differ (especially for satellite/DEM-derived RGB or 16-bit scans); a single global stretch shifts color balance and can collapse a band to near-constant. This is a correctness defect for the historical map image channel that Phase 3 trains on. Also `if hi > lo` else branch does `bands.astype(np.uint8)` on potentially out-of-range float/int16 data → silent wraparound.
**Fix:** Normalize per band (`axis=(1,2)` min/max with broadcasting), and clip to `[0,255]` before the `astype(np.uint8)` cast in the degenerate branch.

### WR-07: Unbounded remote raster open — no decompression-bomb / size guard on WorldCover & Sentinel COG reads

**File:** `scripts/historical/worldcover.py:119`; `scripts/satellite/fetch.py:74`; `scripts/satellite/coverage.py:121`
**Issue:** The IIIF path has an explicit 200 MB guard (T-02-04), but `rasterio.open(url)` on untrusted remote GeoTIFFs/COGs has no analogous protection. A maliciously crafted or corrupt COG with absurd `width`/`height` or block size can drive `ds.read([1,2,3], window=...)` / `ds.read(1, out_shape=...)` to allocate huge arrays. `fetch.py` reads a bounded window so it is partially mitigated, but `worldcover.fetch_worldcover` reprojects into `(map_ds.height, map_ds.width)` with no upper bound on the *map* dimensions (which for satellite come from the fetched window, OK; for historical come from the downloaded GeoTIFF, bounded by the 4096 IIIF cap — acceptable). The unguarded one is `coverage.build_summary` reading every global tile with a fixed `out_shape`, which is safe, but the WorldCover full-grid `dst = np.full((height, width), ...)` in `fetch_worldcover` trusts `map_ds` dimensions implicitly.
**Fix:** Add a defensive cap: reject/skip any opened remote dataset whose `ds.width * ds.height` exceeds a sane threshold, and bound the historical GeoTIFF dimensions before allocating `dst`.

### WR-08: `template_key` silently misclassifies common stems → split stratification skew

**File:** `scripts/build_dataset.py:85-97`
**Issue:** Verified behavior of the regex `^(.*?)[ _\-]*\d+$`:
- `m12` → `m` (the `1` is consumed as the numeric suffix start, leaving stem `m`)
- `plate2_v3` → `plate2_v` (only the final `3` stripped — internal digit kept)
- `12` / `007` → `12` / `007` (pure-numeric stem becomes its own stratum)
- `asia2024map` → `asia2024map` (no trailing digit → whole stem, fine)

For real Azgaar exports the trailing-index assumption is plausible, but stems like `world2`, `r12map`, or any numeric-suffixed continent name with internal digits will be grouped incorrectly, distorting the stratified hold-out proportions for EVAL-01. The phase notes this key is provisional, but there is no validation/warning emitted at split time showing the derived strata for human review before `split.json` is frozen.
**Fix:** Before freezing `split.json`, print the `{template_key: [member_ids]}` grouping so a human can sanity-check stratification; consider requiring a `_`/`-` separator before the numeric suffix (`^(.+?)[ _\-]+\d+$`) so `m12`/`world2` are not split mid-token.

### WR-09: `_to_wgs84_bbox` 4-corner envelope is wrong for projections that bow between corners / cross the antimeridian

**File:** `scripts/historical/label.py:54-73`
**Issue:** The WGS84 bbox is computed from only the 4 corners. For UTM/conic projections over large extents the reprojected edges bow outward, so the true min/max lon/lat lie on an *edge*, not a corner — the envelope under-covers, and the subsequent WorldCover/DEM tile enumeration misses border tiles, producing NODATA stripes in `land_cover.png`/`topography.png`. Antimeridian-crossing extents yield an inside-out bbox (min lon > max lon) that silently fetches the wrong tiles. Historical sources are written EPSG:4326 by `georef.write_georeferenced_geotiff` (so the `ds.crs == _WGS84` fast path dominates), but the manual-QGIS-export path explicitly supports arbitrary CRS, so this is reachable.
**Fix:** Densify the boundary (sample N points per edge, e.g. via `rasterio.warp.transform_bounds` with `densify_pts`) instead of transforming only 4 corners; handle the antimeridian (split the query) or at least detect and warn on `min_lon > max_lon`.

## Info

### IN-01: `enumerate_pyramids` comment references a non-existent `+1`

**File:** `scripts/tiling.py:58-60`
**Issue:** The comment says "+1 so an origin landing exactly at width/height ... is still considered" but the loops are `while y < height` / `while x < width` with no `+1`. An origin exactly at `width`/`height` is never generated (loop condition is strict `<`). The comment describes behavior the code does not implement; the behavior is arguably correct (a 0%-on pyramid is pointless) but the comment is misleading.
**Fix:** Delete or correct the comment to match the strict `<` bound.

### IN-02: `_richness_score` "City maps are too small-scale" comment contradicts the +2 it adds

**File:** `scripts/historical/rumsey.py:130`
**Issue:** `if _field(item, "City"): score += 2  # city maps are too small-scale` — the comment argues city maps are undesirable, yet the code rewards them with +2. Either the comment is stale or the scoring is a bug (likely should be 0 or negative). Affects which maps rank into the top `max_results`.
**Fix:** Reconcile intent — if city presence is undesirable, remove the bonus; otherwise fix the comment.

### IN-03: `rumsey._field(item, "Title")` will never match the LUNA fixture / documented shape

**File:** `scripts/historical/rumsey.py:304, 405`
**Issue:** `title = _field(map_meta, "Title") or _field(map_meta, "title") or item_id`. Per the module docstring and `conftest.sample_luna_item`, LUNA exposes `Short Title` / `Full Title` in `fieldValues` and `displayName` at top level — there is no `Title` / `title` field. So `title` always falls back to `item_id` for log lines and the v2 manifest, losing the human-readable title that the manifest is meant to carry for downstream manual georeferencing.
**Fix:** Use `_field(item, "Short Title") or _field(item, "Full Title") or item.get("displayName") or item_id`.

### IN-04: Duplicated haversine implementation across modules

**File:** `scripts/historical/allmaps.py:163-176` and `scripts/historical/rumsey.py:77-88`
**Issue:** `haversine_km`/`bbox_diagonal_km` (allmaps) and `_haversine_km`/`_bbox_diagonal_km` (rumsey) are byte-for-byte the same math. `rumsey.download_georeferenced` already calls `allmaps.bbox_diagonal_km`, so the private copies in `rumsey.py` are dead code. Divergence risk if one is later tuned.
**Fix:** Delete `rumsey._haversine_km`/`rumsey._bbox_diagonal_km` and use the `allmaps` ones everywhere.

### IN-05: `bbox_diagonal_km` only measures the SW→NE diagonal, ignoring aspect

**File:** `scripts/historical/allmaps.py:173-176`
**Issue:** The "scale window" filter ([100, 2000] km, D-04) uses only `haversine(west, south, east, north)`. A very wide, very short map (e.g. an equatorial strip) has a large diagonal but tiny N–S extent; a tall narrow map likewise. Diagonal alone is a coarse proxy and can admit/reject maps inconsistently with the intended scale band. Not a correctness bug per the locked spec, but worth noting since it directly governs how much data the historical pipeline keeps (D-05 loudness concern).
**Fix:** Consider gating on both the diagonal and the max edge, or document explicitly that diagonal-only is the accepted heuristic.

---

_Reviewed: 2026-05-15T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
