---
phase: 02-build-a-dataset-of-pixel-label-pairs
reviewed: 2026-05-15T00:00:00Z
depth: standard
iteration: 2
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
  critical: 1
  warning: 2
  info: 1
  total: 4
status: issues_found
---

# Phase 2: Code Review Report (Iteration 2)

**Reviewed:** 2026-05-15T00:00:00Z
**Depth:** standard
**Files Reviewed:** 38 (re-review verifying 12 prior fixes across 96b768b..fac2efe)
**Status:** issues_found

## Summary

This is the iteration-2 adversarial re-review. The 12 prior findings (3 BLOCKER + 9 WARNING) were each addressed in a dedicated commit. The full offline suite passes (51/51, up from 50 — one regression test added for CR-02, and `test_season_varies_by_hemisphere` was correctly *strengthened* from prefix-only to full-range assertions, not weakened).

**Verification result — 11 of 12 fixes are correct and regression-free:**

- **CR-01** (seasonal windows) — correctly fixed. Tropics is now exactly `2023-01-01/2023-12-31` (12 months, dead `{1:02d}` removed); southern is one contiguous `2022-12-01/2023-02-28`. The test was strengthened to exact-range assertions, closing the gap that masked the original bug. Verified by reading `coverage.py:67-87`.
- **CR-02** (sanitization collisions) — correctly fixed. Collision detection runs and `sys.exit(1)`s before `load_or_create_split`, so the frozen split is protected. `sys` is imported; a genuine-collision regression test asserts `SystemExit(1)` and absence of `split.json`.
- **WR-01** (assert → RuntimeError) — correct. Survives `python -O`.
- **WR-02** (resume re-validation) — correct. `_is_valid_geotiff` opens with rasterio (CRS + nonzero dims), deletes-and-refetches on failure; the rasterio-absent fallback (`st_size > 0`) is weak but acknowledged and acceptable for offline test envs.
- **WR-03** (distinct drop reason) — correct. `missing_visual_asset` is a separate `drops` key surfaced by the generic summary.
- **WR-04** (no retry on deterministic STAC failure) — correct. Narrow `except (TypeError, ValueError, AttributeError): raise` precedes the broad transient handler.
- **WR-05** (zero-dimension guard) — correct. `int()` coercion in try/except with strict `iw > 0 and ih > 0`; `"0"` now yields `None` → `gcps_insufficient`.
- **WR-06** (per-band normalization) — numerically verified correct. `np.where(span > 0, span, 1)` divisor prevents div-by-zero; constant band → 0; `np.clip(...,0,255)` before the uint8 cast eliminates wraparound.
- **WR-07** (decompression-bomb guard) — correct. `assert_safe_raster_size` is called immediately after every untrusted `rasterio.open(url)` (WorldCover ×2, Sentinel COG ×1); each call site degrades a trip to a skip/drop, not a crash.
- **WR-08** (require separator before numeric suffix) — correct. `^(.+?)[ _\-]+\d+$` no longer splits `m12`/`world2`/`r12map` mid-token; the derived strata are printed before `split.json` freezes.

**However, the adversarial pass found that the CR-03 fix is incomplete and still reachable**, plus the WR-09 fix detects-but-does-not-prevent the antimeridian data corruption it was filed against. Both are documented below with reproductions.

## Critical Issues

### CR-04: CR-03 fix is incomplete — a 3-element coordinate still aborts the whole source (regression of the original CR-03 contract)

**File:** `scripts/label.py:127`; `scripts/render.py:159`
**Issue:** The CR-03 fix hardened the *bbox* path (`_bbox` now uses `pt[0]`/`pt[1]` behind a `len(pt) >= 2` guard) but left the *rasterization* path inconsistent. Both `label.make_label_arrays` and `render.render_style` still do:

```python
coords = [(x - min_x, y - min_y) for x, y in ring]
```

GeoJSON (RFC 7946) explicitly permits a third position element (elevation): `[lon, lat, elevation]`. Azgaar/QGIS exports and many real-world GeoJSON files carry 3-element positions. Such a feature passes the hardened `_bbox` cleanly, then crashes the tuple-unpack `for x, y in ring` with an unhandled `ValueError: too many values to unpack (expected 2)`. Because `build_dataset.build_one_source` only catches at the per-source level, **one feature with elevation coordinates aborts the entire map** — which is precisely the "any one malformed feature drops the entire map" failure mode CR-03 was filed to eliminate. The fix moved the crash from `_bbox` to the draw loop rather than removing it.

Reproduced directly:
```
feat geometry coordinates = [[[0,0,5],[10,0,5],[10,10,5],[0,0,5]]]   # valid 3D GeoJSON
label.make_label_arrays(...) -> ValueError: too many values to unpack (expected 2)
```

This is BLOCKER-class: it is silent data loss on legitimate input (not malicious), it defeats the documented T-02-09 per-source resilience contract, and no test exercises 3-element positions (all fixtures use 2-element coords, which is why the suite stayed green).

**Fix:** Slice each point to its first two ordinates in the rasterization loop, mirroring the `_bbox` hardening:
```python
coords = [(pt[0] - min_x, pt[1] - min_y) for pt in ring if len(pt) >= 2]
if len(coords) < 3:
    continue
```
Apply identically in `label.py:127` and `render.py:159`. Add a fixture/test with 3-element coordinates asserting the source rasterizes (does not raise). Also consider hardening `_rings` for the malformed-`Polygon` case: when `gtype == "Polygon"` and `coords` is a non-list scalar/dict, `_rings` returns it as-is and defers the same crash to the caller — the `try/except TypeError` only guards the MultiPolygon comprehension, not a malformed Polygon.

## Warnings

### WR-10: WR-09 antimeridian "fix" detects but does not prevent the all-NODATA corruption

**File:** `scripts/historical/label.py:58-83`; `scripts/historical/worldcover.py:85-98`
**Issue:** The WR-09 fix correctly replaced the 4-corner envelope with densified `transform_bounds(..., densify_pts=21)` (this part is sound and fixes the bowing-edge under-coverage). But for the antimeridian-crossing case it only prints a warning and *still returns the inside-out bbox* (`west > east`). Downstream, `worldcover._tile_origins` does `lon = lon0; while lon < east` starting from `lon0 = floor(west/3)*3` where `west > east` — the inner loop body never executes, so **zero tiles are yielded and `land_cover.png`/`topography.png` are entirely NODATA** for that source. The original WR-09 ask was to "handle the antimeridian (split the query) **or at least detect and warn**", so warn-only is the documented-acceptable minimum and this is correctly downgraded from BLOCKER to WARNING — but reviewers should know the data corruption itself is *not* fixed, only made observable in logs. An antimeridian source still silently produces a fully-blank label pair that will be ingested by Phase 3 unless a human reads the warning.
**Fix:** Either (a) split the query at ±180 and union the two tile sets in `_tile_origins`, or (b) make `_to_wgs84_bbox` raise on `west > east` so `build_one_source` drop-counts the source loudly (D-13) instead of emitting a blank label pair. Option (b) is the minimal change that converts silent corruption into an accounted drop.

### WR-11: `_read_rgb` per-band normalization does not sanitize NaN/Inf in float source rasters

**File:** `scripts/historical/label.py:96-101`
**Issue:** The WR-06 fix is correct for finite data, but float GeoTIFFs (16-bit-scaled scans, DEM-derived RGB) can carry NaN/Inf nodata. `f.min(axis=(1,2))`/`f.max(...)` propagate NaN across the whole band; `np.clip(nan, 0, 255)` returns `nan`; `nan.astype(np.uint8)` is platform-dependent (commonly 0, but undefined). The result is a silently corrupted image channel that Phase 3 trains on. This is a pre-existing latent issue the WR-06 rework did not introduce, but the rework was the right place to also handle it and did not, so it is in scope for this re-review.
**Fix:** Mask non-finite values before computing per-band min/max, e.g. `f = np.nan_to_num(bands.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)` before the `lo/hi` reduction, or use `np.nanmin`/`np.nanmax` and then `np.nan_to_num` the scaled array.

## Info

### IN-06: Dead `space` alternative in the WR-08 separator class; prior INFO items (IN-01..IN-05) remain unaddressed

**File:** `scripts/build_dataset.py:99`
**Issue:** `_sanitize_stem` (applied at the top of `template_key`) already converts every space to `_` via `re.sub(r"[^\w-]", "_", ...)`, so the space branch in `^(.+?)[ _\-]+\d+$` can never match — it is dead. Harmless, but the docstring claiming "A separator (`_`/`-`/space)" overstates what is reachable. Separately, the 5 INFO findings from iteration 1 (IN-01 misleading `enumerate_pyramids` comment, IN-02 city-map score/comment contradiction, IN-03 `_field("Title")` never matches LUNA, IN-04 duplicated haversine, IN-05 diagonal-only scale gate) were explicitly out of fix scope and remain present — re-flagging only that they were not regressed, not that they were fixed.
**Fix:** Drop the space from the character class (or keep it but correct the docstring to note spaces are pre-sanitized to `_`). IN-01..IN-05 carry forward unchanged.

---

_Reviewed: 2026-05-15T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
_Iteration: 2_
