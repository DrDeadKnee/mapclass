---
phase: 02-build-a-dataset-of-pixel-label-pairs
fixed_at: 2026-05-15T00:00:00Z
review_path: .planning/phases/02-build-a-dataset-of-pixel-label-pairs/02-REVIEW.md
iteration: 1
findings_in_scope: 12
fixed: 12
skipped: 0
status: all_fixed
---

# Phase 2: Code Review Fix Report

**Fixed at:** 2026-05-15T00:00:00Z
**Source review:** .planning/phases/02-build-a-dataset-of-pixel-label-pairs/02-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 12 (3 BLOCKER + 9 WARNING; 5 INFO out of scope)
- Fixed: 12
- Skipped: 0

The full offline test suite (`python3 -m pytest tests/ -q --ignore=tests/integration`)
passes after every fix: baseline 50 passed → 51 passed (one test added for
CR-02; CR-01's prefix-only assertion was strengthened to a full-range
assertion — no test was weakened to pass).

## Fixed Issues

### CR-01: Satellite seasonal datetime windows are wrong

**Files modified:** `scripts/satellite/coverage.py`, `tests/test_coverage.py`
**Commit:** 96b768b
**Applied fix:** `season_for_latitude` now returns a correct trailing-12-month
tropics window (`2023-01-01/2023-12-31`, was a 24-month
`2022-01-01/2023-12-31` with dead `{1:02d}` formatting) and one contiguous
austral summer for the southern band (`2022-12-01/2023-02-28`, was the
2-month-longer split-season `2022-11-01/2023-03-31`). `test_season_varies_by_hemisphere`
was strengthened from prefix-only (`startswith`) to exact full-range string
assertions for all three bands — this is the assertion gap that originally
masked the bug.

### CR-02: Source-ID sanitization collisions silently drop maps / corrupt the frozen split

**Files modified:** `scripts/build_dataset.py`, `tests/test_split.py`
**Commit:** 1a6b778
**Applied fix:** `build()` now detects post-`_sanitize_stem` collisions across
the raw `.geojson` set and `sys.exit(1)`s with the colliding sanitized IDs
and their raw filenames BEFORE the split is computed/frozen — protecting the
EVAL-01 split from cross-boundary leakage and silent overwrites. Added
`test_build_aborts_on_sanitization_collision` (uses a genuine colliding pair
`map.v2` / `map v2`, both → `map_v2`) asserting `SystemExit(1)` and that no
`split.json` is written.

### CR-03: Malformed Azgaar GeoJSON crashes the whole source instead of per-source resilience

**Files modified:** `scripts/label.py`, `scripts/render.py`, `scripts/biome_mapping.py`
**Commit:** 8705573
**Applied fix:** `_rings` in both `label.py` and `render.py` now returns `[]`
for non-Polygon/MultiPolygon, missing, or malformed geometry instead of
indexing blindly. `_bbox` skips unusable geometries and raises a clear
`ValueError` only when no feature has a usable polygon. Per-feature
`biome`/`height` reads are wrapped in `try/except (KeyError, TypeError,
ValueError)` and skip the bad feature rather than aborting the source.
`h_to_landcover` falls back to `bare_sparse` for unknown biome IDs. Added a
`_MAX_CANVAS_DIM` (20000 px) decompression-bomb-analogue guard plus a
degenerate-canvas guard in both `label.make_label_arrays` and the
`render` width/height path.
**Note:** Introduces per-feature skip/fallback logic — flagged for human
verification (the skip semantics and `bare_sparse` fallback choice are
judgement calls).

### WR-01: `assert` used as a runtime data-integrity check

**Files modified:** `scripts/tiling.py`
**Commit:** d1a8000
**Applied fix:** Replaced `assert (pdir / _WEIGHTS_FILE).read_bytes() ==
weights_blob` with an explicit `if ... raise RuntimeError(...)` so the
weight-propagation integrity check survives `python -O`.

### WR-02: Resume path returns a possibly partial/zero `source.tif` without re-validation

**Files modified:** `scripts/historical/rumsey.py`
**Commit:** a1b25d9
**Applied fix:** Added `_is_valid_geotiff` (lazy `rasterio` import; falls back
to a nonzero-size check if rasterio is unavailable) that verifies an existing
`source.tif` opens with a CRS and nonzero dimensions. On the resume path an
invalid file is deleted and re-fetched instead of being silently accepted.

### WR-03: Silent miscounting of "no visual asset" as "no qualifying scene"

**Files modified:** `scripts/build_satellite_dataset.py`
**Commit:** 8c2c495
**Applied fix:** Added a distinct `missing_visual_asset` key to the `drops`
dict and routed the no-`visual`-asset case to it, so the D-13 per-reason
accounting no longer conflates it with `no_qualifying_scene`. The generic
`_print_drop_summary` surfaces the new key automatically.

### WR-04: Broad `except Exception` retries deterministic STAC failures

**Files modified:** `scripts/satellite/stac.py`
**Commit:** 3a6be63
**Applied fix:** Added a preceding `except (TypeError, ValueError,
AttributeError): raise` so deterministic programming/bad-input errors
propagate immediately without the retry/backoff loop; only transient
network/server failures are retried.

### WR-05: GCP scaling divides by `orig_w`/`orig_h` with no zero/None guard

**Files modified:** `scripts/historical/allmaps.py`
**Commit:** cc21109
**Applied fix:** `_parse_annotation` now coerces `width`/`height` inside a
`try/except (TypeError, ValueError)` and requires strictly-positive
integers (`iw > 0 and ih > 0`) before setting `image_size`; a truthy-but-zero
value (e.g. `"0"`) now correctly yields `None` so the annotation is
classified `gcps_insufficient` rather than later causing a
`ZeroDivisionError` miscounted as `download_failed`.

### WR-06: `_read_rgb` global min/max normalization corrupts multi-band radiometry

**Files modified:** `scripts/historical/label.py`
**Commit:** 9c3a5d7
**Applied fix:** Non-uint8 GeoTIFF normalization is now per-band (min/max
over `axis=(1, 2)` with broadcasting) and the result is `np.clip`-ed to
`[0, 255]` before the `uint8` cast, eliminating the cross-band colour shift
and the silent wraparound in the degenerate (`hi == lo`) branch. Verified
numerically that bands normalize independently and a constant band maps to 0.
**Note:** Radiometric correctness change — flagged for human verification.

### WR-07: Unbounded remote raster open — no decompression-bomb / size guard

**Files modified:** `scripts/historical/worldcover.py`, `scripts/satellite/coverage.py`, `scripts/satellite/fetch.py`
**Commit:** e08541b
**Applied fix:** Added `MAX_REMOTE_RASTER_PIXELS` (2e9 px) and a shared
`assert_safe_raster_size(ds, source)` helper in `worldcover.py`, called
immediately after `rasterio.open(url)` at all three untrusted-remote-open
sites (WorldCover tile read in `worldcover.fetch_worldcover` and
`coverage.build_summary`; Sentinel COG read in `fetch.fetch_visual_window`).
Each site's existing broad `except` / `None`-return path degrades the guard
trip to a skipped tile / drop-counted fetch, not a crash.

### WR-08: `template_key` silently misclassifies common stems → split skew

**Files modified:** `scripts/build_dataset.py`
**Commit:** 62e3e47
**Applied fix:** Tightened the suffix regex from `^(.*?)[ _\-]*\d+$` to
`^(.+?)[ _\-]+\d+$` so a separator (`_`/`-`/space) is REQUIRED before the
numeric suffix — `m12`/`world2`/`r12map` are no longer split mid-token
(verified: they now map to themselves; `europe_07`/`europe-7` still →
`europe`). `load_or_create_split` now prints the derived
`{template: [member_ids]}` grouping before freezing `split.json` for human
sanity-check.
**Note:** Stratification-key behaviour change — flagged for human
verification (affects EVAL-01 hold-out grouping for real Azgaar stems).

### WR-09: `_to_wgs84_bbox` 4-corner envelope wrong for bowing / antimeridian

**Files modified:** `scripts/historical/label.py`
**Commit:** fac2efe
**Applied fix:** Replaced the manual 4-corner `pyproj.Transformer` envelope
with `rasterio.warp.transform_bounds(..., densify_pts=21)` so bowing
UTM/conic edges are covered (no NODATA border stripes), and added a loud
`west > east` antimeridian-crossing warning. Removed the now-unused
`pyproj.Transformer` import.
**Note:** bbox-computation correctness change — flagged for human
verification (densify count and antimeridian handling are judgement calls).

---

_Fixed: 2026-05-15T00:00:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
