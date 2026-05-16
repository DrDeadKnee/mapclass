---
phase: 02-build-a-dataset-of-pixel-label-pairs
fixed_at: 2026-05-15T00:00:00Z
review_path: .planning/phases/02-build-a-dataset-of-pixel-label-pairs/02-REVIEW.md
iteration: 2
findings_in_scope: 3
fixed: 3
skipped: 0
status: all_fixed
---

# Phase 2: Code Review Fix Report (Iteration 2)

**Fixed at:** 2026-05-15T00:00:00Z
**Source review:** .planning/phases/02-build-a-dataset-of-pixel-label-pairs/02-REVIEW.md
**Iteration:** 2

**Summary:**
- Findings in scope: 3 (CR-04 BLOCKER, WR-10, WR-11)
- Fixed: 3
- Skipped: 0
- IN-06 is INFO and explicitly out of scope (not addressed, not regressed).

**Test suite:** `python3 -m pytest tests/ -q --ignore=tests/integration` —
57 passed (baseline 51; +6 new regression tests; no tests weakened).

## Fixed Issues

### CR-04: CR-03 fix is incomplete — a 3-element coordinate still aborts the whole source

**Files modified:** `scripts/label.py`, `scripts/render.py`, `tests/test_render.py`
**Commit:** b7b330d
**Applied fix:** Replaced the `for x, y in ring` tuple-unpack in both
rasterization draw loops (`label.make_label_arrays` line ~127 and
`render.render_style` line ~159) with
`[(pt[0] - min_x, pt[1] - min_y) for pt in ring if len(pt) >= 2]`,
mirroring the hardened `_bbox`. RFC 7946 3-element `[lon, lat, elevation]`
positions no longer raise `ValueError: too many values to unpack` and
abort the entire source. Additionally hardened `_rings` in both files: a
malformed `Polygon` whose `coordinates` is a non-list scalar/dict is now
rejected (returns `[]`) instead of deferring the crash to the caller's
draw loop. Added three regression tests: 3D coords rasterize in
`make_label_arrays` and `render_one`, and a malformed scalar-coords
Polygon is skipped while a valid sibling feature still rasterizes
(per-source resilience contract T-02-09).

### WR-10: WR-09 antimeridian fix detects but does not prevent the all-NODATA corruption

**Files modified:** `scripts/historical/label.py`, `tests/test_label.py`
**Commit:** 1497ca7
**Applied fix:** Per WR-10 option (b): `_to_wgs84_bbox` now raises
`ValueError` on the inside-out `west > east` antimeridian bbox instead of
printing a warning and returning the corrupt bbox. `_process_one` in
`build_historical_dataset.py` catches `Exception` and drop-counts the
source (D-13), converting a silently all-NODATA `land_cover.png` /
`topography.png` pair into an accounted, loud drop. Added regression
tests: an antimeridian reprojection raises `ValueError`, and a
well-formed reprojected bbox is still returned intact.

### WR-11: `_read_rgb` per-band normalization does not sanitize NaN/Inf in float source rasters

**Files modified:** `scripts/historical/label.py`, `tests/test_label.py`
**Commit:** 5b0ffd7
**Applied fix:** Inserted
`np.nan_to_num(bands.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)`
before the per-band `lo`/`hi` min/max reduction in `_read_rgb`. Previously
NaN propagated across the whole band through `min`/`max`, survived
`np.clip`, and produced platform-undefined bytes after
`.astype(np.uint8)`. Added a regression test feeding a float band stack
containing `nan`, `+inf`, and `-inf` and asserting the resulting uint8
array is finite and the finite values are still stretched.

## Skipped Issues

None — all in-scope findings were fixed.

---

_Fixed: 2026-05-15T00:00:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 2_
