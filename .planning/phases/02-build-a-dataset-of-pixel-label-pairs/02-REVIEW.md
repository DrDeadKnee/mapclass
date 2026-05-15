---
phase: 02-build-a-dataset-of-pixel-label-pairs
reviewed: 2026-05-15T16:54:29Z
depth: standard
iteration: 3
files_reviewed: 29
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
  warning: 3
  info: 2
  total: 6
status: issues_found
---

# Phase 2: Code Review Report

**Reviewed:** 2026-05-15T16:54:29Z
**Depth:** standard
**Files Reviewed:** 29
**Status:** issues_found

## Summary

Final re-review (iteration 3) of the Phase 2 dataset-build pipeline, focused on
verifying the iteration-2 fixes and re-auditing the per-source-resilience crash
class.

**Iteration-2 fix verification — all three land correctly and are
regression-free at the unit level (`tests/test_render.py` +
`tests/test_label.py`: 11 passed):**

- **CR-04** (b7b330d) — `scripts/label.py` / `scripts/render.py`: the draw
  loops now slice `pt[0]/pt[1]` under a `len(pt) >= 2` guard, and `_rings`
  rejects a non-list/tuple Polygon `coordinates`. 3-element `[lon,lat,elev]`
  positions and scalar Polygon coords are correctly handled. The grep audit
  confirms **no remaining `for x, y in ring` tuple-unpacks** anywhere in the
  draw/ring/bbox paths. Fix is correct *as far as it goes*.
- **WR-10** (1497ca7) — `scripts/historical/label.py:74-87`: the inside-out
  antimeridian bbox now `raise`s `ValueError` instead of returning a corrupt
  bbox that yielded an all-NODATA pair. `_process_one` / `build_one_source`
  drop-count it loudly (D-13). Correct; regression tests cover both the raise
  and the normal-path return.
- **WR-11** (5b0ffd7) — `scripts/historical/label.py:108-110`: `np.nan_to_num`
  is applied to the float band stack *before* the per-band min/max reduction,
  so NaN/Inf no longer poisons the stretch or produces platform-undefined
  uint8. Correct; regression test asserts finiteness and a real stretch range.

**However, the re-review's central question — "does CR-04's fix fully eliminate
the per-source-resilience crash class (no remaining unguarded coordinate
unpacks in label.py/render.py draw or ring paths)?" — the answer is NO.** CR-04
swapped the `for x, y in ring` unpack for a `len(pt) >= 2` filter, but the
filter itself is not crash-safe: a ring containing a **scalar element**
(`[[0,0], 5, [40,0], ...]`) makes `len(5)` raise `TypeError`, and a
**non-numeric string element** (`[[0,0], "ab", ...]`) passes the `len` guard
then crashes the arithmetic / `min()` reduction. Neither is caught (the
`try/except TypeError` lives only inside `_rings`, not in the draw loop or
`_bbox`), so a single malformed ring element still aborts the **entire
source** — precisely the T-02-09 crash class CR-04 was meant to close.
Reproduced empirically against both `label.make_label_arrays` and
`render.render_one` (see CR-01). This is a BLOCKER: the fix is incomplete, not
regression-free against the threat it targets.

## Critical Issues

### CR-01: CR-04 fix is incomplete — malformed ring *elements* still abort the entire source (T-02-09 crash class not fully eliminated)

**File:** `scripts/label.py:78-80,136` and `scripts/render.py:137-139,168` (and `scripts/label.py:64-83` / `scripts/render.py:129-142` `_bbox`)

**Issue:** CR-04 replaced `for x, y in ring` with
`[(pt[0]-min_x, pt[1]-min_y) for pt in ring if len(pt) >= 2]`. The
`len(pt) >= 2` guard is not itself crash-safe, and the `try/except TypeError`
that protects `_rings` does **not** cover the draw loop or `_bbox`:

1. **Scalar element in a ring** — `coordinates: [[[0,0], 5, [40,0], [40,40], [0,40]]]`.
   `len(5)` raises `TypeError: object of type 'int' has no len()`. Uncaught →
   aborts the whole source. Reproduced against `make_label_arrays` *and*
   `render_one`:
   ```
   CRASH: TypeError object of type 'int' has no len()        # label.make_label_arrays
   render_one(flat) CRASH: TypeError object of type 'int' has no len()
   ```
2. **Non-numeric string element** — `coordinates: [[[0,0], "ab", [40,0], ...]]`.
   `len("ab") >= 2` is `True`, so it survives the filter; then
   `"ab"[0] - min_x` (draw loop) or the `min()/max()` reduction in `_bbox`
   crashes:
   ```
   string-elem CRASH: TypeError '<' not supported between instances of 'str' and 'int'
   ```
   (`_bbox` appends `pt[0]='a'` to `xs`, then `min(xs)` raises.)

This is the same RFC-7946-malformed-input / per-source-resilience failure mode
CR-03 and CR-04 were chartered to eliminate (one bad feature must never abort a
batch). The fix narrowed the crash surface (3-element coords, scalar Polygon
`coordinates`) but left two reachable variants that still abort the source.

**Fix:** Validate each point's structure *and* numeric type inside the
comprehension, in both the draw loop and `_bbox`, in both files. Centralising
in a helper avoids the four-site drift:

```python
def _xy(pt):
    """Return (x, y) floats for a GeoJSON position, or None if malformed."""
    if not isinstance(pt, (list, tuple)) or len(pt) < 2:
        return None
    try:
        return float(pt[0]), float(pt[1])
    except (TypeError, ValueError):
        return None
```

Draw loop (label.py:136 / render.py:168):
```python
coords = []
for pt in ring:
    xy = _xy(pt)
    if xy is not None:
        coords.append((xy[0] - min_x, xy[1] - min_y))
if len(coords) < 3:
    continue
```

`_bbox` (label.py:77-80 / render.py:136-139):
```python
for pt in ring:
    xy = _xy(pt)
    if xy is not None:
        xs.append(xy[0]); ys.append(xy[1])
```

Add regression tests for (a) a scalar element inside an otherwise-valid ring
and (b) a non-numeric string element, asserting a valid sibling feature still
rasterizes (the T-02-09 contract the existing
`test_malformed_polygon_scalar_coords_skipped` only partially exercises).

## Warnings

### WR-01: `_bbox` accepts a string point that passes `len(pt) >= 2`, corrupting the canvas bounds before any draw

**File:** `scripts/label.py:77-83`, `scripts/render.py:136-142`

**Issue:** Distinct from CR-01's crash: even when the eventual `min()/max()`
does not raise (e.g. a ring of *all* equal-length strings, or a mix that
happens to compare), `len(pt) >= 2` admits any 2+-char string or 2+-element
non-coordinate sequence into `xs`/`ys`. The bbox — and therefore the inferred
canvas width/height shared across every render style and both label masks — is
computed from garbage, silently producing a mis-sized or degenerate canvas
rather than skipping the bad ring. The `_xy` helper from CR-01 fixes this at
the same site; calling it out separately because it is a *data-corruption*
path independent of the crash and must be covered by its own assertion (bbox
unchanged by an injected malformed point).

### WR-02: `season_for_latitude` southern-temperate window silently truncates leap-year February

**File:** `scripts/satellite/coverage.py:85`

**Issue:** `f"{ref_year - 1}-12-01/{ref_year}-02-28"` hard-codes Feb-28 as the
range end. For any `ref_year` that is a leap year (default `ref_year=2023` is
not, but the parameter is public and 2024/2028/... are common), the austral
summer window silently drops Feb 29 — one day of the lowest-cloud season is
excluded for every southern-temperate region, biasing scene selection. This is
a latent correctness defect the moment a caller passes a leap `ref_year`.

**Fix:** Use the last day of February for the actual year, e.g.
`import calendar; end_day = calendar.monthrange(ref_year, 2)[1]` and format
`f"{ref_year}-02-{end_day:02d}"`, or extend the window to `03-01` (a day of
margin in a seasonal pick is harmless).

### WR-03: `satellite/fetch.py` discards the reproject coverage — partial-coverage windows write opaque-black NODATA into the training channel

**File:** `scripts/satellite/fetch.py:93-103`

**Issue:** `rgb_wgs84 = np.zeros((3, dst_h, dst_w), uint8)` then `reproject`
without `dst_nodata` / `src_nodata`. When the UTM→WGS84 warp of an edge or
rotated window leaves destination pixels uncovered (common for non-axis-aligned
UTM scenes), those pixels stay `0` (pure black) and are written into
`image.png` indistinguishably from real dark imagery. Phase 3 then trains the
model to associate a black wedge with whatever WorldCover land cover underlies
that corner — a silent label/imagery mismatch. The historical path carries an
explicit water/NODATA sentinel discipline; the satellite path drops it here.

**Fix:** Pass `src_nodata`/`dst_nodata` (or capture the reproject coverage
mask) and either crop `rgb_wgs84` to the fully-covered interior or record the
uncovered region so downstream tiling can exclude it, mirroring the NODATA
discipline the rest of the pipeline maintains.

## Info

### IN-01: `_read_rgb` two-band branch fabricates a third channel from the blue band

**File:** `scripts/historical/label.py:119-120`

**Issue:** For a 2-band source, `arr = np.concatenate([arr, arr[:, :, :1]], axis=2)`
duplicates band-1 as the third channel, yielding an (R=b1, G=b2, B=b1) image —
an arbitrary, undocumented colour fabrication for an unusual input. Not a
correctness blocker (2-band georeferenced map scans are rare), but the intent
should be documented or the case rejected explicitly rather than silently
producing a false-colour `image.png`.

**Fix:** Add a comment justifying the channel choice, or raise a clear
`ValueError` for `ds.count == 2` so the source is drop-counted rather than
silently false-coloured.

### IN-02: Obfuscated unpack `_, _, h, w = 1, 1, *rgb.shape[1:]`

**File:** `scripts/satellite/fetch.py:88`

**Issue:** Two throwaway `1`s padding the unpack purely so the statement
parses; `h, w = rgb.shape[1:]` is equivalent and readable. Pure clarity nit,
no behavioural impact.

**Fix:** `h, w = rgb.shape[1], rgb.shape[2]`.

---

_Reviewed: 2026-05-15T16:54:29Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
