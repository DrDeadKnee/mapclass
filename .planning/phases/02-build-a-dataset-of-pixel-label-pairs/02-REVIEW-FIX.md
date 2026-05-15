---
phase: 02-build-a-dataset-of-pixel-label-pairs
fixed_at: 2026-05-15T17:30:00Z
review_path: .planning/phases/02-build-a-dataset-of-pixel-label-pairs/02-REVIEW.md
iteration: 3
findings_in_scope: 4
fixed: 4
skipped: 0
status: all_fixed
cumulative_iterations: 3
cumulative_fixed: 19
cumulative_skipped: 0
---

# Phase 2: Code Review Fix Report (cumulative, iterations 1–3)

**Fixed at:** 2026-05-15T17:30:00Z
**Source review:** .planning/phases/02-build-a-dataset-of-pixel-label-pairs/02-REVIEW.md
**Iteration:** 3 of 3 (FINAL)

**Cumulative summary across all 3 iterations:**
- Iteration 1: 12 in scope (3 BLOCKER + 9 WARNING), 12 fixed, 0 skipped
- Iteration 2: 3 in scope (CR-04 BLOCKER, WR-10, WR-11), 3 fixed, 0 skipped
- Iteration 3: 4 in scope (CR-01 BLOCKER, WR-01, WR-02, WR-03), 4 fixed, 0 skipped
- **Total: 19 findings fixed, 0 skipped.** IN-* findings were INFO and out of
  scope every iteration (not addressed, not regressed).

**Test suite (iteration 3):** `python3 -m pytest tests/ -q --ignore=tests/integration`
— **62 passed** (iteration-2 baseline 57; +5 new regression tests this
iteration; no test weakened to pass). Each fix was committed atomically and
verified green before the next.

---

# Iteration 3 — Fixed Issues

The central iteration-3 finding (CR-01) was that three successive iterations
(CR-03 → CR-04 → CR-01) chased individual malformed-coordinate *element*
shapes (missing z, scalar, string) with piecemeal per-element guards, and each
pass found a new shape. Iteration 3 stops chasing element shapes and makes the
per-feature/per-ring processing **structurally resilient** instead.

### CR-01 (BLOCKER): malformed ring *elements* still abort the entire source — T-02-09 crash class not fully eliminated

**Files modified:** `scripts/label.py`, `scripts/render.py`, `tests/test_render.py`
**Commit:** 7bbfc9c
**Status:** fixed: requires human verification (structural change to the
per-source-resilience contract — recommend confirming the empirical scalar /
string / empty-ring repro from CR-01 no longer aborts a real batch)

**Applied fix:** Replaced the four drifting `len(pt) >= 2` element-shape
guards with a single structural chokepoint plus per-feature / per-ring
try/except in **both** `label.py` and `render.py`:

- Added `_xy(pt)` to both modules: returns `(float, float)` or `None`,
  rejecting non-sequences, short sequences, and non-numeric ordinates
  uniformly. This defensively coerces every coordinate and is the single
  place any malformed element shape is neutralised — no Nth element-shape
  patch is possible because no malformed element can reach the arithmetic /
  `min`/`max` path.
- Wrapped the per-feature and per-ring parse+rasterize blocks in
  `_bbox`, `make_label_arrays` (label.py) and `_bbox`, `render_style`
  (render.py) in `try/except (TypeError, ValueError, KeyError, IndexError)`
  so ANY malformed feature/ring — regardless of element shape — is skipped
  and drop-counted (T-02-09 / D-13) rather than aborting the whole source.
- The two reproduced crash variants from CR-01 (scalar element making
  `len(5)` raise; non-numeric string element surviving the old `len` guard
  then crashing the arithmetic / `min()` reduction) are both eliminated by
  `_xy` at the same site, in both `_bbox` and the draw loop.

**Regression tests added** (`tests/test_render.py`):
`test_mixed_malformed_rings_do_not_abort_label` and
`test_mixed_malformed_rings_do_not_abort_render` feed a feature collection
containing a scalar element, a non-numeric string element, an empty ring, and
a valid 2D sibling — asserting the source still produces output and the valid
sibling rasterizes despite all the malformed siblings.

### WR-01: `_bbox` accepts a string point that passes `len(pt) >= 2`, corrupting canvas bounds

**Files modified:** `scripts/label.py`, `scripts/render.py`, `tests/test_render.py`
**Commit:** 7bbfc9c (same atomic commit as CR-01 — the `_xy` helper fixes
both the crash and the data-corruption path at the same site)
**Applied fix:** `_xy` rejects any non-numeric / short / scalar element
before it can reach `xs`/`ys`, so a malformed point can no longer be folded
into the `min`/`max` that define the shared canvas width/height. This is a
data-corruption path independent of the crash and is covered by its own
assertion.

**Regression test added:** `test_malformed_point_does_not_corrupt_bbox`
asserts the canvas size of a clean ring is byte-identical to the same ring
with scalar / string garbage points spliced in (bad points rejected by
`_xy`, not folded into the bounds).

### WR-02: `season_for_latitude` southern-temperate window silently truncates leap-year February

**Files modified:** `scripts/satellite/coverage.py`, `tests/test_coverage.py`
**Commit:** 1f0c8ad
**Applied fix:** Replaced the hard-coded `02-28` end with a leap-year-correct
last-day-of-February computed from `calendar.monthrange(ref_year, 2)[1]`
(added `import calendar`). A leap `ref_year` now keeps Feb 29 in the
austral-summer window; a non-leap year still ends `02-28`.

**Regression test added:** `test_southern_window_is_leap_year_correct`
asserts `ref_year=2024` → `2023-12-01/2024-02-29` and `ref_year=2023` →
`2022-12-01/2023-02-28`.

### WR-03: satellite reproject discards coverage — partial-coverage windows write opaque-black NODATA into the training channel

**Files modified:** `scripts/satellite/fetch.py`,
`scripts/historical/georef.py`, `tests/test_satellite.py`
**Commit:** 6993d2d
**Applied fix:** Mirrored the rest of the pipeline's NODATA discipline in the
satellite path:

- `fetch.fetch_visual_window` now reprojects an all-ones coverage source
  alongside the RGB warp; destination pixels the warp did not touch are set
  to a `_NODATA = 255` sentinel (a dedicated coverage band is used rather
  than "pixel == 0" because real imagery may legitimately contain 0).
- Extended the SINGLE shared writer `write_georeferenced_geotiff` with an
  additive optional `nodata=` parameter (no forked writer — W-1 dedup
  preserved) and passed `nodata=_NODATA` so downstream tiling can exclude
  the uncovered region instead of training on a black wedge.

**Regression tests added/strengthened** (`tests/test_satellite.py`):
`test_fetch_visual_window_writes_wgs84_rgb_geotiff` now also asserts the
`nodata` tag is set; new `test_fetch_visual_window_marks_uncovered_pixels_nodata`
uses a meridian-rotated high-latitude UTM scene with source values bounded
`< _NODATA` so any sentinel pixel in the output is unambiguously a
reproject-uncovered pixel, asserting uncovered pixels carry the sentinel and
real imagery survives.

## Skipped Issues (iteration 3)

None — all four in-scope findings were fixed.

**Out of scope (INFO, not addressed by design):**
- IN-01: `_read_rgb` two-band branch fabricates a third channel (clarity nit)
- IN-02: obfuscated unpack `_, _, h, w = 1, 1, *rgb.shape[1:]` (clarity nit)

---

# Iterations 1–2 — Prior Fixes (carried forward)

## Iteration 1 (12 fixed)

3 BLOCKER + 9 WARNING fixed. Highlights: CR-01 corrected the satellite
seasonal datetime windows (24-month tropics window and split austral season →
correct trailing-12-month / one-contiguous-summer ranges; assertion
strengthened from `startswith` prefix to exact full-range). All 12 fixes
committed atomically; suite went 50 → 51 passed (one test added, none
weakened). See `02-REVIEW-FIX.iter2.md` for the full iteration-1 detail block.

## Iteration 2 (3 fixed)

- **CR-04** (b7b330d) — replaced the `for x, y in ring` tuple-unpack in both
  rasterization draw loops with a `len(pt) >= 2` filter and hardened `_rings`
  to reject a non-list Polygon `coordinates`. (Iteration 3 supersedes the
  `len`-filter approach with the structural `_xy` chokepoint.)
- **WR-10** (1497ca7) — the inside-out antimeridian bbox now raises
  `ValueError` (accounted drop, D-13) instead of returning a corrupt bbox
  that yielded an all-NODATA pair.
- **WR-11** (5b0ffd7) — `np.nan_to_num` applied to the float band stack
  before the per-band min/max reduction so NaN/Inf no longer poisons the
  stretch or produces platform-undefined uint8.

Suite: 51 → 57 passed (+6 regression tests, none weakened).

---

_Fixed: 2026-05-15T17:30:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 3 of 3 (FINAL) — cumulative report across all iterations_
