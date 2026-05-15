---
phase: 02-build-a-dataset-of-pixel-label-pairs
plan: 05
subsystem: dataset-tiling
tags: [tiling, nested-pyramid, eval-01, multi-scale, integration]
requires: [02-01, 02-02, 02-03, 02-04]
provides:
  - scripts/tiling.py (shared nested-pyramid decomposer + per-pyramid manifest)
  - tiler integrated into all 3 build pipelines
affects:
  - scripts/build_historical_dataset.py
  - scripts/build_dataset.py
  - scripts/build_satellite_dataset.py
tech-stack:
  added: []
  patterns:
    - "PIL Image open/crop/save per-map-dir I/O (copied from scripts/label.py)"
    - "per-pyramid subdir + pyramid.json manifest (RESEARCH Open-Q3)"
    - "default out_root = <map_dir>/pyramids keeps pyramids inside source subtree"
key-files:
  created:
    - scripts/tiling.py
    - .planning/phases/02-build-a-dataset-of-pixel-label-pairs/deferred-items.md
  modified:
    - tests/test_tiling.py
    - scripts/build_historical_dataset.py
    - scripts/build_dataset.py
    - scripts/build_satellite_dataset.py
decisions:
  - "Off-edge fraction is AREA-based (1 - on_area/896^2), not 1-D, so a corner pyramid clipped on both axes is judged by true footprint loss (D-09)"
  - "Pyramid id = grid row/col (origin // stride) — collision-free per source map (T-02-17)"
  - "Tiler defaults out_root to <map_dir>/pyramids so synthetic pyramids never cross the frozen train/test boundary (EVAL-01, T-02-15)"
metrics:
  duration: ~19m
  completed: 2026-05-15
  tasks: 2
  files: 6
---

# Phase 2 Plan 05: Shared multi-scale nested-pyramid tiler Summary

Strict-2×2-nested 1×896 + 4×448 + 16×224 pyramid tiler on a stride-448 grid
with area-based >50%-off-edge drop (D-06..D-09), wired into all three build
pipelines while preserving the synthetic frozen train/test split through tiling.

## What Was Built

**Task 1 — `scripts/tiling.py` (TDD: RED → GREEN):**
- `enumerate_pyramids(W, H)`: 896-px origins on a stride-448 grid anchored at
  the source top-left; drops a pyramid only when the **area** of its 896×896
  footprint lying off the source exceeds 50% (D-09 — a footprint exactly 50%
  or less off IS written).
- `tile(map_dir, out_root=None)`: reads the completed per-map dir
  (`image.png` + `land_cover.png` + `topography.png` + `sample_weights.json`),
  derives `(W,H)` from `image.png`, enumerates kept pyramids, and writes one
  sub-directory per pyramid containing: the 21 cropped PNG triplets (1×896,
  4×448 in strict 2×2, 16×224 in strict 2×2 within each 448), a `pyramid.json`
  manifest listing all 21 tile relative paths + explicit parent→child indices,
  and a **byte-identical** copy of the source `sample_weights.json`.
- Deterministic collision-free pyramid id `py_r{row}_c{col}` from the
  stride-grid cell (T-02-17).
- Internal guard: a per-map dir missing any of the 4 required files is skipped
  + logged, never tiled (T-02-16).
- Six geometry/weight tests in `tests/test_tiling.py`: tile-count, nested
  alignment, stride-448, intra-pyramid disjointness, edge-drop, weight
  propagation — all offline (small synthetic PNGs under `tmp_path`).

**Task 2 — tiler wired into all 3 build pipelines:**
- `build_historical_dataset.py`: `tiling.tile(sample_dir)` in `_process_one`
  after `hist_label.make_labels`.
- `build_dataset.py` (synthetic): `tiling.tile(out)` after `write_sample_weights`
  per `(source × style)` dir. `out` already lives under the train/ or test/
  root the caller routed the whole source into, and the tiler defaults to
  `out/pyramids` — so every pyramid of a held-out source stays test-side,
  never crossing the frozen `split.json` boundary (EVAL-01, T-02-15).
- `build_satellite_dataset.py`: `tiling.tile(sample_dir)` in `_process_one`
  after `sat_weights.write_sample_weights`.
- Each call site adds a pre-tiling missing-file guard (skip + log) so a
  partially-failed map never produces a corrupt pyramid tree (T-02-16).
- `test_split_subtree_preserved` (added in Task 1's test file) asserts a
  synthetic map under `test/` produces its pyramids under `test/` only.

## Verification Results

- `pytest tests/test_tiling.py -x -q --ignore=tests/integration` — **7 passed**
  (6 geometry/weight + split-subtree-preserved).
- Full offline unit suite `pytest tests/ --ignore=tests/integration` —
  **50 passed**, no regressions in any Wave-1/2 module.
- Plan verify command `grep -l tiling ... | wc -l == 3 && pytest tests/test_tiling.py`
  — passes (3/3 build scripts reference `tiling`; call positioned after
  `make_labels`/`write_sample_weights` in each).
- All 4 affected modules import cleanly.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] RED test fixtures encoded incorrect D-09 expectations**
- **Found during:** Task 1 GREEN phase.
- **Issue:** The initial `test_stride_448` and `test_edge_drop` fixtures
  hard-coded origin sets / drop arithmetic that contradicted the locked
  area-based D-09 semantics (e.g. expected only `{0,448,896}` origins on a
  1792-px map, but the 1344 origin is exactly 50% off and is correctly kept;
  the narrow-map drop case was actually only 25% off → kept).
- **Fix:** Recomputed every fixture against the area-based off-fraction
  (`1 - on_area/896²`); `test_stride_448` now asserts the load-bearing
  property (adjacent origins differ by exactly 448 + top-left anchored)
  instead of a brittle hard-coded list; `test_edge_drop` uses explicitly
  worked examples (1244-px → x=448 kept @11% off, x=896 dropped @61% off;
  896-px → x=448 kept at exactly 50% off — the D-09 boundary).
- **Files modified:** `tests/test_tiling.py`
- **Commit:** 17aacc7 (folded into the GREEN implementation commit, since the
  RED expectations were the artifact being corrected to the spec).

## Out-of-Scope / Deferred

- **Integration phase gate (`pytest tests/integration -m integration`) not
  runnable here.** All 6 integration tests are *online* network tests
  (Allmaps / IIIF / Rumsey LUNA / Sentinel-2 STAC / `s3://sentinel-cogs`)
  that hang without network (SIGTERM, exit 143). They exercise Wave-1/2
  fetch/search code, not the 02-05 tiler (pure offline geometry, no network
  code touched). Logged in
  `.planning/phases/02-build-a-dataset-of-pixel-label-pairs/deferred-items.md`
  for the verifier to run in a network-enabled environment.

## Known Stubs

None — `scripts/tiling.py` is fully wired (real PNG crops, real manifest,
real weight propagation) and invoked by all three live build pipelines.

## Threat Coverage

- **T-02-15** (EVAL-01 leakage) — mitigated: tiler defaults to
  `<map_dir>/pyramids`, keeping every pyramid inside the source's already-routed
  train/ or test/ subtree; `test_split_subtree_preserved` enforces it.
- **T-02-16** (corrupt pyramid from partial map) — mitigated: pre-tiling
  missing-file guard at every call site + inside `tile()`.
- **T-02-17** (pyramid id collision) — mitigated: deterministic grid-row/col
  ids, unique per source map; per-pyramid subdir isolates outputs.

No new threat surface introduced (offline filesystem transform only).

## Self-Check: PASSED
- `scripts/tiling.py` — FOUND
- `tests/test_tiling.py` — FOUND (7 tests, all green)
- commit 86b21ea (RED) — FOUND
- commit 17aacc7 (GREEN tiler) — FOUND
- commit 95cd67f (wiring) — FOUND
- `tiling` referenced in all 3 build scripts — VERIFIED (3/3)
