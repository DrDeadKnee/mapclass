---
phase: 02-build-a-dataset-of-pixel-label-pairs
plan: 03
subsystem: synthetic-pipeline
status: paused-at-checkpoint
tags: [synthetic, azgaar, eval-01, stratified-split, loss-weights]
requires:
  - 02-01 (test infra: pytest, conftest fixtures, skeleton test files)
provides:
  - per-(Azgaar source × style) canonical map directories (A4 option a)
  - image.png + sample_weights.json in the synthetic per-map schema
  - locked-shape synthetic sample_weights.json writer (values PROVISIONAL)
  - ROADMAP SC#3 topography-boundary verification gate
affects:
  - scripts/label.py
  - scripts/render.py
  - scripts/build_dataset.py
  - scripts/synthetic_weights.py
  - scripts/biome_mapping.py
  - tests/test_render.py
tech-stack:
  added: []
  patterns:
    - per-(source×style) dir with byte-identical shared label masks
    - locked-shape sample_weights.json writer mirroring historical/label.py
key-files:
  created:
    - scripts/synthetic_weights.py
  modified:
    - scripts/label.py
    - scripts/render.py
    - scripts/build_dataset.py
    - scripts/biome_mapping.py
    - tests/test_render.py
decisions:
  - "Azgaar GeoJSON exposes NO heightmap template name -> stratification falls back to a filename-derived key (documented; user-aware)"
  - "Synthetic weight VALUES deferred to the blocking checkpoint:decision (NOT self-approved)"
metrics:
  tasks_completed: 2
  tasks_total: 4
  checkpoint: blocking checkpoint:decision reached (loss-weight approval)
  completed_date: 2026-05-15
---

# Phase 02 Plan 03: Synthetic Pipeline (per-(source×style), weights, frozen split) Summary

**One-liner:** Restructured the synthetic pipeline to one canonical
`<azgaar_id>__<style>/` directory per (Azgaar source × render style) with
byte-identical shared label masks and the full locked per-map schema, and
installed the ROADMAP SC#3 topography-boundary verification gate — paused at the
blocking loss-weight `checkpoint:decision` before the weights-finalisation and
frozen-split tasks.

## Status

**PAUSED at the blocking `checkpoint:decision` (Task 3 of 4 in plan order).**
Tasks 1 and 2 are complete and committed. Tasks "Create synthetic_weights.py
with approved values" and "Seeded stratified split with frozen split.json"
remain and require the checkpoint decision first.

## What Was Built

### Task 1 — per-(source×style) output + image.png (commit `9ec52e2`)

- `scripts/label.py`: split rasterisation into `make_label_arrays()` (returns the
  shared `land_cover`/`topography` PIL images) and reworked `make_labels()` to
  additionally write `image.png` (from a passed-in render) and
  `sample_weights.json`. Existing `_bbox`/`_rings`/polygon-fill logic and the
  `NODATA`/`WATER_TOPO=255` sentinels are unchanged.
- `scripts/render.py`: added `render_one(geojson_path, style) -> Image` (no disk
  write) so the per-style build hands the rendered image straight to
  `make_labels` as `image.png`. `render_map` retained for backward compatibility.
- `scripts/build_dataset.py`: `build_one_source()` rasterises the shared labels
  **once** per source and writes them byte-identically into each
  `<id>__<style>/` directory (A4 option (a)). Source stems are sanitised
  (`[^\w-]` → `_`, T-02-08). Per-source `try/except` keeps one bad source from
  aborting the batch (T-02-09).
- `tests/test_render.py`: `test_output_dimensions_match`,
  `test_per_source_style_dirs`, `test_shared_label_byte_identical`.

### Task 2 — ROADMAP SC#3 topography-boundary gate (commit `c2347f7`)

- `tests/test_render.py::test_synthetic_topo_locked_boundaries`: concrete
  `assert h_to_topo(...) == TOPO_IDX[...]` at and around the locked cuts
  (flat ≤20 / hilly 20–55 / mountainous >55 over the normalized `[0,100]`
  domain) plus water (`h < H_SEA_LEVEL`) → `None`. It is the SC#3 verification
  gate (not an image-dimension proxy).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Float-precision mis-bin at the hilly/mountainous cut**
- **Found during:** Task 2 (writing the SC#3 boundary gate)
- **Issue:** `biome_mapping.normalize_land_h(64)` evaluates to
  `55.00000000000001` (IEEE-754 division error). The locked rule
  `norm <= 55 → hilly` therefore mis-binned a land cell sitting *exactly* on
  the SC#3 hilly/mountainous cut into **mountainous**. This is a real
  ground-truth-correctness bug, not a test-precision issue.
- **Fix:** `h_to_topo` now rounds the normalized value to 6 decimal places
  (far finer than the smallest reachable spacing of `100/80 = 1.25` per unit
  of integer `h`) before the locked comparison, so the inclusive upper bounds
  (`≤20`, `≤55`) hold exactly. The locked SC#3 boundary values themselves are
  unchanged — the test was not loosened; the source was fixed (per the Task 2
  instruction).
- **Files modified:** `scripts/biome_mapping.py`, `tests/test_render.py`
  (added an explicit `h_to_topo(64) == hilly` regression guard)
- **Commit:** `c2347f7`

## Template-Name Finding (recorded per Task 1)

**Azgaar GeoJSON exposes NO heightmap template name.** There are no raw Azgaar
exports on disk (`data/synthetic/raw/` does not exist), and the Plan-01
`sample_azgaar_geojson` conftest fixture carries only `properties.biome` and
`properties.height` — no per-feature template property and no top-level
`metadata`/`info` block. Per the plan's documented fallback, the seeded
stratified split (remaining Task 4) will derive its stratification key from the
source filename (and may accept an explicit `--template`-style arg). **User
awareness flagged:** if real Azgaar exports are later found to expose a template
field, the split key should be revisited before the split.json is frozen.

## Provisional Artifact Pending Checkpoint

`scripts/synthetic_weights.py` was created in Task 1 because the refactored
`label.make_labels` imports its locked-shape `write_sample_weights()` writer.
The dict **shape** and `topography_weight` are locked; the per-class **float
values** are a clearly-marked PROVISIONAL DEFAULT (uniform 1.0) and are
**explicitly NOT the approved values** — they will be finalised to the
user-approved checkpoint option in the remaining Task 3. No checkpoint
self-approval was performed.

## Remaining Work (after checkpoint decision)

- **Task 3:** finalise `scripts/synthetic_weights.py` to the approved per-class
  floats; add `tests/test_synthetic_weights.py` (9 canonical keys set-equal to
  `HISTORICAL_LC_WEIGHTS`, all floats, JSON round-trips with
  `"source": "synthetic"`). Do NOT modify `historical/label.py`.
- **Task 4:** upgrade `build_dataset.py` to the
  `build_historical_dataset.py` argparse sub-command scaffold; add the seeded
  (`_SPLIT_SEED = 42`) stratified-by-template ~15% hold-out writing/reading a
  frozen `data/synthetic/split.json` (D-15..D-18); document the user-locked
  N=100 / ~15–16-per-template v1 target in the sub-command help; fill
  `tests/test_split.py` (deterministic / no-intersection / frozen-manifest /
  stratified-proportional).

## Self-Check

- `scripts/synthetic_weights.py` — present
- `scripts/label.py`, `scripts/render.py`, `scripts/build_dataset.py`,
  `scripts/biome_mapping.py` — modified, committed
- `tests/test_render.py` — 4 tests passing (incl. the SC#3 gate)
- Commits `9ec52e2`, `c2347f7` — present on the worktree branch
- Quick suite: 8 passed, 13 skipped, 0 failed

## Self-Check: PASSED
