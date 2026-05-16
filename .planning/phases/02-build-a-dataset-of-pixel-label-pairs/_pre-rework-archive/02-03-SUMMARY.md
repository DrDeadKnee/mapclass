---
phase: 02-build-a-dataset-of-pixel-label-pairs
plan: 03
subsystem: synthetic-pipeline
status: complete
tags: [synthetic, azgaar, eval-01, stratified-split, loss-weights]
requires:
  - 02-01 (test infra: pytest, conftest fixtures, skeleton test files)
provides:
  - per-(Azgaar source × style) canonical map directories (A4 option a)
  - image.png + sample_weights.json in the synthetic per-map schema
  - locked-shape synthetic sample_weights.json writer (uniform-1.0, approved)
  - ROADMAP SC#3 topography-boundary verification gate
  - seeded stratified frozen train/test split (EVAL-01, D-15..D-18)
affects:
  - scripts/label.py
  - scripts/render.py
  - scripts/build_dataset.py
  - scripts/synthetic_weights.py
  - scripts/biome_mapping.py
  - tests/test_render.py
  - tests/test_synthetic_weights.py
  - tests/test_split.py
tech-stack:
  added: []
  patterns:
    - per-(source×style) dir with byte-identical shared label masks
    - locked-shape sample_weights.json writer mirroring historical/label.py
    - seeded stratified-by-template hold-out with a frozen ID-list manifest
key-files:
  created:
    - scripts/synthetic_weights.py
    - tests/test_synthetic_weights.py
  modified:
    - scripts/label.py
    - scripts/render.py
    - scripts/build_dataset.py
    - scripts/biome_mapping.py
    - tests/test_render.py
    - tests/test_split.py
decisions:
  - "Azgaar GeoJSON exposes NO heightmap template name -> stratification falls back to a filename-derived key (documented; user-aware)"
  - "Synthetic loss weights = uniform 1.0 (user-approved checkpoint:decision 2026-05-15, option 'uniform') — synthetic labels are exact by construction; class imbalance deferred to Phase 3/4 DataLoader/sampler"
  - "v1 synthetic source target N=100 Azgaar source maps (user-locked A7; research recommended N=50) — splitter is N-agnostic"
metrics:
  tasks_completed: 4
  tasks_total: 4
  checkpoint: blocking checkpoint:decision RESOLVED (loss-weight approval = uniform 1.0)
  completed_date: 2026-05-15
---

# Phase 02 Plan 03: Synthetic Pipeline (per-(source×style), weights, frozen split) Summary

**One-liner:** Restructured the synthetic pipeline to one canonical
`<azgaar_id>__<style>/` directory per (Azgaar source × render style) with
byte-identical shared label masks, installed the ROADMAP SC#3
topography-boundary verification gate, finalised the synthetic loss weights to
the user-approved uniform-1.0, and implemented the seeded stratified-by-template
train/test split with a frozen `split.json` (EVAL-01, D-15..D-18) at the
user-locked N=100 v1 target.

## Status

**COMPLETE — all 4 tasks executed.** The blocking loss-weight
`checkpoint:decision` was resolved (approved: **uniform 1.0**) and Tasks 3 & 4
were executed in this continuation run.

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

### Checkpoint — loss-weight decision RESOLVED

The blocking `checkpoint:decision` (synthetic per-source loss-weight values)
was presented to and resolved by the user on **2026-05-15**.
**Approved option: `uniform`** — all 9 land-cover classes = `1.0`,
`topography_weight = 1.0`. Rationale: synthetic labels are exact by
construction (rasterised directly from the Azgaar source), so no temporal-drift
discount applies; class-imbalance correction is deferred to the Phase 3/4
DataLoader/sampler, not folded into the per-source weight layer.

### Task 3 — finalised synthetic loss weights (commit `d595d60`)

- `scripts/synthetic_weights.py`: stripped all PROVISIONAL / checkpoint-pending
  markers and comments; `SYNTHETIC_LC_WEIGHTS` finalised to the approved
  uniform `1.0` for all 9 canonical classes; `SYNTHETIC_TOPO_WEIGHT = 1.0`.
  Dict shape is locked set-equal to `HISTORICAL_LC_WEIGHTS`. The
  `write_sample_weights()` writer is unchanged (locked-shape JSON:
  `land_cover_weights` / `topography_weight` / `source` / `map_file`,
  `source == "synthetic"`).
- `tests/test_synthetic_weights.py` (new): asserts the 9-key set equals
  `HISTORICAL_LC_WEIGHTS` keys, every value is a float, the approved
  uniform-1.0 values, and the written JSON round-trips with the locked
  top-level keys + `source == "synthetic"`.
- `scripts/historical/label.py` left unmodified (verified via empty `git diff`).

### Task 4 — seeded stratified frozen split, N=100 v1 (commit `9d40f38`)

- `scripts/build_dataset.py`: upgraded to the `build_historical_dataset.py`
  argparse sub-command scaffold (`add_subparsers(dest="command",
  required=True)` + `add_common`), `build` sub-command.
- `template_key()`: derives the stratification (continent-template) key from
  the source filename — strips the trailing numeric/index suffix
  (`europe_07` → `europe`); a no-prefix source becomes its own singleton
  stratum (never starved). This is the documented fallback because Azgaar
  GeoJSON exports expose no heightmap-template field.
- `stratified_split()`: deterministic; seeded with the fixed constant
  `_SPLIT_SEED = 42` (salted per-template for stability as sources are
  appended); `~15%` per-template hold-out (`round(n * 0.15)`, ≥1 for a
  non-empty template); operates at the WHOLE Azgaar source-map level so ALL
  render styles of a held-out source go to `test/` (D-15, D-16).
- `load_or_create_split()`: on the FIRST build computes the split and writes
  `data/synthetic/split.json` listing the held-out test IDs; on EVERY
  subsequent build reads it and never recomputes/mutates it (D-17, D-18).
- `build()`: routes each source into sibling `train/<id>__<style>/` vs
  `test/<id>__<style>/` by the frozen test-ID list; new sources after the
  first build always land in `train/`.
- N=100 / ~15–16-per-template v1 target documented in the `build`
  sub-command help line and the module epilog (user-locked A7; research
  recommended N=50).
- `tests/test_split.py` (filled): `test_seeded_split_deterministic`,
  `test_stratified_holdout_proportional`,
  `test_split_is_whole_source_no_template_collision`,
  `test_no_train_test_intersection` (EVAL-01, also asserts all styles of a
  held-out source under `test/` and none under `train/` — D-15),
  `test_split_manifest_frozen` (EVAL-01 / D-18: rebuild with new sources →
  `split.json` byte-unchanged, new IDs land in `train/`).

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

No other deviations — Tasks 3 & 4 executed exactly as written against the
resolved checkpoint decision.

## Template-Name Finding (recorded per Task 1)

**Azgaar GeoJSON exposes NO heightmap template name.** There are no raw Azgaar
exports on disk (`data/synthetic/raw/` does not exist), and the Plan-01
`sample_azgaar_geojson` conftest fixture carries only `properties.biome` and
`properties.height` — no per-feature template property and no top-level
`metadata`/`info` block. Per the plan's documented fallback, the seeded
stratified split derives its stratification key from the source filename
(`template_key()` strips the trailing numeric suffix). **User awareness
flagged:** if real Azgaar exports are later found to expose a template field,
`template_key()` should be revisited **before** the first `split.json` is
frozen (it is frozen on first real build).

## Loss-Weight Decision (resolved)

The synthetic loss-weight `checkpoint:decision` is resolved: **uniform 1.0**
for all 9 land-cover classes and `topography_weight`. `synthetic_weights.py`
has been finalised and all PROVISIONAL markers removed. Class-imbalance
correction is intentionally deferred to the Phase 3/4 DataLoader/sampler, not
the per-source weight layer.

## Self-Check

- `scripts/synthetic_weights.py` — present, finalised (no PROVISIONAL markers)
- `scripts/build_dataset.py` — argparse sub-command scaffold + frozen split
- `tests/test_synthetic_weights.py` — present, 4 tests passing
- `tests/test_split.py` — present, 5 tests passing (EVAL-01 guardrails)
- `scripts/historical/label.py` — unmodified (empty git diff)
- Commits `9ec52e2`, `c2347f7`, `d595d60`, `9d40f38` — present on the worktree branch
- Quick suite: 26 passed, 4 skipped, 0 failed (`pytest tests/ -x --ignore=tests/integration`)

## Self-Check: PASSED
