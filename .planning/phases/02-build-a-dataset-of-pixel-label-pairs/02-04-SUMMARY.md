---
phase: 02-build-a-dataset-of-pixel-label-pairs
plan: 04
subsystem: satellite-pipeline
status: paused-at-checkpoint
tags: [satellite, sentinel-2, stac, cog, coverage-scan]
requires:
  - 02-01
  - 02-02 (scripts/historical/georef.py::write_georeferenced_geotiff — confirmed present)
provides:
  - scripts/satellite/ package (pending — gated by checkpoint)
affects:
  - requirements.txt (pending)
tech-stack:
  added: []
  patterns: []
key-files:
  created: []
  modified: []
decisions: []
metrics:
  duration: in-progress
  completed: null
---

# Phase 02 Plan 04: Satellite Pipeline Summary (PAUSED AT CHECKPOINT)

One-liner: Satellite source family (class-diversity coverage scan + cloud-filtered Sentinel-2 L2A STAC + COG byte-range RGB fetch) — execution paused at the plan's first task, a BLOCKING `checkpoint:decision` for per-source loss-weight value approval.

## Execution Status

The plan's **first task** is a blocking `checkpoint:decision` (Checkpoint: Approve per-source loss-weight values). It gates all three downstream implementation tasks (Task 1 weights.py needs the approved per-class floats; Tasks 2-3 depend on Task 1). Auto-mode is OFF (`workflow.auto_advance=false`, `workflow._auto_chain_active=false`), so the executor must STOP and collect the human decision rather than self-approve. No implementation work could be started before the decision.

## Pre-flight verification completed

- Worktree branch `worktree-agent-a51972f7f46ae438c` confirmed; base reset to `ae5bdbe` (Wave 1 complete).
- Confirmed shared reusable symbols exist verbatim in the base (no fork needed downstream):
  - `scripts/historical/georef.py:43` `write_georeferenced_geotiff(rgb_array, affine, out_path)`
  - `scripts/historical/label.py:95` `make_labels(map_geotiff, output_dir)`
  - `scripts/historical/label.py:40` `HISTORICAL_LC_WEIGHTS` (9 keys), `:51` `HISTORICAL_TOPO_WEIGHT=1.0`
  - `scripts/historical/worldcover.py` `WC_REMAP` (:42), `_tile_origins` (:67), `AWS_NO_SIGN_REQUEST` (:36)
- Locked `sample_weights.json` shape captured from `label.py:139-144`:
  `{"land_cover_weights": <9-key dict>, "topography_weight": <float>, "source": "<src>", "map_file": <name>}`
- The 9 locked `LANDCOVER_CLASSES` keys (must match exactly in `SATELLITE_LC_WEIGHTS`):
  `water, trees, shrubland, grassland, cropland, built_up, bare_sparse, flooded_wetland, snow_ice`

## Checkpoint Decision Requested

**Approve the satellite per-source loss-weight values** for `SATELLITE_LC_WEIGHTS` (same 9 keys as `HISTORICAL_LC_WEIGHTS`) plus `SATELLITE_TOPO_WEIGHT`. The dict shape and the `topography_weight` slot are locked; only the per-class floats need sign-off. WorldCover labels are contemporaneous with the satellite imagery (no temporal drift), so the historical down-weighting does NOT apply.

Options presented to the user:

- **balance-tilt** (recommended, implements CONTEXT/PROJECT.md guidance):
  `trees 0.6, water 0.6, cropland 1.3, built_up 1.3, flooded_wetland 1.3, shrubland 1.0, grassland 1.0, bare_sparse 1.0, snow_ice 1.0`; `SATELLITE_TOPO_WEIGHT 1.0`
- **uniform**: all 9 classes `1.0`; `SATELLITE_TOPO_WEIGHT 1.0` (defer balancing to Phase 3/4 sampler)
- Or: user supplies explicit per-class floats.

The executor did NOT guess or self-approve. No weights code written.

## Tasks Completed

None — the gating checkpoint is the first task.

| Task | Name | Status | Commit |
| ---- | ---- | ------ | ------ |
| 0 | Checkpoint: Approve per-source loss-weight values | AWAITING DECISION | n/a |
| 1 | Create satellite/ package — stac.py, fetch.py, weights.py | blocked by checkpoint | — |
| 2 | Create coverage.py — class-diversity region picker (D-14) | blocked by checkpoint | — |
| 3 | build_satellite_dataset.py — sub-commands (D-13 drop counter) | blocked by checkpoint | — |

## Deviations from Plan

None — execution paused at the plan-defined blocking checkpoint before any implementation.

## Resume Instructions

A fresh executor agent should be spawned with the approved weight option (or explicit per-class floats) supplied in its prompt. It resumes at **Task 1**, hard-coding the approved values into `scripts/satellite/weights.py::SATELLITE_LC_WEIGHTS` / `SATELLITE_TOPO_WEIGHT`, then proceeds through Tasks 1-3 as written.
