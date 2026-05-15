---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: Phase 2 context + research gathered; planning next
last_updated: "2026-05-15T12:38:44.194Z"
last_activity: 2026-05-15 -- Phase 02 planning complete
progress:
  total_phases: 4
  completed_phases: 0
  total_plans: 5
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-14)

**Core value:** A trained segmentation model on HuggingFace producing dense per-pixel
land-cover and topography predictions on illustrated regional maps, evaluated by
joint per-pixel NLL on held-out synthetic maps.
**Current focus:** Phase 2 — Build a dataset of pixel-label pairs

## Current Position

Phase: 2 of 4 (Build a dataset of pixel-label pairs)
Plan: TBD (Phase 2 plans not yet decomposed)
Status: Ready to execute
Last activity: 2026-05-15 -- Phase 02 planning complete
working branch `phase1.5`; Allmaps annotation index integration in progress
(`scripts/historical/allmaps.py` untracked, `scripts/historical/rumsey.py` modified).

Progress: [██░░░░░░░░] 25% (Phase 1 of 4 complete)

## Performance Metrics

**Velocity:**

- Total plans completed: 1 (Phase 1, executed pre-bootstrap)
- Average duration: n/a (pre-bootstrap execution; not tracked)
- Total execution time: n/a

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 1 | n/a | n/a |

**Recent Trend:**

- Last 5 plans: n/a
- Trend: n/a (first measured cycle starts at Phase 2)

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table and the `<decisions>`
block under Constraints. Recent decisions affecting current work:

- Phase 1: LoRA only on SigLIP attention (`q_proj`/`k_proj`/`v_proj`/`out_proj`),
  Gemma fully frozen, projector trainable — preserves geographic terminology.

- Phase 2: Direct S3 fetch of ESA WorldCover and Copernicus DEM (no GEE in the
  critical path); Dynamic World optional only.

- Phase 2: Class-conditional per-source loss weights for historical maps —
  topography trusted, water mostly trusted, trees/built-up/cropland downweighted.

- Phase 2: Georeferencing precedence — Allmaps → unregistered manifest →
  PaliGemma semi-auto (cross-correlation + TPS) → manual MapWarper / QGIS.

### Pending Todos

None yet.

### Blockers/Concerns

None yet.

## Deferred Items

Items acknowledged and carried forward:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Evaluation | Hand-annotated real grand-strategy map test set (Label Studio / CVAT) | Deferred to v2 | Bootstrap (2026-05-14) |
| Backbone | Large first-kernel ConvNet (RepLKNet / SLaK / ConvNeXt) trained from scratch | Stretch goal | Bootstrap (2026-05-14) |
| Backbone | Progressive backbone unfreezing experiment | Deferred to v2 | Bootstrap (2026-05-14) |
| Synthetic | Calibrate Azgaar synthetic-height thresholds against SRTM statistics | Deferred | Bootstrap (2026-05-14) |

## Session Continuity

Last session: 2026-05-15 — session resumed, awaiting next-action selection (plan Phase 2)
Stopped at: Phase 2 context + research gathered; planning next
`phase1.5`; pending working-tree changes — `scripts/historical/rumsey.py`
modified and `scripts/historical/allmaps.py` untracked, both part of the
Allmaps annotation index integration for Phase 2.
Resume file: .planning/phases/02-build-a-dataset-of-pixel-label-pairs/02-CONTEXT.md
