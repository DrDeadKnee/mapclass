---
gsd_state_version: 1.0
milestone: v1.1
milestone_name: Multi-Model x Multi-Map Attribution Notebook
status: planning
last_updated: "2026-05-19T02:30:00.000Z"
last_activity: 2026-05-19 -- roadmap simplified to 1 phase per user
progress:
  total_phases: 1
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-19)

**Core value:** One notebook that runs dynamic-LRP attribution for several models against several maps and shows the heatmaps for a human to eyeball. Models + maps both from the GCS buckets in the README.
**Current focus:** Phase 1 — Multi-Model × Multi-Map Attribution Notebook

## Current Position

Phase: 1 of 1 (Multi-Model × Multi-Map Attribution Notebook)
Plan: — (GSD planning dropped per user, 2026-05-23)
Status: Working hands-on (no GSD loop) — see Session Continuity
Last activity: 2026-05-23 - Dropped GSD; built notebooks/03_single_model.ipynb (load image → run chosen model → heatmap; configurable MODEL_NAME/MAP_INDEX/QUERY); verified headless exit 0 (vit_b16, peak 1.66 GB)

Progress: deliverable shipped via quick task + 03_single_model notebook

### Resume

**WORKING MODE CHANGED (2026-05-23):** User dropped the GSD workflow for this
project (commit 879b480: "the gsd approach is just not working ... back to basic
claude code with active engagement from my end"). `.planning/` is now a RECORD,
not a driver. Do NOT route to `/gsd-*` commands or agents unless the user asks.
Edit notebooks/code directly; the `.ipynb` is the source of truth.

**Where things stand:** The v1.1 deliverable (multi-model dynamic-LRP over maps)
was already built two ways: `notebooks/02_multimodel.ipynb` (4 models × last-50
maps loop, hand-edited by user) and the new `notebooks/03_single_model.ipynb`
(pick ONE map + ONE model, easily configurable — the hands-on flow the user
asked for). Hardware is now a hard constraint documented in CLAUDE.md (single
NVIDIA L4, ~22.5 GB usable VRAM, one model in VRAM at a time).

**Verified result:** vit_b16 produces real heatmaps (single run peaks only
~1.66 GB — far below the leaky 50-map loop's ~19 GB). SigLIP-2/CLIP op-coverage
gaps and PaliGemma OOM remain recorded results (Fallback Ladder declined).
VENDOR_SHA + frozen pins untouched.

## Performance Metrics

**Velocity:**

- Total plans completed: 0
- Average duration: — min
- Total execution time: 0.0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**

- Last 5 plans: —
- Trend: —

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [v1.1 Roadmap]: 5-phase structure (coarse granularity) — research's 7-phase order consolidated by merging parameterized-overlay+attribution with the SigLIP-2 regression oracle (Phase 2), since SigLIP-2 is the only non-new model and both are low-risk standard-pattern work; all hard ordering constraints preserved.
- [v1.1 Roadmap]: Phase 4 (CLIP + PaliGemma) research/decision-flagged — CLIP CLS-token strip geometry must be validated against real CLIP relevance; PaliGemma answer-token target is a per-query modeling decision resolved at the Phase 1 target-decision step.
- [01-03, user, v1.0]: dynamicLRP does NOT cover SigLIP-2-so400m (`split_with_sizes` MAP-pool op) — accepted as a per-model FINDING; entire Fallback Ladder DECLINED. In v1.1 this is the regression-oracle datum Phase 2 must reproduce.
- [Pivot]: Project reframed as a multi-model comparison of dynamic-LRP (SigLIP-2 = one negative-result model); PROJECT.md/REQUIREMENTS.md now reflect this — prior scope tension resolved.

### Pending Todos

None yet.

### Blockers/Concerns

- [v1.1 — design, OPEN until Phase 1] The per-architecture attribution target is the single decision that gates everything; getting it wrong silently produces query-independent heatmaps. Must be resolved + documented for all 4 models in Phase 1 (ADPT-02) before any harness loop.
- [v1.1 — risk, expected] PaliGemma (~3B, ~7.5× the so400m ceiling) likely OOMs the L4; this is an expected, recordable result, not a bug to engineer around (Phase 4 / Phase 5 captioned "no heatmap" tile).
- [v1.1 — scope guard] No sweep, no per-model coverage-gap fixing / vendor patching, no pin bumps. `VENDOR_SHA==405e74243ecaa1f615f418fdc8ba24c3c5889b1e` must stay intact (ADPT-04 pre-flight enforces this).
- [Carried, v1.0] SigLIP-2 single-forward peak VRAM = 4.326 GB (forward-only lower bound; the dynamicLRP relevance-pass peak is unmeasured because the SigLIP-2 relevance pass never completes). Relevant to Phase 5 sequential load/free OOM budgeting.
- [v1.1 — RESULT, quick-260519-2v4] Cross-model dynamic-LRP coverage measured on last-50 Rumsey maps: **ViT-b-16 works (19/50 heatmaps, peak 19.3 GB)** — the positive control. **SigLIP-2 0/50** (`split_with_sizes` gap, known). **CLIP 0/50 — NEW finding**: genuine dynamicLRP op-coverage gap on CLIP engine internals (`Expand should not increase number of dimensions`), inputs/forward verified correct. **PaliGemma-3B 0/50**: OOM at load on the 24 GB L4 (expected). Recorded results, not bugs — Fallback Ladder stays declined.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260519-2v4 | Multi-model (SigLIP-2/CLIP/ViT-b-16/PaliGemma-3B) × last-50 Rumsey maps dynamic-LRP notebook (headless exit 0) | 2026-05-19 | 14fd238 | [260519-2v4-build-one-notebook-that-runs-dynamic-lrp](./quick/260519-2v4-build-one-notebook-that-runs-dynamic-lrp/) |

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Reporting | Structured per-model op-coverage report table | Deferred to future (user chose "overlays only" for v1.1) | v1.1 scope |
| Models | Additional models beyond the 4 | Deferred | v1.1 scope |
| Metrics | Quantitative attribution / faithfulness scoring | Deferred | v1.1 scope |

## Session Continuity

Last session: 2026-05-19T02:00:00.000Z
Stopped at: v1.1 roadmap created (ROADMAP.md, REQUIREMENTS.md traceability, STATE.md written); 5 phases, 14/14 requirements mapped, 100% coverage
Resume file: None
