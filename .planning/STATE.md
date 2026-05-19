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
Plan: — (not yet planned)
Status: Ready to plan
Last activity: 2026-05-19 — roadmap simplified to a single lean phase per user ("getting over complicated — just want a notebook that runs multiple models against multiple maps")

Progress: [░░░░░░░░░░] 0%

### Resume

Roadmap was deliberately collapsed from 5 phases to 1 at the user's request.
Scope: ONE notebook running 4 models (SigLIP-2, CLIP, ViT-b-16, PaliGemma-3B)
against the last 50 Rumsey manifest maps, dynamic-LRP attribution overlays,
failures as captioned "no heatmap" tiles, headless exit 0. Models from
`gs://mapclass-training-northeast1/models/` (SigLIP-2 mirrored; PaliGemma
ALREADY at `models/paligemma-3b-mix-224/` — NOT gated, full code+weights there;
CLIP + ViT-b-16 to be mirrored). Maps from `gs://.../data/` (1,544 mirrored in
v1.0; use manifest[-50:]). Reuse v1.0 code (loaders, attribution, overlay,
mirror_model) — generalize with PLAIN per-model functions; NO adapter Protocol /
PatchGeometry dataclass / conformance-test or pre-flight-guard scaffold (user
called that over-complicated and it is now Out of Scope). Coverage gaps stay
recorded results; VENDOR_SHA + frozen pins untouched. 7 reqs (MODEL-01..02,
ATTR-01..02, NB-01..03) all map to Phase 1. Next: `/gsd-plan-phase 1`.

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
