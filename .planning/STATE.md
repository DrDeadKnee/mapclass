---
gsd_state_version: 1.0
milestone: v1.1
milestone_name: Multi-Model Dynamic-LRP Comparison
status: planning
last_updated: "2026-05-19T02:00:00.000Z"
last_activity: 2026-05-19
progress:
  total_phases: 5
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-19)

**Core value:** A working, repeatable cross-model comparison — pick the locked map + a text query → run dynamic-LRP attribution through SigLIP-2 / CLIP / PaliGemma / a plain ViT → see their heatmaps side-by-side (and, per model, whether dynamic LRP covers that architecture at all) → judge visually. The comparison itself is the product.
**Current focus:** Phase 1 — Adapter Contract + Pre-Flight Guards

## Current Position

Phase: 1 of 5 (Adapter Contract + Pre-Flight Guards)
Plan: — (not yet planned)
Status: Ready to plan
Last activity: 2026-05-19 — v1.1 roadmap created (5 phases, --reset-phase-numbers active; v1.0 archived to .planning/archive/v1.0-milestone/)

Progress: [░░░░░░░░░░] 0%

### Resume

v1.1 roadmap is freshly created. 14 requirements (ADPT-01..04, ATTR-01..03,
MODEL-01..04, CMP-01..03) mapped across 5 phases, 100% coverage. PROJECT.md /
REQUIREMENTS.md are already reconciled to the multi-model framing (the prior
v1.0 scope tension is resolved by the v1.1 milestone definition; SigLIP-2's
recorded `SplitWithSizesBackward0` op-coverage gap is now the regression-oracle
datum, not a bug). v1.0 infra (pinned env, GCS image mirror, loaders, D-09
signed/zero-centered overlay) is reused as-is. Hard ordering preserved: adapter
contract + target decision + guards (Phase 1) → parameterized overlay +
SigLIP-2 oracle (Phase 2) → ViT (Phase 3) → CLIP + PaliGemma (Phase 4) →
side-by-side notebook done-gate (Phase 5). Next: `/gsd:plan-phase 1`.

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
