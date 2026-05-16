---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: paused
stopped_at: Phase 04 paused after wave 3 — 04-01..04-04 complete; 04-05 deferred (blocking GPU-host probe gate, user choice 2026-05-16)
last_updated: "2026-05-16T18:20:00.000Z"
last_activity: 2026-05-16 -- Phase 04 waves 1-3 executed (04-01..04-04); OOM root-caused + fixed; 04-05 paused
progress:
  total_phases: 5
  completed_phases: 3
  total_plans: 15
  completed_plans: 14
  percent: 93
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-14)

**Core value:** A trained segmentation model on HuggingFace producing dense per-pixel
land-cover and topography predictions on illustrated regional maps, evaluated by
joint per-pixel NLL on held-out synthetic maps.
**Current focus:** Phase 04 — fine-tune-and-evaluate-the-segmentation-model-then-upload-to
comparison report. Publication moved to new Phase 5.

## Current Position

Phase: 04 (fine-tune-and-evaluate-the-segmentation-model-then-upload-to) — PAUSED (after wave 3)
Plan: 04-05 of 5 — DEFERRED (blocking gate)
Status: Phase 04 waves 1–3 complete; 04-05 paused by user decision (2026-05-16)
working branch `phase4` (HEAD c6e3af4), in sync with origin/phase4 through 229539d;
session commits 229539d..c6e3af4 NOT yet pushed.

Completed this run:
- 04-01 train_utils (weighted joint loss, teacher-forced c2f step, val carve)
- 04-02 gcs_checkpoint (preemption-safe save/resume, mocked offline suite)
- 04-03 evaluate_seg (joint per-pixel NLL harness, zero-leakage guard)
- 04-04 finetune_seg (probe-mode loop + 04-HUMAN-UAT.md GPU-host gate)

OOM RESOLVED: train_step_variant_a/b deferred-backward bug fixed (commit
229539d, per-tile backward). Verified: wave-1 suite ~13–15.6 GB → ~5 GB;
all waves peaked <4 GB, zero OOM. Diagnosis + logs in oom-diagnostics/
(untracked). Root cause memo: project memory phase4-oom-rootcause.

DEFERRED — 04-05 blocking gate (resume requires BOTH):
1. Run the GPU-host deferred steps in
   .planning/phases/04-*/04-HUMAN-UAT.md (SigLIP Variant B cost probe);
   capture the `PROBE RESULT |` line (GPU-hrs/epoch + projected full-grid hrs).
2. Re-run /gsd:execute-phase 4 --wave 4 — answer the D-05 grid-breadth gate
   (option-a/b/c) with the probe numbers pasted. Then Task 2 (offline
   report_comparison.py + manifest + report) auto-executes scoped to the choice.

new Phase 5 owns deliverable selection + HuggingFace publish (user decision
2026-05-16, see 04-CONTEXT.md `<domain>`).

Progress: [█████████░] 93% (14/15 plans; phase 04 minus the deferred 04-05 gate)

## Performance Metrics

**Velocity:**

- Total plans completed: 6 (Phase 1, executed pre-bootstrap)
- Average duration: n/a (pre-bootstrap execution; not tracked)
- Total execution time: n/a

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 1 | n/a | n/a |
| 02 | 5 | - | - |

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
| Georeferencing | PaliGemma-driven semi-automatic registration (cross-corr + TPS) | Deferred to v2 | Phase 2 planning (2026-05-15) |
| Georeferencing | Manual MapWarper/QGIS GCP placement fallback | Deferred to v2 | Phase 2 planning (2026-05-15) |

## Session Continuity

Last session: 2026-05-16 (resume)
Stopped at: Session resumed; pushed 10 pending phase1.5 commits
(dc4f585..c059f9c) to origin so the networked box can pull. Phase 4 is
planned + checker-verified (PASS iter 2/3). Next action: /gsd-execute-phase 4.

[older note retained for history]
Stopped at: Phase 3 context gathered
iteration), all coverage gates green, committed at `9ddd6ab`. Next action:
`/gsd-execute-phase 2`. Working tree clean; nothing to recover.
Resume notes:

- Resolved-with-user decisions baked into plans: A6 (Allmaps lookup → ALL
  annotations, in 02-01), A4 (synthetic per-(source×style), in 02-03),
  A7 (synthetic target N=100, in 02-03). Do not re-litigate on resume.

- Execution is NOT fully unattended: 02-03 and 02-04 each halt at a
  blocking loss-weight `checkpoint:decision` gate; both are `autonomous: false`.

- Wave order: W0=02-01 (test infra) · W1=02-02+02-03 · W2=02-04 · W3=02-05.

Resume file: .planning/phases/04-fine-tune-and-evaluate-the-segmentation-model-then-upload-to/04-CONTEXT.md
