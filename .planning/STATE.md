---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: 01-03 complete-with-finding (SigLIP-2 dynamicLRP op-coverage gap; D-02/D-03 user-waived)
last_updated: "2026-05-19T12:00:00.000Z"
last_activity: 2026-05-19 -- 01-03 finalized as honest smoke test; SigLIP-2 attribution finding recorded; peak-VRAM blocker closed
progress:
  total_phases: 2
  completed_phases: 0
  total_plans: 3
  completed_plans: 3
  percent: 50
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-18)

**Core value:** A working, repeatable loop — pick a map + a text query → get a dynamic-LRP attribution heatmap overlaid on that map → judge it visually — that scales to a configurable sweep browsable in a notebook.
**Current focus:** Phase 01 — pinned-environment-gcs-mirrors-and-a-verified-single-slice-a

## Current Position

Phase: 01 (pinned-environment-gcs-mirrors-and-a-verified-single-slice-a) — all 3 plans done; 01-03 COMPLETE-WITH-FINDING
Plan: 3 of 3 complete (01-03 finalized as an honest end-to-end smoke test after the user resolved the Task 3 human-verify checkpoint)
Status: 01-03 complete-with-finding — SigLIP-2 dynamicLRP op-coverage gap is an accepted per-model finding; D-02/D-03 visual gate consciously WAIVED for SigLIP-2 by the user; Fallback Ladder DECLINED
Last activity: 2026-05-19 -- 01-03 finalized; honest smoke test executes end-to-end (0 cell errors); peak forward VRAM 4.326 GB recorded; multi-model reframe noted

Progress: [██████████] 3/3 plans complete (01-01, 01-02, 01-03-with-finding). NOTE: 01-03's central deliverable (a control-verified SigLIP-2 heatmap) is NOT met — recorded as a documented negative result under the reframed multi-model purpose, not a clean pass.

### Resume

Phase 1 plans are all executed. 01-03 is **complete-with-finding**, NOT a clean
pass: dynamicLRP does not cover SigLIP-2-so400m's `split_with_sizes`
(`SplitWithSizesBackward0`, MAP-pool head), so `attribute()` produces no
relevance and no heatmap for SigLIP-2. At the 01-03 human-verify checkpoint the
user **declined the entire Fallback Ladder** (no custom Promise / pre-pool /
LXT / captum IG — *"Smoke-test was good, it didn't crash. Let's leave well
enough alone and move on."*) and the Phase 1 D-02/D-03 visual-eyeball gate is
**consciously WAIVED for SigLIP-2**. `01_single_slice.ipynb` was reframed into
an honest end-to-end smoke test (load → forward → peak forward VRAM → coverage
probe → caught finding → source map; 0 cell errors, exit 0). `third_party/
dynamicLRP` is unmodified; VENDOR_SHA intact (no T-01-SC3 deviation taken).

**Open at the next milestone/phase transition (do NOT auto-resolve here):**
PROJECT.md says SigLIP-2 is fixed and model-swap is out of scope, but the
user's decision reframes the project as a **multi-model comparison of
dynamic-LRP** (SigLIP-2 = one negative-result model). PROJECT.md scope wording,
ROADMAP Phase 1 success criteria 4-5 (assume a working SigLIP-2 heatmap), and
the Phase 2 sweep premise all need reconciliation against the multi-model
framing before Phase 2 starts. Executor did NOT rewrite PROJECT.md.

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

- [Roadmap]: 2-phase structure — verify the single-slice attribution primitive (with three sanity controls) before any sweep; Phase 2 is pure orchestration with zero new attribution logic.
- [Roadmap]: Phase 1 flagged NEEDS DEEPER RESEARCH — the SigLIP-2 ↔ dynamicLRP contrastive-encoder adaptation is the central technical risk; sequence as reproduce ViT.ipynb first, then swap SigLIP-2 + similarity target.
- [01-03, user]: dynamicLRP does NOT cover SigLIP-2-so400m (`split_with_sizes` MAP-pool op) — accepted as a per-model FINDING; the ENTIRE Fallback Ladder DECLINED (no custom Promise/pre-pool/LXT/captum IG); D-02/D-03 visual-eyeball gate consciously WAIVED for SigLIP-2.
- [01-03, user]: Project reframed as a multi-model comparison of dynamic-LRP (SigLIP-2 = one negative-result model). Creates a PROJECT.md/ROADMAP scope tension to reconcile at the next milestone/phase transition (executor did not rewrite PROJECT.md).

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 1 — 01-03 ARCHITECTURAL, RESOLVED 2026-05-19] **dynamicLRP op-coverage FAILS on SigLIP-2-so400m as a contrastive MAP-pool encoder.** Confirmed on NVIDIA L4: `LRPEngine.run` rejects all target forms — `logits_per_image[0,0]` (0-dim → IndexError); `[:1,:1]`/`.reshape(1)` → error path at autograd node `SplitWithSizesBackward0`; coverage probe op count = 26. `split_with_sizes` is intrinsic to SigLIP-2's attention/MAP-pool head and is outside the current engine's coverage. **RESOLUTION (user, 01-03 human-verify checkpoint):** the user DECLINED the entire Fallback Ladder (no custom Promise, no pre-pool/`use_attn_lrp`, no LXT, no captum IG) and ACCEPTED this as a per-model FINDING for the reframed multi-model dynamic-LRP comparison; the Phase 1 D-02/D-03 visual gate is consciously WAIVED for SigLIP-2. `01_single_slice.ipynb` reframed to an honest smoke test. `third_party/dynamicLRP` unmodified; no T-01-SC3 deviation taken. Not to be re-attempted without a new explicit user decision.
- [Phase 1 — INFORMATIONAL] dynamicLRP faithfulness on SigLIP-2 is moot for this project: no relevance is produced for SigLIP-2, so the three sanity controls are not applicable (gate user-waived). Relevant only if SigLIP-2 attribution is revisited under the multi-model framing.
- [Phase 1 — RESOLVED 2026-05-19] Peak VRAM blocker. **CLOSED:** the single so400m FORWARD on the locked slice peaks at **4.326 GB (4,644,488,704 bytes)** — measured + recorded in `notebooks/01_single_slice.ipynb` Cell 4. Caveat: this is forward-only; the dynamicLRP relevance-pass peak (Pitfall E activation retention) is unmeasurable for SigLIP-2 because the relevance pass never completes. Phase 2 sweep sizing should treat 4.326 GB as a forward-only lower bound, not the full attribution peak.
- [Phase transition — NEW, OPEN] PROJECT.md/ROADMAP scope tension: PROJECT.md fixes SigLIP-2 and puts model-swap out of scope, but the user's 01-03 decision reframes the project as a multi-model dynamic-LRP comparison (SigLIP-2 = negative result). ROADMAP Phase 1 success criteria 4-5 and the Phase 2 sweep premise assume a working SigLIP-2 heatmap. Reconcile at the next milestone/phase-transition decision before Phase 2. Executor did NOT rewrite PROJECT.md.
- [Phase 1] Rumsey URL health unknown; the per-id outcome manifest from ingestion is the only signal of how many maps are actually available — budget for a fraction being unavailable.

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| *(none)* | | | |

## Session Continuity

Last session: 2026-05-19T12:00:00.000Z
Stopped at: 01-03 complete-with-finding — honest smoke test executed end-to-end (0 cell errors); SigLIP-2 op-coverage finding recorded; D-02/D-03 user-waived; Fallback Ladder declined; peak-VRAM blocker closed; multi-model reframe + PROJECT.md tension flagged for the next transition
Resume file: .planning/phases/01-pinned-environment-gcs-mirrors-and-a-verified-single-slice-a/01-03-SUMMARY.md
