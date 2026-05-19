---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: Phase 1 context gathered
last_updated: "2026-05-19T00:35:11.133Z"
last_activity: 2026-05-19 -- Phase 01 execution started
progress:
  total_phases: 2
  completed_phases: 0
  total_plans: 3
  completed_plans: 2
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-18)

**Core value:** A working, repeatable loop — pick a map + a text query → get a dynamic-LRP attribution heatmap overlaid on that map → judge it visually — that scales to a configurable sweep browsable in a notebook.
**Current focus:** Phase 01 — pinned-environment-gcs-mirrors-and-a-verified-single-slice-a

## Current Position

Phase: 01 (pinned-environment-gcs-mirrors-and-a-verified-single-slice-a) — BLOCKED (01-03 Rule 4 architectural decision)
Plan: 3 of 3 (01-03 in progress; Task 1 committed, Task 2 committed, BLOCKED before Task 3 human-verify by an engine op-coverage failure)
Status: 01-03 blocked at a Rule 4 architectural decision — awaiting human Fallback-Ladder selection
Last activity: 2026-05-19 -- 01-03 dynamicLRP op-coverage failure on SigLIP-2-so400m confirmed on L4; surfaced as the human-verify checkpoint

Progress: [███████░░░] 2/3 plans complete (01-01, 01-02); 01-03 Tasks 1-2 built+committed, blocked at Rule 4 architectural decision

### Resume

01-03 is BLOCKED at a **Rule 4 architectural decision** (NOT auto-selectable):
the as-is dynamicLRP path fails on SigLIP-2-so400m at `SplitWithSizesBackward0`
(see Blockers/Concerns). `01_single_slice.ipynb` cannot execute end-to-end
until the attribution engine path is resolved. Tasks 1-2 (attribution.py,
overlay.py, tests, notebook + generator) are built and committed; the unit
tests pass (16/16).

**Human decision required (Fallback Ladder, 01-03-PLAN.md):**
- Step 2: write/repair a custom dynamicLRP Promise for `split_with_sizes`
  on SigLIP-2's MAP-pool/attention topology. Patches vendored
  `third_party/dynamicLRP/` → deviates from the recorded VENDOR_SHA
  (threat T-01-SC3 — must be recorded as a deliberate deviation). Highest
  fidelity, highest effort, uncertain.
- Step 5: switch the attribution engine to captum Integrated Gradients on
  `logits_per_image[0,0]` (captum is already a pinned dep) — a degraded but
  trivially-correct baseline that unblocks the three sanity controls and the
  Phase 1 finish line without touching the vendored engine.

On resolution: re-run `01_single_slice.ipynb` end-to-end, record peak VRAM,
then the human-verify checkpoint (D-02/D-03 — all three controls visually
PASS AND the 1,544 mirror confirmed complete) → Phase 1 DONE.

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

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 1 — 01-03 ARCHITECTURAL, ACTIVE] **dynamicLRP op-coverage FAILS on SigLIP-2-so400m as a contrastive MAP-pool encoder.** Confirmed by a fresh NVIDIA L4 run (2026-05-19): `LRPEngine.run` rejects all target forms — `logits_per_image[0,0]` (0-dim → IndexError); `[:1,:1]` (2-D) and `.reshape(1)` (1-D) → `'DummyPromise' object is not iterable` / `No valid curnode candidate was found` at autograd node `SplitWithSizesBackward0`; coverage probe op count = 26. `split_with_sizes` is intrinsic to SigLIP-2's attention/MAP-pool head — the engine registers `SplitWithSizesBackward`→`SplitBackwardProp` but its Promise consumer chokes on SigLIP-2's split topology. This is the MEDIUM-LOW research risk realized. `attribute()` raises `RuntimeError`; `01_single_slice.ipynb` cannot execute end-to-end. **Resolution is a Rule 4 architectural decision pending the 01-03 human-verify checkpoint** — Fallback-Ladder step 2 (custom/repaired engine Promise: patches vendored `third_party/dynamicLRP/`, deviates from the recorded SHA, threat T-01-SC3) OR step 5 (switch the attribution engine to captum Integrated Gradients on `logits_per_image[0,0]` as a degraded but trivially-correct baseline). NOT auto-selected by the executor.
- [Phase 1] dynamicLRP faithfulness on SigLIP-2 as a contrastive encoder is unvalidated in the paper (graph coverage only). Resolvable only by running the three sanity controls on real output — do NOT skip them.
- [Phase 1] Peak VRAM for so400m + LRP graph retention is unknown until first run; measure on the first single-image attribution before sizing any sweep. **STILL OPEN** — blocked by the engine op-coverage failure above (no successful relevance pass yet to measure peak VRAM from).
- [Phase 1] Rumsey URL health unknown; the per-id outcome manifest from ingestion is the only signal of how many maps are actually available — budget for a fraction being unavailable.

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| *(none)* | | | |

## Session Continuity

Last session: 2026-05-19T00:35:11.133Z
Stopped at: 01-03 BLOCKED at Rule 4 architectural decision (dynamicLRP op-coverage fails on SigLIP-2-so400m) — surfaced as the 01-03 human-verify checkpoint; awaiting human Fallback-Ladder selection (step 2 custom Promise vs step 5 captum IG)
Resume file: .planning/phases/01-pinned-environment-gcs-mirrors-and-a-verified-single-slice-a/01-03-PLAN.md
