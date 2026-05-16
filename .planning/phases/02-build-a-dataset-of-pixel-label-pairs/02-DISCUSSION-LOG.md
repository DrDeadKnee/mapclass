# Phase 02: build-a-dataset-of-pixel-label-pairs - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-16 (REWORK — supersedes the original 2026-05-15 log)
**Phase:** 02-build-a-dataset-of-pixel-label-pairs
**Areas discussed:** GCS write strategy, Raw-input naming contract, Train/eval data access, Scope: source families

**Trigger:** Data-loss incident discovered during the Phase 4 GPU-host gate —
synthetic dataset + raw Azgaar `.geojson` inputs lost everywhere (local-only
persistence). User decisions: accept full regeneration; reopen/rework Phase 2;
update context + replan all 5 plans (prior verification void).

---

## Rework scope (existing artifacts)

| Option | Description | Selected |
|--------|-------------|----------|
| Update context, replan all | Update CONTEXT.md, replan all 5 plans, void old verification | ✓ |
| Update context, keep plans | Update CONTEXT.md, salvage/patch existing 5 plans | |

**User's choice:** Update context, replan all
**Notes:** 5 prior plans encode reversed decisions (D-15..D-18, local paths) — full replan is correct for a frozen-decision reversal.

---

## GCS write strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Scratch + rsync, split.json early | Build to local scratch, write split.json to GCS at compute, rsync tree up on success | |
| Direct gcsfs streaming | Rewrite build_*/tiling.py to write every artifact straight to gs:// via gcsfs | ✓ |
| Scratch + rsync, split.json with tree | Build to scratch, rsync everything at the end | |

**User's choice:** Direct gcsfs streaming
**Notes:** Chosen over the recommended scratch+rsync with the invasiveness understood. CONTEXT.md flags the tiling.py path-abstraction refactor + small-object streaming throughput/cost as a mandatory researcher investigation before planning.

---

## Raw-input naming contract

| Option | Description | Selected |
|--------|-------------|----------|
| `<template>_<NN>.geojson` + explicit manifest | Filename convention + user-authored raw/manifest.json, build hard-fails on mismatch | ✓ |
| `<template>_<NN>.geojson` only | Convention only, rely on template_key regex | |
| Freeform + manifest is authority | Filenames irrelevant, manifest sole source of truth | |

**User's choice:** `<template>_<NN>.geojson` + explicit manifest (Recommended)
**Notes:** Belt-and-suspenders against a hand-created filename typo silently skewing the EVAL-01 stratified hold-out.

---

## Train/eval data access

| Option | Description | Selected |
|--------|-------------|----------|
| Pull-once to local cache, verify, then train | rsync split subset down at job start, verify vs split.json, train from disk | ✓ |
| Stream from gs:// every epoch | Read tiles via gcsfs each epoch | |
| Pull-once, no verification | rsync down once, skip verification | |

**User's choice:** Pull-once to local cache, verify, then train (Recommended)
**Notes:** Best throughput over many epochs; resumable after preemption; consistent with seg/gcs_checkpoint.py.

---

## Scope: source families

| Option | Description | Selected |
|--------|-------------|----------|
| Synthetic only now; others follow-up | Rework only the lost synthetic stream | |
| All three families GCS-canonical | synthetic + historical + satellite all GCS-canonical | ✓ |
| Synthetic reworked + persist others' outputs | Full synthetic rework + rsync historical/satellite outputs only | |

**User's choice:** All three families GCS-canonical
**Notes:** Maximally consistent, prevents recurrence everywhere. Widens the rework and delays Phase 4 — accepted.

---

## Claude's Discretion

- Exact gcsfs/fsspec abstraction shape, `raw/manifest.json` schema, and
  verification granularity (count vs checksum) — left to researcher/planner
  within RW-01..RW-03.

## Deferred Ideas

None — discussion stayed within phase scope. Lost Azgaar maps are
unrecoverable (full regeneration accepted, not deferred). Differing Phase 4
EVAL-01 hold-out is an accepted consequence (logged, not deferred).
