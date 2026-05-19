# Roadmap: MapClass

## Overview

MapClass delivers a working, repeatable attribution loop: pick a Rumsey map + a text query, get a dynamic-LRP heatmap overlaid on that map, judge it visually in JupyterLab — then scale that to a configurable sweep browsable as a contact sheet. The journey is two phases driven by a strict dependency chain. Phase 1 builds and *verifies correct* the single-slice primitive: pinned environment, idempotent GCS mirrors, loaders, SigLIP-2 load, the dynamic-LRP attribution (the project's central technical risk), heatmap reconstruction, and one inline overlay — gated not on "looks reasonable" but on three sanity controls (query-swap, model-randomization, occlusion). Phase 2 is pure orchestration over the proven primitive: a configurable maps × queries sweep counting down from a high manifest index, run caching, and a contact-sheet notebook browse. The single most important architectural invariant throughout: the same `requires_grad_()` image tensor must flow from the loader through the SigLIP-2 forward into `params_to_interpret` without being cloned or detached.

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Pinned Environment, GCS Mirrors, and a Verified Single-Slice Attribution** - One map + one query produces a correctly-aligned, control-verified heatmap inline in JupyterLab
- [ ] **Phase 2: Configurable Sweep, Run Caching, and Contact-Sheet Browse** - A configurable maps × queries sweep counting down from index N, browsable as a cached contact sheet (v1 done)

## Phase Details

### Phase 1: Pinned Environment, GCS Mirrors, and a Verified Single-Slice Attribution

**Goal**: A verified, correct single-slice attribution pipeline — one map + one query produces a properly-aligned heatmap overlaid on the source map, displayed inline in JupyterLab, that has passed query-swap, model-randomization, and occlusion sanity controls.
**Depends on**: Nothing (first phase)
**Requirements**: ENV-01, DATA-01, DATA-02, DATA-03, DATA-04, MODEL-01, ATTR-01, ATTR-02, ATTR-03, VIZ-01
**Success Criteria** (what must be TRUE):

  1. A fresh environment built from the pinned `requirements.txt` (torch==2.7.1, transformers==4.52.3) with keeinlev/dynamicLRP vendored at a fixed commit SHA runs the reference `ViT.ipynb` end-to-end and reproduces its heatmap (toolchain proven before the SigLIP-2 swap).
  2. Re-running image ingestion and the model mirror converges idempotently: SigLIP-2 weights and the high-index Rumsey map images are present in `gs://mapclass-training-northeast1/`, a per-id outcome manifest records every download's status, and no corrupt/zero-byte objects exist.
  3. Given a manifest index, the loaders return SigLIP-2 from the GCS mirror plus the *same* `requires_grad_()` image tensor (correctly preprocessed via the official SigLIP-2 processor, [-1,1] rescale, 384 squash) and the original PIL image for overlay.
  4. For a single (map, query) pair, the attribution engine runs SigLIP-2 forward + dynamic LRP against the `logits_per_image[0,0]` image-text similarity scalar (detached text embedding) and returns a per-patch relevance grid reconstructed as a 2D heatmap correctly handling the 27×27 grid and 384÷14 6-px edge discard.
  5. A heatmap overlay for the chosen (map, query) renders inline in a JupyterLab cell, aligned to the source map, and has passed all three sanity controls: an unrelated query changes the heatmap substantially, randomized vision weights collapse it to noise, and occluding top-relevance patches drops image-text similarity more than occluding random patches.

**Plans**: 3 plans
Plans:
**Wave 1**

- [x] 01-01-PLAN.md — Pinned environment + vendored dynamicLRP at the fixed SHA + reproduce reference ViT.ipynb heatmap (ENV-01)
- [x] 01-02-PLAN.md — Manifest reader + idempotent full 1,544 Rumsey + SigLIP-2 GCS mirrors + model/data loaders (DATA-01..04, MODEL-01)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 01-03-PLAN.md — Single-slice SigLIP-2 + dynamic-LRP attribution + 27x27 overlay + three sanity controls notebook (ATTR-01..03, VIZ-01) — **COMPLETE-WITH-FINDING:** pipeline/overlay code + 16 unit tests delivered; dynamicLRP does NOT cover SigLIP-2 (`split_with_sizes` MAP-pool op) so NO heatmap is produced; D-02/D-03 visual gate consciously WAIVED for SigLIP-2 and Fallback Ladder DECLINED by the user; recorded as a per-model finding for the reframed multi-model dynamic-LRP comparison. Not a clean pass.

**UI hint**: yes

### Phase 2: Configurable Sweep, Run Caching, and Contact-Sheet Browse

**Goal**: A configurable maps × queries sweep counting down from a high manifest index N, with relevance grids cached so the notebook re-renders a contact-sheet grid of all overlays without recompute. This is the v1 finish line.
**Depends on**: Phase 1
**Requirements**: SWEEP-01, SWEEP-02, VIZ-02
**Success Criteria** (what must be TRUE):

  1. A sweep configured with a start index N, a map count, and a query list iterates maps counting *down* from N over only successfully-mirrored ids, calling the unchanged Phase 1 primitive per (map, query) with per-item try/except so one bad map does not kill the run.
  2. Relevance grids are cached to disk/GCS keyed by (map id, query); re-running the sweep with the same config skips recompute and reloads from cache.
  3. All sweep overlays render inside the notebook as a labeled contact-sheet grid (each tile captioned with map id and query), and re-rendering the contact sheet from cache requires no model recompute.
  4. GPU memory returns to baseline between sweep iterations (LRP graph freed + `empty_cache()` each iteration), so a multi-map sweep completes without OOM mid-run.

**Plans**: TBD (target 1-2)
**UI hint**: yes

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Pinned Environment, GCS Mirrors, and a Verified Single-Slice Attribution | 3/3 | Plans done — 01-03 complete-with-finding (SigLIP-2 dynamicLRP op-coverage gap; success criteria 4-5 NOT met for SigLIP-2; D-02/D-03 user-waived; multi-model reframe pending PROJECT.md reconciliation) |  |
| 2. Configurable Sweep, Run Caching, and Contact-Sheet Browse | 0/TBD | Not started — gated on PROJECT.md/ROADMAP scope reconciliation against the multi-model reframe | - |
