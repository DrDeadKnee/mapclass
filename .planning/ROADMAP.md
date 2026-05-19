# Roadmap: MapClass — v1.1 Multi-Model Dynamic-LRP Comparison

## Overview

v1.1 turns the v1.0 SigLIP-2-only attribution loop into a cross-model comparison
harness. The journey: first define a model-agnostic adapter contract and lock the
two invisible-failure guards (tensor-identity + frozen-stack) so no model can be
wired in unsafely; then generalize the invariant-bearing overlay/attribution
modules and prove the seam is behavior-preserving by running SigLIP-2 behind it
as a regression oracle (must reproduce the v1.0 recorded `SplitWithSizesBackward0`
op-coverage finding); then add ViT (simplest, first positive heatmap, validates
non-SigLIP geometry), then CLIP and PaliGemma (research-flagged: CLIP CLS-token
geometry, PaliGemma answer-token target, OOM expected); finally compose all four
behind one notebook that renders the heatmaps side-by-side and completes headless
at exit 0 with failures shown as captioned "no heatmap" tiles — the milestone
done-gate. Per-model op-coverage gaps are recorded results, not bugs: no sweep,
no vendor patching, no pin bumps.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Adapter Contract + Pre-Flight Guards** - Model-agnostic seam, the 4 documented attribution targets, and the tensor-identity + frozen-stack guards land before any model is wired in
- [ ] **Phase 2: Parameterized Overlay + SigLIP-2 Regression Oracle** - Overlay/attribution generalized by patch geometry; SigLIP-2 runs behind the adapter and reproduces the v1.0 recorded op-coverage finding
- [ ] **Phase 3: ViT Adapter** - A plain ViT runs behind the adapter producing the first positive non-SigLIP heatmap, validating the parameterized overlay
- [ ] **Phase 4: CLIP + PaliGemma Adapters** - CLIP (contrastive contrast datum) and PaliGemma (generative, OOM-expected) run behind the adapter, weights mirrored to GCS
- [ ] **Phase 5: Comparison Notebook (Milestone Done-Gate)** - One notebook runs the locked map + query through all 4 models and renders heatmaps side-by-side, failures as captioned tiles, headless exit 0

## Phase Details

### Phase 1: Adapter Contract + Pre-Flight Guards
**Goal**: A model-agnostic adapter seam exists with the per-architecture attribution target resolved and documented for all 4 models, and the two invisible-failure guards (requires_grad tensor-identity, frozen-stack drift) are enforced — all before any model is wired in.
**Depends on**: Nothing (first phase)
**Requirements**: ADPT-01, ADPT-02, ADPT-03, ADPT-04
**Success Criteria** (what must be TRUE):
  1. A `ModelAdapter` interface (`load`, `build_inputs`, `forward`, `attribution_target`, `patch_geometry`) and a `PatchGeometry` dataclass exist and are importable
  2. The attribution target for each of the 4 models is documented in-repo (SigLIP-2/CLIP `logits_per_image[0,0]`; ViT class logit; PaliGemma answer-token logit) before any harness loop is written
  3. A conformance test fails if the `requires_grad` pixel tensor is not the same Python object through `build_inputs → forward → engine.params_to_interpret`
  4. A pre-flight assertion fails loudly if `transformers != 4.52.3`, torch not 2.7.x, `VENDOR_SHA != 405e74243ecaa1f615f418fdc8ba24c3c5889b1e`, `third_party/dynamicLRP` is modified, or `requirements.txt` changed
**Plans**: TBD

### Phase 2: Parameterized Overlay + SigLIP-2 Regression Oracle
**Goal**: The invariant-bearing overlay and attribution modules are generalized by `PatchGeometry` (no hardcoded 27×27) with D-09 signed/zero-centered rendering preserved, and SigLIP-2 runs behind the adapter reproducing the identical v1.0 recorded op-coverage finding — proving the seam is behavior-preserving on the only fully-characterized model before any new model is added.
**Depends on**: Phase 1
**Requirements**: ATTR-01, ATTR-02, ATTR-03, MODEL-01
**Success Criteria** (what must be TRUE):
  1. `overlay.py` reconstructs the patch grid from a model's `PatchGeometry` (patch size / image dim / CLS-token presence) while preserving the D-09 signed / zero-centered `TwoSlopeNorm` rendering
  2. `attribution.attribute(adapter, …)` runs the forward and `engine.run` in one scope with the adapter-supplied target; `engine.run` is never called inside an adapter
  3. The existing v1.0 unit tests still pass and `test_overlay_grid` is parametrized over 27/16/14 geometries
  4. SigLIP-2 runs behind the adapter and reproduces the identical v1.0 recorded op-coverage finding (`SplitWithSizesBackward0`, caught RuntimeError, exit 0) — a divergence signals seam breakage or stack drift
**Plans**: TBD

### Phase 3: ViT Adapter
**Goal**: A plain ViT runs behind the adapter and produces the first positive non-SigLIP attribution heatmap (or a recorded coverage result), validating the parameterized overlay against non-SigLIP geometry (14×14 + CLS-token strip).
**Depends on**: Phase 2
**Requirements**: MODEL-02
**Success Criteria** (what must be TRUE):
  1. A plain ViT (`google/vit-base-patch16-224`) loads behind the adapter with weights mirrored to GCS
  2. The class-logit attribution target and the documented query→class mapping are used for the locked map + query
  3. The ViT run produces an attribution heatmap rendered on its correct 14×14 + CLS-strip geometry, OR a recorded op-coverage result if dynamic LRP cannot traverse it
**Plans**: TBD
**UI hint**: yes

### Phase 4: CLIP + PaliGemma Adapters
**Goal**: CLIP (contrastive, same `logits_per_image` API as SigLIP-2 but CLS-pool head — the designed informative contrast) and PaliGemma (generative VLM, gated weights, OOM/coverage failure expected) each run behind the adapter with weights mirrored to GCS, producing a heatmap or a recorded coverage/VRAM result. CLIP is wired before PaliGemma (simpler, more likely to traverse; PaliGemma is the hardest case and runs last).
**Depends on**: Phase 3
**Requirements**: MODEL-03, MODEL-04
**Success Criteria** (what must be TRUE):
  1. CLIP loads behind the adapter (weights mirrored to GCS) with `logits_per_image[0,0]` target and correct 16×16 + CLS-strip geometry, producing a heatmap or a recorded coverage result
  2. PaliGemma loads behind the adapter with gated weights mirrored to GCS (HF token + Gemma license at mirror time) and the documented answer-token logit target, producing a heatmap or a recorded VRAM/coverage result
  3. Whether CLIP traverses where SigLIP-2 fails is observable as a recorded comparison datum (no vendor patching, no pin bumps to force traversal)
**Plans**: TBD
**Research-flagged**: yes — CLIP CLS-token strip geometry must be validated against real CLIP relevance (not synthetic gradients); PaliGemma teacher-forced answer-token target (`answer_pos`/`answer_token_id`) for the locked query is a per-query modeling decision resolved at the Phase 1 target-decision step
**UI hint**: yes

### Phase 5: Comparison Notebook (Milestone Done-Gate)
**Goal**: One notebook runs the locked map + a text query through all 4 models sequentially (load → attribute → free) without OOM on the L4 and renders their per-model attribution overlays side-by-side; a model whose dynamic-LRP fails renders a labeled "no heatmap" tile and the notebook completes headless at exit 0 with ≥4 tiles — the milestone done-gate.
**Depends on**: Phase 4
**Requirements**: CMP-01, CMP-02, CMP-03
**Success Criteria** (what must be TRUE):
  1. One notebook runs the locked map + a text query through all 4 models sequentially (load → attribute → free) without OOM on the L4
  2. The 4 per-model attribution overlays render side-by-side, each panel labeled with the model and what its query binds to
  3. A model whose dynamic-LRP fails (op-coverage gap or OOM) renders a labeled "no heatmap" tile instead of crashing
  4. The notebook completes headless via nbconvert at exit 0 with ≥4 tiles — **milestone done**
**Plans**: TBD
**UI hint**: yes

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Adapter Contract + Pre-Flight Guards | 0/TBD | Not started | - |
| 2. Parameterized Overlay + SigLIP-2 Regression Oracle | 0/TBD | Not started | - |
| 3. ViT Adapter | 0/TBD | Not started | - |
| 4. CLIP + PaliGemma Adapters | 0/TBD | Not started | - |
| 5. Comparison Notebook (Milestone Done-Gate) | 0/TBD | Not started | - |
