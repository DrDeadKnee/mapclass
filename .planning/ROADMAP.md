# Roadmap: MapClass — v1.1 Multi-Model × Multi-Map Attribution Notebook

## Overview

v1.1 is deliberately small: one notebook that runs dynamic-LRP attribution for
four models (SigLIP-2, CLIP, ViT-b-16, PaliGemma-3B) against the last 50 Rumsey
maps and renders the heatmaps for visual eyeballing. Models and maps both come
from `gs://mapclass-training-northeast1/` (models/ and data/) per the README.
v1.0 already shipped the pinned env, the Rumsey image mirror + loaders, and the
SigLIP-2 attribution + D-09 signed/zero-centered overlay — this milestone reuses
all of it and only generalizes model loading, the per-model attribution target,
and the patch-grid geometry with plain per-model functions (no adapter Protocol,
no guard/oracle phases). A model dynamic LRP can't traverse just shows a
captioned "no heatmap" tile; coverage gaps are recorded results, not bugs to fix
(VENDOR_SHA stays intact, frozen pins unchanged).

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Multi-Model × Multi-Map Attribution Notebook** - Load 4 models + last-50 maps from GCS, run dynamic-LRP per (map, model), render overlays in one notebook with failures as captioned tiles

## Phase Details

### Phase 1: Multi-Model × Multi-Map Attribution Notebook

**Goal**: One notebook loads SigLIP-2, CLIP, ViT-b-16, and PaliGemma-3B from the GCS models/ bucket and the last 50 Rumsey maps from the GCS data/ bucket, runs dynamic-LRP attribution for every (map, model) pair reusing the v1.0 attribution/overlay code (generalized per-model with plain functions), and renders the per-model overlays grouped per map for visual eyeballing — a failing (map, model) shows a captioned "no heatmap" tile and the run continues, completing headless at exit 0.
**Depends on**: Nothing (reuses v1.0 infrastructure: pinned env, image mirror, loaders, SigLIP-2 attribution + D-09 overlay)
**Requirements**: MODEL-01, MODEL-02, ATTR-01, ATTR-02, NB-01, NB-02, NB-03
**Success Criteria** (what must be TRUE):
  1. SigLIP-2, CLIP, ViT-b-16, and PaliGemma-3B all load from `gs://mapclass-training-northeast1/models/` (CLIP + ViT-b-16 mirrored as needed; PaliGemma read from the existing `models/paligemma-3b-mix-224/`)
  2. Each model's query-conditioned attribution target is wired (SigLIP-2/CLIP `logits_per_image`; ViT class logit; PaliGemma answer-token logit) with the requires_grad image-tensor identity preserved through forward + dynamic LRP
  3. `overlay` reconstructs each model's patch grid from its own patch size (not hardcoded 27×27), preserving the v1.0 D-09 signed / zero-centered rendering
  4. One notebook iterates the last 50 manifest maps × the 4 models, rendering labeled overlays grouped per map; a (map, model) whose LRP fails shows a captioned "no heatmap" tile and the run continues
  5. The notebook completes headless via nbconvert at exit 0 — **milestone done**
**Plans**: TBD
**Research-flagged**: light — PaliGemma answer-token target choice and CLIP/ViT patch-grid + CLS-token handling are the only non-obvious bits; resolve inline during planning, no separate research phase
**UI hint**: yes
