# Roadmap: MapClass

## Overview

MapClass progresses through four sequenced phases, with PaliGemma-3B threading
through all of them. Phase 1 adapts a frozen-Gemma / LoRA-SigLIP PaliGemma to
illustrated map terrain symbols. Phase 2 uses that fine-tune (plus Allmaps-based
georeferencing and ESA WorldCover + Copernicus DEM labels) to assemble a
pixel-label dataset from three source families: historical illustrated maps,
synthetic illustrated maps from Azgaar + Pillow, and satellite-derived imagery.
Phase 3 builds a dense semantic segmentation pipeline on the Phase-1 SigLIP
encoder with two lightweight output heads (land cover + topography) and
coarse-to-fine inference, with DINOv2 and Swin as benchmark baselines. Phase 4
fine-tunes the segmentation model end-to-end on the Phase-2 dataset, evaluates
it on held-out synthetic maps using joint per-pixel NLL, and publishes the
result to HuggingFace as the project's deliverable.

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3, 4): planned milestone work
- Decimal phases (e.g., 2.1): urgent insertions (none yet)

- [x] **Phase 1: Fine-tune PaliGemma-3B on illustrated map terrain symbols** - LoRA-adapt the SigLIP vision encoder to ~120 hex terrain tiles (9 land cover + 3 topography classes) augmented to ~4,760 samples; Gemma frozen
- [x] **Phase 2: Build a dataset of pixel-label pairs** - Produce labelled training data from historical illustrated maps, synthetic illustrated maps (Azgaar + Pillow), and satellite imagery (ESA WorldCover + Copernicus DEM)
- [x] **Phase 3: Build a dense semantic segmentation pipeline** - Construct the dense segmentation model on the Phase-1 SigLIP encoder with two lightweight heads and coarse-to-fine inference; benchmark DINOv2 and Swin
- [ ] **Phase 4: Fine-tune and evaluate the segmentation model** - Train the A/B × SigLIP/DINOv2/Swin matrix end-to-end on the Phase-2 dataset (cost-probe-gated breadth), evaluate on held-out synthetic maps with joint per-pixel NLL, produce a comparison report — stops at numbers + checkpoints + report
- [ ] **Phase 5: Explore, select, and publish the segmentation model** - Hands-on exploration of the trained models/dataset, select the deliverable, publish to HuggingFace with a model card (inference instructions + joint-NLL)

## Phase Details

### Phase 1: Fine-tune PaliGemma-3B on illustrated map terrain symbols

**Goal**: Adapt PaliGemma's SigLIP vision encoder (via LoRA on `q_proj`, `k_proj`,
`v_proj`, `out_proj` only; Gemma fully frozen; multi-modal projector fully
trainable) to recognise illustrated map terrain symbols, using ~120 hex tile
PNGs across 9 land cover classes and 3 topography classes, augmented across four
visual styles into 2×2 region composites paired with five natural-language
prompt templates (~4,760 samples total).
**Depends on**: Nothing (first phase)
**Requirements**: PHASE-01
**Success Criteria** (what must be TRUE):

  1. `scripts/finetune_paligemma.py` runs end-to-end and produces a LoRA-adapted
     checkpoint where Gemma weights are unchanged

  2. LoRA adapters are present ONLY on the SigLIP attention projections
     (`q_proj`, `k_proj`, `v_proj`, `out_proj`); the multi-modal projector is
     trained; Gemma carries no LoRA and no base-weight updates

  3. On held-out hex tiles, the fine-tuned model identifies the correct land
     cover and topography class for each tile via the trained prompt templates

  4. Place names and geographic terminology behaviour from the base Gemma is
     preserved (zero-forgetting check)
**Plans**: complete
**Status**: complete

### Phase 2: Build a dataset of pixel-label pairs

**Goal**: Produce labelled training data from three source streams —
(a) historical illustrated maps from the 16th–17th c. David Rumsey collection,
georeferenced via Allmaps where possible and via PaliGemma-driven semi-automatic
registration (cross-correlation + TPS) as fallback, labelled with ESA WorldCover

+ Copernicus DEM and stamped with class-conditional per-source loss weights;

(b) synthetic illustrated maps from Azgaar's Fantasy Map Generator rendered with
Pillow into auto-labelled `image.png` / `land_cover.png` / `topography.png` triplets;
(c) satellite-derived imagery from ESA WorldCover + Copernicus DEM GLO-30 mapped
to the canonical 9-class taxonomy with the locked schema rules
(Mangroves→Trees, Moss/lichen→Bare/sparse).
**Depends on**: Phase 1 (semi-automatic georeferencing of unregistered maps
consumes the Phase-1 fine-tuned PaliGemma)
**Requirements**: PHASE-02, EVAL-01
**Success Criteria** (what must be TRUE):

  1. The historical pipeline produces, per map, `image.png`, `land_cover.png`,
     `topography.png`, and `sample_weights.json` (with topography fully trusted;
     water / coastlines mostly trusted; bare/sparse and snow/ice reliable;
     trees / built-up / cropland heavily downweighted)

  2. Registered historical maps come from Allmaps directly; unregistered maps
     are emitted to `unregistered_manifest.json` for v2 processing (manual
     fallback + PaliGemma semi-auto deferred), all registered output as
     georeferenced GeoTIFF in EPSG:4326

  3. The synthetic pipeline produces matched `image.png` / `land_cover.png` /
     `topography.png` triplets from Azgaar + Pillow with auto-generated ground
     truth and re-normalised height thresholds (flat ≤20, hilly 20–55,
     mountainous >55 over `[0, 100]`)

  4. The satellite pipeline pulls ESA WorldCover from `s3://esa-worldcover` and
     Copernicus DEM GLO-30 from `s3://copernicus-dem-30m` directly (no GEE in
     the critical path) and re-encodes them in the canonical 9-class /
     3-class taxonomy

  5. A held-out synthetic subset is reserved as the Phase 4 test set and
     guaranteed unseen during training
**Plans**: 5 plans (REPLANNED 2026-05-16 — GCS-canonical rework; prior verification VOID, see _pre-rework-archive/)
Plans:
**Wave 1**

- [x] 02-01-PLAN.md — Wave 0: scripts/gcs_io.py (_GCSWriter/pull/verify/validate_manifest) + offline test infra
- [x] 02-02-PLAN.md — Wave 1: synthetic GCS migration — tiling.py 32-thread _GCSWriter (RW-01), manifest-before-split hard-fail (RW-02), GCS split.json (D-17/D-18 reversed), PROJECT.md update

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 02-03-PLAN.md — Wave 2: historical pipeline GCS-canonical (RW-04)
- [x] 02-04-PLAN.md — Wave 2: satellite pipeline GCS-canonical (RW-04)

**Wave 3** *(blocked on Wave 2 completion)*

- [ ] 02-05-PLAN.md — Wave 3: finetune/evaluate pull-once+verify (RW-03), _BUILD_COMPLETE sentinel + family-rooted layout (OQ1/OQ2), blocking human Azgaar-regeneration gate

**Cross-cutting constraints:**

- the existing ThreadPoolExecutor build path passes the shared fs through, thread-safe

**Status**: replanned (rework — execution pending; `/gsd:execute-phase 2`)

### Phase 3: Build a dense semantic segmentation pipeline

**Goal**: Construct a dense pixel-level segmentation model using the Phase-1
LoRA-adapted SigLIP encoder as the primary backbone, attached to two lightweight
output heads — land cover (9-class) and topography (3-class) — with
coarse-to-fine inference (multi-resolution sliding window on SigLIP/DINOv2 or
native via Swin) for global context plus boundary sharpness. Provide DINOv2 and
Swin Transformer backbones as non-locked benchmark alternatives; a large
first-kernel ConvNet trained from scratch is a stretch goal.
**Depends on**: Phase 1 (reuses the Phase-1 LoRA-adapted SigLIP weights), Phase 2
(needs the dataset shape locked in to size the heads and decoder)
**Requirements**: PHASE-03, EVAL-03
**Success Criteria** (what must be TRUE):

  1. The segmentation model accepts a regional map image (100–2000 km extent)
     and outputs a dense per-pixel land-cover probability tensor (9 classes) and
     a dense per-pixel topography probability tensor (3 classes)

  2. The primary backbone path loads the Phase-1 LoRA-adapted SigLIP weights;
     no additional Gemma weights are involved in the segmentation forward pass

  3. Coarse-to-fine inference runs end-to-end (either as a multi-resolution
     sliding window on the ViT-style backbones or natively via Swin's hierarchy),
     producing predictions that incorporate broad spatial context

  4. DINOv2 and Swin Transformer backbone variants run on the same input
     pipeline and are usable as benchmark baselines for Phase 4 evaluation
**Plans**: 5 plans
Plans:

- [x] 03-01-PLAN.md — Wave 0: timm + transformers-path resolution; seg test infra (mini-pyramid fixture + 6 scaffolds)
- [x] 03-02-PLAN.md — Backbone protocol + SigLIP/DINOv2/Swin + Variant-B widened 15-ch patch-embed (D-05/D-03a)
- [x] 03-03-PLAN.md — PyramidDataset over Phase-2 tree, train/-only, weights surfaced unchanged (D-06)
- [x] 03-04-PLAN.md — Shared UPerNet PPM+FPN conv decoder + two thin task heads (D-01/D-02)
- [x] 03-05-PLAN.md — SegModel assembly (both D-03a variants) + recursive c2f orchestrator + smoke (D-03/D-04)

**Status**: complete (UAT accepted-on-evidence 2026-05-16, 5/5 passed — see 03-UAT.md)

### Phase 4: Fine-tune and evaluate the segmentation model

**Goal**: Fine-tune the Phase-3 segmentation model end-to-end on the Phase-2
dataset (applying class-conditional loss weights for historical samples) and
evaluate on the held-out synthetic test set using joint per-pixel NLL =
-(log p_land_cover + log p_topography). Train the A/B × SigLIP/DINOv2/Swin
matrix with cost-probe-gated breadth, produce trained checkpoints + a written
comparison report. Stops before deliverable selection / publication (Phase 5).
**Reshaped 2026-05-16** (user decision, see 04-CONTEXT.md `<domain>`):
publication split out to a new Phase 5 so the user can hands-on test the
trained models/dataset before delivering.
**Depends on**: Phase 3
**Requirements**: EVAL-02 (computation only), EVAL-03
**Success Criteria** (what must be TRUE):

  1. The model is fine-tuned end-to-end on the Phase-2 dataset, honouring the
     per-sample `sample_weights.json` weights for historical maps

  2. Joint per-pixel NLL is computed on the held-out synthetic test set
     (averaged over pixels) and reported as the headline metric, with primary
     SigLIP and benchmark DINOv2 / Swin numbers side by side

  3. Trained checkpoints persist to `gs://mapclass-training-northeast1/models/`
     (per-config); a git-committed manifest of URIs + joint-NLL metrics JSON +
     a written A/B × backbone comparison report are produced

  4. A cost-probe (SigLIP Variant B) precedes the full grid; a manual
     checkpoint:decision gate sets grid breadth from measured GPU cost
**Plans**: 5 plans
Plans:
**Wave 1**

- [ ] 04-01-PLAN.md — Wave 0: 3 offline test files + train_utils (weighted joint loss, teacher-forced c2f step, AdamW groups, train→val carve)

**Wave 2** *(blocked on Wave 1 completion)*

- [ ] 04-02-PLAN.md — GCS checkpoint write/auto-resume module + mocked offline round-trip suite (D-07/D-09)
- [ ] 04-03-PLAN.md — Joint per-pixel NLL eval harness over test/ reusing recursive c2f; multi-backbone parity (EVAL-02/EVAL-03)

**Wave 3** *(blocked on Wave 2 completion)*

- [ ] 04-04-PLAN.md — finetune_seg.py training loop + cost-probe mode + deferred GPU-host gate (04-HUMAN-UAT.md)

**Wave 4** *(blocked on Wave 3 completion)*

- [ ] 04-05-PLAN.md — Blocking cost-probe decision gate (D-05) + comparison report renderer + metrics/manifest (D-08)

**Status**: planned

### Phase 5: Explore, select, and publish the segmentation model

**Goal**: Hands-on exploration of the Phase-4 trained models and the dataset by
the researcher, selection of the deliverable model, and publication to
HuggingFace as a public artifact with a model card (minimum inference
instructions + joint per-pixel NLL, optional DINOv2/Swin benchmark numbers).
**Depends on**: Phase 4
**Requirements**: PHASE-04, DELIV-01, DELIV-02, EVAL-02 (reported-with-published-model clause)
**Success Criteria** (what must be TRUE):

  1. The user has hands-on-explored the Phase-4 checkpoints/dataset and selected
     the deliverable model against a recorded criterion

  2. The selected model is uploaded to HuggingFace as a public artifact, with
     minimum inference instructions and pointers to the joint-NLL evaluation

  3. Loading the published model and running it on a fresh regional map image
     yields a dense pixel-level land-cover + topography prediction
**Plans**: TBD
**Status**: not_started

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Fine-tune PaliGemma-3B on illustrated map terrain symbols | 1/1 | Complete | (pre-bootstrap) |
| 2. Build a dataset of pixel-label pairs | 4/5 | In Progress|  |
| 3. Build a dense semantic segmentation pipeline | 5/5 | Complete | 2026-05-16 |
| 4. Fine-tune and evaluate the segmentation model | 0/TBD | Not started | - |
| 5. Explore, select, and publish the segmentation model | 0/TBD | Not started | - |
