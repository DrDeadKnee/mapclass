# Requirements — Milestone v1.1: Multi-Model × Multi-Map Attribution Notebook

**Defined:** 2026-05-19 (simplified per user — drop the adapter/guard scaffold)

**Core value:** One notebook that runs dynamic-LRP attribution for several
models against several maps and shows the heatmaps, so a human can eyeball
them. Models and maps both come from the GCS buckets in the README. A model
that dynamic LRP can't traverse just shows "no heatmap" — not a crash.

## v1.1 Requirements

### Models (MODEL)

- [ ] **MODEL-01**: SigLIP-2, CLIP, ViT-b-16, and PaliGemma-3B all load from `gs://mapclass-training-northeast1/models/` (SigLIP-2 already mirrored; PaliGemma already at `models/paligemma-3b-mix-224/`; CLIP + ViT-b-16 mirrored as needed)
- [ ] **MODEL-02**: Each model has a query-conditioned attribution target wired (SigLIP-2/CLIP `logits_per_image`; ViT class logit; PaliGemma answer-token logit) — plain per-model functions, no framework

### Attribution & Overlay (ATTR)

- [ ] **ATTR-01**: `attribution` runs forward + dynamic LRP in one scope per model, preserving the requires_grad image-tensor identity (no clone/detach)
- [ ] **ATTR-02**: `overlay` reconstructs each model's patch grid from its own patch size (not hardcoded 27×27), keeping the v1.0 D-09 signed / zero-centered rendering

### Notebook (NB)

- [ ] **NB-01**: One notebook iterates the last 50 Rumsey manifest maps (loaded from `gs://.../data/`) × the 4 models, running attribution per (map, model)
- [ ] **NB-02**: Results render as labeled attribution overlays grouped per map across models; a model/map whose LRP fails (op-coverage or OOM) shows a captioned "no heatmap" tile and the run continues
- [ ] **NB-03**: The notebook completes headless via nbconvert at exit 0 — **milestone done**

## Future Requirements (deferred)

- More models (Florence-2, SAM 3, VGG16 from the README candidate list)
- Configurable map selection beyond "last 50"
- Run caching / contact-sheet styling
- Quantitative attribution metrics

## Out of Scope (explicit)

- **Adapter Protocol / PatchGeometry dataclass / conformance-test & pre-flight-guard infrastructure** — over-engineered for one notebook; use plain per-model functions
- **Fixing per-model coverage gaps** (custom Promises / LXT / captum) — a model the engine can't traverse just shows "no heatmap"; `VENDOR_SHA` stays intact
- **Bumping the frozen pins** (torch 2.7.1 / transformers 4.52.3) to satisfy a model
- Fine-tuning/training; labeller/dataset export; non-Rumsey datasets

## Traceability

| REQ-ID | Phase | Status |
|--------|-------|--------|
| MODEL-01 | Phase 1 | Pending |
| MODEL-02 | Phase 1 | Pending |
| ATTR-01 | Phase 1 | Pending |
| ATTR-02 | Phase 1 | Pending |
| NB-01 | Phase 1 | Pending |
| NB-02 | Phase 1 | Pending |
| NB-03 | Phase 1 | Pending |
