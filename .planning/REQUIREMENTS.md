# Requirements — Milestone v1.1: Multi-Model Dynamic-LRP Comparison

**Defined:** 2026-05-19

**Core value:** A working, repeatable cross-model comparison — one locked map + a
text query, dynamic-LRP attribution through SigLIP-2 / CLIP / PaliGemma / a plain
ViT, heatmaps side-by-side in a notebook, judged visually. A model dynamic LRP
cannot traverse is a *recorded result*, not a failure.

## v1.1 Requirements

### Adapter & Guards (ADPT)

- [ ] **ADPT-01**: A model-agnostic `ModelAdapter` interface exists (`load`, `build_inputs`, `forward`, `attribution_target`, `patch_geometry`) that any of the 4 models plugs into
- [ ] **ADPT-02**: The per-architecture attribution target is resolved and documented for all 4 models before any harness loop is written (SigLIP-2/CLIP `logits_per_image[0,0]`; ViT class logit; PaliGemma answer-token logit)
- [ ] **ADPT-03**: A conformance test asserts the `requires_grad` pixel tensor is the same object through `build_inputs → forward → engine.params_to_interpret` (no clone/detach/cast)
- [ ] **ADPT-04**: A pre-flight assertion fails loudly if the frozen stack drifts (`transformers==4.52.3`, `torch` 2.7.x, `VENDOR_SHA==405e74243ecaa1f615f418fdc8ba24c3c5889b1e`, `third_party/dynamicLRP` unmodified, `requirements.txt` unchanged)

### Attribution & Overlay (ATTR)

- [ ] **ATTR-01**: `overlay.py` reconstructs the patch grid from per-model `PatchGeometry` (patch size / image dim / CLS-token presence) instead of hardcoded 27×27, preserving the D-09 signed / zero-centered `TwoSlopeNorm` rendering
- [ ] **ATTR-02**: `attribution.attribute(adapter, …)` runs the forward + `engine.run` in one scope with the adapter-supplied target; `engine.run` is never called inside an adapter
- [ ] **ATTR-03**: The existing v1.0 unit tests still pass and `test_overlay_grid` is parametrized over the new geometries (27/16/14)

### Comparison Models (MODEL)

- [ ] **MODEL-01**: SigLIP-2 runs behind the adapter and reproduces the identical v1.0 recorded op-coverage finding (regression oracle)
- [ ] **MODEL-02**: A plain ViT runs behind the adapter, weights mirrored to GCS, producing an attribution heatmap (or a recorded coverage result)
- [ ] **MODEL-03**: CLIP runs behind the adapter, weights mirrored to GCS, producing an attribution heatmap (or a recorded coverage result)
- [ ] **MODEL-04**: PaliGemma runs behind the adapter, gated weights mirrored to GCS (HF token + Gemma license at mirror time), producing an attribution heatmap (or a recorded VRAM/coverage result)

### Comparison Surface (CMP)

- [ ] **CMP-01**: One notebook runs the locked map + a text query through all 4 models sequentially (load → attribute → free) without OOM on the L4
- [ ] **CMP-02**: The 4 per-model attribution overlays render side-by-side, each panel labeled with the model and what its "query" binds to
- [ ] **CMP-03**: A model whose dynamic-LRP fails (op-coverage gap or OOM) renders a labeled "no heatmap" tile; the notebook completes headless at exit 0 with ≥4 tiles — **milestone done**

## Future Requirements (deferred)

- Structured per-model op-coverage report table (user chose "overlays only" for v1.1)
- Additional models beyond the 4
- Quantitative attribution metrics / faithfulness scoring

## Out of Scope (explicit)

- **Maps × queries sweep + contact-sheet browse** — comparison axis is *model*, not *map count*; dropped from v1.0 Phase 2 (never built)
- **Fixing per-model coverage gaps** (custom Promises / pre-pool / LXT / captum fallback) — a model the engine can't traverse is the *result*; Fallback Ladder declined in v1.0; `VENDOR_SHA` must stay intact
- **Bumping the frozen pins** to satisfy a model — the stack is immovable; a model that needs a newer transformers/torch is out of scope
- LRP method/parameter tuning as an explore axis; per-model sanity controls; fine-tuning/training; labeller/dataset export; non-Rumsey datasets

## Traceability

| REQ-ID | Phase | Status |
|--------|-------|--------|
| *(filled by roadmap)* | | |
