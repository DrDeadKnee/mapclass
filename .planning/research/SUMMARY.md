# Project Research Summary

**Project:** MapClass — v1.1 Multi-Model Dynamic-LRP Comparison
**Domain:** ML interpretability research piping (cross-model attribution comparison)
**Researched:** 2026-05-19
**Confidence:** HIGH

## Executive Summary

MapClass v1.1 is a research comparison harness: one locked Rumsey map + one text query, run through four vision models (SigLIP-2, CLIP, plain ViT, PaliGemma) via the vendored `keeinlev/dynamicLRP` engine, rendered side-by-side in a notebook so a researcher can eyeball whether and where each model attributes. The deliverable is honest piping — a model dynamicLRP cannot traverse shows a labeled "no heatmap" tile, not a crash. The comparison itself is the product.

The frozen stack (torch==2.7.1, transformers==4.52.3, vendored dynamicLRP SHA 405e74243ecaa1f615f418fdc8ba24c3c5889b1e) carries forward unchanged; CLIP and plain ViT add zero new dependencies; PaliGemma adds only a one-time mirror-time HF token + Gemma-license requirement.

**The single design decision that must be made before any harness code is written** is the per-architecture attribution target — the four models do not share one. All four researchers converged on this independently; getting it wrong silently produces query-independent heatmaps. The two v1.0 load-bearing invariants must survive generalization: (1) the `requires_grad_()` pixel tensor must remain the same Python object through `build_inputs → forward → engine.params_to_interpret` (the engine walks the live autograd graph by object identity); (2) the D-09 signed/zero-centered `TwoSlopeNorm` overlay must be parameterized by patch geometry, not forked per-model.

## Key Findings

### Recommended Stack

Zero `requirements.txt` delta. All four model classes are native to the frozen `transformers==4.52.3`; no pin changes, no `sentencepiece`, `accelerate` optional OOM-fallback only.

**Core technologies:**
- `CLIPModel` / `CLIPProcessor` (`openai/clip-vit-large-patch14`, ~428M, not gated) — contrastive comparison datum; same `logits_per_image` API as SigLIP-2 but CLS-pool head
- `ViTForImageClassification` / `ViTImageProcessor` (`google/vit-base-patch16-224`, ~86M, not gated) — dynamicLRP's own example architecture; the "method works at all" reference
- `PaliGemmaForConditionalGeneration` / `PaliGemmaProcessor` (`google/paligemma2-3b-pt-224`, ~3B, **GATED** — HF token + Gemma license at mirror time) — generative VLM; ~7.5× the so400m ceiling, OOM/coverage failure expected and recordable
- Existing SigLIP-2-so400m path — retained, lifted behind the adapter as the regression oracle

### Expected Features

**Must have (table stakes):**
- Model-agnostic adapter contract (`load / build_inputs / forward / attribution_target / patch_geometry`)
- Per-architecture attribution target resolved & documented BEFORE the harness loop
- Per-model patch-grid reconstruction derived from the processor (not hardcoded 27×27; CLS-token strip for CLIP/ViT)
- Graceful per-model degradation → labeled "no heatmap (op-coverage gap)" tile, notebook exit 0
- 4-panel side-by-side notebook with per-panel "what the query binds to" semantics labels

**Should have (competitive):**
- SigLIP-2 vs CLIP framed as the designed informative contrast (same API, different pooling head)

**Defer / anti-features (explicitly out of scope):**
- Op-coverage report table (user chose "overlays only"), maps×queries sweep, quantitative metrics, LRP-param tuning, per-model sanity controls, and **fixing coverage gaps / patching the vendored engine** (Fallback Ladder declined in v1.0)

### Architecture Approach

One new seam absorbs all four model-specific concerns; `manifest`, `ingest_images`, and the vendored engine are reused verbatim.

**Major components:**
1. `ModelAdapter` Protocol + `PatchGeometry` dataclass — the per-model seam (load, inputs, forward, target, geometry)
2. `attribution.attribute(adapter, …)` — owns forward + `engine.run` in ONE scope (engine.run stays OUT of the adapter to preserve tensor identity)
3. Parameterized `overlay.py` — D-09 rendering untouched; only grid dims injected via `PatchGeometry`
4. Ordered model registry `[siglip2, vit, clip, paligemma]` + generalized `mirror_model.py` + `02_multimodel.ipynb`

### Critical Pitfalls

1. **Breaking the requires_grad tensor-identity invariant** — a per-model `.to()`/cast/clone creeping in after `requires_grad_()` silently yields wrong/empty relevance (no error). Prevention: `is`-identity conformance test in Phase 1; engine.run never inside the adapter.
2. **Hardcoded 27×27/14/384 geometry** — silently scrambles (doesn't crash) CLIP/ViT overlays; compounded by the CLS token present in CLIP/ViT but absent in SigLIP-2. Prevention: `PatchGeometry` from processor config + known-answer fixtures parametrized over 27/14/16.
3. **Wrong/ambiguous attribution target** — pooled embeds for ViT/CLIP, undocumented PaliGemma answer-token. Prevention: resolve all four targets as a Phase 1 decision step, label each panel.
4. **Stack/scope drift** — bumping a pin or patching the vendor to satisfy a model, or "fixing" a coverage gap. Prevention: pre-flight assertion suite (`transformers==4.52.3`, torch 2.7.1, `VENDOR_SHA==405e7424…`, `git diff --quiet third_party/dynamicLRP`, `requirements.txt` unchanged) in Phase 1.
5. **Notebook not reaching exit 0** when a model fails — uncaught traceback vs caught FINDING tile. Prevention: sequential load/`del`/`empty_cache` in `finally`, broad `except Exception` → captioned tile (generalize the proven v1.0 pattern).

## Implications for Roadmap

Based on research, suggested phase structure (7 phases, all 4 researchers converged on this order):

### Phase 1: Adapter Contract + Pre-Flight Guards
**Rationale:** The contract and the per-architecture target decision must exist before any model can plug in; the pin/SHA + tensor-identity guards prevent the two invisible-failure classes.
**Delivers:** `ModelAdapter` Protocol, `PatchGeometry`, the documented 4 attribution targets, `test_adapters.py` (`is`-identity assertion), pre-flight pin/SHA assertion suite.
**Avoids:** Pitfalls 1, 3, 4.

### Phase 2: Parameterize Overlay + Attribution
**Rationale:** Invariant-bearing modules generalized before any model exercises them.
**Delivers:** `overlay.py` accepting `PatchGeometry`; `attribution.attribute(adapter, …)`; `test_overlay_grid.py` parametrized over 27/14/16; existing 16 unit tests still green.
**Implements:** Components 2 & 3.

### Phase 3: SigLIP-2 Adapter (Regression Oracle)
**Rationale:** Proves the seam is behavior-preserving on the only fully-characterized model; a divergent finding signals stack drift or a broken seam.
**Delivers:** SigLIP-2 behind the adapter; must reproduce the identical v1.0 recorded op-coverage finding (`SplitWithSizesBackward0`, caught RuntimeError, exit 0).

### Phase 4: ViT Adapter (First Positive Heatmap)
**Rationale:** Highest a-priori coverage probability (dynamicLRP's own example architecture); first positive heatmap validates the parameterized overlay with non-SigLIP geometry.
**Delivers:** `ViTForImageClassification`, class-logit target, documented query→class mapping, 14×14 + CLS strip.

### Phase 5: CLIP Adapter (Contrastive Comparison Datum)
**Rationale:** Same API as SigLIP-2, different pooling head — the designed informative contrast; whether CLIP traverses where SigLIP-2 fails is the primary finding.
**Delivers:** `CLIPModel`, `logits_per_image[0,0]`, 16×16 + CLS strip.

### Phase 6: PaliGemma Adapter (Generative VLM, Hardest Case)
**Rationale:** Hardest adapter; VRAM/coverage failure expected and recorded; runs last in registry.
**Delivers:** `PaliGemmaForConditionalGeneration`, answer-token logit target, bf16, gated weights.

### Phase 7: Mirror Registry + Comparison Notebook (Milestone Done-Gate)
**Rationale:** Composes verified parts; the side-by-side notebook is the milestone gate.
**Delivers:** `config.py` registry, generalized `mirror_model.py`, `_build_02_multimodel.py` + `02_multimodel.ipynb`; headless nbconvert exit 0, ≥4 tiles, failures as captioned tiles.

### Phase Ordering Rationale

- Contract + guards first because both invariants and the target decision gate everything downstream.
- SigLIP-2 before new models: it is the only regression oracle (known recorded finding).
- ViT before CLIP/PaliGemma: simplest, most likely to traverse, validates the parameterized overlay early.
- PaliGemma last: hardest + most likely to OOM; a late failure must not starve earlier panels.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 6 (PaliGemma):** teacher-forced answer-token target (`answer_pos`/`answer_token_id`) for the locked query — a per-query modeling decision; resolve at the Phase 1 target-decision step.
- **Phase 5 (CLIP):** CLS-token strip is the single most-overlooked detail; validate the geometry fixture against real CLIP relevance, not just synthetic gradients.

Phases with standard patterns (skip research-phase):
- **Phase 1** (Protocol/dataclass), **Phase 2** (refactor gated by 16 tests), **Phase 3** (lifts v1.0 verbatim), **Phase 4** (dynamicLRP's own `ViT.ipynb` is the reference), **Phase 7** (composes verified parts; `_build_01` is the template).

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Classes/ids verified against transformers v4.52.3 docs; PaliGemma gating from model card |
| Features | HIGH | Tied to PROJECT.md v1.1 Active/Out-of-Scope + the 01-03 user decision |
| Architecture | HIGH | Derived from the four hard-wire points read in actual v1.0 source + `lrp.py` |
| Pitfalls | HIGH | Every pitfall traces to a directly-read v1.0 mechanism or recorded decision |

**Overall confidence:** HIGH

### Gaps to Address

- **Per-model dynamicLRP op coverage** — intentionally UNKNOWN; this is the data v1.1 produces, not a gap to pre-resolve. CLIP coverage outcome is the experiment's primary observable.
- **PaliGemma answer-token** for query `"a river"` — decide at Phase 1 target-decision step.
- **CLIP size fallback** (`clip-vit-base-patch32` if L/14 OOMs the LRP pass) — decide at Phase 5 if triggered.
- **PaliGemma / CLIP vision-tower patch geometry & CLS presence** — derive from processor config at adapter-load time; verify in Phases 5–6, do not hardcode.

## Sources

### Primary (HIGH confidence)
- v1.0 source read directly: `src/mapclass/{attribution,overlay,data_loader,model_loader,config}.py`, vendored `lrp.py` (`make_graph_iter` identity resolution)
- transformers v4.52.3 model docs (CLIP, ViT, PaliGemma/paligemma2 classes & processors)
- PaliGemma HF model card (gated + Gemma license)
- v1.0 `01-03-SUMMARY.md` / `01-03-PLAN.md` (recorded SigLIP-2 op-coverage finding, caught-error→exit-0 pattern), `PROJECT.md` v1.1 decisions

### Secondary (MEDIUM confidence)
- Parameter-count aggregators (CLIP-L/14 ~428M); exact processor/patch configs to confirm per-adapter at build time

---
*Research completed: 2026-05-19*
*Ready for roadmap: yes*
