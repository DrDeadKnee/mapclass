# Requirements: MapClass

**Defined:** 2026-05-18
**Core Value:** A working, repeatable loop — pick a map + a text query → get a dynamic-LRP attribution heatmap overlaid on that map → judge it visually — that scales to a configurable sweep of maps × queries browsable in a notebook.

## v1 Requirements

Requirements for the initial exploration pipeline. Each maps to a roadmap phase.

### Environment & Mirrors

- [ ] **ENV-01**: Reproducible Python environment pinned to dynamicLRP's `requirements.txt` (torch==2.7.1, transformers==4.52.3), with keeinlev/dynamicLRP vendored at a pinned commit SHA (git clone + sys.path; no pip package exists)
- [ ] **DATA-01**: Idempotently mirror Rumsey map images from the manifest `image_url`s into `gs://mapclass-training-northeast1/data/` (skip-if-exists, records failures, originals preserved)
- [ ] **DATA-02**: Mirror SigLIP-2-so400m-patch14-384 weights from HuggingFace into `gs://mapclass-training-northeast1/models/`

### Data Access

- [ ] **DATA-03**: Manifest access exposing an ordered index → entry resolution over the 1,544-entry manifest, supporting selection counting *down* from a high index N
- [ ] **DATA-04**: Image loader that resolves a manifest `id` to its GCS-mirrored object and returns a usable image, with a local byte cache to avoid repeated GCS reads in a sweep

### Model & Attribution

- [ ] **MODEL-01**: Load SigLIP-2-so400m once from the GCS model mirror with model-aligned preprocessing (resize/normalize to 384, patchify) that records the resize transform for later overlay alignment
- [~] **ATTR-01**: Single forward + dynamic-LRP attribution for one (map, query) pair, attributing the image↔detached-text similarity scalar (`logits_per_image`), reducing image-token relevance to a per-patch grid — **CODE DELIVERED, NOT MET FOR SigLIP-2 (01-03 FINDING):** `attribute()` is implemented correctly (forward + LRP one scope, `logits_per_image[0,0]`) but dynamicLRP does NOT cover SigLIP-2's `split_with_sizes` MAP-pool op, so it raises `RuntimeError` and produces NO relevance for SigLIP-2. User-accepted per-model finding for the reframed multi-model comparison; Fallback Ladder DECLINED.
- [~] **ATTR-02**: Reconstruct patch relevance into a 2D heatmap with a fixed normalization, correctly handling SigLIP-2's 27×27 patch grid and the 384÷14 non-integer edge discard — **CODE DELIVERED + UNIT-TESTED, NOT END-TO-END VERIFIED FOR SigLIP-2 (01-03 FINDING):** `overlay.py` (signed/magnitude 27×27 grid, 6-px discard, zero-centered composite) passes 16/16 units but is never exercised on real SigLIP-2 relevance (none is produced).
- [~] **ATTR-03**: Three near-zero-cost sanity controls (query-swap, model-randomization, occlusion) implemented as a Phase 1 correctness gate — a (map, query) attribution is only trusted if an unrelated query changes the heatmap, randomized vision weights collapse it to noise, and occluding top-relevance patches drops image-text similarity more than occluding random patches — **NOT MET FOR SigLIP-2 (01-03 FINDING):** the controls were authored but cannot be rendered/eyeballed because no SigLIP-2 relevance exists; D-02/D-03 visual-eyeball gate consciously WAIVED for SigLIP-2 by the user.

### Visualization

- [~] **VIZ-01**: Heatmap overlaid on the source map (alpha-blended), displayed inline in a JupyterLab cell for a single (map, query) slice — **Phase 1 finish line** — **NOT MET FOR SigLIP-2 (01-03 FINDING):** `composite()` is implemented + unit-tested but no SigLIP-2 heatmap is produced (dynamicLRP op-coverage gap); `01_single_slice.ipynb` instead executes as an honest end-to-end smoke test (forward + peak VRAM + caught coverage finding + source map shown). User-waived gate, recorded as a per-model finding.
- [ ] **VIZ-02**: Contact-sheet browse of all sweep overlays inside the notebook (grid of maps × queries with labels) — **v1 finish line**

### Sweep

- [ ] **SWEEP-01**: Configurable sweep over maps counting down from a high manifest index N (configurable start + count) crossed with a configurable query list, as pure orchestration over the ATTR/VIZ primitive
- [ ] **SWEEP-02**: Run caching of relevance grids keyed by (map id, query) to disk/GCS so the notebook re-renders contact sheets without recompute

## v2 Requirements

Acknowledged but deferred. Not in the current roadmap. Add only after the v1 loop is visually trusted.

### Diagnostics

- **DIAG-02**: Attribution sanity readout (top patches, score range) to detect silent adaptation failure
- **PROV-01**: Per-run provenance metadata sidecar (map id, query, model/LRP config, timestamp) recorded with each cached artifact
- **SWEEP-03**: Resumable / skip-if-cached sweep with `tqdm` progress

### Interaction

- **VIZ-03**: Side-by-side per-map query comparison view
- **VIZ-04**: Overlay tuning params surfaced (alpha, colormap, signed vs abs, percentile clip)
- **VIZ-05**: ipywidgets interactive query/map selector

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Quantitative attribution metrics (faithfulness/ABPC, pixel-flipping, IoU) | v1 judgment is purely visual; metrics need ground-truth masks that don't exist yet |
| Model swapping as a first-class axis (Florence-2, PaliGemma, ViT…) | Model fixed to SigLIP-2 in v1; each model is its own LRP-adaptation risk |
| LRP method/parameter tuning as a first-class axis | LRP config fixed in v1; vary map + query only; tuning explodes the sweep space |
| Open-source labeller integration (CVAT/Label Studio) | README "phase 2"; depends on a trusted exploration loop existing first |
| Polished mask/dataset export (binary masks, COCO/PNG) | Downstream milestone; label spec not yet known; keep raw `.npy` for now |
| Fine-tuning / any model training | The abandoned prior approach this restart deliberately replaces |
| Other dataset categories (Toons, Fantasy, MapMaker, Satellite) | Rumsey historical only for v1; others add loader variance with no loop payoff |
| Web app / dashboard / served UI | Contradicts remote-VM + JupyterLab-over-SSH constraint; zero research payoff |
| Multi-GPU / distributed sweep | Premature optimization; single GPU VM is the stated runtime |

## Traceability

Which phases cover which requirements. Populated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| ENV-01 | Phase 1 | Pending |
| DATA-01 | Phase 1 | Pending |
| DATA-02 | Phase 1 | Pending |
| DATA-03 | Phase 1 | Pending |
| DATA-04 | Phase 1 | Pending |
| MODEL-01 | Phase 1 | Pending |
| ATTR-01 | Phase 1 | Code delivered; NOT met for SigLIP-2 (01-03 finding — dynamicLRP op-coverage gap, no relevance) |
| ATTR-02 | Phase 1 | Code delivered + unit-tested; not end-to-end verified for SigLIP-2 (01-03 finding) |
| ATTR-03 | Phase 1 | Controls authored; NOT met for SigLIP-2 (01-03 finding — no relevance to eyeball; D-02/D-03 user-waived) |
| VIZ-01 | Phase 1 | Code delivered; NOT met for SigLIP-2 (01-03 finding — no heatmap; smoke test instead) |
| VIZ-02 | Phase 2 | Pending |
| SWEEP-01 | Phase 2 | Pending |
| SWEEP-02 | Phase 2 | Pending |

**Coverage:**
- v1 requirements: 13 total
- Mapped to phases: 13 (Phase 1: 10, Phase 2: 3)
- Unmapped: 0 ✓

---
*Requirements defined: 2026-05-18*
*Last updated: 2026-05-18 after roadmap creation*
