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
- [ ] **ATTR-01**: Single forward + dynamic-LRP attribution for one (map, query) pair, attributing the image↔detached-text similarity scalar (`logits_per_image`), reducing image-token relevance to a per-patch grid
- [ ] **ATTR-02**: Reconstruct patch relevance into a 2D heatmap with a fixed normalization, correctly handling SigLIP-2's 27×27 patch grid and the 384÷14 non-integer edge discard

### Visualization

- [ ] **VIZ-01**: Heatmap overlaid on the source map (alpha-blended), displayed inline in a JupyterLab cell for a single (map, query) slice — **Phase 1 finish line**
- [ ] **VIZ-02**: Contact-sheet browse of all sweep overlays inside the notebook (grid of maps × queries with labels) — **v1 finish line**

### Sweep

- [ ] **SWEEP-01**: Configurable sweep over maps counting down from a high manifest index N (configurable start + count) crossed with a configurable query list, as pure orchestration over the ATTR/VIZ primitive
- [ ] **SWEEP-02**: Run caching of relevance grids keyed by (map id, query) to disk/GCS so the notebook re-renders contact sheets without recompute

## v2 Requirements

Acknowledged but deferred. Not in the current roadmap. Add only after the v1 loop is visually trusted.

### Diagnostics

- **DIAG-01**: Attribution sanity controls (query-swap, model-randomization, occlusion) as a diagnostic readout — *originally proposed as ATTR-03; descoped from v1 to keep evaluation strictly visual*
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
| ENV-01 | TBD | Pending |
| DATA-01 | TBD | Pending |
| DATA-02 | TBD | Pending |
| DATA-03 | TBD | Pending |
| DATA-04 | TBD | Pending |
| MODEL-01 | TBD | Pending |
| ATTR-01 | TBD | Pending |
| ATTR-02 | TBD | Pending |
| VIZ-01 | TBD | Pending |
| VIZ-02 | TBD | Pending |
| SWEEP-01 | TBD | Pending |
| SWEEP-02 | TBD | Pending |

**Coverage:**
- v1 requirements: 12 total
- Mapped to phases: 0 (pending roadmap)
- Unmapped: 12 ⚠️

---
*Requirements defined: 2026-05-18*
*Last updated: 2026-05-18 after initial definition*
