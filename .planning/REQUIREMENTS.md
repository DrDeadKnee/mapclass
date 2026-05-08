# Requirements

**Project:** MapClass / GeoViLM
**Milestone:** v1
**Source of truth:** Extracted from `.planning/PROJECT.md` Active list. PROJECT.md remains canonical — if this file drifts, PROJECT.md wins.

## v1 Requirements

### Data — DATA

- [ ] **DATA-05**: OSM road-tile dataset sub-pipeline as the third v1 source
- [ ] **DATA-06**: Per-source class-conditional loss weights (topography fully trusted; water mostly trusted; trees/built-up/cropland heavily downweighted on historical maps; bare/sparse and snow/ice trusted)

### Auto-georeferencing — GEOREF

- [ ] **GEOREF-01**: Auto-georeferencing tool (cross-correlation against WorldCover + DEM reference grid → thin-plate-spline warping) — used as a training-data-prep tool, not an inference component
- [ ] **GEOREF-02**: Bootstrap loop: train v0 on already-registered historical maps → use v0 to register currently-unregistered Rumsey maps → retrain v1 on the expanded dataset

### Model components — MODEL

- [ ] **MODEL-01**: Small VL backbone for the v1 ship target — the inference module's backbone (specific candidate selected in research phase; PaliGemma-3B is excluded as the *ship* backbone by inference budget)
- [ ] **MODEL-02**: Rotationally-invariant OCR module for curved / rotated map text (CRAFT or ABCNet as candidate starting points; specific choice in research phase)
- [ ] **MODEL-03**: Two lightweight dense segmentation heads on the backbone's vision features (one per output attribute: land cover, topography)
- [ ] **MODEL-04**: PaliGemma-3B benchmark variant — same GeoViLM stack (OCR + seg heads + auto-georef bootstrap) with PaliGemma-3B swapped in as the backbone in place of MODEL-01. Trained as a bellwether upper bound to validate the small-backbone choice; not shipped as inference

### Training — TRAIN

- [ ] **TRAIN-01**: End-to-end training of GeoViLM v0 with per-source loss weighting on the synthetic + already-registered-historical + OSM dataset, run for both backbones (MODEL-01 small VL + MODEL-04 PaliGemma)
- [ ] **TRAIN-02**: End-to-end retraining of GeoViLM v1 on the bootstrap-expanded dataset, run for both backbones

### Evaluation — EVAL

- [ ] **EVAL-01**: Held-out joint per-pixel NLL `−(log p_land_cover + log p_topography)` on synthetic + historical test splits as the v1 ship metric (computed for the small-backbone variant; PaliGemma variant evaluated under EVAL-03)
- [ ] **EVAL-02**: Qualitative spot-check renders on Tolkien Middle-Earth, Westeros, Abercrombie First-Law Circle of the World, and Warhammer Old World as ship demos (no ground truth, no metric); rendered for both backbones
- [ ] **EVAL-03**: PaliGemma-vs-small-backbone NLL comparison on the same held-out splits — bellwether for the small-backbone choice. If the gap is too large to accept, the small-backbone candidate (MODEL-01) is reconsidered before ship

### Ship — SHIP

- [ ] **SHIP-01**: Inference Python module + trained checkpoint of the small-backbone variant, packaged for the separate hex-grid app to consume; runs on single CPU or 4–8 GB consumer GPU. PaliGemma checkpoint is retained as a research/benchmark artifact, not packaged for the app

## Already Validated (existing on `refactor_paper` branch)

These are pre-existing in the codebase and treated as fixed for v1 — not phase deliverables. Listed here for completeness and traceability.

- ✓ **DATA-01**: Synthetic dataset pipeline (Azgaar Fantasy Map Generator + custom Pillow renderer with multi-style outputs) — `scripts/build_dataset.py`, `scripts/render.py`, `scripts/label.py`, `scripts/biome_mapping.py`, `scripts/augment.py`
- ✓ **DATA-02**: Historical dataset pipeline (David Rumsey LUNA API + ESA WorldCover S3 + Copernicus DEM S3) — `scripts/build_historical_dataset.py`, `scripts/historical/{rumsey,worldcover,dem,label}.py`
- ✓ **DATA-03**: Canonical 9-class land-cover taxonomy mapped from ESA WorldCover; Mangroves→Trees and Moss/Lichen→Bare-sparse fold
- ✓ **DATA-04**: 3-class topography taxonomy (flat <2°, hilly 2–15°, mountainous >15°) from Copernicus DEM GLO-30 slope

## Out of Scope (v1)

Anchored in PROJECT.md "Out of Scope (v1)". Listed verbatim here so the roadmapper does not re-promote these during phase decomposition.

- **PaliGemma-3B as the v1 ship / inference backbone** — 6 GB FP16 weights break the inference budget. PaliGemma *training* is in scope as a v1 benchmark/bellwether (MODEL-04, EVAL-03); only its use as the packaged inference backbone is excluded.
- **Auto-georef as an inference / app feature** — fantasy maps lack real-world coordinates; auto-georef is a training-data-prep tool only.
- **Hex-grid aggregation, polygon tracing, region delineation** — owned by the separate hex-grid app downstream.
- **EU4 / CK3 / HoI4 engine-specific output formats** — output is generic per-pixel class probabilities.
- **Hand-annotated fantasy test set** — qualitative spot-checks only at v1.
- **Zero-shot CLIP / SigLIP / OpenCLIP / PaliGemma baselines** — milestone 2.
- **Dynamic-LRP mechanistic failure analysis** — milestone 2.
- **Component ablations** (backbone-only vs +OCR vs +seg-heads vs full system) — milestone 2.
- **Paper draft / write-up** — milestone 2.
- **Redistribution of the combined training dataset** — license risk; v1 ships weights + code only.
- **Hand-annotation tooling integration** (Label Studio / CVAT) — only relevant if a real-domain annotated test set is added later.
- **Cloud / docker / VM provisioning** — handled by a separate VM-provisioning repo.
- **Tests, CI, linting / formatting tooling** — single-author research codebase, accepted risk for v1.

## Traceability

Filled in by the roadmapper after `ROADMAP.md` is generated.

| REQ-ID | Phase |
|--------|-------|
| DATA-05 | — |
| DATA-06 | — |
| GEOREF-01 | — |
| GEOREF-02 | — |
| MODEL-01 | — |
| MODEL-02 | — |
| MODEL-03 | — |
| MODEL-04 | — |
| TRAIN-01 | — |
| TRAIN-02 | — |
| EVAL-01 | — |
| EVAL-02 | — |
| EVAL-03 | — |
| SHIP-01 | — |

---
*Generated: 2026-05-08 from PROJECT.md Active list. Update this file only if PROJECT.md changes; keep in sync.*
