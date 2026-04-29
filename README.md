# MapClass

A system architecture for stylized map understanding: dense per-pixel land cover and
topography prediction across satellite, road, historical, and artistic maps. Target use
case is grand strategy game cartography (EU4 / CK3 / HoI4 style) at 100–2000 km scale,
with the framework generalising to any stylized cartographic source.

The academic framing — motivation, contributions, target venue — lives in
[mockup.md](./mockup.md). This README is the implementation companion: phases,
components, repo layout.

## What MapClass produces

A **dense pixel-level prediction**, two attributes per pixel:

- **Land cover** (9 classes): water, trees, shrubland, grassland, cropland, built-up,
  bare/sparse, flooded/wetland, snow/ice. Canonical taxonomy mapped from ESA WorldCover;
  Mangroves fold into Trees, Moss/lichen folds into Bare/sparse.
- **Topography** (3 classes): flat (<2°), hilly (2–15°), mountainous (>15°). Derived
  from Copernicus DEM GLO-30 via slope classification.

Pixel predictions are designed to support downstream region delineation (polygon
tracing); that step lives in a separate project.

## GeoViLM architecture

GeoViLM is a multi-component system, not a new model class. Four components:

1. **Vision-language backbone** — PaliGemma-3B is the front-runner candidate (SigLIP
   vision encoder + Gemma language decoder). Benchmarked zero-shot first, then
   fine-tuned in-system.
2. **Rotationally-invariant OCR module** — separate component for reading curved /
   rotated map text (place names, region labels). Adapted from scene-text recognition
   work (CRAFT, ABCNet are candidate starting points).
3. **Dense segmentation heads** — two lightweight heads on the backbone's vision
   features, one per output attribute (land cover + topography).
4. **Auto-georeferencing pipeline** — aligns stylized maps to real-world coordinates
   via cross-correlation against a WorldCover + DEM reference grid, followed by
   thin-plate-spline warping. Bootstrapped iteratively: v0 GeoViLM trained on
   already-registered maps → use v0 to georeference unregistered maps → v1 retraining
   on the expanded dataset. Intended as a reusable component for other stylized-map
   work, not just an internal label-prep step.

Backbone alternatives for the segmentation head (DINOv2, Swin, large-kernel CNNs) and
the rationale for hierarchical vs. coarse-to-fine inference are discussed in
[notes/architectural_references.md](./notes/architectural_references.md).

## Phase plan

### 1. Dataset construction

Four open-source sources covering visual diversity:

- **Synthetic illustrated maps** — Azgaar's Fantasy Map Generator + custom Pillow
  renderer (`scripts/build_dataset.py`, `render.py`, `label.py`, `biome_mapping.py`).
  Multiple rendering styles (flat / illustrated / satellite). Pixel-perfect ground
  truth since the renderer controls placement. Cropland, built-up, and flooded/wetland
  are absent from synthetic maps and appear only in satellite-derived data
  (up-weighted in training).
- **Historical illustrated maps** — David Rumsey collection, registered to modern
  coordinates and overlaid with WorldCover + DEM labels
  (`scripts/build_historical_dataset.py`, `scripts/historical/`). v0 scoped to maps
  already registered in the Georeferencer service; v1+ expands via the bootstrapped
  auto-georeferencing component. Coverage restricted to regional-scale maps
  (100–2000 km extent); city plans excluded.
- **Satellite imagery** — ESA WorldCover (`s3://esa-worldcover`, CC-BY 4.0) + Copernicus
  DEM GLO-30 (`s3://copernicus-dem-30m`) at regional scale. Provides ground-truth labels
  and geophysical co-occurrence priors.
- **Road maps** — TBD source (OSM tile renders are the obvious candidate). Not yet
  implemented.

Per-source class-conditional loss weights reflect temporal reliability: topography is
fully trusted (geology is stable); water/coastlines mostly trusted at regional scale;
trees, built-up, and cropland heavily downweighted on historical maps because land use
has shifted substantially since the 16th–17th century. Bare/sparse and snow/ice are
treated as reliable.

### 2. Failure analysis benchmark

Zero-shot evaluation of CLIP, SigLIP, OpenCLIP, and PaliGemma on the held-out stylized
test set, with per-class accuracy and confusion matrices. Combined with **dynamic-LRP**
analysis ([arXiv:2512.07010](https://arxiv.org/pdf/2512.07010)) to localise where in
each model the failures originate. Replaces hand-wavy claims with measured evidence and
provides the entry point for the mechanistic-failure contribution. Not yet implemented;
tracked in [executive_TODO.md](./executive_TODO.md).

### 3. GeoViLM construction

- Adapt the OCR module for curved / rotated text (separate sub-pipeline, may need its
  own training data).
- Add the dense segmentation heads on the chosen backbone.
- Train end-to-end with per-source loss weighting.
- Bootstrap auto-georeferencing: v0 → register more historical maps → v1 retraining.
  The v0-vs-v1 dataset gap also serves as an ablation for the system-paper claim that
  the auto-georeferencing component contributes meaningfully.

### 4. Evaluation + write-up

- **Test sets**: held-out synthetic maps + held-out registered historical maps.
- **Metric**: joint per-pixel negative log-likelihood
  `−(log p_land_cover + log p_topography)`. Joint rather than independent because land
  cover and topography are not statistically orthogonal (water is always flat,
  cropland is rarely mountainous).
- **Component ablations** — backbone-only vs. +OCR vs. +segmentation heads vs. full
  system, supporting the system-paper claim that each component is jointly necessary.
- Hand-annotation of real grand strategy maps is deferred; Label Studio / CVAT are
  candidates if a real-domain test set is added later.

## Repo layout

```
data/
  raw/               Azgaar GeoJSON exports (synthetic input)
  renders/           Synthetic map outputs (image + label PNGs)
  labels/            Pixel-level label rasters
  historical/        Historical map sources and derived label outputs
  toons/             Tile assets — kept as validation set / future
                     pre-labelled low-level feature library
  TODO.md            Open data-side todos
notes/
  architectural_references.md   Backbone choice, hierarchical-vs-large-kernel refs
scripts/
  build_dataset.py             Synthetic map dataset orchestrator
  render.py                    Synthetic map renderer (multi-style)
  label.py                     Synthetic map label generator
  biome_mapping.py             Azgaar biome → canonical taxonomy
  augment.py                   Image augmentation utilities (parchment, sepia, …)
  toon_mapping.py              Biome → tile filename index
  build_historical_dataset.py  Historical map dataset orchestrator
  historical/
    rumsey.py                  David Rumsey LUNA API client
    worldcover.py              ESA WorldCover S3 fetcher
    dem.py                     Copernicus DEM S3 fetcher
    label.py                   Historical map label generation
models/                        Trained model checkpoints (currently empty)
mockup.md                      Academic framing
README.md                      This file
executive_TODO.md              Owner-actionable todos (licensing, benchmarks)
LICENSE
requirements.txt
```

## Status

**In progress.** Repo is being reworked from an earlier hex-tile fine-tuning plan to
the system-paper plan in mockup.md. Working: synthetic dataset pipeline, historical
map sourcing and labelling. Not yet implemented: zero-shot benchmark harness, OCR
module, segmentation heads, auto-georeferencing pipeline, road-map source. Open
licensing and benchmark items live in [executive_TODO.md](./executive_TODO.md).

## Cloud / training workflow

Cloud training is handled by a separate VM-provisioning repo with an SSH workflow.
This repo does not carry docker or cloud build infrastructure.
