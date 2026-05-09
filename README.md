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

1. **Vision-language backbone** — a *small* VL backbone is the v1 ship target, chosen
   to fit the inference budget (single CPU host or 4–8 GB consumer GPU). The specific
   candidate is selected in the research phase. PaliGemma-3B is **not** the ship
   backbone — at 6 GB FP16 it breaks the inference budget — but the same GeoViLM stack
   is trained with PaliGemma-3B swapped in as a **benchmark / bellwether variant**.
   Comparing the two NLL numbers on the held-out splits validates that the small
   backbone choice is good enough; if the gap is unacceptable, the small candidate is
   reconsidered before ship. The PaliGemma checkpoint is retained as a research
   artifact, not packaged for the hex-grid app.
2. **Rotationally-invariant OCR module** — separate component for reading curved /
   rotated map text (place names, region labels). Adapted from scene-text recognition
   work (CRAFT, ABCNet are candidate starting points).
3. **Dense segmentation heads** — two lightweight heads on the backbone's vision
   features, one per output attribute (land cover + topography).
4. **Auto-georeferencing pipeline** — aligns stylized maps to real-world coordinates
   via cross-correlation against a WorldCover + DEM reference grid, followed by
   thin-plate-spline warping. **Training-data-prep tool only**, not an inference /
   app feature: fantasy maps lack real-world coordinates by construction. Bootstrapped
   iteratively: v0 GeoViLM trained on already-registered maps → use v0 to
   georeference unregistered Rumsey maps → v1 retraining on the expanded dataset.
   The v0→v1 dataset gap is also a clean ablation if the milestone-2 paper happens.

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
- **Road maps** — OSM tile renders, the third v1 dataset source (DATA-05). A small
  additive sub-pipeline alongside the working synthetic and historical pipelines.
  Not yet implemented.

Per-source class-conditional loss weights reflect temporal reliability: topography is
fully trusted (geology is stable); water/coastlines mostly trusted at regional scale;
trees, built-up, and cropland heavily downweighted on historical maps because land use
has shifted substantially since the 16th–17th century. Bare/sparse and snow/ice are
treated as reliable.

### 2. PaliGemma-3B benchmark variant

The same GeoViLM stack — OCR module + dense segmentation heads + auto-georef
bootstrap — is trained a second time with **PaliGemma-3B swapped in for the small
VL backbone**. This variant runs through the same v0 → register → v1 bootstrap as
the ship backbone. Same data, same heads, same loss weighting; only the backbone
differs.

The PaliGemma variant is the **bellwether**: it gives an upper-bound NLL on the same
held-out splits. The shipping small-backbone NLL is compared against it; an
acceptably small gap validates the small-backbone choice for v1 ship, and an
unacceptable gap forces the small-backbone candidate to be reconsidered before
ship. The PaliGemma checkpoint is retained as a research artifact and is **not**
packaged for the hex-grid app.

Zero-shot CLIP / SigLIP / OpenCLIP / PaliGemma comparisons and dynamic-LRP
mechanistic failure analysis are **deferred to milestone 2** as paper-flavoured
items. They are not in v1 scope; tracked in
[executive_TODO.md](./executive_TODO.md).

### 3. GeoViLM construction

- Adapt the OCR module for curved / rotated text (separate sub-pipeline, may need its
  own training data).
- Add the dense segmentation heads on the chosen backbone.
- Train end-to-end with per-source loss weighting.
- Bootstrap auto-georeferencing: v0 → register more historical maps → v1 retraining.
  The v0-vs-v1 dataset gap also serves as an ablation for the system-paper claim that
  the auto-georeferencing component contributes meaningfully.

### 4. Evaluation

- **Test sets**: held-out synthetic maps + held-out registered historical maps,
  split deterministically by sample-id hash (not glob order).
- **Ship metric**: joint per-pixel negative log-likelihood
  `−(log p_land_cover + log p_topography)`. Joint rather than independent because
  land cover and topography are not statistically orthogonal (water is always flat,
  cropland is rarely mountainous). Computed for both backbones; the small-vs-PaliGemma
  gap is the bellwether (see phase 2).
- **Qualitative spot checks**: rendered predictions on four target fantasy maps —
  Tolkien Middle-Earth, Westeros, Abercrombie First Law (Circle of the World), and
  the Warhammer Old World. No ground truth, no metric — these are ship demos.
- **Component ablations** (backbone-only vs. +OCR vs. +seg heads vs. full system) and
  hand-annotation of real grand strategy maps are **deferred to milestone 2**;
  Label Studio / CVAT are candidates only if a real-domain annotated test set is
  added later.

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

**In progress.** Repo is being reworked from an earlier hex-tile fine-tuning plan
to a leaner v1 scope: a small VL backbone + OCR + dense seg heads + auto-georef
bootstrap, with a PaliGemma-3B benchmark variant trained alongside for
bellwether comparison. Output is generic per-pixel class probabilities consumed
by a separate hex-grid app downstream of this project; engine-specific packaging
is out of scope here.

**Working:** synthetic dataset pipeline, historical map sourcing and labelling.

**Not yet implemented:** OSM road-tile sub-pipeline, OCR module, dense
segmentation heads, auto-georef bootstrap, training entrypoint, evaluation
harness, inference module, PaliGemma-3B benchmark variant.

**Deferred to milestone 2** (paper-flavoured): zero-shot CLIP / SigLIP /
OpenCLIP / PaliGemma comparisons, dynamic-LRP mechanistic failure analysis,
component ablations, hand-annotated fantasy test set, paper write-up.

The locked v1 scope, constraints, decisions, and out-of-scope list live in
[.planning/PROJECT.md](./.planning/PROJECT.md). Open licensing items live in
[executive_TODO.md](./executive_TODO.md).

## Quick start (Phase 1 walking skeleton)

This is the **Phase 1 Mock-backbone smoke setup**. The pipeline is the deliverable here,
not the score — the model is a tiny learnable conv stub used to prove the gradient path
flows end-to-end. SmolVLM lands in Phase 2; PaliGemma2 lands in Phase 5.

### Install

```sh
pip install -e .
python -c "import mapclass"   # smoke import
```

### Smoke run (Mock backbone gradient-path proof)

```sh
# 5-iteration loss-decrease check on the Mock backbone (D-01 — proves gradient flow).
python -m mapclass.train --config mapclass/configs/v0.yaml --smoke
# writes models/geovilm_phase1_mock.pt with embedded safetensors metadata
# (model_version, dataset_manifest_sha, taxonomy_hash, training_seed,
#  source_class_weights_hash, backbone, processor_identity).
```

### Inference

```sh
python -m mapclass.infer models/geovilm_phase1_mock.pt path/to/image.png
# prints land_cover (9, H, W) + topography (3, H, W) probability tensors;
# WATER_TOPO=255 sentinel overlaid where land_cover argmax == water.
# load_model() refuses to load if the checkpoint's taxonomy_hash does not
# match the live mapclass.data.taxonomy.taxonomy_hash() (PITFALL 4 #3).
```

### Eval

```sh
python -m mapclass.eval --checkpoint models/geovilm_phase1_mock.pt \
                        --config mapclass/configs/v0.yaml \
                        --split heldout --report eval_report.json
# writes eval_report.json with per-source × per-class NLL, per-source mean,
# overall mean, and a calibration: null field reserved for Phase 2+.
# Refuses to run if the test split overlaps the train split
# (SplitsContaminationError; PITFALL 5 prevention #3).
```

### Caveats

Phase 1 numbers will be near `log(num_classes)` because the backbone is Mock
(D-01 in `.planning/phases/01-end-to-end-skeleton/01-CONTEXT.md`). The pipeline
is the deliverable, not the score; SmolVLM lands in Phase 2 and is the first
backbone whose NLL numbers are meaningful. The Mock backbone is intentionally
a learnable conv stub (~19k params) — you should expect modest improvement
over the class-frequency prior, nothing more.

## Cloud / training workflow

Cloud training is handled by a separate VM-provisioning repo with an SSH workflow.
This repo does not carry docker or cloud build infrastructure.
