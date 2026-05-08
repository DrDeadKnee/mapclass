# Architecture Research

**Domain:** Multi-source ML training + dense-prediction inference for stylized cartography
**Researched:** 2026-05-08
**Confidence:** HIGH for component boundaries and data flow (anchored in existing code); MEDIUM for the small-VL backbone interface (the specific backbone choice is the STACK research question, not this one — this doc only specifies the *swap surface*).

## Scope of this document

This is the **integration architecture** for adding three new components to a brownfield codebase that already ships two working data pipelines (synthetic, historical). It does **not** redesign the existing pipelines — see `.planning/codebase/ARCHITECTURE.md` for those, and treat their on-disk contract (`image*.png + land_cover.png + topography.png [+ sample_weights.json]`) as load-bearing.

The three new components:

1. **OSM road-tile sub-pipeline** — third dataset source.
2. **Auto-georef tool** with v0→register→v1 bootstrap loop.
3. **GeoViLM model** — small VL backbone + rotationally-invariant OCR + two dense seg heads, plus the training and inference entry points.

Plus the **`mapclass/` Python package** that the existing `scripts/` directory has been waiting for (per `STRUCTURE.md` and `CONVENTIONS.md` notes about lifting shared code).

## Standard Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  RAW INPUTS (external, anonymous-public-S3 or developer-supplied)           │
├─────────────────────────────────────────────────────────────────────────────┤
│  Azgaar      David Rumsey    OSM (PBF or       ESA WorldCover   Copernicus  │
│  GeoJSON     LUNA + WMS      Overpass API or   (S3, anon)       DEM (S3)    │
│  (local)     (HTTPS)         tile-renderer)                                 │
└──────┬────────────┬─────────────────┬────────────────┬───────────┬──────────┘
       │            │                 │                │           │
       ▼            ▼                 ▼                ▼           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PER-SOURCE DATASET CONSTRUCTORS  (filesystem-to-filesystem batch ETL)      │
├──────────────────────┬──────────────────────┬───────────────────────────────┤
│ scripts/             │ scripts/             │ scripts/                      │
│ build_dataset.py     │ build_historical_    │ build_road_dataset.py  (NEW)  │
│ (synthetic, exists)  │ dataset.py (exists)  │                               │
│   ↓                  │   ↓                  │   ↓                           │
│ mapclass.synthetic.* │ mapclass.historical.*│ mapclass.road.*  (NEW)        │
│                      │   + georef bootstrap │                               │
│                      │   when v0 exists     │                               │
└──────────────────────┴──────────┬───────────┴───────────────────────────────┘
                                  │  (every source emits the SAME sample contract)
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  SAMPLE STORE  (filesystem; gitignored data/)                               │
│   data/renders/<name>/{flat,illustrated,satellite}.png + LAND/TOPO/MANIFEST │
│   data/historical/dataset/<id>/image.png + LAND/TOPO + sample_weights.json  │
│   data/road/dataset/<id>/image.png + LAND/TOPO + sample_weights.json (NEW)  │
└────────────────────────────────────────┬────────────────────────────────────┘
                                         │  (Source interface enumerates samples)
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  GeoViLM MODEL  (mapclass/, NEW — currently empty `models/` dir gets used)  │
├─────────────────────────────────────────────────────────────────────────────┤
│   Backbone interface  ─────►  small VL (SigLIP-2-B / DINOv2-S — STACK doc)  │
│        │                                                                    │
│        ├──► OCR head  ──────► curved-text reader (CRAFT/ABCNet adapter)     │
│        │                                                                    │
│        ├──► Land-cover head ► dense logits (9 classes, sample-resolution)   │
│        │                                                                    │
│        └──► Topography head ─► dense logits (3 classes + WATER_TOPO=255)    │
└────────────────────────────────────────┬────────────────────────────────────┘
                                         │
            ┌────────────────────────────┼────────────────────────────┐
            ▼                            ▼                            ▼
┌────────────────────┐      ┌────────────────────────┐    ┌──────────────────┐
│ Training entry     │      │ Inference module       │    │ Eval entry       │
│ python -m          │      │ mapclass.infer         │    │ python -m        │
│ mapclass.train     │      │  (consumed by external │    │ mapclass.eval    │
│  --config v0|v1    │      │  hex-grid app)         │    │  --split heldout │
│                    │      │                        │    │                  │
│  reads YAML        │      │  CPU-only or 4–8 GB    │    │  joint per-pixel │
│  reads samples     │      │  GPU; per-pixel        │    │  NLL on synth +  │
│  reads loss-       │      │  probability tensors   │    │  historical      │
│  weights schema    │      │  out                   │    │  test split      │
└────────┬───────────┘      └────────────────────────┘    └──────────────────┘
         │
         │  (when v0 checkpoint exists)
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  AUTO-GEOREF TOOL  (scripts/georef/, NEW)                                   │
│                                                                             │
│   Online bootstrap mode:                                                    │
│     v0 inference on unregistered Rumsey TIF → predicted land_cover/topo →   │
│     cross-correlate against WorldCover + DEM at lat/lng grid →              │
│     select best-fit GCPs → thin-plate-spline warp →                         │
│     emit a "georeferenced" GeoTIFF compatible with the existing             │
│     scripts/historical/label.py:make_labels()                               │
│                                                                             │
│   Offline sanity-check mode (independent of v0):                            │
│     run on already-registered Rumsey TIFs, hide their CRS, see if the       │
│     tool recovers the known coords. Useful immediately, no model needed.    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | New / Existing | Responsibility | Implementation |
|-----------|----------------|----------------|----------------|
| `mapclass.synthetic.*` | Existing (renamed-in-place from `scripts/`) | Render Azgaar GeoJSON to multi-style PNGs + label rasters | `Pillow`, pure-Python; no change to logic |
| `mapclass.historical.*` | Existing (moved into package) | Rumsey LUNA + WorldCover S3 + DEM S3 + per-GeoTIFF labels | `rasterio`, `pyproj`, `requests`; no logic change |
| `mapclass.road.*` | **NEW** | OSM tile/PBF → rasterised road map + WorldCover/DEM-derived labels for the same bbox | `osmnx` or `pyrosm` for PBF; `mapnik`/`pillow` for tile render; reuse existing `worldcover.py`/`dem.py` for labels |
| `scripts/build_road_dataset.py` | **NEW** | Top-level orchestrator, mirrors the historical entry point exactly | `argparse` subcommands `search/build/full` |
| `mapclass.georef` | **NEW** | Auto-georef tool: cross-correlation against WorldCover + DEM, TPS warp, GCP scoring | `scikit-image` (phase corr), `rasterio.warp`, in-house TPS or `gdal_translate` shell-out |
| `scripts/georef.py` | **NEW** | CLI: `bootstrap` (v0-driven), `sanity-check` (use known coords), `register` (single TIF) | `argparse`, no extra deps |
| `mapclass.data` | **NEW** | `Source` enumerator interface; `MapImage`/`LabelPair` data types; loss-weights loader; PyTorch `Dataset` adapter | `torch.utils.data`, `numpy`, `Pillow` |
| `mapclass.model` | **NEW** | `Backbone` interface, OCR head, two dense seg heads, end-to-end module | `torch`, `transformers`, `peft` |
| `mapclass.train` | **NEW** | Training entry point; reads config YAML; mixes sources by weight | `torch`, `accelerate`, `bitsandbytes` |
| `mapclass.infer` | **NEW** | Inference module: load checkpoint → image → per-pixel probabilities (the hex-grid app's import surface) | `torch`, `numpy`, `Pillow` |
| `mapclass.eval` | **NEW** | Held-out joint NLL on synth + historical; spot-check renders for the four target fantasy maps | `torch`, `numpy`, `Pillow` |

## Recommended Project Structure

The recommended layout is **additive**: existing files keep their relative locations (so historical pipeline imports keep working today), but the codebase grows a top-level `mapclass/` Python package for the new modelling code, and the `scripts/` directory stays as the user-facing entry-point layer.

```
mapclass/                       # NEW — top-level Python package
├── __init__.py                 # exports: Source, MapImage, LabelPair, LossWeights, INFER (the API)
├── data/
│   ├── __init__.py
│   ├── sources.py              # Source ABC + 4 concrete implementations
│   │                           #   SyntheticSource, HistoricalSource,
│   │                           #   HistoricalBootstrappedSource, RoadSource
│   ├── types.py                # MapImage, LabelPair, SampleManifest dataclasses
│   ├── loss_weights.py         # YAML schema + loader for per-source × per-class weights
│   ├── dataset.py              # torch.utils.data.Dataset wrapper over Source
│   └── transforms.py           # train-time augmentation, calls into existing scripts/augment.py
├── model/
│   ├── __init__.py
│   ├── backbone.py             # Backbone ABC + concrete impls (chosen in STACK research)
│   ├── ocr_head.py             # rotation-invariant OCR (CRAFT/ABCNet starting point)
│   ├── seg_heads.py            # land-cover head, topography head (two thin upsampling stacks)
│   ├── geovilm.py              # the full module: backbone → {ocr, lc, topo} multi-task
│   └── losses.py               # per-source × per-class weighted NLL
├── train.py                    # __main__: python -m mapclass.train --config configs/v0.yaml
├── infer.py                    # public API for the downstream hex-grid app
│                               #   load_model(ckpt) -> Predictor
│                               #   Predictor.predict(image) -> (lc_logits, topo_logits)
├── eval.py                     # __main__: python -m mapclass.eval --split heldout
└── configs/
    ├── v0.yaml                 # synthetic + already-registered-historical + OSM
    ├── v1.yaml                 # v0 sources + bootstrap-expanded historical
    └── loss_weights.yaml       # the per-source × per-class schema (canonical home)

scripts/                        # EXISTS — thin entry-point wrappers, no logic
├── build_dataset.py            # EXISTS — synthetic
├── build_historical_dataset.py # EXISTS — historical
├── build_road_dataset.py       # NEW — OSM (mirrors build_historical_dataset.py exactly)
├── georef.py                   # NEW — `bootstrap | sanity-check | register` CLI
├── render.py                   # EXISTS — synthetic renderer (gradually grows package shims)
├── label.py                    # EXISTS — synthetic labels
├── biome_mapping.py            # EXISTS — canonical taxonomy (the linchpin file)
├── augment.py                  # EXISTS
├── toon_mapping.py             # EXISTS
├── historical/                 # EXISTS — sub-package for historical sub-pipeline
│   ├── __init__.py
│   ├── rumsey.py
│   ├── worldcover.py
│   ├── dem.py
│   └── label.py
├── road/                       # NEW — sub-package for OSM sub-pipeline
│   ├── __init__.py
│   ├── osm.py                  # PBF or Overpass query + tile render
│   ├── label.py                # per-tile labels (reuses scripts/historical/{worldcover,dem}.py)
│   └── (no rumsey-equivalent — OSM has no search step)
└── georef/                     # NEW — sub-package for the auto-georef tool
    ├── __init__.py
    ├── crosscorr.py            # phase-correlation against (WorldCover-derived) reference
    ├── tps.py                  # thin-plate-spline warp
    ├── gcp.py                  # GCP scoring + selection
    └── bootstrap.py            # the v0-uses-model-to-register-rumsey-maps loop

models/                         # EXISTS (empty) — checkpoints land here
├── geovilm_v0.pt               # output of train.py --config v0
├── geovilm_v1.pt               # output of train.py --config v1
└── (gitignored once non-trivial)

data/                           # EXISTS (gitignored) — sample store, see below
└── road/                       # NEW
    └── dataset/<id>/{image,land_cover,topography}.png + sample_weights.json + manifest.json
```

### Structure Rationale

- **`mapclass/` package added without touching existing code.** The historical pipeline already does the right thing as a package (`scripts/historical/__init__.py` exists). The synthetic-side bare-name imports (`from biome_mapping import …`) are flagged in `.planning/codebase/CONVENTIONS.md` as fragile but they work; the new model code does **not** depend on them — it imports `mapclass.data.sources`, which in turn calls into the existing scripts via shim functions. The shim functions (`mapclass/data/sources.py:_load_synthetic_sample()` and friends) are the *only* place that has to know about the `scripts/` layout. This keeps the brownfield risk localised.
- **`scripts/` remains the user-facing entry-point layer.** The cloud-VM SSH workflow invokes `python scripts/build_*.py` and `python -m mapclass.train`. Mixing both styles is acceptable because `scripts/` is for *humans running things* and `mapclass/` is for *code importing things*. CONVENTIONS.md item 5 codifies this naming pattern.
- **`scripts/road/` mirrors `scripts/historical/` deliberately.** Same `__init__.py`, same `worldcover.py + dem.py + label.py` pattern (the road sub-pipeline reuses `historical/worldcover.py` and `historical/dem.py` as-is — they don't care whether their reference image came from a Rumsey scan or an OSM tile render, only the bbox). New file: `scripts/road/osm.py` (analogue of `historical/rumsey.py`) for the OSM-specific source code path.
- **`scripts/georef/` is a sub-package, not a flat module.** The auto-georef tool has three independent pieces (cross-correlation, TPS, GCP scoring) plus the bootstrap loop. Splitting them into separate files makes the offline-sanity-check mode (which uses crosscorr + tps + gcp without bootstrap) testable and useful before v0 exists.
- **`models/` is unchanged from the README** — checkpoints land there; trainer writes `models/geovilm_v0.pt` etc.
- **`mapclass/configs/` holds training configs.** YAML, not Python. Two configs (`v0.yaml`, `v1.yaml`) plus the loss-weights schema. Keeping configs in-package (rather than at repo root) lets `python -m mapclass.train --config v0.yaml` resolve relative to the package on the cloud VM.

## Architectural Patterns

### Pattern 1: Per-source pipelines write to a single sample contract

**What:** Every dataset constructor — synthetic, historical, road, and bootstrap-expanded historical — emits a directory containing `image.png` (or named-style PNGs for synthetic), `land_cover.png`, `topography.png`, and (new for all sources) `sample_weights.json` + `manifest.json`. Class indices are the canonical taxonomy from `scripts/biome_mapping.py:LANDCOVER_CLASSES` and `TOPO_CLASSES`. Sentinel `NODATA = 255` is shared.

**When to use:** Always. This is the single most important invariant of the system.

**Trade-offs:**
- Pro: every new data source costs O(one folder convention) to integrate; the trainer never special-cases a source at the file-IO level.
- Pro: the historical and road pipelines can share `worldcover.py`/`dem.py` because the contract is bbox-in, label-png-out.
- Con: implicit. The contract is enforced by convention, not by a schema check. **Risk flag:** the `LossWeights` and `manifest.json` schemas are new (the existing pipeline has only `sample_weights.json`); both are flagged as untested in `.planning/codebase/TESTING.md` gap-analysis item 6. Add a `mapclass.data.contract.assert_sample_valid(path)` helper invoked at every load to fail loudly when the contract drifts.

**Example contract files:**

```
data/historical/dataset/luna_4567/
├── image.png            # uint8 RGB
├── land_cover.png       # uint8 single channel, values 0-8 + NODATA(255)
├── topography.png       # uint8 single channel, values 0-2 + WATER_TOPO(255)
├── sample_weights.json  # existing — per-class weight dict
└── manifest.json        # NEW — see below
```

**`manifest.json` schema (NEW):**
```json
{
  "source": "historical",
  "source_subtype": "rumsey_registered" | "rumsey_bootstrapped" | "synthetic_azgaar" | "road_osm",
  "source_id": "luna_4567",
  "registered_via": "georeferencer_wms" | "auto_georef_v0" | "manual_qgis" | "synthetic_pixel_space" | "osm_tile",
  "image_files": ["image.png"],          // synthetic emits multiple styles
  "label_files": ["land_cover.png", "topography.png"],
  "weights_file": "sample_weights.json",
  "bbox_wgs84": [west, south, east, north] | null,   // null for synthetic
  "created": "2026-05-08T..."
}
```

### Pattern 2: Loss weights as a (source, class) table loaded once

**What:** Per-source class-conditional loss weights live in `mapclass/configs/loss_weights.yaml` as a 2-D table indexed by `(source_subtype, class_name)`. The trainer loads it once at startup. The per-sample `sample_weights.json` (already emitted by the historical pipeline) is preserved as a per-sample override channel, but the *bulk* weight comes from the source-level YAML. Topography weights are scalar per source (one column).

**When to use:** Always for the trainer. The existing `HISTORICAL_LC_WEIGHTS` constant in `scripts/historical/label.py` becomes the *initial values* of the historical row in the YAML, but the YAML is the single source of truth going forward.

**Trade-offs:**
- Pro: one place to edit weights; one schema; one validator.
- Pro: makes the v0/v1 distinction explicit — both configs can share `loss_weights.yaml` or v1 can override.
- Con: introduces a real config file in a codebase that has historically used module-level constants only (per CONVENTIONS.md). Justified because `.planning/codebase/TESTING.md` gap-item 6 already flags this dict as schema-fragile and silently wrong-on-rename.

**Example:**

```yaml
# mapclass/configs/loss_weights.yaml
sources:
  synthetic_azgaar:
    land_cover:
      water: 1.0
      trees: 1.0
      shrubland: 1.0
      grassland: 1.0
      cropland: 0.0       # absent from synthetic
      built_up: 0.0       # absent from synthetic
      bare_sparse: 1.0
      flooded_wetland: 0.0  # absent from synthetic
      snow_ice: 1.0
    topography: 1.0

  rumsey_registered:
    land_cover:
      water: 0.9
      trees: 0.2          # 16th-c forest cover ≠ modern
      shrubland: 0.5
      grassland: 0.5
      cropland: 0.15      # heavily moved since 1500-1700
      built_up: 0.1       # urban footprint very different
      bare_sparse: 0.8
      flooded_wetland: 0.5
      snow_ice: 0.9
    topography: 1.0       # geology stable

  rumsey_bootstrapped:
    # same as rumsey_registered but with an extra confidence factor
    inherit: rumsey_registered
    confidence: 0.5       # multiplied through

  road_osm:
    land_cover:
      water: 1.0
      trees: 0.7          # OSM landuse=forest is opportunistic
      shrubland: 0.5
      grassland: 0.5
      cropland: 0.7
      built_up: 1.0       # OSM is strong here
      bare_sparse: 0.5
      flooded_wetland: 0.7
      snow_ice: 0.7
    topography: 1.0       # WorldCover/DEM derived, fully trusted
```

### Pattern 3: Backbone interface as a swap surface

**What:** `mapclass.model.backbone.Backbone` is an ABC with a single method, `extract_features(image: Tensor) -> dict[str, Tensor]`, returning a dictionary of feature maps at one or more scales. Concrete subclasses wrap a specific small VL choice (SigLIP-2-B / DINOv2-S / etc., decided in the STACK research phase). The seg heads and OCR head consume the dict by stage name — they don't import the backbone class directly.

**When to use:** Always. The stack-research phase explicitly defers the backbone choice; this codebase needs the swap surface defined *before* the backbone is picked so training-loop code is not on the critical path of that decision.

**Trade-offs:**
- Pro: decouples the stack-research timeline from the training-pipeline implementation. The seg-heads team can develop against a `MockBackbone` that returns random feature maps of the right shape until the real choice is made.
- Pro: ablation friendly (paper milestone-2 wants backbone-only vs. +OCR vs. +seg-heads ablations; this is trivial when components are interchangeable).
- Con: an over-abstracted ABC here would be a YAGNI penalty. Keep it strictly to one method (`extract_features`) and one optional method (`text_encode` for OCR-prompt-conditioning if/when used). Resist adding hooks until a second backbone exists.

**Example:**

```python
# mapclass/model/backbone.py
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch

class Backbone(ABC):
    """Swap surface for the small VL backbone."""

    image_size: int                # input resolution (e.g. 384 for SigLIP-2-B)
    feature_channels: dict[str, int]   # name -> channel count, e.g. {"stage4": 768}

    @abstractmethod
    def extract_features(self, image: "torch.Tensor") -> dict[str, "torch.Tensor"]:
        """Returns a dict of feature maps. Caller is the seg-head / OCR-head."""
        ...

# mapclass/model/seg_heads.py — consumes the dict
class LandCoverHead(torch.nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 9): ...
    def forward(self, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        # picks `feats["stage4"]` or whatever key it was configured for
        ...
```

### Pattern 4: Bootstrap loop is filesystem-mediated, not in-process

**What:** The auto-georef bootstrap step is a *separate batch run* between v0 training and v1 training, not an in-process callback. v0 trains and writes `models/geovilm_v0.pt`. Then `python -m scripts.georef bootstrap --model models/geovilm_v0.pt --manifest data/historical/raw/unregistered_manifest.json` reads the manifest, runs v0 inference + cross-correlation + TPS, writes new GeoTIFFs into `data/historical/raw/georeferenced/` (the existing input dir for the historical labeller), and updates `manifest.json` per sample with `registered_via: auto_georef_v0`. Then `python scripts/build_historical_dataset.py build` (the existing entry point) processes the newly-registered TIFs into the standard sample contract. Then `python -m mapclass.train --config v1.yaml` retrains.

**When to use:** Always. There is no need for the bootstrap to run in-process; everything is filesystem-mediated already, and an in-process loop would conflate training and dataset-construction concerns.

**Trade-offs:**
- Pro: each step is independently invocable, debuggable, and resumable. If TPS warping fails on 10 maps, the user can fix and re-run only `bootstrap`.
- Pro: matches the existing pipeline philosophy (filesystem state, no DB, no queue).
- Pro: the v1 training run is *byte-identical* to a fresh v0-style run on a larger dataset, so no special v1 code path exists.
- Con: requires the user to manually sequence three commands. Mitigated by adding a `Makefile` target or a thin `python -m mapclass.bootstrap_full` orchestrator if this becomes annoying.

**Example sequence:**

```bash
# v0 training (after datasets exist)
python -m mapclass.train --config mapclass/configs/v0.yaml

# Bootstrap: register the unregistered Rumsey maps using v0
python scripts/georef.py bootstrap \
    --model models/geovilm_v0.pt \
    --manifest data/historical/raw/unregistered_manifest.json \
    --out-dir data/historical/raw/georeferenced/

# Re-build the historical dataset (now includes the newly-registered ones)
python scripts/build_historical_dataset.py build

# v1 training on the expanded dataset
python -m mapclass.train --config mapclass/configs/v1.yaml
```

## Data Flow

### End-to-end training-and-inference flow

```
[1] RAW
    Azgaar GeoJSON ────────► scripts/build_dataset.py ──────────────────┐
    Rumsey LUNA + WMS ─────► scripts/build_historical_dataset.py ──────┤
    OSM PBF/tile ─────────► scripts/build_road_dataset.py    [NEW] ────┤
    ESA WorldCover S3 ─┐                                                │
                       ├──► (consumed inside historical + road) ────────┤
    Copernicus DEM S3 ─┘                                                │
                                                                        ▼
[2] SAMPLE STORE   data/{renders,historical,road}/.../*.png + JSON      │
                                                                        │
[3] LOSS WEIGHTS   mapclass/configs/loss_weights.yaml ──────────────────┤
                                                                        │
[4] DATASET ADAPTER   mapclass.data.dataset wraps [2] + [3] as torch ───┤
    Dataset; emits (image, lc_label, topo_label, lc_weight, topo_weight)│
                                                                        │
[5] v0 TRAINING       python -m mapclass.train --config v0.yaml ────────┤
                       reads [4]; writes models/geovilm_v0.pt ─────────►models/geovilm_v0.pt
                                                                        │
[6] BOOTSTRAP         scripts/georef.py bootstrap reads               ──┘
                       data/historical/raw/unregistered_manifest.json
                       + models/geovilm_v0.pt
                       writes new GeoTIFFs to
                       data/historical/raw/georeferenced/
                       (re-running step [1] for the historical pipeline
                        now picks them up automatically)

[7] v1 TRAINING       python -m mapclass.train --config v1.yaml
                       (same code path, larger dataset)
                       writes models/geovilm_v1.pt

[8] INFERENCE         from mapclass.infer import load_model
                       p = load_model("models/geovilm_v1.pt")
                       lc_probs, topo_probs = p.predict(image)
                       ── consumed by the external hex-grid app

[9] EVAL              python -m mapclass.eval --split heldout
                       --checkpoint models/geovilm_v1.pt
                       prints per-pixel NLL + qualitative renders for
                       Tolkien / Westeros / Abercrombie / Warhammer
```

### Per-batch training data flow

```
mapclass.data.dataset.MultiSourceDataset
    │  (mixes synthetic_azgaar, rumsey_registered, rumsey_bootstrapped,
    │   road_osm by configurable per-source sample probability)
    ▼
{image: HxWx3 uint8,
 land_cover: HxW uint8,
 topography: HxW uint8,
 source_subtype: str,
 sample_id: str}
    │
    ▼
mapclass.data.transforms (image → tensor; augment via existing scripts/augment.py)
    │
    ▼
mapclass.data.collate  (batch + per-pixel weight construction:
                        weight[i,j] = loss_weights[source_subtype][class_at[i,j]])
    │
    ▼
mapclass.model.geovilm.GeoViLM(image)
    │
    ├── backbone.extract_features(image) → {"stage_k": features_k}
    │
    ├── seg_heads.LandCoverHead(features) → lc_logits HxWx9
    │
    ├── seg_heads.TopographyHead(features) → topo_logits HxWx3
    │
    └── ocr_head(features [+ image]) → text-extraction loss component
                                       (auxiliary, used only when curated text labels are present)
    ▼
mapclass.model.losses.weighted_nll(lc_logits, lc_label, lc_weight)
                + weighted_nll(topo_logits, topo_label, topo_weight)
                + λ * ocr_loss   (λ scheduled / off if labels absent)
```

### Inference flow (the hex-grid app's consumed surface)

```python
# What the external hex-grid app does:
from mapclass.infer import load_model

predictor = load_model("models/geovilm_v1.pt", device="cpu")  # or "cuda"

# Image is anything the user gives — fantasy map JPG/PNG, satellite tile, etc.
import PIL.Image
image = PIL.Image.open("middle_earth.png").convert("RGB")

result = predictor.predict(image)
# result.land_cover_probs:  numpy float32, shape (H, W, 9)
# result.topography_probs:  numpy float32, shape (H, W, 3)
# result.sentinel_mask:     numpy bool,   shape (H, W)   # True = predict-with-low-confidence
```

This `Predictor.predict` signature is the **stable contract** with the downstream hex-grid app. Everything else inside `mapclass/` can change without breaking the consumer.

## Build Order (Dependency DAG)

```
Independent (Phase 6 / parallel-execution-safe):

  ┌───────────────────────────────────────────────┐
  │ A. Sample-contract finalization               │   ← BLOCKS B, C, D, E, F
  │    - manifest.json schema                     │
  │    - mapclass.data.contract validator         │
  │    - mapclass/configs/loss_weights.yaml seed  │
  └───────────────────────────────────────────────┘

  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
  │ B. OSM sub-pipeline  │  │ C. Auto-georef       │  │ D. Backbone interface│
  │    scripts/road/     │  │    OFFLINE mode      │  │    + Mock backbone   │
  │                      │  │    (sanity-check on  │  │                      │
  │ Depends on: A        │  │    known-coord TIFs) │  │ Depends on: A        │
  │                      │  │ Depends on: A        │  │                      │
  └──────┬───────────────┘  └──────┬───────────────┘  └──────┬───────────────┘
         │                         │                         │
         │                         │                         │
         └────┐                    │                ┌────────┘
              │                    │                │
              ▼                    │                ▼
        ┌────────────────────────────────────────────────┐
        │ E. mapclass.data.* (sources, dataset, types,   │
        │    loss_weights loader, transforms, collate)   │
        │                                                │
        │ Depends on: A + B  (needs OSM source impl to   │
        │             be a Source-conformant case)       │
        └──────────────────┬─────────────────────────────┘
                           │
                           ▼
                    ┌──────────────────────────────┐
                    │ F. mapclass.model.* (heads,  │
                    │    GeoViLM module, losses)   │
                    │                              │
                    │ Depends on: D + E (needs the │
                    │             dataset shape)   │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ G. mapclass.train +          │
                    │    mapclass.eval +           │
                    │    mapclass.infer            │
                    │                              │
                    │ Depends on: F                │
                    └──────────────┬───────────────┘
                                   │
                                   ▼ (specific backbone replaces Mock)
                    ┌──────────────────────────────┐
                    │ H. v0 training run           │
                    │    (cloud VM, SSH workflow)  │
                    │ Produces: models/geovilm_v0.pt│
                    │                              │
                    │ Depends on: G + STACK choice │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ I. Auto-georef BOOTSTRAP mode│
                    │    (scripts/georef bootstrap)│
                    │                              │
                    │ Depends on: C (sanity-check  │
                    │             implementation)  │
                    │           + H (v0 checkpoint)│
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ J. v1 training run           │
                    │    Same code as H, larger    │
                    │    dataset.                  │
                    │ Depends on: I + G            │
                    └──────────────────────────────┘
```

### What can run in parallel

The user picked parallel execution; the following pairs/triples have **zero shared work** and can be implemented simultaneously by independent agents:

- **B (OSM sub-pipeline) + C (Auto-georef offline mode) + D (Backbone interface)** — all three depend only on A (sample-contract). They're the "Phase 1 fan-out."
- **F.OCR head + F.LandCoverHead + F.TopographyHead** — three heads, three independent files, share only `Backbone.feature_channels`.
- **B + I.refactor** — once v0 exists, the OSM source can be tuned in parallel with bootstrap-loop refinement.

What **cannot** run in parallel:

- **A** is the choke point: every other piece consumes the sample contract. Lock the schema (including `manifest.json`) before fanning out.
- **H (v0 training) and I (bootstrap)** are sequential: bootstrap needs v0.
- **J (v1 training) and I (bootstrap)** are sequential: v1 retraining needs the bootstrap-expanded dataset.

## Key Abstractions (the four worth building once)

### 1. `Source` interface

```python
# mapclass/data/sources.py
from abc import ABC, abstractmethod
from pathlib import Path
from .types import LabelPair

class Source(ABC):
    """A dataset source. One per data origin (synthetic, historical, OSM)."""

    name: str                       # "synthetic_azgaar", "rumsey_registered", ...
    root: Path                      # where samples live, e.g. data/historical/dataset/

    @abstractmethod
    def __iter__(self): ...         # yields sample IDs
    @abstractmethod
    def __len__(self) -> int: ...
    @abstractmethod
    def load(self, sample_id: str) -> LabelPair:
        """Returns the sample contract: (image, land_cover, topography, weights, manifest)."""
        ...
```

Concrete impls: `SyntheticSource`, `HistoricalSource` (registered), `HistoricalBootstrappedSource` (registered_via=auto_georef_v0), `RoadSource`. The trainer mixes them via configurable per-source sampling probability.

**Why build it once:** the trainer is the only place that needs to know about source identity (for loss-weight lookup); everything downstream of `load()` is identical.

### 2. `LossWeights` schema

The YAML in `mapclass/configs/loss_weights.yaml` (shown above) is parsed into a typed object once at startup:

```python
# mapclass/data/loss_weights.py
from dataclasses import dataclass

@dataclass(frozen=True)
class LossWeights:
    """Per-source × per-class weights, loaded once from YAML."""
    land_cover: dict[str, dict[str, float]]   # source_subtype -> class_name -> weight
    topography: dict[str, float]              # source_subtype -> weight

    @classmethod
    def load(cls, path: Path) -> "LossWeights": ...

    def lookup(self, source_subtype: str, lc_class_name: str) -> tuple[float, float]:
        """Returns (lc_weight, topo_weight)."""
        ...
```

**Why build it once:** TESTING.md gap-item 6 calls this dict out specifically as schema-fragile. A dataclass loader + a one-line schema test pins it.

### 3. `MapImage` / `LabelPair` types

```python
# mapclass/data/types.py
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class MapImage:
    rgb: np.ndarray          # uint8, HxWx3
    style: str               # "satellite" | "flat" | "illustrated" | "historical" | "road"

@dataclass
class LabelPair:
    image: MapImage
    land_cover: np.ndarray   # uint8, HxW, values 0-8 + 255
    topography: np.ndarray   # uint8, HxW, values 0-2 + 255
    weights_path: Path       # location of sample_weights.json (per-sample override channel)
    manifest: SampleManifest # parsed manifest.json
    source_subtype: str      # canonical name: synthetic_azgaar, rumsey_registered, ...
    sample_id: str
```

**Why build it once:** every consumer (Dataset, eval, infer) needs the same fields. Defining them as a dataclass instead of a dict catches schema drift at import time.

### 4. `Backbone` interface

Already shown in Pattern 3. The reason it's a top-3 abstraction is that the STACK-research phase has **not yet picked the backbone** — defining the swap surface unblocks parallel work on the heads, dataset, and training loop.

## Entry Points (cloud-VM SSH workflow contract)

These are the commands the cloud VM runs over SSH. They MUST keep working without docker/containers (per the `Constraints` section of `PROJECT.md`).

| Command | Purpose | Status | Deps |
|---------|---------|--------|------|
| `python scripts/build_dataset.py <raw_dir> <output_dir>` | Synthetic dataset | EXISTS | A |
| `python scripts/build_historical_dataset.py {search\|build\|full}` | Historical dataset | EXISTS | A |
| `python scripts/build_road_dataset.py {search\|build\|full}` | OSM dataset | NEW | A, B |
| `python scripts/georef.py sanity-check <tif_dir>` | Offline sanity-check (no model) | NEW | A, C |
| `python scripts/georef.py bootstrap --model models/geovilm_v0.pt --manifest <path>` | v0-driven Rumsey registration | NEW | C, H |
| `python -m mapclass.train --config mapclass/configs/v0.yaml` | v0 training | NEW | A–G |
| `python -m mapclass.train --config mapclass/configs/v1.yaml` | v1 training | NEW | A–G, I |
| `python -m mapclass.eval --split heldout --checkpoint <ckpt>` | Held-out NLL + qualitative renders | NEW | F, G |

The `python -m mapclass.X` style requires `mapclass/` to be a real package on `sys.path`; on the cloud VM this means `pip install -e .` from the repo root, which in turn means we **MUST** add a minimal `pyproject.toml` (currently absent per `STACK.md`). This is a small but real new artefact for v1.

For the existing `python scripts/build_*.py` style, no `pyproject.toml` is required — they keep using the bare-name imports per the existing convention. So the addition is additive: `mapclass/` is installable; `scripts/` keeps working as before.

## Scaling Considerations

This is a research codebase with one developer; "scaling" here means dataset size and training epochs, not concurrent users. The relevant cliffs:

| Scale | Architecture impact | What gives |
|-------|---------------------|-----------|
| <1k samples per source | Filesystem layout fine; load entire manifest in memory | Nothing |
| 1k–10k samples | Pre-shuffle index file per source; lazy load images | Stream sample IDs from a per-source manifest.json instead of `os.listdir()` |
| 10k–100k samples | Sample-id index becomes a SQLite or LMDB; raw PNGs stay on disk | Image loading throughput; consider WebDataset shards |
| >100k samples | Out of v1 scope. WebDataset / TFRecord / lakeFS territory | Reconsider with the team that exists at that scale (i.e. not now) |

Inference scaling is fixed by the inference budget (single CPU or 4–8 GB GPU); architecture supports neither batched server inference nor multi-image throughput optimisation in v1, by design — the consumer is one hex-grid app processing one map at a time.

## Anti-Patterns

### Anti-Pattern 1: Mixing source identity into the model

**What people do:** Pass `source_subtype` (or worse, a one-hot source embedding) into the model so it "knows" which dataset a sample came from.

**Why it's wrong:** The whole point of multi-source training is that the model generalises across sources. If the model conditions on source identity at training time, inference (which has no source label) collapses to whichever source happened to be most common.

**Do this instead:** Source identity flows only into the **loss weights**, not into the forward pass. The model sees a uniform `(image → labels)` task; the trainer sees a non-uniform `(image, labels, weights)` task. This is what the YAML-loaded `LossWeights.lookup()` does in Pattern 2.

### Anti-Pattern 2: In-process bootstrap loop

**What people do:** Call the auto-georef code from inside the training loop, e.g. "every 5 epochs, register more maps and add them to the training set."

**Why it's wrong:** Conflates training (in-process, GPU-bound, stateful) with dataset construction (filesystem-bound, embarrassingly parallel, debuggable). When auto-georef fails on a single map, the training loop crashes with a TPS error, and the developer has no way to bisect. Also: registration-then-relabel is a multi-minute-per-map operation; doing it inside training wastes GPU time.

**Do this instead:** Filesystem-mediated as in Pattern 4 — `train v0`, then a separate `bootstrap` command, then `train v1`. Each step is independently invocable.

### Anti-Pattern 3: Ad-hoc per-source code paths in the trainer

**What people do:** `if source == "historical": apply_special_logic_X()` inside the training loop or the dataset.

**Why it's wrong:** Source-specific logic is exactly what `Source` and `LossWeights` are designed to encapsulate. Once an `if source == ...` branch shows up in the trainer, the abstraction has leaked, and the next source addition will require touching the trainer instead of just adding a new YAML row.

**Do this instead:** Push source-specific logic into the `Source` subclass's `load()` (data shape) or into the YAML (weights). The trainer should be source-blind.

### Anti-Pattern 4: Coupling inference to the dataset pipeline

**What people do:** `mapclass.infer` imports `mapclass.data.sources` to "reuse the loader."

**Why it's wrong:** Dataset loading is a training-time concern with multi-source mixing, augmentation, and collation. Inference takes a single PIL image from the user's hex-grid app and returns probabilities. They should not share imports — and crucially, the inference module should not require `rasterio` or `torch.utils.data.DataLoader` to be importable, because those drag in heavy deps that the inference deployment doesn't need.

**Do this instead:** `mapclass.infer` imports `mapclass.model.geovilm` and nothing from `mapclass.data`. The `Predictor.predict(image: PIL.Image) -> Result` API is purely model-facing.

### Anti-Pattern 5: Engine-specific output formats sneaking into `mapclass.infer`

**What people do:** Add an `infer.predict_eu4_terrain_bmp()` convenience method because someone wants it.

**Why it's wrong:** Locks the model API to a specific game engine. The `PROJECT.md` constraints explicitly route engine-specific packaging to the downstream hex-grid app.

**Do this instead:** `Predictor.predict(image)` returns generic per-pixel probability tensors; the hex-grid app does whatever post-processing produces `terrain.bmp` / `heightmap.png` / etc. Quality-gate item: if a file in `mapclass/` ever imports anything game-engine-specific, the gate fails.

## Integration Points

### External Services (no change from existing pipeline)

| Service | Integration | Notes |
|---------|------------|-------|
| David Rumsey LUNA | `requests` + retry, existing | unchanged |
| Georeferencer WMS | `requests`, existing | unchanged |
| ESA WorldCover S3 | `rasterio` + `AWS_NO_SIGN_REQUEST=YES` | unchanged; **reused by OSM sub-pipeline** |
| Copernicus DEM S3 | `rasterio` + `AWS_NO_SIGN_REQUEST=YES` | unchanged; **reused by OSM sub-pipeline** |
| OSM Overpass API or Geofabrik PBF | `requests` (Overpass) or `pyrosm` (PBF) | NEW; STACK research phase decides which |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `scripts/*` ↔ `mapclass/*` | Filesystem-only (sample contract) | Keeps the scripts side from being import-coupled to the model package |
| `mapclass.data` ↔ `mapclass.model` | Tensor + metadata dict, no filesystem | The model never reads disk |
| `mapclass.train` ↔ checkpoint | `torch.save` to `models/*.pt` | Checkpoint format is a separate (already-stable) PyTorch contract |
| `mapclass.infer` ↔ external hex-grid app | Python import; `Predictor.predict(image) -> Result` | **Stable contract**; everything else can change |
| Auto-georef bootstrap ↔ historical pipeline | Filesystem (`data/historical/raw/georeferenced/*.tif`) | Bootstrap writes TIFs that look identical to WMS-downloaded ones; the historical labeller is unchanged |

## Risk Flags (read alongside `.planning/codebase/TESTING.md` and `CONCERNS.md`)

These are architecture-level risks specific to the new components. Each maps to a concern that should be deepened in phase-specific research later.

1. **Sample contract is enforced by convention, not schema.** Adding `manifest.json` and a `LossWeights` YAML are both new and untested. **Mitigation:** add `mapclass.data.contract.assert_sample_valid(path)` invoked at every load; pin its schema with a unit test if we ever stop deferring tests.

2. **`scripts/` bare-name imports persist as fragility.** This was already flagged. Adding `mapclass/` doesn't fix it; it just gives new code a clean home. The fragility lives on for the existing pipelines until/unless they're moved into `mapclass.synthetic` and `mapclass.historical` (a refactor whose cost > value at v1).

3. **OSM data licensing.** ODbL share-alike on derived datasets. PROJECT.md already routes "don't redistribute the dataset" — but the OSM-derived training data adds a fourth licensing layer to the existing three (Rumsey, DEM, Azgaar). Verify this is contemplated in the executive_TODO.md license review.

4. **Auto-georef quality on heavily-stylized maps.** Cross-correlation against WorldCover is a strong cue for water/coastlines; it's a weak cue for forest cover that has changed since the 17th century. The bootstrap step will introduce *some* mis-registered maps into v1 training. The `confidence: 0.5` multiplier on `rumsey_bootstrapped` weights is the v1 mitigation; tune in research-phase.

5. **Backbone choice timeline.** F (model heads) and G (training entry point) need *some* backbone interface defined to proceed. The `MockBackbone` returning random-shape feature maps unblocks parallel work but adds a step (replace Mock → real backbone) at the v0-training boundary. **Mitigation:** lock the `Backbone.feature_channels` shape during stack-research even if the specific class isn't picked yet.

6. **The cloud-VM SSH workflow needs a `pyproject.toml`.** Currently absent (`STACK.md`). To support `python -m mapclass.train`, the package needs to be `pip install -e .`-able. This is one ~15-line file but it does mean introducing a build-system config that doesn't exist today.

## Sources

- `.planning/codebase/ARCHITECTURE.md` (existing pipelines, anchor for integration)
- `.planning/codebase/STRUCTURE.md` (directory layout, "Where to Add New Code" section)
- `.planning/codebase/STACK.md` (existing deps; `pyproject.toml` absence)
- `.planning/codebase/CONVENTIONS.md` (naming, error-handling, CLI conventions to preserve)
- `.planning/codebase/TESTING.md` (six untested risk areas that this architecture must not worsen)
- `.planning/PROJECT.md` (locked v1 scope and Constraints section)
- `README.md` and `mockup.md` (component list and academic framing)
- [SigLIP 2 paper (HuggingFace blog)](https://huggingface.co/blog/siglip2) — confirms ViT-B-86M small-VL viability for the dense-prediction backbone (specific choice deferred to STACK research)
- [SigLIP 2 arXiv:2502.14786](https://arxiv.org/abs/2502.14786) — same, for the dense-features claim
- [Dynamic-LRP arXiv:2512.07010](https://arxiv.org/pdf/2512.07010) — referenced in `mockup.md` as the milestone-2 mechanistic-failure-analysis tool; not consumed by v1 architecture but flagged so the `Backbone` interface stays compatible

---

*Architecture research for: MapClass / GeoViLM (multi-source ML training + dense-prediction inference, brownfield)*
*Researched: 2026-05-08*
