# Codebase Structure

**Analysis Date:** 2026-05-08

## Directory Layout

```
mapclass/
├── README.md                       # Implementation companion: phases, components, repo layout
├── mockup.md                       # Academic framing for the system paper
├── executive_TODO.md               # Owner-actionable todos (licensing, zero-shot benchmarks)
├── LICENSE
├── requirements.txt                # Pillow, numpy, rasterio, pyproj, requests, torch, transformers, peft, accelerate, bitsandbytes
├── .gitignore                      # `data/` is gitignored
│
├── notes/
│   └── architectural_references.md # Backbone / hierarchical-vs-large-kernel discussion
│
├── scripts/                        # All implementation code (synthetic + historical pipelines)
│   ├── build_dataset.py            # Synthetic orchestrator — entry point
│   ├── render.py                   # Multi-style polygon rasteriser (flat / illustrated / satellite)
│   ├── label.py                    # Synthetic label generator (land_cover.png + topography.png)
│   ├── biome_mapping.py            # Canonical taxonomy + Azgaar biome → canonical mapping
│   ├── augment.py                  # Pillow-based augmentations (parchment, sepia, faded, grids)
│   ├── toon_mapping.py             # Hex-tile filename → (land_cover, topo); PaliGemma prompt helpers
│   ├── build_historical_dataset.py # Historical orchestrator — entry point (search/build/full)
│   └── historical/                 # Sub-package for historical-map sub-pipeline
│       ├── __init__.py             # (empty; marks the package)
│       ├── rumsey.py               # David Rumsey LUNA API client + Georeferencer WMS download
│       ├── worldcover.py           # ESA WorldCover S3 fetcher + canonical remap
│       ├── dem.py                  # Copernicus DEM GLO-30 S3 fetcher + slope-based topo classifier
│       └── label.py                # Per-GeoTIFF label generator (image + masks + sample_weights.json)
│
├── data/                           # NOT TRACKED IN GIT — populated at runtime
│   ├── raw/                        # Azgaar GeoJSON exports (synthetic input)
│   ├── renders/                    # Synthetic outputs: <name>/{flat,illustrated,satellite,land_cover,topography}.png
│   ├── labels/                     # Pixel-level label rasters (per README; in practice labels live alongside renders today)
│   ├── historical/                 # Historical pipeline working tree
│   │   ├── raw/
│   │   │   ├── georeferenced/      # GeoTIFFs downloaded from Georeferencer WMS
│   │   │   └── unregistered_manifest.json   # Maps needing manual QGIS GCPs
│   │   └── dataset/                # Per-map outputs: <id>/{image,land_cover,topography}.png + sample_weights.json
│   ├── toons/                      # Hex tile asset library (validation set / pre-labelled features)
│   └── TODO.md                     # Open data-side todos
│
├── models/                         # Trained model checkpoints — currently empty (planned)
│
└── .planning/
    └── codebase/                   # GSD codebase mapping documents (this file)
```

## Directory Purposes

### `scripts/` (top level)

- Purpose: synthetic-side modules and the two orchestrator entry points.
- Contains: flat-module Python files. Not currently a Python package — there
  is no `scripts/__init__.py` — but its modules import each other by bare
  module name (e.g. `from biome_mapping import …`), which works only when
  `scripts/` is on `sys.path`. The historical orchestrator sets this up
  explicitly at import time (`scripts/build_historical_dataset.py:33-35`).
- Key files:
  - `build_dataset.py` — synthetic entry point
  - `build_historical_dataset.py` — historical entry point
  - `biome_mapping.py` — canonical taxonomy (referenced by all other modules)

### `scripts/historical/`

- Purpose: data fetching and labelling for the historical-map source.
- Contains: a real Python package (has `__init__.py`).
- Key files:
  - `rumsey.py` — only entry point that talks to a non-S3 external service
  - `worldcover.py`, `dem.py` — pure S3-fetching + reprojection
  - `label.py` — per-map orchestrator, imported by the top-level historical
    orchestrator and also runnable standalone via `python -m
    historical.label`.

### `notes/`

- Purpose: design rationale and reading lists, kept out of `README.md` so
  the README stays focused on the implementation phase plan.
- Contains today: `architectural_references.md` only.
- Expected to grow with `mockup.md`-adjacent discussion documents (loss
  design, ablation rationale, etc.).

### `data/` (gitignored, populated at runtime)

- Purpose: every byte of training input and output. The README enumerates
  the layout; the actual on-disk shape is created by the orchestrators on
  first run.
- Subdirectories:
  - `data/raw/` — drop Azgaar `.geojson` exports here.
  - `data/renders/<name>/` — synthetic pipeline output, one directory per
    input GeoJSON.
  - `data/labels/` — listed in the README but the synthetic pipeline writes
    labels next to renders (`data/renders/<name>/land_cover.png`); this
    directory is reserved for cross-source aggregated labels if/when that
    pattern emerges.
  - `data/historical/raw/georeferenced/` — Rumsey WMS-downloaded GeoTIFFs.
  - `data/historical/raw/unregistered_manifest.json` — list of maps that
    need manual QGIS GCP placement.
  - `data/historical/dataset/<id>/` — historical pipeline output, one
    directory per map.
  - `data/toons/` — hex tile assets (validation / pre-labelled feature
    library); consumed by `scripts/toon_mapping.py:load_labeled_tiles()`.
  - `data/TODO.md` — data-side todo list (separate from `executive_TODO.md`).

### `models/` (empty)

- Purpose: trained model checkpoints. Listed in `README.md` Repo layout.
- Currently empty. Will hold checkpoints for the (planned) GeoViLM
  components — backbone fine-tuning weights, segmentation head weights, OCR
  module weights.

### `.planning/codebase/`

- Purpose: GSD pipeline outputs (this document and its peers). Consumed by
  the `/gsd-plan-phase` and `/gsd-execute-phase` commands.

## Key File Locations

### Entry points

- `scripts/build_dataset.py` — synthetic dataset orchestrator
- `scripts/build_historical_dataset.py` — historical dataset orchestrator
- `scripts/historical/label.py` — per-GeoTIFF historical label generator
  (also runnable standalone)

### Configuration

- `requirements.txt` — flat pip dependency list. No version pinning beyond
  `transformers>=4.41` and `peft>=0.10`.
- There is no `pyproject.toml`, `setup.py`, `setup.cfg`, `tox.ini`,
  `.pre-commit-config.yaml`, `pytest.ini`, or `conftest.py`.
- There is no `.env`, `.env.example`, or any other secret-bearing file.

### Core logic

- Canonical taxonomy: `scripts/biome_mapping.py` —
  `LANDCOVER_CLASSES`, `TOPO_CLASSES`, `h_to_landcover`, `h_to_topo`.
- Synthetic rasterisation: `scripts/render.py:render_map()`,
  `scripts/label.py:make_labels()`.
- Historical fetchers: `scripts/historical/worldcover.py:fetch_worldcover()`,
  `scripts/historical/dem.py:fetch_topo()`.
- Historical label generator: `scripts/historical/label.py:make_labels()`.

### Documentation

- Implementation plan: `README.md` (Phase plan section)
- Academic framing: `mockup.md`
- Architectural rationale (backbones, hierarchical vs. large-kernel):
  `notes/architectural_references.md`
- Owner-action TODOs: `executive_TODO.md`

### Testing

- **None.** There is no `tests/` directory and no test files anywhere.

## Naming Conventions

### Python files

- Lowercase, underscore-separated: `build_dataset.py`,
  `build_historical_dataset.py`, `biome_mapping.py`, `toon_mapping.py`.
- Top-level orchestrators are prefixed `build_` and have a counterpart
  in-pipeline module (`render.py`, `label.py`, `historical/label.py`).
- Pipeline-internal modules are named after the data source they wrap:
  `rumsey.py`, `worldcover.py`, `dem.py`.

### Output sample directories

- Synthetic: one directory per input GeoJSON, named after its stem
  (`<name>` from `<name>.geojson`).
  - Files: `flat.png`, `illustrated.png`, `satellite.png`, `land_cover.png`,
    `topography.png`.
- Historical: one directory per input GeoTIFF, named after its stem (the
  Rumsey item `id` for WMS-downloaded maps, or whatever the user named the
  manually-registered file).
  - Files: `image.png`, `land_cover.png`, `topography.png`,
    `sample_weights.json`.

### Constants / enums

- Class lists: ALL-UPPER plural — `LANDCOVER_CLASSES`, `TOPO_CLASSES`,
  `_MAP_TYPES`.
- Lookup tables: ALL-UPPER, source name first — `BIOME_TO_LANDCOVER_NAME`,
  `WC_REMAP`, `HISTORICAL_LC_WEIGHTS`, `TERRAIN_DESCRIPTIONS`.
- Sentinels: ALL-UPPER short — `NODATA = 255`, `WATER_TOPO = 255`.
- Module-private helpers / constants are `_underscore_prefixed` — `_FLAT`,
  `_LUNA_SEARCH`, `_haversine_km`, `_get_json`.

### Hex tile filenames

- Convention: `hex_<terrain>_<variant>_<style>_<n>.png` (see
  `scripts/toon_mapping.py:6`).
- Terrain prefix uniquely identifies the (land_cover, topography) class
  pair; matching is done by longest-prefix-first against `_PREFIX_MAP`.

## Where to Add New Code

The README's phase plan is the authority on where each planned component
lands. Below is the mapping from phase to expected on-disk location, plus
guidance for additions inside the implemented pipelines.

### New synthetic-side feature (style, augmentation, label channel)

- New rendering style: extend the palette dicts and `BORDER_STYLE`,
  `BACKGROUND` in `scripts/render.py:29-83`. The orchestrator will pick it
  up automatically because the styles list is parameterised.
- New augmentation: add to `scripts/augment.py`. Keep stateless unless
  prefixed `random_`.
- New label channel beyond land_cover + topography: add a third
  `Image.new("L", …)` and corresponding draw call in
  `scripts/label.py:make_labels()`, and a parallel rendering pass in
  `scripts/render.py` if the channel needs a visualisation.

### New historical-side data source

- New geospatial reference layer (e.g. OSM water / road rasters): add a
  fetcher module in `scripts/historical/` modelled on
  `worldcover.py` (S3 + remap) or `dem.py` (S3 + per-pixel derivation), and
  invoke it from `scripts/historical/label.py:make_labels()`.
- New metadata source for the search phase: extend
  `scripts/historical/rumsey.py:_richness_score()` and
  `_field()` lookups.

### New raw map source (the README's "road maps" gap)

- Per the README, this is "TBD" with OSM tile renders as the obvious
  candidate.
- Expected location: `scripts/road_maps.py` (or `scripts/road/`) plus a new
  top-level orchestrator `scripts/build_road_dataset.py`.
- Expected output: `data/road/dataset/<id>/{image,land_cover,topography}.png
  + sample_weights.json` to match the existing sample contract.

### Phase 2 — Failure-analysis benchmark (planned)

- New top-level location is **not** dictated by the README. Reasonable
  candidates: a new `benchmark/` package at the repo root, or
  `scripts/benchmark/` if we want to keep all scripts under one tree.
- Expected modules:
  - Zero-shot evaluation harness for CLIP / SigLIP / OpenCLIP / PaliGemma
    against the held-out stylized test set.
  - Dynamic-LRP analysis (per `mockup.md`,
    [arXiv:2512.07010](https://arxiv.org/pdf/2512.07010)) for failure
    localisation.
- Outputs: per-class accuracy tables, confusion matrices, LRP attribution
  maps. Likely lands in `data/benchmark/` or `models/benchmarks/`.

### Phase 3 — GeoViLM construction (planned)

The README lists four components. Suggested placement (none of these exist
yet):

- **Vision-language backbone** — `geovilm/backbone.py` or
  `models/backbone/`. PaliGemma-3B loading via `transformers`/`peft` (already
  in `requirements.txt`). LoRA fine-tuning config alongside.
- **Rotationally-invariant OCR** — `geovilm/ocr/` (sub-package). May need
  its own training data pipeline; expect a `scripts/build_ocr_dataset.py`
  parallel to the existing two orchestrators.
- **Dense segmentation heads** — `geovilm/heads/landcover.py`,
  `geovilm/heads/topography.py`. Two lightweight heads on the backbone's
  vision features.
- **Auto-georeferencing pipeline** — `geovilm/georef/`. Cross-correlation
  against WorldCover + DEM reference grid + thin-plate-spline warping.
  Bootstrapped iteratively (v0 → v1).
- **Training loop** — `geovilm/train.py` or `train.py` at repo root. Must
  consume `sample_weights.json` from the historical pipeline.
- **Evaluation** — `geovilm/eval.py`. Joint per-pixel NLL metric per
  README Phase 4.

The exact top-level package name (`geovilm/`, `mapclass/`, `models/<x>/`) is
**not** decided in the codebase today; the README only states that
checkpoints land in `models/`. The first phase planning step that touches
modelling code should pick this name explicitly and stick with it.

### Utilities

- Shared geometry helpers (e.g. `_bbox`, `_rings` currently duplicated in
  `render.py` and `label.py`): extract to
  `scripts/geojson_utils.py` or fold into `biome_mapping.py`.
- Anything that becomes used across both pipelines: lift into
  `scripts/` top level (next to `biome_mapping.py`), or — preferably — into
  a new top-level `mapclass/` package once the directory is created for the
  modelling layer.

## Special Directories

### `data/`

- Purpose: dataset working tree (raw inputs and rendered outputs).
- Generated: yes (by `build_dataset.py` and `build_historical_dataset.py`).
- Committed: **no** (gitignored at repo root).
- Implication: nothing here is required to read or build the codebase. A
  fresh clone has no `data/` directory; it is created lazily by the
  orchestrators.

### `models/`

- Purpose: trained model checkpoints.
- Generated: planned, not yet generated.
- Committed: not currently in repo. Likely gitignored once populated, given
  checkpoint sizes.

### `notes/`

- Purpose: design discussion that is too long for the README and too
  technical for `mockup.md`.
- Generated: no — hand-written.
- Committed: yes.

### `.planning/codebase/`

- Purpose: GSD command outputs (architecture maps, conventions, etc.).
- Generated: yes (by GSD `/gsd-map-codebase`).
- Committed: yes.

---

*Structure analysis: 2026-05-08*
