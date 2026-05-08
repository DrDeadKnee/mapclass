<!-- refreshed: 2026-05-08 -->
# Architecture

**Analysis Date:** 2026-05-08

## System Overview

MapClass / GeoViLM is a research project mid-refactor on branch `refactor_paper`,
moving from an earlier hex-tile fine-tuning plan toward a system-paper plan
(see `mockup.md`). The codebase today is **two batch data-construction
pipelines** plus a (planned, not-yet-coded) modelling stack. There is no
training loop, no model code, no evaluation harness — only data preparation.

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                       IMPLEMENTED — Dataset construction                 │
├──────────────────────────────────┬──────────────────────────────────────┤
│  Synthetic pipeline              │  Historical pipeline                 │
│  `scripts/build_dataset.py`      │  `scripts/build_historical_dataset.py`│
│         │                        │             │                        │
│         ▼                        │             ▼                        │
│  `scripts/render.py`             │  `scripts/historical/rumsey.py`      │
│  `scripts/label.py`              │  `scripts/historical/worldcover.py`  │
│  `scripts/biome_mapping.py`      │  `scripts/historical/dem.py`         │
│  `scripts/augment.py`            │  `scripts/historical/label.py`       │
│  `scripts/toon_mapping.py`       │                                      │
└─────────────┬────────────────────┴───────────────┬──────────────────────┘
              ▼                                    ▼
      `data/renders/<name>/`              `data/historical/dataset/<id>/`
       flat.png + illustrated.png +        image.png + land_cover.png +
       satellite.png + land_cover.png +    topography.png + sample_weights.json
       topography.png

╔═════════════════════════════════════════════════════════════════════════╗
║                  PLANNED — GeoViLM model components (NOT IN CODE)        ║
╠═════════════════════════════════════════════════════════════════════════╣
║  1. Vision-language backbone   — PaliGemma-3B (SigLIP + Gemma)          ║
║  2. Rotationally-invariant OCR — CRAFT / ABCNet starting points         ║
║  3. Dense segmentation heads   — land cover head + topography head      ║
║  4. Auto-georeferencing        — cross-correlation + thin-plate-spline  ║
║                                                                          ║
║  Plus: zero-shot benchmark harness, dynamic-LRP failure analysis,        ║
║  road-map source, end-to-end training loop, evaluation metric            ║
║  (joint per-pixel NLL).                                                  ║
║                                                                          ║
║  Target output directory: `models/` (currently empty)                    ║
╚═════════════════════════════════════════════════════════════════════════╝
```

The asymmetry is the dominant fact of this codebase: every box in the top
half exists; every box in the bottom half is described in `README.md` and
`mockup.md` but has zero code attached.

## Component Responsibilities

### Implemented — Synthetic dataset pipeline

| Component | Responsibility | File |
|-----------|----------------|------|
| Synthetic orchestrator | Iterate Azgaar GeoJSON exports, drive render + label per map | `scripts/build_dataset.py` |
| Multi-style renderer | Rasterise GeoJSON polygons to flat / illustrated / satellite PNGs | `scripts/render.py` |
| Synthetic label generator | Rasterise GeoJSON polygons to `land_cover.png` and `topography.png` masks | `scripts/label.py` |
| Biome / canonical taxonomy | Map Azgaar biome IDs and height to canonical 9-class land cover and 3-class topography | `scripts/biome_mapping.py` |
| Hex-tile prefix taxonomy | Map hex tile filenames (`hex_<terrain>_*.png`) to (land_cover, topography) labels | `scripts/toon_mapping.py` |
| Image augmentation | Parchment / sepia / faded / grayscale style transforms; grid compositor | `scripts/augment.py` |

### Implemented — Historical dataset pipeline

| Component | Responsibility | File |
|-----------|----------------|------|
| Historical orchestrator | `search` / `build` / `full` subcommands; thread-pool fan-out over GeoTIFFs | `scripts/build_historical_dataset.py` |
| David Rumsey LUNA client | Search 16th–17th c. maps; rank by metadata richness; download Georeferencer WMS GeoTIFFs; emit manifest of unregistered maps | `scripts/historical/rumsey.py` |
| WorldCover fetcher | Pull ESA WorldCover 10 m tiles from S3, reproject into target raster grid, remap to canonical 9-class taxonomy | `scripts/historical/worldcover.py` |
| DEM + slope classifier | Pull Copernicus DEM GLO-30 tiles, compute slope in a local LAEA projection, classify into flat / hilly / mountainous | `scripts/historical/dem.py` |
| Historical label generator | Per-GeoTIFF: emit `image.png`, `land_cover.png`, `topography.png`, `sample_weights.json` (per-class temporal-reliability weights) | `scripts/historical/label.py` |

### Planned — GeoViLM model components (no code yet)

| Component | Responsibility | Status |
|-----------|----------------|--------|
| Vision-language backbone | PaliGemma-3B benchmarked zero-shot, then fine-tuned in-system | Not implemented |
| Rotationally-invariant OCR | Read curved / rotated map text (place names, region labels) | Not implemented |
| Dense segmentation heads | Two lightweight heads on backbone vision features (land cover, topography) | Not implemented |
| Auto-georeferencing pipeline | Cross-correlation against WorldCover + DEM reference grid, thin-plate-spline warping; bootstrapped v0 → v1 | Not implemented |
| Zero-shot benchmark harness | CLIP / SigLIP / OpenCLIP / PaliGemma evaluation on held-out stylized test set + dynamic-LRP failure localisation | Not implemented (tracked in `executive_TODO.md`) |
| Road-map data source | OSM tile renders (TBD) | Not implemented |
| Training loop | End-to-end with per-source class-conditional loss weighting | Not implemented |
| Evaluation metric | Joint per-pixel NLL `−(log p_land_cover + log p_topography)` | Not implemented |

## Pattern Overview

**Overall:** Batch ETL data pipelines (Python scripts, no service / daemon /
notebook). Each pipeline is a flat module set in `scripts/` invoked by a
top-level orchestrator script.

**Key Characteristics:**
- Two parallel pipelines (synthetic, historical) with **the same output
  contract**: a directory containing image(s) + `land_cover.png` +
  `topography.png` (the historical pipeline additionally writes
  `sample_weights.json`).
- A single shared canonical taxonomy in `scripts/biome_mapping.py`
  (`LANDCOVER_CLASSES`, `TOPO_CLASSES`, `NODATA = 255`, `WATER_TOPO = 255`).
  Both pipelines and all external sources (Azgaar, ESA WorldCover, Copernicus
  DEM) are remapped into this taxonomy.
- No inter-pipeline imports apart from this shared taxonomy. The synthetic
  side is pure-Python + Pillow; the historical side adds rasterio + pyproj +
  requests for cloud-hosted geospatial data.
- No persistent state, no DB, no task queue. Each invocation is a fresh
  filesystem-to-filesystem pass.
- `scripts/historical/` is a Python package (`__init__.py` is present); the
  rest of `scripts/` is a flat module directory imported by sibling-on-sys.path
  hacks.

## Layers

### Data-input layer

- Purpose: ingest raw inputs (Azgaar GeoJSON exports, David Rumsey API,
  ESA WorldCover S3, Copernicus DEM S3).
- Location: `scripts/historical/rumsey.py`, `scripts/historical/worldcover.py`,
  `scripts/historical/dem.py`. Synthetic side reads GeoJSON in the renderer
  and label generator directly.
- Depends on: `requests`, `rasterio`, `pyproj`. Public S3 buckets accessed
  anonymously via `AWS_NO_SIGN_REQUEST=YES`.
- Used by: the orchestrators, indirectly through label generators.

### Taxonomy / mapping layer

- Purpose: collapse heterogeneous source labels into the canonical 9-class
  land cover + 3-class topography taxonomy.
- Location: `scripts/biome_mapping.py` (Azgaar → canonical),
  `scripts/historical/worldcover.py` (`WC_REMAP`),
  `scripts/historical/dem.py` (slope thresholds → topo class),
  `scripts/toon_mapping.py` (hex-tile filename → canonical).
- Depends on: nothing (pure-Python lookup tables and small numerical helpers).
- Used by: every label generator and renderer.

### Rasterisation / labelling layer

- Purpose: produce per-pixel masks aligned to a single reference image.
- Location: `scripts/render.py`, `scripts/label.py` (synthetic);
  `scripts/historical/label.py` (historical).
- Depends on: Pillow (synthetic), rasterio + numpy (historical).
- Used by: the orchestrators.

### Orchestration layer

- Purpose: glue file discovery, parallelism, and per-sample error handling.
- Location: `scripts/build_dataset.py`, `scripts/build_historical_dataset.py`.
- Depends on: the rasterisation and data-input layers.
- Used by: humans running `python scripts/build_dataset.py …`.

### (Planned) Modelling layer

- Purpose: training, inference, evaluation of the four GeoViLM components.
- Location: `models/` (currently empty per `README.md`); module placement TBD.
- Status: no code yet. The README phase plan and `mockup.md` describe what
  goes here.

## Data Flow

### Synthetic primary path

1. User exports a `.geojson` from Azgaar's Fantasy Map Generator into
   `data/raw/`.
2. `scripts/build_dataset.py:build()` (`scripts/build_dataset.py:25`) globs
   `*.geojson` in the input dir.
3. For each map, `render_map()` (`scripts/render.py:138`) rasterises the
   polygons under three palettes (flat / illustrated / satellite) inferred
   from `_FLAT`, `_ILLUSTRATED`, `_SATELLITE` in `scripts/render.py:29-69`.
   Land cover is determined per polygon via
   `h_to_landcover(h, biome)` (`scripts/biome_mapping.py:90`).
4. `make_labels()` (`scripts/label.py:51`) rasterises the same polygons into
   `land_cover.png` and `topography.png` using the same coordinate frame.
5. Outputs land in `data/renders/<name>/` (per `README.md` Repo layout).

### Historical primary path

1. `cmd_search()` (`scripts/build_historical_dataset.py:45`) calls
   `rumsey.search_maps()` (`scripts/historical/rumsey.py:192`), which
   collects a year-by-year pool from the LUNA API
   (`https://www.davidrumsey.com/luna/servlet/as/search`), then ranks by
   `_richness_score()` (`scripts/historical/rumsey.py:128`).
2. Each candidate is fed to `download_georeferenced()`
   (`scripts/historical/rumsey.py:319`), which fetches a GeoTIFF via the
   Georeferencer WMS service when registered, or otherwise records the map
   in `unregistered_manifest.json` for manual GCP placement in QGIS.
3. `cmd_build()` (`scripts/build_historical_dataset.py:82`) iterates all
   GeoTIFFs (with optional `ThreadPoolExecutor` fan-out) and calls
   `historical.label.make_labels()` (`scripts/historical/label.py:95`).
4. `make_labels()` opens the GeoTIFF, derives a WGS84 bbox via
   `_to_wgs84_bbox()`, then:
   - reads the RGB bands → `image.png`
   - calls `worldcover.fetch_worldcover()`
     (`scripts/historical/worldcover.py:94`) → `land_cover.png`
   - derives a `water_mask` (`lc_arr == 0`) and calls `dem.fetch_topo()`
     (`scripts/historical/dem.py:95`) → `topography.png`
   - writes `sample_weights.json` from `HISTORICAL_LC_WEIGHTS`
     (`scripts/historical/label.py:40`).
5. Outputs land in `data/historical/dataset/<id>/`.

### (Planned) Training / inference flow

The README describes — but no code implements — the following:
- v0 GeoViLM trained on synthetic + already-registered historical maps.
- v0 used to auto-georeference unregistered Rumsey maps (the manifest
  emitted by `rumsey.emit_manifest()`).
- v1 retraining on the expanded dataset.
- Per-source class-conditional loss weights consumed from the
  `sample_weights.json` files emitted by the historical pipeline.

**State Management:** None at runtime; everything is filesystem state. The
synthetic pipeline is fully deterministic given a GeoJSON. The historical
pipeline has external API non-determinism (LUNA result ordering, S3
availability) and uses `np.random` for paper-grain noise in
`scripts/augment.py`.

## Key Abstractions

### Canonical 9-class land cover taxonomy

- Purpose: single class-index space across all data sources.
- Definition: `LANDCOVER_CLASSES` in `scripts/biome_mapping.py:38`.
- Pattern: every source ships with a remap table — `BIOME_TO_LANDCOVER_NAME`
  (Azgaar), `WC_REMAP` (WorldCover), `_RAW_MAP` (hex tiles). All collapse into
  these nine indices.
- Sentinel values: `NODATA = 255` for unlabelled pixels; `WATER_TOPO = 255`
  for water-class pixels in topography masks (water has no topography class).

### Canonical 3-class topography taxonomy

- Purpose: same job, for slope-derived class.
- Definition: `TOPO_CLASSES = ["flat", "hilly", "mountainous"]` in
  `scripts/biome_mapping.py:51`.
- Thresholds:
  - Synthetic side: bands of `normalize_land_h(h)` in
    `scripts/biome_mapping.py:97-107` (≤20 flat, ≤55 hilly, else mountainous).
  - Historical side: slope-degree thresholds in `scripts/historical/dem.py:41-42`
    (`FLAT_MAX_DEG = 2.0`, `HILLY_MAX_DEG = 15.0`).

### Per-sample loss weights (temporal reliability)

- Purpose: encode the README claim that historical maps under-weight
  fast-changing classes (cropland, built-up, trees) but trust topography and
  water/coastlines.
- Definition: `HISTORICAL_LC_WEIGHTS` in `scripts/historical/label.py:40`.
- Output: written as `sample_weights.json` next to each label pair. Will be
  consumed by the (planned) training loop.

### Output sample contract

- Each dataset sample is a directory with image(s) + `land_cover.png` +
  `topography.png`. The historical side adds `sample_weights.json`.
- Class indices and sentinel values are identical across sides, by design.

## Entry Points

### `scripts/build_dataset.py`

- Location: `scripts/build_dataset.py`
- Triggers: `python scripts/build_dataset.py <raw_dir> <output_dir> [styles…]`
- Responsibilities: walk Azgaar `*.geojson` files; for each, call
  `render_map()` and `make_labels()`.

### `scripts/build_historical_dataset.py`

- Location: `scripts/build_historical_dataset.py`
- Triggers:
  `python scripts/build_historical_dataset.py {search|build|full} [flags]`
- Responsibilities: dispatch on subcommand; run search and/or build; manage
  worker thread pool; surface per-map errors without aborting the batch.

### `scripts/historical/label.py` (`__main__`)

- Location: `scripts/historical/label.py:152`
- Triggers: `python -m historical.label <map.tif> <output_dir>`
- Responsibilities: per-GeoTIFF execution path, useful for debugging one map
  without invoking the orchestrator.

### `scripts/render.py`, `scripts/label.py`, `scripts/toon_mapping.py` (`__main__`)

- All three expose `if __name__ == "__main__":` blocks for ad-hoc CLI use on
  a single map / tile directory.

## Architectural Constraints

- **Threading:** Synthetic pipeline is single-threaded and CPU-bound on
  Pillow. Historical pipeline supports a `--workers N` thread pool
  (`scripts/build_historical_dataset.py:104`). Workers share the GIL but
  spend most of their time blocked on S3 / WMS I/O, so the threading model
  is appropriate.
- **Global state:** Module-level constants only — palettes, remap tables,
  loss weights. No mutable singletons. `_RAW_MAP` is sorted into
  `_PREFIX_MAP` at import time in `scripts/toon_mapping.py:84`.
- **Import path hack:** `scripts/build_historical_dataset.py:33-35` inserts
  its own directory onto `sys.path` so that `from historical import label`
  works from either the repo root or the `scripts/` directory. Synthetic
  scripts (`build_dataset.py`, `render.py`, `label.py`) rely on being run
  with `scripts/` as the working directory or on `scripts/` already being on
  `sys.path` — they do `from label import …`, `from biome_mapping import …`
  with no path manipulation. This is fragile.
- **Coordinate systems:** Synthetic GeoJSON lives in Azgaar's native pixel
  space (no real CRS). Historical GeoTIFFs carry a real CRS; slope is
  computed in a per-map LAEA projection
  (`scripts/historical/dem.py:_laea_crs`). Re-projection happens via
  `rasterio.warp.reproject` rather than at the application layer.
- **Cloud / S3:** Anonymous public-bucket access via
  `AWS_NO_SIGN_REQUEST=YES` set at module import in
  `scripts/historical/worldcover.py:36` and `scripts/historical/dem.py:33`.
  No AWS credentials are required or supported.
- **No environment variables required** for normal operation; no
  `.env`-style secret loading; no API keys hard-coded. The only knob is
  `AWS_NO_SIGN_REQUEST` and that defaults itself.
- **No tests, no CI, no linter config**: there is no `tests/`, no
  `pyproject.toml`, no `.pre-commit`, no `pytest.ini`, no `conftest.py`. The
  codebase is being built without an automated regression net.

## Anti-Patterns

### `sys.path` injection over a real package

**What happens:** `scripts/build_historical_dataset.py:33-35` does
`sys.path.insert(0, str(_HERE))` and the synthetic scripts assume `scripts/`
is on `sys.path` implicitly so that `from label import make_labels` works.

**Why it's wrong here:** It makes the import graph depend on cwd, breaks
when the scripts are reorganised into a real package, and prevents the same
modules from being imported from a hypothetical training script in
`models/`.

**Do this instead:** Convert `scripts/` into a real package
(`scripts/__init__.py`), or move the importable code into a top-level
package such as `mapclass/` and keep `scripts/` for thin entry-point
wrappers. The historical sub-pipeline already does this correctly with
`scripts/historical/__init__.py`.

### Two parallel `_bbox()` and `_rings()` implementations

**What happens:** `scripts/render.py:86-105` and `scripts/label.py:28-48`
each define their own `_bbox()` and `_rings()` helpers with the same logic.

**Why it's wrong here:** The renderer and the label generator must agree on
image dimensions and coordinate origin pixel-exactly, or labels will
silently misalign with renders. Today they agree because the code is
duplicated; the moment one is updated and the other is not, the synthetic
dataset becomes silently corrupt.

**Do this instead:** Hoist `_bbox()` and `_rings()` into
`scripts/biome_mapping.py` (or a new `scripts/geojson_utils.py`) and import
from both call sites.

### Non-pip-installed cross-script imports

**What happens:** `scripts/render.py:23` does `from biome_mapping import …`
with no package qualifier; `scripts/build_dataset.py:21` does
`from label import make_labels`.

**Why it's wrong here:** It works only when the cwd is `scripts/` or
`scripts/` is on `sys.path`. Running `python scripts/build_dataset.py …`
from the repo root currently relies on `scripts/` not being a package, so
Python falls back to inserting the script's directory onto `sys.path`. This
is the same anti-pattern as the first one, just from the synthetic side.

**Do this instead:** Same fix — package-ify and use absolute imports.

## Error Handling

**Strategy:** Each per-sample failure is caught and reported, but the batch
continues. There is no retry queue.

**Patterns:**
- `scripts/build_historical_dataset.py:73-79`: `_process_one()` catches every
  exception, returns `(tif, exc)`, and the main loop accumulates errors and
  prints them at the end.
- `scripts/historical/rumsey.py:161-185`: `_get_json()` retries with
  exponential backoff on HTTP 429/503, raises after `_MAX_RETRIES = 4`.
- `scripts/historical/worldcover.py:131-132`: per-tile failures log a
  warning but do not abort the whole map.
- `scripts/historical/dem.py:127-128`: missing DEM tiles (over-ocean 404s)
  are silently swallowed — this is intentional and documented in the module
  docstring.
- Synthetic side has no real error handling — Pillow / JSON exceptions
  propagate to the orchestrator, which has no try/except and lets them kill
  the run. This is acceptable because the synthetic input is local files
  that the user controls.

## Cross-Cutting Concerns

**Logging:** Plain `print()` everywhere. No `logging` module, no log levels,
no structured logs. Acceptable for batch scripts; will need to be replaced
when training is added.

**Validation:** Minimal. The historical pipeline checks `ds.crs is not
None` (`scripts/historical/label.py:111`) and bbox-diagonal range
(`scripts/historical/rumsey.py:344-346`); the synthetic side trusts its
inputs.

**Authentication:** None. All external sources (LUNA API, ESA WorldCover S3,
Copernicus DEM S3) are anonymously accessible; the LUNA client only sets a
`User-Agent` header (`scripts/historical/rumsey.py:41`).

**Configuration:** Hard-coded constants in module-level globals. No config
files. No CLI flag exposes the date range, scale-band thresholds, or loss
weights — to change them you edit the source.

---

*Architecture analysis: 2026-05-08*
