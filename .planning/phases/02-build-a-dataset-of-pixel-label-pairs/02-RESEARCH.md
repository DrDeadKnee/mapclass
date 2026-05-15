# Phase 2: Build a dataset of pixel-label pairs - Research

**Researched:** 2026-05-15
**Domain:** Geospatial ML data pipelines — IIIF + Allmaps georeferencing, STAC + Sentinel-2 satellite, Pillow-rendered synthetic, multi-scale tiled output
**Confidence:** HIGH on Allmaps + Sentinel-2 + IIIF + rasterio paths (verified live);
MEDIUM on Azgaar template-name extraction (no exposed schema field — needs file inspection);
LOW on multi-scale pyramid storage convention (no canonical pattern in the ecosystem — Claude's discretion).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Allmaps wiring (historical pipeline)

- **D-01: Delete WMS code path entirely.** LUNA never exposes WMS URLs or bounding boxes — the old `_wms_url`, `_parse_bbox`, `_download_wms_geotiff` functions are dead code. Remove them. The historical pipeline flows LUNA → Allmaps → IIIF image → GeoTIFF only.
- **D-02: Fixed 4096 max-edge IIIF fetch.** Request `/full/!4096,4096/0/default.jpg`. Scale the Allmaps GCPs (which are in original IIIF pixel coords) by the same factor before writing the GeoTIFF.
- **D-03: Affine least-squares fit over all Allmaps GCPs.** Solve a 6-parameter affine; write as the GeoTIFF's `transform` via `rasterio`. No TPS pre-warp at this stage. `historical/label.py` works directly off `ds.crs` + `ds.transform` and needs no resampling layer.
- **D-04: Failure handling — drop out-of-scale silently, manifest the rest.** Maps whose GCP-derived bbox diagonal falls outside [100, 2000] km are dropped without entering the manifest. Maps with `not_in_allmaps` (404) or `gcps_insufficient` (<3 GCPs) are emitted to `unregistered_manifest.json` with a `status` field distinguishing reason.
- **D-05: Surface per-reason drop counts at end of `cmd_search`.** Print drop counts for `out_of_scale`, `not_in_allmaps`, `gcps_insufficient`, `download_failed`. A high `out_of_scale` rate (e.g. >50% of search results) is a signal to revisit the scale window — Phase 2 is not "done" if the cleanup is hiding most of the dataset.

#### Sample yield (all three source families)

- **D-06: Pre-tile at build time, with overlap.** Tiles are materialised on disk during `build_*_dataset.py`. The training DataLoader reads by index, not by random crop.
- **D-07: Multi-scale nested pyramids.** Each pyramid contains 1×(896×896) + 4×(448×448) + 16×(224×224) tiles in strict 2×2 spatial nesting. A given 896 region's prediction maps directly to its 4 inner 448 predictions and 16 inner 224 predictions — exploitable by Phase 3's coarse-to-fine model.
- **D-08: Pyramid-to-pyramid stride 448 (50% top-scale overlap).** Within a pyramid, siblings at the same scale do **not** overlap. Across pyramids, the 896 footprints step by 448 across the source map. A 4096-px map yields ~64 pyramids × 21 tiles = ~1344 sample tiles.
- **D-09: Edge policy.** Drop pyramids whose 896 footprint falls more than 50% off the source map.

#### Satellite-source RGB imagery

- **D-10: Satellite is a true third training source (image→label pairs).** The satellite stream is *not* label-only adjacency priors — the model is trained on Sentinel-2 RGB tiles paired with WorldCover labels alongside historical and synthetic samples. Cross-domain generalisation back to illustrated is part of the bet.
- **D-11: Sentinel-2 L2A RGB from `s3://sentinel-cogs`.** Anonymous S3, STAC catalogue, bands B04/B03/B02. Locked-against-GEE-as-primary constraint in `PROJECT.md` survives because GEE is *not* used for fetching; the data source is the same Sentinel-2 source whether fetched via GEE or S3.
- **D-12: ESA WorldCover labels (already locked primary).** No Dynamic World secondary labels in v1. Removing the 10→9-class remap and the secondary-source machinery keeps the satellite path symmetric with the historical path (both label from WorldCover + DEM).
- **D-13: Cloud handling — STAC query with `eo:cloud_cover < 10`, single-date fetch.** For each target region, pick the lowest-cloud scene from the chosen season. Region/season combos with no qualifying scene are dropped; log the drop count for the same per-reason transparency as Allmaps.
- **D-14: Class-diversity stratified coverage selection.** Use a coarse 1 km WorldCover summary to find regions rich in cropland, built-up, and flooded/wetland — the classes synthetic doesn't produce and that PROJECT.md already up-weights. Sample 4096-px windows from those regions. Do not sample uniformly globally (would oversample forest and water).

#### Held-out synthetic test set (EVAL-01)

- **D-15: Hold out whole Azgaar source maps end-to-end.** All renderer styles, all pyramids, all tiles from a held-out map go to test. Zero spatial leakage by construction. The nested-pyramid overlap discussion is moot because adjacent pyramids share a source map and a split.
- **D-16: ~15% by seeded random, stratified across Azgaar continent templates.** Each template contributes proportionally to the test set. Splitter uses a fixed seed for reproducibility.
- **D-17: Sibling directories: `data/synthetic/train/` and `data/synthetic/test/`.** Filesystem-level separation. Training DataLoader points at `train/`; it literally cannot see `test/`. Per-map subdirectory structure is identical inside each.
- **D-18: Snapshot at first build, frozen by ID list.** First `build_dataset.py` run computes the seeded stratified split and writes `data/synthetic/split.json` listing test-set map IDs. Subsequent builds read the manifest; map IDs in the list go to `test/`, everything else (including newly generated maps) goes to `train/`. **The Phase 4 test set never changes after first recording.**

### Claude's Discretion

- IIIF image fetcher implementation (HTTP client choice, retry policy, on-disk caching of intermediate JPEGs) — researcher/planner to decide.
- Storage layout for the multi-scale pyramid (per-pyramid subdirectory vs flat naming convention vs sqlite index) — Claude's call during planning unless the researcher surfaces a specific tradeoff.
- Sentinel-2 STAC client library (`pystac-client` vs raw `requests`) — Claude's call.
- Per-pyramid `sample_weights.json` scoping (per-source map vs per-pyramid vs per-tile) — Claude's call during planning, but the per-source values from `scripts/historical/label.py:40` are locked and must propagate to every tile from that source.

### Deferred Ideas (OUT OF SCOPE)

#### v2 (post-HuggingFace upload)

- **PaliGemma-driven semi-automatic georeferencing.** The full pipeline described in PROJECT.md `DECISION-georeferencing-pipeline` step 3 — generate dense terrain predictions on unregistered maps with the Phase-1 fine-tuned PaliGemma, cross-correlate against a WorldCover + Copernicus DEM reference grid for rigid alignment, then refine with thin-plate-spline warping anchored on coastlines / mountain ranges / major water bodies. **Note:** the locked precedence in `DECISION-georeferencing-pipeline` survives — we just don't *implement* steps 3 and 4 in v1. Add `GEOREF-V2` to `.planning/REQUIREMENTS.md`.
- **Manual MapWarper / QGIS GCP placement.** Same v2 bucket as the PaliGemma path. Provides a per-map manual fallback for the unregistered manifest.

#### Future iterations within Phase 2 (not in plan-01)

- **Per-source loss weights for the synthetic stream.** Currently only the historical source has weights defined (`scripts/historical/label.py:40`). Synthetic and satellite need their own — the planner should propose values and surface them for user approval. Synthetic might be uniform-1.0 (its labels are ground-truth by construction); satellite should down-weight forest/water (over-represented) and up-weight cropland/built-up/wetland (matches PROJECT.md schema constraint).
- **Storage versioning.** Dataset reproducibility — should the build write a `MANIFEST.json` with checksums + source code commit hash? Worth a follow-up plan once the pipelines stabilise.

#### Documentation edits triggered by this discussion

- `.planning/ROADMAP.md` Phase 2 Success Criterion #2 — drop the PaliGemma semi-auto requirement; replace with "unregistered maps are emitted to `unregistered_manifest.json` for v2 processing (manual fallback + PaliGemma semi-auto deferred)".
- `.planning/REQUIREMENTS.md` — add `GEOREF-V2` under v2 Requirements, covering both PaliGemma semi-auto and manual fallback.
- `.planning/STATE.md` Deferred Items — add the two v2 georeferencing items.

These edits should be folded into the Phase 2 plan-01 commit, since they're load-bearing for the plan's scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PHASE-02 | Phase 2 complete — labelled pixel-pair dataset produced from three source families (historical illustrated maps, synthetic illustrated maps, satellite-derived imagery) with `image.png`, `land_cover.png`, `topography.png`, and `sample_weights.json` per map | Standard Stack (rasterio + pystac-client + Pillow) and Architecture Patterns 1-5 cover all three families; the existing `historical/{label,worldcover,dem}.py` modules are reused verbatim by the satellite path; the new tiler module is the cross-family integration point |
| EVAL-01 | Held-out synthetic test set exists, generated by the same Azgaar + Pillow pipeline as training but unseen during training | CONTEXT D-15..D-18 lock the filesystem-level separation (`data/synthetic/{train,test}/`); the seeded stratified-by-Azgaar-template split + frozen `split.json` manifest are documented in "Validation Architecture" with two specific tests (`test_no_train_test_intersection`, `test_split_manifest_frozen`); Open Question #5 surfaces the synthetic dataset target size (~50 source maps) needed to make a 15% stratified hold-out statistically reasonable |
</phase_requirements>

## Summary

Phase 2 produces a pre-tiled, nested multi-scale (224 / 448 / 896) pixel-label dataset from
three independent source families, with a fourth orchestration layer (the tiler) shared by
all three. The high-level shape is fully locked by `02-CONTEXT.md`; this research surfaces
the *plannable details* the planner needs to write tasks against.

**Three load-bearing findings:**

1. **Allmaps publishes a 176 MB bulk open-data dump at `https://files.allmaps.org/maps.geojsonl`.**
   `[VERIFIED: HTTP GET 2026-05-15]` Total maps: **58,666 georeferenced annotations across all
   contributing institutions**. **David Rumsey share: 10,455 georeferenced canvases under 337
   distinct atlas manifests.** Filtering by century requires hitting LUNA for date metadata
   (the dump's `label` field carries titles, not publication dates). The plan should
   *download the dump offline once* and intersect with LUNA year-by-year results, instead of
   probing `annotations.allmaps.org` per-result. This is a free order-of-magnitude reduction
   in API calls and removes Allmaps rate-limit as a phase risk.

2. **Sentinel-2 L2A via STAC + COG is a near-zero-friction third source.**
   `[VERIFIED: live STAC query 2026-05-15]` Collection `sentinel-2-l2a` at
   `https://earth-search.aws.element84.com/v1` exposes a `visual` asset (pre-stacked
   TCI True-Color Image, B04+B03+B02 as 3-band uint8 COG). For our RGB-only requirement this
   eliminates the need to fetch and stack three separate band COGs. Pair with the existing
   `worldcover.py` + `dem.py` fetchers (which already accept any rasterio dataset as the target
   grid) and the labels half of the satellite pipeline is **zero new code**.

3. **The `unregistered_manifest.json` artifact is the v2 hand-off, not a phase-2 deliverable.**
   `02-CONTEXT.md` defers the PaliGemma semi-auto path (DECISION-georeferencing-pipeline
   step 3) to v2 but keeps the locked precedence. Phase 2 plans should emit the manifest
   for any LUNA result that misses Allmaps, with the manifest entries enriched enough that
   v2 can pick up cleanly (LUNA item id + IIIF manifest URL + best-effort thumbnail).

**Primary recommendation:** structure Phase 2 as **four parallel-track plans**: (1) Historical
pipeline refactor — Allmaps wiring, IIIF fetch, GCP affine, `unregistered_manifest` shape;
(2) Synthetic pipeline — Azgaar build + stratified train/test split + held-out frozen manifest;
(3) Satellite pipeline — STAC client + class-diversity coverage scan + RGB fetch reusing
existing label functions; (4) Tiler — nested-pyramid materialisation shared by all three. The
tiler is the integration point — every source-family per-map directory feeds into it identically.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Search Rumsey by year | Python module (`historical/rumsey.py`) | LUNA REST API | LUNA is the only date-aware index for Rumsey; CONTEXT D-01 deletes WMS code so this is the single search path |
| Resolve manifest → georeferencing | Python module (`historical/allmaps.py`) | Allmaps annotations REST API OR offline dump | Two valid resolution strategies — the offline dump approach is plannable optimization |
| Fetch IIIF image | Python module (HTTP client) | David Rumsey IIIF Image API (Cantaloupe/LUNA-bridge server) | LUNA hosts the IIIF surface; standard IIIF 3.0 size/region semantics apply |
| Affine GCP fit | Python (`rasterio.transform.from_gcps`) | rasterio + GDAL | GDAL's `GDALGCPsToGeoTransform` is the canonical least-squares affine — no need to hand-roll |
| Fetch WorldCover labels | Python (`historical/worldcover.py`) | GDAL VSI-CURL → s3://esa-worldcover | EXISTING — reuse verbatim for satellite pipeline |
| Fetch DEM topography | Python (`historical/dem.py`) | GDAL VSI-CURL → s3://copernicus-dem-30m | EXISTING — reuse verbatim |
| STAC search Sentinel-2 | Python (`pystac-client` recommended) | Element84 earth-search-aws v1 | Standard STAC API; pystac-client adds `eo:cloud_cover` query helpers |
| COG byte-range read | Python (`rasterio.open(url).read(window=...)`) | GDAL VSI-CURL | Byte-range reads are GDAL's job; we never download a full S2 tile |
| Class-diversity coverage scan | Python (`historical/worldcover.py` reused at coarse res) | Locally cached coarse-WC summary | Build a one-shot ~1 km global summary file, query it per region |
| Render synthetic | Python Pillow (`scripts/render.py`) | Pillow `ImageDraw.polygon` | EXISTING — reuse |
| Synthetic split | Python (`build_dataset.py`) — new sub-command | seeded RNG + `split.json` manifest | Filesystem-level separation per D-15..D-18 |
| Multi-scale pyramid tiling | Python module (NEW: `scripts/tiling.py`) | Pillow + numpy | No library does this with strict-nested 1+4+16 geometry; custom but small |
| Per-map output schema | Filesystem convention | `image.png` / `land_cover.png` / `topography.png` / `sample_weights.json` | EXISTING — all three families conform |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| rasterio | 1.5.0 | GeoTIFF I/O, GCP→affine, reprojection, COG byte-range reads | EXISTING in `requirements.txt`; canonical Python wrapper over GDAL; `from_gcps` is the GDAL-blessed affine fit `[VERIFIED: pypi 2026-01-05]` |
| pyproj | 3.7.2 | CRS / coordinate-system transforms | EXISTING; standard for any non-trivial reprojection `[VERIFIED: pypi 2025-08-14]` |
| Pillow | 12.2.0 | Raster image I/O for PNG outputs, polygon rasterisation in synthetic renderer | EXISTING; project's canonical 2-D raster lib `[VERIFIED: pypi 2026-04-01]` |
| numpy | 2.4.4 | array math, masking, slope gradients | EXISTING `[VERIFIED: pypi 2026-03-29]` |
| requests | (any 2.x) | HTTP for LUNA, Allmaps, IIIF | EXISTING |
| pystac-client | 0.9.0 | STAC API search with property-query helpers | Standard for Element84 earth-search; supports dict-style `query={'eo:cloud_cover': {'lt': 10}}` `[VERIFIED: pypi 2025-07-18, docs read 2026-05-15]` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pystac | 1.14.3 | parse/manipulate STAC items returned by the search | Implicit dependency of pystac-client; no direct calls needed in our code `[VERIFIED]` |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| pystac-client | raw `requests.post` against the STAC API `/search` endpoint | Saves a dep but loses query-syntax sugar and the iterator-with-paging helper; not worth it `[ASSUMED: standard tradeoff]` |
| `rasterio.transform.from_gcps` (affine LSQ via GDAL) | hand-rolled numpy LSQ over the 6-param affine equations | The hand-roll is ~20 lines and gives more control over residual reporting, but adds a tested-by-us layer that GDAL already has battle-hardened. **Recommendation: use `from_gcps`; only fall back to hand-roll if Phase 2 wants to report per-GCP residuals for QA.** `[CITED: rasterio docs]` |
| Element84 earth-search-aws | Microsoft Planetary Computer STAC | MS PC has the same Sentinel-2 L2A collection but requires SAS-token sign-in for some assets; earth-search is anonymous and matches our `AWS_NO_SIGN_REQUEST` pattern. `[VERIFIED: earth-search anonymous, registry.opendata.aws]` |
| Custom tiling | torchgeo `GridGeoSampler` | torchgeo's samplers run inside the DataLoader (random crops at training time), but CONTEXT D-06 *locks* pre-tile-at-build-time. Use torchgeo as a downstream reader in Phase 3 if useful; don't pull it into the build pipeline. `[CITED: torchgeo docs]` |

**Installation (add to `requirements.txt`):**
```
pystac-client>=0.9
```
(`pystac` arrives transitively. Everything else is already pinned.)

**Version verification:** All versions above queried live from pypi on 2026-05-15.

## Architecture Patterns

### System Architecture Diagram

```
                                  +-------------------------------------+
                                  |  PHASE 2: BUILD DATASETS            |
                                  +-------------------------------------+

  ===== HISTORICAL =====        ===== SYNTHETIC =====           ===== SATELLITE =====

  LUNA search by year             Azgaar GeoJSON dir              Coarse-WC class summary
  scripts/historical/rumsey.py    data/synthetic/raw/             (NEW: scripts/satellite/
        |                                |                         coverage.py — 1 km
        |                                |                         WorldCover summary file
        v                                v                         scanned for cropland/built-up/
  Allmaps lookup ----------> 404? -> emit to                       wetland-rich cells)
  (annotations.allmaps.org           unregistered_                       |
   per-result, OR offline dump         manifest.json                     v
   intersection)                       (v2 hand-off)              STAC search per region
        |                                |                       earth-search.aws.element84.com
        v                                |                       collections=[sentinel-2-l2a]
  IIIF Image fetch                       |                       query={'eo:cloud_cover':{'lt':10}}
  /full/!4096,4096/0/default.jpg         |                              |
        |                                |                              v
        v                                |                       COG byte-range RGB fetch
  Scale GCPs by S=4096/orig_dim          |                       (visual asset: pre-stacked TCI)
        |                                |                              |
        v                                |                              v
  rasterio.transform.from_gcps           |                       Write 4096-px RGB GeoTIFF
        |                                |                              |
        v                                |                              |
  Write GeoTIFF(EPSG:4326, affine)       |                              |
        |                                |                              |
        v                                v                              v
  +---------------------------------------------------------------------------+
  | per-map directory:                                                        |
  |   image.png  /  land_cover.png  /  topography.png  /  sample_weights.json |
  | (label generation reuses existing scripts/historical/worldcover.py +      |
  |  dem.py for ALL THREE families — same code path)                          |
  +---------------------------------------------------------------------------+
                                  |
                                  v
                         +-----------------+
                         |  NEW: tiling.py |
                         |  multi-scale    |
                         |  nested-pyramid |
                         |  decomposer     |
                         +-----------------+
                                  |
              +-------------------+-------------------+
              v                                       v
       data/{source}/train/                    data/synthetic/test/
       <map_id>/                               (synthetic only — D-17)
         <pyramid_id>/                         <map_id>/<pyramid_id>/...
           896.png + 896_*.png (lc, topo)
           448_0.png .. 448_3.png  (4 children)
           224_0.png .. 224_15.png (16 grandchildren)
           ... or whatever Claude picks for storage layout (Discretion D-Storage)
```

### Recommended Project Structure
```
scripts/
├── historical/
│   ├── __init__.py
│   ├── rumsey.py            # EXISTING — refactor per D-01..D-05
│   ├── allmaps.py           # EXISTING (untracked — commit in plan-01)
│   ├── label.py             # EXISTING — frozen HISTORICAL_LC_WEIGHTS
│   ├── worldcover.py        # EXISTING — REUSED by satellite path
│   ├── dem.py               # EXISTING — REUSED by satellite path
│   ├── iiif.py              # NEW — IIIF image fetcher (size-best-fit, retries, scale return)
│   └── georef.py            # NEW — GCP scale + affine fit + GeoTIFF write
├── satellite/               # NEW package
│   ├── __init__.py
│   ├── coverage.py          # NEW — coarse-WC global summary; class-diverse cell picker
│   ├── stac.py              # NEW — pystac-client wrapper, cloud-filtered search
│   └── fetch.py             # NEW — COG byte-range RGB fetch → 4096-px GeoTIFF
├── tiling.py                # NEW — multi-scale nested pyramid (shared by all 3 families)
├── build_historical_dataset.py  # EXISTING — re-wire per D-04, D-05
├── build_synthetic_dataset.py   # EXISTING (build_dataset.py) — add train/test split per D-15..D-18
├── build_satellite_dataset.py   # NEW — coverage-scan / search / build sub-commands
├── render.py                # EXISTING
├── label.py                 # EXISTING (synthetic-side)
├── biome_mapping.py         # EXISTING
├── toon_mapping.py          # EXISTING
└── augment.py               # EXISTING
```
**Note:** the existing `scripts/build_dataset.py` orchestrates synthetic. Either keep that name
or rename to `build_synthetic_dataset.py` for symmetry with the other two. The planner should
decide and the rename is trivial.

### Pattern 1: Allmaps Lookup with Offline Dump Acceleration
**What:** Pre-download `https://files.allmaps.org/maps.geojsonl` once. Build a Python `set`
of Rumsey manifest URLs that have annotations. For each LUNA result, check set membership
first; only hit `annotations.allmaps.org/?url=<manifest>` when the set hit confirms it.
**When to use:** Anytime we know in advance we're querying a single institution's
manifests in bulk. The dump is regenerated nightly per the Allmaps docs.
**Example:**
```python
# Source: https://files.allmaps.org/maps.geojsonl (CC0 open data)
import json, requests, pathlib

DUMP_URL = "https://files.allmaps.org/maps.geojsonl"
DUMP_CACHE = pathlib.Path("data/historical/raw/allmaps_maps.geojsonl")

def build_rumsey_index() -> set[str]:
    """Return the set of Rumsey manifest URLs in Allmaps. Caches locally."""
    if not DUMP_CACHE.exists():
        DUMP_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(DUMP_URL, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            with open(DUMP_CACHE, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
    manifests: set[str] = set()
    with open(DUMP_CACHE) as f:
        for line in f:
            if "davidrumsey" not in line:
                continue
            feat = json.loads(line)
            for canvas in feat.get("properties", {}).get("resource", {}).get("partOf") or []:
                for m in canvas.get("partOf") or []:
                    if "manifest" in m.get("id", ""):
                        manifests.add(m["id"])
    return manifests
```
`[VERIFIED: 2026-05-15 — dump is 176 MB, contains 58,666 annotations, 10,455 of which are Rumsey
under 337 distinct manifest URLs]`

### Pattern 2: STAC Cloud-Filtered Sentinel-2 Search
**What:** Use pystac-client with the dict-style `query` param for `eo:cloud_cover`. Use the
`visual` asset (pre-stacked B04/B03/B02 TCI) instead of fetching three single-band COGs.
**When to use:** Every satellite-source fetch in this phase.
**Example:**
```python
# Source: https://pystac-client.readthedocs.io/en/stable/quickstart.html
# Source: live STAC query against earth-search.aws.element84.com (2026-05-15)
import os
import pystac_client
import rasterio
from rasterio.windows import Window

os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")

ENDPOINT = "https://earth-search.aws.element84.com/v1"

def find_lowest_cloud_scene(bbox, datetime_range, max_cloud=10):
    client = pystac_client.Client.open(ENDPOINT)
    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=datetime_range,
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    items = list(search.items())
    if not items:
        return None
    return min(items, key=lambda it: it.properties.get("eo:cloud_cover", 100))

def fetch_visual_window(item, dst_window_px: int = 4096):
    """Read a centred 4096x4096 window from the TCI/visual asset (no full download)."""
    url = item.assets["visual"].href
    with rasterio.open(url) as ds:
        # centre window
        cx, cy = ds.width // 2, ds.height // 2
        half = dst_window_px // 2
        win = Window(cx - half, cy - half, dst_window_px, dst_window_px)
        rgb = ds.read([1, 2, 3], window=win)  # (3, H, W) uint8
        win_transform = ds.window_transform(win)
        return rgb, win_transform, ds.crs
```
`[VERIFIED: STAC item S2B_54KYD_20260515_0_L2A returned `visual` asset = TCI 3-band COG, 2026-05-15]`

### Pattern 3: IIIF Image Fetch with 4096 Max-Edge
**What:** Build the IIIF URL as `<image_service>/full/!4096,4096/0/default.jpg`. The `!w,h`
form is **size-best-fit**: server scales so neither dimension exceeds 4096 while preserving
aspect ratio. Compute scale = 4096 / max(orig_w, orig_h), apply to all Allmaps GCPs before
the affine fit.
**When to use:** Every Allmaps-resolved historical map.
**Example:**
```python
# Source: https://iiif.io/api/image/3.0/ (size parameter, !w,h form)
def build_iiif_url(image_service_id: str, max_edge: int = 4096) -> str:
    base = image_service_id.rstrip("/")
    # Rumsey image services are IIIF Image API 2.x; the !w,h form works identically.
    # `default.jpg` = "server default quality, JPEG". 0 = no rotation.
    return f"{base}/full/!{max_edge},{max_edge}/0/default.jpg"

def scale_gcps(gcps_orig, orig_width, orig_height, fetched_width, fetched_height):
    """Allmaps gives GCPs in original-image pixel space. After fetching at smaller size,
    scale by the SAME factor in x and y (size-best-fit preserves aspect ratio)."""
    sx = fetched_width / orig_width
    sy = fetched_height / orig_height
    scaled = [
        ((px * sx, py * sy), (lng, lat))
        for ((px, py), (lng, lat)) in gcps_orig
    ]
    return scaled
```
**Note:** the Rumsey-Stanford IIIF image surface is IIIF Image API 2.x (judged from existing
URLs like `https://www.davidrumsey.com/luna/servlet/iiif/RUMSEY~8~1~...`). The `!w,h` and
`/full/.../0/default.jpg` syntax is **identical between IIIF 2.x and 3.x**, so we don't need
to special-case versions. `[CITED: iiif.io/api/image/3.0 and iiif.io/api/image/2.1]`

### Pattern 4: Affine GCP Fit with rasterio
**What:** Use `rasterio.transform.from_gcps(gcps)` to get the 6-param affine. This calls
GDAL's `GDALGCPsToGeoTransform` under the hood — least-squares fit. Returns an `Affine`
object directly usable as the `transform` argument of `rasterio.open(..., 'w', ...)`.
**When to use:** Every Allmaps-resolved historical map after GCPs are scaled.
**Example:**
```python
# Source: https://rasterio.readthedocs.io/en/stable/api/rasterio.transform.html
from rasterio.control import GroundControlPoint
from rasterio.transform import from_gcps
from rasterio.crs import CRS

def gcps_to_affine(scaled_gcps):
    """scaled_gcps: list of ((px, py), (lng, lat)).
    GroundControlPoint takes (row=y, col=x, x=lng, y=lat) -- mind the convention."""
    rasterio_gcps = [
        GroundControlPoint(row=py, col=px, x=lng, y=lat)
        for (px, py), (lng, lat) in scaled_gcps
    ]
    return from_gcps(rasterio_gcps)  # returns Affine

def write_georeferenced_geotiff(rgb_array, affine, out_path):
    """rgb_array: (3, H, W) uint8."""
    h, w = rgb_array.shape[1:]
    with rasterio.open(
        out_path, "w",
        driver="GTiff",
        height=h, width=w,
        count=3, dtype="uint8",
        crs=CRS.from_epsg(4326),
        transform=affine,
    ) as ds:
        ds.write(rgb_array)
```
`[CITED: rasterio docs, GDAL GDALGCPsToGeoTransform]`

### Pattern 5: Class-Diversity Stratified Coverage (Pattern A from research question 3)

**Recommendation: build a one-shot coarse global WorldCover summary and pick class-diverse
cells from it.** This matches the stratified-sampling pattern used by every large-scale
geospatial ML dataset (SatlasPretrain, Globe230k, the 1984-2020 global land cover training
dataset). `[CITED: nature.com/articles/s41597-023-02798-5, satlas-pretrain.allen.ai]`

**Implementation sketch:**
1. **Coarse-WC summary (one-shot, cached)** — at ~1 km resolution, read all WorldCover tiles
   (3°×3° each) and compute, per 1°×1° cell, the count of each canonical class. Persist as a
   single sidecar GeoTIFF or compact JSON. Size estimate: ~36 MB at 1° resolution × 9 classes.
2. **Region picker** — load the summary, rank 1° cells by class diversity (Shannon entropy
   over the 9-class distribution, or a custom score that up-weights the three synthetic-absent
   classes per `02-CONTEXT.md` D-14). Pick top-N cells with a seasonal datetime per cell.
3. **STAC fetch per cell** — for each picked cell, STAC-search the lowest-cloud scene in the
   chosen season, fetch the 4096-px window from the scene's `visual` asset.
4. **Drop-count surfacing** — log per-cell drop reasons (`no_qualifying_scene`,
   `stac_search_failed`, `fetch_failed`) per `02-CONTEXT.md` D-13 / D-05.

**Alternative (Pattern B — biome-first):** would oversample forest/water because most of the
Earth's land is forest/water. Reject. `[ASSUMED]`

### Anti-Patterns to Avoid

- **Hitting `annotations.allmaps.org/?url=<manifest>` per LUNA result.** With the open-data
  dump available (Pattern 1), this is 10,000× more API calls than needed and exposes the
  pipeline to Allmaps rate-limit risk. Use the dump.
- **Building the IIIF URL with `/full/full/`.** IIIF Image API 3.0 deprecates `full` as the
  size parameter (still allowed for region). Use `!w,h` size syntax. `[VERIFIED: iiif.io/api/image/3.0]`
- **Downloading the full Sentinel-2 scene (10980×10980 px, ~600 MB) when we only need a 4096-px
  window.** Use `rasterio.windows.Window` with COG byte-range reads — GDAL VSI-CURL streams
  only the requested bytes. Standard COG practice. `[CITED: GDAL COG driver docs]`
- **Random-crop tiling at training time.** CONTEXT D-06 locks pre-tile-at-build-time. Don't
  pull in torchgeo's `RandomGeoSampler` here.
- **Stratifying the train/test split by anything other than whole-map ID.** CONTEXT D-15
  prohibits adjacent-pyramid spatial leakage by construction; the split is at the *source map*
  level, not the *pyramid* or *tile* level.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Affine least-squares from N GCPs | numpy lstsq over the 6-param equation system | `rasterio.transform.from_gcps` | GDAL's `GDALGCPsToGeoTransform` is the canonical fit; battle-tested; one function call |
| STAC query construction | `requests.post(url, json={'collections': [...], 'query': {...}})` | `pystac_client.Client.open(...).search(...)` | Handles pagination, gives an iterator, dict-style query syntax matches the docs |
| COG byte-range reads over HTTP | manual HTTP `Range` header logic | `rasterio.open(url).read(window=Window(...))` | GDAL VSI-CURL transparently handles range requests for COG — already proven in existing `worldcover.py` |
| WorldCover class remapping | hard-coded `if/elif` chains | EXISTING `WC_REMAP` dict in `historical/worldcover.py` | Already locked, used by historical pipeline, reuse verbatim for satellite |
| Slope from DEM | reproject + manual `np.gradient` | EXISTING `historical/dem.py:fetch_topo` | Already does the LAEA-projection trick correctly |
| 1° / 3° tile origin enumeration | reimplemented `math.floor` loops | EXISTING `_tile_origins` in `worldcover.py` and `_tile_urls` in `dem.py` | Already correct and tested |
| IIIF manifest parsing | custom JSON walk for label/canvas/source | continue with the pattern in `historical/allmaps.py:_parse_annotation` | The annotation shape is locked by W3C / IIIF Image API; just keep it |
| Anonymous S3 to public buckets | boto3 + signed-request hackery | `os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")` + GDAL VSI-CURL | EXISTING pattern, works for `esa-worldcover`, `copernicus-dem-30m`, and `sentinel-cogs` identically |

**Key insight:** the satellite-source pipeline reuses ~90% of the historical pipeline's
infrastructure. The new code is one STAC client + one coverage-scan module + one Sentinel-2
fetch — everything else (label generation, anonymous S3 access, per-map output schema) is
shared with `historical/`.

## Runtime State Inventory

*Phase 2 is greenfield with one rename consideration (the optional renaming of
`build_dataset.py` → `build_synthetic_dataset.py` for symmetry). Sections marked "None — N/A":*

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — `data/` is currently empty save for the `data/toons/` Phase-1 training set. No databases involved. | None |
| Live service config | None — Phase 2 is offline scripts only. | None |
| OS-registered state | None — no scheduled tasks, no daemons. | None |
| Secrets / env vars | Only `AWS_NO_SIGN_REQUEST=YES` (set via `os.environ.setdefault`, no secret value). | None — keep existing pattern |
| Build artifacts | None — Python-only, no installed package. The historical/ subpackage is an in-repo import. | None |

**If `build_dataset.py` is renamed:** the only consumer is the README's section-2 prose; no
import dependencies (only `__main__` execution). Safe rename.

## Common Pitfalls

### Pitfall 1: GCP coordinate convention mismatch
**What goes wrong:** `rasterio.control.GroundControlPoint` takes `(row=y, col=x, x=lng, y=lat)`.
Allmaps gives `((x_px, y_px), (lng, lat))`. Mistakenly passing `row=px_x, col=px_y` will flip
the image.
**Why it happens:** rasterio's `row, col` matches numpy's `(H, W)` axis order — image y-axis
first. Allmaps follows IIIF convention — image x-axis first.
**How to avoid:** wrap GCP construction in a single helper (Pattern 4 example does this
explicitly with named keyword args).
**Warning signs:** the resulting GeoTIFF has the WorldCover overlay at the wrong pixel
location — the labels and the map are off by a non-trivial transform.

### Pitfall 2: IIIF `!w,h` is best-fit, NOT exact
**What goes wrong:** Asking for `!4096,4096` on a 6000×3000 image returns a 4096×2048
image (preserves aspect ratio, neither dimension exceeds 4096). Code that assumes both
dimensions equal 4096 will compute the wrong scale factor.
**Why it happens:** IIIF `!w,h` means "fit within this box," not "exact dimensions."
**How to avoid:** after fetching, read the actual returned image dimensions; compute scale
as `fetched_max_edge / original_max_edge` and apply identically to both GCP axes
(size-best-fit preserves the aspect ratio so scale_x == scale_y).
**Warning signs:** GCPs are off by a consistent multiplicative factor along one axis only.

### Pitfall 3: Allmaps multi-canvas atlases vs single-map manifests
**What goes wrong:** A Rumsey "atlas" manifest contains many canvases (one per plate). Each
canvas may have its own Allmaps georeferencing. The current `historical/allmaps.py:lookup`
returns the **first** annotation only. For atlases with N georeferenced plates, this drops
N-1 plates silently.
**Why it happens:** `items[0]` shortcut in `lookup()` is documented as "the canonical /
latest one" but for atlases it's "plate 1 of N" instead.
**How to avoid:** when more than one item is returned, the planner should decide: either
loop over all of them (each becomes a separate map in `data/historical/raw/`), or reject the
manifest as "out-of-scope" because we want single-plate regional maps not multi-plate atlases.
**Warning signs:** the 10,455 / 337 Rumsey-canvas / Rumsey-manifest ratio in the Allmaps dump
shows atlases ARE the norm — average 31 canvases per manifest. The pipeline will under-yield
by 30× if it stops at items[0].

### Pitfall 4: Sentinel-2 datetime + cloud-cover combinatorics
**What goes wrong:** Hardcoding a single datetime range (e.g. "2024-06-01 to 2024-09-30")
across all picked regions will miss low-cloud scenes for tropical regions whose dry season
is winter and Southern Hemisphere regions whose summer is December.
**Why it happens:** "low cloud cover" is region-and-season dependent. A single global
datetime is the wrong abstraction.
**How to avoid:** per-region picker should select an appropriate season per latitude band
(rough rule: pick Northern-summer May-Sep for northern temperate, Southern-summer Nov-Mar
for southern temperate, dry season per Köppen zone for tropics). Or: just search the past
12 months and let `eo:cloud_cover < 10` do the filtering. Simpler.
**Warning signs:** `drop_count[no_qualifying_scene]` is high (> 10% of picked regions).

### Pitfall 5: Topography label width inheritance from historical map
**What goes wrong:** `historical/dem.py:fetch_topo` projects DEM into the *target map_ds* grid.
For a 4096-px IIIF-fetched map at 100 km diagonal extent, the per-pixel DEM resolution is
~25 m — close to GLO-30's native 30 m, fine. For a 2000-km map at the same 4096 px, per-pixel
resolution is ~500 m — slope from such coarsely-sampled DEM under-reports gradients (because
gradient magnitude scales inversely with pixel size).
**Why it happens:** the `_slope_degrees` function uses `np.gradient(dem, res_m, res_m)` where
`res_m = abs(metric_transform.a)`. Coarse `res_m` flattens slopes.
**How to avoid:** compute slope *at DEM native resolution* (30 m), classify, then downsample
the classified topo into the map_ds grid with `Resampling.mode`. The existing code partially
does this (computes slope in a metric CRS intermediate at target dims), but the intermediate
is at MAP resolution, not DEM resolution.
**Warning signs:** large-extent historical maps (1500-2000 km diagonal) show all-flat
topography even where mountains are visually present.
**Triage:** this is a known limitation of the existing `dem.py`; the planner should surface it
as a follow-up task ("Phase 2 tightening") but **does NOT need to block Phase 2 on it** — the
class-conditional loss weight for topography is `1.0`, so any per-pixel error here costs
training accuracy proportionally. Confirm with user during planning.

### Pitfall 6: Per-map directory output schema drift
**What goes wrong:** The synthetic pipeline currently writes `flat.png`/`illustrated.png`/
`satellite.png` as the image variants — *three different rendering styles per source map*.
But the per-map output schema locked by CONTEXT requires a single `image.png`.
**Why it happens:** `scripts/render.py` and `scripts/build_dataset.py` were written before the
schema was canonicalised.
**How to avoid:** either (a) treat each style as a separate "map" with its own `image.png` —
multiplies synthetic sample count by 3, may bloat dataset; or (b) pick one canonical style per
Azgaar source map (probably `illustrated`) and write that as `image.png` — keeps sample count
modest, loses style augmentation diversity.
**Recommendation:** **option (a)** — treat each (Azgaar source, style) pair as its own map,
share the same `land_cover.png`/`topography.png` across styles, use directory naming
`<azgaar_id>__<style>/`. This way the held-out test split (whole-Azgaar-source-map) still
guarantees zero spatial leakage but training sees 3× the visual diversity. Surface this for
user approval in plan-01.

## Code Examples

(Already given in Patterns 1-5 above. Cross-reference rather than duplicate here.)

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Georeferencer.com WMS download | Allmaps Web-Annotation index | Allmaps reached production maturity, ~2023 | Phase 2 D-01 — delete WMS code path entirely |
| WMS bounding-box-in-metadata | Allmaps GCPs + per-map affine fit | Same as above | Phase 2 D-03 — `rasterio.transform.from_gcps` |
| Fetching three Sentinel-2 bands and stacking | Use Element84's `visual` TCI asset (pre-stacked) | earth-search v1 added TCI 2023 | Reduces our STAC + COG code from 3 fetches+1 stack to 1 fetch |
| Polling `annotations.allmaps.org` per manifest | Offline `maps.geojsonl` dump from `files.allmaps.org` | Allmaps open-data dump introduced ~2024 | Order-of-magnitude API reduction; no Allmaps rate-limit risk |
| Random-crop tiling at training time | Pre-tile at build time, materialise to disk | CONTEXT D-06 / D-07 lock this | Allows the nested-pyramid 1+4+16 geometry to be inspectable on disk |

**Deprecated / outdated:**
- IIIF Image API 3.0 deprecates `full` as the **size** parameter (still valid as the
  **region** parameter). Use `max` or `!w,h`. `[VERIFIED: iiif.io/api/image/3.0]`
- The `sentinel-2-c1-l2a` collection name appeared in some 2024 docs — the current canonical
  name on earth-search v1 is `sentinel-2-l2a`. `[VERIFIED: live STAC query 2026-05-15]`

## Assumptions Log

> Every claim tagged `[ASSUMED]` above is listed here for user confirmation.

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The IIIF 2.x vs 3.0 size syntax `!w,h` works identically on the Rumsey IIIF surface | Pattern 3 | LOW — fallback is to query the `/info.json` for the service profile and adapt; one extra HTTP per map |
| A2 | The "biome-first" alternative to class-diversity sampling oversamples forest/water and is wrong | Pattern 5 | LOW — easy to validate empirically at coverage-scan time |
| A3 | Slope-from-coarse-DEM is a known limitation but not a Phase 2 blocker | Pitfall 5 | MEDIUM — if topography is critically wrong for large-extent historical maps, the topography head will train poorly. **Recommend surfacing this for user approval as part of plan-01.** |
| A4 | Synthetic per-map = (Azgaar source × style) is the right multiplexing | Pitfall 6 | MEDIUM — affects training data volume by 3×. **Needs user approval in plan-01.** |
| A5 | Standard `pystac-client` over raw `requests` to the STAC API is the right call | Stack alternatives | LOW — easy to switch; both work |
| A6 | The Allmaps `lookup()` should be extended to return ALL annotations, not items[0] | Pitfall 3 | HIGH — currently silently drops most plates of atlases. **Plan-01 should fix.** |
| A7 | Synthetic dataset target size of "at least 50 Azgaar source maps before splitting" gives 7-8 maps per template for a stratified 15% hold-out | Synthetic sizing (research Q8) | LOW — easy to add more maps |

**A6 is particularly load-bearing** — without it, the historical pipeline yields ~337 maps
instead of ~10,455. Surface to user as part of plan-01.

## Open Questions

1. **How many Allmaps-georeferenced Rumsey maps fall in the 1500-1700 LUNA-Type=Map subset?**
   - What we know: 10,455 Rumsey canvases in Allmaps; LUNA can filter by year/type.
   - What's unclear: the *intersection* — Allmaps doesn't surface LUNA's date metadata in
     the dump; LUNA doesn't surface Allmaps coverage.
   - Recommendation: this is *empirically discoverable during plan execution* — run
     `rumsey.search_maps(date_start=1500, date_end=1700)` (already implemented), intersect
     with the offline dump's manifest URL set, count. **The drop-count instrumentation in
     D-05 will surface this naturally on the first run.** Don't pre-compute; let the
     pipeline tell us.

2. **Should the Sentinel-2 satellite source use the `visual` TCI asset or fetch B04/B03/B02
   separately?**
   - What we know: `visual` is a pre-stacked TCI COG, uint8, 3-band, native 10 m resolution,
     standard Sentinel-2 colour-balanced product.
   - What's unclear: whether the colour-balancing applied by the TCI processor introduces
     a domain-gap from raw reflectance that affects training. For our use case (illustrated
     map segmentation), TCI is arguably *closer to what an illustrator would paint*, so the
     gap might actually help.
   - Recommendation: use `visual` for v1. Reserve "switch to band-stacked B04/B03/B02 if
     training surfaces a saturation/clipping issue" as a follow-up.

3. **Should the multi-scale pyramid be stored as per-pyramid subdirectories, flat naming, or
   a sqlite index?**
   - What we know: CONTEXT D-Storage marks this as Claude's discretion.
   - What's unclear: Phase 3's DataLoader access pattern (sequential? random with replacement?
     parent-aware mini-batching for coarse-to-fine?).
   - Recommendation: **per-pyramid subdirectories with a single `pyramid.json` per subdir
     listing the 21 file paths and parent→child indices.** Filesystem-native, easy to debug,
     easy to delete a broken pyramid, easy to ship via a tar/rsync. Add a single top-level
     `index.parquet` if the DataLoader needs O(1) lookup; that's a Phase 3 concern.

4. **Should `historical/allmaps.py:lookup()` return ALL items or just items[0]?**
   - See Pitfall 3 / Assumption A6. **Surface to user in plan-01.**

5. **What's the synthetic dataset target size?**
   - SPEC and CONTEXT are silent. Phase-1 fine-tune used ~4760 augmented samples from 120
     base tiles. For Phase 2 segmentation training (a much harder task), reasonable target
     is at least 50-100 Azgaar source maps × 3 styles × ~1344 pyramid tiles = 200-400k tiles.
     Plenty for segmentation. **Recommendation: target N=50 Azgaar source maps for v1, allow
     planning to revise up if the user wants more diversity.** This gives 7-8 maps per
     Azgaar template (with ~12 templates known) — enough for a stratified 15% hold-out.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3 | All scripts | ✓ | 3.13+ inferred from `bool` ↔ `is_integer()` use | — |
| `rasterio` | All georeferenced paths | ✓ | 1.5.0 (latest on pypi) | none required |
| `pyproj` | DEM LAEA reprojection | ✓ | 3.7.2 | none required |
| `Pillow` | Image I/O | ✓ | 12.2.0 | none required |
| `numpy` | Array math | ✓ | 2.4.4 | none required |
| `requests` | LUNA, Allmaps, IIIF | ✓ | listed in requirements.txt | none required |
| `pystac-client` | Sentinel-2 STAC search | **MISSING** (not in `requirements.txt`) | needs `pystac-client>=0.9` | raw `requests` against the STAC API works but loses query helpers |
| GDAL (via rasterio) | All raster ops | ✓ (bundled with rasterio wheel) | bundled | — |
| Network access to `*.s3.amazonaws.com` | Anonymous public-bucket reads | ✓ | — | — |
| Network access to `*.davidrumsey.com` | LUNA search, IIIF image fetch | ✓ (assumed) | — | — |
| Network access to `annotations.allmaps.org`, `files.allmaps.org` | Allmaps lookup, offline dump | ✓ (verified 2026-05-15) | — | — |
| Network access to `earth-search.aws.element84.com` | Sentinel-2 STAC | ✓ (assumed standard internet) | — | — |
| Disk for `data/historical/raw/` GeoTIFFs | Historical pipeline | varies by `--max-maps` | ~5-15 MB per IIIF image at 4096 px | reduce `max-maps` |
| Disk for `data/synthetic/{train,test}/` | Synthetic pipeline | varies | small (a synthetic Azgaar map is < 5 MB) | — |
| Disk for `data/satellite/` | Satellite pipeline | varies | ~50 MB per 4096-px window × N regions | reduce regions |
| Disk for Allmaps dump cache | Historical pipeline acceleration | 176 MB one-shot | — | none required |

**Missing dependencies with no fallback:** none — `pystac-client` has a working raw-requests fallback.

**Missing dependencies with fallback:**
- `pystac-client` — install recommended; raw `requests` works.

## Validation Architecture

(Per `.planning/config.json`: nyquist_validation key is absent, so include this section.)

### Test Framework
| Property | Value |
|----------|-------|
| Framework | **None currently** — no `tests/` directory, no test files, no pytest config detected `[VERIFIED: ls scripts/]` |
| Config file | none — see Wave 0 |
| Quick run command | TBD (Wave 0 creates the framework) |
| Full suite command | TBD |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PHASE-02 (historical) | LUNA search returns N ≥ 1 result for 1500-1700 query | integration (online) | `pytest tests/test_rumsey.py::test_search_returns_results -x` | ❌ Wave 0 |
| PHASE-02 (historical) | Allmaps lookup returns GCPs for a known-georeferenced Rumsey manifest | integration (online) | `pytest tests/test_allmaps.py::test_known_rumsey_manifest_has_gcps -x` | ❌ Wave 0 |
| PHASE-02 (historical) | IIIF fetch at `!4096,4096` returns ≤ 4096 in both dimensions | integration (online) | `pytest tests/test_iiif.py::test_max_edge_fetch -x` | ❌ Wave 0 |
| PHASE-02 (historical) | GCP affine fit + GeoTIFF write round-trips: read CRS=4326 and `transform` survives | unit | `pytest tests/test_georef.py::test_affine_roundtrip -x` | ❌ Wave 0 |
| PHASE-02 (historical) | `make_labels()` produces all 4 output files for a tiny synthetic 256-px GeoTIFF | integration (offline, fixture-based) | `pytest tests/test_label.py::test_make_labels_writes_all_outputs -x` | ❌ Wave 0 |
| PHASE-02 (synthetic) | `render_map` produces matching `land_cover.png` and `topography.png` dimensions | unit | `pytest tests/test_render.py::test_output_dimensions_match -x` | ❌ Wave 0 |
| PHASE-02 (synthetic) | The seeded train/test split is deterministic across re-invocations | unit | `pytest tests/test_split.py::test_seeded_split_deterministic -x` | ❌ Wave 0 |
| PHASE-02 (satellite) | STAC search returns at least one item for a bbox with known coverage and cloud<10 | integration (online) | `pytest tests/test_stac.py::test_known_bbox_returns_results -x` | ❌ Wave 0 |
| PHASE-02 (satellite) | COG byte-range window read returns shape (3, 4096, 4096) for `visual` asset | integration (online) | `pytest tests/test_satellite.py::test_window_fetch_shape -x` | ❌ Wave 0 |
| PHASE-02 (tiler) | 1×896 + 4×448 + 16×224 nested pyramid: child tiles' aggregated extent equals parent's | unit | `pytest tests/test_tiling.py::test_nested_alignment -x` | ❌ Wave 0 |
| PHASE-02 (tiler) | Edge-policy: pyramid >50% off the source map is dropped | unit | `pytest tests/test_tiling.py::test_edge_drop -x` | ❌ Wave 0 |
| EVAL-01 | After build, `data/synthetic/test/` contains map IDs from `split.json` and `train/` contains the rest, **with NO intersection** | integration | `pytest tests/test_split.py::test_no_train_test_intersection -x` | ❌ Wave 0 |
| EVAL-01 | `split.json` is frozen — re-running build does not change the test-set ID list | integration | `pytest tests/test_split.py::test_split_manifest_frozen -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/ -x --ignore=tests/integration` (unit tests only, < 30s)
- **Per wave merge:** `pytest tests/` (all unit + offline integration, < 5min)
- **Phase gate:** full suite (incl. online integration tests) green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/conftest.py` — shared fixtures (sample LUNA item, sample Allmaps annotation,
      tiny 256-px GeoTIFF, sample Azgaar GeoJSON, mocked-STAC item)
- [ ] `tests/test_rumsey.py` — LUNA search + filter behaviour
- [ ] `tests/test_allmaps.py` — Allmaps lookup + offline-dump intersection
- [ ] `tests/test_iiif.py` — IIIF size syntax + scale-factor logic
- [ ] `tests/test_georef.py` — affine fit + GeoTIFF I/O
- [ ] `tests/test_label.py` — `make_labels` end-to-end on a fixture
- [ ] `tests/test_render.py` — synthetic render dimensions
- [ ] `tests/test_split.py` — train/test deterministic split + frozen manifest
- [ ] `tests/test_stac.py` — STAC client wrapper
- [ ] `tests/test_satellite.py` — satellite fetcher + RGB GeoTIFF write
- [ ] `tests/test_tiling.py` — nested-pyramid tiler
- [ ] `tests/integration/` — subdir for online tests (network required)
- [ ] Framework install: add `pytest>=8` and `pytest-mock>=3` to requirements.txt;
      `pytest tests/` runnable

## Security Domain

`.planning/config.json` does not include a `security_enforcement` key, but the project posture
is "single researcher, offline ML data pipeline, no production service, no user-supplied
data, no auth surface." The standard ASVS categories largely do not apply.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No auth surface — anonymous S3 + public LUNA/Allmaps/IIIF |
| V3 Session Management | no | No sessions |
| V4 Access Control | no | No multi-user model |
| V5 Input Validation | **partial** | The LUNA / Allmaps response parsers in `rumsey.py` and `allmaps.py` already validate field presence and types. Keep that pattern — never trust upstream JSON without checking field types before float casts. |
| V6 Cryptography | no | No secrets, no PII, no signing |

### Known Threat Patterns for this Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malformed JSON from third-party APIs causing `KeyError`/`TypeError` crashes mid-build | Denial of Service (against ourselves) | Wrap parser code in try/except per item; continue with next item; log to manifest. EXISTING pattern in `allmaps.py:_parse_annotation`. |
| Path traversal via item IDs ending up in filenames | Tampering | Item IDs are tilde-delimited LUNA IDs like `RUMSEY~8~1~123~456`. No `/` or `..`. **Sanitise anyway** when writing output filenames — replace any non-`[\w-]` chars with `_`. |
| Zip-slip-style risks in tile output paths | Tampering | We never read user-supplied archives. N/A. |
| Resource exhaustion from a huge IIIF response (e.g. server ignores `!4096,4096`) | Denial of Service | After fetching, before parsing as JPEG, check `Content-Length`. If > 200 MB (sanity ceiling for any sensible 4096-px JPEG), abort. |

## Sources

### Primary (HIGH confidence — verified live or from official docs)

- Allmaps open-data dump: https://files.allmaps.org/maps.geojsonl — VERIFIED 2026-05-15:
  176 MB, 58,666 maps total, 10,455 from David Rumsey under 337 manifest URLs.
- Allmaps annotations API: https://annotations.allmaps.org/ — VERIFIED 2026-05-15: returns
  `{name: "annotations", version: "2.5.0-beta.0"}` at root; `/maps` returns a paginated
  AnnotationPage (capped at ~750 items); the offline dump is the right bulk path.
- Element84 earth-search v1: https://earth-search.aws.element84.com/v1 — VERIFIED 2026-05-15
  via live STAC item retrieval; `sentinel-2-l2a` collection confirmed; `visual` asset is a
  pre-stacked TCI 3-band COG; cloud-cover filter syntax `query={"eo:cloud_cover": {"lt": 10}}`.
- IIIF Image API 3.0 spec: https://iiif.io/api/image/3.0/ — VERIFIED: `!w,h` size syntax
  is best-fit; `default.jpg` = "server default quality, JPEG"; rotation `0` = no rotation.
- IIIF Image API 2.1 spec: https://iiif.io/api/image/2.1/ — VERIFIED: same `!w,h` semantics
  (relevant because Rumsey IIIF surface is 2.x).
- Rasterio docs (transform module): https://rasterio.readthedocs.io/en/stable/api/rasterio.transform.html
  — VERIFIED: `from_gcps(gcps) -> Affine`, uses GDAL's `GDALGCPsToGeoTransform` (LSQ).
- Rasterio docs (georeferencing): https://rasterio.readthedocs.io/en/stable/topics/georeferencing.html
- pystac-client docs (quickstart): https://pystac-client.readthedocs.io/en/stable/quickstart.html
  — VERIFIED: dict-style query syntax, endpoint, collection names.
- Sentinel-2 L2A COG registry: https://registry.opendata.aws/sentinel-2-l2a-cogs/ — VERIFIED:
  bucket `s3://sentinel-cogs` in `us-west-2`, anonymous access.
- PyPI version queries (2026-05-15): `pystac-client` 0.9.0, `pystac` 1.14.3, `rasterio` 1.5.0,
  `pyproj` 3.7.2, `numpy` 2.4.4, `Pillow` 12.2.0.

### Secondary (MEDIUM confidence)

- Allmaps Rumsey-scripts repo: https://github.com/allmaps/rumsey-scripts — VERIFIED via
  WebFetch: no bulk-download tools; the open-data dump is the right approach.
- Observable notebook (David Rumsey / Allmaps): https://observablehq.com/@allmaps/rumsey
  — CITED: 10,126 Rumsey maps as of Feb 2025, consistent with our 2026-05-15 dump count of
  10,455 (a 3.3% increase over ~15 months, plausible cadence).
- Azgaar Fantasy Map Generator GIS export wiki: https://github.com/Azgaar/Fantasy-Map-Generator/wiki/GIS-data-export
  — Documents what GeoJSON exports include, but does NOT specify whether the heightmap template
  name is exposed in the export.
- Azgaar template list: ~12 named templates exist (High Island, Low Island, Continents,
  Archipelago, Atoll, Mediterranean, Peninsula, Volcano, Pangea, Shattered, Two Continents,
  Fractured). Source: https://azgaar.wordpress.com/2017/10/05/templates/ —
  CITED but list is not authoritative; Azgaar's `heightmap-templates.js` would be the source
  of truth. **Plan-01 should inspect an actual exported GeoJSON file to confirm whether the
  template name is in the feature properties or in a top-level `metadata` block.**
- SatlasPretrain dataset paper: https://arxiv.org/abs/2211.15660 — CITED: pattern of
  geographic stratification + WorldCover-derived labels at scale.
- Global land cover training dataset (Stanimirova et al. 2023): https://www.nature.com/articles/s41597-023-02798-5
  — CITED: stratified-sampling-by-ecoregion pattern; reference for our class-diversity approach.

### Tertiary (LOW confidence — flagged for validation)

- Multi-scale nested pyramid storage convention: no canonical pattern found in the
  geospatial-ML ecosystem (torchgeo's pre-tiled datasets like LandCoverAI use a flat directory
  with paired image+mask files). **Recommendation: per-pyramid subdirectory with `pyramid.json`
  manifest, plus an optional top-level `index.parquet` if Phase 3 needs O(1) lookup.**
- Synthetic dataset target size of 50-100 source maps: heuristic based on Phase-1's 4760 samples
  scaling; no authoritative source.
- Slope-from-coarse-DEM limitation severity: known mathematical issue but no measurement of
  practical impact on segmentation accuracy. Surface as user decision.

## Metadata

**Confidence breakdown:**
- Allmaps wiring: HIGH — verified the open-data dump exists, has Rumsey content (10,455
  canvases under 337 manifests), and that the existing `historical/allmaps.py` lookup logic
  matches the IIIF Annotation shape.
- Sentinel-2 + STAC: HIGH — verified the endpoint, collection, asset names, and cloud-cover
  query syntax against a live earth-search response.
- IIIF size syntax: HIGH — verified against the IIIF 3.0 spec and the 2.x spec; same semantics.
- Rasterio GCP affine: HIGH — verified function signature and behaviour from docs.
- Multi-scale nested pyramid tiler: LOW — no canonical library; custom implementation.
- Azgaar template-name extraction: LOW — not documented in the GeoJSON export schema. Plan-01
  needs to inspect an actual export file to confirm where the template name lives.
- Class-stratified satellite coverage scan: MEDIUM — pattern matches what SatlasPretrain and
  the 1984-2020 GLC training dataset do; implementation details are our own.

**Research date:** 2026-05-15
**Valid until:** 2026-06-15 for Sentinel-2 / STAC + Allmaps (the dump regenerates daily; the
endpoint contract is stable). 2026-08-15 for rasterio + pystac-client + IIIF (slow-moving
infrastructure). Re-validate the Allmaps dump URL pattern before plan-01 commits work.
