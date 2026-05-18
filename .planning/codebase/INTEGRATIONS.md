# External Integrations

**Analysis Date:** 2026-05-08

The MapClass repo currently only integrates with **read-only public data sources** for dataset construction. There are no outbound webhooks, no databases, no auth providers, no error trackers, no CI pipelines, and no proprietary APIs. Every external interaction below is a one-directional fetch.

## APIs & External Services

### David Rumsey Map Collection — LUNA Search API

**Purpose:** Catalogue search for 16th–17th century regional maps. Returns map metadata (id, title, author, date, geography fields, image URLs, and where available a Georeferencer WMS link).

**Consumed by:** `scripts/historical/rumsey.py` — function `search_maps()`. Orchestrated from `scripts/build_historical_dataset.py` `cmd_search()`.

**Endpoint:**
- `GET https://www.davidrumsey.com/luna/servlet/as/search`
- Query parameters used (`scripts/historical/rumsey.py:222-227`):
  - `q` — free-text query, populated with a single year (`str(year)`) because LUNA's Lucene-style date-range syntax returns zero results.
  - `output=json` — request JSON response.
  - `bs=50` — batch size (`_BATCH_SIZE`).
  - `os=<offset>` — page offset, in increments of `_BATCH_SIZE`.

**Auth:** None. Public endpoint. No API key, no token, no cookies.

**Identification headers:**
- `User-Agent: mapclass-dataset-builder/0.1` (set in `_LUNA_HEADERS`, `scripts/historical/rumsey.py:41`).

**Rate limits / backoff:**
- The API has no published rate limit. The client implements defensive retry logic in `_get_json()` (`scripts/historical/rumsey.py:161-185`):
  - Catches HTTP **429** (Too Many Requests) and **503** (Service Unavailable) and waits `_BACKOFF_BASE ** attempt` seconds (base 2.0, max 4 attempts → up to 8 s).
  - Catches `requests.RequestException` with the same exponential backoff.
  - 30 second timeout per request.
- Search calls run year-by-year over the full window (1500–1700 → 201 years × `pages_per_year`); a default `max_results=500` run with `pages_per_year=1` issues at least 201 search requests.

**Response format:** JSON. The expected shape (per `scripts/historical/rumsey.py` docstring and `_field()` parser):
```json
{
  "results": [
    {
      "id": "...",
      "urlSize0": "...", "urlSize1": "...", ..., "urlSize4": "...",
      "iiifManifest": "...",
      "links": [{"href": "...", "rel": "..."}, ...],
      "fieldValues": [
        {"Author": ["..."]},
        {"Date": ["1570"]},
        {"Type": ["Atlas Map"]},
        {"Country": ["..."]},
        {"Scale 1": ["..."]},
        ...
      ]
    },
    ...
  ]
}
```
Sparse fields (Country, Scale 1, Region, World Area) appear in 30–50% of items per the file's docstring. **Bounding boxes are not provided by LUNA** — this is the central difficulty of using the source.

**Error handling specifics:**
- Non-JSON responses raise a `ValueError` that includes status code, content type, and the first 300 bytes of the body, to make rate-limit HTML pages identifiable (`scripts/historical/rumsey.py:172-180`).

---

### David Rumsey Georeferencer — WMS GetMap

**Purpose:** Fetch a georeferenced GeoTIFF for a Rumsey map that has already been registered in the third-party Georeferencer service.

**Consumed by:** `scripts/historical/rumsey.py` — function `_download_wms_geotiff()`, called from `download_georeferenced()`. Orchestrated from `scripts/build_historical_dataset.py` `cmd_search()`.

**Endpoint pattern:**
- WMS URLs are not constructed by the client; they are extracted from the LUNA item's metadata. `_wms_url()` (`scripts/historical/rumsey.py:93-106`) looks for them in two places:
  1. A `wms_url` or `WMS URL` field in `fieldValues`.
  2. Any `links[].href` containing `georeferencer.com` or `maps.georeferencer`.
- The host is therefore `https://maps.georeferencer.com/...` per the docstring (`scripts/historical/rumsey.py:8-9`).

**Request:**
- `GET <wms_url>` with WMS 1.3.0 GetMap parameters (`scripts/historical/rumsey.py:289-301`):
  - `SERVICE=WMS`, `VERSION=1.3.0`, `REQUEST=GetMap`, `LAYERS=0`, `STYLES=`
  - `CRS=EPSG:4326`
  - `BBOX={south},{west},{north},{east}` (WMS 1.3.0 axis order)
  - `WIDTH=4096`, `HEIGHT=4096` (default)
  - `FORMAT=image/geotiff`

**Auth:** None. Public access.

**Headers:** Same `mapclass-dataset-builder/0.1` User-Agent.

**Timeout:** 120 seconds, streamed in 64 KiB chunks.

**Rate limits:** Not documented; no explicit retry logic for the WMS download (a single failure simply returns `False`).

**Output format:** GeoTIFF, written to `<output_dir>/<item_id>.tif`. The handler validates that `Content-Type` contains `tiff` or `image` before writing (`scripts/historical/rumsey.py:305-307`).

**Filtering applied before download** (`download_georeferenced()`, `scripts/historical/rumsey.py:319-356`):
- The map must have a Georeferencer WMS URL (otherwise it's added to the unregistered manifest instead).
- The bounding box diagonal must be in `[100, 2000]` km (regional scale; excludes city plans and continental sheets).
- If the file already exists on disk, the download is skipped (idempotent).

**Failure mode for unregistered maps:**
Maps without a WMS URL are written to `data/historical/raw/unregistered_manifest.json` by `emit_manifest()` (`scripts/historical/rumsey.py:363-428`) for **manual GCP placement in QGIS**. Each manifest entry includes thumbnail URL, image URL (`urlSize4`), IIIF manifest URL, and a `rumsey_page` link of the form `https://www.davidrumsey.com/luna/servlet/detail/<item_id>`.

---

### ESA WorldCover 10m — AWS Open Data S3 (HTTPS, anonymous)

**Purpose:** 10-metre global land cover raster (year 2021, version v200). Source of land-cover ground truth for satellite-imagery and historical-map training samples.

**Consumed by:** `scripts/historical/worldcover.py` — function `fetch_worldcover()`. Called from `scripts/historical/label.py` `make_labels()`.

**Endpoint pattern:**
- Bucket: `s3://esa-worldcover` (also referenced as `s3://esa-worldcover` in `README.md`).
- HTTPS access via `https://esa-worldcover.s3.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{NS}{LAT:02d}{EW}{LON:03d}_Map.tif`.
- Tile naming uses SW-corner integer coordinates on a **3°×3°** grid. `_tile_origins()` (`scripts/historical/worldcover.py:67-80`) yields all tile origins overlapping the requested WGS84 bbox. `_tile_name()` (`scripts/historical/worldcover.py:60-64`) formats e.g. `N48E000`, `S03W009`.

**Auth:** None — anonymous S3 access. The client sets `AWS_NO_SIGN_REQUEST=YES` via `os.environ.setdefault` (`scripts/historical/worldcover.py:36`) to tell GDAL not to sign requests.

**Access mechanism:** GDAL VSI-CURL streaming through `rasterio.open(url)` — the file is **not** downloaded as a whole; only the byte ranges needed for reprojection are fetched. This is essential for keeping `data/historical/` small: only the cropped, reprojected output PNG is persisted.

**Rate limits:** S3 has effectively no per-request rate limit for this bucket; standard AWS S3 backoff conventions apply. The client does not implement explicit retry — failures are caught per-tile and logged as warnings (`scripts/historical/worldcover.py:131-132`), allowing partial coverage.

**Class taxonomy & remapping** (`scripts/historical/worldcover.py:42-54`):

| WorldCover value | Meaning | Canonical class index | Canonical name |
|---|---|---|---|
| 10  | Tree cover | 1 | trees |
| 20  | Shrubland | 2 | shrubland |
| 30  | Grassland | 3 | grassland |
| 40  | Cropland | 4 | cropland |
| 50  | Built-up | 5 | built_up |
| 60  | Bare/sparse vegetation | 6 | bare_sparse |
| 70  | Snow and ice | 8 | snow_ice |
| 80  | Permanent water bodies | 0 | water |
| 90  | Herbaceous wetland | 7 | flooded_wetland |
| 95  | Mangroves | 1 | trees (folded) |
| 100 | Moss and lichen | 6 | bare_sparse (folded) |

NODATA sentinel: `255`.

**Output format:** `np.ndarray` shape `(map_ds.height, map_ds.width)`, dtype `uint8`, in the caller's target CRS / pixel grid (reprojected via `rasterio.warp.reproject` with `Resampling.nearest`).

---

### Copernicus DEM GLO-30 — AWS Open Data S3 (HTTPS, anonymous)

**Purpose:** Global 30-metre digital elevation model. Used to derive the topography label (flat / hilly / mountainous) by computing slope and bucketing.

**Consumed by:** `scripts/historical/dem.py` — function `fetch_topo()`. Called from `scripts/historical/label.py` `make_labels()`.

**Endpoint pattern:**
- Bucket: `s3://copernicus-dem-30m` (referenced as `s3://copernicus-dem-30m` in `README.md`).
- HTTPS: `https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{NS}{LAT:02d}_00_{EW}{LON:03d}_00_DEM/Copernicus_DSM_COG_10_{NS}{LAT:02d}_00_{EW}{LON:03d}_00_DEM.tif` (`scripts/historical/dem.py:35-39`).
- Tile naming uses SW-corner integer coordinates on a **1°×1°** grid (`_tile_urls()`, `scripts/historical/dem.py:47-60`).

**Auth:** None — anonymous S3 access via `AWS_NO_SIGN_REQUEST=YES` (`scripts/historical/dem.py:33`).

**Access mechanism:** Same GDAL VSI-CURL streaming pattern as WorldCover.

**Behaviour over open ocean:** Tiles for fully oceanic 1° cells **do not exist** in the bucket; HTTP 404 is the normal response and is silently swallowed (`scripts/historical/dem.py:124-128` and the explicit comment in the docstring at lines 11-12). If every tile is 404 the function returns a uniform `WATER_TOPO` raster (`scripts/historical/dem.py:132-136`).

**Rate limits:** Same as WorldCover — no explicit retry, S3 default behaviour.

**Topography classification** (`scripts/historical/dem.py:41-43, 86-92`):

| Slope | Class index | Class name |
|---|---|---|
| < 2°    | 0 | flat |
| 2°–15°  | 1 | hilly |
| ≥ 15°   | 2 | mountainous |
| —       | 255 | water / nodata (`WATER_TOPO`) |

Slope is computed in a **Lambert Azimuthal Equal Area** projection centred on the bbox (`_laea_crs()`, `scripts/historical/dem.py:63-72`) so that `np.gradient` returns metric values directly. The DEM is reprojected to LAEA, slope computed, then the classified topo raster is reprojected back to the caller's target grid.

**Output format:** `np.ndarray` shape `(map_ds.height, map_ds.width)`, dtype `uint8`, in the caller's target CRS / pixel grid. The land-cover water mask (passed in as `water_mask`) overrides slope-derived classes after reprojection.

---

### Azgaar's Fantasy Map Generator — manual upstream

**Purpose:** Generates the synthetic illustrated fantasy maps that anchor the synthetic-source training data. The generator produces a procedural world with biome/height fields per cell.

**Consumed by:** `scripts/build_dataset.py`, indirectly via `scripts/render.py` and `scripts/label.py`. Both `render.py` and `label.py` parse the GeoJSON and rely on properties:
- `properties.height` (integer, 0–100; <20 is water).
- `properties.biome` (integer, 0–21; mapped via `AZGAAR_BIOMES` in `scripts/biome_mapping.py`).

**Integration mode:** **No API.** The Fantasy Map Generator is a separate browser-hosted JavaScript application; the developer runs it manually and exports GeoJSON via the menu path *Map → Save → GeoJSON with all layers* or *Tools → Export → GeoJSON cells* (`scripts/build_dataset.py:14-15`). The exported `.geojson` files are dropped into `data/raw/` and consumed by `scripts/build_dataset.py`.

**Auth / rate limits:** N/A — purely offline.

**Coordinate system:** Azgaar's native pixel space (no CRS). `scripts/label.py:13` and `scripts/render.py:8` both compute the output image extent from the bounding box of all features.

**Schema dependencies (load-bearing assumptions made by `scripts/render.py` and `scripts/label.py`):**
- `data["features"]` is a list of GeoJSON Features.
- Each feature has `geometry.type` of `Polygon` or `MultiPolygon` and properties `height` and `biome`.
- Biome IDs 0–21 cover the full Azgaar palette per `AZGAAR_BIOMES` (`scripts/biome_mapping.py:10-33`); any biome ID outside this range will raise a `KeyError` at `BIOME_TO_LANDCOVER_NAME[biome]` (`scripts/biome_mapping.py:90-94`).

---

## Data Storage

**Databases:** None.

**File Storage:** Local filesystem only. All produced artefacts land under `data/` (gitignored):
- `data/raw/` — Azgaar GeoJSON inputs.
- `data/renders/` — synthetic map renders.
- `data/labels/` — pixel-level label rasters.
- `data/historical/raw/georeferenced/` — downloaded historical GeoTIFFs.
- `data/historical/raw/unregistered_manifest.json` — list of Rumsey items needing manual GCP placement in QGIS.
- `data/historical/dataset/<item_id>/` — per-map outputs: `image.png`, `land_cover.png`, `topography.png`, `sample_weights.json`.
- `data/toons/` — hex tile assets.

**Caching:** None at the application layer. GDAL's VSI-CURL has its own internal caching for S3 byte-range reads but it is not configured here.

**Cloud storage:** Read-only, anonymous. The repo never writes to any S3 bucket.

## Authentication & Identity

**Auth Provider:** None — no user accounts, no auth flows, no SSO. Every external endpoint accessed is fully public.

## Monitoring & Observability

**Error Tracking:** None (no Sentry, no Rollbar, no Bugsnag).

**Logs:** Plain `print()` statements only. Examples:
- Search progress (`scripts/historical/rumsey.py:217, 259, 261`).
- Download status (`scripts/historical/rumsey.py:354, 314`).
- Per-tile WorldCover / DEM warnings (`scripts/historical/worldcover.py:131-132`, `scripts/historical/dem.py` warns implicitly through the swallowed exception path).
- Build-step phases in `scripts/historical/label.py:108, 117-118, 122, 126, 129, 133, 135, 144`.

**Metrics:** None.

## CI/CD & Deployment

**Hosting:** Local development + a separate VM-provisioning repo for cloud training (per `README.md` Cloud / training workflow). This repo carries no Docker, no Terraform, no Helm.

**CI Pipeline:** None. There is no `.github/workflows/`, no `.gitlab-ci.yml`, no `Jenkinsfile`, no `circle.yml`.

## Environment Configuration

**Environment variables read by code:**
- `AWS_NO_SIGN_REQUEST` — set defensively to `"YES"` via `os.environ.setdefault` in `scripts/historical/worldcover.py:36` and `scripts/historical/dem.py:33`. If already set, the existing value is preserved.

**Environment variables expected externally:**
- None. No `.env` file is loaded; no `python-dotenv` dependency.

**Secrets location:** Not applicable — no secrets are used. Every external service is anonymous.

**Default file paths (CLI defaults, `scripts/build_historical_dataset.py:134, 145, 152`):**
- `--raw-dir` defaults to `data/historical/raw`.
- `--out-dir` defaults to `data/historical/dataset`.
- `--max-maps` defaults to 500.
- `--workers` defaults to 1 (multi-threaded downloads via `concurrent.futures.ThreadPoolExecutor` when >1).

## Webhooks & Callbacks

**Incoming:** None.

**Outgoing:** None.

## Planned but Unimplemented Integrations

These are referenced in `README.md` and `mockup.md` but have no code yet. Future planning will need to wire them up:

- **Hugging Face Hub** — implicit via `transformers` for downloading PaliGemma-3B / SigLIP / CLIP / OpenCLIP weights for the zero-shot benchmark and fine-tuned backbone. No HF token currently required for public models, but rate limits on `huggingface.co` may matter once benchmarking starts.
- **OpenStreetMap** — `README.md` lists "OSM tile renders" as the obvious candidate for the road-map source, but no client exists yet.
- **Georeferencer.com (account API, not public WMS)** — currently only the public WMS GetMap endpoint is used. The auto-georeferencing pipeline planned in `README.md` § GeoViLM construction would supplement this.
- **Annotation tools** — `README.md` mentions Label Studio / CVAT as candidates if a hand-annotated real-domain test set is later added; not integrated.

## Licensing implications (cross-reference)

Per `executive_TODO.md`, the integrations above feed into a proposed open-source dataset, but redistribution rights are unverified for:
- Copernicus DEM GLO-30 — derivative product (slope-derived topography) license terms TBD.
- David Rumsey digitisations — public-domain originals, but Rumsey's institutional terms on the digitisations and georeferenced GeoTIFFs are uncertain.
- Azgaar Fantasy Map Generator outputs — generator code is MIT, but generated-output redistribution terms TBD.

This file documents *how* the code touches each service; the *legal* dimension is owned by `executive_TODO.md`.

---

*Integration audit: 2026-05-08*
