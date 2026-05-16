# Phase 2: Build a dataset of pixel-label pairs - Pattern Map

**Mapped:** 2026-05-15
**Files analyzed:** 13 (5 new, 5 modified/refactored, 3 reused-verbatim)
**Analogs found:** 13 / 13 (every new file has a strong in-repo analog — this phase is a parallel-track extension of an existing pipeline, not greenfield)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `scripts/historical/rumsey.py` | service (search/download) | request-response + file-I/O | itself (in-place refactor per D-01..D-05) | self / exact |
| `scripts/historical/allmaps.py` | service (lookup) | request-response | itself (extend `lookup()` per A6/Pitfall 3) | self / exact |
| `scripts/historical/iiif.py` | service (fetcher) | request-response → file-I/O | `historical/worldcover.py` (HTTP-fetch + retry) + `rumsey._get_json` (backoff) | role-match |
| `scripts/historical/georef.py` | utility (transform) | transform | `historical/dem.py` (rasterio reproject/transform math) | role-match |
| `scripts/build_historical_dataset.py` | orchestrator (CLI) | batch + event-driven | itself (re-wire `cmd_search` per D-04/D-05) | self / exact |
| `scripts/satellite/coverage.py` | service (region picker) | transform + file-I/O | `historical/worldcover.py` (tile enumeration + remap) | role-match |
| `scripts/satellite/stac.py` | service (search) | request-response | `historical/allmaps.py` (lookup + retry/backoff) | role-match |
| `scripts/satellite/fetch.py` | service (fetcher) | streaming (COG byte-range) → file-I/O | `historical/worldcover.py` (`rasterio.open(url)` + window read) | exact (same GDAL VSI-CURL pattern) |
| `scripts/build_satellite_dataset.py` | orchestrator (CLI) | batch | `scripts/build_historical_dataset.py` (sub-commands + threadpool) | exact |
| `scripts/tiling.py` | utility (decomposer) | transform + file-I/O | `scripts/label.py` (per-map dir, Pillow I/O) — closest, but novel pyramid geometry | role-match (geometry is novel) |
| `scripts/build_dataset.py` (synthetic) | orchestrator (CLI) | batch + file-I/O | itself (add split per D-15..D-18) + `build_historical_dataset.py` (sub-command shape) | self / exact |
| `scripts/historical/label.py` | service (label gen) | transform → file-I/O | REUSED VERBATIM by satellite path (do not modify; weights locked line 40) | reuse |
| `scripts/historical/{worldcover,dem}.py` | service (label gen) | streaming → transform | REUSED VERBATIM by satellite path | reuse |

## Pattern Assignments

### `scripts/historical/rumsey.py` (service, in-place refactor — D-01..D-05)

**Analog:** itself. The WMS path is being deleted; the Allmaps path replaces it. Keep every pattern below; swap only the download body.

**KEEP — module structure / constants** (lines 41–68): the `_LUNA_*`, `_MIN/MAX_DIAG_KM`, `_DATE_*`, `_BATCH_SIZE`, `_MAX_RETRIES`, `_BACKOFF_BASE`, `_MAP_TYPES` block is the template. Satellite constants should mirror this layout.

**KEEP — exponential backoff HTTP helper** (`_get_json`, lines 173–197):
```python
def _get_json(url: str, params: dict) -> dict:
    """GET with exponential-backoff retry on 429/503."""
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, headers=_LUNA_HEADERS, timeout=30)
            if resp.status_code in (429, 503):
                wait = _BACKOFF_BASE ** attempt
                time.sleep(wait)
                continue
            resp.raise_for_status()
            ...
        except requests.RequestException as exc:
            if attempt == _MAX_RETRIES - 1:
                raise
            time.sleep(_BACKOFF_BASE ** attempt)
```
This is the canonical retry shape — `iiif.py` and `satellite/stac.py` must copy it.

**KEEP verbatim** — `search_maps` (204–288), `_richness_score` (140–170), `_field` (89–102), `_haversine_km`/`_bbox_diagonal_km` (75–86). CONTEXT explicitly says preserve `_richness_score` as-is.

**DELETE per D-01** — `_wms_url` (105–118), `_parse_bbox` (121–137), `_download_wms_geotiff` (295–328). Dead code. Note `build_historical_dataset.py:59` also calls `rumsey._wms_url` — that call site must be removed in the same diff.

**REWRITE — `download_georeferenced`** (331–368): replace WMS body with the Allmaps → IIIF → GeoTIFF flow. The drop-classification skeleton already lives in the current docstring (lines 336–342). New return contract should distinguish drop reasons for D-04/D-05 (e.g. return a `(path | None, status)` tuple or raise typed sentinels). The existing early-return-with-print idiom (lines 351–364) is the style to follow:
```python
    if not (_MIN_DIAG_KM <= diag <= _MAX_DIAG_KM):
        print(f"  {item_id}: diagonal {diag:.0f} km out of range, skipping")
        return None
```
Use `manifest_url = map_meta.get("iiifManifest")` (confirmed available per the LUNA field inventory in the module docstring, lines 31–38) → `allmaps.lookup(manifest_url)` → scale GCPs → `georef` affine + GeoTIFF write.

**REWRITE — `emit_manifest`** (375–440): keep the rich per-item dict (412–435) almost verbatim — it already emits `iiif_manifest`, `image_url`, `thumbnail_url`, `rumsey_page` which are exactly the v2 hand-off fields RESEARCH §finding-3 requires. Change only: drop the `_wms_url(item)` skip on line 385, and replace the hardcoded `"status": "needs_gcps"` (line 434) with the D-04 per-reason status (`not_in_allmaps` | `gcps_insufficient`).

---

### `scripts/historical/allmaps.py` (service, extend `lookup()` — Pitfall 3 / A6)

**Analog:** itself. Module is untracked in git — commit it in plan-01 (CONTEXT specifics).

**KEEP — retry/backoff + status-code switch** (`lookup`, lines 67–103). Same shape as `rumsey._get_json` but for the Allmaps endpoint; 404/500 → `None`, 429/503 → backoff-retry, other → raise `AllmapsLookupError`. This is the second instance of the canonical pattern.

**KEEP — typed error class** (lines 41–42): `class AllmapsLookupError(RuntimeError)`. Satellite STAC client should define an analogous `StacLookupError`.

**KEEP — annotation parser** (`_parse_annotation`, 106–153): the defensive feature loop with per-feature `try/except (KeyError, TypeError, IndexError, ValueError): continue` (117–124) and the `len(gcps) < 3 → None` guard (125–126) are locked by the W3C contract. RESEARCH "Don't Hand-Roll" says keep this verbatim.

**CHANGE per A6/Pitfall 3** — line 100 `ann = items[0]` silently drops N-1 plates of multi-canvas atlases. Plan-01 decision (surface to user): either return `list[dict]` (all annotations, each → separate map) or filter atlases. Whichever — the `_parse_annotation` helper is reused per-item unchanged; only the `lookup()` aggregation changes.

**REUSE — geo helpers** `haversine_km` / `bbox_diagonal_km` (156–169): note these duplicate `rumsey._haversine_km`. Planner may consolidate, but not required.

---

### `scripts/historical/iiif.py` (NEW — service, fetcher)

**Analog:** `historical/worldcover.py` for the rasterio/HTTP boundary; `rumsey._get_json` for retry.

**Imports + module constants pattern** — copy `worldcover.py:27–39` shape (stdlib, then numpy/rasterio, then module constants and base URLs):
```python
import io, time
from pathlib import Path
import requests
_IIIF_HEADERS = {"User-Agent": "mapclass-dataset-builder/0.1"}   # match rumsey._LUNA_HEADERS
_MAX_RETRIES = 4
_BACKOFF_BASE = 2.0
```

**Retry pattern** — copy `rumsey._get_json` (173–197) structure but for a binary image body (`resp.content` / `resp.iter_content`) instead of `.json()`. Reuse `_download_wms_geotiff`'s streaming-write idiom (`rumsey.py:321–325`, the only part of the WMS code worth salvaging before deletion):
```python
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
```

**Core pattern** — RESEARCH Pattern 3 (`build_iiif_url` / `scale_gcps`, lines 354–369). The function must return the *actual* fetched (w, h) so the caller computes `scale = fetched_max_edge / orig_max_edge` (Pitfall 2 — `!w,h` is best-fit not exact). On-disk JPEG caching is Claude's discretion (CONTEXT) — mirror the `worldcover.py` URL-open simplicity unless caching is wanted.

---

### `scripts/historical/georef.py` (NEW — utility, transform)

**Analog:** `historical/dem.py` — the in-repo authority on rasterio CRS/transform/reproject mechanics.

**Imports pattern** — copy `dem.py:23–33`:
```python
import numpy as np
import rasterio
from rasterio.crs import CRS
```
plus `from rasterio.control import GroundControlPoint` and `from rasterio.transform import from_gcps` (new, per RESEARCH Pattern 4).

**Core pattern** — RESEARCH Pattern 4 (`gcps_to_affine` / `write_georeferenced_geotiff`, lines 388–409). The `GroundControlPoint(row=py, col=px, x=lng, y=lat)` keyword convention is Pitfall 1 — wrap GCP construction in one helper with named kwargs (RESEARCH says so explicitly).

**GeoTIFF write idiom** — model the `rasterio.open(out_path, "w", driver="GTiff", ...)` context-manager on the existing read-side context managers in `label.py:110` and `worldcover.py:119`. Write `crs=CRS.from_epsg(4326)`, `transform=affine` so `label.make_labels` consumes `ds.crs`/`ds.transform` unchanged (D-03).

---

### `scripts/build_historical_dataset.py` (orchestrator — re-wire `cmd_search`, D-04/D-05)

**Analog:** itself. The threadpool `cmd_build` (82–117) is the reusable template for both other build scripts — DO NOT change it.

**KEEP — threadpool worker pattern** (`_process_one` 73–79, `cmd_build` 82–117):
```python
def _process_one(tif, out_dir):
    try:
        hist_label.make_labels(tif, sample_dir)
        return tif, None
    except Exception as exc:
        return tif, exc
...
    if workers == 1:
        for tif in tifs: ...
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_one, tif, out_dir): tif for tif in tifs}
            for fut in as_completed(futures): ...
```
`build_satellite_dataset.py` and the synthetic build copy this verbatim.

**KEEP — argparse sub-command scaffold** (`main` 124–164): `sub = parser.add_subparsers(dest="command", required=True)` + the `add_common(p)` shared-args helper. This is the template for `build_satellite_dataset.py` (`coverage-scan`/`search`/`build`) and the synthetic split sub-command.

**REWRITE — `cmd_search`** (45–66): remove the `rumsey._wms_url(item)` call (line 59 — dead after D-01). Replace the `downloaded` / `unregistered` two-bucket counting with the D-05 per-reason drop counter:
```python
drops = {"out_of_scale": 0, "not_in_allmaps": 0,
         "gcps_insufficient": 0, "download_failed": 0}
# ... increment per rumsey.download_georeferenced return status ...
print("\nSearch summary:")
for reason, n in drops.items():
    print(f"  {reason}: {n}")
```
The existing final-summary print block (63–66) is the formatting style to extend. RESEARCH "Dropped maps must be loud" — surface `out_of_scale` rate as a Phase-2-not-done signal.

---

### `scripts/satellite/stac.py` (NEW — service, search)

**Analog:** `historical/allmaps.py` (lookup + retry + typed error).

**Module structure** — copy `allmaps.py:35–42`: endpoint constant, headers, `_MAX_RETRIES`, `_BACKOFF_BASE`, a `class StacLookupError(RuntimeError)`.

**Anonymous-S3 env pattern** — copy `worldcover.py:36` exactly: `os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")` at module top. Same pattern works for `sentinel-cogs` (RESEARCH "Don't Hand-Roll").

**Core pattern** — RESEARCH Pattern 2 (`find_lowest_cloud_scene`, lines 318–329): `pystac_client.Client.open(ENDPOINT).search(collections=["sentinel-2-l2a"], bbox=..., datetime=..., query={"eo:cloud_cover": {"lt": max_cloud}})`, then `min(items, key=lambda it: it.properties.get("eo:cloud_cover", 100))`. `None` on no qualifying scene → drop-counted by caller (D-13, same as Allmaps 404).

**New dep:** add `pystac-client>=0.9` to `requirements.txt` (RESEARCH Standard Stack; `pystac` arrives transitively).

---

### `scripts/satellite/fetch.py` (NEW — service, COG byte-range fetcher)

**Analog:** `historical/worldcover.py` — exact match for the `rasterio.open(url)` + windowed read + write-to-target-grid pattern.

**Core pattern** — RESEARCH Pattern 2 (`fetch_visual_window`, lines 331–341): read the `visual` (TCI) asset with `rasterio.windows.Window`, never download the full scene (anti-pattern in RESEARCH). Mirror `worldcover.py:118–130`'s `with rasterio.open(url) as tile_ds:` + `try/except → print Warning` resilience idiom. Write the 4096-px RGB to a GeoTIFF via the **same `georef.write_georeferenced_geotiff` helper** built above (uses `win_transform` + `ds.crs` from the STAC item instead of GCP-derived affine).

**Then reuse verbatim:** the resulting GeoTIFF goes straight into `historical.label.make_labels` — zero new label code (RESEARCH key insight; CONTEXT code_context). Satellite needs its own `*_LC_WEIGHTS` dict (deferred — planner proposes values, see "Shared Patterns").

---

### `scripts/build_satellite_dataset.py` (NEW — orchestrator)

**Analog:** `scripts/build_historical_dataset.py` — parallel structure, sub-commands `coverage-scan` / `search` / `build`.

Copy the entire CLI scaffold (`main` 124–164, `add_common` 133–135), the `_process_one`/threadpool worker loop (73–117), and the per-reason drop-counter from the re-wired `cmd_search` (D-13 mandates the same transparency as D-05). `build` sub-command's worker calls `satellite.fetch.fetch_visual_window` → `georef` write → `historical.label.make_labels`.

---

### `scripts/satellite/coverage.py` (NEW — service, class-diversity region picker)

**Analog:** `historical/worldcover.py` — tile-origin enumeration + WC class remap.

**Reuse** — `worldcover._tile_origins` (67–81) and `WC_REMAP` (42–54) for the coarse global summary. RESEARCH Pattern 5: build a one-shot ~1 km / 1°-cell class-count summary (cached sidecar), rank cells by class-diversity (Shannon entropy up-weighting cropland/built_up/flooded_wetland per D-14), emit picked cells + per-cell season. Drop-reason logging (`no_qualifying_scene`, `stac_search_failed`, `fetch_failed`) follows the D-05 counter shape.

---

### `scripts/tiling.py` (NEW — utility, nested-pyramid decomposer; SHARED by all 3 families)

**Analog:** `scripts/label.py` (synthetic) for the Pillow `Image` open/crop/save + per-map-directory I/O idiom. The 1+4+16 strict-2×2-nested geometry is novel — no library or in-repo analog (RESEARCH: "custom but small").

**I/O idiom to copy** — `label.py:64–88` (open PNGs, derive width/height, iterate, `.save(output_dir / "...")`). Input is a completed per-map dir (`image.png`/`land_cover.png`/`topography.png`/`sample_weights.json`); output is the pyramid tree.

**Geometry (locked, D-07/D-08/D-09):** per pyramid 1×896 + 4×448 + 16×224 in strict 2×2 nesting; pyramid stride 448 (50% top-scale overlap); drop pyramids whose 896 footprint is >50% off the source map. Storage layout is Claude's discretion (RESEARCH Open Q3 recommends per-pyramid subdir + `pyramid.json` listing 21 paths + parent→child indices).

**Integration:** called by all three `build_*` scripts after their per-map dir completes. `sample_weights.json` propagates per-tile (per-source values locked; scoping is Claude's discretion per CONTEXT).

---

### `scripts/build_dataset.py` (synthetic orchestrator — add split, D-15..D-18)

**Analog:** itself + `build_historical_dataset.py` sub-command scaffold.

**KEEP** — the `render_map` → `make_labels` per-geojson loop (36–42).

**CHANGE per Pitfall 6** — currently writes `flat.png`/`illustrated.png`/`satellite.png` (3 styles, no `image.png`). RESEARCH recommends option (a): treat each `(azgaar_id, style)` as its own map dir `<azgaar_id>__<style>/` with its own `image.png`, sharing `land_cover.png`/`topography.png`. Surface for user approval (A4). The synthetic `label.make_labels` (`scripts/label.py`) writes only `land_cover.png`/`topography.png` — no `sample_weights.json` yet; add a synthetic weights dict (see Shared Patterns).

**ADD per D-15..D-18** — a seeded stratified-by-Azgaar-template splitter writing `data/synthetic/split.json` on first run (frozen thereafter); route IDs into sibling `data/synthetic/train/` vs `test/` dirs (filesystem-level separation — train DataLoader literally cannot see `test/`). Template-name extraction is MEDIUM-confidence (RESEARCH: needs Azgaar GeoJSON field inspection). Upgrade the CLI to the `build_historical_dataset.py` argparse sub-command scaffold; optional rename to `build_synthetic_dataset.py` for symmetry (safe — only README prose consumes the name).

---

## Shared Patterns

### Exponential Backoff HTTP Retry
**Source:** `scripts/historical/rumsey.py:173–197` (`_get_json`) and `scripts/historical/allmaps.py:67–103` (`lookup`)
**Apply to:** `historical/iiif.py`, `satellite/stac.py`
```python
for attempt in range(_MAX_RETRIES):
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=30)
        if resp.status_code in (429, 503):
            time.sleep(_BACKOFF_BASE ** attempt); continue
        resp.raise_for_status()
        ...
    except requests.RequestException as exc:
        if attempt == _MAX_RETRIES - 1: raise
        time.sleep(_BACKOFF_BASE ** attempt)
```
Constants `_MAX_RETRIES` / `_BACKOFF_BASE = 2.0` and `_HEADERS = {"User-Agent": "mapclass-dataset-builder/0.1"}` are the established values — reuse them.

### Anonymous Public-S3 Access
**Source:** `scripts/historical/worldcover.py:36` (also `dem.py:33`)
**Apply to:** `satellite/stac.py`, `satellite/fetch.py`, `satellite/coverage.py`
```python
os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
```
Module-top, before any `rasterio.open(url)`. No boto3. Works identically for `esa-worldcover`, `copernicus-dem-30m`, `sentinel-cogs`.

### Typed Lookup Error
**Source:** `scripts/historical/allmaps.py:41–42`
**Apply to:** `satellite/stac.py`
```python
class AllmapsLookupError(RuntimeError):
    """Raised for unexpected (non-404/500) failures during Allmaps lookup."""
```
404/missing → return `None` (caller drop-counts); unexpected → raise typed error.

### Per-Reason Drop Counter (D-05 / D-13)
**Source:** new in re-wired `build_historical_dataset.py:cmd_search`; pattern extends the existing summary-print block at `build_historical_dataset.py:63–66`
**Apply to:** `build_historical_dataset.py`, `build_satellite_dataset.py`, `satellite/coverage.py`
A `dict[str, int]` of drop reasons, incremented in the worker loop, printed at end of the search/build command. "A clean pipeline that quietly throws away most of the data is not job done" (CONTEXT specifics).

### Per-Map Output Schema (mandatory, all 3 families)
**Source:** `scripts/historical/label.py:95–149` (canonical writer)
**Apply to:** historical (have it), satellite (reuse `label.make_labels` verbatim), synthetic (`scripts/label.py` must add `image.png` + `sample_weights.json`)
Every per-map dir: `image.png` + `land_cover.png` + `topography.png` + `sample_weights.json`. The `sample_weights.json` dict shape is locked:
```python
{"land_cover_weights": {<class>: float, ...},
 "topography_weight": float,
 "source": "<historical|synthetic|satellite>",
 "map_file": "<name>"}
```
`HISTORICAL_LC_WEIGHTS` (`label.py:40–51`) is **frozen — do not modify**. Synthetic + satellite need their own dicts of identical shape (deferred; planner proposes values for user approval — synthetic likely uniform-1.0, satellite down-weight forest/water, up-weight cropland/built_up/flooded_wetland per PROJECT.md).

### Threadpool Worker Loop
**Source:** `scripts/build_historical_dataset.py:73–117`
**Apply to:** `build_satellite_dataset.py`, synthetic build
`_process_one(item) → (item, exc|None)` returning exceptions instead of raising, dispatched via `ThreadPoolExecutor` + `as_completed`, with a `workers == 1` synchronous fast path.

### argparse Sub-Command Scaffold
**Source:** `scripts/build_historical_dataset.py:124–164`
**Apply to:** `build_satellite_dataset.py` (`coverage-scan`/`search`/`build`), synthetic build (add split sub-command)
`add_subparsers(dest="command", required=True)` + a local `add_common(p)` for shared `--raw-dir`/`--out-dir` args; module docstring doubles as `epilog` via `RawDescriptionHelpFormatter`.

## No Analog Found

None. Every new file maps to an in-repo analog. The only genuinely novel logic is the **nested-pyramid tile geometry** inside `scripts/tiling.py` (1+4+16 strict-2×2 nesting, stride 448, >50%-off-edge drop) — the file *I/O scaffolding* copies `scripts/label.py`, but the geometry has no analog and the planner should treat D-07/D-08/D-09 + RESEARCH Open-Q3 as the spec. Confidence on tile storage layout is LOW (Claude's discretion); confidence on Azgaar template-name extraction for the synthetic split is MEDIUM (needs GeoJSON field inspection during planning).

## Metadata

**Analog search scope:** `scripts/`, `scripts/historical/` (full read of all 13 Python modules, 2321 LOC total)
**Files scanned:** 13 source files + `requirements.txt` + CONTEXT.md + RESEARCH.md (668 lines, read in 2 non-overlapping passes)
**Tests:** none exist in repo — EVAL-01's `test_no_train_test_intersection` / `test_split_manifest_frozen` (RESEARCH Validation Architecture) will be the first tests; no test analog to copy, follow RESEARCH's described structure.
**Pattern extraction date:** 2026-05-15
</content>
