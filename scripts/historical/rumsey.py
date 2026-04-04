"""
Search the David Rumsey Map Collection for 16th–17th century regional maps
and download georeferenced map images as GeoTIFFs.

Uses two endpoints:
  - LUNA search API: https://www.davidrumsey.com/luna/servlet/as/search
    Returns JSON with map metadata.
  - WMS GetMap (Georeferencer): https://maps.georeferencer.com/georeferences/...
    For maps already registered in the Georeferencer service.

Maps that are not yet georeferenced are written to an unregistered_manifest.json
for manual GCP placement in QGIS.

Search strategy:
  The LUNA API does not support date-range queries (Lucene syntax returns zero
  results). We query year by year and collect a pool capped at pages_per_year
  pages per year, then rank the pool by metadata richness and return the top
  max_results items. This ensures even temporal and geographic distribution
  and biases toward well-documented maps.

Metadata available per item (always present unless noted):
  Top-level: id, urlSize0–urlSize4 (JPEG image URLs), iiifManifest
  fieldValues: Author, Date, Short Title, Full Title, Type, Obj Height cm,
    Obj Width cm, Publisher, Publisher Location, Pub Title, Pub Type
  Sparse (30–50% of items): Scale 1, Country, City, World Area, Region,
    Reference, Engraver or Printer

Not available: bounding boxes / coordinates, orientation/bearing.
"""

import json
import math
import time
from pathlib import Path

import requests

# David Rumsey LUNA API
_LUNA_SEARCH = "https://www.davidrumsey.com/luna/servlet/as/search"
_LUNA_HEADERS = {"User-Agent": "mapclass-dataset-builder/0.1"}

# Extent filters (kilometres, diagonal of bounding box)
_MIN_DIAG_KM = 100
_MAX_DIAG_KM = 2000

# Date range for 16th–17th century maps
_DATE_START = 1500
_DATE_END = 1700

_BATCH_SIZE = 50
_MAX_RETRIES = 4
_BACKOFF_BASE = 2.0  # seconds

# David Rumsey Type field values that represent actual maps (not text pages, covers, etc.)
_MAP_TYPES = {"Atlas Map", "Separate Map", "Map", "Historical Map", "Wall Map", "Globe Map"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _haversine_km(lon1, lat1, lon2, lat2) -> float:
    """Return great-circle distance in km between two WGS84 points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _bbox_diagonal_km(west, south, east, north) -> float:
    return _haversine_km(west, south, east, north)


def _field(item: dict, name: str, default=None):
    """
    Extract a named field from a LUNA item's fieldValues list.

    The LUNA API returns fieldValues as a list of single-key dicts:
      [{"Date": ["1570"]}, {"Short Title": ["..."]}, ...]
    """
    for fv in item.get("fieldValues", []):
        for key, vals in fv.items():
            if key.lower() == name.lower():
                if isinstance(vals, list):
                    return vals[0] if vals else default
                return vals if vals is not None else default
    return default


def _wms_url(item: dict) -> str | None:
    """Return the Georeferencer WMS URL for this item, or None."""
    # Check explicit WMS field
    wms = _field(item, "wms_url") or _field(item, "WMS URL")
    if wms and "georeferencer" in wms.lower():
        return wms

    # Check all links in the item for georeferencer.com patterns
    for link in item.get("links", []):
        href = link.get("href", "")
        if "georeferencer.com" in href or "maps.georeferencer" in href:
            return href

    return None


def _parse_bbox(item: dict) -> tuple[float, float, float, float] | None:
    """
    Try to extract a (west, south, east, north) bounding box from LUNA
    item metadata. Returns None if not determinable.
    """
    for field_name in ("bbox", "Bounding Box", "Coverage", "coverage"):
        raw = _field(item, field_name)
        if raw:
            # Common formats: "W E S N" or "minX,minY,maxX,maxY"
            parts = raw.replace(",", " ").split()
            if len(parts) == 4:
                try:
                    vals = [float(p) for p in parts]
                    return vals[0], vals[2], vals[1], vals[3]  # W,S,E,N
                except ValueError:
                    pass
    return None


def _richness_score(item: dict) -> int:
    """
    Score an item by how many useful metadata fields are populated.
    Higher = better documented = preferred for training.

    Geographic context fields are weighted highest because they tell us
    what terrain the map depicts, which is the most operationally useful
    information for our pipeline.
    """
    score = 0
    # Geographic context — most valuable for terrain expectations
    if _field(item, "Country"):    score += 4
    if _field(item, "World Area"): score += 4
    if _field(item, "City"):       score += 2  # city maps are too small-scale
    if _field(item, "Region"):     score += 2
    # Scale denominator — useful for ruling out city/world maps
    if _field(item, "Scale 1"):    score += 3
    # Attribution quality indicators
    if _field(item, "Reference"):          score += 2
    if _field(item, "Engraver or Printer"): score += 1
    # Physical dimensions (filter out very small maps)
    h = _field(item, "Obj Height cm")
    w = _field(item, "Obj Width cm")
    if h and w:
        try:
            area = float(h) * float(w)
            if area >= 500:   score += 1   # ≥ ~A3 size
            if area >= 1200:  score += 1   # ≥ ~A2 size
        except ValueError:
            pass
    return score


def _get_json(url: str, params: dict) -> dict:
    """GET with exponential-backoff retry on 429/503."""
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, headers=_LUNA_HEADERS, timeout=30)
            if resp.status_code in (429, 503):
                wait = _BACKOFF_BASE ** attempt
                print(f"  Rate limited ({resp.status_code}), waiting {wait:.0f}s…")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            try:
                return resp.json()
            except ValueError:
                snippet = resp.text[:300] if resp.text else "(empty body)"
                ct = resp.headers.get("Content-Type", "?")
                raise ValueError(
                    f"Non-JSON response [status={resp.status_code}, "
                    f"Content-Type={ct}]: {snippet!r}"
                )
        except requests.RequestException as exc:
            if attempt == _MAX_RETRIES - 1:
                raise
            time.sleep(_BACKOFF_BASE ** attempt)
    return {}


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_maps(
    date_start: int = _DATE_START,
    date_end: int = _DATE_END,
    max_results: int = 500,
    pages_per_year: int = 1,
) -> list[dict]:
    """
    Search David Rumsey LUNA API for maps dated between date_start and date_end.

    Two-phase strategy to avoid ordering/distribution bias:
      Phase 1 — Pool collection: query each year independently, take up to
        pages_per_year pages per year. Each year contributes equally, so the
        pool spans the full date range regardless of max_results.
      Phase 2 — Richness ranking: score every pooled item by how many useful
        metadata fields are populated, then return the top max_results.

    This biases toward well-documented maps (known location, scale, author)
    while preserving temporal and geographic diversity across the period.

    pages_per_year=1 → ~50 raw results/year → ~10–15 maps/year after filtering
    → pool of ~2000–3000 maps for 1500–1700, then rank to get max_results.
    Increase pages_per_year for a larger pool at the cost of more API calls.
    """
    pool: dict[str, dict] = {}  # id → item

    print(f"Collecting pool: {date_start}–{date_end}, {pages_per_year} page(s)/year")

    for year in range(date_start, date_end + 1):
        for page in range(pages_per_year):
            try:
                data = _get_json(_LUNA_SEARCH, {
                    "q": str(year),
                    "output": "json",
                    "bs": _BATCH_SIZE,
                    "os": page * _BATCH_SIZE,
                })
            except Exception as exc:
                print(f"  Warning: year {year} page {page} failed: {exc}")
                break

            batch = data.get("results", [])
            if not isinstance(batch, list) or not batch:
                break

            for item in batch:
                item_id = item.get("id", "")
                if not item_id or item_id in pool:
                    continue

                item_type = _field(item, "Type") or ""
                if item_type and item_type not in _MAP_TYPES:
                    continue

                item_date = _field(item, "Date")
                if item_date:
                    try:
                        if not (date_start <= int(str(item_date).strip()[:4]) <= date_end):
                            continue
                    except ValueError:
                        pass

                pool[item_id] = item

            if len(batch) < _BATCH_SIZE:
                break  # no more pages for this year

        if year % 25 == 0:
            print(f"  Through {year}: {len(pool)} maps in pool")

    print(f"Pool complete: {len(pool)} maps. Ranking by metadata richness…")
    ranked = sorted(pool.values(), key=_richness_score, reverse=True)

    # Print richness distribution for the selected set
    selected = ranked[:max_results]
    scores = [_richness_score(item) for item in selected]
    if scores:
        print(f"  Score range in top {len(selected)}: {min(scores)}–{max(scores)}")
        n_with_geo = sum(
            1 for item in selected
            if _field(item, "Country") or _field(item, "World Area") or _field(item, "City")
        )
        print(f"  Maps with geographic context field: {n_with_geo}/{len(selected)}")

    print(f"Search complete: returning {len(selected)} maps")
    return selected


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _download_wms_geotiff(wms_url: str, bbox: tuple, output_path: Path, size: int = 4096) -> bool:
    """
    Download a GeoTIFF from a Georeferencer WMS GetMap request.

    Returns True on success, False on failure.
    """
    west, south, east, north = bbox
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": "0",
        "STYLES": "",
        "CRS": "EPSG:4326",
        "BBOX": f"{south},{west},{north},{east}",
        "WIDTH": size,
        "HEIGHT": size,
        "FORMAT": "image/geotiff",
    }
    try:
        resp = requests.get(wms_url, params=params, headers=_LUNA_HEADERS, timeout=120, stream=True)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "tiff" not in content_type.lower() and "image" not in content_type.lower():
            print(f"  Unexpected content type: {content_type}")
            return False
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        return True
    except Exception as exc:
        print(f"  WMS download failed: {exc}")
        return False


def download_georeferenced(
    map_meta: dict,
    output_dir: Path,
) -> Path | None:
    """
    Download a georeferenced historical map as a GeoTIFF.

    Returns the output path on success, or None if:
      - the map has no WMS georeferencing URL
      - the bounding box diagonal falls outside [100, 2000] km
      - the download fails
    """
    item_id = map_meta.get("id", "unknown")
    title = _field(map_meta, "Title") or _field(map_meta, "title") or item_id

    wms = _wms_url(map_meta)
    if not wms:
        return None

    bbox = _parse_bbox(map_meta)
    if bbox is None:
        print(f"  {item_id}: no bounding box in metadata, skipping")
        return None

    west, south, east, north = bbox
    diag = _bbox_diagonal_km(west, south, east, north)
    if not (_MIN_DIAG_KM <= diag <= _MAX_DIAG_KM):
        print(f"  {item_id}: diagonal {diag:.0f} km out of range, skipping")
        return None

    out_path = output_dir / f"{item_id}.tif"
    if out_path.exists():
        print(f"  {item_id}: already downloaded, skipping")
        return out_path

    print(f"  Downloading: {title[:60]!r} ({diag:.0f} km)")
    success = _download_wms_geotiff(wms, bbox, out_path)
    return out_path if success else None


# ---------------------------------------------------------------------------
# Manifest for unregistered maps
# ---------------------------------------------------------------------------

def emit_manifest(items: list[dict], output_path: Path) -> None:
    """
    Write a JSON manifest of maps that have no WMS georeferencing URL, for
    manual GCP registration in QGIS.

    Only includes maps whose bbox (if determinable) is within the scale range,
    or whose bbox is unknown (included with a note).
    """
    manifest = []
    for item in items:
        if _wms_url(item):
            continue  # already georeferenced

        item_id = item.get("id", "")
        title = _field(item, "Title") or _field(item, "title") or item_id
        date = _field(item, "Date") or _field(item, "Pub Date") or ""
        bbox = _parse_bbox(item)

        if bbox is not None:
            west, south, east, north = bbox
            diag = _bbox_diagonal_km(west, south, east, north)
            if not (_MIN_DIAG_KM <= diag <= _MAX_DIAG_KM):
                continue  # wrong scale
            scale_note = f"{diag:.0f} km diagonal"
        else:
            scale_note = "unknown extent — verify before registering"

        # Best-effort thumbnail and detail page URLs
        thumb = next(
            (lnk["href"] for lnk in item.get("links", []) if "thumb" in lnk.get("rel", "")),
            None,
        )
        detail = next(
            (lnk["href"] for lnk in item.get("links", []) if "detail" in lnk.get("rel", "")),
            f"https://www.davidrumsey.com/luna/servlet/detail/{item_id}",
        )

        manifest.append({
            "id": item_id,
            "title": title,
            "date": date,
            "author": _field(item, "Author"),
            "publisher": _field(item, "Publisher"),
            "publisher_location": _field(item, "Publisher Location"),
            "country": _field(item, "Country"),
            "world_area": _field(item, "World Area"),
            "region": _field(item, "Region"),
            "city": _field(item, "City"),
            "scale": _field(item, "Scale 1"),
            "obj_height_cm": _field(item, "Obj Height cm"),
            "obj_width_cm": _field(item, "Obj Width cm"),
            "full_title": _field(item, "Full Title"),
            "reference": _field(item, "Reference"),
            "richness_score": _richness_score(item),
            "scale_note": scale_note,
            "thumbnail_url": item.get("urlSize0"),
            "image_url": item.get("urlSize4"),
            "iiif_manifest": item.get("iiifManifest"),
            "rumsey_page": f"https://www.davidrumsey.com/luna/servlet/detail/{item_id}",
            "status": "needs_gcps",
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written: {output_path} ({len(manifest)} maps need GCPs)")
