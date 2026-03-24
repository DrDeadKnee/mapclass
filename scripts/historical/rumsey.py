"""
Search the David Rumsey Map Collection for 16th–17th century regional maps
and download georeferenced map images as GeoTIFFs.

Uses two endpoints:
  - LUNA search API: https://www.davidrumsey.com/luna/servlet/as/search
    Returns JSON with map metadata including any WMS georeferencing URLs.
  - WMS GetMap (Georeferencer): https://maps.georeferencer.com/georeferences/...
    For maps already registered in the Georeferencer service.

Maps that are not yet georeferenced are written to an unregistered_manifest.json
for manual GCP placement in QGIS.

Scale filtering:
  Maps whose spatial extent diagonal is outside [100 km, 2000 km] are excluded.
  The Haversine formula is used to compute diagonal from the bounding box.
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
    """Extract a named field from a LUNA item's fieldValues list."""
    for fv in item.get("fieldValues", []):
        if fv.get("name", "").lower() == name.lower():
            return fv.get("value", default)
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
            return resp.json()
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
) -> list[dict]:
    """
    Search David Rumsey LUNA API for maps dated between date_start and date_end.

    Returns a list of raw item dicts as returned by the API, unfiltered.
    Spatial filtering (scale) is applied in download_georeferenced / emit_manifest.
    """
    q = f'type:Map AND date:[{date_start} TO {date_end}]'
    items = []
    lc = 1  # list cursor (1-based)

    print(f"Searching David Rumsey: {q}")
    while len(items) < max_results:
        data = _get_json(_LUNA_SEARCH, {
            "q": q,
            "output": "json",
            "bs": _BATCH_SIZE,
            "lc": lc,
        })

        batch = data.get("results", {}).get("items", [])
        if not batch:
            break

        items.extend(batch)
        print(f"  Fetched {len(items)} maps so far…")
        lc += _BATCH_SIZE

        if len(batch) < _BATCH_SIZE:
            break  # last page

    print(f"Search complete: {len(items)} maps found")
    return items[:max_results]


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
            "scale_note": scale_note,
            "thumbnail_url": thumb,
            "rumsey_page": detail,
            "status": "needs_gcps",
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written: {output_path} ({len(manifest)} maps need GCPs)")
