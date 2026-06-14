"""
Search the David Rumsey Map Collection for 16th–17th century regional maps
and download georeferenced map images as GeoTIFFs.

Two data sources are stitched together:

  - LUNA search API: https://www.davidrumsey.com/luna/servlet/as/search
    Returns JSON with map metadata, IIIF manifest URLs, and JPEG image URLs.
    LUNA itself does *not* expose bounding boxes, WMS URLs, or any indication
    of which maps are georeferenced.

  - Allmaps annotation index: https://annotations.allmaps.org
    Public W3C Web-Annotation index of community-contributed georeferencing
    for IIIF maps, keyed by IIIF manifest URL. Provides ground control points
    (pixel ↔ WGS84) for the Rumsey subset that has been crowdsourced.

For each ranked LUNA result we ask Allmaps for georeferencing. If GCPs exist
and the resulting WGS84 bbox falls within the scale range, the IIIF image is
downloaded, the GCPs are scaled to that size, and a GeoTIFF is written using
an affine transform fit. Maps absent from Allmaps are emitted to an
unregistered manifest for downstream semi-automatic (PaliGemma) or manual
(MapWarper / QGIS) georeferencing.

Search strategy:
  The LUNA API does not support date-range queries (Lucene syntax returns zero
  results). We query year by year and collect a pool capped at pages_per_year
  pages per year, then rank the pool by metadata richness and return the top
  max_results items. This ensures even temporal and geographic distribution
  and biases toward well-documented maps.

Metadata available per LUNA item (always present unless noted):
  Top-level: id, urlSize0–urlSize4 (JPEG image URLs), iiifManifest
  fieldValues: Author, Date, Short Title, Full Title, Type, Obj Height cm,
    Obj Width cm, Publisher, Publisher Location, Pub Title, Pub Type
  Sparse (30–50% of items): Scale 1, Country, City, World Area, Region,
    Reference, Engraver or Printer

Not available from LUNA: bounding boxes, WMS URLs, georeferencer.com links.
"""

import json
import math
import re
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from historical import allmaps, georef, iiif

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


def _sanitize_id(item_id: str) -> str:
    """
    Sanitize a LUNA item id for use as a filesystem directory component.

    LUNA ids are tilde-delimited (RUMSEY~8~1~123~456) with no path separators,
    but any character outside ``[\\w-]`` is replaced with ``_`` regardless
    (threat T-02-02: externally-derived string in a filesystem path).
    """
    return re.sub(r"[^\w-]", "_", item_id or "unknown")


def _is_valid_geotiff(path: Path) -> bool:
    """True if ``path`` opens as a GeoTIFF with a CRS and nonzero dims.

    Used to re-validate a resumed ``source.tif`` (WR-02): a run that
    crashed mid-write leaves a truncated/invalid file that must not be
    silently accepted as a completed plate.
    """
    try:
        import rasterio  # lazy: keeps offline tests rasterio-free
    except ImportError:
        # Cannot validate — fall back to a nonzero-size sanity check.
        try:
            return path.stat().st_size > 0
        except OSError:
            return False
    try:
        with rasterio.open(path) as ds:
            return (
                ds.crs is not None
                and ds.width > 0
                and ds.height > 0
            )
    except Exception:
        return False


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

def download_georeferenced(
    map_meta: dict,
    output_dir: Path,
) -> tuple[list[Path], str]:
    """
    Resolve a LUNA item through the Allmaps → IIIF → georeferenced-GeoTIFF flow.

    Flow (D-01..D-04):
      1. ``allmaps.lookup(item["iiifManifest"])`` → ``list[dict]`` of
         per-annotation georeferencing (a multi-canvas atlas yields N entries).
      2. For each annotation: compute the GCP-derived bbox diagonal; if it falls
         outside ``[_MIN_DIAG_KM, _MAX_DIAG_KM]`` ([100, 2000] km) the annotation
         is dropped SILENTLY and counted ``out_of_scale`` (D-04: not emitted to
         the unregistered manifest).
      3. Otherwise the IIIF image is fetched at !4096,4096 best-fit, the GCPs are
         scaled by the *actual* fetched/original ratio, and a georeferenced
         GeoTIFF is written to
         ``output_dir/<sanitized_id>__plate<i>/source.tif``.

    Returns ``(written_paths, status)`` where ``status`` is one of::

        ok                — at least one plate written
        out_of_scale      — every annotation was outside the scale window
        not_in_allmaps    — manifest absent / no parseable annotations
        gcps_insufficient — annotations present but no usable GCPs/image size
        download_failed   — IIIF fetch or GeoTIFF write raised for every plate

    A single bad annotation never aborts the batch (threat T-02-05): per-plate
    failures are caught and classified, not raised.
    """
    item_id = map_meta.get("id", "unknown")
    title = _field(map_meta, "Title") or _field(map_meta, "title") or item_id
    manifest_url = map_meta.get("iiifManifest")

    if not manifest_url:
        return [], "not_in_allmaps"

    try:
        annotations = allmaps.lookup(manifest_url)
    except allmaps.AllmapsLookupError as exc:
        print(f"  {item_id}: Allmaps lookup error: {exc}")
        return [], "download_failed"

    if not annotations:
        # Empty list covers 404 (not in Allmaps), 500 (invalid IIIF), and the
        # all-malformed / <3-GCP case — classified as not_in_allmaps (D-04).
        return [], "not_in_allmaps"

    safe_id = _sanitize_id(item_id)
    written: list[Path] = []
    saw_out_of_scale = False
    saw_gcps_insufficient = False
    saw_download_failed = False

    for i, ann in enumerate(annotations):
        gcps = ann.get("gcps") or []
        image_id = ann.get("image_id")
        image_size = ann.get("image_size")
        if len(gcps) < 3 or not image_id or not image_size:
            saw_gcps_insufficient = True
            continue

        diag = allmaps.bbox_diagonal_km(ann["bbox"])
        if not (_MIN_DIAG_KM <= diag <= _MAX_DIAG_KM):
            # D-04: out-of-scale maps are dropped SILENTLY (no manifest entry);
            # the caller's per-reason counter still surfaces the count loudly.
            saw_out_of_scale = True
            continue

        plate_dir = output_dir / f"{safe_id}__plate{i}"
        out_path = plate_dir / "source.tif"
        if out_path.exists():
            # WR-02: a prior run may have crashed mid-write, leaving a
            # truncated/invalid source.tif. Re-validate (openable + has a
            # CRS + nonzero dims) before trusting it; if invalid, delete
            # and re-fetch rather than handing a corrupt GeoTIFF onward.
            if _is_valid_geotiff(out_path):
                print(f"  {item_id} plate{i}: already written, skipping")
                written.append(out_path)
                continue
            print(f"  {item_id} plate{i}: existing source.tif invalid — "
                  f"re-fetching")
            try:
                out_path.unlink()
            except OSError:
                pass

        try:
            orig_w, orig_h = image_size
            jpeg_path = plate_dir / "_iiif.jpg"
            jpeg_path, fetched_w, fetched_h = iiif.fetch_iiif_image(
                image_id, jpeg_path
            )
            scaled = iiif.scale_gcps(
                gcps, orig_w, orig_h, fetched_w, fetched_h
            )
            affine = georef.gcps_to_affine(scaled)
            with Image.open(jpeg_path) as img:
                rgb = np.asarray(img.convert("RGB"))  # (H, W, 3)
            rgb = np.moveaxis(rgb, -1, 0)  # (3, H, W)
            georef.write_georeferenced_geotiff(rgb, affine, out_path)
            jpeg_path.unlink(missing_ok=True)
            print(
                f"  Wrote: {title[:50]!r} plate{i} "
                f"({diag:.0f} km, {fetched_w}x{fetched_h})"
            )
            written.append(out_path)
        except Exception as exc:  # noqa: BLE001 — one bad map must not abort batch
            print(f"  {item_id} plate{i}: download/write failed: {exc}")
            saw_download_failed = True
            continue

    if written:
        return written, "ok"
    if saw_out_of_scale:
        return [], "out_of_scale"
    if saw_download_failed:
        return [], "download_failed"
    if saw_gcps_insufficient:
        return [], "gcps_insufficient"
    return [], "not_in_allmaps"


# ---------------------------------------------------------------------------
# Manifest for unregistered maps
# ---------------------------------------------------------------------------

def emit_manifest(items: list[tuple[dict, str]], output_path: Path) -> None:
    """
    Write the v2 hand-off manifest of maps that did NOT produce a georeferenced
    GeoTIFF, for downstream semi-automatic (PaliGemma) or manual (MapWarper /
    QGIS) registration. (RESEARCH finding 3: this is the v2 hand-off — no
    PaliGemma / manual registration is attempted here; deferred per Plan 01.)

    ``items`` is a list of ``(luna_item, status)`` pairs where ``status`` is the
    D-04 per-reason classification from :func:`download_georeferenced`. Only
    ``not_in_allmaps`` and ``gcps_insufficient`` belong in the manifest —
    ``out_of_scale`` maps are dropped silently (D-04) and ``ok`` maps are
    already georeferenced, so the caller must not pass those here.
    """
    manifest = []
    for item, status in items:
        item_id = item.get("id", "")
        title = _field(item, "Title") or _field(item, "title") or item_id
        date = _field(item, "Date") or _field(item, "Pub Date") or ""

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
            "thumbnail_url": item.get("urlSize0"),
            "image_url": item.get("urlSize4"),
            "iiif_manifest": item.get("iiifManifest"),
            "rumsey_page": f"https://www.davidrumsey.com/luna/servlet/detail/{item_id}",
            "status": status,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written: {output_path} ({len(manifest)} maps need GCPs)")
