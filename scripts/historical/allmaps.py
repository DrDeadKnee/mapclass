"""
Look up georeferencing data for IIIF-hosted historical maps via Allmaps.

Allmaps (https://allmaps.org) is a public W3C Web-Annotation index of community-
contributed georeferencing for IIIF maps, including the David Rumsey collection.
Given a IIIF manifest URL, the annotations endpoint returns an AnnotationPage
with ground control points (pixel ↔ WGS84) and a polygon mask delimiting the
map's actual content area.

Lookup endpoint:
    GET https://annotations.allmaps.org/?url=<URL-encoded manifest URL>

Outcomes:
    200 with items[] — map is in Allmaps; ALL parseable annotations returned
                        (a multi-canvas atlas yields one entry per canvas)
    404 "Manifest not found" — IIIF manifest is valid but not in Allmaps
    500 "Invalid IIIF data" — manifest URL is malformed; skip
    Other — transient network/server issues

GCP shape (per body.features[i]):
    properties.resourceCoords = [x_px, y_px]   (pixel coords on the source image)
    geometry.coordinates      = [lng, lat]     (WGS84)

Note: the GCPs are expressed in the *original* IIIF image's pixel space
(target.source.width × target.source.height). If the image is downloaded at a
smaller size, GCP pixel coords must be scaled by the same factor.
"""

import math
import time
from typing import Optional
from urllib.parse import quote

import requests

_ALLMAPS_LOOKUP = "https://annotations.allmaps.org/"
_HEADERS = {"User-Agent": "mapclass-dataset-builder/0.1"}
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0


class AllmapsLookupError(RuntimeError):
    """Raised for unexpected (non-404/500) failures during Allmaps lookup."""


def lookup(manifest_url: str) -> list[dict]:
    """
    Return parsed georeferencing data for every annotation Allmaps holds for
    a IIIF manifest. A multi-canvas atlas is georeferenced canvas-by-canvas;
    each canvas is an independent map and is returned as its own dict.

    Each returned dict:
        {
            "gcps":          list of ((x_px, y_px), (lng, lat)) tuples,
            "bbox":          (west, south, east, north) WGS84 envelope of GCPs,
            "image_id":      IIIF image service URL (target.source.id),
            "image_size":    (width, height) at the GCP coordinate scale,
            "transformation": str | None — Allmaps recommended transform type,
            "mask_svg":       SVG polygon string (or None) bounding map content,
            "allmaps_id":     stable Allmaps map ID for citation/caching,
            "annotation_id":  W3C Annotation @id for this map's record,
        }

    Returns an empty list ``[]`` for "not in Allmaps" (404), "invalid IIIF"
    (500), no items, and the all-malformed case. Callers (rumsey.py) treat an
    empty list as not_in_allmaps / gcps_insufficient for D-04 drop
    classification, and a list of length N as N separate maps (each becomes
    its own ``<map>__plate<i>/`` directory downstream).
    """
    url = f"{_ALLMAPS_LOOKUP}?url={quote(manifest_url, safe='')}"

    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=30)
        except requests.RequestException as exc:
            if attempt == _MAX_RETRIES - 1:
                raise AllmapsLookupError(f"network error: {exc}") from exc
            time.sleep(_BACKOFF_BASE ** attempt)
            continue

        if resp.status_code in (404, 500):
            return []  # not in Allmaps / invalid manifest — same caller action

        if resp.status_code in (429, 503):
            time.sleep(_BACKOFF_BASE ** attempt)
            continue

        if resp.status_code != 200:
            raise AllmapsLookupError(
                f"unexpected status {resp.status_code}: {resp.text[:200]}"
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise AllmapsLookupError(f"non-JSON response: {resp.text[:200]}") from exc

        items = data.get("items") or []
        if not items:
            return []

        # A manifest can carry multiple georeferencings — one per canvas of a
        # multi-canvas atlas, or several contributor records. Parse every one;
        # _parse_annotation returns None for malformed / <3-GCP records, which
        # we drop. Each surviving annotation is an independent map downstream
        # (resolved decision A6 / Pitfall 3).
        parsed = [_parse_annotation(ann) for ann in items]
        return [p for p in parsed if p is not None]

    raise AllmapsLookupError(f"exhausted retries for {url}")


def _parse_annotation(ann: dict) -> Optional[dict]:
    """Extract GCPs, bbox, and image metadata from a single Allmaps annotation."""
    body = ann.get("body") or {}
    target = ann.get("target") or {}
    source = target.get("source") or {}

    features = body.get("features") or []
    if not features:
        return None

    gcps: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for feat in features:
        try:
            px = feat["properties"]["resourceCoords"]
            ll = feat["geometry"]["coordinates"]
            gcps.append(((float(px[0]), float(px[1])), (float(ll[0]), float(ll[1]))))
        except (KeyError, TypeError, IndexError, ValueError):
            continue  # malformed feature — skip silently

    if len(gcps) < 3:
        return None  # need at least 3 GCPs for a meaningful affine fit

    lons = [g[1][0] for g in gcps]
    lats = [g[1][1] for g in gcps]
    bbox = (min(lons), min(lats), max(lons), max(lats))

    image_id = source.get("id")
    width = source.get("width")
    height = source.get("height")
    image_size = (int(width), int(height)) if width and height else None

    mask_svg = None
    selector = target.get("selector") or {}
    if isinstance(selector, dict) and selector.get("type") == "SvgSelector":
        mask_svg = selector.get("value")

    allmaps_meta = body.get("_allmaps") or {}

    return {
        "gcps": gcps,
        "bbox": bbox,
        "image_id": image_id,
        "image_size": image_size,
        "transformation": (body.get("transformation") or {}).get("type"),
        "mask_svg": mask_svg,
        "allmaps_id": allmaps_meta.get("id"),
        "annotation_id": ann.get("id"),
    }


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in km between two WGS84 points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def bbox_diagonal_km(bbox: tuple[float, float, float, float]) -> float:
    """Return the great-circle diagonal of a (W, S, E, N) bbox, in km."""
    west, south, east, north = bbox
    return haversine_km(west, south, east, north)
