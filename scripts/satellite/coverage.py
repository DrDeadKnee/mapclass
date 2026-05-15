"""
Class-diversity coarse-WorldCover region picker (D-14).

RESEARCH Pattern 5 (Pattern A — class-diversity stratified coverage): instead
of sampling the globe uniformly (which over-samples forest/water, since most of
the Earth's land is forest or water — the rejected Pattern B), build a one-shot
coarse global WorldCover class-count summary at ~1° cell resolution, cache it,
then rank 1° cells by a class-diversity score (Shannon entropy over the canonical
9-class distribution) with an extra multiplicative up-weight on the three
synthetic-absent classes (cropland / built_up / flooded_wetland) per D-14.

Reuses ``historical.worldcover._tile_origins`` (3° WorldCover tile enumeration)
and ``WC_REMAP`` (WorldCover value → canonical 9-class index) verbatim — no
forked tile math.

Each picked cell carries a latitude-appropriate seasonal STAC datetime range
(Pitfall 4: northern temperate May–Sep, southern temperate Nov–Mar, tropics
search the trailing 12 months and let the cloud filter decide) so
``stac.find_lowest_cloud_scene`` is never handed a hemisphere-wrong season.

The build scan is expensive (it reads every 3° WorldCover tile), so the
summary is a MANDATORY cache: if the sidecar exists it is loaded, never
rescanned.
"""

import json
import math
import os
from pathlib import Path

import numpy as np
import rasterio

from historical.worldcover import (
    WC_REMAP,
    _WC_BASE,
    _WC_FILENAME,
    _tile_name,
    _tile_origins,
)

os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")

# 9 canonical land-cover class indices (0..8), matching biome_mapping.
_N_CLASSES = 9
# Canonical indices that are synthetic-absent and PROJECT.md / D-14 up-weighted.
_UPWEIGHT_CLASSES = {
    4,  # cropland
    5,  # built_up
    7,  # flooded_wetland
}
_DIVERSITY_UPWEIGHT = 1.5  # multiplicative bonus per up-weighted class present

_DEFAULT_SUMMARY_PATH = Path("data/satellite/coverage_summary.json")

# Global WorldCover footprint (v200 covers -180..180, -60..84).
_GLOBAL_BBOX = (-180.0, -60.0, 180.0, 84.0)
# Coarse decimated read: sample each 3° tile on a small grid (~1 km-ish at
# 3°/_TILE_SAMPLE; keeps the one-shot scan tractable).
_TILE_SAMPLE = 64


# ---------------------------------------------------------------------------
# seasonal datetime per latitude band (Pitfall 4)
# ---------------------------------------------------------------------------

def season_for_latitude(lat: float, ref_year: int = 2023) -> str:
    """
    Return a STAC ``datetime`` range string appropriate for ``lat``.

    Northern temperate (lat > 23.5): boreal summer May–Sep (5 months,
    within ``ref_year``).
    Southern temperate (lat < -23.5): one contiguous austral summer
    Dec–Feb (3 months, ``ref_year - 1`` Dec through ``ref_year`` Feb) —
    a single season, not two split summers.
    Tropics (|lat| <= 23.5): exactly the trailing 12 months of
    ``ref_year`` — let the cloud filter decide (cloud cover, not season,
    dominates tropical scene quality).
    """
    if lat > 23.5:
        return f"{ref_year}-05-01/{ref_year}-09-30"
    if lat < -23.5:
        # one contiguous austral summer (Dec–Feb), no double-season
        return f"{ref_year - 1}-12-01/{ref_year}-02-28"
    # tropics: exactly the trailing 12 months of ref_year
    return f"{ref_year}-01-01/{ref_year}-12-31"


# ---------------------------------------------------------------------------
# coarse summary (one-shot, cached)
# ---------------------------------------------------------------------------

def _class_counts_for_tile(arr: np.ndarray) -> dict[int, int]:
    """Count canonical 9-class occurrences in a decimated WorldCover array."""
    counts = {c: 0 for c in range(_N_CLASSES)}
    for wc_val, canonical in WC_REMAP.items():
        counts[canonical] += int(np.count_nonzero(arr == wc_val))
    return counts


def build_summary(
    summary_path: Path = _DEFAULT_SUMMARY_PATH,
    bbox: tuple[float, float, float, float] = _GLOBAL_BBOX,
    force: bool = False,
) -> dict:
    """
    Build (or load) the coarse per-1°-cell WorldCover class-count summary.

    If ``summary_path`` already exists and ``force`` is False, the cached
    summary is loaded and returned WITHOUT rescanning (mandatory cache — the
    scan reads every 3° WorldCover tile and is expensive).

    Returns a dict: ``{"cells": [{"lat": int, "lon": int,
    "counts": [9 ints]}, ...]}`` (1° SW-corner per cell).
    """
    summary_path = Path(summary_path)
    if summary_path.exists() and not force:
        return json.loads(summary_path.read_text())

    west, south, east, north = bbox
    cells: list[dict] = []

    for lat_sw, lon_sw in _tile_origins(west, south, east, north):
        tile_id = _tile_name(lat_sw, lon_sw)
        url = f"{_WC_BASE}/{_WC_FILENAME.format(tile=tile_id)}"
        try:
            with rasterio.open(url) as ds:
                # Decimated read: out_shape downsamples the 3° tile in-flight.
                arr = ds.read(
                    1,
                    out_shape=(_TILE_SAMPLE, _TILE_SAMPLE),
                )
        except Exception as exc:  # ocean / missing tile — expected
            print(f"  Warning: could not read WorldCover tile {tile_id}: {exc}")
            continue

        # Split the 3° tile into its nine 1° cells; count classes per cell.
        step = _TILE_SAMPLE // 3
        for dlat in range(3):
            for dlon in range(3):
                sub = arr[
                    dlat * step:(dlat + 1) * step,
                    dlon * step:(dlon + 1) * step,
                ]
                if sub.size == 0:
                    continue
                counts = _class_counts_for_tile(sub)
                if sum(counts.values()) == 0:
                    continue  # all-ocean / nodata cell
                cells.append({
                    "lat": int(lat_sw + dlat),
                    "lon": int(lon_sw + dlon),
                    "counts": [counts[c] for c in range(_N_CLASSES)],
                })

    summary = {"cells": cells}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary))
    print(f"  Coverage summary: {len(cells)} cells → {summary_path}")
    return summary


# ---------------------------------------------------------------------------
# class-diversity region picker
# ---------------------------------------------------------------------------

def _shannon_entropy(counts: list[int]) -> float:
    """Shannon entropy (nats) of a class-count distribution."""
    total = sum(counts)
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log(p)
    return h


def _diversity_score(counts: list[int]) -> float:
    """
    Class-diversity score: Shannon entropy multiplicatively boosted for every
    synthetic-absent class (cropland/built_up/flooded_wetland) that is present
    in the cell (D-14 — bias selection toward the under-represented classes).
    """
    base = _shannon_entropy(counts)
    boost = 1.0
    for idx in _UPWEIGHT_CLASSES:
        if idx < len(counts) and counts[idx] > 0:
            boost *= _DIVERSITY_UPWEIGHT
    return base * boost


def pick_regions(
    n: int,
    seed: int,
    summary: dict | None = None,
    summary_path: Path = _DEFAULT_SUMMARY_PATH,
) -> list[dict]:
    """
    Rank cells by the class-diversity score and return the top ``n`` as
    region dicts, each with a 1° ``bbox``, a stable ``region_id``, and a
    latitude-appropriate ``datetime_range`` (Pitfall 4).

    ``seed`` makes the selection reproducible: ties (equal score) are broken
    by a seeded shuffle so two calls with the same seed pick identical cells.
    """
    if summary is None:
        summary = build_summary(summary_path)
    cells = list(summary["cells"])

    rng = np.random.default_rng(seed)
    # Seeded shuffle first so equal-score ties resolve deterministically.
    order = rng.permutation(len(cells))
    cells = [cells[i] for i in order]
    cells.sort(key=lambda c: _diversity_score(c["counts"]), reverse=True)

    picked: list[dict] = []
    for cell in cells[:n]:
        lat, lon = cell["lat"], cell["lon"]
        picked.append({
            "region_id": f"wc_{lat:+03d}_{lon:+04d}",
            "bbox": [float(lon), float(lat), float(lon + 1), float(lat + 1)],
            "datetime_range": season_for_latitude(lat),
            "diversity_score": _diversity_score(cell["counts"]),
        })
    return picked
