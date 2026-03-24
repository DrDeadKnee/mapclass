"""
Generate pixel-level labels from a georeferenced historical map GeoTIFF.

Given a GeoTIFF (either downloaded from the Georeferencer WMS or manually
registered in QGIS), this module:
  1. Fetches ESA WorldCover land cover for the map's extent
  2. Fetches Copernicus DEM GLO-30 and derives topography classes
  3. Exports image.png, land_cover.png, topography.png, sample_weights.json

Output layout (one directory per map):
    <output_dir>/
        image.png           — RGB historical map render (uint8)
        land_cover.png      — uint8 canonical land cover class indices
        topography.png      — uint8 topography class indices (255 = water)
        sample_weights.json — per-class loss weights encoding temporal reliability

Loss weights reflect temporal drift since the 16th–17th century:
  - Topography (flat/hilly/mountainous): fully trusted (geology is stable)
  - Water / coastlines: highly reliable at regional scale
  - Trees, cropland, built_up: heavily downweighted (significant land-use change)
"""

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.crs import CRS

from historical.dem import fetch_topo
from historical.worldcover import fetch_worldcover

_WGS84 = CRS.from_epsg(4326)

# Per-class land cover loss weights for the historical source.
# Keys match LANDCOVER_CLASSES from biome_mapping.py.
HISTORICAL_LC_WEIGHTS: dict[str, float] = {
    "water":           1.0,
    "trees":           0.3,   # significant deforestation / reforestation
    "shrubland":       0.7,
    "grassland":       0.7,
    "cropland":        0.15,  # agricultural change is dramatic
    "built_up":        0.1,   # medieval cities vs. modern footprints
    "bare_sparse":     1.0,
    "flooded_wetland": 0.5,
    "snow_ice":        1.0,
}
HISTORICAL_TOPO_WEIGHT: float = 1.0


def _to_wgs84_bbox(ds: rasterio.DatasetReader) -> tuple[float, float, float, float]:
    """Return (west, south, east, north) in WGS84 for a rasterio dataset."""
    if ds.crs is None:
        raise ValueError("Dataset has no CRS — was it properly georeferenced?")
    if ds.crs == _WGS84:
        b = ds.bounds
        return b.left, b.bottom, b.right, b.top

    transformer = Transformer.from_crs(ds.crs, _WGS84, always_xy=True)
    b = ds.bounds
    # Transform all four corners and take the enclosing envelope
    corners = [
        transformer.transform(b.left,  b.bottom),
        transformer.transform(b.right, b.bottom),
        transformer.transform(b.left,  b.top),
        transformer.transform(b.right, b.top),
    ]
    lons = [c[0] for c in corners]
    lats = [c[1] for c in corners]
    return min(lons), min(lats), max(lons), max(lats)


def _read_rgb(ds: rasterio.DatasetReader) -> np.ndarray:
    """Read the first three bands as (H, W, 3) uint8."""
    n = min(ds.count, 3)
    bands = ds.read(list(range(1, n + 1)))  # (C, H, W)
    if bands.dtype != np.uint8:
        # Normalise to uint8 range
        lo, hi = bands.min(), bands.max()
        if hi > lo:
            bands = ((bands - lo) / (hi - lo) * 255).astype(np.uint8)
        else:
            bands = bands.astype(np.uint8)
    arr = np.moveaxis(bands, 0, -1)  # (H, W, C)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.shape[2] == 2:
        arr = np.concatenate([arr, arr[:, :, :1]], axis=2)
    return arr[..., :3]


def make_labels(map_geotiff: Path, output_dir: Path) -> None:
    """
    Produce image.png, land_cover.png, topography.png, and sample_weights.json
    for a single georeferenced historical map GeoTIFF.

    Parameters
    ----------
    map_geotiff : path to a georeferenced GeoTIFF (WMS download or QGIS export)
    output_dir  : directory to write outputs (created if it does not exist)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    map_geotiff = Path(map_geotiff)

    print(f"Processing: {map_geotiff.name}")

    with rasterio.open(map_geotiff) as ds:
        if ds.crs is None:
            print(f"  ERROR: {map_geotiff.name} has no CRS — skipping")
            return

        wgs84_bbox = _to_wgs84_bbox(ds)
        west, south, east, north = wgs84_bbox
        print(f"  Extent: W={west:.2f} S={south:.2f} E={east:.2f} N={north:.2f}")
        print(f"  Grid: {ds.width}×{ds.height} px, CRS: {ds.crs.to_epsg() or ds.crs.to_string()}")

        # 1. Map image
        rgb = _read_rgb(ds)
        Image.fromarray(rgb).save(output_dir / "image.png")
        print(f"  Saved image.png")

        # 2. Land cover
        print(f"  Fetching WorldCover…")
        lc_arr = fetch_worldcover(ds, wgs84_bbox)
        Image.fromarray(lc_arr).save(output_dir / "land_cover.png")
        print(f"  Saved land_cover.png")

        # 3. Topography
        water_mask = lc_arr == 0
        print(f"  Fetching Copernicus DEM…")
        topo_arr = fetch_topo(ds, wgs84_bbox, water_mask)
        Image.fromarray(topo_arr).save(output_dir / "topography.png")
        print(f"  Saved topography.png")

    # 4. Sample weights
    weights = {
        "land_cover_weights": HISTORICAL_LC_WEIGHTS,
        "topography_weight": HISTORICAL_TOPO_WEIGHT,
        "source": "historical",
        "map_file": map_geotiff.name,
    }
    with open(output_dir / "sample_weights.json", "w") as f:
        json.dump(weights, f, indent=2)
    print(f"  Saved sample_weights.json")

    print(f"  Done → {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m historical.label <map.tif> <output_dir>")
        sys.exit(1)
    make_labels(Path(sys.argv[1]), Path(sys.argv[2]))
