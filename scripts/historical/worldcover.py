"""
Fetch ESA WorldCover 10m land cover tiles for a given bounding box and
reproject them into a target rasterio dataset's pixel grid.

WorldCover tiles are 3°×3°, publicly available via HTTPS on the AWS Open Data
registry (no credentials required):
  https://esa-worldcover.s3.amazonaws.com/v200/2021/map/
  ESA_WorldCover_10m_2021_v200_{NS}{LAT:02d}{EW}{LON:03d}_Map.tif

WorldCover class values → canonical 9-class land cover index
(defined in scripts/biome_mapping.py):
  10 → 1  tree cover      → trees
  20 → 2  shrubland
  30 → 3  grassland
  40 → 4  cropland
  50 → 5  built-up        → built_up
  60 → 6  bare/sparse     → bare_sparse
  70 → 8  snow and ice    → snow_ice
  80 → 0  permanent water → water
  90 → 7  herbaceous wetland → flooded_wetland
  95 → 1  mangroves       → trees
 100 → 6  moss and lichen → bare_sparse

NODATA sentinel: 255 (matches label.py convention).
"""

import math
import os

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import Resampling, reproject

# Allow anonymous access to public S3 buckets via GDAL VSI-CURL.
os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")

_WC_BASE = "https://esa-worldcover.s3.amazonaws.com/v200/2021/map"
_WC_FILENAME = "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"

# WR-07: decompression-bomb guard for untrusted remote rasters/COGs. A
# maliciously crafted or corrupt dataset can advertise absurd width/height
# and drive an unbounded array allocation on read. ~2e9 px (≈ a 45000²
# raster) is far above any legitimate 3° WorldCover / windowed Sentinel
# tile but caps a hostile one.
MAX_REMOTE_RASTER_PIXELS = 2_000_000_000


def assert_safe_raster_size(ds, source: str = "remote raster") -> None:
    """Raise ``ValueError`` if an opened dataset exceeds the pixel cap."""
    px = int(ds.width) * int(ds.height)
    if px > MAX_REMOTE_RASTER_PIXELS:
        raise ValueError(
            f"{source} {ds.width}x{ds.height} ({px} px) exceeds the "
            f"{MAX_REMOTE_RASTER_PIXELS} px safety cap — refusing to read "
            f"(decompression-bomb guard, WR-07)"
        )

# ESA WorldCover class value → canonical 9-class index
WC_REMAP: dict[int, int] = {
    10: 1,   # tree cover
    20: 2,   # shrubland
    30: 3,   # grassland
    40: 4,   # cropland
    50: 5,   # built-up
    60: 6,   # bare/sparse vegetation
    70: 8,   # snow and ice
    80: 0,   # permanent water bodies
    90: 7,   # herbaceous wetland
    95: 1,   # mangroves → trees
    100: 6,  # moss and lichen → bare_sparse
}

NODATA = 255
_WGS84 = CRS.from_epsg(4326)


def _tile_name(lat_sw: int, lon_sw: int) -> str:
    """Format the WorldCover tile ID from its SW-corner integer coordinates."""
    ns = "N" if lat_sw >= 0 else "S"
    ew = "E" if lon_sw >= 0 else "W"
    return f"{ns}{abs(lat_sw):02d}{ew}{abs(lon_sw):03d}"


def _tile_origins(west: float, south: float, east: float, north: float):
    """
    Yield (lat_sw, lon_sw) SW-corner integer coordinates of all 3° WorldCover
    tiles whose footprint overlaps the given WGS84 bounding box.
    """
    lat0 = math.floor(south / 3) * 3
    lon0 = math.floor(west / 3) * 3
    lat = lat0
    while lat < north:
        lon = lon0
        while lon < east:
            yield lat, lon
            lon += 3
        lat += 3


def _remap(arr: np.ndarray) -> np.ndarray:
    """Map WorldCover class values to canonical 9-class indices in-place."""
    out = np.full_like(arr, NODATA, dtype=np.uint8)
    unknown = set(np.unique(arr)) - set(WC_REMAP) - {0}
    if unknown:
        print(f"  Warning: unexpected WorldCover values {unknown}; mapped to NODATA")
    for wc_val, canonical in WC_REMAP.items():
        out[arr == wc_val] = canonical
    return out


def fetch_worldcover(
    map_ds: "rasterio.DatasetReader",
    wgs84_bbox: tuple[float, float, float, float],
) -> np.ndarray:
    """
    Fetch ESA WorldCover tiles covering wgs84_bbox, reproject into map_ds's
    pixel grid, and return a uint8 array of canonical land cover class indices.

    Parameters
    ----------
    map_ds : open rasterio dataset defining the target CRS, transform, and size
    wgs84_bbox : (west, south, east, north) in WGS84 degrees

    Returns
    -------
    np.ndarray shape (map_ds.height, map_ds.width), dtype uint8
    """
    west, south, east, north = wgs84_bbox
    height, width = map_ds.height, map_ds.width
    dst = np.full((height, width), NODATA, dtype=np.uint8)

    for lat_sw, lon_sw in _tile_origins(west, south, east, north):
        tile_id = _tile_name(lat_sw, lon_sw)
        url = f"{_WC_BASE}/{_WC_FILENAME.format(tile=tile_id)}"
        try:
            with rasterio.open(url) as tile_ds:
                assert_safe_raster_size(tile_ds, f"WorldCover tile {tile_id}")
                reproject(
                    source=rasterio.band(tile_ds, 1),
                    destination=dst,
                    src_transform=tile_ds.transform,
                    src_crs=tile_ds.crs,
                    dst_transform=map_ds.transform,
                    dst_crs=map_ds.crs,
                    resampling=Resampling.nearest,
                    src_nodata=0,
                    dst_nodata=NODATA,
                )
        except Exception as exc:
            print(f"  Warning: could not read WorldCover tile {tile_id}: {exc}")

    return _remap(dst)
