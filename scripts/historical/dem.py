"""
Fetch Copernicus DEM GLO-30 tiles for a given bounding box, derive terrain
slope, and classify into topography classes matching label.py's convention.

Tiles are 1°×1°, publicly available via HTTPS (no credentials required):
  https://copernicus-dem-30m.s3.amazonaws.com/
  Copernicus_DSM_COG_10_{NS}{LAT:02d}_00_{EW}{LON:03d}_00_DEM/
  Copernicus_DSM_COG_10_{NS}{LAT:02d}_00_{EW}{LON:03d}_00_DEM.tif

Important: tiles over open ocean simply do not exist in the bucket — HTTP 404
is the normal response for fully-oceanic 1° cells and is silently ignored.

Topography classes:
  0 = flat        (slope < 2°)
  1 = hilly       (2° ≤ slope < 15°)
  2 = mountainous (slope ≥ 15°)
255 = water / nodata  (matches WATER_TOPO sentinel in label.py)

Slope is computed in a metric equal-area projection centred on the bbox to
avoid the per-row latitude scaling required for geographic (degree) gradients.
"""

import math
import os

import numpy as np
import rasterio
from pyproj import CRS as ProjCRS
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.warp import Resampling, reproject

os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")

_COP_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"
_COP_PATH = (
    "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM/"
    "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif"
)

FLAT_MAX_DEG = 2.0
HILLY_MAX_DEG = 15.0
WATER_TOPO = 255
_WGS84 = CRS.from_epsg(4326)


def _tile_urls(west: float, south: float, east: float, north: float) -> list[str]:
    """Return S3 HTTPS URLs for all 1° Copernicus DEM tiles covering the bbox."""
    urls = []
    lat = math.floor(south)
    while lat < north:
        lon = math.floor(west)
        while lon < east:
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            path = _COP_PATH.format(ns=ns, lat=abs(lat), ew=ew, lon=abs(lon))
            urls.append(f"{_COP_BASE}/{path}")
            lon += 1
        lat += 1
    return urls


def _laea_crs(centre_lat: float, centre_lon: float) -> CRS:
    """Return a Lambert Azimuthal Equal Area CRS centred on the given point."""
    proj = ProjCRS.from_dict({
        "proj": "laea",
        "lat_0": centre_lat,
        "lon_0": centre_lon,
        "datum": "WGS84",
        "units": "m",
    })
    return CRS.from_wkt(proj.to_wkt())


def _slope_degrees(dem: np.ndarray, res_m: float) -> np.ndarray:
    """
    Compute terrain slope in degrees from a DEM in a metric CRS.
    NaN values (nodata) propagate through gradient computation.
    """
    dz_dy, dz_dx = np.gradient(dem, res_m, res_m)
    with np.errstate(invalid="ignore"):
        slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    return slope


def _classify(slope: np.ndarray, water_mask: np.ndarray) -> np.ndarray:
    topo = np.zeros(slope.shape, dtype=np.uint8)
    topo[slope >= FLAT_MAX_DEG] = 1
    topo[slope >= HILLY_MAX_DEG] = 2
    topo[np.isnan(slope)] = WATER_TOPO
    topo[water_mask] = WATER_TOPO
    return topo


def fetch_topo(
    map_ds: "rasterio.DatasetReader",
    wgs84_bbox: tuple[float, float, float, float],
    water_mask: np.ndarray,
) -> np.ndarray:
    """
    Fetch Copernicus DEM tiles for wgs84_bbox, compute slope in a local
    equal-area projection, classify, and return a uint8 topography array
    in map_ds's pixel grid.

    Parameters
    ----------
    map_ds : open rasterio dataset defining the target grid
    wgs84_bbox : (west, south, east, north) in WGS84 degrees
    water_mask : bool array (H, W) in map_ds's grid — True where land cover = water

    Returns
    -------
    np.ndarray shape (map_ds.height, map_ds.width), dtype uint8
    """
    west, south, east, north = wgs84_bbox
    centre_lat = (south + north) / 2
    centre_lon = (west + east) / 2

    # Build a metric CRS for slope computation
    metric_crs = _laea_crs(centre_lat, centre_lon)

    # Collect DEM tiles; silently skip ocean tiles (404)
    dem_tiles: list[rasterio.DatasetReader] = []
    for url in _tile_urls(west, south, east, north):
        try:
            dem_tiles.append(rasterio.open(url))
        except Exception:
            pass  # ocean tile or missing — expected

    height, width = map_ds.height, map_ds.width

    if not dem_tiles:
        # Entirely oceanic bbox
        topo = np.full((height, width), WATER_TOPO, dtype=np.uint8)
        topo[water_mask] = WATER_TOPO
        return topo

    # Reproject each tile into the metric CRS at native ~30 m resolution,
    # accumulating into a shared intermediate array at the target extent.
    # We use the target map grid dimensions for the intermediate to stay
    # memory-bounded; slope is only slightly affected by this downsampling.
    dem_metric = np.full((height, width), np.nan, dtype=np.float32)

    # Compute the metric transform for map_ds extent
    from rasterio.warp import calculate_default_transform
    metric_transform, metric_w, metric_h = calculate_default_transform(
        map_ds.crs, metric_crs,
        map_ds.width, map_ds.height,
        *map_ds.bounds,
    )

    dem_intermediate = np.full((metric_h, metric_w), np.nan, dtype=np.float32)

    for tile_ds in dem_tiles:
        tile_nodata = tile_ds.nodata if tile_ds.nodata is not None else -9999
        src_arr = tile_ds.read(1).astype(np.float32)
        src_arr[src_arr == tile_nodata] = np.nan
        reproject(
            source=src_arr,
            destination=dem_intermediate,
            src_transform=tile_ds.transform,
            src_crs=tile_ds.crs,
            dst_transform=metric_transform,
            dst_crs=metric_crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
        tile_ds.close()

    res_m = abs(metric_transform.a)
    slope = _slope_degrees(dem_intermediate, res_m)
    topo_metric = _classify(slope, np.zeros((metric_h, metric_w), dtype=bool))

    # Reproject classified topo back to map_ds grid
    dst = np.full((height, width), WATER_TOPO, dtype=np.uint8)
    reproject(
        source=topo_metric,
        destination=dst,
        src_transform=metric_transform,
        src_crs=metric_crs,
        dst_transform=map_ds.transform,
        dst_crs=map_ds.crs,
        resampling=Resampling.nearest,
        src_nodata=WATER_TOPO,
        dst_nodata=WATER_TOPO,
    )

    # Apply water mask last
    dst[water_mask] = WATER_TOPO
    return dst
