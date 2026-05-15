"""
COG byte-range fetch of a Sentinel-2 ``visual`` (TCI) window.

RESEARCH Pattern 2 / threat T-02-10: read ONLY a centred
``rasterio.windows.Window`` of the scene's ``visual`` asset — a full
Sentinel-2 scene is ~600 MB and must NEVER be downloaded. GDAL's VSI-CURL
driver issues HTTP range requests for the windowed blocks only.

The fetched (3, H, W) uint8 RGB plus the window's affine and the scene CRS are
written to a georeferenced GeoTIFF using the SINGLE shared writer from Plan 02
(``historical.georef.write_georeferenced_geotiff``) — there is deliberately NO
inline fallback writer here (W-1 dedup: one canonical implementation). The
resulting GeoTIFF is consumed verbatim by ``historical.label.make_labels``.

The remote-read resilience idiom (``with rasterio.open(url) ... except →
print Warning, return None``) mirrors ``historical/worldcover.py:118-132`` and
maps a malformed item / missing asset to a ``fetch_failed`` outcome at the
orchestrator (threat T-02-11) instead of crashing the batch.
"""

import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.windows import Window

from historical.georef import write_georeferenced_geotiff
from historical.worldcover import assert_safe_raster_size

_WGS84 = CRS.from_epsg(4326)

# NODATA sentinel for reproject-uncovered destination pixels (WR-03).
# 0 collides with legitimate dark imagery, so use the uint8 max — a value
# that does not appear in the Sentinel-2 TCI dynamic range for a covered
# pixel often enough to matter, and is explicitly tagged on the GeoTIFF
# so downstream tiling treats it as nodata rather than real black.
_NODATA = 255

# Allow anonymous access to public S3 buckets via GDAL VSI-CURL (sentinel-cogs).
os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")


def _centred_window(width: int, height: int, dst_window_px: int) -> Window:
    """
    Return a ``dst_window_px``-sized window centred on the raster, clamped so
    it never extends past the raster bounds (edge/partial scenes stay valid).
    """
    size = min(dst_window_px, width, height)
    col_off = max(0, (width - size) // 2)
    row_off = max(0, (height - size) // 2)
    return Window(col_off, row_off, size, size)


def fetch_visual_window(item, dst_path: Path, dst_window_px: int = 4096):
    """
    Read a centred ``dst_window_px``×``dst_window_px`` RGB window from
    ``item.assets["visual"]`` via COG byte-range and write it as a
    georeferenced GeoTIFF using the shared Plan-02 writer.

    Parameters
    ----------
    item : a STAC item (or offline stand-in) with
           ``assets["visual"].href`` and ``properties``
    dst_path : output GeoTIFF path
    dst_window_px : window edge length in pixels (default 4096)

    Returns
    -------
    Path to the written GeoTIFF, or ``None`` if the asset is missing /
    unreadable (caller drop-counts as ``fetch_failed``, threat T-02-11).
    """
    try:
        url = item.assets["visual"].href
    except (KeyError, AttributeError, TypeError) as exc:
        print(f"  Warning: STAC item has no 'visual' asset: {exc}")
        return None

    try:
        with rasterio.open(url) as ds:
            assert_safe_raster_size(ds, "Sentinel COG")
            win = _centred_window(ds.width, ds.height, dst_window_px)
            rgb = ds.read([1, 2, 3], window=win)  # (3, H, W) uint8
            win_transform = ds.window_transform(win)
            src_crs = ds.crs

            # The shared Plan-02 writer hard-codes crs=EPSG:4326. Sentinel-2
            # scenes are in a UTM CRS, so the windowed UTM affine must be
            # reprojected to WGS84 before it is written under the EPSG:4326
            # label — otherwise make_labels' WorldCover/DEM alignment (which
            # consumes ds.crs/ds.transform, D-03) would be geographically
            # corrupt. We only ever warp the 4096-px window, never the scene.
            _, _, h, w = 1, 1, *rgb.shape[1:]
            dst_transform, dst_w, dst_h = calculate_default_transform(
                src_crs, _WGS84, w, h,
                *rasterio.windows.bounds(win, ds.transform),
            )
            rgb_wgs84 = np.zeros((3, dst_h, dst_w), dtype="uint8")
            for b in range(3):
                reproject(
                    source=rgb[b],
                    destination=rgb_wgs84[b],
                    src_transform=win_transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=_WGS84,
                    resampling=Resampling.bilinear,
                )

            # WR-03: the UTM→WGS84 warp of an edge / rotated window leaves
            # destination pixels uncovered. Left as the zero-fill they are
            # written as pure-black (0,0,0) and become indistinguishable
            # from real dark imagery — a silent label/imagery mismatch in
            # Phase-3 training. Mirror the rest of the pipeline's NODATA
            # discipline: reproject an all-ones coverage source so any
            # destination pixel the warp did NOT touch is identifiable, set
            # those pixels to the NODATA sentinel, and tag it on the file so
            # downstream tiling can exclude the uncovered region. A separate
            # coverage band (not "pixel == 0") is used because real imagery
            # may legitimately contain 0.
            coverage = np.zeros((dst_h, dst_w), dtype="uint8")
            reproject(
                source=np.ones(rgb[0].shape, dtype="uint8"),
                destination=coverage,
                src_transform=win_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=_WGS84,
                resampling=Resampling.nearest,
            )
            uncovered = coverage == 0
            rgb_wgs84[:, uncovered] = _NODATA
    except Exception as exc:
        print(f"  Warning: could not read visual window from {url}: {exc}")
        return None

    return write_georeferenced_geotiff(
        rgb_wgs84, dst_transform, Path(dst_path), nodata=_NODATA
    )
