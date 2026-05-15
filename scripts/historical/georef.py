"""
Fit a least-squares affine transform from scaled GCPs and write a
georeferenced GeoTIFF in EPSG:4326.

The affine is fit by GDAL's ``GDALGCPsToGeoTransform`` (least squares) via
``rasterio.transform.from_gcps`` — it is NOT hand-rolled.

Pitfall 1: ``GroundControlPoint`` takes ``row`` (image Y / pixel-py),
``col`` (image X / pixel-px), ``x`` (world lng), ``y`` (world lat). The GCP
construction is wrapped in a single helper with named kwargs so the convention
cannot silently drift.

The written GeoTIFF carries ``crs=EPSG:4326`` and ``transform=<affine>`` so
``historical.label.make_labels`` consumes ``ds.crs`` / ``ds.transform``
directly with no resampling layer (D-03).
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS
from rasterio.transform import from_gcps


def gcps_to_affine(scaled_gcps: list):
    """
    Fit an :class:`affine.Affine` from ``scaled_gcps``.

    ``scaled_gcps`` is a list of ``((px, py), (lng, lat))`` tuples (pixel coords
    already scaled to the fetched image resolution). ``GroundControlPoint`` is
    built with named kwargs ``row=py, col=px, x=lng, y=lat`` (Pitfall 1); the
    fit is delegated to GDAL via ``rasterio.transform.from_gcps``.
    """
    rasterio_gcps = [
        GroundControlPoint(row=py, col=px, x=lng, y=lat)
        for (px, py), (lng, lat) in scaled_gcps
    ]
    return from_gcps(rasterio_gcps)


def write_georeferenced_geotiff(rgb_array: np.ndarray, affine, out_path: Path) -> Path:
    """
    Write a 3-band uint8 ``(3, H, W)`` array as a georeferenced GeoTIFF.

    ``crs`` is fixed to EPSG:4326 and ``transform`` to the supplied affine so the
    file is directly consumable by ``historical.label.make_labels`` (D-03).
    Returns the written path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = rgb_array.shape[1:]
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=3,
        dtype="uint8",
        crs=CRS.from_epsg(4326),
        transform=affine,
    ) as ds:
        ds.write(rgb_array)
    return out_path
