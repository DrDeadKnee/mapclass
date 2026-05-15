"""
Offline tests for ``historical.georef`` — GCP affine least-squares fit + the
georeferenced-GeoTIFF writer (plan 02-02, D-03).

``test_affine_roundtrip`` builds a known affine from >=4 synthetic GCPs, writes
a GeoTIFF, and asserts CRS=EPSG:4326 + transform survive a reopen.
``test_gcp_convention`` guards Pitfall 1: GroundControlPoint must be built with
row=py, col=px, x=lng, y=lat — an axis-swapped input must produce a detectably
different transform.
"""

import numpy as np
import rasterio
from rasterio.crs import CRS

from historical import georef


def _known_gcps():
    """4 GCPs sampled from a simple affine: lng = 2 + 0.01*px, lat = 51 - 0.01*py."""
    pts = [(0.0, 0.0), (4000.0, 0.0), (4000.0, 3000.0), (0.0, 3000.0)]
    gcps = []
    for px, py in pts:
        lng = 2.0 + 0.01 * px
        lat = 51.0 - 0.01 * py
        gcps.append(((px, py), (lng, lat)))
    return gcps


def test_affine_roundtrip(tmp_path):
    """gcps_to_affine + write_georeferenced_geotiff round-trips CRS=4326 + transform."""
    gcps = _known_gcps()
    affine = georef.gcps_to_affine(gcps)

    h, w = 300, 400
    rgb = np.zeros((3, h, w), dtype=np.uint8)
    out = tmp_path / "georef.tif"
    georef.write_georeferenced_geotiff(rgb, affine, out)

    with rasterio.open(out) as ds:
        assert ds.crs == CRS.from_epsg(4326)
        assert ds.count == 3
        assert (ds.height, ds.width) == (h, w)
        # transform recovered from the same GCPs must match within tolerance
        for a, b in zip(tuple(ds.transform)[:6], tuple(affine)[:6]):
            assert abs(a - b) < 1e-6


def test_gcp_convention():
    """Axis-swapped GCP input yields a detectably different transform (Pitfall 1)."""
    gcps = _known_gcps()
    correct = georef.gcps_to_affine(gcps)

    # Deliberately swap px<->py and lng<->lat going in: a correct implementation
    # (row=py, col=px, x=lng, y=lat named kwargs) must NOT be invariant to this.
    swapped = [((py, px), (lat, lng)) for ((px, py), (lng, lat)) in gcps]
    other = georef.gcps_to_affine(swapped)

    assert tuple(correct)[:6] != tuple(other)[:6]
