"""
Offline test for ``historical.label.make_labels`` (plan 02-02, D-05 verify).

This does NOT modify ``label.py`` — it pins the existing per-map output contract
that the historical GeoTIFF written by ``georef.write_georeferenced_geotiff``
must satisfy: ``image.png`` + ``land_cover.png`` + ``topography.png`` +
``sample_weights.json`` with the locked HISTORICAL_LC_WEIGHTS keys.

WorldCover / Copernicus DEM fetches are S3-backed, so they are stubbed here to
keep the test offline (the contract under test is the writer, not the fetch).
"""

import json

import numpy as np
import pytest

from historical import label as hist_label
from historical.label import HISTORICAL_LC_WEIGHTS, _to_wgs84_bbox


def test_make_labels_writes_all_outputs(tiny_geotiff, tmp_path, mocker):
    """make_labels produces image.png + land_cover.png + topography.png + weights."""
    h = w = 256

    def _fake_wc(ds, bbox):
        return np.zeros((ds.height, ds.width), dtype=np.uint8)

    def _fake_topo(ds, bbox, water_mask):
        return np.zeros((ds.height, ds.width), dtype=np.uint8)

    mocker.patch.object(hist_label, "fetch_worldcover", side_effect=_fake_wc)
    mocker.patch.object(hist_label, "fetch_topo", side_effect=_fake_topo)

    out_dir = tmp_path / "sample"
    hist_label.make_labels(tiny_geotiff, out_dir)

    for name in ("image.png", "land_cover.png", "topography.png", "sample_weights.json"):
        assert (out_dir / name).exists(), f"missing {name}"

    weights = json.loads((out_dir / "sample_weights.json").read_text())
    assert weights["source"] == "historical"
    assert set(weights["land_cover_weights"].keys()) == set(HISTORICAL_LC_WEIGHTS.keys())
    assert "topography_weight" in weights


# ---------------------------------------------------------------------------
# WR-10 regression: an antimeridian-crossing extent yields an inside-out
# WGS84 bbox (west > east). Tile enumeration would produce an all-NODATA
# label pair, so _to_wgs84_bbox must RAISE (accounted drop) rather than
# warn-and-return the corrupt bbox.
# ---------------------------------------------------------------------------


class _Bounds:
    def __init__(self, left, bottom, right, top):
        self.left, self.bottom, self.right, self.top = left, bottom, right, top


class _FakeDS:
    """Minimal rasterio-dataset stand-in for _to_wgs84_bbox."""

    def __init__(self, crs, bounds):
        self.crs = crs
        self.bounds = bounds


def test_antimeridian_bbox_raises_instead_of_silent_nodata(mocker):
    """west > east must raise so the source is drop-counted (WR-10)."""
    from rasterio.crs import CRS

    # A non-WGS84 CRS forces the densified transform_bounds path.
    ds = _FakeDS(CRS.from_epsg(3857), _Bounds(0, 0, 10, 10))
    # Simulate an antimeridian-crossing reprojection: west > east.
    mocker.patch(
        "rasterio.warp.transform_bounds",
        return_value=(170.0, -10.0, -170.0, 10.0),
    )
    with pytest.raises(ValueError, match="antimeridian"):
        _to_wgs84_bbox(ds)


def test_normal_reprojected_bbox_still_returned(mocker):
    """A well-formed reprojected bbox (west < east) is returned intact."""
    from rasterio.crs import CRS

    ds = _FakeDS(CRS.from_epsg(3857), _Bounds(0, 0, 10, 10))
    mocker.patch(
        "rasterio.warp.transform_bounds",
        return_value=(4.0, 50.0, 5.0, 51.0),
    )
    assert _to_wgs84_bbox(ds) == (4.0, 50.0, 5.0, 51.0)
