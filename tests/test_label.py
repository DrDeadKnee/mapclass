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

from historical import label as hist_label
from historical.label import HISTORICAL_LC_WEIGHTS


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
