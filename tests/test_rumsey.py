"""
Offline tests for ``historical.rumsey`` — the Allmaps → IIIF → GeoTIFF flow
and the v2 hand-off manifest (plan 02-02, D-01..D-04).

``allmaps.lookup`` and ``iiif.fetch_iiif_image`` are mocked so no network is
touched; the GeoTIFF write is real (rasterio, local tmp_path).
"""

import json

import numpy as np
import pytest
from PIL import Image

from historical import rumsey


def _gcps(bbox_w, bbox_s, bbox_e, bbox_n):
    """4 corner GCPs spanning the requested WGS84 bbox at 4000x3000 px."""
    return [
        ((0.0, 0.0), (bbox_w, bbox_n)),
        ((4000.0, 0.0), (bbox_e, bbox_n)),
        ((4000.0, 3000.0), (bbox_e, bbox_s)),
        ((0.0, 3000.0), (bbox_w, bbox_s)),
    ]


def _annotation(gcps, bbox, image_size=(4000, 3000)):
    return {
        "gcps": gcps,
        "bbox": bbox,
        "image_id": "https://iiif.example/img",
        "image_size": image_size,
        "transformation": "polynomial",
        "mask_svg": None,
        "allmaps_id": "abc",
        "annotation_id": "anno",
    }


def _fake_fetch(tmp_path):
    """Return a stand-in for iiif.fetch_iiif_image writing a real small JPEG."""
    def _fetch(image_service_id, output_path, max_edge=4096):
        from pathlib import Path

        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        arr = (np.random.default_rng(1).integers(0, 256, (60, 80, 3))).astype(np.uint8)
        Image.fromarray(arr).save(p, format="JPEG")
        return p, 80, 60

    return _fetch


def test_download_not_in_allmaps(mocker, tmp_path):
    """Empty Allmaps lookup → status not_in_allmaps, no paths written."""
    mocker.patch.object(rumsey.allmaps, "lookup", return_value=[])
    item = {"id": "RUMSEY~8~1~1~1", "iiifManifest": "https://m/manifest"}
    paths, status = rumsey.download_georeferenced(item, tmp_path)
    assert paths == []
    assert status == "not_in_allmaps"


def test_download_out_of_scale(mocker, tmp_path):
    """A bbox far larger than _MAX_DIAG_KM → out_of_scale, dropped silently."""
    # ~continental span: diagonal well above 2000 km
    bbox = (-30.0, -30.0, 60.0, 60.0)
    ann = _annotation(_gcps(*bbox), bbox)
    mocker.patch.object(rumsey.allmaps, "lookup", return_value=[ann])
    mocker.patch.object(rumsey.iiif, "fetch_iiif_image", side_effect=_fake_fetch(tmp_path))
    item = {"id": "RUMSEY~8~1~1~2", "iiifManifest": "https://m/manifest"}
    paths, status = rumsey.download_georeferenced(item, tmp_path)
    assert paths == []
    assert status == "out_of_scale"


def test_download_ok_writes_geotiff(mocker, tmp_path):
    """An in-scale annotation → status ok with a written source.tif (CRS=4326)."""
    import rasterio
    from rasterio.crs import CRS

    # ~5 deg box near the equator → diagonal ~hundreds of km, in [100, 2000].
    bbox = (4.0, 48.0, 9.0, 52.0)
    ann = _annotation(_gcps(*bbox), bbox)
    mocker.patch.object(rumsey.allmaps, "lookup", return_value=[ann])
    mocker.patch.object(rumsey.iiif, "fetch_iiif_image", side_effect=_fake_fetch(tmp_path))
    item = {"id": "RUMSEY~8~1~1~3", "iiifManifest": "https://m/manifest"}
    paths, status = rumsey.download_georeferenced(item, tmp_path)
    assert status == "ok"
    assert len(paths) == 1
    assert paths[0].name == "source.tif"
    # sanitized id used as directory component
    assert "RUMSEY_8_1_1_3__plate0" in str(paths[0])
    with rasterio.open(paths[0]) as ds:
        assert ds.crs == CRS.from_epsg(4326)
        assert ds.count == 3


def test_emit_manifest_per_reason_status(tmp_path):
    """emit_manifest carries the D-04 per-reason status; out_of_scale is absent."""
    out = tmp_path / "unregistered_manifest.json"
    items = [
        ({"id": "A", "iiifManifest": "https://m/a"}, "not_in_allmaps"),
        ({"id": "B", "iiifManifest": "https://m/b"}, "gcps_insufficient"),
    ]
    rumsey.emit_manifest(items, out)
    data = json.loads(out.read_text())
    assert {e["id"]: e["status"] for e in data} == {
        "A": "not_in_allmaps",
        "B": "gcps_insufficient",
    }
    assert all(e["status"] != "out_of_scale" for e in data)
    assert all("iiif_manifest" in e for e in data)  # v2 hand-off field retained
