"""
Online integration tests for the satellite fetch + end-to-end build
(plan 02-04).

@pytest.mark.integration — hits the live STAC API and reads a real
Sentinel-2 ``visual`` COG via S3 byte-range. Run with:
    pytest tests/integration/test_satellite_online.py -m integration
"""

import sys
from pathlib import Path

import pytest
import rasterio

from satellite import fetch, stac

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


@pytest.mark.integration
def test_window_fetch_shape(tmp_path):
    """A real visual-asset window fetch produces a 3-band uint8 WGS84 GeoTIFF."""
    item = stac.find_lowest_cloud_scene(
        bbox=(4.0, 50.0, 6.0, 52.0),
        datetime_range="2023-06-01/2023-08-31",
        max_cloud=40,
    )
    assert item is not None
    out = fetch.fetch_visual_window(item, tmp_path / "scene.tif", dst_window_px=4096)
    assert out is not None
    with rasterio.open(out) as ds:
        assert ds.count == 3
        assert ds.dtypes[0] == "uint8"
        # Reprojected to WGS84; window is ~4096 px before warp, similar after.
        assert ds.width > 1000 and ds.height > 1000


@pytest.mark.integration
def test_build_one_region_end_to_end(tmp_path):
    """
    search → build for one known-coverage region yields a per-map dir with
    image.png + land_cover.png + topography.png + sample_weights.json.
    """
    import build_satellite_dataset as bsd

    manifest = tmp_path / "manifest.json"
    out_dir = tmp_path / "dataset"

    # One hand-specified region (skip the expensive global coverage scan).
    region = {
        "region_id": "europe_test",
        "bbox": [4.0, 50.0, 6.0, 52.0],
        "datetime_range": "2023-06-01/2023-08-31",
    }
    bsd.cmd_search_regions([region], manifest, max_cloud=40)
    assert manifest.exists()

    bsd.cmd_build(manifest, out_dir, workers=1)

    dirs = [p for p in out_dir.iterdir() if p.is_dir()]
    assert dirs, "no per-map dir produced"
    d = dirs[0]
    for name in ("image.png", "land_cover.png", "topography.png", "sample_weights.json"):
        assert (d / name).exists(), f"missing {name}"
