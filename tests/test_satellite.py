"""
Wave 0 skeleton — satellite fetcher + RGB GeoTIFF write for ``satellite.fetch``.

Body filled by plan 02-04 (satellite pipeline, D-11). The COG window read is
S3-backed; the offline assertion will mock rasterio.open.
"""

import pytest


@pytest.mark.skip(reason="Wave 2 — implemented in plan-02-04 (satellite fetch)")
def test_window_fetch_writes_rgb_geotiff(tmp_path):
    """fetch_visual_window writes a 3-band RGB GeoTIFF via georef.write_georeferenced_geotiff."""
