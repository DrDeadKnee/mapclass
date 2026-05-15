"""
Wave 0 skeleton — ``historical.label.make_labels`` end-to-end on a fixture.

Body filled by plan 02-02. The WorldCover / DEM fetches are S3-backed, so the
real assertion will mock them; left skipped here.
"""

import pytest


@pytest.mark.skip(reason="Wave 1 — implemented in plan-02-02 (label make_labels)")
def test_make_labels_writes_all_outputs(tiny_geotiff, tmp_path):
    """make_labels produces image.png + land_cover.png + topography.png + sample_weights.json."""
