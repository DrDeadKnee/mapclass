"""
Wave 0 skeleton — synthetic render dimensions for ``scripts/render.py``.

Body filled by plan 02-03 (synthetic pipeline).
"""

import pytest


@pytest.mark.skip(reason="Wave 1 — implemented in plan-02-03 (synthetic render)")
def test_output_dimensions_match(sample_azgaar_geojson, tmp_path):
    """render_map's image dims match the land_cover.png / topography.png dims."""
