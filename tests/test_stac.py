"""
Wave 0 skeleton — STAC client wrapper for ``satellite.stac``.

Body filled by plan 02-04 (satellite pipeline, D-13). ``mock_stac_item``
fixture is available now for the offline lowest-cloud-pick assertion.
"""

import pytest


@pytest.mark.skip(reason="Wave 2 — implemented in plan-02-04 (satellite STAC client)")
def test_picks_lowest_cloud_scene(mock_stac_item):
    """find_lowest_cloud_scene returns the item with minimum eo:cloud_cover."""
