"""
Online integration tests for ``satellite.stac`` (plan 02-04).

@pytest.mark.integration — hits the live Element84 Earth Search STAC API.
Run with: pytest tests/integration/test_stac_online.py -m integration
"""

import pytest

from satellite import stac


@pytest.mark.integration
def test_known_bbox_returns_results():
    """A cloud-tolerant search over a large land bbox returns a Sentinel-2 item."""
    # Central Europe, a generous month window, relaxed cloud bound so the
    # assertion is about connectivity + parsing, not about scene scarcity.
    item = stac.find_lowest_cloud_scene(
        bbox=(4.0, 50.0, 6.0, 52.0),
        datetime_range="2023-06-01/2023-08-31",
        max_cloud=40,
    )
    assert item is not None
    assert "visual" in item.assets
    assert item.properties.get("eo:cloud_cover") < 40
