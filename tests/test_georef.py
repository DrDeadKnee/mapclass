"""
Wave 0 skeleton — affine fit + GeoTIFF I/O for ``historical.georef``.

Body filled by plan 02-02 (D-03 affine least-squares fit). ``tiny_geotiff``
fixture is available now for the round-trip assertion.
"""

import pytest


@pytest.mark.skip(reason="Wave 1 — implemented in plan-02-02 (georef affine fit)")
def test_affine_roundtrip(tiny_geotiff):
    """gcps_to_affine + write_georeferenced_geotiff round-trips CRS=4326 + transform."""
