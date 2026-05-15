"""
Wave 0 skeleton — IIIF size syntax + scale-factor logic for ``historical.iiif``.

Bodies are filled by plan 02-02. The module does not yet exist, so the skip
keeps collection green.
"""

import pytest


@pytest.mark.skip(reason="Wave 1 — implemented in plan-02-02 (historical iiif fetcher)")
def test_max_edge_size_syntax():
    """build_iiif_url emits /full/!4096,4096/0/default.jpg (size-best-fit)."""


@pytest.mark.skip(reason="Wave 1 — implemented in plan-02-02 (historical iiif fetcher)")
def test_scale_gcps_uses_fetched_dimensions():
    """GCP pixel coords are scaled by fetched_max_edge / original_max_edge (Pitfall 2)."""
