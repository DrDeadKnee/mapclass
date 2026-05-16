"""
Online integration test for ``historical.iiif.fetch_iiif_image`` against a live
David Rumsey IIIF image service. Network-required — marked ``integration`` so
the per-commit offline run skips it (run by the phase gate).
"""

import pytest

from historical import iiif

# A David Rumsey IIIF image service known to resolve (IIIF Image API 2.x).
# Verified live 2026-05-16 (11651x14997 source). The prior fixture
# (RUMSEY~8~1~24694~890095) was retired upstream by David Rumsey.
_KNOWN_IMAGE_SERVICE = (
    "https://www.davidrumsey.com/luna/servlet/iiif/RUMSEY~8~1~292315~90066993"
)


@pytest.mark.integration
def test_max_edge_fetch(tmp_path):
    """A real IIIF fetch returns an image with both dims <= 4096 and the actual (w, h)."""
    out = tmp_path / "fetched.jpg"
    path, w, h = iiif.fetch_iiif_image(_KNOWN_IMAGE_SERVICE, out)
    assert path.exists()
    assert w <= 4096 and h <= 4096
    assert w > 0 and h > 0
