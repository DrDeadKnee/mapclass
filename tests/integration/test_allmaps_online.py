"""
Online integration test for ``historical.allmaps.lookup`` against the live
Allmaps annotations endpoint. Network-required — marked ``integration`` so
the per-commit offline run skips it (run by the phase gate).
"""

import pytest

from historical import allmaps

# A David Rumsey manifest known to be georeferenced in Allmaps.
# Verified live 2026-05-16 (19 GCPs). The prior fixture
# (RUMSEY~8~1~24694~890095) was retired upstream by David Rumsey.
_KNOWN_RUMSEY_MANIFEST = (
    "https://www.davidrumsey.com/luna/servlet/iiif/m/RUMSEY~8~1~292315~90066993/manifest"
)


@pytest.mark.integration
def test_known_rumsey_manifest_has_gcps():
    """A known-georeferenced Rumsey manifest returns >=1 annotation with >=3 GCPs."""
    result = allmaps.lookup(_KNOWN_RUMSEY_MANIFEST)
    assert isinstance(result, list)
    assert len(result) >= 1
    assert len(result[0]["gcps"]) >= 3
