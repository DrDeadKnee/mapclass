"""
Wave 0 skeleton — LUNA search + filter behaviour for ``historical.rumsey``.

Bodies are filled by plan 02-02 (historical pipeline, D-01..D-05). They are
skipped here so ``pytest tests/`` collects green; the online search test lives
under ``tests/integration/``.
"""

import pytest


@pytest.mark.skip(reason="Wave 1 — implemented in plan-02-02 (historical pipeline)")
def test_search_filters_out_of_scale():
    """download_georeferenced drops maps whose bbox diagonal is outside [100, 2000] km."""


@pytest.mark.skip(reason="Wave 1 — implemented in plan-02-02 (historical pipeline)")
def test_emit_manifest_per_reason_status():
    """unregistered_manifest.json carries a per-reason status (not_in_allmaps / gcps_insufficient)."""
