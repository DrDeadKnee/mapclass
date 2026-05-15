"""
Online integration test for ``historical.rumsey.search_maps`` against the live
David Rumsey LUNA API. Network-required — marked ``integration`` so the
per-commit offline run skips it (run by the phase gate).
"""

import pytest

from historical import rumsey


@pytest.mark.integration
def test_search_returns_results():
    """search_maps over a 16th–17th c. window returns at least one ranked map."""
    results = rumsey.search_maps(date_start=1500, date_end=1700, max_results=10)
    assert isinstance(results, list)
    assert len(results) >= 1
    assert "id" in results[0]
