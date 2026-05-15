"""
Wave 0 skeleton — Allmaps lookup() behaviour for ``historical.allmaps``.

The offline assertions (multi-annotation, 404-empty, malformed-skip) are
filled by Task 2 of plan 02-01 once ``lookup()`` returns ``list[dict]``.
"""

import pytest


@pytest.mark.skip(reason="plan-02-01 Task 2 — filled after lookup() multi-annotation fix")
def test_lookup_returns_all_annotations(sample_allmaps_multi):
    """lookup() returns every parseable annotation in items[], not just items[0]."""
