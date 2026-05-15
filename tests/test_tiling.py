"""
Wave 0 skeleton — nested-pyramid tiler for ``scripts/tiling.py``.

Bodies filled by plan 02-05 (shared tiler, D-06..D-09): strict 2x2 nesting
(1x896 + 4x448 + 16x224) and the >50%-off-edge drop policy.
"""

import pytest


@pytest.mark.skip(reason="Wave 3 — implemented in plan-02-05 (nested-pyramid tiler, D-07)")
def test_nested_alignment(tmp_path):
    """Each 896 pyramid's footprint exactly contains its 4 x 448 and 16 x 224 children."""


@pytest.mark.skip(reason="Wave 3 — implemented in plan-02-05 (nested-pyramid tiler, D-09)")
def test_edge_drop(tmp_path):
    """A pyramid whose 896 footprint is >50% off the source map is dropped."""
