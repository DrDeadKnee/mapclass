"""
Wave 0 skeleton — train/test deterministic split + frozen manifest (EVAL-01).

Bodies filled by plan 02-03 (D-15..D-18). These are the EVAL-01 guardrails:
no train/test ID intersection and a frozen split.json.
"""

import pytest


@pytest.mark.skip(reason="Wave 1 — implemented in plan-02-03 (synthetic split, D-16)")
def test_seeded_split_deterministic(tmp_path):
    """The seeded stratified split produces the same test-set IDs across re-invocations."""


@pytest.mark.skip(reason="Wave 1 — implemented in plan-02-03 (EVAL-01, D-17)")
def test_no_train_test_intersection(tmp_path):
    """data/synthetic/test/ map IDs never appear under data/synthetic/train/."""


@pytest.mark.skip(reason="Wave 1 — implemented in plan-02-03 (EVAL-01, D-18)")
def test_split_manifest_frozen(tmp_path):
    """Rebuilding does not change the recorded split.json test-set ID list."""
