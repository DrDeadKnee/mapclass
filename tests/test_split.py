"""
Seeded stratified train/test split + frozen manifest (EVAL-01, plan 02-03 T4).

These are the EVAL-01 guardrails (D-15..D-18):
  * deterministic seeded split across re-invocations,
  * zero train/test source-ID intersection,
  * a frozen split.json (rebuilding never changes the test-set ID list),
  * stratified-proportional hold-out (each Azgaar template ≈15%, not a flat
    global 15% that could starve a small template),
  * whole-source-map split — every render style of a held-out source → test/.

Fully offline. The build is exercised against copies of the minimal
``sample_azgaar_geojson`` fixture, so no network / no real Azgaar export.
"""

import json
import shutil
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import build_dataset as bd  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw_dir(tmp_path, sample_azgaar_geojson, source_ids):
    """Materialise <id>.geojson for every id by copying the fixture."""
    raw = tmp_path / "raw"
    raw.mkdir()
    for sid in source_ids:
        shutil.copy(sample_azgaar_geojson, raw / f"{sid}.geojson")
    return raw


def _fixture_ids(n_per_template, templates):
    """Build N source IDs of the form ``<template>_<NN>`` across templates."""
    ids = []
    for t in templates:
        for i in range(1, n_per_template + 1):
            ids.append(f"{t}_{i:02d}")
    return ids


# ---------------------------------------------------------------------------
# Pure-splitter determinism (offline, no rendering)
# ---------------------------------------------------------------------------

def test_seeded_split_deterministic():
    """Two splitter invocations on the same source-id set yield identical IDs."""
    ids = _fixture_ids(15, [f"cont{n}" for n in range(12)])  # ~180 sources
    first = bd.stratified_split(ids)
    second = bd.stratified_split(ids)
    assert first == second
    assert first == sorted(first)  # returned sorted (stable ordering)
    # Order of the input must not change the result.
    assert bd.stratified_split(list(reversed(ids))) == first


def test_stratified_holdout_proportional():
    """Each template contributes ≈15% (±1) of its sources to test."""
    templates = [f"cont{n}" for n in range(12)]
    ids = _fixture_ids(16, templates)  # 16/template ≈ N=192, ~15-16/template
    test_ids = set(bd.stratified_split(ids))

    for t in templates:
        members = [s for s in ids if bd.template_key(s) == t]
        held = [s for s in members if s in test_ids]
        expected = round(len(members) * bd._TEST_FRACTION)
        assert abs(len(held) - expected) <= 1, (
            f"template {t}: held {len(held)} expected≈{expected} "
            f"of {len(members)}"
        )
        # No template is starved: a non-empty template always gets >=1.
        assert len(held) >= 1


def test_split_is_whole_source_no_template_collision():
    """A singleton-template source is its own stratum (never starved)."""
    ids = ["europe_01", "europe_02", "asia_01", "lonelyisland"]
    test_ids = set(bd.stratified_split(ids))
    # 'lonelyisland' has its own template key → must be held out (>=1 rule).
    assert "lonelyisland" in test_ids


# ---------------------------------------------------------------------------
# Build-level guarantees (offline, tmp dirs)
# ---------------------------------------------------------------------------

def test_no_train_test_intersection(tmp_path, sample_azgaar_geojson):
    """test/ and train/ source-ID sets are disjoint (EVAL-01)."""
    source_ids = _fixture_ids(4, ["alpha", "beta", "gamma"])  # 12 sources
    raw = _make_raw_dir(tmp_path, sample_azgaar_geojson, source_ids)
    out = tmp_path / "synthetic"

    bd.build(raw, out, styles=("flat", "illustrated"))

    def source_ids_under(root):
        if not root.exists():
            return set()
        return {d.name.split(bd._STYLE_SEP)[0] for d in root.iterdir() if d.is_dir()}

    train = source_ids_under(out / "train")
    test = source_ids_under(out / "test")
    assert train and test
    assert train.isdisjoint(test), f"leak: {train & test}"

    # D-15: ALL styles of a held-out source live under test/ (none under train/).
    for sid in test:
        for style in ("flat", "illustrated"):
            assert (out / "test" / f"{sid}{bd._STYLE_SEP}{style}").is_dir()
            assert not (out / "train" / f"{sid}{bd._STYLE_SEP}{style}").exists()


def test_split_manifest_frozen(tmp_path, sample_azgaar_geojson):
    """Rebuilding with new sources never changes split.json's test-ID list."""
    source_ids = _fixture_ids(4, ["alpha", "beta", "gamma"])
    raw = _make_raw_dir(tmp_path, sample_azgaar_geojson, source_ids)
    out = tmp_path / "synthetic"

    bd.build(raw, out, styles=("flat",))
    split_path = out / bd._SPLIT_FILENAME
    assert split_path.exists()
    first_bytes = split_path.read_bytes()
    first_test = set(json.loads(first_bytes)["test"])

    # Add brand-new sources and rebuild.
    new_ids = ["alpha_99", "delta_01", "delta_02"]
    for sid in new_ids:
        shutil.copy(sample_azgaar_geojson, raw / f"{sid}.geojson")
    bd.build(raw, out, styles=("flat",))

    # split.json is byte-identical (frozen — D-18).
    assert split_path.read_bytes() == first_bytes
    assert set(json.loads(split_path.read_text())["test"]) == first_test

    # New sources are NOT in the frozen test list → they land in train/.
    for sid in new_ids:
        assert sid not in first_test
        assert (out / "train" / f"{sid}{bd._STYLE_SEP}flat").is_dir()
        assert not (out / "test" / f"{sid}{bd._STYLE_SEP}flat").exists()
