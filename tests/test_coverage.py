"""
Offline tests for ``satellite.coverage`` — the class-diversity coarse-WC
region picker (plan 02-04, D-14).

No network: a synthetic in-memory summary stands in for the global scan.
"""

import json

from satellite import coverage


def _cell(lat, lon, counts):
    return {"lat": lat, "lon": lon, "counts": counts}


def _summary(cells):
    return {"cells": cells}


def test_picker_is_deterministic_for_fixed_seed():
    """Two pick_regions calls with the same seed return identical picks."""
    cells = [
        _cell(40, 0, [10, 10, 10, 10, 10, 10, 10, 10, 10]),
        _cell(41, 0, [50, 50, 0, 0, 0, 0, 0, 0, 0]),
        _cell(42, 0, [5, 5, 5, 5, 30, 0, 0, 0, 0]),
        _cell(-30, 10, [0, 0, 0, 0, 10, 10, 0, 10, 0]),
        _cell(10, 20, [20, 20, 20, 0, 0, 0, 0, 0, 0]),
    ]
    s = _summary(cells)
    a = coverage.pick_regions(3, seed=7, summary=s)
    b = coverage.pick_regions(3, seed=7, summary=s)
    assert [r["region_id"] for r in a] == [r["region_id"] for r in b]


def test_upweighted_cell_outranks_equal_entropy_forest_cell():
    """
    A cell rich in cropland/built_up/flooded_wetland outranks a cell with the
    same Shannon entropy but only forest/water (D-14 up-weight).
    """
    # Forest/water-only cell: entropy from a 2-class even split.
    forest = _cell(50, 0, [50, 50, 0, 0, 0, 0, 0, 0, 0])
    # Cropland/built_up even split: SAME 2-class entropy, but both classes are
    # in the up-weighted set → score is multiplicatively boosted.
    synthetic = _cell(51, 0, [0, 0, 0, 0, 50, 50, 0, 0, 0])

    base = coverage._shannon_entropy(forest["counts"])
    assert abs(
        coverage._shannon_entropy(synthetic["counts"]) - base
    ) < 1e-9  # identical raw entropy
    assert coverage._diversity_score(synthetic["counts"]) > coverage._diversity_score(
        forest["counts"]
    )

    picks = coverage.pick_regions(1, seed=0, summary=_summary([forest, synthetic]))
    assert picks[0]["region_id"] == "wc_+51_+000"


def test_season_varies_by_hemisphere():
    """A southern-hemisphere cell gets a different season than a northern one."""
    north = coverage.season_for_latitude(45.0)
    south = coverage.season_for_latitude(-35.0)
    tropic = coverage.season_for_latitude(5.0)
    assert north != south
    assert north.startswith("2023-05-01")          # boreal summer
    assert south.startswith("2022-11-01")          # austral summer (prev yr)
    assert tropic != north and tropic != south     # trailing 12 months


def test_build_summary_reloads_cache_without_rescanning(tmp_path, mocker):
    """If the sidecar exists, build_summary loads it and never opens a tile."""
    cache = tmp_path / "coverage_summary.json"
    payload = {"cells": [_cell(0, 0, [1, 0, 0, 0, 0, 0, 0, 0, 0])]}
    cache.write_text(json.dumps(payload))

    spy = mocker.patch("satellite.coverage.rasterio.open")
    out = coverage.build_summary(summary_path=cache)
    assert out == payload
    spy.assert_not_called()  # cache hit — no tile reads


def test_pick_regions_emits_bbox_and_region_metadata():
    """Each picked region carries a 1° bbox, stable id, and datetime range."""
    s = _summary([_cell(48, 2, [10, 0, 10, 0, 20, 0, 0, 0, 0])])
    r = coverage.pick_regions(1, seed=1, summary=s)[0]
    assert r["bbox"] == [2.0, 48.0, 3.0, 49.0]
    assert r["region_id"] == "wc_+48_+002"
    assert "/" in r["datetime_range"]
