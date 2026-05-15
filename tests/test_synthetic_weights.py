"""
Synthetic per-source loss-weight tests (plan 02-03 Task 3).

Verifies the user-approved uniform-1.0 ``SYNTHETIC_LC_WEIGHTS`` (checkpoint
decision 2026-05-15, option "uniform"): the dict shape is locked to mirror
``HISTORICAL_LC_WEIGHTS``, every value is the float ``1.0``, and the written
``sample_weights.json`` round-trips with the locked top-level keys and
``source == "synthetic"``.

Fully offline — writes only under pytest ``tmp_path``.
"""

import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from historical.label import HISTORICAL_LC_WEIGHTS  # noqa: E402
from synthetic_weights import (  # noqa: E402
    SYNTHETIC_LC_WEIGHTS,
    SYNTHETIC_TOPO_WEIGHT,
    write_sample_weights,
)

_CANONICAL_KEYS = {
    "water",
    "trees",
    "shrubland",
    "grassland",
    "cropland",
    "built_up",
    "bare_sparse",
    "flooded_wetland",
    "snow_ice",
}


def test_keys_set_equal_to_historical():
    """SYNTHETIC_LC_WEIGHTS has exactly the 9 canonical keys (== HISTORICAL)."""
    assert set(SYNTHETIC_LC_WEIGHTS) == set(HISTORICAL_LC_WEIGHTS)
    assert set(SYNTHETIC_LC_WEIGHTS) == _CANONICAL_KEYS


def test_all_values_are_floats():
    """Every land-cover weight is a float; topography weight is a float."""
    for cls, w in SYNTHETIC_LC_WEIGHTS.items():
        assert isinstance(w, float), f"{cls} weight is not a float: {w!r}"
    assert isinstance(SYNTHETIC_TOPO_WEIGHT, float)


def test_approved_uniform_one_values():
    """Approved checkpoint decision: uniform 1.0 for all classes + topo."""
    assert all(w == 1.0 for w in SYNTHETIC_LC_WEIGHTS.values())
    assert SYNTHETIC_TOPO_WEIGHT == 1.0


def test_written_json_round_trips_locked_shape(tmp_path):
    """sample_weights.json has the locked top-level keys and source=synthetic."""
    out = write_sample_weights(tmp_path, "azgaar_0042.geojson")
    assert out == tmp_path / "sample_weights.json"
    assert out.exists()

    data = json.loads(out.read_text())
    assert set(data) == {
        "land_cover_weights",
        "topography_weight",
        "source",
        "map_file",
    }
    assert data["source"] == "synthetic"
    assert data["map_file"] == "azgaar_0042.geojson"
    assert data["topography_weight"] == 1.0
    assert set(data["land_cover_weights"]) == _CANONICAL_KEYS
    assert all(v == 1.0 for v in data["land_cover_weights"].values())
