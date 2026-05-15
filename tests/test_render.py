"""
Synthetic render / per-(source×style) output tests for the synthetic pipeline.

Filled by plan 02-03 (synthetic pipeline):
  - test_output_dimensions_match      — image/label dims agree within a style dir
  - test_per_source_style_dirs        — A4 option (a): one dir per (source×style)
  - test_shared_label_byte_identical  — labels shared byte-for-byte across styles
  - test_synthetic_topo_locked_boundaries — ROADMAP SC#3 height-class gate
"""

import json

from PIL import Image

from biome_mapping import H_SEA_LEVEL, TOPO_IDX, h_to_topo, normalize_land_h
from build_dataset import build_one_source
from label import make_label_arrays
from render import render_one


def test_output_dimensions_match(sample_azgaar_geojson, tmp_path):
    """image.png matches land_cover.png / topography.png dims within a style dir."""
    dirs = build_one_source(sample_azgaar_geojson, tmp_path, styles=("flat",))
    d = dirs[0]
    img = Image.open(d / "image.png")
    lc = Image.open(d / "land_cover.png")
    topo = Image.open(d / "topography.png")
    assert lc.size == topo.size
    assert img.size == lc.size


def test_per_source_style_dirs(sample_azgaar_geojson, tmp_path):
    """A4 option (a): two styles → two <id>__<style>/ dirs, each fully populated."""
    dirs = build_one_source(
        sample_azgaar_geojson, tmp_path, styles=("flat", "illustrated")
    )
    assert len(dirs) == 2
    names = sorted(d.name for d in dirs)
    assert names == ["azgaar_sample__flat", "azgaar_sample__illustrated"]
    for d in dirs:
        for fname in (
            "image.png",
            "land_cover.png",
            "topography.png",
            "sample_weights.json",
        ):
            assert (d / fname).is_file(), f"{fname} missing in {d.name}"


def test_shared_label_byte_identical(sample_azgaar_geojson, tmp_path):
    """land_cover.png is byte-identical across the two style dirs of one source."""
    dirs = build_one_source(
        sample_azgaar_geojson, tmp_path, styles=("flat", "illustrated")
    )
    a, b = sorted(dirs, key=lambda p: p.name)
    assert (a / "land_cover.png").read_bytes() == (b / "land_cover.png").read_bytes()
    assert (a / "topography.png").read_bytes() == (b / "topography.png").read_bytes()


# ---------------------------------------------------------------------------
# CR-04 regression: GeoJSON (RFC 7946) permits a third position element
# (elevation). A 3-element coordinate must NOT crash the rasterization
# tuple-unpack and abort the entire source. A malformed Polygon whose
# coordinates is a scalar/dict must be skipped, not deferred to a crash.
# ---------------------------------------------------------------------------


def _write_geojson(tmp_path, features):
    fc = {"type": "FeatureCollection", "features": features}
    path = tmp_path / "src.geojson"
    path.write_text(json.dumps(fc))
    return path


def test_3d_coordinates_do_not_abort_label(tmp_path):
    """A feature with [lon, lat, elevation] positions rasterizes (CR-04)."""
    path = _write_geojson(
        tmp_path,
        [
            {
                "type": "Feature",
                "properties": {"biome": 6, "height": 35},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[0, 0, 5], [100, 0, 5], [100, 100, 5], [0, 100, 5], [0, 0, 5]]
                    ],
                },
            }
        ],
    )
    # Must not raise ValueError: too many values to unpack.
    lc_img, topo_img = make_label_arrays(path)
    assert lc_img.size == topo_img.size
    assert lc_img.size[0] > 0 and lc_img.size[1] > 0
    # The 3D feature was actually drawn (not silently dropped): some pixel
    # carries its land cover class rather than the NODATA fill (255).
    assert min(lc_img.getdata()) < 255


def test_3d_coordinates_do_not_abort_render(tmp_path):
    """render_one tolerates [lon, lat, elevation] positions (CR-04)."""
    path = _write_geojson(
        tmp_path,
        [
            {
                "type": "Feature",
                "properties": {"biome": 1, "height": 80},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[0, 0, 9], [50, 0, 9], [50, 50, 9], [0, 50, 9], [0, 0, 9]]
                    ],
                },
            }
        ],
    )
    img = render_one(path, "flat")
    assert img.size[0] > 0 and img.size[1] > 0


def test_malformed_polygon_scalar_coords_skipped(tmp_path):
    """A Polygon whose coordinates is a scalar is skipped, not crashed (CR-04).

    A valid sibling feature must still rasterize so one bad feature does
    not abort the whole source (T-02-09 per-source resilience contract).
    """
    path = _write_geojson(
        tmp_path,
        [
            {
                "type": "Feature",
                "properties": {"biome": 6, "height": 35},
                "geometry": {"type": "Polygon", "coordinates": 42},
            },
            {
                "type": "Feature",
                "properties": {"biome": 1, "height": 80},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [40, 0], [40, 40], [0, 40], [0, 0]]],
                },
            },
        ],
    )
    lc_img, topo_img = make_label_arrays(path)
    # The valid sibling was drawn despite the malformed feature.
    assert min(lc_img.getdata()) < 255


# ---------------------------------------------------------------------------
# Task 2 — ROADMAP SC#3 LOCKED topography height-class boundaries.
#
# flat ≤20 / hilly 20–55 / mountainous >55 over the normalized [0,100] domain.
# This is the SC#3 verification gate: it asserts biome_mapping.h_to_topo
# directly with concrete values, NOT an image-dimension proxy. If h_to_topo
# drifts off the locked cuts this test FAILS — do not loosen it to match an
# off-by-one implementation; fix the source instead.
# ---------------------------------------------------------------------------


def _h_for_norm(norm: float) -> float:
    """Inverse of normalize_land_h: the raw Azgaar h whose normalized value == norm."""
    return norm / 100 * (100 - H_SEA_LEVEL) + H_SEA_LEVEL


def test_synthetic_topo_locked_boundaries():
    flat = TOPO_IDX["flat"]
    hilly = TOPO_IDX["hilly"]
    mountainous = TOPO_IDX["mountainous"]

    # --- flat: norm ≤ 20 (20 inclusive upper) ---
    assert h_to_topo(int(round(_h_for_norm(0)))) == flat
    h20 = _h_for_norm(20)
    # sanity: this raw h normalizes to exactly 20 (within float tolerance)
    assert abs(normalize_land_h(int(round(h20))) - 20) < 1e-6
    assert h_to_topo(int(round(h20))) == flat          # norm == 20 → flat

    # explicit on-the-cut raw values: h=36 → norm 20.0 (flat), h=64 → norm 55.0
    # (hilly, NOT mountainous — the float-precision regression guard)
    assert normalize_land_h(36) == 20.0 or abs(normalize_land_h(36) - 20.0) < 1e-6
    assert h_to_topo(36) == flat
    assert h_to_topo(64) == hilly
    assert h_to_topo(int(round(_h_for_norm(19)))) == flat  # just below 20 → flat

    # --- hilly: 20 < norm ≤ 55 (55 inclusive upper) ---
    assert h_to_topo(int(round(_h_for_norm(21)))) == hilly  # just above 20 → hilly
    assert h_to_topo(int(round(_h_for_norm(55)))) == hilly  # norm == 55 → hilly
    assert h_to_topo(int(round(_h_for_norm(54)))) == hilly  # just below 55 → hilly

    # --- mountainous: norm > 55 (strict) ---
    assert h_to_topo(int(round(_h_for_norm(56)))) == mountainous   # just above 55
    assert h_to_topo(int(round(_h_for_norm(100)))) == mountainous  # top of domain

    # --- water (h < H_SEA_LEVEL) → no topography class (WATER_TOPO sentinel) ---
    assert h_to_topo(H_SEA_LEVEL - 1) is None
    assert h_to_topo(0) is None
