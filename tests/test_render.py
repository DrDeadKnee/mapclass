"""
Synthetic render / per-(source×style) output tests for the synthetic pipeline.

Filled by plan 02-03 (synthetic pipeline):
  - test_output_dimensions_match      — image/label dims agree within a style dir
  - test_per_source_style_dirs        — A4 option (a): one dir per (source×style)
  - test_shared_label_byte_identical  — labels shared byte-for-byte across styles
  - test_synthetic_topo_locked_boundaries — ROADMAP SC#3 height-class gate
"""

from PIL import Image

from biome_mapping import H_SEA_LEVEL, TOPO_IDX, h_to_topo, normalize_land_h
from build_dataset import build_one_source


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
