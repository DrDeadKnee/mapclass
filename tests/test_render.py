"""
Synthetic render / per-(source×style) output tests for the synthetic pipeline.

Filled by plan 02-03 (synthetic pipeline):
  - test_output_dimensions_match      — image/label dims agree within a style dir
  - test_per_source_style_dirs        — A4 option (a): one dir per (source×style)
  - test_shared_label_byte_identical  — labels shared byte-for-byte across styles
"""

from PIL import Image

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
