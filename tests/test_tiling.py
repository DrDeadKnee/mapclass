"""
Offline geometry/weight tests for ``scripts/tiling.py`` (plan 02-05, D-06..D-09).

The nested-pyramid tiler is the only genuinely novel geometry in Phase 2:
each pyramid is 1x896 + 4x448 + 16x224 in strict 2x2 spatial nesting, pyramids
step across the source map by stride 448 (50% top-scale overlap), and a pyramid
whose 896 footprint is >50% off the source map is dropped (D-09).

Every test is fully offline: small synthetic PNGs written under pytest tmp_path,
no network. ``sample_weights.json`` propagates byte-identically into each
pyramid subdir, and the synthetic train/test boundary (EVAL-01) survives tiling.
"""

import io
import itertools
import json
import sys
from pathlib import Path

import fsspec
import pytest
from PIL import Image

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from tiling import (
    PYRAMID_896,
    PYRAMID_448,
    PYRAMID_224,
    PYRAMID_STRIDE,
    enumerate_pyramids,
    tile,
)
from gcs_io import _GCSWriter

WEIGHTS_BLOB = json.dumps(
    {
        "land_cover_weights": {"trees": 0.3, "cropland": 0.15},
        "topography_weight": 1.0,
        "source": "synthetic",
        "map_file": "fixture",
    },
    indent=2,
)


def _make_map_dir(base, width, height):
    """Write a complete per-map dir (image/lc/topo PNG + sample_weights.json)."""
    base.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), (10, 20, 30)).save(base / "image.png")
    Image.new("L", (width, height), 1).save(base / "land_cover.png")
    Image.new("L", (width, height), 2).save(base / "topography.png")
    (base / "sample_weights.json").write_text(WEIGHTS_BLOB)
    return base


def _load_manifest(pyramid_dir):
    return json.loads((pyramid_dir / "pyramid.json").read_text())


def _box(entry):
    """(x0, y0, x1, y1) from a manifest tile entry."""
    return (
        entry["x"],
        entry["y"],
        entry["x"] + entry["size"],
        entry["y"] + entry["size"],
    )


def _contains(outer, inner):
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


def _disjoint(a, b):
    return a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]


# A 1792x1792 map: 896 pyramid origins on a stride-448 grid sit at
# {0, 448, 896, 1344} per axis. Origins 0/448/896 are fully on-map; the
# 1344 origin's 896 footprint is exactly 50% off (kept by D-09 — "<=50% off
# IS written"). So the full grid is 4x4 = 16 pyramids, the first (origin 0,0)
# being a fully-on-map pyramid used for the nesting/overlap assertions.
FULL_W = FULL_H = 896 + 2 * PYRAMID_STRIDE  # 1792


def test_pyramid_tile_count(tmp_path):
    """A source large enough for >=1 full pyramid yields exactly 21 tiles."""
    src = _make_map_dir(tmp_path / "map", FULL_W, FULL_H)
    out = tile(src)

    pyramids = sorted(p for p in out.iterdir() if p.is_dir())
    assert pyramids, "expected at least one pyramid"
    for pdir in pyramids:
        m = _load_manifest(pdir)
        sizes = sorted(t["size"] for t in m["tiles"])
        assert sizes.count(PYRAMID_896) == 1
        assert sizes.count(PYRAMID_448) == 4
        assert sizes.count(PYRAMID_224) == 16
        assert len(m["tiles"]) == 21
        for t in m["tiles"]:
            assert (pdir / t["image"]).exists()
            assert (pdir / t["land_cover"]).exists()
            assert (pdir / t["topography"]).exists()


def test_nested_alignment(tmp_path):
    """448 children fully inside the 896; 224 inside their 448; siblings tile parent."""
    src = _make_map_dir(tmp_path / "map", FULL_W, FULL_H)
    out = tile(src)

    pdir = next(p for p in sorted(out.iterdir()) if p.is_dir())
    m = _load_manifest(pdir)
    by_id = {t["id"]: t for t in m["tiles"]}

    parent896 = next(t for t in m["tiles"] if t["size"] == PYRAMID_896)
    box896 = _box(parent896)

    children448 = [by_id[i] for i in parent896["children"]]
    assert len(children448) == 4
    # each 448 fully contained in the 896
    for c in children448:
        assert _contains(box896, _box(c))
    # the 4 448 exactly tile the 896 (sum of areas == area, pairwise disjoint)
    assert sum((c["size"] ** 2) for c in children448) == PYRAMID_896**2
    for a, b in itertools.combinations(children448, 2):
        assert _disjoint(_box(a), _box(b))

    for c448 in children448:
        box448 = _box(c448)
        grand = [by_id[i] for i in c448["children"]]
        assert len(grand) == 4
        for g in grand:
            assert _contains(box448, _box(g))
        assert sum((g["size"] ** 2) for g in grand) == PYRAMID_448**2
        for a, b in itertools.combinations(grand, 2):
            assert _disjoint(_box(a), _box(b))


def test_stride_448(tmp_path):
    """Adjacent pyramid 896 origins differ by exactly 448 px across the source."""
    origins = enumerate_pyramids(FULL_W, FULL_H)
    xs = sorted({x for x, y in origins})
    ys = sorted({y for x, y in origins})
    # First origin anchors the source top-left; >1 origin per axis here.
    assert xs[0] == 0 and ys[0] == 0
    assert len(xs) > 1 and len(ys) > 1
    # Adjacent pyramid 896 origins differ by exactly the locked stride (D-08).
    for a, b in zip(xs, xs[1:]):
        assert b - a == PYRAMID_STRIDE
    for a, b in zip(ys, ys[1:]):
        assert b - a == PYRAMID_STRIDE


def test_no_intra_pyramid_overlap(tmp_path):
    """The 4 448 siblings are pairwise disjoint; same for the 16 224."""
    src = _make_map_dir(tmp_path / "map", FULL_W, FULL_H)
    out = tile(src)
    pdir = next(p for p in sorted(out.iterdir()) if p.is_dir())
    m = _load_manifest(pdir)

    boxes448 = [_box(t) for t in m["tiles"] if t["size"] == PYRAMID_448]
    for a, b in itertools.combinations(boxes448, 2):
        assert _disjoint(a, b)

    boxes224 = [_box(t) for t in m["tiles"] if t["size"] == PYRAMID_224]
    for a, b in itertools.combinations(boxes224, 2):
        assert _disjoint(a, b)


def test_edge_drop(tmp_path):
    """A pyramid >50% off the source map is dropped; <=50% off is kept (D-09).

    Off-fraction is area-based: ``1 - (on_w*on_h)/(896*896)``. Height is kept
    tall so the y=0 row is fully on vertically (on_h=896); only the
    horizontal axis varies the off-fraction.
    """
    # width 1244, height FULL_H (y=0 row fully on vertically):
    #   x=448  -> on_w = 1244-448 = 796 -> off = 1 - 796/896 = 0.11  -> KEEP
    #   x=896  -> on_w = 1244-896 = 348 -> off = 1 - 348/896 = 0.61  -> DROP
    origins = enumerate_pyramids(1244, FULL_H)
    assert (PYRAMID_STRIDE, 0) in origins          # 11% off -> kept
    assert (2 * PYRAMID_STRIDE, 0) not in origins  # 61% off -> dropped (D-09)

    # Exactly-50%-off edge case: width 896 so the x=448 origin's footprint
    # [448,1344) overlaps the map only on [448,896): on_w = 448 exactly,
    # off = 1 - 448/896 = 0.50 -> EXACTLY 50% off IS written (D-09 boundary).
    origins2 = enumerate_pyramids(896, FULL_H)
    assert (PYRAMID_STRIDE, 0) in origins2          # exactly 50% off -> kept
    assert (2 * PYRAMID_STRIDE, 0) not in origins2  # 100% off -> dropped


def test_weight_propagation(tmp_path):
    """Every pyramid subdir has a sample_weights.json byte-identical to source."""
    src = _make_map_dir(tmp_path / "map", FULL_W, FULL_H)
    out = tile(src)

    source_blob = (src / "sample_weights.json").read_bytes()
    pyramids = [p for p in out.iterdir() if p.is_dir()]
    assert pyramids
    for pdir in pyramids:
        assert (pdir / "sample_weights.json").read_bytes() == source_blob


def test_split_subtree_preserved(tmp_path):
    """A synthetic map under test/ produces its pyramids under test/ only (EVAL-01)."""
    synthetic = tmp_path / "synthetic"
    test_map = _make_map_dir(synthetic / "test" / "europe_3__flat", FULL_W, FULL_H)
    train_map = _make_map_dir(synthetic / "train" / "europe_1__flat", FULL_W, FULL_H)

    test_out = tile(test_map)
    train_out = tile(train_map)

    # test pyramids live under .../test/..., never under .../train/...
    assert "test" in test_out.parts
    assert "train" not in test_out.parts
    assert str(test_out).startswith(str(synthetic / "test"))

    assert "train" in train_out.parts
    assert "test" not in train_out.parts
    assert str(train_out).startswith(str(synthetic / "train"))

    # No cross-subtree leakage: test pyramids are not written under train/.
    train_pyr_dirs = {p.name for p in train_out.iterdir() if p.is_dir()}
    test_pyr_dirs = {p.name for p in test_out.iterdir() if p.is_dir()}
    assert train_pyr_dirs and test_pyr_dirs


# ---------------------------------------------------------------------------
# Task 1 (02-02): _GCSWriter out_root + ThreadPoolExecutor tests
# ---------------------------------------------------------------------------

def _local_gcs_writer(tmp_path, rel_prefix: str) -> "_GCSWriter":
    """Return a _GCSWriter backed by a local fsspec filesystem."""
    fs = fsspec.filesystem("file")
    return _GCSWriter(fs, str(tmp_path / rel_prefix))


def test_tile_to_gcs_writer(tmp_path):
    """tile() with a _GCSWriter out_root produces complete pyramid dirs.

    Structure must be identical to the local-Path output: image/land_cover/
    topography PNGs + pyramid.json + sample_weights.json in each subdir.
    Tests RW-01: _GCSWriter-aware tile() with 32-thread write pool.
    """
    src = _make_map_dir(tmp_path / "map", FULL_W, FULL_H)
    out_writer = _local_gcs_writer(tmp_path, "train")
    result = tile(src, out_root=out_writer)

    # result must be the _GCSWriter (or a Path-like representation)
    # Inspect the local fs at the prefix directly
    train_dir = tmp_path / "train"
    assert train_dir.exists(), "out_root directory not created"

    pyramid_dirs = [p for p in train_dir.iterdir() if p.is_dir()]
    assert pyramid_dirs, "no pyramid dirs produced under _GCSWriter out_root"

    for pdir in pyramid_dirs:
        # Check pyramid.json exists
        assert (pdir / "pyramid.json").exists(), f"missing pyramid.json in {pdir.name}"
        manifest = json.loads((pdir / "pyramid.json").read_text())
        assert len(manifest["tiles"]) == 21, f"expected 21 tiles, got {len(manifest['tiles'])}"

        # Check sample_weights.json exists and is byte-identical to source
        assert (pdir / "sample_weights.json").exists(), \
            f"missing sample_weights.json in {pdir.name}"
        assert (pdir / "sample_weights.json").read_bytes() == \
            (src / "sample_weights.json").read_bytes(), \
            "sample_weights.json content mismatch"

        # Check all tile PNGs exist
        for t in manifest["tiles"]:
            assert (pdir / t["image"]).exists(), \
                f"missing image tile {t['image']} in {pdir.name}"
            assert (pdir / t["land_cover"]).exists(), \
                f"missing land_cover tile {t['land_cover']} in {pdir.name}"
            assert (pdir / t["topography"]).exists(), \
                f"missing topography tile {t['topography']} in {pdir.name}"


def test_tile_local_path_unchanged(tmp_path):
    """tile() with out_root=None still produces local pyramids (no regression).

    Verifies that the _GCSWriter branch does not break the existing local-Path
    code path.
    """
    src = _make_map_dir(tmp_path / "map", FULL_W, FULL_H)
    out = tile(src, out_root=None)

    # Should be a Path under src/pyramids (default)
    assert isinstance(out, Path), "expected Path for local out_root=None"
    pyramid_dirs = [p for p in out.iterdir() if p.is_dir()]
    assert pyramid_dirs, "no pyramid dirs produced for local out_root=None"
    for pdir in pyramid_dirs:
        assert (pdir / "pyramid.json").exists()
        assert (pdir / "sample_weights.json").exists()
