"""
Shared multi-scale nested-pyramid tiler (D-06..D-09).

Consumed by all three build pipelines (historical / synthetic / satellite)
after a per-map directory completes. The geometry is the only genuinely novel
logic in Phase 2 and is exploited by Phase 3's coarse-to-fine model:

  * Each *pyramid* is 1x(896x896) + 4x(448x448) + 16x(224x224) tiles in
    strict 2x2 spatial nesting (D-07): the 896 region is tiled by 4
    non-overlapping 448 children; each 448 is tiled by 4 non-overlapping
    224 grandchildren. 21 tiles per pyramid.
  * Pyramids step across the source map by ``PYRAMID_STRIDE`` = 448 px
    (50% top-scale overlap, D-08). Same-scale siblings WITHIN a pyramid do
    not overlap.
  * A pyramid whose 896 footprint lies >50% off the source map is dropped
    (D-09). A footprint exactly 50%-or-less off IS written.

Storage layout (RESEARCH Open-Q3, Claude's discretion D-Storage): one
sub-directory per pyramid containing the 21 cropped PNG triplets, a
``pyramid.json`` manifest listing every tile path + parent->child indices,
and a byte-identical copy of the source ``sample_weights.json`` (per-source
loss weights propagate unchanged to every tile of that map).

I/O idiom copies ``scripts/label.py`` (PIL Image open/crop/save, per-map-dir).
The crop of an edge pyramid whose footprint runs partly off the source is
zero-padded by PIL beyond the image extent — acceptable for the <=50%-off
tiles D-09 keeps; the in-pyramid geometry is exact regardless of clipping.
"""

import json
import shutil
from pathlib import Path

from PIL import Image

# Locked geometry (D-07/D-08).
PYRAMID_896 = 896
PYRAMID_448 = 448
PYRAMID_224 = 224
PYRAMID_STRIDE = 448  # D-08: 50% top-scale overlap across the source map.

_REQUIRED_FILES = ("image.png", "land_cover.png", "topography.png",
                   "sample_weights.json")
_WEIGHTS_FILE = "sample_weights.json"


def enumerate_pyramids(width: int, height: int) -> list[tuple[int, int]]:
    """Return the kept 896-px pyramid origins ``(x, y)`` on a stride-448 grid.

    Origins are laid out at multiples of ``PYRAMID_STRIDE`` from the top-left
    so that the source's top-left always anchors a pyramid (D-08). A pyramid
    is dropped when more than 50% of its 896x896 footprint falls off the
    source extent (D-09); a footprint exactly 50%-or-less off is kept.
    """
    origins: list[tuple[int, int]] = []
    full_area = PYRAMID_896 * PYRAMID_896
    # +1 so an origin landing exactly at width/height (or partly off) is still
    # considered and then accepted/rejected by the >50%-off rule.
    y = 0
    while y < height:
        x = 0
        while x < width:
            on_w = max(0, min(x + PYRAMID_896, width) - x)
            on_h = max(0, min(y + PYRAMID_896, height) - y)
            on_area = on_w * on_h
            off_fraction = 1.0 - (on_area / full_area)
            if off_fraction <= 0.5:  # D-09: drop only when >50% off
                origins.append((x, y))
            x += PYRAMID_STRIDE
        y += PYRAMID_STRIDE
    return origins


def _pyramid_tiles(ox: int, oy: int) -> list[dict]:
    """Build the 21-tile manifest entries for one pyramid at origin (ox, oy).

    Strict 2x2 nesting: the 896 root, its 4 non-overlapping 448 children
    (2x2 quadrants), and each child's 4 non-overlapping 224 grandchildren.
    IDs are stable and parent->child indices are explicit so Phase 3 can map
    a 896 prediction onto its 4 448 / 16 224 sub-predictions.
    """
    tiles: list[dict] = []

    def _add(tid, x, y, size):
        tiles.append({
            "id": tid,
            "x": x,
            "y": y,
            "size": size,
            "children": [],
            "image": f"{tid}_image.png",
            "land_cover": f"{tid}_land_cover.png",
            "topography": f"{tid}_topography.png",
        })
        return tiles[-1]

    root = _add("896", ox, oy, PYRAMID_896)

    for ci, (cdx, cdy) in enumerate(
        ((0, 0), (1, 0), (0, 1), (1, 1))
    ):
        cx = ox + cdx * PYRAMID_448
        cy = oy + cdy * PYRAMID_448
        child = _add(f"448_{ci}", cx, cy, PYRAMID_448)
        root["children"].append(child["id"])

        for gi, (gdx, gdy) in enumerate(
            ((0, 0), (1, 0), (0, 1), (1, 1))
        ):
            gx = cx + gdx * PYRAMID_224
            gy = cy + gdy * PYRAMID_224
            grand = _add(f"224_{ci}_{gi}", gx, gy, PYRAMID_224)
            child["children"].append(grand["id"])

    return tiles


def _pyramid_id(ox: int, oy: int) -> str:
    """Deterministic, collision-free pyramid id from its stride-grid cell.

    Grid row/col are ``origin // PYRAMID_STRIDE`` — unique per source map, so
    per-pyramid subdirs never collide (threat T-02-17).
    """
    col = ox // PYRAMID_STRIDE
    row = oy // PYRAMID_STRIDE
    return f"py_r{row:03d}_c{col:03d}"


def tile(map_dir, out_root=None) -> Path:
    """Decompose a completed per-map dir into nested pyramids.

    Parameters
    ----------
    map_dir : a directory containing ``image.png`` + ``land_cover.png`` +
              ``topography.png`` + ``sample_weights.json``.
    out_root : where to write the pyramid tree. Defaults to
               ``<map_dir>/pyramids`` so the tiler writes INSIDE the source's
               own subtree — for synthetic this keeps every pyramid on the
               same side of the frozen train/test split (EVAL-01, T-02-15).

    Returns the pyramid-tree root directory. If any required per-map file is
    missing the map is skipped (logged) and an (empty) root is returned, so a
    partially-failed map never produces a corrupt pyramid tree (T-02-16).
    """
    map_dir = Path(map_dir)
    out = Path(out_root) if out_root is not None else map_dir / "pyramids"

    missing = [f for f in _REQUIRED_FILES if not (map_dir / f).exists()]
    if missing:
        print(f"  SKIP tiling {map_dir}: missing {', '.join(missing)}")
        out.mkdir(parents=True, exist_ok=True)
        return out

    img = Image.open(map_dir / "image.png")
    lc = Image.open(map_dir / "land_cover.png")
    topo = Image.open(map_dir / "topography.png")
    width, height = img.size

    out.mkdir(parents=True, exist_ok=True)
    weights_blob = (map_dir / _WEIGHTS_FILE).read_bytes()

    n = 0
    for ox, oy in enumerate_pyramids(width, height):
        pid = _pyramid_id(ox, oy)
        pdir = out / pid
        pdir.mkdir(parents=True, exist_ok=True)

        tiles = _pyramid_tiles(ox, oy)
        for t in tiles:
            x, y, s = t["x"], t["y"], t["size"]
            box = (x, y, x + s, y + s)
            img.crop(box).save(pdir / t["image"])
            lc.crop(box).save(pdir / t["land_cover"])
            topo.crop(box).save(pdir / t["topography"])

        manifest = {
            "pyramid_id": pid,
            "origin": [ox, oy],
            "source_map": str(map_dir),
            "source_size": [width, height],
            "scales": [PYRAMID_896, PYRAMID_448, PYRAMID_224],
            "stride": PYRAMID_STRIDE,
            "tiles": tiles,
        }
        (pdir / "pyramid.json").write_text(json.dumps(manifest, indent=2))

        # Per-source loss weights propagate UNCHANGED to every pyramid of the
        # map (D-claude-discretion; per-source values locked upstream).
        shutil.copyfile(map_dir / _WEIGHTS_FILE, pdir / _WEIGHTS_FILE)
        # WR-01: a runtime data-integrity check, NOT an assert — `assert`
        # is stripped under `python -O` (exactly the production batch-build
        # scenario where weight-propagation corruption matters).
        if (pdir / _WEIGHTS_FILE).read_bytes() != weights_blob:
            raise RuntimeError(
                f"weight propagation corrupted for pyramid {pdir.name}"
            )
        n += 1

    print(f"  tiled {map_dir.name}: {n} pyramid(s) "
          f"({width}x{height}px, {n * 21} tiles)")
    return out


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python tiling.py <map_dir> [out_root]")
        sys.exit(1)
    tile(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
