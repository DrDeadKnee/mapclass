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

RW-01: per-pyramid tile writes run under ThreadPoolExecutor(max_workers=32).
_GCSWriter instances are NOT thread-safe — one instance is created per pyramid
write task; the underlying gcsfs.GCSFileSystem IS thread-safe.
"""

import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

import sys as _sys
_HERE = Path(__file__).parent
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))

from gcs_io import _GCSWriter, GCS_PROJECT

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


def _write_pyramid(pdir, tiles, img, lc, topo, weights_blob, manifest):
    """Write one pyramid's tiles + manifest + weights to pdir (_GCSWriter or Path).

    Called from the ThreadPoolExecutor worker. pdir is a fresh per-pyramid
    _GCSWriter (or Path) — not shared across threads.
    """
    pdir.mkdir(parents=True, exist_ok=True)

    for t in tiles:
        x, y, s = t["x"], t["y"], t["size"]
        box = (x, y, x + s, y + s)

        buf = io.BytesIO()
        img.crop(box).save(buf, format="PNG")
        (pdir / t["image"]).write_bytes(buf.getvalue())

        buf = io.BytesIO()
        lc.crop(box).save(buf, format="PNG")
        (pdir / t["land_cover"]).write_bytes(buf.getvalue())

        buf = io.BytesIO()
        topo.crop(box).save(buf, format="PNG")
        (pdir / t["topography"]).write_bytes(buf.getvalue())

    (pdir / "pyramid.json").write_text(json.dumps(manifest, indent=2))

    # Per-source loss weights propagate UNCHANGED to every pyramid of the map
    # (D-claude-discretion; per-source values locked upstream).
    (pdir / _WEIGHTS_FILE).write_bytes(weights_blob)

    # WR-01: a runtime data-integrity check, NOT an assert — `assert` is
    # stripped under `python -O` (exactly the production batch-build scenario
    # where weight-propagation corruption matters).
    # Integrity check only applicable for local Path (GCS pipe_file is atomic
    # — no partial-write risk; round-trip read would add unnecessary latency).
    if isinstance(pdir, Path):
        if (pdir / _WEIGHTS_FILE).read_bytes() != weights_blob:
            raise RuntimeError(
                f"weight propagation corrupted for pyramid {pdir.name}"
            )


def tile(map_dir, out_root=None, max_workers: int = 32):
    """Decompose a completed per-map dir into nested pyramids.

    Parameters
    ----------
    map_dir : a directory containing ``image.png`` + ``land_cover.png`` +
              ``topography.png`` + ``sample_weights.json``.
    out_root : where to write the pyramid tree. Accepts:
               - None: defaults to ``<map_dir>/pyramids`` (local Path, D-08)
               - a ``_GCSWriter`` instance: write directly to GCS (RW-01)
               - a ``str`` starting with ``gs://``: constructs a _GCSWriter
               - a ``str`` or ``Path`` (local): wraps in Path
    max_workers : thread pool size for concurrent per-pyramid writes (RW-01b).

    Returns the pyramid-tree root (_GCSWriter or Path). If any required
    per-map file is missing the map is skipped (logged) and the root is
    returned empty, so a partially-failed map never produces a corrupt
    pyramid tree (T-02-16).
    """
    map_dir = Path(map_dir)

    # Branch on out_root type — RW-01 _GCSWriter support.
    if isinstance(out_root, _GCSWriter):
        out = out_root
    elif isinstance(out_root, str) and out_root.startswith("gs://"):
        # Lazy-import gcsfs — same pattern as gcs_checkpoint.py lines 36-39.
        try:
            import gcsfs  # type: ignore[import]
        except ModuleNotFoundError:
            raise ImportError(
                "gcsfs is required to write pyramids to GCS. "
                "Install with: pip install gcsfs"
            )
        bare = out_root[len("gs://"):]
        fs = gcsfs.GCSFileSystem(project=GCS_PROJECT)
        out = _GCSWriter(fs, bare)
    elif out_root is not None:
        out = Path(out_root)
    else:
        out = map_dir / "pyramids"

    missing = [f for f in _REQUIRED_FILES if not (map_dir / f).exists()]
    if missing:
        print(f"  SKIP tiling {map_dir}: missing {', '.join(missing)}")
        out.mkdir(parents=True, exist_ok=True)
        return out

    img = Image.open(map_dir / "image.png")
    lc = Image.open(map_dir / "land_cover.png")
    topo = Image.open(map_dir / "topography.png")
    width, height = img.size
    # Pre-load image data into memory before passing to thread pool.
    # PIL lazily decodes PNG files; concurrent crop() calls on lazily-loaded
    # images are not thread-safe. load() forces full decode once on the main
    # thread so workers only call crop() on fully-decoded in-memory data.
    img.load()
    lc.load()
    topo.load()

    out.mkdir(parents=True, exist_ok=True)
    weights_blob = (map_dir / _WEIGHTS_FILE).read_bytes()

    origins = enumerate_pyramids(width, height)

    # Build list of (pdir, tiles, manifest) for all pyramids.
    pyramid_tasks = []
    for ox, oy in origins:
        pid = _pyramid_id(ox, oy)
        pdir = out / pid
        tiles = _pyramid_tiles(ox, oy)
        manifest = {
            "pyramid_id": pid,
            "origin": [ox, oy],
            "source_map": str(map_dir),
            "source_size": [width, height],
            "scales": [PYRAMID_896, PYRAMID_448, PYRAMID_224],
            "stride": PYRAMID_STRIDE,
            "tiles": tiles,
        }
        pyramid_tasks.append((pdir, tiles, manifest))

    # RW-01b: 32-thread write pool — mirror build_historical_dataset.py L138-144.
    # Each pyramid gets its own pdir (_GCSWriter or Path) — not shared across
    # threads (_GCSWriter instances are not thread-safe; gcsfs.GCSFileSystem is).
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_write_pyramid, pdir, tiles, img, lc, topo,
                        weights_blob, manifest): pid
            for pdir, tiles, manifest in pyramid_tasks
            for pid in [manifest["pyramid_id"]]
        }
        for fut in as_completed(futures):
            fut.result()  # re-raises any exception from the worker

    n = len(pyramid_tasks)
    print(f"  tiled {map_dir.name}: {n} pyramid(s) "
          f"({width}x{height}px, {n * 21} tiles)")
    return out


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python tiling.py <map_dir> [out_root]")
        sys.exit(1)
    tile(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
