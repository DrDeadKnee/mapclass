"""
Build the H3 hex soft-label dataset for the terrain prior (Phase A).

Per region: fetch ESA WorldCover + Copernicus DEM onto a coarse WGS84 grid,
map to composite terrain classes, aggregate per H3 hex, and write one .npz:

  <out>/<name>_res<h3_res>.npz
    cells    (n,) uint64          — H3 cell ids
    counts   (n, 25) uint32       — composite-class pixel counts (soft label
                                    = counts / counts.sum(axis=1))
    n_pixels (n,) uint32          — total grid pixels per cell (coverage)
  <out>/<name>_res<h3_res>.meta.json — bbox, resolutions, class names

Usage:
  python scripts/hexprior/build_hex_dataset.py \
      --name iberia --bbox -10 36 3 44 \
      --out-dir gs://mapclass-training-northeast1/data/hexprior

--out-dir accepts a local path or a gs:// prefix.
"""

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

# Allow running as `python scripts/hexprior/build_hex_dataset.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from historical.dem import fetch_topo
from historical.worldcover import fetch_worldcover
from hexprior.aggregate import (
    DEFAULT_GRID_RES_DEG,
    DEFAULT_H3_RES,
    aggregate_counts,
    cells_for_grid,
    open_grid_dataset,
)
from hexprior.labels import COMPOSITE_NAMES, composite_from_rasters


def build_region(
    bbox: tuple[float, float, float, float],
    h3_res: int = DEFAULT_H3_RES,
    grid_res_deg: float = DEFAULT_GRID_RES_DEG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fetch, classify, and aggregate one region. Returns aggregate_counts output."""
    with open_grid_dataset(bbox, grid_res_deg) as grid_ds:
        print(f"  grid {grid_ds.height}x{grid_ds.width} @ {grid_res_deg} deg")
        lc = fetch_worldcover(grid_ds, bbox)
        topo = fetch_topo(grid_ds, bbox, water_mask=(lc == 0))
    composite = composite_from_rasters(lc, topo)
    cells = cells_for_grid(bbox, h3_res, grid_res_deg)
    return aggregate_counts(cells, composite)


def _write_bytes(dest: str, data: bytes) -> None:
    if dest.startswith("gs://"):
        import gcsfs
        fs = gcsfs.GCSFileSystem()
        fs.pipe_file(dest[len("gs://"):], data)
    else:
        path = Path(dest)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def save_region(
    out_dir: str,
    name: str,
    bbox: tuple[float, float, float, float],
    unique_cells: np.ndarray,
    counts: np.ndarray,
    n_pixels: np.ndarray,
    h3_res: int,
    grid_res_deg: float,
) -> str:
    """Write the npz + meta.json pair; returns the npz destination."""
    stem = f"{name}_res{h3_res}"
    npz_dest = f"{out_dir.rstrip('/')}/{stem}.npz"

    buf = io.BytesIO()
    np.savez_compressed(buf, cells=unique_cells, counts=counts, n_pixels=n_pixels)
    _write_bytes(npz_dest, buf.getvalue())

    meta = {
        "name": name,
        "bbox_wsen": list(bbox),
        "h3_res": h3_res,
        "grid_res_deg": grid_res_deg,
        "n_cells": int(unique_cells.size),
        "composite_classes": COMPOSITE_NAMES,
        "worldcover": "ESA WorldCover 10m v200 (2021)",
        "dem": "Copernicus DEM GLO-30",
    }
    _write_bytes(f"{out_dir.rstrip('/')}/{stem}.meta.json",
                 json.dumps(meta, indent=2).encode())
    return npz_dest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="region identifier (filename stem)")
    p.add_argument("--bbox", required=True, nargs=4, type=float,
                   metavar=("W", "S", "E", "N"), help="WGS84 bounding box")
    p.add_argument("--h3-res", type=int, default=DEFAULT_H3_RES)
    p.add_argument("--grid-res", type=float, default=DEFAULT_GRID_RES_DEG,
                   help="sampling grid resolution in degrees")
    p.add_argument("--out-dir", default="gs://mapclass-training-northeast1/data/hexprior")
    args = p.parse_args()

    bbox = tuple(args.bbox)
    print(f"Building hex dataset for {args.name} bbox={bbox} h3_res={args.h3_res}")
    unique_cells, counts, n_pixels = build_region(bbox, args.h3_res, args.grid_res)

    valid_frac = counts.sum() / max(1, int(n_pixels.sum()))
    print(f"  {unique_cells.size} hexes, {int(n_pixels.sum())} pixels, "
          f"{valid_frac:.1%} valid")

    dest = save_region(args.out_dir, args.name, bbox,
                       unique_cells, counts, n_pixels, args.h3_res, args.grid_res)
    print(f"  wrote {dest}")


if __name__ == "__main__":
    main()
