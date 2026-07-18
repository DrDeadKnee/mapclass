"""
Select and batch-build training regions for the hex terrain prior.

Ranks 1° cells by the class-diversity score from satellite/coverage.py, then
greedily picks cells subject to a minimum spacing (so one high-entropy area,
e.g. Lake Victoria, cannot dominate the training set) and optional exclusion
bboxes (regions already built, e.g. iberia).

Usage:
  # dry run — print the picks
  python scripts/hexprior/select_regions.py --n 16 --exclude-bbox -10 36 3 44

  # build each picked region and upload
  python scripts/hexprior/select_regions.py --n 16 --exclude-bbox -10 36 3 44 \
      --build --out-dir gs://mapclass-training-northeast1/data/hexprior
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satellite.coverage import pick_regions
from hexprior.build_hex_dataset import build_region, save_region
from hexprior.aggregate import DEFAULT_GRID_RES_DEG, DEFAULT_H3_RES

# Rank pool depth: deep enough that spacing/exclusion filters never exhaust it.
_POOL_SIZE = 500


def _center(bbox: list[float]) -> tuple[float, float]:
    w, s, e, n = bbox
    return (w + e) / 2.0, (s + n) / 2.0


def _in_bbox(lon: float, lat: float, bbox: tuple[float, float, float, float]) -> bool:
    w, s, e, n = bbox
    return w <= lon <= e and s <= lat <= n


def spaced_pick(
    ranked: list[dict],
    n: int,
    min_spacing_deg: float = 3.0,
    exclude_bboxes: list[tuple[float, float, float, float]] | None = None,
) -> list[dict]:
    """
    Greedy pick over a score-ranked region list: take each region unless its
    center is within ``min_spacing_deg`` (Chebyshev distance) of an already
    picked region or inside an exclusion bbox. Preserves ranking order.
    """
    picked: list[dict] = []
    centers: list[tuple[float, float]] = []
    for region in ranked:
        lon, lat = _center(region["bbox"])
        if any(_in_bbox(lon, lat, b) for b in exclude_bboxes or []):
            continue
        if any(max(abs(lon - plon), abs(lat - plat)) < min_spacing_deg
               for plon, plat in centers):
            continue
        picked.append(region)
        centers.append((lon, lat))
        if len(picked) >= n:
            break
    return picked


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=16, help="number of regions to pick")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-spacing", type=float, default=3.0,
                   help="minimum center-to-center spacing in degrees (Chebyshev)")
    p.add_argument("--exclude-bbox", nargs=4, type=float, action="append",
                   metavar=("W", "S", "E", "N"), default=[],
                   help="skip cells inside this bbox (repeatable)")
    p.add_argument("--build", action="store_true",
                   help="build + upload each picked region (default: print only)")
    p.add_argument("--h3-res", type=int, default=DEFAULT_H3_RES)
    p.add_argument("--grid-res", type=float, default=DEFAULT_GRID_RES_DEG)
    p.add_argument("--out-dir", default="gs://mapclass-training-northeast1/data/hexprior")
    args = p.parse_args()

    ranked = pick_regions(n=_POOL_SIZE, seed=args.seed)
    picked = spaced_pick(ranked, args.n, args.min_spacing,
                         [tuple(b) for b in args.exclude_bbox])

    for r in picked:
        print(f"{r['region_id']}  bbox={r['bbox']}  score={r['diversity_score']:.3f}")
    if len(picked) < args.n:
        print(f"Warning: only {len(picked)}/{args.n} regions satisfied the "
              f"spacing/exclusion constraints (pool={_POOL_SIZE})")

    if not args.build:
        return

    for i, r in enumerate(picked, 1):
        bbox = tuple(r["bbox"])
        print(f"[{i}/{len(picked)}] building {r['region_id']} bbox={bbox}")
        unique_cells, counts, n_pixels = build_region(bbox, args.h3_res, args.grid_res)
        valid_frac = counts.sum() / max(1, int(n_pixels.sum()))
        print(f"  {unique_cells.size} hexes, {valid_frac:.1%} valid")
        dest = save_region(args.out_dir, r["region_id"], bbox, unique_cells,
                           counts, n_pixels, args.h3_res, args.grid_res)
        print(f"  wrote {dest}")


if __name__ == "__main__":
    main()
