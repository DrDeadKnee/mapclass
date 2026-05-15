"""
Assemble the historical map training dataset.

Two sub-commands:

  search  — Query David Rumsey, download georeferenced maps to raw_dir,
             write unregistered_manifest.json for maps needing manual GCPs.

  build   — Process all GeoTIFFs in raw_dir into training samples.

  full    — search + build in one pass.

Usage
-----
  python build_historical_dataset.py search [--raw-dir RAW] [--max-maps N]
  python build_historical_dataset.py build  [--raw-dir RAW] [--out-dir OUT] [--workers N]
  python build_historical_dataset.py full   [--raw-dir RAW] [--out-dir OUT] [--max-maps N] [--workers N]

Defaults:
  --raw-dir   data/historical/raw
  --out-dir   data/historical/dataset
  --max-maps  500
  --workers   1
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Adjust sys.path so that 'historical' package and sibling scripts are importable
# when running from the repo root or from the scripts/ directory.
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from historical import label as hist_label
from historical import rumsey


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def cmd_search(raw_dir: Path, max_maps: int) -> None:
    geo_dir = raw_dir / "georeferenced"
    manifest_path = raw_dir / "unregistered_manifest.json"

    print("=== David Rumsey search ===")
    items = rumsey.search_maps(max_results=max_maps)

    # D-05: per-reason drop accounting. "A clean pipeline that quietly throws
    # away most of the data is not job done" — every reason is surfaced loudly.
    drops = {
        "ok": 0,
        "out_of_scale": 0,
        "not_in_allmaps": 0,
        "gcps_insufficient": 0,
        "download_failed": 0,
    }
    unregistered: list[tuple[dict, str]] = []
    for item in items:
        _, status = rumsey.download_georeferenced(item, geo_dir)
        drops[status] = drops.get(status, 0) + 1
        # out_of_scale maps are dropped SILENTLY (D-04 — not in the manifest);
        # ok maps are already georeferenced. Only the two registration-needed
        # reasons go to the v2 hand-off manifest.
        if status in ("not_in_allmaps", "gcps_insufficient"):
            unregistered.append((item, status))

    rumsey.emit_manifest(unregistered, manifest_path)

    total = len(items)
    print(f"\nSearch summary:")
    for reason, n in drops.items():
        print(f"  {reason}: {n}")
    print(f"  manifest (needs GCPs): {len(unregistered)}  → {manifest_path}")

    if total > 0 and drops["out_of_scale"] / total > 0.5:
        print(
            f"\n⚠ out_of_scale rate >50% "
            f"({drops['out_of_scale']}/{total}) — the scale window may be "
            f"hiding most of the dataset; Phase 2 is not done (D-05)."
        )


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def _process_one(tif: Path, out_dir: Path) -> tuple[Path, Exception | None]:
    sample_dir = out_dir / tif.stem
    try:
        hist_label.make_labels(tif, sample_dir)
        return tif, None
    except Exception as exc:
        return tif, exc


def cmd_build(raw_dir: Path, out_dir: Path, workers: int) -> None:
    tifs = sorted((raw_dir / "georeferenced").glob("*.tif"))
    # Also pick up any manually registered GeoTIFFs placed directly in raw_dir
    tifs += sorted(raw_dir.glob("*.tif"))
    tifs = list(dict.fromkeys(tifs))  # deduplicate, preserve order

    if not tifs:
        print(f"No GeoTIFFs found under {raw_dir}")
        sys.exit(1)

    print(f"=== Building labels for {len(tifs)} map(s) ===")
    out_dir.mkdir(parents=True, exist_ok=True)

    errors: list[tuple[Path, Exception]] = []

    if workers == 1:
        for tif in tifs:
            _, exc = _process_one(tif, out_dir)
            if exc:
                print(f"  ERROR {tif.name}: {exc}")
                errors.append((tif, exc))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_one, tif, out_dir): tif for tif in tifs}
            for fut in as_completed(futures):
                tif, exc = fut.result()
                if exc:
                    print(f"  ERROR {tif.name}: {exc}")
                    errors.append((tif, exc))

    print(f"\nBuild complete.")
    print(f"  Success: {len(tifs) - len(errors)}/{len(tifs)}")
    if errors:
        print(f"  Failed:")
        for tif, exc in errors:
            print(f"    {tif.name}: {exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble historical map training dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Common args
    def add_common(p):
        p.add_argument("--raw-dir", type=Path, default=Path("data/historical/raw"),
                       help="Directory for downloaded/registered GeoTIFFs")

    # search
    p_search = sub.add_parser("search", help="Download maps from David Rumsey")
    add_common(p_search)
    p_search.add_argument("--max-maps", type=int, default=500)

    # build
    p_build = sub.add_parser("build", help="Generate labels from GeoTIFFs")
    add_common(p_build)
    p_build.add_argument("--out-dir", type=Path, default=Path("data/historical/dataset"))
    p_build.add_argument("--workers", type=int, default=1,
                         help="Parallel worker threads for label generation")

    # full
    p_full = sub.add_parser("full", help="search + build")
    add_common(p_full)
    p_full.add_argument("--out-dir", type=Path, default=Path("data/historical/dataset"))
    p_full.add_argument("--max-maps", type=int, default=500)
    p_full.add_argument("--workers", type=int, default=1)

    args = parser.parse_args()

    if args.command == "search":
        cmd_search(args.raw_dir, args.max_maps)
    elif args.command == "build":
        cmd_build(args.raw_dir, args.out_dir, args.workers)
    elif args.command == "full":
        cmd_search(args.raw_dir, args.max_maps)
        cmd_build(args.raw_dir, args.out_dir, args.workers)


if __name__ == "__main__":
    main()
