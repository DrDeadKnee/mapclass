"""
Assemble the historical map training dataset.

Two sub-commands:

  search  — Query David Rumsey, download georeferenced maps to raw_dir,
             write unregistered_manifest.json (to GCS at
             gs://mapclass-training-northeast1/data/historical/raw/).
             Raw Rumsey GeoTIFF downloads remain local scratch (re-downloadable
             from LUNA — RW-04 research finding A-R5).

  build   — Process all GeoTIFFs in raw_dir into training samples, streaming
             pyramid output to GCS via _GCSWriter (RW-04).

  full    — search + build in one pass.

Usage
-----
  python build_historical_dataset.py search [--raw-dir RAW] [--max-maps N]
  python build_historical_dataset.py build  [--raw-dir RAW] [--out-dir OUT] [--workers N] [--local-ok]
  python build_historical_dataset.py full   [--raw-dir RAW] [--out-dir OUT] [--max-maps N] [--workers N] [--local-ok]

Defaults:
  --raw-dir   data/historical/raw  (local scratch for raw GeoTIFF downloads)
  --out-dir   gs://mapclass-training-northeast1/data/historical/dataset
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

# ---------------------------------------------------------------------------
# Lazy gcsfs import — same pattern as scripts/seg/gcs_checkpoint.py lines 36-39
# ---------------------------------------------------------------------------
try:
    import gcsfs  # type: ignore[import]
except ModuleNotFoundError:
    gcsfs = None  # type: ignore[assignment]

import tiling
from gcs_io import _GCSWriter, GCS_PROJECT, DATA_PREFIX, mark_build_complete, is_build_complete
from historical import label as hist_label
from historical import rumsey

_REQUIRED_MAP_FILES = (
    "image.png", "land_cover.png", "topography.png", "sample_weights.json",
)

# GCS key for the historical unregistered manifest (v2 hand-off artifact)
_HISTORICAL_MANIFEST_GCS_KEY = f"{DATA_PREFIX}/historical/raw/unregistered_manifest.json"


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def cmd_search(raw_dir: Path, max_maps: int) -> None:
    raw_dir = Path(raw_dir)
    geo_dir = raw_dir / "georeferenced"
    # Keep manifest_path local scratch so rumsey.emit_manifest can write normally.
    manifest_path = raw_dir / "unregistered_manifest.json"

    # Ensure raw_dir (and geo_dir) exist on local scratch before downloading.
    geo_dir.mkdir(parents=True, exist_ok=True)

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

    # Write manifest to local scratch first (rumsey.emit_manifest uses local Path).
    # Do NOT modify rumsey.py — PATTERNS.md option (b).
    rumsey.emit_manifest(unregistered, manifest_path)

    # RW-04: push the manifest to GCS so it survives ephemeral compute.
    # The manifest is the v2 hand-off artifact and must land in GCS.
    manifest_bytes = manifest_path.read_bytes()
    fs = gcsfs.GCSFileSystem(project=GCS_PROJECT)
    fs.pipe_file(_HISTORICAL_MANIFEST_GCS_KEY, manifest_bytes)
    print(f"  manifest written to GCS: gs://{_HISTORICAL_MANIFEST_GCS_KEY}")

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

def _process_one(
    tif: Path,
    out_dir: "str | Path",
    fs: "object | None",
    gcs_out_prefix: "str | None",
) -> "tuple[Path, Exception | None]":
    """Process one GeoTIFF into a training sample.

    Parameters
    ----------
    tif:            input GeoTIFF path (local scratch)
    out_dir:        output root — either a local Path or a gs:// string.
                    When gcs_out_prefix is set, out_dir is ignored for writing
                    (pyramid output goes to GCS); only the stem is used.
    fs:             gcsfs.GCSFileSystem instance (thread-safe shared instance)
                    or None for local writes.
    gcs_out_prefix: bare GCS prefix (no gs://) for pyramid output, or None
                    for local writes.  Example:
                    "mapclass-training-northeast1/data/historical/dataset"
    """
    sample_name = tif.stem
    if gcs_out_prefix is not None:
        # Sample dir lives in local scratch for reads (hist_label.make_labels
        # writes the schema files locally); tiling writes to GCS.
        # We need a local scratch sample dir.
        import tempfile
        scratch_base = Path(tempfile.mkdtemp(prefix="bhd_"))
        sample_dir = scratch_base / sample_name
    else:
        sample_dir = Path(out_dir) / sample_name
    try:
        hist_label.make_labels(tif, sample_dir)
        # Decompose into the shared nested pyramids once the full per-map
        # schema exists. Guard: a partially-failed map (missing any of the 4
        # required files) is skipped + logged, never tiled (T-02-16).
        missing = [
            f for f in _REQUIRED_MAP_FILES if not (sample_dir / f).exists()
        ]
        if missing:
            print(f"  SKIP tiling {sample_dir.name}: missing "
                  f"{', '.join(missing)}")
        else:
            if gcs_out_prefix is not None and fs is not None:
                # RW-04: stream pyramids to GCS. Each _GCSWriter is per-pyramid
                # (not shared) — thread safety relies on shared gcsfs.GCSFileSystem.
                gcs_map_prefix = f"{gcs_out_prefix}/{sample_name}"
                # OQ1: skip only if _BUILD_COMPLETE sentinel is present (Pitfall R-2).
                if is_build_complete(fs, gcs_map_prefix):
                    print(f"  SKIP (complete) {sample_name}")
                    return tif, None
                gcs_prefix = f"{gcs_map_prefix}/pyramids"
                tiling.tile(sample_dir, out_root=_GCSWriter(fs, gcs_prefix))
                # OQ1: write sentinel LAST after all pyramid objects (T-02-40).
                mark_build_complete(fs, gcs_map_prefix)
            else:
                tiling.tile(sample_dir)
        return tif, None
    except Exception as exc:
        return tif, exc


def cmd_build(raw_dir: "Path | str", out_dir: "Path | str", workers: int) -> None:
    raw_dir = Path(raw_dir)
    tifs = sorted((raw_dir / "georeferenced").glob("*.tif"))
    # Also pick up any manually registered GeoTIFFs placed directly in raw_dir
    tifs += sorted(raw_dir.glob("*.tif"))
    tifs = list(dict.fromkeys(tifs))  # deduplicate, preserve order

    if not tifs:
        print(f"No GeoTIFFs found under {raw_dir}")
        sys.exit(1)

    print(f"=== Building labels for {len(tifs)} map(s) ===")

    # Determine if out_dir is GCS or local
    out_dir_str = str(out_dir)
    is_gcs = out_dir_str.startswith("gs://")

    if is_gcs:
        # RW-04: stream pyramid output to GCS; no local out_dir.mkdir needed.
        gcs_out_prefix = out_dir_str[len("gs://"):]
        fs = gcsfs.GCSFileSystem(project=GCS_PROJECT)
        # mkdir is a no-op for GCS prefix; _GCSWriter.mkdir handles this inside tile()
    else:
        gcs_out_prefix = None
        fs = None
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    errors: list[tuple[Path, Exception]] = []

    if workers == 1:
        for tif in tifs:
            _, exc = _process_one(tif, out_dir, fs, gcs_out_prefix)
            if exc:
                print(f"  ERROR {tif.name}: {exc}")
                errors.append((tif, exc))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_process_one, tif, out_dir, fs, gcs_out_prefix): tif
                for tif in tifs
            }
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
                       help="Local scratch directory for downloaded raw GeoTIFFs")

    # search
    p_search = sub.add_parser("search", help="Download maps from David Rumsey")
    add_common(p_search)
    p_search.add_argument("--max-maps", type=int, default=500)

    # build
    p_build = sub.add_parser("build", help="Generate labels from GeoTIFFs")
    add_common(p_build)
    p_build.add_argument(
        "--out-dir",
        default="gs://mapclass-training-northeast1/data/historical/dataset",
        help="Output root: gs:// URI (default) or local path with --local-ok",
    )
    p_build.add_argument("--workers", type=int, default=1,
                         help="Parallel worker threads for label generation")
    p_build.add_argument(
        "--local-ok",
        action="store_true",
        help="Allow non-gs:// --out-dir (for offline testing)",
    )

    # full
    p_full = sub.add_parser("full", help="search + build")
    add_common(p_full)
    p_full.add_argument(
        "--out-dir",
        default="gs://mapclass-training-northeast1/data/historical/dataset",
        help="Output root: gs:// URI (default) or local path with --local-ok",
    )
    p_full.add_argument("--max-maps", type=int, default=500)
    p_full.add_argument("--workers", type=int, default=1)
    p_full.add_argument(
        "--local-ok",
        action="store_true",
        help="Allow non-gs:// --out-dir (for offline testing)",
    )

    args = parser.parse_args()

    # Pitfall R-1 guard: non-gs:// out-dir without --local-ok hard-errors.
    if hasattr(args, "out_dir") and hasattr(args, "local_ok"):
        out_dir_str = str(args.out_dir)
        if not out_dir_str.startswith("gs://") and not args.local_ok:
            parser.error(
                f"--out-dir {out_dir_str!r} is not a gs:// URI. "
                "Use --local-ok to allow local output (offline/testing only)."
            )

    if args.command == "search":
        cmd_search(args.raw_dir, args.max_maps)
    elif args.command == "build":
        cmd_build(args.raw_dir, args.out_dir, args.workers)
    elif args.command == "full":
        cmd_search(args.raw_dir, args.max_maps)
        cmd_build(args.raw_dir, args.out_dir, args.workers)


if __name__ == "__main__":
    main()
