"""
Assemble the satellite (Sentinel-2 + ESA WorldCover) training dataset.

Three sub-commands (mirrors ``build_historical_dataset.py``):

  coverage-scan — Build (and cache) the one-shot coarse global WorldCover
                  class-count summary used by the region picker.

  search        — Pick the top-N class-diverse regions, STAC-search the
                  lowest-cloud Sentinel-2 L2A scene per region, and write a
                  resolved-scene manifest. Per-reason drop accounting (D-13).

  build         — For each resolved scene: fetch the 4096-px visual window
                  (COG byte-range), reuse historical.label.make_labels
                  verbatim, and emit the satellite sample_weights.json.

Usage
-----
  python build_satellite_dataset.py coverage-scan [--summary PATH]
  python build_satellite_dataset.py search [--summary PATH] [--manifest M]
                                           [--n-regions N] [--seed S]
                                           [--max-cloud C]
  python build_satellite_dataset.py build  [--manifest M] [--out-dir OUT]
                                           [--workers N] [--local-ok]

Defaults (GCS-canonical, RW-04):
  --summary    gs://mapclass-training-northeast1/data/satellite/coverage_summary.json
  --manifest   gs://mapclass-training-northeast1/data/satellite/resolved_scenes.json
  --out-dir    gs://mapclass-training-northeast1/data/satellite/dataset
  --n-regions  200
  --seed       42
  --max-cloud  10
  --workers    1
"""

import argparse
import json
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Make the 'historical'/'satellite' packages importable from repo root or scripts/.
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import tiling
from gcs_io import _GCSWriter, GCS_PROJECT, DATA_PREFIX
from historical import label as hist_label
from satellite import coverage, fetch, stac
from satellite import weights as sat_weights

# Lazy gcsfs import — same pattern as scripts/seg/gcs_checkpoint.py lines 36-39.
# The module can be imported on the planning VM where gcsfs is absent.
# Tests patch ``build_satellite_dataset.gcsfs`` the same way tests patch
# ``seg.gcs_checkpoint.gcsfs``.
try:
    import gcsfs  # type: ignore[import]
except ModuleNotFoundError:
    gcsfs = None  # type: ignore[assignment]

_REQUIRED_MAP_FILES = (
    "image.png", "land_cover.png", "topography.png", "sample_weights.json",
)

# GCS-canonical defaults (RW-04). All three must start with "gs://".
_DEFAULT_SUMMARY = (
    f"gs://{DATA_PREFIX}/satellite/coverage_summary.json"
)
_DEFAULT_MANIFEST = (
    f"gs://{DATA_PREFIX}/satellite/resolved_scenes.json"
)
_DEFAULT_OUT = (
    f"gs://{DATA_PREFIX}/satellite/dataset"
)

# Threat T-02-12: region ids become a filesystem path component.
_UNSAFE = re.compile(r"[^\w-]")


def _sanitize(region_id: str) -> str:
    """Replace any non-``[\\w-]`` char with ``_`` before joining into a path."""
    return _UNSAFE.sub("_", region_id)


def _print_drop_summary(drops: dict[str, int], label: str) -> None:
    """Print the D-13 per-reason drop counter (same shape/loudness as D-05)."""
    total = sum(drops.values())
    print(f"\n{label} summary:")
    for reason, n in drops.items():
        print(f"  {reason}: {n}")
    if total > 0 and drops.get("ok", 0) / total < 0.5:
        print(
            f"\n⚠ resolved/built rate <50% ({drops.get('ok', 0)}/{total}) — "
            f"the satellite pipeline is dropping most candidate regions; "
            f"Phase 2 is not done (D-13)."
        )


def _make_fs():
    """Instantiate a GCSFileSystem (lazily — raises ImportError if not installed)."""
    if gcsfs is None:
        raise ImportError(
            "gcsfs is not installed. Install it on the training host with: "
            "pip install gcsfs"
        )
    return gcsfs.GCSFileSystem(project=GCS_PROJECT)


def _is_gcs_path(path_str: str) -> bool:
    """Return True if the string is a GCS URI."""
    return str(path_str).startswith("gs://")


def _bare(gcs_uri: str) -> str:
    """Strip the ``gs://`` scheme from a GCS URI (bucket/path form)."""
    return gcs_uri[len("gs://"):]


# ---------------------------------------------------------------------------
# coverage-scan
# ---------------------------------------------------------------------------

def cmd_coverage_scan(summary_path) -> None:
    """Build the coarse WorldCover class-count summary and persist to GCS."""
    print("=== Coverage scan (coarse WorldCover summary) ===")

    summary_path_str = str(summary_path)
    if _is_gcs_path(summary_path_str):
        # coverage.build_summary requires a local Path to write to.
        # Write to a local temp file, then pipe to GCS.
        fs = _make_fs()
        with tempfile.NamedTemporaryFile(
            suffix="_coverage_summary.json", delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            summary = coverage.build_summary(tmp_path)
            # Pipe the produced JSON bytes to GCS (fs.pipe_file is atomic).
            gcs_dest = _bare(summary_path_str)
            fs.pipe_file(gcs_dest, json.dumps(summary).encode())
            print(f"  Coverage summary written to {summary_path_str}")
        finally:
            tmp_path.unlink(missing_ok=True)
    else:
        summary = coverage.build_summary(Path(summary_path))

    print(f"  {len(summary['cells'])} cells available for region picking")


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def cmd_search_regions(
    regions: list[dict],
    manifest_path,
    max_cloud: float = 10,
    fs=None,
) -> dict[str, int]:
    """
    Resolve a pre-picked list of regions to lowest-cloud scenes and write the
    manifest. Returns the drop counter. Each region dict:
    ``{"region_id", "bbox": [w,s,e,n], "datetime_range"}``.
    """
    drops = {
        "ok": 0,
        "no_qualifying_scene": 0,
        "missing_visual_asset": 0,
        "stac_search_failed": 0,
    }
    resolved: list[dict] = []

    for region in regions:
        rid = region["region_id"]
        bbox = tuple(region["bbox"])
        dt = region["datetime_range"]
        try:
            item = stac.find_lowest_cloud_scene(bbox, dt, max_cloud=max_cloud)
        except stac.StacLookupError as exc:
            print(f"  {rid}: STAC search failed: {exc}")
            drops["stac_search_failed"] += 1
            continue

        if item is None:
            print(f"  {rid}: no qualifying scene (cloud<{max_cloud})")
            drops["no_qualifying_scene"] += 1
            continue

        try:
            visual_href = item.assets["visual"].href
        except (KeyError, AttributeError, TypeError):
            # WR-03: a resolved scene that lacks the TCI/visual asset is a
            # DISTINCT drop reason from "no scene under the cloud
            # threshold" — conflating them corrupts the D-13 per-reason
            # accounting that the phase requires to be loud and accurate.
            print(f"  {rid}: resolved scene has no 'visual' asset")
            drops["missing_visual_asset"] += 1
            continue

        resolved.append({
            "region_id": rid,
            "bbox": list(bbox),
            "datetime_range": dt,
            "visual_href": visual_href,
            "eo_cloud_cover": item.properties.get("eo:cloud_cover"),
        })
        drops["ok"] += 1

    manifest_path_str = str(manifest_path)
    manifest_json = json.dumps({"scenes": resolved}, indent=2).encode()

    if _is_gcs_path(manifest_path_str):
        # Write resolved_scenes.json directly to GCS (Threat T-02-30 mitigation).
        if fs is None:
            fs = _make_fs()
        gcs_dest = _bare(manifest_path_str)
        fs.pipe_file(gcs_dest, manifest_json)
        print(f"  Resolved {len(resolved)} scenes → {manifest_path_str}")
    else:
        local = Path(manifest_path)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(manifest_json)
        print(f"  Resolved {len(resolved)} scenes → {local}")

    _print_drop_summary(drops, "Search")
    return drops


def cmd_search(
    summary_path,
    manifest_path,
    n_regions: int,
    seed: int,
    max_cloud: float,
    fs=None,
) -> None:
    print("=== Region pick + STAC resolve ===")
    # coverage.pick_regions may need a local summary if summary is on GCS;
    # for the search command a local file is fine as a Path cache.
    regions = coverage.pick_regions(n_regions, seed, summary_path=summary_path)
    print(f"  Picked {len(regions)} class-diverse regions (seed={seed})")
    cmd_search_regions(regions, manifest_path, max_cloud=max_cloud, fs=fs)


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

class _SceneItem:
    """Minimal STAC-item stand-in carrying the manifest's resolved visual href."""

    def __init__(self, href: str):
        self.assets = {"visual": _Asset(href)}
        self.properties: dict = {}


class _Asset:
    def __init__(self, href: str):
        self.href = href


def _process_one(scene: dict, out_dir, fs=None) -> tuple[str, str]:
    """
    Fetch one resolved scene's window, label it (reused historical code), and
    write the satellite weights. Routes tiling.tile through _GCSWriter when
    out_dir is a GCS URI (RW-04).

    Returns ``(region_id, status)`` where status is one of ``ok`` /
    ``fetch_failed`` (exceptions are returned, not raised — threat T-02-11,
    threadpool resilience).

    ``fs`` is a shared gcsfs.GCSFileSystem passed in from the caller — one
    instance per cmd_build call, shared thread-safely across workers. A fresh
    per-pyramid _GCSWriter is constructed inside this function (not shared).
    """
    rid = scene["region_id"]
    safe = _sanitize(rid)
    out_dir_str = str(out_dir)

    # local scratch always — raw COG reads (VSI-CURL) are in-memory and never
    # staged; only the reprojected window GeoTIFF lands in local scratch.
    with tempfile.TemporaryDirectory() as _tmp_root:
        sample_dir = Path(_tmp_root) / safe
        sample_dir.mkdir(parents=True, exist_ok=True)
        try:
            item = _SceneItem(scene["visual_href"])
            geotiff = sample_dir / f"{safe}.tif"
            written = fetch.fetch_visual_window(item, geotiff)
            if written is None:
                return rid, "fetch_failed"

            # Reuse the historical label generator VERBATIM (zero new label code).
            hist_label.make_labels(written, sample_dir)
            # Satellite supplies its OWN weights dict (same locked JSON shape).
            sat_weights.write_sample_weights(sample_dir, geotiff.name)

            # Guard: skip + log a partial map rather than tiling a corrupt dir.
            missing = [
                f for f in _REQUIRED_MAP_FILES if not (sample_dir / f).exists()
            ]
            if missing:
                print(f"  SKIP tiling {safe}: missing {', '.join(missing)}")
            else:
                if _is_gcs_path(out_dir_str):
                    # Per-pyramid _GCSWriter — NOT shared across threads
                    # (_GCSWriter instances are not thread-safe; the shared fs IS).
                    gcs_prefix = _bare(out_dir_str)
                    out_root = _GCSWriter(fs, f"{gcs_prefix}/{safe}/pyramids")
                    tiling.tile(sample_dir, out_root=out_root)
                else:
                    local_out = Path(out_dir) / safe
                    local_out.mkdir(parents=True, exist_ok=True)
                    tiling.tile(sample_dir, out_root=local_out / "pyramids")

            return rid, "ok"
        except Exception as exc:  # noqa: BLE001 — surface, don't crash the batch
            print(f"  ERROR {rid}: {exc}")
            return rid, "fetch_failed"


def cmd_build(manifest_path, out_dir, workers: int) -> dict[str, int]:
    manifest_path_str = str(manifest_path)
    out_dir_str = str(out_dir)

    # Pitfall R-1 guard (T-02-30): non-gs:// --out-dir requires --local-ok.
    # This function receives local_ok from main(); check via the flag embedded
    # in the out_dir value — the check is done before any I/O.
    # NOTE: cmd_build itself does not receive local_ok; callers (main or tests)
    # are responsible for pre-checking. The check in main() enforces R-1.

    # Read manifest — supports both GCS and local paths.
    if _is_gcs_path(manifest_path_str):
        fs = _make_fs()
        manifest_gcs = _bare(manifest_path_str)
        try:
            raw = fs.open(manifest_gcs, "rb").read()
        except FileNotFoundError:
            print(f"No manifest at {manifest_path_str} — run 'search' first")
            sys.exit(1)
        scenes = json.loads(raw).get("scenes", [])
    else:
        local = Path(manifest_path)
        if not local.exists():
            print(f"No manifest at {local} — run 'search' first")
            sys.exit(1)
        scenes = json.loads(local.read_text()).get("scenes", [])
        fs = None

    if not scenes:
        print(f"Manifest {manifest_path_str} has no resolved scenes")
        sys.exit(1)

    print(f"=== Building labels for {len(scenes)} scene(s) ===")

    # Instantiate the shared fs once (thread-safe gcsfs.GCSFileSystem).
    if _is_gcs_path(out_dir_str) and fs is None:
        fs = _make_fs()
    elif not _is_gcs_path(out_dir_str):
        Path(out_dir_str).mkdir(parents=True, exist_ok=True)

    drops = {"ok": 0, "fetch_failed": 0}

    if workers == 1:
        for scene in scenes:
            _, status = _process_one(scene, out_dir, fs=fs)
            drops[status] = drops.get(status, 0) + 1
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_process_one, scene, out_dir, fs): scene
                for scene in scenes
            }
            for fut in as_completed(futures):
                _, status = fut.result()
                drops[status] = drops.get(status, 0) + 1

    print(f"\nBuild complete.")
    print(f"  Success: {drops['ok']}/{len(scenes)}")
    _print_drop_summary(drops, "Build")
    return drops


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble the satellite training dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("coverage-scan", help="Build the coarse WC summary")
    p_scan.add_argument("--summary", default=_DEFAULT_SUMMARY)

    p_search = sub.add_parser("search", help="Pick regions + STAC resolve scenes")
    p_search.add_argument("--summary", default=_DEFAULT_SUMMARY)
    p_search.add_argument("--manifest", default=_DEFAULT_MANIFEST)
    p_search.add_argument("--n-regions", type=int, default=200)
    p_search.add_argument("--seed", type=int, default=42)
    p_search.add_argument("--max-cloud", type=float, default=10)

    p_build = sub.add_parser("build", help="Fetch + label resolved scenes")
    p_build.add_argument("--manifest", default=_DEFAULT_MANIFEST)
    p_build.add_argument("--out-dir", default=_DEFAULT_OUT)
    p_build.add_argument("--workers", type=int, default=1)
    p_build.add_argument(
        "--local-ok",
        action="store_true",
        default=False,
        help=(
            "Allow a non-gs:// --out-dir (for offline testing/debugging). "
            "Without this flag a local --out-dir hard-errors (Pitfall R-1, T-02-30)."
        ),
    )

    args = parser.parse_args()

    if args.command == "coverage-scan":
        cmd_coverage_scan(args.summary)
    elif args.command == "search":
        cmd_search(
            args.summary, args.manifest, args.n_regions, args.seed, args.max_cloud
        )
    elif args.command == "build":
        # Pitfall R-1 guard: non-gs:// out-dir requires --local-ok (T-02-30).
        if not _is_gcs_path(str(args.out_dir)) and not args.local_ok:
            print(
                f"ERROR: --out-dir {args.out_dir!r} is not a gs:// URI. "
                f"If you intend to write to local disk, pass --local-ok. "
                f"(Pitfall R-1 / T-02-30)"
            )
            sys.exit(1)
        cmd_build(args.manifest, args.out_dir, args.workers)


if __name__ == "__main__":
    main()
