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
                                           [--workers N]

Defaults:
  --summary    data/satellite/coverage_summary.json
  --manifest   data/satellite/resolved_scenes.json
  --out-dir    data/satellite/dataset
  --n-regions  200
  --seed       42
  --max-cloud  10
  --workers    1
"""

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Make the 'historical'/'satellite' packages importable from repo root or scripts/.
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import tiling
from historical import label as hist_label
from satellite import coverage, fetch, stac
from satellite import weights as sat_weights

_REQUIRED_MAP_FILES = (
    "image.png", "land_cover.png", "topography.png", "sample_weights.json",
)

_DEFAULT_SUMMARY = Path("data/satellite/coverage_summary.json")
_DEFAULT_MANIFEST = Path("data/satellite/resolved_scenes.json")
_DEFAULT_OUT = Path("data/satellite/dataset")

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


# ---------------------------------------------------------------------------
# coverage-scan
# ---------------------------------------------------------------------------

def cmd_coverage_scan(summary_path: Path) -> None:
    print("=== Coverage scan (coarse WorldCover summary) ===")
    summary = coverage.build_summary(summary_path)
    print(f"  {len(summary['cells'])} cells available for region picking")


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def cmd_search_regions(
    regions: list[dict],
    manifest_path: Path,
    max_cloud: float = 10,
) -> dict[str, int]:
    """
    Resolve a pre-picked list of regions to lowest-cloud scenes and write the
    manifest. Returns the drop counter. Each region dict:
    ``{"region_id", "bbox": [w,s,e,n], "datetime_range"}``.
    """
    drops = {
        "ok": 0,
        "no_qualifying_scene": 0,
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
            print(f"  {rid}: resolved scene has no 'visual' asset")
            drops["no_qualifying_scene"] += 1
            continue

        resolved.append({
            "region_id": rid,
            "bbox": list(bbox),
            "datetime_range": dt,
            "visual_href": visual_href,
            "eo_cloud_cover": item.properties.get("eo:cloud_cover"),
        })
        drops["ok"] += 1

    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"scenes": resolved}, indent=2))
    print(f"  Resolved {len(resolved)} scenes → {manifest_path}")
    _print_drop_summary(drops, "Search")
    return drops


def cmd_search(
    summary_path: Path,
    manifest_path: Path,
    n_regions: int,
    seed: int,
    max_cloud: float,
) -> None:
    print("=== Region pick + STAC resolve ===")
    regions = coverage.pick_regions(n_regions, seed, summary_path=summary_path)
    print(f"  Picked {len(regions)} class-diverse regions (seed={seed})")
    cmd_search_regions(regions, manifest_path, max_cloud=max_cloud)


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


def _process_one(scene: dict, out_dir: Path) -> tuple[str, str]:
    """
    Fetch one resolved scene's window, label it (reused historical code), and
    write the satellite weights. Returns ``(region_id, status)`` where status
    is one of ``ok`` / ``fetch_failed`` (exceptions are returned, not raised —
    threat T-02-11, threadpool resilience).
    """
    rid = scene["region_id"]
    safe = _sanitize(rid)
    sample_dir = out_dir / safe
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
        # Tile into nested pyramids now the full per-map schema is written.
        # Guard: skip + log a partial map rather than tiling a corrupt dir.
        missing = [
            f for f in _REQUIRED_MAP_FILES if not (sample_dir / f).exists()
        ]
        if missing:
            print(f"  SKIP tiling {safe}: missing {', '.join(missing)}")
        else:
            tiling.tile(sample_dir)
        return rid, "ok"
    except Exception as exc:  # noqa: BLE001 — surface, don't crash the batch
        print(f"  ERROR {rid}: {exc}")
        return rid, "fetch_failed"


def cmd_build(manifest_path: Path, out_dir: Path, workers: int) -> dict[str, int]:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        print(f"No manifest at {manifest_path} — run 'search' first")
        sys.exit(1)

    scenes = json.loads(manifest_path.read_text()).get("scenes", [])
    if not scenes:
        print(f"Manifest {manifest_path} has no resolved scenes")
        sys.exit(1)

    print(f"=== Building labels for {len(scenes)} scene(s) ===")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    drops = {"ok": 0, "fetch_failed": 0}

    if workers == 1:
        for scene in scenes:
            _, status = _process_one(scene, out_dir)
            drops[status] = drops.get(status, 0) + 1
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_process_one, scene, out_dir): scene
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
    p_scan.add_argument("--summary", type=Path, default=_DEFAULT_SUMMARY)

    p_search = sub.add_parser("search", help="Pick regions + STAC resolve scenes")
    p_search.add_argument("--summary", type=Path, default=_DEFAULT_SUMMARY)
    p_search.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    p_search.add_argument("--n-regions", type=int, default=200)
    p_search.add_argument("--seed", type=int, default=42)
    p_search.add_argument("--max-cloud", type=float, default=10)

    p_build = sub.add_parser("build", help="Fetch + label resolved scenes")
    p_build.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    p_build.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT)
    p_build.add_argument("--workers", type=int, default=1)

    args = parser.parse_args()

    if args.command == "coverage-scan":
        cmd_coverage_scan(args.summary)
    elif args.command == "search":
        cmd_search(
            args.summary, args.manifest, args.n_regions, args.seed, args.max_cloud
        )
    elif args.command == "build":
        cmd_build(args.manifest, args.out_dir, args.workers)


if __name__ == "__main__":
    main()
