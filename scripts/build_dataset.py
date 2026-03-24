"""
Orchestrate the synthetic dataset build from a directory of Azgaar GeoJSON exports.

Usage:
    python build_dataset.py <raw_dir> <output_dir> [--styles flat illustrated satellite]

For each <name>.geojson in raw_dir, produces:
    <output_dir>/<name>/flat.png
    <output_dir>/<name>/illustrated.png
    <output_dir>/<name>/satellite.png
    <output_dir>/<name>/land_cover.png
    <output_dir>/<name>/topography.png

Raw GeoJSON files are expected to come from Azgaar's GIS export
(Map menu → Save → GeoJSON with all layers, or Tools → Export → GeoJSON cells).
"""

import sys
from pathlib import Path

from label import make_labels
from render import render_map


def build(raw_dir: str | Path, output_dir: str | Path, styles=("flat", "illustrated", "satellite")) -> None:
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    geojsons = sorted(raw_dir.glob("*.geojson"))
    if not geojsons:
        print(f"No .geojson files found in {raw_dir}")
        sys.exit(1)

    print(f"Found {len(geojsons)} map(s) in {raw_dir}")

    for geojson_path in geojsons:
        name = geojson_path.stem
        out = output_dir / name
        print(f"\n--- {name} ---")
        render_map(geojson_path, out, styles=styles)
        make_labels(geojson_path, out)

    print(f"\nDone. Dataset written to {output_dir}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 2:
        print("Usage: python build_dataset.py <raw_dir> <output_dir> [style ...]")
        sys.exit(1)
    raw_dir, output_dir = args[0], args[1]
    styles = args[2:] if len(args) > 2 else ("flat", "illustrated", "satellite")
    build(raw_dir, output_dir, styles)
