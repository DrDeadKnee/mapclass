"""
Orchestrate the synthetic dataset build from a directory of Azgaar GeoJSON exports.

Per resolved decision A4 (option (a)): each (Azgaar source × render style) is its
own canonical per-map directory ``<azgaar_id>__<style>/`` containing exactly one
``image.png`` (the S-style render) plus ``land_cover.png`` / ``topography.png``
(identical across styles for the same source — generated once and shared) plus
``sample_weights.json``.

Usage:
    python build_dataset.py <raw_dir> <output_dir> [--styles flat illustrated satellite]

For each <name>.geojson in raw_dir and each style S, produces:
    <output_dir>/<name>__<S>/image.png
    <output_dir>/<name>__<S>/land_cover.png   (shared across styles of <name>)
    <output_dir>/<name>__<S>/topography.png   (shared across styles of <name>)
    <output_dir>/<name>__<S>/sample_weights.json

Raw GeoJSON files are expected to come from Azgaar's GIS export
(Map menu → Save → GeoJSON with all layers, or Tools → Export → GeoJSON cells).

Template-name note (A4 / split stratification): Azgaar GeoJSON exports do NOT
expose the heightmap template name in feature ``properties`` or a top-level
metadata block (confirmed against the Plan-01 conftest fixture; no raw exports
on disk to inspect). The seeded stratified split (Task 4) therefore falls back
to a filename-derived stratification key — see ``build_dataset`` split logic.
"""

import re
import sys
from pathlib import Path

from label import make_label_arrays
from render import render_one
from synthetic_weights import write_sample_weights

_STYLE_SEP = "__"


def _sanitize_stem(stem: str) -> str:
    """Sanitize an Azgaar source stem for safe filesystem path use (T-02-08)."""
    return re.sub(r"[^\w-]", "_", stem)


def build_one_source(
    geojson_path: Path,
    output_root: Path,
    styles=("flat", "illustrated", "satellite"),
) -> list[Path]:
    """Build every per-(source×style) dir for a single Azgaar source.

    ``land_cover.png`` / ``topography.png`` are rasterised once and saved
    byte-identically into each style dir (shared label, D-15 / A4 option (a)).
    Returns the list of created style directories.
    """
    geojson_path = Path(geojson_path)
    src_id = _sanitize_stem(geojson_path.stem)

    # Rasterise the shared labels exactly once for this source.
    lc_img, topo_img = make_label_arrays(geojson_path)

    created: list[Path] = []
    for style in styles:
        out = Path(output_root) / f"{src_id}{_STYLE_SEP}{style}"
        out.mkdir(parents=True, exist_ok=True)

        img = render_one(geojson_path, style)
        img.save(out / "image.png")
        lc_img.save(out / "land_cover.png")
        topo_img.save(out / "topography.png")
        write_sample_weights(out, geojson_path.stem)

        created.append(out)
        print(f"  built {out.name}  ({lc_img.width}x{lc_img.height}px)")
    return created


def build(raw_dir: str | Path, output_dir: str | Path, styles=("flat", "illustrated", "satellite")) -> None:
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    geojsons = sorted(raw_dir.glob("*.geojson"))
    if not geojsons:
        print(f"No .geojson files found in {raw_dir}")
        sys.exit(1)

    print(f"Found {len(geojsons)} source map(s) in {raw_dir}")

    for geojson_path in geojsons:
        print(f"\n--- {geojson_path.stem} ---")
        try:
            build_one_source(geojson_path, output_dir, styles=styles)
        except Exception as exc:  # per-source resilience (T-02-09)
            print(f"  FAILED {geojson_path.name}: {exc}")

    print(f"\nDone. Dataset written to {output_dir}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 2:
        print("Usage: python build_dataset.py <raw_dir> <output_dir> [style ...]")
        sys.exit(1)
    raw_dir, output_dir = args[0], args[1]
    styles = args[2:] if len(args) > 2 else ("flat", "illustrated", "satellite")
    build(raw_dir, output_dir, styles)
