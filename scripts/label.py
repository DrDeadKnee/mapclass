"""
Generate pixel-level land cover and topography label masks from an Azgaar GeoJSON export.

Usage:
    python label.py <geojson_path> <output_dir>

Outputs (one per-map directory):
    image.png       — the rendered style image (RGB), passed in by the caller
    land_cover.png  — class index per pixel (see LANDCOVER_CLASSES in biome_mapping.py)
    topography.png  — class index per pixel (0=flat, 1=hilly, 2=mountainous)
                      255 = water (no topography class assigned)
    sample_weights.json — locked-shape per-class loss weights (synthetic source)

The GeoJSON coordinate system is Azgaar's native pixel space; image dimensions are
inferred from the bounding box of all features. ``land_cover.png`` and
``topography.png`` are identical across render styles for the same Azgaar source —
the caller generates them once and shares them into each per-(source×style) dir.
"""

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from biome_mapping import h_to_landcover, h_to_topo
from synthetic_weights import write_sample_weights

NODATA = 255  # fill value for pixels not covered by any cell (shouldn't occur)
WATER_TOPO = 255  # sentinel: water cells have no topography class


def _bbox(features: list) -> tuple[float, float, float, float]:
    """Return (min_x, min_y, max_x, max_y) over all polygon rings."""
    xs, ys = [], []
    for feat in features:
        geom = feat["geometry"]
        rings = (
            geom["coordinates"]
            if geom["type"] == "Polygon"
            else [ring for poly in geom["coordinates"] for ring in poly]
        )
        for ring in rings:
            for x, y in ring:
                xs.append(x)
                ys.append(y)
    return min(xs), min(ys), max(xs), max(ys)


def _rings(geom: dict) -> list[list[tuple[float, float]]]:
    if geom["type"] == "Polygon":
        return geom["coordinates"]
    return [ring for poly in geom["coordinates"] for ring in poly]


def make_label_arrays(geojson_path: str | Path):
    """
    Rasterise the GeoJSON into (land_cover_img, topography_img) PIL images.

    The two masks share dimensions inferred from the feature bbox and are
    identical for every render style of the same Azgaar source map.
    """
    geojson_path = Path(geojson_path)
    with open(geojson_path) as f:
        data = json.load(f)
    features = data["features"]

    min_x, min_y, max_x, max_y = _bbox(features)
    width = int(max_x - min_x) + 1
    height = int(max_y - min_y) + 1

    lc_img = Image.new("L", (width, height), NODATA)
    topo_img = Image.new("L", (width, height), NODATA)
    lc_draw = ImageDraw.Draw(lc_img)
    topo_draw = ImageDraw.Draw(topo_img)

    for feat in features:
        props = feat["properties"]
        biome = int(props["biome"])
        h = int(props["height"])

        lc_class = h_to_landcover(h, biome)
        topo_class = h_to_topo(h)
        topo_fill = topo_class if topo_class is not None else WATER_TOPO

        for ring in _rings(feat["geometry"]):
            # Shift coordinates so origin is (0, 0)
            coords = [(x - min_x, y - min_y) for x, y in ring]
            if len(coords) < 3:
                continue
            lc_draw.polygon(coords, fill=lc_class)
            topo_draw.polygon(coords, fill=topo_fill)

    return lc_img, topo_img


def make_labels(
    geojson_path: str | Path,
    output_dir: str | Path,
    image: "str | Path | Image.Image | None" = None,
    map_file: str | None = None,
) -> None:
    """
    Write the full locked per-map schema into ``output_dir``:
    ``image.png`` + ``land_cover.png`` + ``topography.png`` + ``sample_weights.json``.

    Parameters
    ----------
    geojson_path : Azgaar GeoJSON export for this source map
    output_dir   : per-(source×style) directory to populate
    image        : the rendered style image — a PIL Image or a path to a PNG.
                   Written verbatim as ``image.png``. If omitted, ``image.png``
                   is not written (kept for backward compatibility / labels-only).
    map_file     : the source identifier recorded in ``sample_weights.json``
                   (defaults to the GeoJSON stem).
    """
    geojson_path = Path(geojson_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lc_img, topo_img = make_label_arrays(geojson_path)

    if image is not None:
        if isinstance(image, (str, Path)):
            img = Image.open(image)
        else:
            img = image
        img.save(output_dir / "image.png")

    lc_img.save(output_dir / "land_cover.png")
    topo_img.save(output_dir / "topography.png")

    write_sample_weights(output_dir, map_file or geojson_path.stem)

    print(
        f"Saved per-map dir → {output_dir}  "
        f"({lc_img.width}x{lc_img.height}px)"
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python label.py <geojson_path> <output_dir>")
        sys.exit(1)
    make_labels(sys.argv[1], sys.argv[2])
