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

# Decompression-bomb-analogue guard for the synthetic path (threat parity
# with T-02-04): refuse to allocate a canvas larger than this on any edge.
_MAX_CANVAS_DIM = 20000


def _rings(geom: dict) -> list[list[tuple[float, float]]]:
    """Return the coordinate rings of a Polygon/MultiPolygon geometry.

    Returns ``[]`` for any other / malformed geometry so callers can skip
    the feature rather than crash (CR-03).
    """
    if not isinstance(geom, dict):
        return []
    gtype = geom.get("type")
    coords = geom.get("coordinates")
    if coords is None:
        return []
    try:
        if gtype == "Polygon":
            # A malformed Polygon whose coordinates is a scalar/dict (not a
            # list of rings) would defer the crash to the caller's draw
            # loop; reject it here so the feature is skipped (CR-04).
            if not isinstance(coords, (list, tuple)):
                return []
            return coords
        if gtype == "MultiPolygon":
            return [ring for poly in coords for ring in poly]
    except TypeError:
        return []
    return []


def _bbox(features: list) -> tuple[float, float, float, float]:
    """Return (min_x, min_y, max_x, max_y) over all usable polygon rings.

    Non-Polygon/MultiPolygon, missing, or malformed geometries are skipped
    rather than crashing. Raises ``ValueError`` only when NO feature has a
    usable polygon (CR-03).
    """
    xs, ys = [], []
    for feat in features:
        geom = (feat or {}).get("geometry") or {}
        if geom.get("type") not in ("Polygon", "MultiPolygon"):
            continue
        for ring in _rings(geom):
            for pt in ring:
                if len(pt) >= 2:
                    xs.append(pt[0])
                    ys.append(pt[1])
    if not xs:
        raise ValueError("GeoJSON has no usable Polygon/MultiPolygon geometry")
    return min(xs), min(ys), max(xs), max(ys)


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
    if width <= 0 or height <= 0:
        raise ValueError(
            f"degenerate canvas {width}x{height} from feature bbox"
        )
    if width > _MAX_CANVAS_DIM or height > _MAX_CANVAS_DIM:
        raise ValueError(
            f"canvas {width}x{height} exceeds max {_MAX_CANVAS_DIM}px "
            f"(pathological coordinates — refusing to allocate)"
        )

    lc_img = Image.new("L", (width, height), NODATA)
    topo_img = Image.new("L", (width, height), NODATA)
    lc_draw = ImageDraw.Draw(lc_img)
    topo_draw = ImageDraw.Draw(topo_img)

    for feat in features:
        props = (feat or {}).get("properties") or {}
        try:
            biome = int(props["biome"])
            h = int(props["height"])
        except (KeyError, TypeError, ValueError):
            # Missing/non-numeric biome|height — skip this feature, do not
            # abort the whole source (CR-03).
            continue

        lc_class = h_to_landcover(h, biome)
        topo_class = h_to_topo(h)
        topo_fill = topo_class if topo_class is not None else WATER_TOPO

        for ring in _rings((feat or {}).get("geometry") or {}):
            # Shift coordinates so origin is (0, 0). Slice each position to
            # its first two ordinates: GeoJSON (RFC 7946) permits a third
            # element (elevation), and a 3-element coord would otherwise
            # crash the tuple-unpack and abort the entire source (CR-04),
            # mirroring the _bbox `len(pt) >= 2` hardening.
            coords = [(pt[0] - min_x, pt[1] - min_y) for pt in ring if len(pt) >= 2]
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
