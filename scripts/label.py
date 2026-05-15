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


def _xy(pt):
    """Return ``(x, y)`` floats for a GeoJSON position, or ``None`` if the
    element is malformed in *any* way.

    Three iterations (CR-03 → CR-04 → CR-01) chased individual malformed
    element shapes (missing z, scalar element, string element) with
    piecemeal per-element guards. This helper is the single structural
    chokepoint: it rejects non-sequences, short sequences, and
    non-numeric ordinates uniformly, so no element shape can reach the
    arithmetic / ``min``/``max`` path and abort the source (T-02-09).
    """
    if not isinstance(pt, (list, tuple)) or len(pt) < 2:
        return None
    try:
        return float(pt[0]), float(pt[1])
    except (TypeError, ValueError):
        return None


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
        # Structurally resilient: ANY malformed feature/ring (regardless of
        # element shape) is skipped, never aborts the whole source (CR-01 /
        # WR-01 / T-02-09 / D-13). _xy rejects non-numeric / short / scalar
        # elements so a single bad point cannot corrupt the canvas bounds.
        try:
            geom = (feat or {}).get("geometry") or {}
            if geom.get("type") not in ("Polygon", "MultiPolygon"):
                continue
            for ring in _rings(geom):
                try:
                    for pt in ring:
                        xy = _xy(pt)
                        if xy is not None:
                            xs.append(xy[0])
                            ys.append(xy[1])
                except (TypeError, ValueError, KeyError, IndexError):
                    continue
        except (TypeError, ValueError, KeyError, IndexError):
            continue
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
        # Per-feature structural resilience: any malformed feature/ring —
        # regardless of element shape (missing z, scalar, string, empty
        # ring) — is skipped + drop-counted, never aborts the source
        # (CR-01 / T-02-09 / D-13). One try/except replaces the Nth
        # element-shape patch.
        try:
            props = (feat or {}).get("properties") or {}
            try:
                biome = int(props["biome"])
                h = int(props["height"])
            except (KeyError, TypeError, ValueError):
                # Missing/non-numeric biome|height — skip this feature, do
                # not abort the whole source (CR-03).
                continue

            lc_class = h_to_landcover(h, biome)
            topo_class = h_to_topo(h)
            topo_fill = topo_class if topo_class is not None else WATER_TOPO

            for ring in _rings((feat or {}).get("geometry") or {}):
                try:
                    # Shift coordinates so origin is (0, 0). _xy defensively
                    # coerces each position and rejects any element that is
                    # not a real (x, y) pair, so no element shape can crash
                    # the draw loop (CR-01 / WR-01).
                    coords = []
                    for pt in ring:
                        xy = _xy(pt)
                        if xy is not None:
                            coords.append((xy[0] - min_x, xy[1] - min_y))
                    if len(coords) < 3:
                        continue
                    lc_draw.polygon(coords, fill=lc_class)
                    topo_draw.polygon(coords, fill=topo_fill)
                except (TypeError, ValueError, KeyError, IndexError):
                    continue
        except (TypeError, ValueError, KeyError, IndexError):
            continue

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
