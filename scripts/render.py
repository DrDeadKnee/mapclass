"""
Render an Azgaar GeoJSON export as a styled raster image.

Usage:
    python render.py <geojson_path> <output_dir> [--styles flat illustrated satellite]

Each requested style produces one PNG in output_dir named <style>.png.
Image dimensions are inferred from the GeoJSON bounding box, matching label.py.

Styles
------
flat        — solid biome-representative fill, 1px borders
illustrated — muted parchment palette with thicker borders, mimicking hand-drawn maps
satellite   — realistic muted palette, no visible borders
"""

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from biome_mapping import LANDCOVER_CLASSES, h_to_landcover

# ---------------------------------------------------------------------------
# Palette: canonical land cover index → RGB for each style
# ---------------------------------------------------------------------------

_FLAT = {
    0: (60,  120, 200),   # water
    1: (34,  100,  34),   # trees
    2: (140, 170,  70),   # shrubland
    3: (180, 210, 100),   # grassland
    4: (220, 200,  80),   # cropland  (not in synthetic data but defined for completeness)
    5: (160, 150, 140),   # built_up
    6: (210, 190, 150),   # bare_sparse
    7: ( 70, 150, 140),   # flooded_wetland
    8: (235, 245, 255),   # snow_ice
}

_ILLUSTRATED = {
    0: ( 90, 150, 200),   # water — washed-out blue
    1: ( 80, 130,  60),   # trees — dark olive green
    2: (160, 175,  90),   # shrubland — khaki
    3: (200, 215, 130),   # grassland — pale yellow-green
    4: (215, 195,  90),   # cropland
    5: (185, 170, 155),   # built_up — warm grey
    6: (220, 200, 165),   # bare_sparse — parchment tan
    7: ( 90, 160, 155),   # flooded_wetland
    8: (240, 245, 250),   # snow_ice — off-white
}

_SATELLITE = {
    0: ( 40,  80, 130),   # water — deep blue
    1: ( 25,  80,  30),   # trees — dark green
    2: (110, 130,  55),   # shrubland
    3: (140, 165,  75),   # grassland
    4: (175, 160,  55),   # cropland
    5: (130, 120, 110),   # built_up
    6: (185, 170, 135),   # bare_sparse
    7: ( 50, 120, 115),   # flooded_wetland
    8: (220, 230, 235),   # snow_ice
}

PALETTES = {
    "flat": _FLAT,
    "illustrated": _ILLUSTRATED,
    "satellite": _SATELLITE,
}

# Border colour and width per style
BORDER_STYLE = {
    "flat":         ((100, 100, 100), 1),
    "illustrated":  (( 80,  65,  45), 2),
    "satellite":    (None,             0),   # no border
}

# Background fill (for gaps / ocean areas beyond cells)
BACKGROUND = {
    "flat":         (200, 220, 240),
    "illustrated":  (210, 195, 165),   # parchment
    "satellite":    ( 20,  50,  90),
}


def _xy(pt):
    """Return ``(x, y)`` floats for a GeoJSON position, or ``None`` if the
    element is malformed in *any* way.

    Single structural chokepoint mirroring ``label._xy``: rejects
    non-sequences, short sequences, and non-numeric ordinates uniformly so
    no element shape (missing z, scalar, string, empty) can reach the
    arithmetic / ``min``/``max`` path and abort the source (CR-01 /
    WR-01 / T-02-09).
    """
    if not isinstance(pt, (list, tuple)) or len(pt) < 2:
        return None
    try:
        return float(pt[0]), float(pt[1])
    except (TypeError, ValueError):
        return None


def _rings(geom):
    """Coordinate rings of a Polygon/MultiPolygon; ``[]`` for anything
    else / malformed so callers skip rather than crash (CR-03)."""
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


# Decompression-bomb-analogue guard for the synthetic render path
# (threat parity with T-02-04 / label._MAX_CANVAS_DIM).
_MAX_CANVAS_DIM = 20000


def _canvas_dims(min_x, min_y, max_x, max_y):
    """Return a validated (width, height), guarding degenerate/huge canvases."""
    width = int(max_x - min_x) + 1
    height = int(max_y - min_y) + 1
    if width <= 0 or height <= 0:
        raise ValueError(f"degenerate canvas {width}x{height} from feature bbox")
    if width > _MAX_CANVAS_DIM or height > _MAX_CANVAS_DIM:
        raise ValueError(
            f"canvas {width}x{height} exceeds max {_MAX_CANVAS_DIM}px "
            f"(pathological coordinates — refusing to allocate)"
        )
    return width, height


def _bbox(features):
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


def render_style(features, min_x, min_y, width, height, style: str) -> Image.Image:
    palette = PALETTES[style]
    border_color, border_width = BORDER_STYLE[style]
    bg = BACKGROUND[style]

    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)

    for feat in features:
        # Per-feature structural resilience: any malformed feature/ring —
        # regardless of element shape (missing z, scalar, string, empty
        # ring) — is skipped, never aborts the source (CR-01 / T-02-09 /
        # D-13). One try/except replaces the Nth element-shape patch.
        try:
            props = (feat or {}).get("properties") or {}
            try:
                h = int(props["height"])
                biome = int(props["biome"])
            except (KeyError, TypeError, ValueError):
                continue  # skip malformed feature (CR-03), do not abort source
            lc_idx = h_to_landcover(h, biome)
            fill = palette[lc_idx]

            for ring in _rings((feat or {}).get("geometry") or {}):
                try:
                    # _xy defensively coerces each position and rejects any
                    # element that is not a real (x, y) pair, so no element
                    # shape can crash the draw loop (CR-01 / WR-01).
                    coords = []
                    for pt in ring:
                        xy = _xy(pt)
                        if xy is not None:
                            coords.append((xy[0] - min_x, xy[1] - min_y))
                    if len(coords) < 3:
                        continue
                    draw.polygon(coords, fill=fill)
                    if border_color and border_width > 0:
                        draw.polygon(
                            coords, outline=border_color, width=border_width
                        )
                except (TypeError, ValueError, KeyError, IndexError):
                    continue
        except (TypeError, ValueError, KeyError, IndexError):
            continue

    if style == "satellite":
        # Slight blur to mimic sensor smoothing
        img = img.filter(ImageFilter.GaussianBlur(radius=0.8))

    return img


def render_one(geojson_path: str | Path, style: str) -> Image.Image:
    """Render a single style and return the PIL image (no disk write).

    Used by the per-(source×style) build so the rendered image can be handed
    to ``label.make_labels`` as ``image.png`` alongside the shared label masks.
    """
    geojson_path = Path(geojson_path)
    with open(geojson_path) as f:
        data = json.load(f)
    features = data["features"]

    min_x, min_y, max_x, max_y = _bbox(features)
    width, height = _canvas_dims(min_x, min_y, max_x, max_y)

    if style not in PALETTES:
        raise ValueError(f"Unknown style '{style}'")
    return render_style(features, min_x, min_y, width, height, style)


def render_map(geojson_path: str | Path, output_dir: str | Path, styles=("flat", "illustrated", "satellite")) -> None:
    geojson_path = Path(geojson_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(geojson_path) as f:
        data = json.load(f)
    features = data["features"]

    min_x, min_y, max_x, max_y = _bbox(features)
    width, height = _canvas_dims(min_x, min_y, max_x, max_y)

    for style in styles:
        if style not in PALETTES:
            print(f"Unknown style '{style}', skipping.")
            continue
        img = render_style(features, min_x, min_y, width, height, style)
        out_path = output_dir / f"{style}.png"
        img.save(out_path)
        print(f"Saved {style} render → {out_path}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 2:
        print("Usage: python render.py <geojson_path> <output_dir> [style ...]")
        sys.exit(1)
    geojson_path, output_dir = args[0], args[1]
    styles = args[2:] if len(args) > 2 else list(PALETTES.keys())
    render_map(geojson_path, output_dir, styles)
