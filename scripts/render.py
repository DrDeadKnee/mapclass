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


def _bbox(features):
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


def _rings(geom):
    if geom["type"] == "Polygon":
        return geom["coordinates"]
    return [ring for poly in geom["coordinates"] for ring in poly]


def render_style(features, min_x, min_y, width, height, style: str) -> Image.Image:
    palette = PALETTES[style]
    border_color, border_width = BORDER_STYLE[style]
    bg = BACKGROUND[style]

    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)

    for feat in features:
        props = feat["properties"]
        h = int(props["height"])
        biome = int(props["biome"])
        lc_idx = h_to_landcover(h, biome)
        fill = palette[lc_idx]

        for ring in _rings(feat["geometry"]):
            coords = [(x - min_x, y - min_y) for x, y in ring]
            if len(coords) < 3:
                continue
            draw.polygon(coords, fill=fill)
            if border_color and border_width > 0:
                draw.polygon(coords, outline=border_color, width=border_width)

    if style == "satellite":
        # Slight blur to mimic sensor smoothing
        img = img.filter(ImageFilter.GaussianBlur(radius=0.8))

    return img


def render_map(geojson_path: str | Path, output_dir: str | Path, styles=("flat", "illustrated", "satellite")) -> None:
    geojson_path = Path(geojson_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(geojson_path) as f:
        data = json.load(f)
    features = data["features"]

    min_x, min_y, max_x, max_y = _bbox(features)
    width = int(max_x - min_x) + 1
    height = int(max_y - min_y) + 1

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
