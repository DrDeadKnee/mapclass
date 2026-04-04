"""
Map hex tile filenames to canonical land cover and topography class labels.

Filename convention: hex_{terrain}_{variant}_{style}_{n}.png
The terrain prefix uniquely identifies the class pair (land_cover, topo).

Label indices match biome_mapping.py:
  Land cover: 0=water, 1=trees, 2=shrubland, 3=grassland, 4=cropland,
              5=built_up, 6=bare_sparse, 7=flooded_wetland, 8=snow_ice
  Topography: 0=flat, 1=hilly, 2=mountainous
  Topography None = not applicable (water tiles)
"""

from pathlib import Path

from biome_mapping import LANDCOVER_CLASSES, TOPO_CLASSES

# ---------------------------------------------------------------------------
# Prefix → (land_cover_idx, topo_idx | None)
# Sorted longest-first so more specific prefixes match before shorter ones.
# ---------------------------------------------------------------------------

_RAW_MAP: list[tuple[str, int, int | None]] = [
    # Water
    ("hex_water_ocean",            0, None),
    ("hex_water_swamp",            7, 0),     # flooded wetland, flat
    ("hex_coast_beach",            6, 0),     # bare/sparse (sand), flat

    # Wetlands
    ("hex_wetlands_damp",          7, 0),     # flooded wetland, flat

    # Snow / ice  (check snowy variants before forest/hills/mountains)
    ("hex_snow",                   8, 0),     # snow field, flat
    ("hex_sparse_trees_snowy",     8, 0),     # snow-covered sparse trees → snow_ice, flat

    # Forest
    ("hex_forest_conifer_snowy",   1, 2),     # snowy conifers → trees, mountainous
    ("hex_forest_conifer_lush",    1, 1),     # conifer forest → trees, hilly
    ("hex_forest_deciduous_lush",  1, 0),     # deciduous → trees, flat
    ("hex_forest_mixed_lush",      1, 0),     # mixed forest → trees, flat
    ("hex_sparse_trees_lush",      1, 0),     # sparse trees → trees, flat
    ("hex_hill_with_tree",         1, 1),     # treed hill → trees, hilly

    # Mountains (snowy/rocky/lush variants; snowy checked before lush)
    ("hex_mountains_foothills_snowy",  8, 1),
    ("hex_mountains_foothills_rocky",  6, 1),
    ("hex_mountains_foothills_lush",   3, 1),  # foothills w/ grass → grassland, hilly
    ("hex_mountains_low_snowy",        8, 2),
    ("hex_mountains_low_rocky",        6, 2),
    ("hex_mountains_low_lush",         1, 2),
    ("hex_mountains_medium_snowy",     8, 2),
    ("hex_mountains_medium_rocky",     6, 2),
    ("hex_mountains_medium_lush",      1, 2),
    ("hex_mountains_peak_snowy",       8, 2),
    ("hex_mountains_peak_rocky",       6, 2),
    ("hex_mountains_peak_lush",        6, 2),  # peaks are rocky even if "lush"-labelled
    ("hex_mountain_volcano_snowy",     8, 2),
    ("hex_mountain_volcano_rocky",     6, 2),
    ("hex_mountain_volcano_lush",      1, 2),

    # Hills
    ("hex_hills_snowy",            8, 1),
    ("hex_hills_desert",           6, 1),    # bare/sparse, hilly
    ("hex_hills_lush",             3, 1),    # grassland, hilly

    # Plains
    ("hex_plains_farmland",        4, 0),    # cropland, flat
    ("hex_plains_desert",          6, 0),    # bare/sparse, flat
    ("hex_plains_damp",            3, 0),    # grassland (damp meadow), flat
    ("hex_plains_lush",            3, 0),    # grassland, flat

    # Urban
    ("hex_urban_farmland",         4, 0),    # cropland, flat
    ("hex_urban_farm",             4, 0),    # cropland, flat
    ("hex_urban_city",             5, 0),    # built_up, flat
    ("hex_urban_modern_town",      5, 0),    # built_up, flat
    ("hex_urban_monastery",        5, 0),    # built_up, flat
    ("hex_urban_tower",            5, 0),    # built_up, flat
    ("hex_urban_town",             5, 0),    # built_up, flat
]

# Sort longest prefix first so specific entries shadow shorter ones
_PREFIX_MAP: list[tuple[str, int, int | None]] = sorted(
    _RAW_MAP, key=lambda x: len(x[0]), reverse=True
)


def label_for_filename(filename: str) -> tuple[int, int | None] | None:
    """
    Return (land_cover_idx, topo_idx) for a hex tile filename, or None if
    the file has no terrain class (decorative UI element, river, path, etc.).
    """
    stem = Path(filename).stem
    for prefix, lc, topo in _PREFIX_MAP:
        if stem.startswith(prefix):
            return lc, topo
    return None


def load_labeled_tiles(toon_dir: str | Path) -> list[tuple[Path, int, int | None]]:
    """
    Scan toon_dir for hex tiles, assign labels, and return a list of
    (path, land_cover_idx, topo_idx) tuples.

    Files with no recognised terrain prefix are silently skipped.
    """
    toon_dir = Path(toon_dir)
    results = []
    for path in sorted(toon_dir.rglob("hex_*.png")):
        label = label_for_filename(path.name)
        if label is not None:
            lc, topo = label
            results.append((path, lc, topo))
    return results


# ---------------------------------------------------------------------------
# Text description utilities for PaliGemma fine-tuning
# ---------------------------------------------------------------------------

# Human-readable terrain descriptions for (lc, topo) pairs.
# Multiple alternatives per pair provide prompt diversity.
TERRAIN_DESCRIPTIONS: dict[tuple[int, int | None], list[str]] = {
    (0, None): ["ocean", "sea", "open water", "coastal water"],
    (1, 0):    ["flat woodland", "lowland forest", "deciduous forest"],
    (1, 1):    ["forested hills", "wooded highlands", "hillside forest"],
    (1, 2):    ["mountain forest", "alpine forest", "forested mountains"],
    (2, 0):    ["shrubland", "scrubland", "low bushland"],
    (2, 1):    ["scrubby hills", "shrubby highlands"],
    (3, 0):    ["grassland", "plains", "flat meadow", "open fields"],
    (3, 1):    ["rolling hills", "grassy hills", "verdant highlands"],
    (4, 0):    ["farmland", "cropland", "agricultural fields"],
    (5, 0):    ["town", "city", "urban settlement", "built-up area"],
    (6, 0):    ["desert", "arid plains", "bare terrain", "sandy flats"],
    (6, 1):    ["rocky hills", "barren highlands", "desert hills"],
    (6, 2):    ["rocky mountains", "barren peaks", "high rocky mountains"],
    (7, 0):    ["wetland", "swamp", "marsh", "flooded terrain"],
    (7, 1):    ["wetland", "swamp"],    # swamp tiles classified as flooded_wetland
    (8, 0):    ["snow field", "snow plain", "ice field"],
    (8, 1):    ["snowy hills", "snow-covered highlands"],
    (8, 2):    ["snowy mountains", "snow-capped peaks", "alpine snowfield"],
}

PROMPT_TEMPLATES: list[str] = [
    "What terrain type does this illustrated map tile show?",
    "Identify the land cover and topography in this map tile.",
    "What terrain is depicted in this fantasy map illustration?",
    "Describe the terrain type shown in this map tile.",
    "What kind of landscape does this map tile represent?",
]


def label_to_answer(lc: int, topo: int | None, description_idx: int = 0) -> str:
    """
    Return a short natural-language answer for a (lc, topo) label pair.
    description_idx selects among synonyms for variety.
    """
    lc_name = LANDCOVER_CLASSES[lc]
    descs = TERRAIN_DESCRIPTIONS.get((lc, topo), [lc_name])
    desc = descs[description_idx % len(descs)]

    if topo is not None:
        topo_name = TOPO_CLASSES[topo]
        return f"{desc} ({topo_name})"
    return desc


if __name__ == "__main__":
    import sys
    toon_dir = sys.argv[1] if len(sys.argv) > 1 else "data/toons"
    tiles = load_labeled_tiles(toon_dir)
    print(f"{len(tiles)} labeled tiles found in {toon_dir}")
    from collections import Counter
    lc_counts = Counter(lc for _, lc, _ in tiles)
    for lc, n in sorted(lc_counts.items()):
        print(f"  {LANDCOVER_CLASSES[lc]:20s}: {n}")
