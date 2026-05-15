"""
Azgaar Fantasy Map Generator → canonical taxonomy mappings.

Azgaar biome IDs (0-21) and height values (h) are exported in GeoJSON
as properties 'biome' and 'height' respectively.
Height range: 0-19 = water, 20-100 = land.
"""

# fmt: off
AZGAAR_BIOMES = {
    0:  "Hot desert",
    1:  "Savanna",
    2:  "Tropical dry forest",
    3:  "Tropical wet forest",
    4:  "Xeric shrubland",
    5:  "Temperate dry grassland",
    6:  "Temperate wet grassland",
    7:  "Temperate deciduous forest",
    8:  "Subtropical rain forest",
    9:  "Cold desert",
    10: "Temperate rain forest",
    11: "Coniferous wet forest",
    12: "Temperate coniferous forest",
    13: "Subtaiga",
    14: "Boreal wet forest",
    15: "Boreal dry forest",
    16: "Subpolar scrub",
    17: "Subpolar desert",
    18: "Tundra",
    19: "Rocky desert",
    20: "Polar desert",
    21: "Glacier",
}

# Canonical 9-class land cover taxonomy (class index → name).
# Classes 4 (cropland), 5 (built_up), 7 (flooded_wetland) do not appear in
# synthetic data; they come from satellite-derived data only.
LANDCOVER_CLASSES = [
    "water",           # 0
    "trees",           # 1
    "shrubland",       # 2
    "grassland",       # 3
    "cropland",        # 4  — satellite only
    "built_up",        # 5  — satellite only
    "bare_sparse",     # 6
    "flooded_wetland", # 7  — satellite only
    "snow_ice",        # 8
]
LANDCOVER_IDX = {name: i for i, name in enumerate(LANDCOVER_CLASSES)}

TOPO_CLASSES = ["flat", "hilly", "mountainous"]
TOPO_IDX = {name: i for i, name in enumerate(TOPO_CLASSES)}

# Azgaar biome ID → canonical land cover name.
# Water cells (h < 20) are handled separately by h_to_landcover().
BIOME_TO_LANDCOVER_NAME = {
    0:  "bare_sparse",   # Hot desert
    1:  "grassland",     # Savanna
    2:  "trees",         # Tropical dry forest
    3:  "trees",         # Tropical wet forest
    4:  "shrubland",     # Xeric shrubland
    5:  "grassland",     # Temperate dry grassland
    6:  "grassland",     # Temperate wet grassland
    7:  "trees",         # Temperate deciduous forest
    8:  "trees",         # Subtropical rain forest
    9:  "bare_sparse",   # Cold desert
    10: "trees",         # Temperate rain forest
    11: "trees",         # Coniferous wet forest
    12: "trees",         # Temperate coniferous forest
    13: "trees",         # Subtaiga
    14: "trees",         # Boreal wet forest
    15: "trees",         # Boreal dry forest
    16: "shrubland",     # Subpolar scrub
    17: "bare_sparse",   # Subpolar desert
    18: "bare_sparse",   # Tundra
    19: "bare_sparse",   # Rocky desert
    20: "bare_sparse",   # Polar desert
    21: "snow_ice",      # Glacier
}
# fmt: on

H_SEA_LEVEL = 20  # cells with h < H_SEA_LEVEL are water


def normalize_land_h(h: int) -> float:
    """Re-normalize Azgaar land height [20, 100] → [0, 100]."""
    return (h - H_SEA_LEVEL) / (100 - H_SEA_LEVEL) * 100


def h_to_landcover(h: int, biome: int) -> int:
    """Map (h, biome) → canonical land cover class index."""
    if h < H_SEA_LEVEL:
        return LANDCOVER_IDX["water"]
    return LANDCOVER_IDX[BIOME_TO_LANDCOVER_NAME[biome]]


def h_to_topo(h: int) -> int | None:
    """Map raw Azgaar h → topography class index. Returns None for water cells.

    ROADMAP SC#3 LOCKED boundaries over the normalized [0,100] domain:
    flat ≤20, hilly 20–55 (55 inclusive), mountainous >55.

    ``normalize_land_h`` is integer-derived, so its theoretical value is exact
    at every reachable point; the float division can introduce ~1e-14 error
    (e.g. h=64 → 55.00000000000001) that would otherwise mis-bin a cell
    sitting exactly on the hilly/mountainous cut into mountainous. Round to a
    precision far finer than any reachable spacing (the smallest step is
    100/80 = 1.25 per unit of h) before the locked comparison so the inclusive
    upper bounds hold exactly.
    """
    if h < H_SEA_LEVEL:
        return None
    norm = round(normalize_land_h(h), 6)
    if norm <= 20:
        return TOPO_IDX["flat"]
    elif norm <= 55:
        return TOPO_IDX["hilly"]
    else:
        return TOPO_IDX["mountainous"]
