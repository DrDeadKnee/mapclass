"""
Phase A: terrain prior over H3 hexes with soft labels.

Pipeline: ESA WorldCover (land cover) + Copernicus DEM (slope topography)
→ per-pixel composite terrain classes on a coarse WGS84 grid
→ aggregated per H3 hex into soft labels (area-fraction distributions)
→ training data for a masked-hex prediction model (the terrain prior).
"""
