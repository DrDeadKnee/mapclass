# MapClass

Detecting and classifying terrain regions in map images, targeting grand strategy style campaigns at 100–2000 km scale. Multiple map image styles should be supported, from illustrated grand strategy aesthetics (EU4/CK3/HoI4 style) to satellite-derived imagery.

The output is **region-level polygons**, each labeled with two orthogonal attributes:

- **Land cover** (what covers the surface): drawn from Google Earth Engine land cover products (e.g. ESA WorldCover or Dynamic World) — water, trees, grassland, shrubland, cropland, built-up, bare ground, flooded vegetation, snow/ice, etc.
- **Topography** (shape of the terrain): derived from GEE elevation data (SRTM / ASTER DEM) — flat, hilly, mountainous

Illustrated is the primary target domain. Satellite data serves as a source of ground-truth labels and geophysically realistic terrain adjacency statistics, which encode valid co-occurrence priors (e.g. ocean borders coast borders lowland).


## Project Plan
This project will progress in phases:

1. **Build a dataset of region-polygon/label pairs.** Sources:
    - **Synthetic illustrated maps** — use an agentic model to generate maps via tools such as Azgaar's Fantasy Map Generator or a custom Pillow-based renderer. Because the agent controls asset placement, ground truth polygon annotations (land cover + topography) are generated automatically in the same pass.
    - **Satellite-derived imagery** — use GEE land cover products (ESA WorldCover, Dynamic World) combined with SRTM/ASTER DEM topography at the appropriate regional scale (100–2000 km). Satellite adjacency statistics provide geophysically grounded priors on terrain co-occurrence.

2. **Build a region segmentation and classification pipeline.**
    - Regional context is critical: the identity of a region depends as much on its neighbors as on its own appearance. Classifying regions in isolation will not work well.
    - The preferred architecture is **recursive coarse-to-fine segmentation** using a frozen pre-trained backbone (e.g. SAM or CLIP ViT): divide the map into coarse tiles, obtain per-class probability estimates, use these as priors when segmenting at the next finer scale. This approach is more compute-efficient than training a large-kernel convnet from scratch, and leverages existing pre-trained representations.
    - An alternative — a novel convnet with a large first convolution kernel to capture broad spatial context — remains worth benchmarking if compute allows. See `notes/architectural_references.md` for discussion and citations.

3. **Fine-tune and evaluate the model, then upload to HuggingFace.**
