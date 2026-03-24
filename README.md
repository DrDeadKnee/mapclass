# MapClass

Detecting and classifying terrain regions in map images, targeting grand strategy style campaigns at 100–2000 km scale. Multiple map image styles should be supported, from illustrated grand strategy aesthetics (EU4/CK3/HoI4 style) to satellite-derived imagery.

The output is a **dense pixel-level prediction**, assigning two distinct attributes to every pixel:

- **Land cover** (what covers the surface): 9-class canonical taxonomy derived from two GEE products (ESA WorldCover and Dynamic World), harmonised into a single label set — water, trees, shrubland, grassland, cropland, built-up, bare/sparse, flooded/wetland, snow/ice. ESA Mangroves fold into Trees; ESA Moss/lichen folds into Bare/sparse.
- **Topography** (shape of the terrain): derived from GEE elevation data (SRTM / ASTER DEM) — flat, hilly, mountainous

Illustrated is the primary target domain. Satellite data serves as a source of ground-truth labels and geophysically realistic terrain adjacency statistics, which encode valid co-occurrence priors (e.g. ocean borders coast borders lowland).

The pixel-level predictions are designed to support downstream region delineation (tracing contiguous same-class areas into polygons), which is handled in a separate project.


## Project Plan
This project will progress in phases:

1. **Build a dataset of pixel-label pairs.** Sources:
    - **Historical illustrated maps** — 16th–17th century maps (Ortelius, Mercator, Blaeu school) are the direct aesthetic ancestor of grand strategy game cartography and provide authentic illustrated style that synthetic generation cannot replicate. Labels are derived by registering each map to modern coordinates and overlaying ESA WorldCover (land cover) and SRTM (topography). Labels are applied with class-conditional per-source loss weights reflecting temporal reliability: topography (flat/hilly/mountainous) is fully trusted (geology is stable); water/coastlines are mostly trusted at regional scale; trees, built-up, and cropland are heavily downweighted (land use has shifted substantially since the 16th–17th century). Bare/sparse and snow/ice are treated as reliable. The **David Rumsey Map Collection** is the primary source — thousands of high-resolution scans, many already georeferenced as GeoTIFF. Unregistered maps are warped via rubber-sheet transformation using manually placed ground control points (GCPs) in QGIS/GDAL. Coverage is restricted to regional-scale maps (100–2000 km extent); city plans and large-scale surveys are excluded. This source primarily strengthens the topography head and increases illustrated-domain style diversity.
    - **Synthetic illustrated maps** — use an agentic model to generate maps via Azgaar's Fantasy Map Generator and a custom Pillow-based renderer. Because the agent controls asset placement, ground truth pixel annotations (land cover + topography) are generated automatically in the same pass. Multiple rendering styles are used to maximise visual diversity. Azgaar's biome taxonomy is mapped to the canonical 9-class land cover taxonomy; cropland, built-up, and flooded/wetland are absent from synthetic maps and appear only in satellite-derived data (up-weighted in training). Topography labels are derived from Azgaar's normalised heightmap: raw height h ∈ [20, 100] is re-normalised to [0, 100] before applying thresholds (flat ≤ 20, hilly 20–55, mountainous > 55); calibration against SRTM statistics is left as future work.
    - **Satellite-derived imagery** — use both ESA WorldCover and Dynamic World from GEE, mapped to the canonical 9-class taxonomy, combined with SRTM/ASTER DEM topography at the appropriate regional scale (100–2000 km). Satellite adjacency statistics provide geophysically grounded priors on terrain co-occurrence.

2. **Build a dense semantic segmentation pipeline.**
    - Regional context is critical: the identity of a region depends as much on its neighbors as on its own appearance. Classifying regions in isolation will not work well.
    - The preferred architecture is **dense semantic segmentation** using a frozen pre-trained backbone with two output heads (land cover + topography). The primary candidate is **DINOv2** (self-supervised ViT; strong dense-prediction transfer with minimal fine-tuning, fully frozen backbone). The benchmark alternative backbone is **Swin Transformer** (hierarchical ViT; natively produces multi-scale feature maps suited to coarse-to-fine inference). Both share a frozen backbone and train only a lightweight segmentation head.
    - Coarse-to-fine inference is retained as a multi-scale strategy: coarse feature maps establish global context; finer maps sharpen boundaries. This is native to Swin's architecture and implementable as a multi-resolution sliding window with DINOv2.
    - A large first-kernel ConvNet trained from scratch remains a stretch-goal benchmark. See `notes/architectural_references.md` for discussion and citations.

3. **Fine-tune and evaluate the model, then upload to HuggingFace.**
    - **Test set**: held-out synthetic maps (same generation pipeline, unseen during training).
    - **Metric**: joint per-pixel negative log-likelihood — `-(log p_land_cover + log p_topography)` averaged over pixels. Land cover and topography are not statistically orthogonal (e.g. water is always flat, cropland is rarely mountainous), so a joint metric is more appropriate than evaluating heads independently.
    - Hand-annotation of real grand strategy map images is deferred; Label Studio and CVAT are candidate tools if a real-domain test set is added later.
