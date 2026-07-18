# MapClass

Inferring terrain characteristics for every region of a **hand-drawn map**. Given an illustrated / fantasy / historical map image, produce a dense field of terrain labels (per ~10 km hex) suitable for driving grand-strategy-style campaigns at 100–2000 km scale.

## Architecture (2026-07 reboot)

Hand-drawn maps carry *sparse, semantic* evidence: a sea name floating in blank space, a coastline stroke hundreds of pixels away, stipple marks meaning forest. The design splits the problem into two kinds of knowledge that want different data:

```
hand-drawn map ──► anchor extractor ──► sparse soft hexes ──► terrain prior ──► dense terrain field
                   (OCR'd place names,    (clamped as            (iterative masked-hex
                    coastlines, symbols)   evidence)              decoding fills the rest)
```

1. **Terrain prior** ("terrain physics") — a masked-hex prediction model over H3 hexes with **soft labels**: each hex carries a probability distribution over terrain classes, computed as area fractions of ESA WorldCover land cover × Copernicus DEM slope classes. Trained with high, variable mask ratios (50–95%, contiguous blobs) and a KL loss, so it learns to diffuse terrain from *sparse* evidence — real-Earth adjacency statistics (ocean borders coast, coast borders lowland) at effectively infinite data scale.
2. **Anchor extraction** ("reading the map") — turns ink into sparse, high-confidence terrain anchors. No labeled hand-drawn data exists, so this is bootstrapped from (a) **synthetic rendering**: terrain hex fields rendered as hand-drawn-style maps with free pixel-perfect labels, and (b) **VLM pseudo-labeling** of real scraped maps. A small hand-labeled set (~10–20 maps) is reserved for evaluation only.
3. **Fusion** is native to the prior: clamp anchor hexes as (possibly soft) evidence, iteratively decode the masked remainder.

### Taxonomy

- **Land cover** (9 classes): water, trees, shrubland, grassland, cropland, built-up, bare/sparse, flooded/wetland, snow/ice — from ESA WorldCover (S3, no GEE).
- **Topography** (3 classes): flat (<2°), hilly (2–15°), mountainous (>15°) — from Copernicus DEM GLO-30 slope.
- Hex soft labels live over the joint (land cover × topography) space. The visual hex-tile vocabulary (`gs://mapclass-training-northeast1/data/toons/`) maps onto this taxonomy via `scripts/toon_mapping.py`.

## Repository layout

| Path | Purpose |
|---|---|
| `scripts/hexprior/` | Phase A: WorldCover + DEM → H3 hex soft-label dataset; masked-hex prior model |
| `scripts/historical/` | Geo data fetch: WorldCover, Copernicus DEM, Rumsey/IIIF/Allmaps historical maps, georeferencing |
| `scripts/satellite/` | STAC scene lookup, COG fetch, WorldCover coverage/diversity ranking (region picker) |
| `scripts/gcs_io.py` | GCS dataset layout conventions and IO |
| `scripts/render.py`, `scripts/augment.py` | Azgaar GeoJSON → styled raster rendering + parchment/faded augmentation (seed of the synthetic hand-drawn renderer) |
| `scripts/biome_mapping.py`, `scripts/toon_mapping.py` | Canonical taxonomy and hex-tile vocabulary mapping |

Data and models live in `gs://mapclass-training-northeast1/` (`data/`, `models/`).

## History

Previous iterations pursued pixel-level semantic segmentation (PaliGemma/SigLIP backbone, coarse-to-fine decoding). That code was removed in the 2026-07 reboot — see git history before the `reboot` commit if needed. The surviving modules above are approach-agnostic infrastructure.

## Dev notes

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -x -q          # offline unit tests
.venv/bin/python -m pytest tests/integration -q  # live-network tests
```

Jupyter-on-GCP workflow: start `jupyter lab --no-browser --port 8888 --ip 127.0.0.1` on the VM, then `gcloud compute ssh <vm_name> -- -N -L 8888:127.0.0.1:8888` locally.
