# MapClass

**Project code:** MAPCLASS

## What This Is

MapClass is a research project that trains a dense pixel-level terrain segmentation
model for grand-strategy-style map imagery at 100–2000 km regional scale. Each pixel
receives two independent labels — land cover (9 canonical classes) and topography
(3 slope-derived classes). It is built and used by a single researcher; the
deliverable is a fine-tuned model published on HuggingFace, not a production service.

## Core Value

A trained segmentation model on HuggingFace that produces dense per-pixel
land-cover and topography predictions on illustrated regional maps, evaluated by
joint per-pixel NLL = -(log p_land_cover + log p_topography) on held-out
synthetic maps.

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

- ✓ Phase 1: PaliGemma-3B LoRA fine-tune on ~120 hex tile PNGs across 9 land cover
  and 3 topography classes, augmented to ~4,760 samples — `scripts/finetune_paligemma.py`
- ✓ Phase 2: Labelled pixel-pair dataset from three source streams — historical
      (Rumsey via Allmaps→IIIF→GeoTIFF; PaliGemma-semi-auto + manual fallback
      deferred to v2 as GEOREF-V2), synthetic (Azgaar+Pillow, per-(source×style),
      uniform-1.0 weights, frozen seeded stratified split at N=100 — EVAL-01),
      satellite (Sentinel-2/WorldCover/DEM from S3, balance-tilt weights), unified
      by a shared nested-pyramid tiler — Validated in Phase 2 (5/5 ROADMAP criteria,
      offline; online integration gate tracked in 02-HUMAN-UAT.md)

### Active

<!-- Current scope. Building toward these. -->

- [ ] Phase 3: Build the dense semantic segmentation pipeline on the Phase-1
      SigLIP encoder with land-cover + topography heads and coarse-to-fine inference
- [ ] Phase 4: Fine-tune segmentation end-to-end, evaluate on held-out synthetic
      maps with joint per-pixel NLL, and upload to HuggingFace

### Out of Scope

<!-- Explicit boundaries. Includes reasoning to prevent re-adding. -->

- City-plan and large-scale survey map imagery — extent restricted to 100–2000 km regional scale
- Polygonisation / region-delineation tooling — responsibility of a separate downstream project
- Hand annotation of real grand-strategy game map imagery for evaluation — deferred; synthetic test set only
- A production inference service — deliverable is the trained model on HuggingFace, nothing more
- Google Earth Engine as the primary label-fetching path — direct S3 access is the locked source
- Unfreezing the Gemma language model under any LoRA or full-weight scheme — guarantees zero forgetting of place names / geographic terminology
- LoRA on any SigLIP layer other than the attention projections — locked to `q_proj`, `k_proj`, `v_proj`, `out_proj`

## Context

Background that informs implementation:

- **Why context-aware segmentation.** Terrain identity depends on surroundings as
  much as local appearance. Geophysical co-occurrence statistics (oceans border
  coasts, coasts border lowlands, water is flat, cropland is rarely mountainous)
  are strong priors. This motivates dense (not tile-classification) prediction
  and a coarse-to-fine recursive design.
- **Preferred backbone direction (from DOC).** Recursive coarse-to-fine: divide
  the map into coarse tiles, get per-class distributions from a pre-trained
  backbone, use them as priors at the next finer scale. The SPEC locks the
  *primary* backbone to the Phase-1 fine-tuned SigLIP encoder; DINOv2 and Swin
  Transformer remain non-locked benchmark alternatives. A large first-kernel
  ConvNet from scratch (RepLKNet / SLaK / ConvNeXt direction) is a stretch goal.
- **Planned experiment — progressive backbone unfreezing.** If illustrated-map
  performance plateaus with a fully frozen backbone, progressively unfreeze from
  the top down and run attribution (GradCAM / attention rollout) at each stage
  to localise the domain gap inside the network. Informs whether the gap is
  low-level texture, mid-level composition, or high-level semantics.
- **Current execution state.** Phase 1 is complete (`scripts/finetune_paligemma.py`).
  Phase 2 is in progress: historical dataset construction
  (`scripts/build_historical_dataset.py`, `scripts/historical/rumsey.py`,
  Allmaps annotation index integration via `scripts/historical/allmaps.py`).
  Working branch is `phase1.5`; `scripts/historical/allmaps.py` is untracked
  and `scripts/historical/rumsey.py` is modified.
- **Reference reading** (from `notes/architectural_references.md`):
  Swin Transformer (arxiv 2103.14030), AerialFormer (2306.06842), DINOv2
  (2304.07193), Global Convolutional Network (1703.02719), RepLKNet (2203.06717),
  SLaK (2207.03620), ConvNeXt (2201.03545).

## Constraints

All ten constraints below are LOCKED — sourced from the SPEC (`README.md`,
classified high confidence). They are treated as immutable until superseded
by a future ADR.

- **NFR (product scope)**: Dense pixel-level prediction over grand-strategy-style
  map imagery at regional scale (100–2000 km extent). Two independent per-pixel
  labels: land cover and topography. City-plan and large-scale survey maps are
  out of scope.
- **Schema (land cover taxonomy)**: 9 canonical classes — water, trees, shrubland,
  grassland, cropland, built-up, bare/sparse, flooded/wetland, snow/ice. ESA
  Mangroves fold into Trees; ESA Moss/lichen folds into Bare/sparse. Cropland,
  built-up, and flooded/wetland are not produced by Azgaar synthetic maps and
  must be sourced from satellite imagery (up-weighted in training).
- **Schema (topography taxonomy)**: 3 classes derived from slope thresholds on
  Copernicus DEM GLO-30 — flat (<2°), hilly (2–15°), mountainous (>15°). For
  Azgaar synthetic maps, raw height `h ∈ [20, 100]` is re-normalised to `[0, 100]`
  before applying thresholds (flat ≤20, hilly 20–55, mountainous >55).
  Calibration of synthetic thresholds against SRTM statistics is deferred.
- **API contract (data sources)**: ESA WorldCover fetched from `s3://esa-worldcover`;
  Copernicus DEM GLO-30 fetched from `s3://copernicus-dem-30m`; Dynamic World
  (GEE) is an optional secondary land-cover source for temporal diversity.
  Historical maps from the David Rumsey Map Collection via the LUNA API
  (`scripts/build_historical_dataset.py search`). Regional-scale only (100–2000 km).
- **Protocol (PaliGemma fine-tuning)**: PaliGemma-3B — Gemma language model fully
  frozen (no LoRA, no base-weight updates); LoRA applied ONLY to SigLIP attention
  layers `q_proj`, `k_proj`, `v_proj`, `out_proj`; multi-modal projector fully
  trainable. Training data: `data/toons/` (~120 hex tile PNGs across 9 land cover
  + 3 topography classes), augmented into four visual styles
  (original, grayscale, parchment/sepia, faded), tiled into 2×2 region composites,
  paired with five natural-language prompt templates (~4,760 samples).
- **Protocol (georeferencing pipeline)**: Precedence — (1) Allmaps/Georeferencer
  for already-registered maps; (2) unregistered maps emitted to
  `unregistered_manifest.json`; (3) semi-automatic registration using the
  Phase-1 fine-tuned PaliGemma to generate dense terrain predictions, matched
  against a WorldCover + Copernicus DEM reference grid via cross-correlation
  (rigid) then refined with thin-plate spline warping; coastlines, mountain
  ranges, and major water bodies as anchor features. (4) Fallback: manual GCP
  placement via MapWarper or QGIS. Output: warped GeoTIFF in EPSG:4326.
- **Protocol (class-conditional loss weights)**: Historical-map labels weighted
  by temporal reliability — topography fully trusted; water/coastlines mostly
  trusted at regional scale; bare/sparse and snow/ice reliable; trees, built-up,
  and cropland heavily downweighted (land use has shifted since the 16th–17th c.).
  Per-map weights persisted in `sample_weights.json` alongside `image.png`,
  `land_cover.png`, and `topography.png`.
- **Protocol (segmentation backbone)**: Primary backbone — the Phase-1
  LoRA-adapted SigLIP encoder; two lightweight output heads (land cover 9-class,
  topography 3-class); coarse-to-fine inference via multi-resolution sliding
  window on SigLIP/DINOv2 or natively on Swin. Benchmark alternatives
  (non-locked): DINOv2, Swin Transformer. Stretch goal: large first-kernel
  ConvNet trained from scratch.
- **Protocol (evaluation metric)**: Held-out synthetic maps from the same
  Azgaar + Pillow pipeline used for training but unseen during training.
  Metric: joint per-pixel NLL = -(log p_land_cover + log p_topography) averaged
  over pixels. Rationale: the two heads are not statistically orthogonal
  (water is always flat, cropland is rarely mountainous), so a joint metric is
  more appropriate than evaluating heads independently. Hand-annotation of real
  grand-strategy imagery deferred (Label Studio / CVAT are candidates if added).
- **NFR (deliverable)**: Final deliverable is the fine-tuned segmentation model
  uploaded to HuggingFace. Pixel-level predictions are designed to support
  downstream region-delineation tooling (polygonisation), which is a separate
  project's responsibility.

<decisions>
The block below records each LOCKED constraint as an addressable decision.
Format mirrors what a future ADR promotion would look like.

- **DECISION-product-scope** (LOCKED, NFR): Dense pixel-level prediction at 100–2000 km
  regional scale, two labels (land cover, topography). City plans and large-scale
  surveys out of scope. *Source: README.md (SPEC).*
- **DECISION-land-cover-taxonomy** (LOCKED, schema): 9 canonical classes — water,
  trees, shrubland, grassland, cropland, built-up, bare/sparse, flooded/wetland,
  snow/ice. ESA Mangroves → Trees; ESA Moss/lichen → Bare/sparse. Cropland,
  built-up, flooded/wetland must come from satellite imagery (up-weighted).
  *Source: README.md.*
- **DECISION-topography-taxonomy** (LOCKED, schema): 3 classes from Copernicus DEM
  GLO-30 slope — flat <2°, hilly 2–15°, mountainous >15°. Azgaar height
  re-normalised from [20,100] to [0,100] before thresholding (flat ≤20, hilly
  20–55, mountainous >55). SRTM calibration deferred. *Source: README.md.*
- **DECISION-data-sources** (LOCKED, api-contract): ESA WorldCover from
  `s3://esa-worldcover`, Copernicus DEM GLO-30 from `s3://copernicus-dem-30m`,
  Dynamic World (GEE) optional secondary source. Historical maps from the David
  Rumsey Map Collection via LUNA API. Regional scale only. *Source: README.md.*
- **DECISION-paligemma-finetuning** (LOCKED, protocol): PaliGemma-3B — Gemma
  fully frozen; LoRA only on SigLIP `q_proj` / `k_proj` / `v_proj` / `out_proj`;
  multi-modal projector fully trainable. Training corpus ~4,760 samples from
  `data/toons/` augmented across 4 visual styles, 2×2 composites, 5 prompt
  templates. Entrypoint: `scripts/finetune_paligemma.py`. *Source: README.md.*
- **DECISION-georeferencing-pipeline** (LOCKED, protocol): Precedence Allmaps →
  unregistered manifest → PaliGemma semi-auto (cross-correlation + TPS refinement,
  coastline/range/water anchors) → manual GCP fallback (MapWarper / QGIS).
  Output GeoTIFF in EPSG:4326. *Source: README.md.*
- **DECISION-class-conditional-loss-weights** (LOCKED, protocol): Historical-map
  per-source weights — topography trusted; water/coastlines mostly trusted;
  bare/sparse and snow/ice reliable; trees / built-up / cropland heavily
  downweighted. Persisted in `sample_weights.json`. *Source: README.md.*
- **DECISION-segmentation-backbone** (LOCKED primary, non-locked alternatives):
  Primary backbone — Phase-1 LoRA-adapted SigLIP encoder with two lightweight
  heads (9-class land cover, 3-class topography); coarse-to-fine inference.
  Non-locked benchmark alternatives: DINOv2, Swin Transformer. Stretch goal:
  large first-kernel ConvNet from scratch. *Source: README.md.*
- **DECISION-evaluation-metric** (LOCKED, protocol): Joint per-pixel NLL
  = -(log p_land_cover + log p_topography) on held-out synthetic maps from the
  same Azgaar + Pillow pipeline. Real-domain hand annotation deferred.
  *Source: README.md.*
- **DECISION-deliverable** (LOCKED, NFR): Fine-tuned segmentation model uploaded
  to HuggingFace. Polygonisation / region delineation is a downstream project.
  *Source: README.md.*
</decisions>

## Key Decisions

<!-- Decisions that constrain future work. Add throughout project lifecycle. -->

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Freeze Gemma entirely; LoRA only on SigLIP attention projections | Guarantees zero forgetting of text understanding so place names and geographic terminology remain preserved at inference | — Pending (validated by Phase 1 training; downstream perf TBD) |
| Joint per-pixel NLL as the single metric | Land cover and topography are not orthogonal (water is flat, cropland rarely mountainous) — joint NLL captures the dependency | — Pending |
| Direct S3 fetch for ESA WorldCover + Copernicus DEM (no GEE in the critical path) | Removes a heavyweight dependency, keeps the data pipeline reproducible without API keys / quota | — Pending |
| Three-source dataset (historical + synthetic + satellite) with class-conditional weights | Each source has different reliability per class (historical land-use is stale; synthetic lacks built-up/cropland/wetland) — weighting reflects that honestly | — Pending |
| Coarse-to-fine inference over the Phase-1 SigLIP encoder, with DINOv2 / Swin as benchmark alternatives | Captures spatial context cheaply, reuses Phase-1 LoRA investment, keeps two strong baselines available | — Pending |
| HuggingFace upload as the only deliverable; no polygonisation | Keeps scope contained — region delineation is genuinely a separate problem | — Pending |
| **[REVERSED 2026-05-16, 02-02] D-06: tiles stream to GCS, not canonical-on-disk** | Root-cause of synthetic data loss: local-only pyramid writes on ephemeral compute were destroyed on preemption. tiling.py now writes pyramids through `_GCSWriter` with 32-thread pool. Local Path path preserved for tests/offline use. | — Applied in 02-02 |
| **[REVERSED 2026-05-16, 02-02] D-17: `gs://.../data/synthetic/{train,test}/` canonical; filesystem separation now transient scratch** | GCS is the canonical store for all synthetic dataset outputs. Local scratch is used only for rendering/labelling reads; all pyramid writes stream to GCS. | — Applied in 02-02 |
| **[REVERSED 2026-05-16, 02-02] D-18: split.json relocated to `gs://.../data/synthetic/split.json`; prior frozen split GONE; NEW split freezes on regenerated Azgaar sources** | Prior split.json (if any) referenced Azgaar raw inputs that were lost with ephemeral compute. The GCS-canonical split.json is frozen on first build; --refreeze-split is the only deliberate recompute path (Pitfall R-3). validate_manifest hard-fails BEFORE the split is frozen (RW-02). See 02-CONTEXT.md for full rework rationale. | — Applied in 02-02 |

---
*Last updated: 2026-05-16 — Phase 2 rework: D-06/D-17/D-18 REVERSED (GCS-canonical synthetic pipeline); see 02-02-SUMMARY.md*
