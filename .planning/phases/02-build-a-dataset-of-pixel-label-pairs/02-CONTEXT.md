# Phase 2: Build a dataset of pixel-label pairs - Context

**Gathered:** 2026-05-15
**Status:** Ready for planning

<domain>
## Phase Boundary

Produce labelled pixel-pair training data from three source families and a held-out synthetic test set:

1. **Historical illustrated maps** — 16th–17th c. David Rumsey collection, georeferenced via Allmaps (community-contributed W3C Web-Annotation index), labelled by ESA WorldCover + Copernicus DEM with class-conditional per-source loss weights already locked in `scripts/historical/label.py`.
2. **Synthetic illustrated maps** — Azgaar Fantasy Map Generator GeoJSON exports rendered by `scripts/render.py` (Pillow), labelled automatically from the same export with the canonical 9-class / 3-class taxonomy.
3. **Satellite-derived imagery** — Sentinel-2 L2A RGB from `s3://sentinel-cogs` paired with ESA WorldCover labels + Copernicus DEM topography, class-diversity stratified to oversample cropland / built-up / flooded-wetland.

Each map of every family ships `image.png`, `land_cover.png`, `topography.png`, `sample_weights.json` plus the multi-scale nested-pyramid tile decomposition.

A held-out synthetic test set is reserved for Phase 4 evaluation and is *guaranteed* unseen during training by filesystem-level separation.

**Explicitly out of scope for this phase plan:**

- PaliGemma-driven semi-automatic georeferencing of the Allmaps-missing maps — deferred to v2 (see Deferred Ideas).
- Manual MapWarper / QGIS GCP placement of individual maps — deferred to v2 (same v2 bucket as PaliGemma path).
- Any modification to the locked taxonomies, locked data sources, or locked historical loss weights — those come from the SPEC (`README.md`) and are recorded as `DECISION-*` entries in `PROJECT.md`.

</domain>

<decisions>
## Implementation Decisions

### Allmaps wiring (historical pipeline)

- **D-01: Delete WMS code path entirely.** LUNA never exposes WMS URLs or bounding boxes — the old `_wms_url`, `_parse_bbox`, `_download_wms_geotiff` functions are dead code. Remove them. The historical pipeline flows LUNA → Allmaps → IIIF image → GeoTIFF only.
- **D-02: Fixed 4096 max-edge IIIF fetch.** Request `/full/!4096,4096/0/default.jpg`. Scale the Allmaps GCPs (which are in original IIIF pixel coords) by the same factor before writing the GeoTIFF.
- **D-03: Affine least-squares fit over all Allmaps GCPs.** Solve a 6-parameter affine; write as the GeoTIFF's `transform` via `rasterio`. No TPS pre-warp at this stage. `historical/label.py` works directly off `ds.crs` + `ds.transform` and needs no resampling layer.
- **D-04: Failure handling — drop out-of-scale silently, manifest the rest.** Maps whose GCP-derived bbox diagonal falls outside [100, 2000] km are dropped without entering the manifest. Maps with `not_in_allmaps` (404) or `gcps_insufficient` (<3 GCPs) are emitted to `unregistered_manifest.json` with a `status` field distinguishing reason.
- **D-05: Surface per-reason drop counts at end of `cmd_search`.** Print drop counts for `out_of_scale`, `not_in_allmaps`, `gcps_insufficient`, `download_failed`. A high `out_of_scale` rate (e.g. >50% of search results) is a signal to revisit the scale window — Phase 2 is not "done" if the cleanup is hiding most of the dataset.

### Sample yield (all three source families)

- **D-06: Pre-tile at build time, with overlap.** Tiles are materialised on disk during `build_*_dataset.py`. The training DataLoader reads by index, not by random crop.
- **D-07: Multi-scale nested pyramids.** Each pyramid contains 1×(896×896) + 4×(448×448) + 16×(224×224) tiles in strict 2×2 spatial nesting. A given 896 region's prediction maps directly to its 4 inner 448 predictions and 16 inner 224 predictions — exploitable by Phase 3's coarse-to-fine model.
- **D-08: Pyramid-to-pyramid stride 448 (50% top-scale overlap).** Within a pyramid, siblings at the same scale do **not** overlap. Across pyramids, the 896 footprints step by 448 across the source map. A 4096-px map yields ~64 pyramids × 21 tiles = ~1344 sample tiles.
- **D-09: Edge policy.** Drop pyramids whose 896 footprint falls more than 50% off the source map.

### Satellite-source RGB imagery

- **D-10: Satellite is a true third training source (image→label pairs).** The satellite stream is *not* label-only adjacency priors — the model is trained on Sentinel-2 RGB tiles paired with WorldCover labels alongside historical and synthetic samples. Cross-domain generalisation back to illustrated is part of the bet.
- **D-11: Sentinel-2 L2A RGB from `s3://sentinel-cogs`.** Anonymous S3, STAC catalogue, bands B04/B03/B02. Locked-against-GEE-as-primary constraint in `PROJECT.md` survives because GEE is *not* used for fetching; the data source is the same Sentinel-2 source whether fetched via GEE or S3.
- **D-12: ESA WorldCover labels (already locked primary).** No Dynamic World secondary labels in v1. Removing the 10→9-class remap and the secondary-source machinery keeps the satellite path symmetric with the historical path (both label from WorldCover + DEM).
- **D-13: Cloud handling — STAC query with `eo:cloud_cover < 10`, single-date fetch.** For each target region, pick the lowest-cloud scene from the chosen season. Region/season combos with no qualifying scene are dropped; log the drop count for the same per-reason transparency as Allmaps.
- **D-14: Class-diversity stratified coverage selection.** Use a coarse 1 km WorldCover summary to find regions rich in cropland, built-up, and flooded/wetland — the classes synthetic doesn't produce and that PROJECT.md already up-weights. Sample 4096-px windows from those regions. Do not sample uniformly globally (would oversample forest and water).

### Held-out synthetic test set (EVAL-01)

- **D-15: Hold out whole Azgaar source maps end-to-end.** All renderer styles, all pyramids, all tiles from a held-out map go to test. Zero spatial leakage by construction. The nested-pyramid overlap discussion is moot because adjacent pyramids share a source map and a split.
- **D-16: ~15% by seeded random, stratified across Azgaar continent templates.** Each template contributes proportionally to the test set. Splitter uses a fixed seed for reproducibility.
- **D-17: Sibling directories: `data/synthetic/train/` and `data/synthetic/test/`.** Filesystem-level separation. Training DataLoader points at `train/`; it literally cannot see `test/`. Per-map subdirectory structure is identical inside each.
- **D-18: Snapshot at first build, frozen by ID list.** First `build_dataset.py` run computes the seeded stratified split and writes `data/synthetic/split.json` listing test-set map IDs. Subsequent builds read the manifest; map IDs in the list go to `test/`, everything else (including newly generated maps) goes to `train/`. **The Phase 4 test set never changes after first recording.**

### Claude's Discretion

- IIIF image fetcher implementation (HTTP client choice, retry policy, on-disk caching of intermediate JPEGs) — researcher/planner to decide.
- Storage layout for the multi-scale pyramid (per-pyramid subdirectory vs flat naming convention vs sqlite index) — Claude's call during planning unless the researcher surfaces a specific tradeoff.
- Sentinel-2 STAC client library (`pystac-client` vs raw `requests`) — Claude's call.
- Per-pyramid `sample_weights.json` scoping (per-source map vs per-pyramid vs per-tile) — Claude's call during planning, but the per-source values from `scripts/historical/label.py:40` are locked and must propagate to every tile from that source.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project planning artifacts
- `.planning/PROJECT.md` — Locked SPEC decisions (10 `DECISION-*` entries), constraint prose, key decisions table. Source of truth for taxonomies, data sources, georeferencing precedence, evaluation metric, deliverable.
- `.planning/REQUIREMENTS.md` — Outcome-level acceptance criteria including PHASE-02 (this phase) and EVAL-01 (held-out test set). Phase 2 SC #5 requires the held-out subset.
- `.planning/ROADMAP.md` §"Phase 2" — Goal + Success Criteria. **One edit required after this CONTEXT.md commits:** SC #2 must drop the PaliGemma semi-auto georeferencing requirement (see Deferred Ideas).
- `.planning/INGEST-CONFLICTS.md` — Confirms 0 blockers / 0 warnings; the SPEC and DOC ingest set is consistent.
- `README.md` — Original SPEC. Section "2. Build a dataset of pixel-label pairs" describes the three source families verbatim.

### Existing code (historical pipeline)
- `scripts/build_historical_dataset.py` — `search` / `build` / `full` subcommands. Threadpool-based label generation. The `search` command must be re-wired so `download_georeferenced` calls Allmaps instead of WMS.
- `scripts/historical/rumsey.py` — LUNA search with year-by-year pooling and `_richness_score` ranking. **Currently mid-refactor:** imports `allmaps` but `download_georeferenced` still calls the WMS path. D-01–D-05 govern the cleanup.
- `scripts/historical/allmaps.py` — Allmaps lookup module (currently untracked in git). Returns `{gcps, bbox, image_id, image_size, transformation, mask_svg, allmaps_id, annotation_id}`.
- `scripts/historical/label.py` — `make_labels(geotiff, output_dir)` produces image.png + land_cover.png + topography.png + sample_weights.json. **Historical loss weights are locked at line 40** — `HISTORICAL_LC_WEIGHTS` with trees 0.3, cropland 0.15, built_up 0.1, etc. Do not modify.
- `scripts/historical/worldcover.py` — `fetch_worldcover(map_ds, wgs84_bbox)`. Already uses `s3://esa-worldcover` anonymously via GDAL VSI-CURL. Reusable verbatim for the satellite pipeline (the function takes any rasterio dataset as the target grid).
- `scripts/historical/dem.py` — Copernicus DEM GLO-30 fetcher + slope→topography classifier. Same reusability as worldcover.py.

### Existing code (synthetic pipeline scaffolding)
- `scripts/build_dataset.py` — Orchestrator for the Azgaar GeoJSON → Pillow render → labels pipeline. Iterates `*.geojson` in a raw dir; calls `render_map` then `make_labels`. Currently has no train/test split logic — D-15 to D-18 to be wired in.
- `scripts/render.py` — Pillow renderer for Azgaar maps; multiple visual styles.
- `scripts/label.py` (root level) — Synthetic-side label generator (different from `historical/label.py`).
- `scripts/biome_mapping.py` — Azgaar biome → canonical 9-class mapping.
- `scripts/toon_mapping.py` — Style/toon mapping for the renderer.
- `scripts/augment.py` — Augmentation utilities.

### Existing code (Phase 1, reference only)
- `scripts/finetune_paligemma.py` — Phase 1 fine-tune entrypoint. Not consumed by Phase 2, but the LoRA-adapted SigLIP checkpoint it produces is the architectural target that justifies the multi-scale pyramid tile geometry (D-07).

### External docs (read during discussion)
- Allmaps W3C Web-Annotation contract — documented in `scripts/historical/allmaps.py` module docstring; covers GCP shape (`properties.resourceCoords`, `geometry.coordinates`), endpoint behaviour (200/404/500/429), original-image-pixel-space scaling rule.
- IIIF Image API size syntax — used by D-02. Standard `/full/!w,h/0/default.jpg` for size-best-fit.
- Sentinel-2 L2A STAC catalogue on AWS — `s3://sentinel-cogs`, `eo:cloud_cover` property used by D-13.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- **`fetch_worldcover` / `fetch_topo`**: written generically against an arbitrary rasterio target grid. Reusable verbatim for the satellite pipeline — write a Sentinel-2 RGB tile to a GeoTIFF, then call `fetch_worldcover(ds, bbox)` and `fetch_topo(ds, bbox, water_mask)` to produce co-registered labels. No new code needed for the labels half of the satellite pipeline.
- **`HISTORICAL_LC_WEIGHTS` schema in `scripts/historical/label.py:40`**: the per-class loss-weight dict pattern is reusable for synthetic and satellite. Suggested values to be confirmed during planning, but the schema is set: dict by canonical class name → float, with a sibling `*_TOPO_WEIGHT` float.
- **`rumsey._richness_score`**: useful for any future LUNA-based source selection; preserve as-is.
- **`build_historical_dataset.py` thread-pool pattern**: the `cmd_build` worker loop is reusable for both the satellite pipeline (parallel STAC fetches) and the synthetic pipeline (parallel render+label).

### Established Patterns

- **Per-map directory output schema** (`image.png` + `land_cover.png` + `topography.png` + `sample_weights.json`): mandatory across all three source families. New pipelines must conform.
- **Anonymous S3 via GDAL VSI-CURL** (`os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")`): already established in `worldcover.py`. Sentinel-2 access reuses this pattern — no boto3 needed for raster reads.
- **Exponential backoff + `_MAX_RETRIES`** in `allmaps.py` and `rumsey.py`: same pattern for the new Sentinel-2 STAC client.

### Integration Points

- **`build_historical_dataset.py` `cmd_search` orchestration**: the Allmaps refactor (D-01 through D-05) is the immediate diff. Once `rumsey.download_georeferenced` switches to the Allmaps path, the manifest emission needs the `status`-field changes per D-04, and the summary print needs the per-reason counts per D-05.
- **New `build_satellite_dataset.py`**: parallel structure to `build_historical_dataset.py`, with sub-commands `coverage-scan` (find class-diverse regions), `search` (resolve to Sentinel-2 scenes via STAC), `build` (fetch RGB + run `fetch_worldcover` + `fetch_topo`).
- **New tiler module** (`scripts/tiling.py` or similar): takes a per-map directory, slices the four PNGs into the nested-pyramid tile structure, writes the tile tree. Used by all three source pipelines after their respective per-map output completes.

</code_context>

<specifics>
## Specific Ideas

- **PaliGemma-3B SigLIP resolution.** The nested-pyramid 224 / 448 / 896 scale choice is anchored to SigLIP's native + interpolated input geometry (224 native; 448 = 2× via patch-grid interpolation; 896 = 4×). Position embeddings will be interpolated for the larger scales.
- **Dropped maps must be loud, not silent.** D-05's per-reason drop counter applies to satellite too (D-13). The user is explicit: a clean pipeline that quietly throws away most of the data is not "job done".
- **`scripts/historical/allmaps.py` is untracked in git.** Should be committed as part of the Phase 2 plan-01 work, alongside the `rumsey.py` refactor.

</specifics>

<deferred>
## Deferred Ideas

### v2 (post-HuggingFace upload)

- **PaliGemma-driven semi-automatic georeferencing.** The full pipeline described in PROJECT.md `DECISION-georeferencing-pipeline` step 3 — generate dense terrain predictions on unregistered maps with the Phase-1 fine-tuned PaliGemma, cross-correlate against a WorldCover + Copernicus DEM reference grid for rigid alignment, then refine with thin-plate-spline warping anchored on coastlines / mountain ranges / major water bodies. **Note:** the locked precedence in `DECISION-georeferencing-pipeline` survives — we just don't *implement* steps 3 and 4 in v1. Add `GEOREF-V2` to `.planning/REQUIREMENTS.md`.
- **Manual MapWarper / QGIS GCP placement.** Same v2 bucket as the PaliGemma path. Provides a per-map manual fallback for the unregistered manifest.

### Future iterations within Phase 2 (not in plan-01)

- **Per-source loss weights for the synthetic stream.** Currently only the historical source has weights defined (`scripts/historical/label.py:40`). Synthetic and satellite need their own — the planner should propose values and surface them for user approval. Synthetic might be uniform-1.0 (its labels are ground-truth by construction); satellite should down-weight forest/water (over-represented) and up-weight cropland/built-up/wetland (matches PROJECT.md schema constraint).
- **Storage versioning.** Dataset reproducibility — should the build write a `MANIFEST.json` with checksums + source code commit hash? Worth a follow-up plan once the pipelines stabilise.

### Documentation edits triggered by this discussion

- `.planning/ROADMAP.md` Phase 2 Success Criterion #2 — drop the PaliGemma semi-auto requirement; replace with "unregistered maps are emitted to `unregistered_manifest.json` for v2 processing (manual fallback + PaliGemma semi-auto deferred)".
- `.planning/REQUIREMENTS.md` — add `GEOREF-V2` under v2 Requirements, covering both PaliGemma semi-auto and manual fallback.
- `.planning/STATE.md` Deferred Items — add the two v2 georeferencing items.

These edits should be folded into the Phase 2 plan-01 commit, since they're load-bearing for the plan's scope.

</deferred>

---

*Phase: 2-build-a-dataset-of-pixel-label-pairs*
*Context gathered: 2026-05-15*
