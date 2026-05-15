# Phase 2: Build a dataset of pixel-label pairs - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-15
**Phase:** 2-build-a-dataset-of-pixel-label-pairs
**Areas discussed:** Allmaps wiring, Sample yield, Satellite RGB, Held-out test set

---

## Allmaps wiring strategy

### What happens to the old WMS code path in `rumsey.py`?

| Option | Description | Selected |
|--------|-------------|----------|
| Delete it entirely | LUNA never exposes WMS URLs or bboxes — the path is dead. Removing it simplifies the flow to pure LUNA → Allmaps → IIIF → GeoTIFF. | ✓ |
| Keep as opportunistic fallback | Keep `_wms_url` + `_download_wms_geotiff` for stray georeferencer links in `links[]`. | |
| Keep but gate behind a flag | Delete from default flow; leave behind `--use-wms` CLI flag for debugging. | |

**User's choice:** Delete it entirely.

### What IIIF image size should we request?

| Option | Description | Selected |
|--------|-------------|----------|
| Fixed 4096 max-edge | Match old WMS size. `/full/!4096,4096/0/default.jpg`. | ✓ |
| Fixed 2048 max-edge | Faster, ~1/4 storage, may lose fine illustrative detail. | |
| Adaptive by GSD | Target a uniform metres-per-pixel from bbox diagonal. | |
| Original IIIF size | `/full/full/0/default.jpg` — highest fidelity, 100+ MB per map. | |

**User's choice:** Fixed 4096 max-edge.

### How should we convert Allmaps GCPs into a GeoTIFF transform?

| Option | Description | Selected |
|--------|-------------|----------|
| Affine least-squares fit | 6-parameter affine via least squares over all GCPs. Simple, rasterio-friendly, "good enough" at regional scale. | ✓ |
| Trust Allmaps `transformation` field | Honour Allmaps's recommended transform type per map. Adds branching; TPS needs pre-warping. | |
| TPS warp then write with affine | Pre-warp the IIIF JPEG via thin-plate-spline GCPs, write plain rectilinear GeoTIFF. Highest fidelity for distorted historical maps. | |

**User's choice:** Affine least-squares fit.

### How should Allmaps failures (404 / <3 GCPs / out-of-scale bbox) be handled?

| Option | Description | Selected |
|--------|-------------|----------|
| Single unified manifest, reason field | All failures emitted with `status` distinguishing reason. | |
| Drop out-of-scale, manifest the rest | Out-of-scale silently skipped; `not_in_allmaps` + `gcps_insufficient` enter manifest. | ✓ |
| Two separate manifests | Categorical (no registration data) vs quality (bad data) failures split. | |
| Drop everything; no manifest | Allmaps-only pipeline; defer all manifest design. | |

**User's choice:** Drop out-of-scale, manifest the rest.

**Notes:** "I'll go with option 2, but make a note that the number of maps dropped for this reason needs to be tracked and attended to if large. We can't say 'job done' if we're dropping 3/4 of the maps because it makes the pipeline cleaner." → captured as D-05 (per-reason drop counter at end of `cmd_search`).

### Is the PaliGemma-driven semi-automatic georeferencing in scope?

(Re-asked after a clarification: user did not know the term "registration"; reformulated using "georeferencing" consistent with the SPEC.)

| Option | Description | Selected |
|--------|-------------|----------|
| In scope, later plan within Phase 2 | Phase 2 includes Allmaps + synthetic + satellite + PaliGemma semi-auto + dataset finalisation. | |
| Defer to Phase 2.1 (decimal insertion) | Allmaps + synthetic + satellite = Phase 2; PaliGemma semi-auto = Phase 2.1 urgent insertion. | |
| Defer to v2 | Drop the PaliGemma semi-auto requirement from Phase 2 SC entirely; train v1 on Allmaps + synthetic + satellite only. Add the path post-v1 if dataset size proves insufficient. | ✓ |

**User's choice:** Defer to v2.
**Notes:** Triggers downstream doc edits — ROADMAP.md Phase 2 SC #2, REQUIREMENTS.md (new GEOREF-V2 entry), STATE.md Deferred Items.

---

## Sample yield per map (tile/crop strategy)

### Where should sample crops be materialised?

| Option | Description | Selected |
|--------|-------------|----------|
| Whole-map per GeoTIFF; crop at train time | Smallest disk footprint; most flexible; slightly slower per-step I/O. | |
| Pre-tiled at build time, non-overlapping | Disk scales by tile count; tiny DataLoader; fixed-grid. | |
| Pre-tiled at build time, with overlap | Doubles–quadruples sample count; better edge coverage; mild data-leak risk if not split properly. | ✓ |
| Whole-map + crops.jsonl index | Whole maps on disk + index of valid crop coordinates. | |

**User's choice:** Pre-tiled at build time, with overlap.

### What tile size?

| Option | Description | Selected |
|--------|-------------|----------|
| 224×224 (SigLIP native) | Smallest, finest granularity; multi-resolution must come from aggregation. | |
| 384×384 (SigLIP large) | ~4× area vs 224; SigLIP handles via interpolated position embeddings. | |
| 512×512 (clean power-of-2) | Geographically reasonable; requires resize or sliding-window for SigLIP. | |
| Multi-scale (224 + 448 + 896, aligned) | Directly supports Phase 3 coarse-to-fine; ~3× disk; more complex DataLoader. | ✓ |

**User's choice:** Multi-scale (224 + 448 + 896, all aligned).

### How should the multi-scale tiles align across resolutions?

| Option | Description | Selected |
|--------|-------------|----------|
| Nested pyramid (strict 2×2 nesting per scale, no sibling overlap within parent) | A 896 prediction maps directly to 4 inner 448s and 16 inner 224s. | ✓ |
| Independent sampling per scale | Each scale tiled with its own stride; no parent–child relation. | |
| Nested pyramid with overlap at finest scale only | Strict pyramid + 50% overlap among 224 tiles within parent. | |
| Multi-scale crops sharing a centre point | Per crop position, co-centred trio (224 + 448 + 896). | |

**User's choice:** Nested pyramid.

### How does "with overlap" compose with strict nested pyramids?

| Option | Description | Selected |
|--------|-------------|----------|
| Overlap between pyramids at the top scale only | Each pyramid strict; adjacent pyramids stride < 896. | ✓ |
| Non-overlapping pyramids (drop the overlap decision) | Pyramids tile at stride 896; smallest dataset. | |
| Two overlap levels (between pyramids + among 224 children of different 448 parents) | Maximum samples, highest correlation risk. | |
| Half-pyramid stride only at boundaries | Interior non-overlapping; half-stepped pyramid near edges. | |

**User's choice:** Overlap between pyramids at the top scale only. Stride 448 (50% pyramid overlap). ~64 pyramids × 21 tiles = ~1344 sample tiles per 4096-px map.

---

## Satellite-source RGB imagery

### What is the satellite stream actually contributing?

| Option | Description | Selected |
|--------|-------------|----------|
| Image→label pairs (true third source) | Satellite RGB + labels fed as samples like historical and synthetic. Cross-domain generalisation bet. | ✓ |
| Label-only adjacency priors | No satellite RGB; WorldCover + DEM provide structured prior on class co-occurrence. | |
| Render satellite labels back into illustrated style | WorldCover + DEM → toon renderer → illustrated maps with real-world geography. | |
| Image→label pairs with style transfer | Sentinel-2 RGB stylised to illustrated via a frozen style network. | |

**User's choice:** Image→label pairs (true third training source).

### Which satellite RGB source?

| Option | Description | Selected |
|--------|-------------|----------|
| Sentinel-2 L2A via AWS | 10 m, anonymous S3 (`s3://sentinel-cogs`), STAC catalogue. | |
| Landsat 9 Collection 2 via AWS | 30 m, larger swaths, less downsampling needed. | |
| Pre-rendered annual composite | Cloud-free static mosaic — sample like any COG. | |
| HLS (Harmonized Landsat–Sentinel-2) | 30 m fused L8/L9 + S2; sensor-agnostic. | |

**User's choice:** Asked a clarifying question about Google Earth + "Google's pixel-level classification" — assumed GEE was the satellite source. Re-asked with clarification:

| Option | Description | Selected |
|--------|-------------|----------|
| S2 L2A RGB + ESA WorldCover labels | Both already locked primary sources; co-registered on Sentinel-2 10 m UTM grid. | ✓ |
| S2 L2A RGB + WorldCover (primary) + Dynamic World (secondary) | Two label sources; multi-task training or temporal-diversity sampling. | |
| S2 L2A RGB + Dynamic World labels | DW near-real-time; needs 10→9 class remap. | |
| Something I'm missing | Reopen the "satellite role" question. | |

**User's choice:** Sentinel-2 L2A RGB + ESA WorldCover labels.

**Notes:** Clarification covered: GEE is a compute platform, not an imagery source; Sentinel-2 from S3 vs GEE yields the same pixels; Google Earth basemap tiles are commercially licensed and not usable for ML; Dynamic World is the "Google model" the user was thinking of and is allowed as secondary, but not needed for v1.

### How should Sentinel-2 cloud cover be handled?

| Option | Description | Selected |
|--------|-------------|----------|
| Filter scenes by cloud cover, single-date fetch | STAC `eo:cloud_cover < 10`; lowest-cloud scene per region/season. | ✓ |
| Pre-computed annual median composite | Build-once global mosaic; uniform quality. | |
| Multi-date stack, season-conditioned | 4 scenes per region; random-pick per training step. | |
| Cloud-mask in COG, drop high-cloud tiles at sample time | Raw scenes + per-pixel SCL mask. | |

**User's choice:** Filter scenes by cloud cover, single-date fetch.

### How to choose which regions to sample?

| Option | Description | Selected |
|--------|-------------|----------|
| Class-diversity stratified sampling | Coarse 1 km WorldCover summary → regions rich in cropland / built-up / wetland. | ✓ |
| Match historical-map geographic distribution | Mostly Europe + Med + N. Africa; under-represents tropics + high latitudes. | |
| Uniform global random sampling | Real-world class distribution dominates; lots of forest and water. | |
| Biome-stratified by WWF 14-biome | Each biome proportional; doesn't directly target 9-class diversity. | |

**User's choice:** Class-diversity stratified sampling.

---

## Held-out synthetic test set (EVAL-01)

### What's the held-out test set strategy?

| Option | Description | Selected |
|--------|-------------|----------|
| Hold out whole Azgaar maps | All renderings, all pyramids, all tiles from N maps reserved. Zero spatial leakage. | ✓ |
| Hold out by seed range | Reserve seeds [N..M] for test; requires fully-seeded generator. | |
| Hold out a renderer style | Train on 3 styles, test on 1 — risky as headline metric. | |
| Hold out a continent template | Train on subset of templates, test on rest — risky as headline metric. | |

**User's choice:** Hold out whole Azgaar maps.

### How many, and how selected?

| Option | Description | Selected |
|--------|-------------|----------|
| ~15% by seeded random, stratified across continent templates | Reproducible; balanced per template; ~160k test tiles. | ✓ |
| Fixed N=20 maps | Hard-coded count; less flexible at scale. | |
| ~10% seeded random, no stratification | Simplest; risk of biome / template bias. | |
| ~25% seeded random, stratified | More statistical power on per-class NLL; costs more training data. | |

**User's choice:** ~15% by seeded random, stratified.

### How is the split surfaced on disk?

| Option | Description | Selected |
|--------|-------------|----------|
| Sibling directories `data/synthetic/{train,test}` | Filesystem-level separation; impossible to leak. | ✓ |
| Single tree + `split.json` manifest | Loader filters by manifest; flexible but easier to bug. | |
| Per-map metadata field in `meta.json` | Scan-time filter; easy to accidentally include test in a glob loader. | |

**User's choice:** Sibling directories `data/synthetic/{train,test}`.

### How does the split behave when the dataset grows later?

| Option | Description | Selected |
|--------|-------------|----------|
| Snapshot at first build, frozen by ID list | First build writes `data/synthetic/split.json` with test IDs; subsequent builds read it. New maps default to train. Test set never changes. | ✓ |
| Deterministic recomputation each build | Re-run splitter every build; test composition drifts with dataset growth. | |
| Monotonic test growth | Test only grows; union of old + new test IDs. | |

**User's choice:** Snapshot at first build, frozen by ID list.

---

## Claude's Discretion

- IIIF image-fetcher implementation details (HTTP client, retry policy, intermediate caching).
- Storage layout for the nested pyramid (per-pyramid subdir vs flat naming vs sqlite index).
- Sentinel-2 STAC client library (`pystac-client` vs raw `requests`).
- `sample_weights.json` per-source vs per-pyramid vs per-tile scoping.
- Synthetic and satellite per-source loss weight values (planner proposes, user approves).

## Deferred Ideas

### v2 (post-HuggingFace upload)

- **PaliGemma-driven semi-automatic georeferencing** (PROJECT.md `DECISION-georeferencing-pipeline` step 3).
- **Manual MapWarper / QGIS GCP placement** (PROJECT.md `DECISION-georeferencing-pipeline` step 4).

### Within Phase 2 but not in plan-01

- Per-source loss weights for synthetic and satellite streams.
- Dataset storage versioning + reproducibility manifest with source commit hash.

### Documentation edits triggered by this discussion

- ROADMAP.md Phase 2 SC #2 — drop PaliGemma semi-auto requirement.
- REQUIREMENTS.md — add GEOREF-V2 under v2 Requirements.
- STATE.md — add the two v2 georeferencing items to Deferred Items table.

### Terminology feedback

- User flagged "registration" as unfamiliar jargon mid-discussion; switched to "georeferencing" (project-native term) throughout. Saved as a feedback memory for future sessions.
