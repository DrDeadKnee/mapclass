# Roadmap: MapClass / GeoViLM

## Overview

Vertical-MVP roadmap: each phase delivers an end-to-end runnable training+inference slice of the GeoViLM stack at progressively greater depth, rather than building all data pipelines first and risking month-6 discovery that the model architecture doesn't work. Phase 1 wires the whole pipeline end-to-end on synthetic data with a Mock backbone (random outputs, but correct shapes and contracts). Phases 2-3 grow the model and data: real small VL backbone, then OSM + OCR + full v0 training on all three sources. Phase 4 closes the bootstrap loop (auto-georef + v1 retrain) with the mandatory v0-vs-v1 kill-switch. Phase 5 trains the PaliGemma bellwether and runs the EVAL-03 NLL comparison. Phase 6 packages the small-backbone ship: inference module, qualitative renders on the four fantasy maps, model card. v1 ends at model handoff (checkpoint + inference module + benchmark numbers).

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: End-to-End Skeleton (Synthetic + Mock Backbone)** - Wire the whole pipeline end-to-end on synthetic data with a Mock backbone so the contracts, configs, and entry points exist before any real model lands.
- [ ] **Phase 2: Small VL Backbone Slice (Synthetic + Historical)** - Swap Mock → SmolVLM-500M and run a thin per-source-weighted v0 training on synthetic + already-registered-historical so the first real probabilities exist.
- [ ] **Phase 3: OSM + OCR + Full v0 Training (small backbone)** - Add the OSM road-tile source and the docTR OCR module, then train v0 to completion on all three v0 sources.
- [ ] **Phase 4: Auto-Georef Bootstrap + v1 Retrain (small backbone)** - Land the auto-georef tool, run the bootstrap to register Rumsey maps, retrain v1 on the expanded dataset, and gate ship with the v0-vs-v1 NLL kill-switch.
- [ ] **Phase 5: PaliGemma Bellwether + EVAL-03** - Train MODEL-04 v0 and v1 on byte-identical configs differing only in the backbone field, then compute the small-backbone-vs-PaliGemma NLL gap.
- [ ] **Phase 6: Ship the Small-Backbone Variant (SHIP-01 + EVAL-02)** - Finalize the inference module contract, render the four qualitative spot-checks, embed model+dataset versioning, and package the v1 small-backbone checkpoint for the hex-grid app.

## Phase Details

### Phase 1: End-to-End Skeleton (Synthetic + Mock Backbone)
**Goal**: A runnable end-to-end training+inference pipeline on synthetic data with a Mock backbone — outputs are random but the shape, config, checkpoint, and inference contracts are all real and locked.
**Mode:** mvp
**Depends on**: Nothing (first phase; brownfield: synthetic + historical dataset pipelines already exist on `refactor_paper`)
**Requirements**: MODEL-03
**Success Criteria** (what must be TRUE):
  1. `python -m mapclass.train --config mapclass/configs/v0.yaml` runs to completion on a small synthetic-only sample and writes a safetensors checkpoint with embedded `{model_version, dataset_manifest_sha, taxonomy_hash, training_seed, source_class_weights_hash}` metadata.
  2. `python -m mapclass.infer <checkpoint> <image>` loads the checkpoint and returns a dict with `land_cover` shape `[9, H, W]` and `topography` shape `[3, H, W]` whose class axes sum to 1.0 within fp16 epsilon (probabilities, not logits).
  3. `python -m mapclass.eval --split heldout --checkpoint <ckpt>` computes per-pixel NLL on a deterministic-by-sample-id-hash held-out split and writes `eval_report.json` with per-source per-class numbers (numbers will be near `log(num_classes)` since the backbone is Mock — the *pipeline* is the deliverable, not the score).
  4. `mapclass.data.contract.assert_sample_valid(path)` rejects a tampered sample (e.g. wrong class indices, missing manifest.json) with a loud schema error at training startup.
  5. `pip install -e .` from a fresh clone of `refactor_paper` succeeds against the committed `pyproject.toml` + `requirements.lock.txt` + `.python-version` and `python -c "import mapclass"` works.
  6. A committed `mapclass/configs/EVAL-03_protocol.md` (one page) names the held-out split hash, the loss-weights file hash, the augmentation policy, the fine-tune budget shape, the evaluation seed, and the kill-switch criterion that Phase 4 and Phase 5 must respect — pre-registered before any real training run starts.
**Plans**: TBD

**Pitfalls addressed:** PITFALL 3 (loss-weight schema validator at startup), PITFALL 4 (Backbone ABC owns `preprocess()` from day one), PITFALL 5 (deterministic hash-based splits committed once), PITFALL 15 (seeding utility), PITFALL 19 (lockfile non-deferrable). EVAL-03 protocol pre-registration here is the prevention for PITFALL 2 — must land in Phase 1 before MODEL-01 / MODEL-04 training begins. Brownfield concerns addressed: CONCERNS.md 8a (no seeds), 8b (no dataset versioning), 8c (no lockfile), TESTING.md gap-item 6 (loss-weight schema fragility).

### Phase 2: Small VL Backbone Slice (Synthetic + Historical)
**Goal**: Replace the Mock backbone with `HuggingFaceTB/SmolVLM-500M-Instruct` and produce the first real per-pixel probability maps from a thin v0-style training run on synthetic + already-registered-historical, with per-source class-conditional loss weights wired through.
**Mode:** mvp
**Depends on**: Phase 1
**Requirements**: MODEL-01, DATA-06
**Success Criteria** (what must be TRUE):
  1. `python -m mapclass.train --config mapclass/configs/v0_thin.yaml` trains a `SmolVLMBackbone` + seg-heads model end-to-end on synthetic + already-registered-historical and produces a safetensors checkpoint that loads at FP16 in under 4 GB GPU VRAM and runs CPU inference on a 1024×1024 image in seconds.
  2. `mapclass.data.loss_weights.LossWeights.load("mapclass/configs/loss_weights.yaml")` parses the per-source × per-class table (incl. `synthetic_azgaar`, `rumsey_registered` rows) and `lookup(source_subtype, class_name)` returns the configured weight; the trainer's per-pixel weight tensor reflects these values for batches mixing the two sources (verifiable from a logged batch sample).
  3. The eval script reports v0-thin held-out NLL per source per class, and the per-source numbers separate (i.e. they are not identical) — confirms the loss-weight signal actually flowed through training.
  4. TensorBoard `runs/` directory contains loss curves and sample-prediction images for each source, so source-shift is visually inspectable.
  5. The checkpoint metadata header includes the SmolVLM processor identity hash and the taxonomy hash; `mapclass.infer.load_model` refuses to load on a hash mismatch with a clear error.
**Plans**: TBD

**Pitfalls addressed:** PITFALL 4 (preprocessor identity in metadata, hash-checked at load), PITFALL 7 (LoRA-vs-full-FT-vs-frozen decision must be made and documented in this phase's plan; default per STACK.md is full FT for SmolVLM-500M with frozen baseline as comparison), PITFALL 10 (focal CE + per-source weights × pixel-frequency weights — must be wired in here, not deferred), PITFALL 11 (label smoothing 0.05 default during training to start mitigating segmentation miscalibration). Brownfield concern 2a (`augment.py` dead code) — wire D-2 artifact augmentation into the synthetic training path as part of this phase.

### Phase 3: OSM + OCR + Full v0 Training (small backbone)
**Goal**: Land the OSM road-tile source and the docTR-pretrained OCR module, then train v0 to completion on all three v0 sources (synthetic + already-registered-historical + OSM) for the small backbone — this is the v0 ship candidate before the bootstrap loop runs.
**Mode:** mvp
**Depends on**: Phase 2
**Requirements**: DATA-05, MODEL-02, TRAIN-01
**Success Criteria** (what must be TRUE):
  1. `python scripts/build_road_dataset.py {search,build,full}` mirrors the historical orchestrator and emits OSM-tile-derived samples conforming to the sample contract (`image.png + land_cover.png + topography.png + sample_weights.json + manifest.json` with `source_subtype: road_osm`) for at least one bbox region; `assert_sample_valid()` accepts every emitted sample.
  2. `mapclass.model.ocr_head` (docTR-pretrained starting point) runs as a separate sub-pipeline that processes a held-out map image and returns detection bboxes + recognition strings; an isolation diagnostic (per-batch detection-count + accept rate) is logged before joint training is enabled.
  3. `python -m mapclass.train --config mapclass/configs/v0.yaml` trains end-to-end on all three sources with the `road_osm` row of `loss_weights.yaml` active, OCR auxiliary loss with `λ=0.1` (and λ-off mode preserved as a fallback), and produces `models/geovilm_v0_small.pt` whose held-out NLL improves over the Phase-2 v0-thin baseline on at least the synthetic and historical splits.
  4. The final v0 eval report (per-source per-class NLL JSON + reliability diagram + ECE) is committed to `eval_reports/v0_small.json` and per-source numbers are non-degenerate (no source dominates the loss with weight 0 by accident).
  5. Single-CPU inference on a 1024×1024 fantasy map completes within 10 s on a modern laptop, and FP16 GPU inference fits in a 4 GB envelope with multiple GB to spare — the inference budget is verified, not assumed.
**Plans**: TBD

**Pitfalls addressed:** PITFALL 8 (OCR trained as separate sub-pipeline first, joint-trained only after isolation diagnostic; per-batch detection-count check as gating; λ=0.1 default), PITFALL 9 (fixed-vocabulary or near-zero-LM-weight in beam search to avoid hallucinating fantasy names; document Latin-only English-trained limitation in inference README), PITFALL 16 (OSM ODbL attribution captured in OSM-source manifest.json and propagated into checkpoint metadata), PITFALL 17 (OSM CRS round-trip verified against WorldCover/DEM bbox at sub-pipeline construction).

### Phase 4: Auto-Georef Bootstrap + v1 Retrain (small backbone)
**Goal**: Land the auto-georef tool (sanity-check first, then bootstrap mode), run the v0-driven bootstrap to register the unregistered Rumsey maps, retrain v1 on the expanded dataset, and gate ship with the v0-vs-v1 NLL kill-switch — if v1 doesn't beat v0 by ≥2% the ship target is v0, not v1.
**Mode:** mvp
**Depends on**: Phase 3
**Requirements**: GEOREF-01, GEOREF-02, TRAIN-02, EVAL-01
**Success Criteria** (what must be TRUE):
  1. `python scripts/georef.py sanity-check <known-coords-tif-dir>` runs WITHOUT a model, hides the input CRS, recovers GCPs via cross-correlation against WorldCover + DEM, applies regularised TPS warp, and reports per-map residuals; the residual on at least 80% of already-registered Rumsey maps is below a configured threshold (rejected residuals are logged, not crashed on).
  2. `python scripts/georef.py bootstrap --model models/geovilm_v0_small.pt --manifest data/historical/raw/unregistered_manifest.json` runs v0 inference + cross-correlation + TPS, writes new GeoTIFFs into `data/historical/raw/georeferenced/`, updates each sample's `manifest.json` with `registered_via: auto_georef_v0`, and rejects configs where GCP-convex-hull < 30% of image area or peak-significance ratio < 1.5 (rejections recorded as manifest entries, not silent drops).
  3. `python scripts/build_historical_dataset.py build` re-runs over the bootstrap-expanded TIFs and produces a larger dataset whose new samples carry `source_subtype: rumsey_bootstrapped` (and the matching loss-weight row with `confidence: 0.5` multiplier active).
  4. `python -m mapclass.train --config mapclass/configs/v1.yaml` produces `models/geovilm_v1_small.pt`; `python -m mapclass.eval --split heldout` runs against BOTH v0 and v1 on the SAME committed `splits.json`, and the v0-vs-v1 NLL delta is reported in `eval_reports/v1_small.json`.
  5. A documented kill-switch decision is recorded: if v1 NLL beats v0 by ≥2%, v1 is the small-backbone ship target; otherwise v0 is the ship target and the bootstrap is documented as net-negative for this run (not retried in v1).
**Plans**: TBD

**Pitfalls addressed:** PITFALL 1 CRITICAL (v0-vs-v1 NLL kill-switch — if v1 doesn't beat v0 by ≥2% margin, ship v0 and document the bootstrap as net-negative; bootstrap depth = 1 by contract, no recursive bootstrap), PITFALL 5 (bootstrap can only ADD to the train split via the committed splits.json bucket policy; eval-time overlap assertion runs and fails loudly on contamination), PITFALL 13 (regularised TPS with smoothing, GCP cap 50–200, hull-area threshold ≥30% of image area, sanity-check mode landed before bootstrap), PITFALL 14 (peak-significance ratio threshold ≥1.5, GCP coverage threshold, manifest-as-rejection-record on no-good-match).

### Phase 5: PaliGemma Bellwether + EVAL-03
**Goal**: Train MODEL-04 (`google/paligemma2-3b-pt-224`) v0 and v1 on byte-identical configs that differ only in the `backbone:` field, compute the small-backbone-vs-PaliGemma NLL gap on the same committed held-out splits, and decide whether the small backbone is good enough to ship.
**Mode:** mvp
**Depends on**: Phase 4 (needs the locked splits.json, the `loss_weights.yaml`, the augmentation policy, and the bootstrap-expanded dataset that Phase 4 produced; needs the Phase-1 EVAL-03 protocol document as the entry criterion)
**Requirements**: MODEL-04, EVAL-03
**Success Criteria** (what must be TRUE):
  1. `mapclass.model.backbone.PaliGemma2Backbone` is implemented as a parallel concrete subclass of the same `Backbone` ABC; loading via `PaliGemmaForConditionalGeneration` with NF4 + LoRA on the SigLIP-So400m vision tower succeeds on a 24 GB GPU with gradient checkpointing.
  2. `python -m mapclass.train --config mapclass/configs/v0_paligemma.yaml` and `--config v1_paligemma.yaml` produce `models/benchmark/geovilm_paligemma_v0.pt` and `_v1.pt`; the configs differ from the small-backbone v0/v1 configs only in the `backbone:` field, the `dtype` (NF4 + LoRA), and the input resolution (224); a hash-assertion test confirms `splits.json`, `loss_weights.yaml`, and the augmentation policy are byte-identical to Phase 3/4 runs.
  3. `eval_reports/eval_03_comparison.json` reports per-source per-class NLL for both backbones on the same splits; the small-backbone-vs-PaliGemma gap is documented globally and per-source.
  4. PaliGemma checkpoints land under `models/benchmark/` (separated from the ship checkpoint per Gemma TOU vs Apache-2.0 license split) with the Gemma Terms of Use string embedded in their safetensors metadata; they are explicitly NOT packaged with the ship checkpoint.
  5. A documented decision is recorded: if the EVAL-03 gap is within the protocol's accept threshold, the small backbone (Phase-4 ship target) proceeds to Phase 6; if the gap is too large, MODEL-01 is reconsidered before ship and the recovery options (Moondream2 fallback per STACK.md, retrain with longer budget, drop OCR aux loss, etc.) are noted.
**Plans**: TBD

**Pitfalls addressed:** PITFALL 2 CRITICAL (pre-registered EVAL-03 protocol from Phase 1 enforced via hash-assertion at compare time; only the backbone field differs between configs — no augmentation, splits, or loss-weights drift), PITFALL 4 (`PaliGemma2Backbone` owns its own `preprocess()` and the processor identity is captured in its checkpoint metadata, separate from the SmolVLM processor identity), PITFALL 6 (PaliGemma VRAM blowup — one-batch dry-run on the cloud VM before scheduling either full training run; default to QLoRA + gradient checkpointing + 224 input, not 448 or 896).

### Phase 6: Ship the Small-Backbone Variant (SHIP-01 + EVAL-02)
**Goal**: Finalize the inference module as the stable contract with the hex-grid app, render the four qualitative spot-checks (Tolkien Middle-Earth / Westeros / Abercrombie Circle of the World / Warhammer Old World), embed model + dataset + taxonomy versioning in the checkpoint metadata, and package the v1 (or v0, per Phase 4 kill-switch) small-backbone variant for downstream consumption.
**Mode:** mvp
**Depends on**: Phase 5 (needs EVAL-03 sign-off that the small backbone is good enough; uses Phase-4 ship-target decision)
**Requirements**: EVAL-02, SHIP-01
**Success Criteria** (what must be TRUE):
  1. `from mapclass.infer import load_model; predictor = load_model(<checkpoint>, device='auto')` works on both CPU-only and 4–8 GB GPU hosts; `predictor.predict(image)` returns the documented contract (`land_cover` probabilities `[9, H, W]`, `topography` probabilities `[3, H, W]`, `model_version`, `taxonomy`, `WATER_TOPO=255` sentinel) and `mapclass/infer/README.md` documents the output shape + dtype contract.
  2. `mapclass.infer` does not import anything from `mapclass.data` — verified by a startup import check; the inference module's deps are the model + numpy + PIL only (no rasterio, no torch.utils.data).
  3. Tile-and-stitch helper (D-9) is wired into `predict()` so that fantasy maps larger than the SmolVLM 384-input native size are tiled with overlap-and-feather (cosine-window) blending and stitched without visible seams; spot-check on a 4096×4096 fantasy map produces a probability map with no checkerboard artifacts.
  4. Qualitative renders for the four target fantasy maps (Tolkien Middle-Earth, Westeros, Abercrombie First-Law Circle of the World, Warhammer Old World) are produced under `eval_reports/qualitative/<map>.png` for the ship checkpoint AND for the PaliGemma checkpoint (no metric, no ground truth — visual inspection only).
  5. The shipped checkpoint embeds `{model_version, dataset_manifest_sha, taxonomy_hash, training_seed, source_class_weights_hash, processor_identity, license: "Apache-2.0", attribution: "OSM ODbL, Copernicus DEM, ..."}` in safetensors metadata; the model card under `models/MODEL_CARD.md` documents the inference budget, the four qualitative renders, and the EVAL-01 + EVAL-03 numbers; the hex-grid app contract is pinned (any future change to it is a breaking change marked by a new `model_version`).
**Plans**: TBD
**UI hint**: yes

**Pitfalls addressed:** PITFALL 9 (Latin-only English-trained OCR limitation documented in inference README), PITFALL 11 (TS-6 reliability diagram + ECE in the model card; calibrated probabilities by default in `predict()`; temperature scaling factor stored in checkpoint metadata if Phase 4/5 found ECE > acceptable), PITFALL 12 (tile-and-stitch with cosine-window blending is now a hard ship requirement, not a deferred D-9 differentiator, since SmolVLM's 384 native input is well below typical 4096²+ fantasy map sizes), PITFALL 16 (OSM ODbL attribution + Copernicus DEM attribution string in checkpoint metadata; PaliGemma checkpoint kept under `models/benchmark/` separate from the Apache-2.0-compatible ship checkpoint).

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. End-to-End Skeleton | 0/TBD | Not started | - |
| 2. Small VL Backbone Slice | 0/TBD | Not started | - |
| 3. OSM + OCR + Full v0 Training | 0/TBD | Not started | - |
| 4. Auto-Georef Bootstrap + v1 Retrain | 0/TBD | Not started | - |
| 5. PaliGemma Bellwether + EVAL-03 | 0/TBD | Not started | - |
| 6. Ship the Small-Backbone Variant | 0/TBD | Not started | - |
