# Project Research Summary

**Project:** MapClass / GeoViLM
**Domain:** Multi-source-trained dense-prediction VL+OCR+heads model (model-handoff package, brownfield mid-refactor)
**Researched:** 2026-05-08
**Confidence:** HIGH overall, with one localized MEDIUM area (OCR module on map text)

## Executive Summary

MapClass v1 ships a small vision-language backbone (`HuggingFaceTB/SmolVLM-500M-Instruct`, Apache-2.0, ~1 GB FP16) with two dense seg heads, an auxiliary rotation-invariant OCR head, and a packaged inference module — a model-handoff to a separate hex-grid app, **not** an end-user product. A parallel benchmark variant trains the same stack with `google/paligemma2-3b-pt-224` swapped in as the backbone (MODEL-04) to produce a bellwether NLL number against which the small backbone is judged (EVAL-03). The architectural lever that makes both runs cheap is the `Backbone` ABC (ARCHITECTURE.md Pattern 3) — one swap surface, one trainer, one eval, two concrete subclasses; the data, loss-weights, augmentation, and split files must be byte-identical between the two backbones for EVAL-03 to be a clean comparison.

The dominant risk is **silent regression** — five CRITICAL pitfalls all share the property that nothing crashes and the loss curve looks fine while the eval number is contaminated, the bootstrap regresses, or the backbone comparison is confounded. Mitigation is observability-over-tests: schema validators that fail loudly at startup (PITFALLS 3, 4), a v0-vs-v1 NLL kill-switch (PITFALLS 1; FEATURES D-5), a pre-registered EVAL-03 protocol (PITFALLS 2), hash-stable splits (PITFALLS 5), and seeds + lockfile + preflight (PITFALLS 15, 19). PROJECT.md accepts no tests / no CI at v1; that decision is survivable only if these specific deliberate mitigations land in v1, not "post-ship."

The bootstrap loop (GEOREF-02: v0 → auto-register Rumsey → retrain v1) is the single highest-leverage and highest-risk feature in v1: it can poison v1 (Pitfall 1) or land badly-warped GeoTIFFs (Pitfalls 13, 14). Architectural mitigation is filesystem-mediated three-step orchestration (ARCHITECTURE Pattern 4) with hard kill-switches; if the bootstrap is net-negative on held-out NLL, v0 is the ship target and that path must be a first-class outcome, not an embarrassed fallback.

## Key Findings

### Recommended Stack — concrete picks

Bare PyTorch + HuggingFace transformers + accelerate + safetensors + OmegaConf + TensorBoard, with a single new pyproject.toml making `mapclass/` `pip install -e .`-able for the cloud-VM SSH workflow. Detailed in [STACK.md](STACK.md).

**Core technologies (versions are minima):**

| Technology | Version | Purpose | Why this pick |
|------------|---------|---------|---------------|
| Python | `3.10+` (3.12 verified) | Project language | Already pinned via PEP 604/585 type hints in tree |
| PyTorch | `>=2.4,<3.0` | Tensor / autograd / serialization | Required by transformers 4.47+; cuDNN SDPA speedups for ViTs |
| transformers | `>=4.47,<5.0` | Backbone loader, processors | 4.47 introduced PaliGemma 2 support; <5.0 defers breaking-change migration |
| accelerate | `>=0.34` | Mixed-precision + device + `prepare()` | Right level of abstraction for a research codebase — not Lightning, not Trainer |
| PEFT | `>=0.11` | LoRA on PaliGemma vision tower | Optional for SmolVLM full FT; required for PaliGemma NF4+LoRA training |
| bitsandbytes | `>=0.43` | NF4 (PaliGemma training) + INT8 dynamic quant (post-ship optional) | Required for PaliGemma 3B on a 24 GB GPU |
| safetensors | `>=0.4.5` | Checkpoint format with metadata header | TS-2 / TS-8: embeds versioning hashes; never pickle (`.bin`) |
| OmegaConf | `>=2.3` | YAML + dataclass config system | Lighter than Hydra; sufficient for single-developer multirun-free workflow |
| TensorBoard | `>=2.18` | Training-time logging (loss curves + sample preds) | TS-7; local files, no network, rsyncs cleanly off the cloud VM |
| python-doctr[torch] | `>=1.0.1` | OCR module starting point (MODEL-02) | Active maintenance, end-to-end, rotation-aware, Apache-2.0 |
| omegaconf, scikit-image, opencv-python-headless, torchmetrics, tqdm | recent | Supporting libs | Per STACK.md "Supporting Libraries" |

**Concrete model IDs:**

- **MODEL-01 ship backbone:** `HuggingFaceTB/SmolVLM-500M-Instruct` — Apache-2.0, ~1 GB FP16, vision encoder is the 93M SigLIP-B/16 ViT (clean dense features, no `trust_remote_code`), inference budget comfortable on a 4–8 GB GPU and CPU-usable at ~2–4 s per 1024×1024.
- **MODEL-04 benchmark backbone:** `google/paligemma2-3b-pt-224` — Gemma Terms of Use, ~6 GB FP16; `pt-224` (not `mix-224`) for a clean fine-tune comparison; trained with NF4 + LoRA on a 24 GB GPU.
- **Fallbacks documented in STACK.md:** SmolVLM-256M (CPU floor), Moondream2 (`vikhyatk/moondream2`, costs `trust_remote_code`), Florence-2-base (DaViT encoder).

**License split (must propagate to roadmap and model cards):** SmolVLM-derived ship checkpoint can be Apache-2.0; PaliGemma-derived benchmark checkpoint inherits Gemma TOU and **must NOT be packaged with the ship checkpoint** — keep it in `models/benchmark/` separately (per STACK.md "License flag (loud)" and PITFALL 16).

**Avoid:** PyTorch Lightning, full HF `Trainer`, full Hydra, wandb (single-author), MMOCR / mmcv, pickle `.bin`, TensorFlow / JAX paths in any library that offers a choice (per STACK.md "What NOT to Use").

### Architecture Approach — phase-relevant slice

Filesystem-mediated, additive-to-the-brownfield. The existing `scripts/` entry-point layer is preserved unchanged; a new `mapclass/` Python package is added alongside it for the model code. Detailed in [ARCHITECTURE.md](ARCHITECTURE.md) (System Overview; Component Responsibilities; Patterns 1–4; Build Order).

**Component boundaries (each new line = one v1 component):**

1. **Sample contract** (ARCHITECTURE.md Pattern 1) — every source emits the same on-disk schema (`image*.png + land_cover.png + topography.png + sample_weights.json + manifest.json`). The `manifest.json` schema is **new** and load-bearing. Validator `mapclass.data.contract.assert_sample_valid` fails loudly on drift.
2. **Loss weights as `(source, class)` table** (ARCHITECTURE.md Pattern 2) — `mapclass/configs/loss_weights.yaml`, loaded once at startup. Strict schema validator at startup is the mitigation for PITFALL 3.
3. **Backbone interface as swap surface** (ARCHITECTURE.md Pattern 3) — `Backbone` ABC with `extract_features(image) -> dict[str, Tensor]` and (per PITFALL 4 prevention) **also** `preprocess(image) -> Tensor`. SmolVLMBackbone and PaliGemma2Backbone are parallel concrete subclasses. Heads consume the dict by stage name.
4. **OCR module** (`mapclass/model/ocr_head.py`) — docTR pretrained as starting point; auxiliary loss with low λ (~0.1); train as a separate sub-pipeline first, integrate as auxiliary loss only after isolation (PITFALL 8 prevention).
5. **Two dense seg heads** (`mapclass/model/seg_heads.py`) — land-cover (9 classes) + topography (3 classes + WATER_TOPO=255 sentinel), thin upsampling stacks on the backbone's vision features.
6. **Auto-georef tool** (`scripts/georef/` sub-package) — three independent pieces (cross-correlation, TPS, GCP scoring) plus the bootstrap orchestrator. Sanity-check mode runs with no model and is independently useful.
7. **Bootstrap loop** (ARCHITECTURE.md Pattern 4, filesystem-mediated, NEVER in-process per Anti-Pattern 2) — three-step sequence: `train v0` → `scripts/georef.py bootstrap` → re-run `build_historical_dataset.py build` → `train v1`.
8. **Training entry point** (`python -m mapclass.train --config v0|v1`) — single trainer for both backbones (just different `backbone:` field in the config); hand-written loop with `accelerator.prepare()`.
9. **Eval entry point** (`python -m mapclass.eval --split heldout --checkpoint <ckpt>`) — held-out NLL per source × per class + JSON report + reliability diagram + ECE + qualitative renders for the four target fantasy maps.
10. **Inference module** (`mapclass.infer`) — the **stable contract** with the hex-grid app. `load_model(ckpt) -> Predictor`; `predictor.predict(image) -> {land_cover, topography, model_version, taxonomy}`. Output is **probabilities by default**, not logits; output spatial shape matches input.

**Suggested build order (ARCHITECTURE.md "Build Order (Dependency DAG)"):**

A (sample contract) → fan out to **B (OSM sub-pipeline) + C (auto-georef offline) + D (Backbone interface + Mock backbone)** → E (`mapclass.data.*`) → F (`mapclass.model.*` heads + losses) → G (`train`/`eval`/`infer`) → **H (v0 training, cloud VM)** → I (auto-georef bootstrap mode) → J (v1 training).

**Parallel-safe pairs:** B + C + D after A; LandCoverHead + TopographyHead + OCR-head within F.

**Sequential bottlenecks:** A is the choke point; H → I → J cannot be parallelised.

### Table-stakes (TS-1..TS-11) and anti-features

Detailed in [FEATURES.md](FEATURES.md). All eleven table-stakes verified item-by-item against PROJECT.md Out-of-Scope; none collide.

**Table-stakes (must land in v1; map to PROJECT.md REQ-IDs as noted):**

| # | Feature | Maps to / supports |
|---|---------|---------------------|
| TS-1 | Reproducible training entrypoint with config + seeding | TRAIN-01, TRAIN-02 |
| TS-2 | Safetensors checkpoint format (no pickle `.bin`) | SHIP-01 |
| TS-3 | Inference Python module with stable single-call API (`load_model`, `predict`, `predict_batched`) | SHIP-01 — the contract with the hex-grid app |
| TS-4 | Probability output (not argmax / not logits) on both heads | SHIP-01, EVAL-01 |
| TS-5 | Held-out eval script reporting NLL per source per class + JSON report | EVAL-01, EVAL-03 |
| TS-6 | Calibration sanity (reliability diagram + ECE) appended to eval | EVAL-01, SHIP-01 (Pitfall 11 mitigation) |
| TS-7 | Training-time logging (loss curves + sample preds) | All TRAIN; observability-replaces-tests |
| TS-8 | Model + dataset versioning embedded in safetensors metadata header | SHIP-01 (also Pitfall 4, 5, 16 mitigation) |
| TS-9 | CPU and 4–8 GB GPU inference paths via `device='auto'` | SHIP-01 |
| TS-10 | Deterministic train/val/test split from sample-id hash, committed | EVAL-01, EVAL-03 (Pitfall 5 mitigation) |
| TS-11 | Output shape + dtype contract documented in inference module README | SHIP-01 |

**Cheap differentiators worth landing in v1 (FEATURES D-list, S-cost):** D-1 (multi-style synthetic — already in tree, regression-protect), D-2 (artifact augmentation wired in — fixes CONCERNS.md 2a dead-code), D-3 (rotation augmentation), **D-5 (bootstrap-loop quality monitor — this is the kill-switch for Pitfall 1; non-negotiable)**, D-6 (per-source weights as configurable hook), **D-10 (lockfile — non-negotiable per Pitfall 19)**.

**Defer to v1.x post-ship:** D-4 (OSM stylesheet diversity), D-7 (INT8/INT4 inference quantisation), D-8 (batched inference helper), **D-9 (tile-and-stitch helper)** — promote D-9 to v1 table-stakes only if MODEL-01's native input is < 512 (likely; SmolVLM uses 384 patches and fantasy maps are routinely 4096²+, so D-9 will probably move up).

**Anti-features (PROJECT.md Out-of-Scope — DO NOT re-promote during planning):** polygon tracing / region delineation, hex-grid aggregation, EU4/CK3/HoI4 engine-specific output formats, hand-annotated fantasy test set, hand-annotation tooling integration (Label Studio / CVAT), auto-georef as inference/app feature, **PaliGemma-3B as the v1 ship/inference backbone** (it stays as MODEL-04 benchmark only), zero-shot CLIP / SigLIP / OpenCLIP / PaliGemma baselines, dynamic-LRP mechanistic failure analysis, component ablations, paper write-up, open-dataset redistribution, tests / CI / linting, cloud / docker / VM provisioning, web UI / Gradio demo, multi-language OCR (Cyrillic / CJK / Arabic), full temperature-scaling fit (TS-6 diagnostic only).

> **Reconciliation note for the roadmapper:** FEATURES.md was written before MODEL-04 / EVAL-03 were locked, so its anti-features list still says "PaliGemma-3B as the inference backbone" without distinguishing inference-vs-benchmark. PROJECT.md's current locked scope is the truth: **PaliGemma-3B ship/inference backbone is excluded; PaliGemma-3B benchmark training under MODEL-04 with EVAL-03 comparison is in scope.** STACK.md and PITFALLS.md account for the split correctly.

### Top pitfalls — by severity

Detailed in [PITFALLS.md](PITFALLS.md). 19 pitfalls total; 5 CRITICAL + 9 HIGH + 5 MEDIUM. The roadmapper should treat the CRITICAL list as gating items.

**CRITICAL (5):**

| # | Pitfall | REQ-IDs | One-line prevention |
|---|---------|---------|---------------------|
| 1 | **Bootstrap-loop error compounding** (v0 → v1 gets worse, not better) | GEOREF-02, TRAIN-02, EVAL-01, EVAL-03 | Mandatory v0-vs-v1 NLL kill-switch (FEATURES D-5); ship v0 if v1 doesn't beat it by ≥2%. Bootstrap depth = 1 by contract. |
| 2 | **PaliGemma-vs-small-backbone comparison contamination** (EVAL-03 invalid) | EVAL-03, MODEL-04, TRAIN-01, TRAIN-02 | **Pre-registered EVAL-03 protocol committed before either training run starts** — same `splits.json`, same `loss_weights.yaml`, same augmentation, same fine-tune budget; only `backbone:` field differs. Hash assertion at compare time. |
| 3 | **Loss-weight schema silently misaligned with canonical taxonomy** | DATA-03, DATA-06, MODEL-03, TRAIN-01, TRAIN-02 | Strict schema validator at training startup in `LossWeights.load()`; taxonomy hash embedded in checkpoint metadata (TS-8); refuse to evaluate on hash mismatch. |
| 4 | **Tokenizer/processor mismatch when swapping backbones** | MODEL-01, MODEL-04, TRAIN-01 | `Backbone` ABC owns `preprocess()`; preprocessor identity in checkpoint metadata; reference-image hash-check on inference load. |
| 5 | **Training-eval contamination via held-out split drift** | EVAL-01, EVAL-03, TRAIN-01, TRAIN-02 | Split by sample-id hash (not glob order); `splits.json` committed once at v0 build time; bootstrap can only add to train (filter buckets 8–9); eval-time overlap assertion. |

**HIGH (9), compact table:**

| # | Pitfall | REQ-IDs | Prevention summary |
|---|---------|---------|--------------------|
| 6 | PaliGemma VRAM blowup during training | MODEL-04, TRAIN-01, TRAIN-02 | One-batch dry-run on cloud VM before scheduling full run; default to QLoRA + gradient checkpointing; train at 224 first |
| 7 | Frozen-vs-LoRA-vs-full mis-pick on small backbone | MODEL-01, TRAIN-01 | Default LoRA on attention Q/V only, rank=16, alpha=32; run a 30-epoch frozen baseline first; track per-source NLL |
| 8 | OCR false positives on textured map elements | MODEL-02 | Train OCR sub-pipeline-first, joint-train only after isolation; λ=0.1 default; mine hard negatives from cartographic textures; per-batch detection-count diagnostic |
| 9 | OCR LM-bias hallucinating fantasy names | MODEL-02 | Fixed-vocabulary recognizer or near-zero LM weight in beam search; document Latin-only English-trained limitation in inference README |
| 10 | Class imbalance dominates segmentation training | DATA-06, MODEL-03, TRAIN-01 | Focal CE (gamma=2); pixel-frequency class weights × per-source weights; gate ship on worst-class NLL not just mean |
| 11 | Segmentation-head miscalibration (overconfident wrong predictions) | MODEL-03, EVAL-01, EVAL-02, SHIP-01 | TS-6 reliability diagram + ECE in eval; temperature scaling in checkpoint metadata; label smoothing 0.05 during training; calibrated probs by default in inference |
| 12 | Tile-boundary edge effects in dense segmentation inference | MODEL-03, SHIP-01 | Overlap with feathered (cosine-window) blending; reflection padding to multiple of tile size; **promote D-9 to v1 table-stakes if MODEL-01 native input < 512** |
| 13 | TPS-warping pathologies on auto-georef | GEOREF-01, GEOREF-02 | Reject GCP configs with hull <30% of image area; regularised TPS with smoothing; cap GCPs at 50–200; sanity-check mode passes on known-good maps first |
| 14 | Auto-georef silent "no good match" failure | GEOREF-01, GEOREF-02 | Peak-significance ratio threshold (≥1.5); GCP coverage threshold; pre-classify Rumsey maps by LUNA scale/region tags; manifest-as-rejection-record |

**MEDIUM (5):** seeds (Pitfall 15), OSM ODbL attribution (16), OSM CRS / freshness mismatches (17), single-author bus factor (18), training-script-repo / VM-provisioning-repo drift (19). Roadmapper: read PITFALLS.md directly for these if/when relevant; only #19 (lockfile) is non-deferrable per the task brief.

## Implications for Roadmap

### Phase-shaping recommendations

The research collectively suggests phase boundaries that follow the architecture's build-order DAG, with three roadmap-level decisions that cannot live inside a single phase. The roadmapper structures phases; what follows are the cross-cutting items the roadmapper must explicitly schedule.

**Roadmap-level decisions (cannot be punted to a phase planner):**

1. **EVAL-03 fairness protocol must be pre-committed before either training run starts.** PITFALL 2 is CRITICAL and the prevention strategy explicitly requires `EVAL-03_protocol.md` (one page: held-out split hash, loss-weights hash, augmentation hash, fine-tune approach, evaluation seed, kill-switch criterion) committed before MODEL-01 and MODEL-04 phases begin. This is a roadmap-level artifact, sequenced before the training phases — not a deliverable inside one of them. Recommended placement: at the end of the phase that delivers the sample contract + splits + loss-weights schema (the "data foundations" phase, build-step A in the architecture DAG), or as a dedicated tiny pre-training-phase artifact.

2. **Bootstrap quality-monitor / kill-switch (FEATURES D-5) must be a v1 deliverable, not a v1.x.** This is the prevention for PITFALL 1 (CRITICAL) and the gate that determines whether v1 ships v0's checkpoint or v1's. The roadmap's GEOREF-02 / TRAIN-02 phase must include the eval-on-both, NLL-comparison gate, and an explicit "ship v0 if v1 regresses" success-criterion — that path is a first-class outcome, not a fallback.

3. **Lockfile (FEATURES D-10) cannot be deferred.** PITFALL 19 is MEDIUM but the prevention is "lockfile is mandatory, not optional" because the cloud-VM SSH workflow's two-repo split makes deferral break the training run remotely. Recommended placement: in the earliest infrastructure phase (alongside `pyproject.toml`, `.python-version`, and a 30-line `mapclass.preflight_check()`).

**Suggested phase grouping (the roadmapper assigns IDs and exact boundaries):**

- **Foundation phase** — sample-contract finalization (manifest.json schema + validator), loss-weights YAML schema + validator, taxonomy-hash utility, deterministic split utility, `pyproject.toml` + `.python-version` + lockfile + preflight, seeding utility. Build-step A. Mitigates Pitfalls 3, 5, 15, 19.
- **Parallel fan-out phase** — OSM sub-pipeline (build-step B), auto-georef offline / sanity-check mode (build-step C), Backbone ABC + Mock backbone (build-step D). Mitigates Pitfalls 13, 14 (sanity-check mode lets TPS pathologies be diagnosed before any model exists), 17 (OSM CRS round-trip).
- **Model-assembly phase** — `mapclass.data.*` (build-step E) → `mapclass.model.*` heads + losses (build-step F) → `train` / `eval` / `infer` entry points (build-step G). Eval script delivers TS-5 + TS-6 (calibration). Inference module delivers TS-3 + TS-4 + TS-9 + TS-11. Mitigates Pitfalls 4, 8, 11, 12.
- **Training phases** — v0 (build-step H) and the parallel PaliGemma v0 training together; the bellwether discipline (PITFALL 2 prevention) means these need to run from byte-identical configs differing only in `backbone:`. Embed EVAL-03 fairness protocol as the entry criterion. Mitigates Pitfalls 6, 7, 10, 15.
- **Bootstrap + retrain phase** — auto-georef bootstrap (build-step I), v1 training for both backbones (build-step J), v0-vs-v1 NLL kill-switch evaluation. Final EVAL-01 + EVAL-03 numbers produced here. Mitigates Pitfalls 1, 13, 14.
- **Ship phase** — packaging the inference module, OSM attribution + Gemma license attribution in checkpoint metadata, EVAL-02 qualitative renders, README finalisation. Mitigates Pitfalls 9, 11, 16.

The architecture DAG's parallel-safe pairs (B+C+D, then heads within F) are the natural intra-phase parallelism opportunities; the orchestrator can dispatch parallel agents inside the fan-out phase and the model-assembly phase.

### Research Flags

Phases likely needing deeper `/gsd-research-phase` runs during planning:

- **OCR module phase (MODEL-02 implementation):** STACK.md explicitly flags MEDIUM confidence on the OCR starting point — docTR pretrained may need fine-tuning on map text, and there is no off-the-shelf 2025 SOTA for "rotated, curved, often-stylized text on stylized maps." If docTR-pretrained fails on synthetic + historical training images, a phase-specific research block should evaluate map-text fine-tuning datasets (DocBank, MapText competition data, synthetic curved-text generation) before committing more time to MODEL-02. Fallback: drop OCR for v1 ship — the seg-head NLL is the actual ship metric (Pitfall 8 recovery).
- **Auto-georef bootstrap phase (GEOREF-02):** TPS pathologies (PITFALL 13) and silent no-match failures (PITFALL 14) are well-known but the specific thresholds (peak-significance ratio, GCP convex-hull coverage, regularised-TPS smoothing, GCP cap) need to be tuned on the project's own sanity-check set. If sanity-check residuals on already-registered Rumsey maps are bad, the TPS / cross-correlation pipeline itself is suspect and the research should reopen the auto-georef approach (maybe SuperPoint+SuperGlue feature matching instead of phase correlation against WorldCover).
- **EVAL-03 fairness protocol:** not a deeper-research item — it's a 1-page protocol document — but it is a roadmap-shaping artifact that the roadmapper must explicitly schedule before MODEL-01 and MODEL-04 training phases.

Phases with standard patterns (skip phase-internal research):

- **Sample contract + validators + lockfile + seeding** — well-trodden engineering; STACK.md and ARCHITECTURE.md cover the patterns directly.
- **Backbone interface (Pattern 3) + heads + losses + training loop** — standard PyTorch + transformers fine-tuning shape. ARCHITECTURE.md's swap-surface and STACK.md's "Stack Patterns by Variant" sections cover both backbones explicitly.
- **Inference module (`mapclass.infer`)** — FEATURES.md TS-3 specifies the API surface; ARCHITECTURE.md specifies the data-flow contract. Implementation is pattern-following.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | **HIGH** | Backbone IDs and licenses verified on HF; transformers 4.47 PaliGemma 2 support confirmed in release notes; Apache-2.0 vs Gemma TOU split clearly understood. ONE MEDIUM area: OCR module (`docTR`) may need fine-tuning on map text. |
| Features | **HIGH** | Boundaries are tightly specified by PROJECT.md; only feasibility/cost ratings carry MEDIUM confidence. All 11 table-stakes verified item-by-item against Out-of-Scope list; no collisions. |
| Architecture | **HIGH** | Component boundaries anchored in existing code; sample-contract / loss-weights / swap-surface / filesystem-bootstrap patterns all verified against the brownfield. ONE MEDIUM area: specific backbone choice was deferred to STACK research, now locked to SmolVLM-500M and PaliGemma2-3b-pt-224. |
| Pitfalls | **HIGH** on ML-literature pitfalls (well-documented), MEDIUM on project-management pitfalls (single-author specific), MEDIUM-HIGH on OSM licensing. |

**Overall confidence:** HIGH. The project is well-bounded by PROJECT.md, the architecture matches the brownfield, and the stack picks are conservative. The dominant risk is execution discipline around the five CRITICAL pitfalls — none of which are unknown unknowns.

### Gaps to Address

These are open items the roadmapper or phase planners must answer; each cites the source-file question for traceability.

- **Tile-and-stitch promotion** — STACK.md confirms SmolVLM-500M's vision encoder is SigLIP-B/16 with 384-input patches; FEATURES D-9 says "promote to v1 table-stakes if backbone native input ≤ 512." Decision: **promote D-9 to table-stakes for v1**, given the consumer's input distribution (fantasy maps routinely 4096²+). The roadmapper should encode this decision in the inference-module phase. Source: FEATURES.md "Dependency notes" + PITFALL 12.
- **Frozen-vs-LoRA-vs-full-finetune for SmolVLM** — STACK.md says "full fine-tune is feasible (~500M params on a 16 GB VM is comfortable). PEFT/LoRA is optional." PITFALL 7 says "default to LoRA on attention-Q/V only, rank=16, alpha=32; run a 30-epoch frozen-backbone baseline first." Phase planner for MODEL-01 / TRAIN-01 must pick the default and the budget for the frozen-baseline comparison. Source: PITFALL 7 prevention strategies 1–2.
- **OCR map-text fine-tuning data** — STACK.md flags this as the LOW-confidence item: "no off-the-shelf 2025 SOTA model that solves rotated, curved, often-stylized text on stylized maps." The MODEL-02 phase needs a deferred-research block (see Research Flags above). v1 fallback path: ship without OCR if it's net-negative. Source: STACK.md "OCR module — MODEL-02" LOW-confidence flag + PITFALL 8 recovery.
- **OSM source implementation choice** (Overpass API vs PBF + tile renderer) — ARCHITECTURE.md says "STACK research phase decides between Overpass and PBF." STACK.md does not redecide — implicit punt to the DATA-05 phase planner. Source: ARCHITECTURE.md "Pitfall 17 prevention strategy 2."
- **Bootstrap kill-switch margin** (PITFALL 1 prevention says "by at least 2% — a small but non-zero margin") — exact margin is configurable; the roadmapper should treat 2% as a sensible default but flag for adjustment if v0 NLL is unstable. Source: PITFALL 1 prevention strategy 1.

## Sources

### Primary research files (authoritative, in this repo)

- `.planning/research/STACK.md` — backbone IDs, framework, checkpoint format, OCR starting point, alternatives, what NOT to use
- `.planning/research/FEATURES.md` — table-stakes (TS-1..11), differentiators (D-1..11), anti-features list, inference-module API sketch, prioritization matrix
- `.planning/research/ARCHITECTURE.md` — system overview, component responsibilities, four patterns (sample contract, loss weights, backbone swap surface, filesystem bootstrap), data flow, build-order DAG, anti-patterns
- `.planning/research/PITFALLS.md` — 19 pitfalls (5 CRITICAL + 9 HIGH + 5 MEDIUM), prevention strategies, recovery costs, "looks-done-but-isn't" checklist, pitfall-to-phase mapping
- `.planning/PROJECT.md` — locked v1 scope (DATA-01..06, GEOREF-01..02, MODEL-01..04, TRAIN-01..02, EVAL-01..03, SHIP-01); Out-of-Scope list; constraints; key decisions

### Brownfield context (anchoring)

- `.planning/codebase/STACK.md` — existing pinned-but-unused `torch / transformers / peft / accelerate / bitsandbytes`; absence of `pyproject.toml`
- `.planning/codebase/ARCHITECTURE.md` — sample-contract on disk in the existing pipelines
- `.planning/codebase/CONCERNS.md` — items 2a (`augment.py` dead code), 8a (no seeds), 8b (no dataset versioning), 8c (no lockfile)
- `.planning/codebase/TESTING.md` — six untested high-risk areas; gap-item 6 (loss-weight schema fragility) directly informs PITFALL 3

### External (high confidence; full bibliographies in source files)

- HuggingFace model cards for SmolVLM, Moondream2, PaliGemma 2, Florence-2 (in STACK.md)
- transformers 4.47 release notes (PaliGemma 2 support)
- Arazo et al. 2019 (pseudo-labeling confirmation bias) → PITFALL 1
- Mask-TS Net 2024 (segmentation miscalibration) → PITFALL 11
- Unified Focal Loss → PITFALL 10
- Tiling artifacts and feature normalization → PITFALL 12
- OSMF Attribution Guidelines + ODbL → PITFALL 16
- GDAL TPS docs + TPS pathology literature → PITFALL 13

---
*Research completed: 2026-05-08*
*Ready for roadmap: yes*
