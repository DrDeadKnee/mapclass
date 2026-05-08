# Walking Skeleton — MapClass / GeoViLM

**Phase:** 1
**Generated:** 2026-05-08

## Capability Proven End-to-End

A developer can run `python -m mapclass.train --config mapclass/configs/v0.yaml` against ~100 synthetic Azgaar samples, write a `safetensors` checkpoint with embedded `{model_version, dataset_manifest_sha, taxonomy_hash, training_seed, source_class_weights_hash, backbone}` metadata, then run `python -m mapclass.infer <ckpt> <image>` to get land-cover `[9, H, W]` + topography `[3, H, W]` probability tensors that sum to 1.0, and finally `python -m mapclass.eval --split heldout --checkpoint <ckpt>` to write a per-source per-class NLL `eval_report.json`. The Mock backbone is a learnable conv stub so the loss curve visibly decreases — proof the gradient path is real.

## Architectural Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Package layout | `mapclass/` package alongside existing `scripts/`; `scripts/build_dataset.py` and `scripts/build_historical_dataset.py` stay where they are. `scripts/__init__.py` (new, empty) makes `from scripts.biome_mapping import ...` clean. | Brownfield: synthetic + historical pipelines are validated DATA-01..04. Phase 1 only adds new code, never moves existing code. CONTEXT.md `## Claude's Discretion`. |
| Model framework | PyTorch `>=2.4,<3.0` + HF transformers `>=4.47,<5.0` + safetensors `>=0.4.5`; no Lightning, no custom logger. | research/STACK.md "Recommended Stack". |
| Backbone swap surface | `Backbone(ABC)` with `extract_features(image) -> dict[str, Tensor]` and `preprocess(image) -> Tensor`. Phase 1 implements `MockBackbone` (~10–100k params). Phase 2 swaps in `SmolVLMBackbone`; Phase 5 swaps in `PaliGemma2Backbone`. | research/ARCHITECTURE.md Pattern 3 + PITFALL 4 prevention #1. |
| Mock backbone fidelity | Learnable conv stub (D-01), single-key feature dict `{"features": [B, 64, H/?, W/?]}` (D-02), seeded init (D-03), `backbone: 'mock'` in metadata (D-04). | CONTEXT.md `## Implementation Decisions / Mock Backbone Design`. |
| Sample contract | On-disk per-sample directory with `image*.png + land_cover.png + topography.png + sample_weights.json + manifest.json`. `mapclass.data.contract.assert_sample_valid()` raises `SampleContractError(ValueError)` LOUDLY at training startup. | research/ARCHITECTURE.md Pattern 1 + CONTEXT.md `assert_sample_valid()` scope. |
| Per-source loss-weights schema | `mapclass/configs/loss_weights.yaml`, sources × land-cover-class × float plus per-source topography float. Strict validator: every source row MUST list ALL 9 land-cover classes (explicit `0.0` for absent). `LossWeights.load()` raises `LossWeightsSchemaError(ValueError)` on schema drift; computes `source_class_weights_hash = sha256(yaml_text)`. | research/ARCHITECTURE.md Pattern 2 + PITFALL 3 prevention #1. |
| Deterministic splits | Sample-id-hash buckets (sha256 first-32-bits → bucket 0–9): 0–7 train, 8 val, 9 test. Committed `mapclass/configs/splits.json`; eval-time overlap assertion raises `SplitsContaminationError(ValueError)`. | PITFALL 5 prevention + CONTEXT.md splits.json structure. |
| Checkpoint format | `safetensors.torch.save_file(state_dict, path, metadata={...})`. Metadata is `dict[str, str]` — every value JSON-encoded. Required keys: `model_version`, `dataset_manifest_sha`, `taxonomy_hash`, `training_seed`, `source_class_weights_hash`, `backbone`, `processor_identity`. `mapclass.infer.load_model` raises `TaxonomyHashMismatchError(ValueError)` on taxonomy hash drift. | FEATURES.md TS-2 + TS-8 + PITFALL 4 prevention #3. |
| Config system | OmegaConf YAML at `mapclass/configs/v0.yaml`. Light dataclass typing later; Phase 1 is a flat YAML. | research/STACK.md "OmegaConf + dataclass-typed configs". |
| Logging | `print()` only. TensorBoard for loss curves under `runs/`. No `logging` module, no `warnings.warn`. | codebase/CONVENTIONS.md. |
| Seeding | `mapclass.seeding.set_global_seed(seed)` pins random + numpy + torch + cudnn deterministic + `PYTHONHASHSEED`. Single value, no `--no-seed` override. | PITFALL 15 prevention #1 + CONTEXT.md D-03. |
| Lockfile | `pyproject.toml` + `requirements.lock.txt` + `.python-version` ALL land in Phase 1. Generated via `uv pip compile pyproject.toml -o requirements.lock.txt`. | PITFALL 19 (non-deferrable) + CONCERNS.md 8c. |
| EVAL-03 protocol | Markdown-only at `mapclass/configs/EVAL-03_protocol.md`. Contains the literal string `DECIDE_AT_PHASE_5` (D-08) so Phase 5's `eval_03_compare.py` can grep-and-block. | CONTEXT.md D-05..D-08 + PITFALL 2 prevention setup. |
| Inference module isolation | `mapclass.infer` imports only `mapclass.data.taxonomy` (re-export shim) + numpy + PIL + safetensors + torch. NEVER imports `mapclass.data.dataset`/`contract`/`loss_weights`/`splits`. Enforcement: Phase 6 startup-import test; Phase 1 honors by discipline. | research/ARCHITECTURE.md Anti-Pattern 4. |

## Stack Touched in Phase 1

- [x] Project scaffold — `pyproject.toml` (setuptools, `requires-python = ">=3.10"`), `.python-version` (`3.10`), `requirements.lock.txt` (uv-compiled exact pins), empty `mapclass/{__init__,data/__init__,model/__init__}.py`, empty `scripts/__init__.py`.
- [x] Routing — `python -m mapclass.train`, `python -m mapclass.infer`, `python -m mapclass.eval` (three `__main__` entrypoints).
- [x] Real I/O — one real `safetensors.torch.save_file` write at the end of training, one real `safetensors.torch.load_file` + metadata-string-decode read at inference. No DB; this is a single-author training pipeline.
- [x] User interaction — CLI: `python -m mapclass.train --config mapclass/configs/v0.yaml` produces `models/geovilm_phase1_mock.pt`; `python -m mapclass.infer <ckpt> <image>` returns probabilities; `python -m mapclass.eval --split heldout --checkpoint <ckpt>` writes `eval_report.json`. No web UI.
- [x] Deployment — `pip install -e .` from a fresh clone of `refactor_paper` succeeds against `pyproject.toml + requirements.lock.txt + .python-version`; `python -c "import mapclass"` works. No Docker, no cloud build (per PROJECT.md).

## Out of Scope (Deferred to Later Slices)

These are NOT in the Phase 1 walking skeleton. They belong to specifically named later phases:

- **Real backbone (SmolVLM-500M)** — Phase 2. Phase 1 only ships `MockBackbone`.
- **Per-source loss-weight WIRING through training** — Phase 2. Phase 1 ships the schema + validator + hash, but does NOT yet multiply per-pixel losses by per-source weights inside the trainer (D-02 implication: keep training trivial in Phase 1).
- **Multi-stage feature dict (`{"stage_2": ..., "stage_3": ...}`)** — Phase 2. Phase 1's MockBackbone emits single-key `{"features": ...}` (D-02).
- **PaliGemma backbone** — Phase 5.
- **OCR module (`mapclass.model.ocr_head`)** — Phase 3.
- **OSM road-tile source** — Phase 3.
- **Auto-georef sanity-check + bootstrap loop** — Phase 4.
- **v1 retrain + v0-vs-v1 NLL kill-switch** — Phase 4.
- **Tile-and-stitch helper for >384px inputs** — Phase 6.
- **Qualitative renders (Tolkien / Westeros / Abercrombie / Warhammer)** — Phase 6.
- **Calibration / reliability diagram / ECE in eval report** — Phase 2+ (TS-6). Phase 1's `eval_report.json` schema reserves an optional `calibration` field.
- **Tests, CI, linting/formatting** — Out of scope at v1 per PROJECT.md.
- **Mock-checkpoint refusal at infer/ship time** — discipline-only at v1 per D-04 (revisit at milestone 2 if shipped accidentally).
- **Permitted-Differences table for EVAL-03** — explicitly rejected (D-07: strict fairness). Phase 5 may renegotiate after a feasibility dry-run.

## Subsequent Slice Plan

Each later phase adds one vertical slice ON TOP of this skeleton without altering the architectural decisions above:

- **Phase 2** — `MockBackbone` → `SmolVLMBackbone(Backbone)` swap. Wire per-source loss weights through the trainer's per-pixel loss tensor. Refactor seg heads to consume multi-stage feature dict per ARCHITECTURE.md Pattern 3 (the D-02 debt). Add `rumsey_registered` source to the dataset; adds TensorBoard sample-prediction images.
- **Phase 3** — Add `road_osm` source (DATA-05) via a new `scripts/build_road_dataset.py`. Add `mapclass.model.ocr_head` (docTR pretrained). Train v0 to completion on all three sources; commit `models/geovilm_v0_small.pt` and an updated `eval_reports/v0_small.json` with per-source NLL + reliability diagram + ECE (TS-6 lands here).
- **Phase 4** — Add `scripts/georef.py` (sanity-check + bootstrap modes). Run bootstrap loop; produce `rumsey_bootstrapped` source rows; retrain v1; gate ship with v0-vs-v1 NLL kill-switch (≥2% margin).
- **Phase 5** — Add `mapclass.model.paligemma_backbone.PaliGemma2Backbone` (parallel concrete subclass of the same `Backbone` ABC). Run EVAL-03 NLL comparison against the locked Phase 1 `splits.json` and `loss_weights.yaml` (byte-identical hashes assert at compare time). REPLACE the `DECIDE_AT_PHASE_5` literal in `EVAL-03_protocol.md` with a numeric threshold via a separate commit (D-08 enforcement).
- **Phase 6** — Finalize `mapclass.infer.load_model()` API contract; add tile-and-stitch (cosine-window blending) for >384px inputs; render four qualitative spot-check images; embed full `{license, attribution, processor_identity, ...}` in shipped checkpoint metadata; ship the small-backbone variant.
