# Phase 1: End-to-End Skeleton (Synthetic + Mock Backbone) - Context

**Gathered:** 2026-05-08
**Status:** Ready for planning

<domain>
## Phase Boundary

Wire the whole training + inference + eval pipeline end-to-end on synthetic data with a Mock backbone. The deliverable is the **plumbing**, not the score: contracts, configs, entry points, and the EVAL-03 protocol document are real and locked before any real model lands. Eval numbers will be near `log(num_classes)` because the backbone is Mock — that is expected and correct.

**In scope (mapped from ROADMAP.md success criteria 1–6):**
- `mapclass/` Python package with `mapclass.train`, `mapclass.infer`, `mapclass.eval`, `mapclass.data`, `mapclass.model` modules
- `Backbone` ABC + `MockBackbone` concrete subclass (per research/ARCHITECTURE.md Pattern 3)
- Sample contract validator `mapclass.data.contract.assert_sample_valid()`
- `LossWeights.load()` with strict YAML schema validator (PITFALL 3 prevention)
- Deterministic sample-id-hash splits → committed `splits.json` (PITFALL 5 prevention)
- Safetensors checkpoints with embedded metadata (TS-2, TS-8 from research/FEATURES.md)
- `pyproject.toml` + `.python-version` + `requirements.lock.txt` (PITFALL 19 / FEATURES D-10, non-deferrable)
- Seeding utility (PITFALL 15)
- `mapclass/configs/EVAL-03_protocol.md` — pre-registered bellwether contract (PITFALL 2 prevention setup)

**Out of scope (belongs to other phases):**
- Real backbone (SmolVLM/PaliGemma) — Phases 2 / 5
- Per-source loss-weight wiring through training — Phase 2
- OCR module — Phase 3
- OSM source — Phase 3
- Auto-georef tool — Phase 4
- Bootstrap loop — Phase 4
- Tile-and-stitch / qualitative renders — Phase 6

</domain>

<decisions>
## Implementation Decisions

### Mock Backbone Design

- **D-01: Output is a tiny learnable conv stub.** Mock is `nn.Module` subclass with ~10–100k learnable parameters (1–2 conv layers RGB → features). Training actually backprops through it. Phase 1 exercises the full training path: optimizer, scheduler, mixed precision, gradient flow. Eval NLL may improve modestly over the class-frequency prior — that's a positive diagnostic ("trainer is doing something"), not a ship metric.
  - **Why:** Pure-random Mock would only test shapes. A learnable stub forces the trainer's gradient path to be real before SmolVLM lands in Phase 2.

- **D-02: Feature dict has a single key, single tensor.** `extract_features(image) -> {"features": [B, C, H/p, W/p]}`. Multi-stage skip-connection plumbing is NOT exercised in Phase 1 — that test waits for Phase 2 when SmolVLM emits real multi-stage features. Seg heads in Phase 1 consume the single tensor directly.
  - **Why:** MVP simplest-end-to-end-first; multi-stage is a Phase-2 concern.
  - **Phase-2 implication:** When SmolVLM lands, the head code MUST be re-shaped to consume multi-stage `dict[str, Tensor]` per ARCHITECTURE.md Pattern 3. Plan Phase 2 with this refactor scope explicit.

- **D-03: Mock's init is seeded from config `training_seed`.** Two runs with the same seed produce byte-identical loss curves. Single seed value, no `--no-seed` override flag (PITFALL 15 prevention from day one). Real backbones in Phase 2+ inherit the same seeding utility.
  - **Why:** Diff-testing future pipeline changes requires a deterministic baseline.

- **D-04: Mock-trained checkpoints carry `backbone: 'mock'` in safetensors metadata.** No enforcement at load or ship time — discipline-only. `mapclass.infer.load_model()` accepts Mock checkpoints normally. Phase 6 packaging accepts them too.
  - **Why:** Lightest-weight option. Risk that someone ships a Mock checkpoint accepted as a discipline trade-off — the operator-friction cost of refusal logic was judged not worth it for a single-author project.
  - **Phase-6 implication:** Phase 6 plan should still verify `backbone:` field in metadata at packaging time (sanity check), even if not blocking.

### EVAL-03 Protocol Artifact

- **D-05: Protocol is a single Markdown file.** `mapclass/configs/EVAL-03_protocol.md`, one page, human-readable. NOT JSON, NOT a hybrid. Phase 5 reads and respects it via discipline; no programmatic hash-comparison against the protocol itself.
  - **Why:** Single-author project; the hash-assertion test in Phase 5 compares the configs (`v0.yaml` vs `v0_paligemma.yaml`) against each other, not against the protocol. The protocol is the audit trail, not the enforcement mechanism.
  - **Required content (from ROADMAP.md Phase 1 success criterion 6):** held-out split hash, loss-weights file hash, augmentation policy, fine-tune budget shape, evaluation seed, kill-switch criterion (per next decision).

- **D-06: Comparison logic is locked in Phase 1; numeric threshold is deferred to Phase 5 with `DECIDE_AT_PHASE_5` placeholder.** The protocol commits the rule structure (per-source per-class NLL gap, decision logic — "small backbone ships if NLL gap to PaliGemma is within X% relative on the held-out splits, otherwise reconsider MODEL-01") but leaves `X` as the literal string `DECIDE_AT_PHASE_5`. Phase 5 fills it in based on Phase 3 v0 results.
  - **Why:** Pre-registers the structure (PITFALL 2 partial prevention); allows the threshold to be calibrated to actual v0 NLL stability before Phase 5 runs. Pure pre-registration would force a guess; pure deferral would invite goalpost shifting. Compromise picks the structure now, the number later.

- **D-07: EVAL-03 fairness is STRICT — only the `backbone:` field differs.** PaliGemma trains at SmolVLM's resolution / batch-size / dtype / precision. No Permitted Differences table.
  - **Why:** Pure controlled experiment. Maximum bellwether interpretability.
  - **⚠ KNOWN TENSION (must surface in Phase 5 plan):** STACK.md says PaliGemma-3B needs NF4 + LoRA on a 24 GB GPU, and SmolVLM's native input is 384 vs PaliGemma's 224. Strict fairness may make the PaliGemma run infeasible (OOM at 384 + FP16) or distort it (PaliGemma trained at non-native resolution). Phase 5 planner must verify a one-batch dry-run on the cloud VM before committing to strict fairness — if it OOMs, this decision is re-opened in Phase 5 (NOT in Phase 1).
  - **Phase-5 implication:** Phase 5 plan MUST include a "PaliGemma feasibility dry-run" task gated BEFORE the full training run; if dry-run fails, return-to-discuss this decision.

- **D-08: Pre-registration of the deferred threshold is enforced by a two-commit policy + script-level guard.** Protocol.md states: "The `DECIDE_AT_PHASE_5` placeholder MUST be replaced with a numeric threshold in a SEPARATE commit before `eval_03_compare.py` runs." Phase 5's `eval_03_compare.py` script reads the protocol file at startup and refuses to run (raises `RuntimeError`) if the literal string `DECIDE_AT_PHASE_5` is still present.
  - **Why:** Mechanical block — discipline alone is fragile. Ensures the threshold is set BEFORE comparison numbers are visible.
  - **Phase-5 implication:** Phase 5 plan MUST include "commit threshold value" as a discrete, named task that runs before the compare script.

### Claude's Discretion

The following Phase 1 implementation decisions were not selected for discussion. Defaults below — Phase 1 planner can reopen them if needed during planning, but should not surface them again to the user without cause.

- **Package layout for `mapclass/`:** New `mapclass/` package introduced alongside existing `scripts/` tree. `scripts/build_dataset.py` and `scripts/build_historical_dataset.py` stay where they are (they are entry points already validated as DATA-01..04). Synthetic / historical dataset code in `scripts/` is NOT moved into `mapclass/`. Rationale: minimal churn, keeps validated pipelines untouched, lets Phase 1 focus on the new model code. `mapclass.data.contract` (the validator) is new code in the new package.
- **Skeleton-run dataset size:** ~100 synthetic samples. Big enough to exercise batching with batch sizes 4–16 (real-ish iterations); small enough that one Phase 1 smoke run completes in under 5 minutes on CPU. Number is a sensible default — Phase 1 planner can adjust if iteration time matters.
- **`assert_sample_valid()` scope:** Strict at startup, run once per training run (not per-batch). Validates: (a) all required files exist, (b) image dimensions match across `image*.png + land_cover.png + topography.png`, (c) label values are in `[0, num_classes)` modulo `WATER_TOPO=255` for topo, (d) `manifest.json` matches the schema, (e) `sample_weights.json` row exists for the sample's `source_subtype`. Failure raises a custom `SampleContractError` exception (loud — NOT a warning, NOT silent).
- **Eval report format:** JSON shape with per-source × per-class NLL, per-source mean NLL, overall mean NLL, sample count per source, and a `notes` field. Phase 1 establishes the schema; Phase 2+ extends with reliability diagram + ECE (TS-6).
- **Taxonomy hash algorithm:** `sha256(json.dumps({"land_cover": LANDCOVER_CLASSES, "topography": TOPO_CLASSES}, sort_keys=True).encode())`. Deterministic, simple, embedded in checkpoint metadata.
- **Splits.json structure:** `{"version": 1, "by_split": {"train": [<sample_ids>], "val": [...], "test": [...]}, "by_id_hash_bucket": {<sample_id>: <bucket_int_0_to_9>}}`. 80/10/10 train/val/test by hash bucket. Bootstrap (Phase 4) ADDS to train via the bucket policy; eval-time overlap assertion enforces no leakage.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project-level decisions and scope
- `.planning/PROJECT.md` — locked v1 scope, key decisions, out-of-scope list. Every Phase 1 plan task must check against PROJECT.md "Out of Scope (v1)" before adding scope.
- `.planning/REQUIREMENTS.md` — REQ-IDs and traceability. Phase 1 maps to MODEL-03 only.
- `.planning/ROADMAP.md` §"Phase 1: End-to-End Skeleton" — the 6 success criteria are the binding contract; this CONTEXT.md does NOT override them, only adds implementation choices for the same criteria.

### Architecture and stack (research)
- `.planning/research/SUMMARY.md` — executive synthesis; phase-shaping recommendations § "Foundation phase" maps directly to this phase.
- `.planning/research/ARCHITECTURE.md` — Pattern 1 (sample contract), Pattern 2 (loss weights table), Pattern 3 (Backbone ABC swap surface), Anti-Pattern 2 (no in-process bootstrap). The Backbone ABC contract for Mock is anchored here.
- `.planning/research/STACK.md` — concrete library versions. Phase 1's `pyproject.toml` should pin minimum versions per the table in this document.
- `.planning/research/FEATURES.md` § "Table-stakes TS-1..TS-11" — Phase 1 lands TS-1 (entrypoint+seeding), TS-2 (safetensors), TS-3 (infer API contract), TS-4 (probabilities), TS-5 (eval JSON), TS-8 (versioning in metadata), TS-10 (deterministic splits), TS-11 (output shape contract documented). TS-6 (calibration) is Phase-2+ extension.
- `.planning/research/PITFALLS.md` — five CRITICAL items relevant here:
  - PITFALL 2 (CRITICAL) — EVAL-03 protocol pre-registration is a Phase-1 deliverable.
  - PITFALL 3 (CRITICAL) — `LossWeights.load()` strict schema validator at startup.
  - PITFALL 4 (CRITICAL) — `Backbone` ABC owns `preprocess()` from day one; processor identity in checkpoint metadata.
  - PITFALL 5 (CRITICAL) — deterministic sample-id-hash splits, committed `splits.json`, eval-time overlap assertion.
  - PITFALL 15 (MEDIUM) — seeding utility.
  - PITFALL 19 (MEDIUM) — lockfile non-deferrable.

### Brownfield context (existing code)
- `.planning/codebase/STRUCTURE.md` — current `scripts/` layout. Phase 1 ADDS `mapclass/` alongside; `scripts/` stays.
- `.planning/codebase/CONVENTIONS.md` — observational style (4-space, ~110-char lines, modern type hints, no formatter/linter). Phase 1 inherits these conventions.
- `.planning/codebase/CONCERNS.md` items 8a (no seeds), 8b (no dataset versioning), 8c (no lockfile), TESTING.md gap-item 6 (loss-weight schema fragility) — Phase 1 directly mitigates all four.
- `scripts/biome_mapping.py` — `LANDCOVER_CLASSES`, `LANDCOVER_IDX`, `TOPO_CLASSES`, `TOPO_IDX` are the canonical taxonomy. Phase 1's taxonomy hash and `assert_sample_valid()` use these constants.
- `scripts/historical/label.py` — `HISTORICAL_LC_WEIGHTS`, `HISTORICAL_TOPO_WEIGHT` show the existing per-source loss weighting shape; Phase 1's `loss_weights.yaml` schema must accommodate these as one row (`source_subtype: rumsey_registered`).
- `scripts/label.py:24-25` — `WATER_TOPO=255` sentinel; Phase 1's validator and seg-head topography output must respect this.

### Project documents
- `mockup.md` — original academic framing. Phase 1 does NOT need to read this; safe to ignore for skeleton plumbing.
- `executive_TODO.md` — owner-facing licensing TODO. Phase 1 does NOT touch licensing — that's a milestone-2 / Phase-6 concern.
- `README.md` — implementation companion; documents the current `scripts/` flow. Phase 1's new entrypoints (`python -m mapclass.train|infer|eval`) get added to README at Phase 1's commit time.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/biome_mapping.py:LANDCOVER_CLASSES, LANDCOVER_IDX, TOPO_CLASSES, TOPO_IDX` — canonical taxonomy. `mapclass.data.taxonomy` should re-export these (do not duplicate). Taxonomy hash is computed from these constants.
- `scripts/label.py:WATER_TOPO=255, NODATA=255` — sentinels. `mapclass.data.contract` honors them in validation; `mapclass.model.seg_heads` topo head outputs `[3, H, W]` and the inference module overlays `WATER_TOPO=255` from the land-cover water class at predict time.
- `scripts/historical/label.py:HISTORICAL_LC_WEIGHTS, HISTORICAL_TOPO_WEIGHT` — existing per-source weighting pattern. The new `loss_weights.yaml` schema generalizes this: `source_subtype × class_name → weight`.
- `scripts/historical/__init__.py` exists; `scripts/__init__.py` does not. The historical sub-pipeline is a real Python package; the synthetic side imports by bare module name with `sys.path` mutation. Phase 1's `mapclass/` is a real package — will not have these issues.
- `requirements.txt` exists with `Pillow, numpy, rasterio, pyproj, requests, torch, transformers>=4.41, peft>=0.10, accelerate, bitsandbytes`. Phase 1's `pyproject.toml` SUPERSEDES this with stricter pins per `research/STACK.md` minimums (transformers >= 4.47, etc.). `requirements.txt` can stay as a developer convenience or be deleted — Phase 1 planner's call.

### Established Patterns
- **Snake_case throughout, leading underscore for module-private constants.** `mapclass/` follows the same conventions.
- **Modern type hints (PEP 604/585) on public surfaces, relaxed on private helpers.** Mock backbone class signature gets full annotations; internal helpers can be relaxed.
- **No formatter / linter configured.** Phase 1 does NOT introduce one (PROJECT.md: tests/CI/linting deferred from v1).
- **Try/except is sparse — only used in the historical pipeline (10 blocks, all in `scripts/historical/*`).** New `mapclass/` code follows the synthetic-side pattern: don't catch what you can't recover from; let exceptions propagate. `assert_sample_valid()` is the exception — it raises `SampleContractError` loudly.

### Integration Points
- **`mapclass.data.taxonomy`** re-exports `scripts.biome_mapping.LANDCOVER_CLASSES` etc. — do NOT copy-paste; do NOT redefine.
- **`mapclass.data.contract.assert_sample_valid(path)`** consumes the on-disk schema produced by `scripts/build_dataset.py` and `scripts/build_historical_dataset.py`. Validator fails loudly on schema drift — this is the brownfield-to-mapclass boundary.
- **`mapclass.train`** reads samples produced by the existing data pipelines. No imports across the boundary except via the validator. Phase 1 does NOT modify `scripts/build_dataset.py` or `scripts/build_historical_dataset.py`.
- **`mapclass.infer.load_model(<ckpt>)`** is the stable contract surfaced to the hex-grid app downstream. Phase 1 establishes the API; Phase 6 packages it for shipping.

</code_context>

<specifics>
## Specific Ideas

- **Mock as a learnable stub, not a fake** — the user explicitly chose the option that makes Phase 1 exercise the real training path (gradient flow, optimizer, scheduler), not just shape contracts. This shapes Phase 1 success: a Mock-trained run should produce a loss curve that visibly decreases (even if the absolute NLL is poor), confirming end-to-end gradient flow is intact. Phase 1 planner should add a Mock-NLL-decreases assertion to the smoke run.

- **Strict EVAL-03 fairness** — the user accepted that PaliGemma may need to train outside its pretrained operating envelope (224 vs 384, FP16 vs NF4), with the explicit caveat that Phase 5 must run a one-batch dry-run on the cloud VM and re-open this decision if it OOMs. The strict-fairness preference is a research-flavored choice (controlled experiment) — Phase 5 must surface dry-run results to the user before locking the PaliGemma config.

- **`DECIDE_AT_PHASE_5` literal sentinel** — the EVAL-03 protocol commits the placeholder string verbatim. Phase 5's `eval_03_compare.py` greps for this exact string and refuses to run if it's still present. Phase 5 plan must include "commit threshold replacement" as a separate, named task that runs before the compare script.

</specifics>

<deferred>
## Deferred Ideas

- **Multi-stage feature dict for Mock** — explicitly punted to Phase 2. When SmolVLM lands, the seg-heads consumer code will need to be re-shaped to handle `dict[str, Tensor]` with multiple stage names. Plan Phase 2 with this refactor scope explicit.
- **Calibration (TS-6 reliability diagram + ECE) in eval report** — Phase 2+ extension. Phase 1's `eval_report.json` schema is forward-compatible (adds optional `calibration` field).
- **Mock-checkpoint refusal at SHIP / infer load** — discipline-only choice in Phase 1. If a Mock checkpoint is accidentally shipped during v1.0 development, revisit in Phase 6 / milestone 2 (introduce `is_ship_eligible` flag + packaging-time guard).
- **Permitted Differences table for EVAL-03** — explicitly rejected in Phase 1 (strict fairness chosen). If Phase 5 dry-run reveals strict fairness is infeasible, Phase 5 may negotiate a Permitted Differences table at that time as a Phase 5 scope amendment.
- **Two-commit + git-history-mtime check for the deferred threshold** — rejected in favor of script-level guard only. If the threshold is set retroactively (after PaliGemma training) and goes undetected, this is a v1.x audit-tooling item, not a Phase 1 concern.

</deferred>

---

*Phase: 01-end-to-end-skeleton*
*Context gathered: 2026-05-08*
