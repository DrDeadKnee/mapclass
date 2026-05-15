# Phase 3: Build a dense semantic segmentation pipeline - Context

**Gathered:** 2026-05-15
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 3 **constructs** (does not train) a dense pixel-level segmentation model:
the Phase-1 LoRA-adapted SigLIP encoder (primary, locked backbone) → a shared
lightweight conv decoder → two thin task heads (9-class land cover, 3-class
topography), plus recursive coarse-to-fine inference machinery and swappable
DINOv2/Swin benchmark backbones. Deliverable is the assembled model + inference
pipeline + a dataloader over the Phase-2 nested-pyramid dataset + a forward/shape
smoke-test on real tiles. **All training, fine-tuning, and the joint-NLL
evaluation are Phase 4.** Requirements: PHASE-03, EVAL-03.

</domain>

<decisions>
## Implementation Decisions

### Decoder & head architecture
- **D-01: Lightweight conv decoder.** A small FPN/UPerNet-style convolutional
  decoder on SigLIP patch-token features (not a linear-probe upsample, not
  DPT-reassemble). Chosen for crisp terrain-boundary quality (coastlines/ranges)
  at modest params with fast Phase-4 training on the frozen encoder. The decoder
  may internally fuse multi-scale backbone features.
- **D-02: Shared decoder trunk + two thin task-specific output layers.** One
  decoder feeds two lightweight heads (9-class land cover, 3-class topography).
  Rationale: exploits the land-cover↔topography correlation the joint-NLL metric
  rewards (water=flat, cropland≠mountainous), fewer params, consistent features.
  NOT two fully independent decoders.

### Coarse-to-fine inference mechanics
- **D-03: Explicit recursive prior.** The coarse 896-scale predicted
  class-probability maps are fed as **extra input channels** into the finer
  448→224 predictions. This implements the project DOC's geophysical-prior
  thesis (predicted distribution as prior) and directly consumes Phase-2's
  nested pyramid (D-07). It is a 2-pass recursive pipeline, not single-pass
  feature fusion. The conv decoder (D-01) may ALSO do internal feature fusion;
  the load-bearing requirement is the coarse→fine predicted-probability feedback.
- **D-04: Inference walks Phase-2 `pyramid.json` parent→child literally.**
  Coarse-to-fine traversal follows the stored 896→4×448→16×224 parent→child
  indices that Phase-2 D-07/D-08 purpose-built. No independent
  sliding-window/resampling reconstruction at inference for Phase-2 inputs —
  this keeps train/infer alignment exact and avoids re-deriving topology.

### Backbone-swap interface
- **D-05: One unified `Backbone` protocol.** Single interface:
  `forward(image) -> feature maps at declared strides`. SigLIP (primary) and
  DINOv2 are columnar ViTs (multi-scale via sliding window over the pyramid);
  Swin is natively hierarchical — each implements the same protocol. The SAME
  conv decoder, recursive-prior wiring, and eval harness sit on top of all
  three, so Phase-4's EVAL-03 comparison is apples-to-apples (differences
  attributable to the backbone, not per-backbone adapter artifacts).

### Phase 3 / Phase 4 scope line
- **D-06: Phase 3 = model + inference construction only.** Deliverable:
  assembled model (backbone + shared conv decoder + 2 heads), recursive
  coarse-to-fine inference, all three backbone variants behind D-05, a
  dataloader over the Phase-2 pyramid dataset (`pyramid.json`,
  image/land_cover/topography/sample_weights, frozen `split.json`), and a
  forward-pass / output-shape smoke-test on real Phase-2 tiles. **Backbone
  frozen by default.** Phase 4 owns ALL training/fine-tuning and the joint
  per-pixel NLL evaluation. No training loop, no overfit-a-batch run, no
  unfreeze hooks in Phase 3.

### Claude's Discretion
- Exact conv-decoder topology (channel widths, number of upsample stages, which
  SigLIP block(s) to tap, norm/activation choices) — planner/researcher's call
  within D-01.
- How the coarse-prior channels are encoded (raw softmax probs vs logits vs
  argmax one-hot; resize/alignment method onto the finer grid) — planner's call
  within D-03, surfaced if research finds a strong tradeoff.
- DINOv2/Swin checkpoint sources and exact stride declarations behind the D-05
  protocol.
- Dataloader internals (batching strategy, pyramid-aware sampling, sample-weight
  plumbing) — implementation detail under D-06.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Locked constraints & requirements
- `.planning/PROJECT.md` — locked SPEC constraints: SigLIP-primary backbone;
  Gemma NEVER in the segmentation forward pass; LoRA only on SigLIP
  `q_proj`/`k_proj`/`v_proj`/`out_proj`; DINOv2/Swin non-locked benchmark
  alternatives; large-first-kernel ConvNet from scratch is a stretch goal only.
- `.planning/REQUIREMENTS.md` — PHASE-03 (dense seg pipeline on Phase-1 SigLIP +
  2 heads + coarse-to-fine; DINOv2/Swin baselines available) and EVAL-03 (DINOv2
  + Swin evaluated on the same metric for direct comparison vs the SigLIP path).
- `.planning/ROADMAP.md` §"Phase 3" — goal + the 4 success criteria.

### Phase-1 backbone (reused weights)
- `scripts/finetune_paligemma.py` — how the Phase-1 PEFT adapter is produced
  and saved (PEFT adapter directory; LoRA on SigLIP attention; Gemma frozen).
  Phase 3's SigLIP backbone path loads these adapter weights; no Gemma weights
  in the forward pass.

### Phase-2 dataset contract (decoder/dataloader sizing)
- `.planning/phases/02-build-a-dataset-of-pixel-label-pairs/02-CONTEXT.md`
  §decisions — D-06..D-09 (pre-tiled nested pyramids 1×896 + 4×448 + 16×224,
  stride-448, edge policy) and D-15..D-18 (frozen seeded stratified
  `data/synthetic/split.json`, `train/` vs `test/` filesystem separation).
- `scripts/tiling.py` — the on-disk per-pyramid contract: `pyramid.json`
  (tile paths + parent→child indices), per-map `image.png` / `land_cover.png` /
  `topography.png` / `sample_weights.json`. This is the exact structure D-04's
  inference traversal and D-06's dataloader consume.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/finetune_paligemma.py::apply_lora` — established pattern for loading
  PaliGemma and isolating the SigLIP vision encoder with LoRA on
  `q_proj/k_proj/v_proj/out_proj` while freezing Gemma. Phase 3's SigLIP
  backbone adapter should mirror this isolation so the segmentation forward
  pass provably never touches Gemma (PHASE-03 success criterion #2).
- `scripts/tiling.py` (`_REQUIRED_FILES`, `pyramid.json` schema, parent→child
  indices) — directly consumed by the D-04 recursive inference walk and the
  D-06 dataloader. No new tile materialisation needed; Phase 3 reads.

### Established Patterns
- Phase-2 frozen split (`data/synthetic/split.json`, sibling `train/`/`test/`
  dirs): the Phase-3 dataloader MUST point only at `train/` for any
  Phase-4-facing use and never enumerate `test/` — preserves EVAL-01's
  zero-leakage guarantee end-to-end.
- Per-source `sample_weights.json` (locked dict shape across historical
  uniform-trust, synthetic uniform-1.0, satellite balance-tilt) — the
  dataloader/loss plumbing in Phase 4 will consume these; Phase 3 must surface
  them through the dataloader unchanged.

### Integration Points
- SigLIP backbone ← Phase-1 PEFT adapter directory (load path/contract from
  `finetune_paligemma.py`).
- Dataloader ← Phase-2 pyramid directories + `pyramid.json` + `split.json`.
- Unified `Backbone` protocol (D-05) is the seam Phase-4 EVAL-03 swaps
  SigLIP/DINOv2/Swin behind.

</code_context>

<specifics>
## Specific Ideas

- Decoder must prioritise **boundary sharpness** for terrain edges
  (coastlines, mountain ranges, major water bodies) — this drove the conv-decoder
  choice over linear-probe.
- The recursive-prior (D-03) is treated as the project's load-bearing
  architectural bet, not an optional nicety — Phase-2's pyramid was built
  specifically to feed it (D-07 explicitly: "exploitable by Phase 3's
  coarse-to-fine model").

</specifics>

<deferred>
## Deferred Ideas

- **OCR / place-name & text reading off maps** — raised during discussion as a
  candidate area. NOT in Phase 3 scope (Phase 3 is dense terrain segmentation;
  the ROADMAP goal and success criteria contain no text/OCR capability). It is
  closest in spirit to the already-deferred **GEOREF-V2** (v2: PaliGemma-driven
  semi-automatic registration / place-name-aware georeferencing — see
  `.planning/REQUIREMENTS.md` §Deferred GEOREF-V2 and STATE.md Deferred Items).
  Captured here so it is not lost; belongs to a future v2 milestone, not Phase 3
  or Phase 4. No action this phase.

</deferred>

---

*Phase: 3-build-a-dense-semantic-segmentation-pipeline*
*Context gathered: 2026-05-15*
