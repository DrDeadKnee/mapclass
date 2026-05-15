# Phase 3: Build a dense semantic segmentation pipeline - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-15
**Phase:** 3-build-a-dense-semantic-segmentation-pipeline
**Areas discussed:** Decoder & head architecture, Coarse-to-fine mechanics, Backbone-swap interface, Phase 3/Phase 4 scope boundary

---

## Decoder architecture

| Option | Description | Selected |
|--------|-------------|----------|
| Lightweight conv decoder | FPN/UPerNet-style conv decoder on SigLIP patch features; crisp boundaries, modest params, fast Phase-4 training | ✓ |
| Linear-probe upsample | Linear + bilinear upsample; near-zero params but blocky/blurry boundaries | |
| DPT-style reassemble | Multi-depth reassemble + fusion; best detail, heaviest; overkill for 9+3 classes on frozen encoder | |

**User's choice:** Lightweight conv decoder (after a requested refresher on SigLIP/ViT patch tokens and the decoder problem).
**Notes:** User initially asked for a SigLIP refresher and more detail before deciding; explanation given (ViT patch grid → dense logits problem, three decoder tiers), then confirmed the recommended conv decoder.

## Head architecture

| Option | Description | Selected |
|--------|-------------|----------|
| Shared decoder trunk + 2 thin task layers | Exploits land-cover↔topography correlation joint-NLL rewards; fewer params | ✓ |
| Fully independent decoders | Max task isolation, ~2× decoder params, no cross-task feature exploitation | |

**User's choice:** Shared decoder, 2 head layers (Recommended).
**Notes:** Decided in the first batch without needing a refresher.

## Coarse-to-fine mechanics

| Option | Description | Selected |
|--------|-------------|----------|
| Explicit recursive prior | Coarse 896 predicted class-probs fed as extra input channels at finer scale; matches DOC thesis + Phase-2 D-07; 2-pass | ✓ |
| Feature-pyramid fusion only | Decoder merges backbone multi-scale features, single pass; drops recursive-prior bet | |
| Both: fusion decoder + coarse-prior channel | Most faithful; slightly more wiring | (folded into D-01/D-03 — decoder may fuse internally AND take the prior channel) |

**User's choice:** Explicit recursive prior (after requesting more detail on feature-pyramid fusion vs the recursive prior).
**Notes:** User asked for a clearer explanation of feature-pyramid fusion before deciding; tradeoff explained (prediction-prior feedback vs generic backbone-feature merge). Confirmed the explicit recursive prior as the project's load-bearing bet; CONTEXT D-01 permits the conv decoder to also fuse features internally.

## Inference: Phase-2 pyramid use

| Option | Description | Selected |
|--------|-------------|----------|
| Use pyramid.json parent→child literally | Walk stored 896→4×448→16×224 indices D-07 was built for; tight train/infer alignment | ✓ |
| Independent multi-res sliding window | Reconstruct windows at inference; flexible for non-Phase-2 maps but re-derives topology, risks mismatch | |

**User's choice:** Use pyramid.json parent→child literally (Recommended).

## Backbone-swap interface

| Option | Description | Selected |
|--------|-------------|----------|
| Unified Backbone protocol | One `forward → feature maps at declared strides` protocol; same decoder/prior/eval on all 3; clean EVAL-03 comparison | ✓ |
| Per-backbone adapters → common head | Per-backbone freedom but fuzzier comparison (adapter artifacts) | |

**User's choice:** Unified Backbone protocol (Recommended).

## Phase 3 / Phase 4 scope line

| Option | Description | Selected |
|--------|-------------|----------|
| Model + inference only; Phase 4 trains | Assembled model + recursive c2f inference + 3 backbones + dataloader + forward/shape smoke-test; backbone frozen; Phase 4 owns all training + eval | ✓ |
| Include a smoke-train loop | Above + minimal training loop + overfit-a-batch sanity in Phase 3 | |
| Expose unfreeze hooks now | Model-only but build v2-deferred progressive-unfreeze seam early | |

**User's choice:** Model + inference only; Phase 4 trains (Recommended).
**Notes:** Keeps progressive-backbone-unfreeze v2-deferred (consistent with STATE.md Deferred Items); no training scaffolding pulled into Phase 3.

---

## Claude's Discretion

- Exact conv-decoder topology (channel widths, upsample stages, which SigLIP block(s) to tap, norm/activation).
- Coarse-prior channel encoding (softmax probs vs logits vs one-hot; resize/alignment onto finer grid).
- DINOv2/Swin checkpoint sources and stride declarations behind the unified protocol.
- Dataloader internals (batching, pyramid-aware sampling, sample-weight plumbing).

## Deferred Ideas

- **OCR / place-name & text reading off maps** — raised as a candidate discussion area; ruled out as Phase 3 scope creep (Phase 3 is dense terrain segmentation, no text/OCR in the ROADMAP goal/criteria). Captured in CONTEXT.md `<deferred>`; closest to the already-deferred GEOREF-V2 v2 path. No Phase 3 / Phase 4 action.
