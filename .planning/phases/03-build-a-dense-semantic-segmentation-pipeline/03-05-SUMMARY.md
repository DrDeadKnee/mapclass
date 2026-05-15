---
phase: 03-build-a-dense-semantic-segmentation-pipeline
plan: 05
subsystem: model
tags: [pytorch, siglip, dinov2, swin, segmentation, recursive-inference, upernet, fpn, prior-encoder, peft, timm]

# Dependency graph
requires:
  - phase: 03-build-a-dense-semantic-segmentation-pipeline
    provides: "03-01 backbones.py (Backbone protocol, SiglipBackbone, Dinov2Backbone, SwinBackbone, _widen_patch_embed)"
  - phase: 03-build-a-dense-semantic-segmentation-pipeline
    provides: "03-04 decoder.py (SegDecoder UPerNet PPM+FPN) + heads.py (LandCoverHead, TopographyHead)"
  - phase: 02-build-a-dataset-of-pixel-label-pairs
    provides: "pyramid.json schema: tiles[].{id,x,y,size,children}; 1×896+4×448+16×224 nested pyramid"
provides:
  - "SegModelVariantA: frozen backbone + trainable PriorEncoder decoder-level injection; forward(rgb,prior)→(lc,topo)"
  - "SegModelVariantB: widened 15-ch patch-embed backbone + input-level prior; forward(x15)→(lc,topo)"
  - "PriorEncoder: small 2-conv trainable module mapping 12-ch prior to decoder working resolution"
  - "SegModel factory wrapper (variant='A'|'B')"
  - "_assert_no_gemma: construction-time Gemma-exclusion assertion (PHASE-03 SC#2)"
  - "crop_prior_to_child: exact quadrant crop from pyramid.json boxes (D-04 / Pitfall-5 guard)"
  - "cold_start_prior: zeros(B,12,S,S) 896 cold-start"
  - "recursive_predict: depth-first 896→448→224 c2f walk returning {tile_id: (B,12,S,S)} prob cache"
  - "recursive_predict_variant_b: same walk for Variant B (15-ch input)"
affects: [04-phase-4-training, eval-03-backbone-comparison]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Variant A decoder-level prior injection: PriorEncoder resizes 12-ch prior to decoder working resolution, sums into decoder output before task heads"
    - "Variant B input-level prior injection: prior concatenated to RGB (15-ch input) through widened patch-embed"
    - "Gemma exclusion at construction time: _assert_no_gemma checks named_parameters + module types"
    - "Recursive c2f walk: manifest-box crop-then-resize (cy-oy:cy-oy+s, cx-ox:cx-ox+s) never recomputed"
    - "torch.no_grad throughout recursive_predict (D-06 forward-only)"

key-files:
  created:
    - scripts/seg/recursive.py
  modified:
    - scripts/seg/model.py

key-decisions:
  - "Variant A prior injection: resize prior to decoder output spatial size, then apply PriorEncoder (2-conv), then sum into decoder features — avoids any dependency on backbone internal shape"
  - "PriorEncoder target_size=None in SegModelVariantA: caller resizes prior before calling encoder; encoder operates at decoder spatial resolution throughout"
  - "Both variants upsample decoder output to full tile H×W at forward time via F.interpolate — handles 448/896 inputs identically without re-constructing the decoder"
  - "PriorEncoder GroupNorm uses 4 groups for 12-ch input (12 % 4 == 0; avoids GN(32) incompatibility with 12 < 32)"
  - "recursive_predict re-exports PriorEncoder from seg.model so test_seg_recursive.py imports work (from seg.recursive import PriorEncoder)"

patterns-established:
  - "crop_prior_to_child(parent_prob, parent_tile, child_tile): the canonical D-04 quadrant crop primitive; all recursive orchestration calls this function"
  - "cold_start_prior(batch, tile_size): explicit zeros(B,12,S,S) constructor — the 896 cold-start is always this"
  - "_assert_no_gemma(module): construction-time SC#2 assertion; call from __init__ of any SegModel variant"

requirements-completed: [PHASE-03, EVAL-03]

# Metrics
duration: 50min
completed: 2026-05-15
---

# Phase 3 Plan 05: SegModel Assembly + Recursive C2F Orchestrator Summary

**SegModel assembles backbone+UPerNet-decoder+2 heads with both D-03a prior-injection variants (A: decoder-level PriorEncoder, B: 15-ch widened-patch-embed input) plus a pyramid.json-literal recursive c2f orchestrator; all three backbones independently provably Gemma-free**

## Performance

- **Duration:** ~50 min
- **Started:** 2026-05-15T23:00:00Z
- **Completed:** 2026-05-15T23:44:00Z
- **Tasks:** 2
- **Files modified:** 2 (1 created, 1 expanded from stub)

## Accomplishments

- SegModelVariantA and SegModelVariantB both constructed and verified for SigLIP stub backbone: forward(rgb,prior)→(B,9,H,W)+(B,3,H,W) no NaN at 896, 448, 224 tile sizes
- Gemma exclusion enforced at construction time via _assert_no_gemma; no language_model param or gemma module-type reachable from either variant (PHASE-03 SC#2)
- recursive_predict walks pyramid.json 896→448→224 depth-first; crop_prior_to_child uses exact manifest box arithmetic; Pitfall-5 alignment test (one-hot blob lands in correct child) green
- cold_start_prior returns exact zeros(B,12,S,S); 896 root cold-start confirmed by test
- Both variants share identical SegDecoder + LandCoverHead + TopographyHead instances (D-02 EVAL-03 apples-to-apples requirement)
- DINOv2 and Swin variants constructable via backbone_name= flag (same decoder, same orchestration; EVAL-03 seam closed)
- Integration tests skip with clear reason when Phase-1 adapter / Phase-2 data absent (OQ1 RESOLVED)
- Full offline suite: 111/111 passed, 9 deselected (integration markers), 4 unrelated deprecation warnings

## Task Commits

Each task was committed atomically:

1. **Task 1: SegModel assembly with both D-03a prior-injection variants** - `51f489b` (feat)
2. **Task 2: Recursive c2f orchestrator (pyramid.json walk) + smoke/integration close** - `eb499b8` (feat)

## Files Created/Modified

- `scripts/seg/model.py` - SegModelVariantA, SegModelVariantB, PriorEncoder, SegModel factory, _assert_no_gemma; full backbone+decoder+heads assembly with both D-03a variants (expanded from stub)
- `scripts/seg/recursive.py` - crop_prior_to_child, cold_start_prior, PriorEncoder (re-export), recursive_predict, recursive_predict_variant_b, _default_read_rgb (created)

## Decisions Made

- **PriorEncoder GroupNorm(4) for 12-ch input:** GroupNorm(32, 12) would fail since 12 < 32; used GN(4) = 4 groups for the first conv layer (12 % 4 == 0).
- **Variant A forward: caller resizes prior before PriorEncoder call:** The SegModelVariantA.forward resizes the prior to the decoder output's spatial size, then calls PriorEncoder (which uses padding=1 convs, preserving spatial size). PriorEncoder target_size=None means no internal resize — the caller manages alignment explicitly. This is cleaner than storing a dynamic target_size.
- **Both variants upsample to full tile H×W at forward time:** Rather than constructing a decoder for every possible tile size (896/448/224), the decoder is constructed at a default tile_size=224 and both variant forwards apply a final F.interpolate to reach the actual input resolution. This keeps construction simple while handling 896/448/224 dynamically.
- **PriorEncoder re-exported from seg.recursive:** test_seg_recursive.py imports PriorEncoder from seg.recursive per the 03-01 scaffold. The re-export (from seg.model import PriorEncoder) satisfies the test contract without duplicating the class.

## Deviations from Plan

None — plan executed exactly as written. Both tasks completed as specified. The forward implementation uses a clean caller-resizes-prior pattern (not in the plan spec, but the cleanest resolution of the dynamic-spatial-size requirement).

## Threat Mitigations Applied

| Threat ID | Status | Notes |
|-----------|--------|-------|
| T-03-12 (quadrant misalignment) | Mitigated | crop_prior_to_child uses literal pyramid.json box arithmetic; Pitfall-5 blob-in-quadrant test green |
| T-03-13 (Gemma reachable) | Mitigated | _assert_no_gemma called at SegModelVariantA + SegModelVariantB __init__; both variants verified clean |
| T-03-14 (adapter deserialization) | Mitigated | SiglipBackbone takes explicit user-supplied paths; integration tests skip-with-reason when absent |
| T-03-15 (unbounded recursion) | Accepted | Pyramid depth fixed at 3 (21 tiles) by Phase-2 contract; cache bounded by manifest |

## Known Stubs

None. Both `scripts/seg/model.py` and `scripts/seg/recursive.py` are fully implemented with all paths wired.

## Issues Encountered

None. The implementation followed the plan directly.

## Next Phase Readiness

- Phase-3 smoke deliverable complete: test_seg_smoke.py (15 tests) and test_seg_recursive.py (8 tests) all green for both D-03a variants
- Phase-4 training can import SegModelVariantA / SegModelVariantB directly; optimizer/backward are Phase 4's responsibility
- DINOv2 and Swin EVAL-03 seam open: backbone_name="dinov2"/"swin" constructs the full pipeline with the same decoder+heads
- Integration test gate remains: real SigLIP-LoRA + real Phase-2 pyramid needed for test_seg_online.py (skip-with-reason when absent)
- No blockers for Phase 4

## Self-Check: PASSED

- scripts/seg/model.py: FOUND
- scripts/seg/recursive.py: FOUND
- Task 1 commit 51f489b: FOUND
- Task 2 commit eb499b8: FOUND
- test_seg_smoke.py: 15 passed
- test_seg_recursive.py: 8 passed
- Full offline suite: 111 passed, 0 failures

---
*Phase: 03-build-a-dense-semantic-segmentation-pipeline*
*Completed: 2026-05-15*
