---
phase: 03-build-a-dense-semantic-segmentation-pipeline
plan: "04"
subsystem: seg-decoder
tags: [pytorch, upernet, fpn, ppm, decoder, task-heads, dense-segmentation]
dependency_graph:
  requires: [03-01, 03-02]
  provides: [scripts/seg/decoder.py, scripts/seg/heads.py]
  affects: [scripts/seg/model.py, tests/test_seg_decoder.py]
tech_stack:
  added: []
  patterns:
    - UPerNet-style PPM+FPN conv decoder on declared-stride backbone feature lists
    - Progressive 2× bilinear+3×3 upsample (boundary-sharpness over single large bilinear)
    - GroupNorm(32) throughout (batch-size-agnostic for B=1 inference)
    - Thin 1×1-conv task heads — no second decoder, no softmax in head
key_files:
  created:
    - scripts/seg/decoder.py
    - scripts/seg/heads.py
  modified: []
decisions:
  - "PPM bins (1,2,3,6) — standard UPerNet setting; decoder working width 256"
  - "FPN fusion at common working resolution = tile_size // min(strides), then progressive 2× upsample"
  - "All lateral convs followed by F.interpolate to working resolution before FPN add (Pitfall 4 guard)"
  - "Sum-fuse all smoothed FPN levels (better multi-scale coverage vs using only finest)"
metrics:
  duration: "~20 minutes (active execution; session elapsed differs)"
  completed: "2026-05-15"
  tasks_completed: 2
  tasks_total: 2
  files_created: 2
  files_modified: 0
---

# Phase 03 Plan 04: Shared UPerNet PPM+FPN Conv Decoder + Two Thin Task Heads Summary

**One-liner:** UPerNet PPM+FPN decoder with dynamic stride/channel reading (ViT/Swin-agnostic) + thin 1×1-conv LC (9-class) and topo (3-class) heads emitting raw logits at full tile resolution.

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Shared UPerNet PPM+FPN conv decoder (D-01) | d47ddf1 | scripts/seg/decoder.py |
| 2 | Two thin task heads LC+topo (D-02) | c73cfb4 | scripts/seg/heads.py |

## What Was Built

### Task 1: `scripts/seg/decoder.py` — `SegDecoder`

A `SegDecoder(feature_strides, feature_channels, tile_size)` class implementing D-01:

- **PPM on the deepest level** (bins 1/2/3/6): `_PPMBlock` applies adaptive average pooling at each bin, projects to `decoder_channels // 4` per bin, upsamples back, and concatenates with the original feature before a 1×1 fusion — adds global context critical for terrain coherence.
- **Lateral 1×1 convs** per level: all inputs projected to 256 channels (GroupNorm+ReLU); deepest level uses the PPM output.
- **Common working resolution**: every level F.interpolated to `tile_size // min(feature_strides)` before FPN add — the Pitfall 4 guard that makes ViT equal-stride lists and Swin's strides 4/8/16/32 work identically with zero code change.
- **FPN top-down fusion**: accumulates from deepest to shallowest; each level gets a 3×3 smoothing conv; all smoothed levels sum-fused.
- **Progressive upsample**: `ceil(log2(tile_size / working_h))` stages of 2× bilinear + 3×3 conv, with a final exact-size interpolate to handle non-power-of-2 ratios — no single large bilinear blow-up.
- Exposes `out_channels` (256) for head construction.

### Task 2: `scripts/seg/heads.py` — `LandCoverHead` + `TopographyHead`

Two thin task heads (D-02):

- `LandCoverHead(in_channels)`: single `Conv2d(in_channels, 9, 1)` → (B,9,H,W) raw logits.
- `TopographyHead(in_channels)`: single `Conv2d(in_channels, 3, 1)` → (B,3,H,W) raw logits.
- No softmax; spatial resolution unchanged (decoder already at tile_size).
- Both share the single decoder trunk — no second decoder.

## Verification

```
python -m pytest tests/test_seg_decoder.py -x -q -m "not integration"
# 9 passed in 38.44s

python -m pytest -q -m "not integration"
# 96 passed (15 pre-existing failures in test_seg_recursive + test_seg_smoke
#  from seg.recursive/seg.model not yet implemented — plan 03-05/06)
```

Manual grep: no `optim`/`.backward(`/linear-probe/DPT-reassemble/softmax in `decoder.py` or `heads.py`.

## Decisions Made

- **Decoder working width**: 256 channels (modest params, sane for research pipeline — D-01 Claude's Discretion).
- **PPM bins**: (1, 2, 3, 6) — standard UPerNet default; adds global context at 4 scales.
- **FPN working resolution**: `tile_size // min(feature_strides)` — finest declared level drives the common resolution; each other level bilinearly interpolated before add.
- **FPN level fusion**: sum all smoothed FPN levels (vs. use only finest) — ensures all scale information contributes to the shared trunk.
- **Progressive upsample**: staged 2× bilinear + 3×3 conv instead of single large-factor; final exact-size clamp handles non-power-of-2.
- **GroupNorm(32)**: batch-size-agnostic for B=1 inference at tile boundaries.

## Deviations from Plan

None — plan executed exactly as written.

Both TDD gates satisfied: test file was RED (ImportError) before implementation; GREEN (9/9 pass) after both decoder.py and heads.py landed. The test file's import guard couples both modules at import time, so Task 1 and Task 2 share the same GREEN gate (by design in the pre-written test file from plan 03-01).

## Known Stubs

None. All decoder and head logic is fully wired. The decoder output feeds the heads correctly at full tile resolution for all three tested tile sizes (224, 448, 896) and both backbone geometries (ViT 1-level and Swin 4-level).

## Threat Flags

None. No new network endpoints, auth paths, file access patterns, or schema changes. The decoder/heads are pure in-process tensor transforms. T-03-10 (hard-coded level count vs Swin/ViT geometry) is mitigated: decoder reads `feature_strides`/`feature_channels` dynamically and uses `F.interpolate` to a common resolution before fusion.

## Self-Check: PASSED

| Check | Result |
|-------|--------|
| `scripts/seg/decoder.py` exists | FOUND |
| `scripts/seg/heads.py` exists | FOUND |
| commit d47ddf1 (decoder) exists | FOUND |
| commit c73cfb4 (heads) exists | FOUND |
| `pytest tests/test_seg_decoder.py -x -q -m "not integration"` | 9 passed |
