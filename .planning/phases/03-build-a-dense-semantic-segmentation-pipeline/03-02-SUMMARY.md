---
phase: 03-build-a-dense-semantic-segmentation-pipeline
plan: "02"
subsystem: seg.backbones
tags: [backbone, siglip, dinov2, swin, peft, gemma-exclusion, variant-b, patch-embed, d05, d06a]

dependency_graph:
  requires:
    - scripts/finetune_paligemma.py   # apply_lora load-order mirror
    - tests/test_seg_backbones.py     # TDD tests from 03-01
    - scripts/seg/__init__.py         # package marker from 03-01
  provides:
    - scripts/seg/backbones.py        # Backbone Protocol + 3 implementations
    - scripts/seg/model.py            # SegModelVariantA / SegModelVariantB stubs
  affects:
    - scripts/seg/decoder.py          # reads feature_strides / feature_channels (03-04)
    - scripts/seg/model.py            # SegModel assembly wiring (03-04)

tech_stack:
  added:
    - timm==1.0.27 (DINOv2 + Swin backbones via get_intermediate_layers / features_only)
    - torch.random.fork_rng (deterministic stub init for offline RGB-preservation test)
  patterns:
    - typing.Protocol (runtime_checkable Backbone D-05 seam)
    - _StubVisionTower (offline test double — no HF checkpoint required)
    - _widen_patch_embed (Conv2d surgery: copy RGB, zero-init extras, requires_grad_(True))
    - Swin (B,H,W,C) → permute(0,3,1,2) → (B,C,H,W) before returning

key_files:
  created:
    - scripts/seg/backbones.py
    - scripts/seg/model.py
  modified: []

decisions:
  - "DINOv2 model tag: vit_small_patch14_dinov2 with dynamic_img_size=True (handles 224/448/896)"
  - "Swin model tag: swin_base_patch4_window7_224 (patch4, features_only, 4 levels strides [4,8,16,32])"
  - "SigLIP tap layers: (depth//3, 2*depth//3, depth-1) → (8,17,26) for 27-layer SigLIP as documented"
  - "_StubPatchEmbed uses fixed seed derived from (hidden_size, patch_size, in_channels) for deterministic RGB-preservation test"
  - "seg.model stub: SegModelVariantA/B created now (plan 03-02) as import gate for test_no_gemma; full implementation in 03-04"

metrics:
  duration: "~25 minutes (2026-05-15T23:14Z)"
  completed: "2026-05-15"
  tasks_completed: 2
  tasks_total: 2
  files_created: 2
  files_modified: 0
---

# Phase 03 Plan 02: Backbone Protocol + SigLIP/DINOv2/Swin + Variant-B patch-embed Summary

Backbone Protocol (D-05) implemented with SiglipBackbone (PEFT LoRA load, Gemma-exclusion, ViT reshape), Dinov2Backbone (timm), and SwinBackbone (timm features_only), plus Variant-B 15-channel widened patch-embed for all three (D-03a / D-06a construction-only).

## Tasks

| # | Name | Status | Commit |
|---|------|--------|--------|
| 1 | Backbone Protocol + SiglipBackbone (PEFT load, Gemma-exclusion, ViT reshape) | Done | 0652623 |
| 2 | DINOv2 + Swin timm backbones + Variant-B widened 15-ch patch-embed | Done | 0652623 |

Both tasks were implemented in a single pass (same output file) and committed atomically.

## What Was Built

### `scripts/seg/backbones.py`

**`Backbone` Protocol (D-05):** `runtime_checkable` typing Protocol with `feature_strides: list[int]`, `feature_channels: list[int]`, and `forward(x) -> list[Tensor]`. The shared decoder (03-04) reads these attributes dynamically.

**`SiglipBackbone`:** mirrors `finetune_paligemma.py::apply_lora` load order exactly:
1. `PaliGemmaForConditionalGeneration.from_pretrained(model_id)`
2. `PeftModel.from_pretrained(base, adapter_dir)` — binds Phase-1 LoRA
3. `.eval().requires_grad_(False)` — D-06 frozen
4. Defensive accessor: `getattr(pg, "vision_tower", None) or pg.model.vision_tower` (transformers 4.x/5.x Pitfall 3)
5. Asserts resolved module is a SigLIP type (T-03-05)
6. Holds `_vision_tower` only — no projector or `language_model` reachable (PHASE-03 SC#2 / T-03-06)

Offline mode via `stub_config`: `_StubVisionTower` with `_StubPatchEmbed` using deterministic seed per (hidden_size, patch_size, in_channels) so RGB-preservation tests work across two independent instances.

Tap layers: `(depth//3, 2*depth//3, depth-1)` → `(8, 17, 26)` for 27-layer SigLIP as documented.

**`Dinov2Backbone`:** `timm.create_model("vit_small_patch14_dinov2", ..., dynamic_img_size=True)` + `get_intermediate_layers(x, n=k, reshape=True)` → list of (B, C, h, w). `dynamic_img_size=True` handles 224/448/896 without reloading. Declared stride 14; channels from `embed_dim`.

**`SwinBackbone`:** `timm.create_model("swin_base_patch4_window7_224", features_only=True, out_indices=(0,1,2,3))`. Strides/channels declared from `feature_info.reduction()` / `feature_info.channels()` (never hard-coded — Pitfall 4). Swin returns (B,H,W,C) per level; permuted to (B,C,H,W) before return.

**`_widen_patch_embed`:** constructs a new `nn.Conv2d` with in_channels+12, copies pretrained RGB weights into channels 0:orig_in verbatim, zero-inits channels orig_in:new_in, marks `requires_grad_(True)`. Called for all three backbones when `variant_b=True`.

**`patch_embed_weight()`:** accessor on all three backbone classes for test introspection.

### `scripts/seg/model.py`

Stub `SegModelVariantA` and `SegModelVariantB` classes with `__init__(self, backbone)` — holds backbone as submodule so `named_parameters()` traversal reaches it for the `test_no_gemma_variant_a` import gate. Full implementation lands in plan 03-04.

## Verification

```
pytest tests/test_seg_backbones.py -x -q -m "not integration"  → 15 passed
pytest tests/test_seg_backbones.py -k "siglip or no_gemma or protocol" -x -q → 12 passed
pytest tests/test_seg_backbones.py -k "dino or swin or variant_b or protocol" -x -q → 9 passed
```

Manual grep: no `optim`, `.backward(`, or training loop in `scripts/seg/backbones.py` (D-06).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Deterministic stub weight initialization for RGB-preservation test**
- **Found during:** Task 1 (first test run)
- **Issue:** `test_variant_b_siglip_rgb_weights_preserved` creates two separate `SiglipBackbone` instances from the same `stub_config` and expects their patch-embed weights (channels 0:3) to be equal. `nn.Conv2d` initializes with random weights per instance — two instances got different random weights.
- **Fix:** `_StubPatchEmbed.__init__` wraps `nn.Conv2d` construction in `torch.random.fork_rng()` with `torch.manual_seed(hidden_size * 1000 + patch_size * 100 + in_channels)`. Any two stubs with the same config shape get identical pretrained-like weights; the global RNG is unaffected.
- **Files modified:** `scripts/seg/backbones.py`
- **Commit:** 0652623

**2. [Rule 2 - Missing] stub `seg.model` module for test_no_gemma_variant_a import gate**
- **Found during:** Task 1 (test analysis)
- **Issue:** `test_no_gemma_variant_a` imports `from seg.model import SegModelVariantA` as a correctness gate (confirming the model module exists and the backbone is the only seg parameter path). `seg.model` did not exist yet (planned for 03-04).
- **Fix:** Created `scripts/seg/model.py` with stub `SegModelVariantA` and `SegModelVariantB` holding the backbone as a submodule. Full implementation deferred to 03-04.
- **Files modified:** `scripts/seg/model.py` (new)
- **Commit:** 0652623

**3. [Rule 1 - Bug] Swin output permutation (B,H,W,C) → (B,C,H,W)**
- **Found during:** Task 2 (pre-implementation inspection)
- **Issue:** timm Swin `features_only` returns tensors in (B,H,W,C) layout. The Backbone protocol contract requires (B,C,h,w). `test_swin_protocol` asserts `f.shape[1] == ch` (channel at dim 1), which would fail with Swin's (B,H,W,C) output.
- **Fix:** `SwinBackbone.forward` applies `f.permute(0, 3, 1, 2).contiguous()` to each level before returning.
- **Files modified:** `scripts/seg/backbones.py`
- **Commit:** 0652623

## Known Stubs

- `scripts/seg/model.py`: `SegModelVariantA` and `SegModelVariantB` have minimal `__init__(backbone)` bodies only. Full decoder+heads wiring (D-01/D-02/D-03a) is in plan 03-04. The stubs do not block plan 03-02's success criteria.

## Threat Surface Scan

No new trust boundaries beyond those declared in the plan's threat model (T-03-04 through T-03-06). The T-03-05 / T-03-06 mitigations are implemented:
- SigLIP type assertion after vision-tower resolution (T-03-05)
- Backbone holds vision tower only; no `language_model`/`gemma` param reachable (T-03-06)

## Self-Check: PASSED
