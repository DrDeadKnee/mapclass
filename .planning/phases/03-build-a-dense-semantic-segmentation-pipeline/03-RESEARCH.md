# Phase 3: Build a dense semantic segmentation pipeline - Research

**Researched:** 2026-05-15
**Domain:** Dense semantic segmentation — frozen ViT (SigLIP/DINOv2) + Swin backbones, conv decoder, recursive coarse-to-fine inference, PyTorch/transformers/timm
**Confidence:** HIGH on backbone/decoder mechanics and library APIs; MEDIUM on recursive-prior channel encoding (a discretion call with a clear recommended default); HIGH on dataloader/test contract (read directly from Phase-1/Phase-2 source).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01: Lightweight conv decoder.** Small FPN/UPerNet-style convolutional decoder on SigLIP patch-token features (NOT a linear-probe upsample, NOT DPT-reassemble). Chosen for crisp terrain-boundary quality at modest params with fast Phase-4 training on the frozen encoder. The decoder may internally fuse multi-scale backbone features.
- **D-02: Shared decoder trunk + two thin task-specific output layers.** One decoder feeds two lightweight heads (9-class land cover, 3-class topography). NOT two fully independent decoders.
- **D-03: Explicit recursive prior.** Coarse 896-scale predicted class-probability maps are fed as **extra input channels** into the finer 448→224 predictions. 2-pass recursive pipeline, not single-pass feature fusion. The conv decoder may ALSO do internal feature fusion; the load-bearing requirement is the coarse→fine predicted-probability feedback.
- **D-04: Inference walks Phase-2 `pyramid.json` parent→child literally.** Coarse-to-fine traversal follows the stored 896→4×448→16×224 parent→child indices. No independent sliding-window/resampling reconstruction at inference for Phase-2 inputs.
- **D-05: One unified `Backbone` protocol.** Single interface: `forward(image) -> feature maps at declared strides`. SigLIP (primary) and DINOv2 are columnar ViTs (multi-scale via sliding window over the pyramid); Swin is natively hierarchical. SAME conv decoder, recursive-prior wiring, and eval harness on all three.
- **D-06: Phase 3 = model + inference construction only.** Deliverable: assembled model (backbone + shared conv decoder + 2 heads), recursive coarse-to-fine inference, all three backbone variants behind D-05, a dataloader over the Phase-2 pyramid dataset, and a forward-pass / output-shape smoke-test on real Phase-2 tiles. **Backbone frozen by default.** Phase 4 owns ALL training/fine-tuning and the joint per-pixel NLL evaluation. NO training loop, NO overfit-a-batch run, NO unfreeze hooks in Phase 3.

### Claude's Discretion
- Exact conv-decoder topology (channel widths, number of upsample stages, which SigLIP block(s) to tap, norm/activation choices) — within D-01.
- How the coarse-prior channels are encoded (raw softmax probs vs logits vs argmax one-hot; resize/alignment method onto the finer grid) — within D-03.
- DINOv2/Swin checkpoint sources and exact stride declarations behind the D-05 protocol.
- Dataloader internals (batching strategy, pyramid-aware sampling, sample-weight plumbing) — under D-06.

### Deferred Ideas (OUT OF SCOPE)
- **OCR / place-name & text reading off maps** — NOT in Phase 3 scope. Belongs to a future v2 milestone (closest to deferred GEOREF-V2). No action this phase.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PHASE-03 | Dense seg pipeline on Phase-1 SigLIP + 2 heads (9-class LC, 3-class topo) + coarse-to-fine; DINOv2/Swin baselines available | §Standard Stack (transformers `PaliGemmaForConditionalGeneration` → `.vision_tower`, PEFT adapter load), §Architecture Patterns (UPerNet-on-ViT decoder, recursive prior), §Code Examples (vision-tower-only forward, ViT token→grid reshape, Swin `features_only`). Gemma-absence provably enforced via `get_image_features` (vision_tower + projector only — never language_model). |
| EVAL-03 | DINOv2 + Swin backbone variants evaluated on the same metric as SigLIP for direct comparison | §Architecture Patterns (unified `Backbone` protocol with declared strides), §Code Examples (DINOv2 `get_intermediate_layers(reshape=True)`, Swin `feature_info.reduction()`), §Common Pitfalls (Swin-vs-ViT stride mismatch in shared decoder). Phase 3 *constructs* the swappable variants; Phase 4 runs the eval. |
</phase_requirements>

## Summary

Phase 3 assembles — does not train — a dense terrain segmentation model. The load-bearing technical facts: (1) The Phase-1 artifact is a **PEFT LoRA adapter directory** wrapping `PaliGemmaForConditionalGeneration`. The SigLIP vision tower is reachable as a submodule, and transformers ships a method, `get_image_features`, that runs **only** `vision_tower` + `multi_modal_projector` and provably never touches the Gemma `language_model`. This is the literal mechanism that satisfies PHASE-03 success criterion #2 ("NO Gemma in the seg forward pass"). The decoder must therefore consume the **pre-projector** SigLIP `last_hidden_state` (the projector maps into Gemma's token space and is irrelevant to dense prediction — tap the vision tower directly, not `get_image_features`'s projected output).

(2) `paligemma-3b-pt-224` uses **SigLIP-So400m/14**: patch size 14, native input 224 → a 16×16 = 256-token grid, hidden_size 1152, 27 layers, NO CLS token. For the 448 and 896 pyramid scales the position embeddings must be interpolated (`interpolate_pos_encoding=True`), yielding 32×32 and 64×64 token grids respectively. The ViT path produces a single coarse feature grid per tile; the conv decoder upsamples it to full tile resolution. DINOv2 (timm `get_intermediate_layers(..., reshape=True)`) behaves the same way. Swin (timm `features_only=True`) is the odd one out — it natively emits a 4-level hierarchical pyramid at strides 4/8/16/32, which the shared decoder must accommodate behind the D-05 protocol.

(3) The recursive prior (D-03) is a 2-pass pipeline: predict 9+3 class probabilities at 896, then for each of the 4 child 448 tiles, crop the corresponding quadrant of the 896 probability map, resize it to the 448 tile's spatial size, and **concatenate it as extra channels onto the RGB input** (or inject at the decoder — recommended: input-level, see Architecture). The Phase-2 `pyramid.json` parent→child indices and pixel `(x, y, size)` boxes make the spatial crop exact and side-step any resampling-topology re-derivation (D-04). Cold start: the 896 pass receives zero-filled prior channels.

**Primary recommendation:** Build a `Backbone` protocol returning `list[Tensor]` feature maps tagged with declared strides; implement `SiglipBackbone` (wrap the Phase-1 PEFT model, call `model.vision_tower` directly with `interpolate_pos_encoding=True`, reshape tokens to `(B, C, h, w)`), `Dinov2Backbone` (timm), `SwinBackbone` (timm `features_only`). One UPerNet-style conv decoder (PPM on the deepest feature + lateral 1×1 convs + FPN top-down fuse + 3×3 smoothing) feeds two 1×1-conv heads. Recursive prior enters as extra input channels at the finer scale, sourced by literal `pyramid.json` index/box lookup. Validate with a pure forward/shape pytest plus a parameter-graph assertion that no `language_model` parameter is reachable from the seg module.

## Architectural Responsibility Map

This is a single-process PyTorch research pipeline (no client/server/CDN tiers). "Tier" here = pipeline stage / module boundary.

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Load Phase-1 LoRA SigLIP weights | Backbone module (`SiglipBackbone`) | — | Owns PEFT adapter loading + Gemma-exclusion; nothing else should touch transformers internals |
| Extract dense feature grid from a tile | Backbone module (per-variant) | — | The only place ViT-token→grid reshape / Swin hierarchy differences live; isolates D-05 seam |
| Multi-scale feature fusion (within a tile) | Conv decoder | — | D-01: internal FPN/PPM fuse is decoder responsibility |
| Per-pixel class logits (9 LC, 3 topo) | Two task heads | Conv decoder (shared trunk) | D-02: shared trunk, thin heads |
| Recursive coarse→fine probability feedback | Inference orchestrator | Dataset/`pyramid.json` reader | D-03/D-04: cross-tile, cross-scale — above the per-tile model; consumes Phase-2 manifest topology |
| Read Phase-2 pyramid tiles + weights + split | Dataset/DataLoader | — | D-06: surfaces `image/land_cover/topography/sample_weights`, train/-only |
| Backbone swap for EVAL-03 | `Backbone` protocol seam | Phase-4 eval harness | D-05: decoder/heads/inference identical across all three |
| Forward/shape + Gemma-absence proof | Test suite (pytest) | — | D-06 deliverable; PHASE-03 SC#2 is a parameter-graph assertion |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | 2.12.0 | Model definition, tensors | Already in `requirements.txt`; project standard |
| transformers | 5.8.1 | `PaliGemmaForConditionalGeneration` / SigLIP vision tower, `get_image_features` | `requirements.txt` pins `>=4.41`; Phase-1 uses it. SigLIP tower + Gemma-exclusion API live here |
| peft | 0.19.1 | Load Phase-1 LoRA adapter onto SigLIP attention | `requirements.txt` pins `>=0.10`; Phase-1 produced a PEFT adapter dir |
| timm | 1.0.27 | DINOv2 + Swin benchmark backbones, `features_only`, `get_intermediate_layers` | Canonical source for both; uniform `feature_info` (channels + stride) interface — the natural backing for the D-05 declared-stride contract |
| Pillow | (pinned, unversioned) | Read Phase-2 tile PNGs | Project standard; matches `scripts/tiling.py` I/O idiom |
| numpy | (pinned, unversioned) | Label array handling | Project standard |

**Version verification (PyPI, checked 2026-05-15):**
`transformers 5.8.1` (2026-05-13) · `peft 0.19.1` (2026-04-16) · `timm 1.0.27` (2026-05-08) · `torch 2.12.0` (2026-05-13). `[VERIFIED: pypi.org/pypi/<pkg>/json]`
**Note (MEDIUM):** `requirements.txt` pins `transformers>=4.41`, `peft>=0.10` (Phase-1 era). The transformers attribute path changed between 4.x and 5.x (see Pitfall 3). The plan must either (a) pin the resolved transformers version, or (b) write the backbone wrapper defensively (getattr fallback across `model.vision_tower` and `model.model.vision_tower`). Do NOT assume the 4.41 layout.

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `timm` | 1.0.27 | `timm.create_model("vit_*.dinov2", features_only=...)` or `get_intermediate_layers` | DINOv2 backbone variant only |
| `torch.nn.functional` | (torch) | `interpolate` for prior-channel resize + final upsample | Decoder + recursive prior |
| pytest / pytest-mock | >=8 / >=3 | Forward/shape smoke-test, Gemma-absence assertion | D-06 deliverable; existing infra |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| transformers `SiglipVisionModel` via PaliGemma wrapper | Standalone `google/siglip-so400m-patch14-224` from HF | REJECTED — Phase-1 LoRA was trained inside the PaliGemma wrapper; the adapter `target_modules` and module names are PaliGemma-scoped. Loading the standalone SigLIP would drop the Phase-1 fine-tune. Must load the PaliGemma model + PEFT adapter, then reach the tower. |
| timm DINOv2/Swin | HF `transformers` `Dinov2Model` / `SwinModel` | Either works; timm gives a *uniform* `feature_info.reduction()` declared-stride API across both — directly serves D-05. transformers DINOv2/Swin have divergent output shapes and no common stride accessor. Recommend timm for the two benchmarks; SigLIP stays on transformers (Phase-1 constraint). |
| UPerNet (PPM + FPN) | Plain FPN, or SegFormer-style all-MLP, or DPT-reassemble | D-01 explicitly picks FPN/UPerNet-style and explicitly excludes DPT-reassemble and linear-probe. UPerNet's PPM adds the global context that helps terrain-boundary coherence — keep it. |

**Installation:**
```bash
pip install "torch==2.12.0" "transformers==5.8.1" "peft==0.19.1" "timm==1.0.27" pillow numpy
```
(Resolve and pin the actual transformers version chosen — see Pitfall 3.)

## Architecture Patterns

### System Architecture Diagram

```
                         Phase-1 PEFT adapter dir            Phase-2 pyramid tree
                         (LoRA on SigLIP attn,                (pyramid.json + per-tile
                          Gemma frozen)                        image/lc/topo/weights)
                                  |                                     |
                                  v                                     v
        ┌─────────────────────────────────────┐       ┌──────────────────────────────┐
        │  Backbone protocol  (D-05 seam)      │       │  PyramidDataset / DataLoader │
        │  forward(img[,prior]) -> [feat maps  │       │  - reads pyramid.json        │
        │   @ declared strides]                │       │  - train/ ONLY (frozen split)│
        │  ┌─────────┬──────────┬───────────┐  │       │  - surfaces sample_weights   │
        │  │ Siglip  │ Dinov2   │  Swin     │  │       │  - parent→child indices      │
        │  │(transf. │ (timm    │ (timm     │  │       └───────────────┬──────────────┘
        │  │ PaliG   │  ViT)    │  features │  │                       │
        │  │ .vision │  reshape │  _only,   │  │                       │
        │  │ _tower) │  =True   │  4 levels)│  │                       │
        │  └─────────┴──────────┴───────────┘  │                       │
        └──────────────────┬───────────────────┘                       │
                           │  feature map(s) @ stride(s)                │
                           v                                            │
        ┌─────────────────────────────────────┐                        │
        │  Shared conv decoder (D-01)          │                        │
        │  PPM(deepest) → lateral 1×1 →         │                        │
        │  FPN top-down fuse → 3×3 smooth →     │                        │
        │  upsample to tile HxW                 │                        │
        └──────────────────┬───────────────────┘                        │
                           │  shared dense features                     │
              ┌────────────┴────────────┐                               │
              v                          v                              │
        ┌───────────┐            ┌───────────┐                          │
        │ LC head   │            │ Topo head │  (D-02 thin 1×1 convs)    │
        │ (B,9,H,W) │            │ (B,3,H,W) │                           │
        └─────┬─────┘            └─────┬─────┘                           │
              └───────────┬────────────┘                                │
                          v                                             │
        ┌──────────────────────────────────────────────────────┐       │
        │ Recursive c2f inference orchestrator (D-03/D-04)       │<──────┘
        │  pass 1: 896 tile, prior = zeros  → softmax probs      │
        │  pass 2: for each 448 child (pyramid.json indices):    │
        │          crop+resize parent prob → extra input chans   │
        │          → predict; repeat 448→224                     │
        └──────────────────────────────────────────────────────┘
                          │
                          v
              dense (B,9,H,W) LC + (B,3,H,W) topo probability tensors
```
Trace the primary use case: a regional map tile enters at the DataLoader, flows through the Backbone (Gemma never on the path), the shared decoder, the two heads, and the recursive orchestrator that re-injects coarse probabilities at finer scales using the Phase-2 manifest topology.

### Recommended Project Structure
```
scripts/
├── seg/
│   ├── __init__.py
│   ├── backbones.py      # Backbone protocol + Siglip/Dinov2/Swin impls (D-05)
│   ├── decoder.py        # UPerNet-style PPM+FPN conv decoder (D-01)
│   ├── heads.py          # two thin 1x1-conv task heads (D-02)
│   ├── model.py          # SegModel = backbone + decoder + heads; accepts prior channels (D-03)
│   ├── recursive.py      # c2f orchestrator walking pyramid.json (D-03/D-04)
│   └── dataset.py        # PyramidDataset over Phase-2 tree, train/-only (D-06)
tests/
├── test_seg_backbones.py # shape + stride contract per variant; Gemma-absence assertion
├── test_seg_decoder.py   # decoder/head output shapes
├── test_seg_recursive.py # prior-channel alignment vs pyramid.json indices
├── test_seg_dataset.py   # split safety, weight surfacing, batch shapes
└── test_seg_smoke.py     # end-to-end forward on a synthetic mini-pyramid (D-06 deliverable)
```
Place under `scripts/seg/` so `pytest.ini`'s `pythonpath = scripts` makes `from seg.backbones import ...` work with zero config change.

### Pattern 1: Vision-tower-only forward (provable Gemma exclusion)
**What:** Run the SigLIP encoder of the Phase-1 PaliGemma model without ever invoking `language_model`.
**When to use:** The `SiglipBackbone.forward`. This is the literal PHASE-03 SC#2 guarantee.
**Key facts** `[CITED: huggingface.co/docs/transformers/model_doc/paligemma, transformers v5.8.1 source]`:
- `PaliGemmaForConditionalGeneration` (transformers 5.x) holds submodules under `self.model`: `self.model.vision_tower` (a `SiglipVisionModel`), `self.model.multi_modal_projector`, `self.model.language_model`.
- `get_image_features(pixel_values)` body is exactly: `image_outputs = self.vision_tower(pixel_values); selected = image_outputs.last_hidden_state; image_features = self.multi_modal_projector(selected); return ...` — **it never calls `language_model`.**
- For dense segmentation, tap `vision_tower(...).last_hidden_state` **directly** (the multimodal projector maps into Gemma's token space — not useful for pixels; skipping it also keeps the decoder input dim = SigLIP hidden_size 1152, not projection_dim 2048).
- The decoder/seg module must hold a reference to the *vision tower only* (or assert no `language_model.*` parameter is in `seg_model.named_parameters()`), so the parameter graph proves Gemma is absent.

### Pattern 2: ViT patch tokens → spatial feature grid
**What:** Reshape `(B, N, C)` token sequence to `(B, C, h, w)`.
**SigLIP-So400m/14 specifics** `[CITED: SigLIP/PaliGemma docs; SigLIP is patch16/14 family]`:
- `paligemma-3b-pt-224` vision tower = SigLIP-So400m, **patch_size 14**, image_size 224, hidden_size **1152**, 27 layers, **NO CLS token** (SigLIP uses no class embedding — every token is a patch, so `h*w == N` exactly, no token-stripping needed).
- 224 input → 16×16 = 256 tokens (224/14 = 16). 448 → 32×32 = 1024. 896 → 64×64 = 4096. **Use `interpolate_pos_encoding=True`** for the 448/896 scales (the Phase-2 pyramid was sized specifically around this — see Phase-2 02-CONTEXT §specifics).
- Reshape: `feat = last_hidden_state.transpose(1,2).reshape(B, C, h, w)` where `h = w = sqrt(N)` (square tiles — Phase-2 tiles are always square 896/448/224, so no non-square edge case **except** edge pyramids that PIL zero-pads to full size; the grid stays square).

### Pattern 3: UPerNet-style conv decoder on a (mostly) single-scale ViT grid
**What:** PPM on the deepest feature + lateral 1×1 + FPN top-down + 3×3 smoothing + upsample to tile resolution. `[VERIFIED: WebSearch — UPerNet-on-ViT is the established ViT→dense pattern (ViT-UPerNet, SETR-MLA, Segmenter); reshaped patch grid feeds FPN levels after up/down scale transforms]`
**When to use:** The shared decoder for all three backbones.
**ViT multi-scale option (D-01 "may internally fuse multi-scale"):** A columnar ViT has one native stride. To give the FPN >1 level, tap **multiple transformer blocks** (e.g. SigLIP layers ~9/18/27 via `output_hidden_states=True`) and treat them as pseudo-pyramid levels (this is the SETR-MLA / ViT-Adapter recipe). Alternatively keep it single-scale + PPM only. Recommend: tap 3–4 evenly-spaced blocks → reshape each → FPN. Swin already gives 4 true levels.
**Boundary sharpness (the D-01 driver):** final upsample should be progressive (e.g. 2× conv-transpose or bilinear + 3×3 conv stages) rather than a single ×14 bilinear blow-up, to keep coastlines/ranges crisp.

### Pattern 4: Explicit recursive prior as extra input channels (D-03)
**What:** Coarse predicted class-probabilities → resized/cropped → concatenated to the finer tile's input.
**Recommended encoding (discretion call within D-03):** **raw softmax probabilities** (9 LC + 3 topo = **12 extra channels**), bilinearly resized — not argmax one-hot (throws away the calibrated distribution that the joint-NLL metric and the geophysical-prior thesis are explicitly about) and not logits (unbounded scale destabilises a frozen-backbone first layer).
**Where it enters:** **At the image input** — concat to RGB so the *first conv of the decoder path* (or a small prior-stem) sees it. Recommended over decoder-level injection because (a) it is backbone-agnostic (works identically for Siglip/Dinov2/Swin behind D-05 — the prior never has to match a backbone's internal feature stride), (b) it matches the project's "predicted distribution as prior" thesis literally. Caveat: a frozen ViT backbone cannot ingest extra input channels (its patch-embed conv is 3-channel and frozen). **Resolution:** route RGB through the frozen backbone as normal, and feed the 12 prior channels into the *decoder* via a tiny trainable prior-encoder that is resized to the decoder's working resolution and summed/concatenated there. This keeps the prior "explicit extra channels" (D-03) while respecting the frozen-backbone constraint (D-06). Document this as the resolved interpretation of D-03 for a frozen ViT.
**Spatial alignment (D-04):** For child tile `448_i`, the parent `896` covers pixel box `(ox, oy, ox+896, oy+896)`; the child covers `(cx, cy, cx+448, cy+448)` from `pyramid.json`. Crop the parent probability map at `(cx-ox, cy-oy, cx-ox+448, cy-oy+448)`, then resize to the child grid. **Use the manifest boxes literally; do not recompute geometry.** Cold start: at 896, prior channels = zeros (the orchestrator owns this; the model just accepts a prior arg that may be all-zero).

### Pattern 5: Unified Backbone protocol with declared strides (D-05)
```python
class Backbone(Protocol):
    feature_strides: list[int]      # declared, e.g. [16] for ViT, [4,8,16,32] for Swin
    feature_channels: list[int]
    def forward(self, x: Tensor) -> list[Tensor]: ...   # one (B,C,h,w) per stride
```
- `SiglipBackbone`: strides `[16]` (patch14 ≈ stride14, but treat the pyramid scale as the effective stride family) or multi-block `[stride]*k`; channels `[1152]*k`. Calls `model.model.vision_tower(pixel_values, interpolate_pos_encoding=True, output_hidden_states=True)`.
- `Dinov2Backbone`: timm `vit_*_patch14_dinov2`; `model.get_intermediate_layers(x, n=k, reshape=True)` → list of `(B, C, h, w)` already reshaped `[VERIFIED: WebSearch — timm get_intermediate_layers(reshape=True) returns 2D spatial maps]`. Declared stride 14.
- `SwinBackbone`: `timm.create_model("swin_*", features_only=True, out_indices=(0,1,2,3))`; declared strides from `model.feature_info.reduction()`, channels from `model.feature_info.channels()` `[VERIFIED: WebSearch — timm features_only + feature_info]`.
- The decoder reads `feature_strides`/`feature_channels` from whatever backbone is plugged in, so the SAME decoder code adapts (Swin → 4 real FPN levels; ViT → k pseudo-levels). This is the EVAL-03 apples-to-apples seam.

### Anti-Patterns to Avoid
- **Calling `get_image_features` for the decoder input.** It applies the multimodal projector → wrong feature space (Gemma token space) and wrong dim. Use `vision_tower(...).last_hidden_state` directly.
- **Loading standalone `google/siglip-*` instead of the Phase-1 PaliGemma+adapter.** Silently discards the entire Phase-1 LoRA fine-tune.
- **argmax/one-hot prior channels.** Discards the calibrated distribution the recursive-prior thesis and joint-NLL metric depend on.
- **Recomputing pyramid geometry at inference.** D-04 says walk `pyramid.json` literally; re-deriving sliding-window topology risks off-by-patch misalignment of the prior.
- **Any training loop, optimizer, `.backward()`, overfit run, or unfreeze hook.** D-06: out of scope. The smoke-test is forward-only / `torch.no_grad()`.
- **Enumerating `data/synthetic/test/`.** Frozen-split leakage; dataset must point only at `train/` (Phase-2 D-17).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| ViT pos-embed interpolation for 448/896 | Custom bicubic resize of position table | transformers `interpolate_pos_encoding=True` arg | SigLIP's adapted (no-CLS) implementation handles it + is jit-traceable; hand-rolling gets the no-CLS layout subtly wrong |
| DINOv2 token→grid reshape | Manual transpose/reshape + CLS-strip guesswork | timm `get_intermediate_layers(..., reshape=True)` | Handles register/CLS tokens and reshape correctly per checkpoint |
| Swin multi-scale feature extraction | Re-implement patch-merging taps | timm `features_only=True, out_indices=...` | Uniform `feature_info` (channels + reduction) — also backs the D-05 stride contract |
| LoRA adapter loading onto SigLIP | Manual state-dict surgery | `peft` `PeftModel.from_pretrained(base, adapter_dir)` | Phase-1 saved a PEFT adapter dir; mirror `finetune_paligemma.py`'s isolation exactly |
| FPN/PPM decoder | Novel decoder | UPerNet recipe (PPM + lateral + top-down + smooth) | D-01 names it; well-trodden, training-cheap on a frozen backbone |
| Pyramid parent→child geometry | Recompute crop boxes | Read `pyramid.json` `tiles[].{x,y,size,children}` | Phase-2 `scripts/tiling.py` already emits exact stable indices (D-04) |

**Key insight:** Every "novel" piece here (pos-embed interp, token reshape, hierarchical taps, adapter load, pyramid topology) is a solved problem in transformers/timm/peft or already materialised by Phase-2. Phase 3's only genuinely new logic is the **wiring**: the Backbone protocol seam, the shared decoder, and the recursive-prior orchestration.

## Common Pitfalls

### Pitfall 1: Mixing Gemma into the forward graph
**What goes wrong:** Calling the full PaliGemma `forward`/`generate`, or `get_image_features` (projector), pulls Gemma (or Gemma-space projection) into the graph — fails PHASE-03 SC#2.
**Why it happens:** The convenient high-level APIs all route toward the LM.
**How to avoid:** `SiglipBackbone` holds and calls **only** `paligemma_model.model.vision_tower`. Add a test asserting `not any("language_model" in n for n,_ in seg_model.named_parameters())` and that no module is an instance of the Gemma model class.
**Warning signs:** Decoder input dim == 2048 (projection_dim) instead of 1152 (SigLIP hidden); VRAM spikes; `language_model` weights in the seg state-dict.

### Pitfall 2: PEFT adapter target-module / wrapper mismatch
**What goes wrong:** Loading the adapter onto a bare SigLIP, or onto a PaliGemma loaded differently than Phase-1, fails to bind LoRA (`target_modules=["q_proj","k_proj","v_proj","out_proj"]` won't match) — the Phase-1 fine-tune is silently lost.
**Why it happens:** `finetune_paligemma.py` applies LoRA to the PaliGemma-wrapped SigLIP via `get_peft_model`; module names are PaliGemma-scoped (`...vision_tower...out_proj`).
**How to avoid:** Mirror `apply_lora` load order: load `PaliGemmaForConditionalGeneration` (same `model-id`, `google/paligemma-3b-pt-224`), then `PeftModel.from_pretrained(model, adapter_dir)`, then reach `.model.vision_tower`. Verify LoRA params are present on the vision tower after load (count trainable-ish / inspect `peft` adapter keys).
**Warning signs:** `peft` warns "no module matched target_modules"; backbone outputs identical to a fresh pretrained SigLIP.

### Pitfall 3: transformers 4.x vs 5.x attribute path
**What goes wrong:** Phase-1 `requirements.txt` pins `transformers>=4.41`. In 4.x the submodules sat on `PaliGemmaForConditionalGeneration` directly (`.vision_tower`); in **5.x they live under `.model`** (`.model.vision_tower`). A hard-coded path breaks on whichever version isn't installed.
**Why it happens:** transformers refactored PaliGemma into a `PaliGemmaModel` + head split. `[CITED: transformers v5.8.1 modeling_paligemma source]`
**How to avoid:** Resolve and **pin** the transformers version in the Phase-3 plan. Write the accessor defensively: `vt = getattr(m, "vision_tower", None) or m.model.vision_tower`. Add a test that asserts the resolved tower is a SigLIP module.
**Warning signs:** `AttributeError: ... has no attribute 'vision_tower'`.

### Pitfall 4: Swin-vs-ViT stride mismatch in the shared decoder
**What goes wrong:** Decoder hard-codes ViT's single stride; Swin's 4-level pyramid (strides 4/8/16/32) then misaligns in the FPN top-down add (spatial sizes don't match).
**Why it happens:** Columnar ViT and hierarchical Swin have fundamentally different feature geometry — the whole reason D-05 exists.
**How to avoid:** Decoder reads `backbone.feature_strides`/`feature_channels` and builds lateral convs + interpolation dynamically; never hard-code level count or channel dims. `F.interpolate` each level to a common decoder resolution before fusion.
**Warning signs:** Shape-mismatch RuntimeError only on the Swin variant; decoder works for SigLIP/DINOv2 but not Swin.

### Pitfall 5: Prior-channel spatial misalignment across scales
**What goes wrong:** Resizing the whole 896 probability map to 448 (instead of cropping the correct quadrant first) feeds a child the parent's *entire* context squashed 2× — spatially wrong by construction.
**Why it happens:** Forgetting that a 448 child is one quadrant of the 896, not a downscale of it.
**How to avoid:** Use `pyramid.json` boxes: crop parent prob at `(cx-ox, cy-oy)` for `size` 448, *then* resize. Add a test that places a known one-hot blob in a parent quadrant and asserts it lands in the right child.
**Warning signs:** Recursive prior degrades rather than improves coherence; prior "leaks" across quadrant boundaries.

### Pitfall 6: Frozen split leakage through the dataloader
**What goes wrong:** Dataset enumerates `data/synthetic/` recursively and sweeps in `test/` pyramids — breaks EVAL-01's zero-leakage guarantee end-to-end.
**Why it happens:** Phase-2 puts `train/` and `test/` as sibling dirs (D-17); a naive `glob("**/pyramid.json")` finds both.
**How to avoid:** Dataset takes an explicit `train/` root (or filters by `split.json`); add a test asserting no path under the dataset contains a `test/` component (mirror Phase-2's `test_split_subtree_preserved`).
**Warning signs:** Dataset length larger than the train split; `test` in sample paths.

### Pitfall 7: Edge-pyramid zero-padding silently feeding the model garbage
**What goes wrong:** Phase-2 keeps pyramids whose 896 footprint is ≤50% off the source; PIL zero-pads the off-map region (`scripts/tiling.py` docstring). Those black borders are valid pixels to the model but meaningless terrain.
**Why it happens:** D-09 edge policy + PIL crop-beyond-extent behaviour.
**How to avoid:** Phase 3 is forward/shape-only (D-06), so this does not block the deliverable — but flag it for Phase-4 (loss masking) and note it in the smoke-test (zero-padded tiles must still produce correctly *shaped* outputs without NaN).
**Warning signs:** N/A for Phase 3 outputs; a Phase-4 concern — recorded here so it isn't lost.

## Code Examples

### SiglipBackbone — Phase-1 adapter load + Gemma-free dense features
```python
# Source: transformers PaliGemma/SigLIP docs (v5.8.1) + scripts/finetune_paligemma.py::apply_lora
import torch
from transformers import PaliGemmaForConditionalGeneration
from peft import PeftModel

def load_siglip_backbone(model_id: str, adapter_dir: str):
    base = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base, adapter_dir)        # binds Phase-1 LoRA
    model.eval().requires_grad_(False)                          # D-06: frozen by default
    # defensive accessor (Pitfall 3): 5.x -> .model.vision_tower ; 4.x -> .vision_tower
    inner = getattr(model, "base_model", model)
    pg = getattr(inner, "model", inner)
    vt = getattr(pg, "vision_tower", None) or pg.model.vision_tower
    assert "language_model" not in type(vt).__name__.lower()
    return vt   # a SiglipVisionModel — Gemma is NOT reachable from here

@torch.no_grad()
def siglip_features(vision_tower, pixel_values, tap_layers=(8, 17, 26)):
    out = vision_tower(pixel_values,
                        interpolate_pos_encoding=True,   # required for 448/896
                        output_hidden_states=True)
    grids = []
    B = pixel_values.shape[0]
    for li in tap_layers:
        tok = out.hidden_states[li]            # (B, N, 1152), N = (S/14)^2, no CLS
        n = tok.shape[1]; h = w = int(round(n ** 0.5))
        grids.append(tok.transpose(1, 2).reshape(B, -1, h, w))  # (B,1152,h,w)
    return grids
```

### DINOv2 / Swin backbones via timm (declared-stride contract)
```python
# Source: WebSearch (HF timm feature_extraction docs; pytorch-image-models discussion #2068)
import timm, torch

# DINOv2 — columnar ViT, reshape=True gives (B,C,h,w)
dino = timm.create_model("vit_base_patch14_reg4_dinov2.lvd142m",
                          pretrained=True, num_classes=0).eval().requires_grad_(False)
@torch.no_grad()
def dino_features(x, k=4):
    return dino.get_intermediate_layers(x, n=k, reshape=True)   # list of (B,C,h,w)

# Swin — native hierarchy, 4 levels at strides 4/8/16/32
swin = timm.create_model("swin_base_patch4_window7_224",
                          pretrained=True, features_only=True,
                          out_indices=(0, 1, 2, 3)).eval().requires_grad_(False)
print(swin.feature_info.reduction())   # [4, 8, 16, 32]  -> Backbone.feature_strides
print(swin.feature_info.channels())    # e.g. [128,256,512,1024] -> feature_channels
```

### Recursive prior — literal pyramid.json walk (D-03/D-04)
```python
# Source: scripts/tiling.py pyramid.json schema (tiles[].{id,x,y,size,children})
import json, torch
import torch.nn.functional as F

def recursive_predict(model, pyramid_dir, read_tile_rgb):
    man = json.loads((pyramid_dir / "pyramid.json").read_text())
    by_id = {t["id"]: t for t in man["tiles"]}
    root = next(t for t in man["tiles"] if t["size"] == 896)

    prob_cache = {}
    def predict(tile, parent):
        rgb = read_tile_rgb(pyramid_dir, tile)                 # (1,3,S,S)
        if parent is None:
            prior = torch.zeros(1, 12, *rgb.shape[-2:])        # cold start at 896
        else:
            p = prob_cache[parent["id"]]                       # (1,12,Sp,Sp) parent probs
            ox, oy = parent["x"], parent["y"]
            cx, cy, s = tile["x"], tile["y"], tile["size"]
            crop = p[..., cy-oy:cy-oy+s, cx-ox:cx-ox+s]        # exact quadrant (D-04)
            prior = F.interpolate(crop, size=rgb.shape[-2:], mode="bilinear",
                                  align_corners=False)
        probs = model(rgb, prior)                              # (1,12,S,S) softmaxed
        prob_cache[tile["id"]] = probs
        for cid in tile["children"]:
            predict(by_id[cid], tile)

    predict(root, None)
    return prob_cache   # keyed by tile id; 224-level entries are the finest output
```

### Gemma-absence assertion (PHASE-03 SC#2 — pytest)
```python
# Source: D-06 deliverable; parameter-graph proof
def test_no_gemma_in_seg_forward(seg_model):
    names = [n for n, _ in seg_model.named_parameters()]
    assert not any("language_model" in n for n in names), \
        "Gemma parameters reachable from segmentation model"
    # also assert no Gemma module type is present
    assert not any("gemma" in type(m).__name__.lower()
                   for m in seg_model.modules())
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| transformers `PaliGemmaForConditionalGeneration.vision_tower` (direct) | `.model.vision_tower` (model/head split) | transformers 5.x | Backbone accessor must be defensive/version-pinned (Pitfall 3) |
| Manual ViT token reshape + CLS strip | timm `get_intermediate_layers(reshape=True)` | timm 0.9+ | Handles register/CLS tokens correctly; do not hand-roll |
| DPT-reassemble / linear probe for ViT dense | UPerNet/FPN-on-ViT, SegFormer, ViT-Adapter | ~2021–2023 | D-01 explicitly picks the UPerNet/FPN family |

**Deprecated/outdated:**
- Assuming a CLS token in SigLIP token reshape — **SigLIP has no class embedding**; `N == h*w` exactly. (DINOv2 *does* have CLS + optional registers — let timm strip them.)

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `paligemma-3b-pt-224` SigLIP vision config = So400m/14: hidden_size 1152, patch 14, 27 layers, no CLS | Summary, Pattern 2 | LOW–MED. These are the well-known published So400m numbers, but the live `config.json` requires HF auth and was not fetched this session. Decoder input dim and token-grid math depend on it — **verify with one line at plan time:** `PaliGemmaForConditionalGeneration.from_pretrained(id).config.vision_config` and assert hidden_size/patch_size/num_hidden_layers before sizing the decoder. |
| A2 | Phase-1 adapter saved at `checkpoints/paligemma-terrain/epochNN/` as a PEFT dir | Stack, Pitfall 2 | LOW–MED. `finetune_paligemma.py` writes there by default, but no `checkpoints/` dir exists in the repo yet (Phase-1 ran pre-bootstrap; artifact location/availability unconfirmed). Plan must take the adapter path as a parameter and surface a clear error if absent — see Environment Availability. |
| A3 | Input-level prior is infeasible through a frozen 3-ch ViT patch-embed; prior must enter at the decoder | Pattern 4 | LOW. Frozen 3-channel patch-embed conv genuinely cannot take 15 channels without modification (would unfreeze/replace it, violating D-06). Decoder-level injection is the standard resolution. Surface to discuss-phase as the resolved reading of D-03 under the frozen-backbone constraint. |
| A4 | Recommended prior encoding = raw 12-channel softmax probs, decoder-injected | Pattern 4 | LOW. Explicitly a D-03 discretion call; recommendation is well-justified by the joint-NLL/prior thesis but is a recommendation, not a locked fact. |
| A5 | timm checkpoint names (`vit_base_patch14_reg4_dinov2.lvd142m`, `swin_base_patch4_window7_224`) | Code Examples | LOW. Family/interface verified; exact tag is a D-05 discretion call — pick final tags at plan time via `timm.list_models("*dinov2*")` / `timm.list_models("swin*")`. |

## Open Questions

1. **Exact Phase-1 adapter directory + epoch to load.**
   - What we know: `finetune_paligemma.py` defaults to `checkpoints/paligemma-terrain/epochNN/`, PEFT format, `model-id google/paligemma-3b-pt-224`.
   - What's unclear: No `checkpoints/` exists in the repo; Phase-1 ran pre-bootstrap (STATE.md). Actual path/availability of the trained adapter is unconfirmed.
   - Recommendation: Plan takes `--adapter-dir` + `--model-id` as parameters; smoke-test xfails/skips with a clear message if the adapter is absent (forward-shape logic can still be unit-tested with a fresh base model or a tiny stub SigLIP config). Surface to the user.

2. **D-03 channel-injection point under a frozen ViT.**
   - What we know: D-03 says "extra input channels"; D-06 says backbone frozen.
   - What's unclear: A frozen 3-channel ViT patch-embed cannot literally accept extra input channels.
   - Recommendation: Resolve as decoder-level prior injection (Pattern 4 / A3); confirm this reading with the user in discuss/plan so the "explicit extra channels" intent is preserved without unfreezing the backbone.

3. **Multi-scale ViT tap vs single-scale + PPM (decoder topology).**
   - What we know: D-01 permits internal multi-scale fusion; ViT is columnar.
   - What's unclear: 3–4 block taps (SETR-MLA style) vs single deepest + PPM — both valid; tradeoff is params/quality.
   - Recommendation: Default to 3–4 evenly-spaced taps + PPM (better boundary detail, the D-01 driver); leave channel widths to planner discretion.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python deps (torch/transformers/peft/timm) | All Phase-3 modules | ✗ (not importable in this shell; conda base only) | requirements.txt pins exist but not installed in researched env | Plan Wave 0: `pip install -r requirements.txt` + add `timm` to requirements.txt (NOT currently listed) |
| `timm` | DINOv2 + Swin backbones (EVAL-03) | ✗ — **not in `requirements.txt`** | needs `timm==1.0.27` | None — must be added to requirements.txt by the plan |
| Phase-1 PEFT adapter dir | `SiglipBackbone` (primary path) | ✗ — no `checkpoints/` dir in repo | n/a | Forward-shape tests can run against base model / stub config; full SigLIP path needs the artifact (Open Q1) |
| Phase-2 pyramid data (`data/synthetic/train/.../pyramid.json`) | DataLoader, recursive smoke-test | ✗ — no `data/` dir built yet | n/a | Tests synthesise mini-pyramids under `tmp_path` (mirror `tests/test_tiling.py::_make_map_dir`); smoke-test on real tiles is gated on a Phase-2 build run |
| GPU/CUDA | Practical SigLIP forward (3B params; only vision tower used ≈ ~0.4B) | unknown | — | CPU forward feasible for shape tests with tiny batch; flag VRAM/time for real-tile smoke-test |

**Missing dependencies with no fallback:**
- `timm` is not in `requirements.txt` — the EVAL-03 DINOv2/Swin variants cannot be built until it is added. **Plan Wave 0 must add `timm` to `requirements.txt`.**

**Missing dependencies with fallback:**
- Phase-1 adapter & Phase-2 data absent in the working tree: shape/contract tests run fully offline against synthesised fixtures + base/stub models; the *real-tile* smoke-test (D-06) and the SigLIP-LoRA path depend on those artifacts and should be gated/skip-with-message rather than block the build.

## Validation Architecture

`workflow.nyquist_validation` not present in `.planning/config.json` → treated as **enabled**.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >=8 (+ pytest-mock >=3) — already in `requirements.txt` |
| Config file | `pytest.ini` (`testpaths = tests`, `pythonpath = scripts`, marker `integration`) |
| Quick run command | `pytest tests/test_seg_*.py -x -q -m "not integration"` |
| Full suite command | `pytest -q` (offline) / `pytest -q -m integration` (network/heavy, phase gate) |

Established conventions to mirror: offline tests in `tests/`, network/heavy tests `@pytest.mark.integration` under `tests/integration/`, fixtures synthesised under `tmp_path` (see `tests/conftest.py`, `tests/test_tiling.py`). `pythonpath = scripts` means modules under `scripts/seg/` import as `from seg.x import ...` with no config change.

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| PHASE-03 | SigLIP backbone yields (B,C,h,w) grids at declared strides for 224/448/896 | unit | `pytest tests/test_seg_backbones.py -x -q` | ❌ Wave 0 |
| PHASE-03 | No Gemma param/module reachable from seg model (SC#2) | unit | `pytest tests/test_seg_backbones.py::test_no_gemma -x` | ❌ Wave 0 |
| PHASE-03 | Decoder + 2 heads emit (B,9,H,W)+(B,3,H,W) at full tile res | unit | `pytest tests/test_seg_decoder.py -x -q` | ❌ Wave 0 |
| PHASE-03 | Recursive prior crops correct quadrant per pyramid.json (D-03/D-04) | unit | `pytest tests/test_seg_recursive.py -x -q` | ❌ Wave 0 |
| PHASE-03 | End-to-end forward on a synthetic mini-pyramid, no NaN, correct shapes | smoke | `pytest tests/test_seg_smoke.py -x -q` | ❌ Wave 0 |
| EVAL-03 | DINOv2 + Swin satisfy the same Backbone protocol (strides/channels declared, same decoder runs) | unit | `pytest tests/test_seg_backbones.py -k "dino or swin" -x` | ❌ Wave 0 |
| EVAL-03 | Real SigLIP-LoRA load + real-tile forward | integration | `pytest tests/integration/test_seg_online.py -m integration` | ❌ Wave 0 (gated on Phase-1 adapter + Phase-2 data) |

### Sampling Rate
- **Per task commit:** `pytest tests/test_seg_*.py -x -q -m "not integration"`
- **Per wave merge:** `pytest -q -m "not integration"` (full offline suite — must stay green; Phase-2 tests included)
- **Phase gate:** full offline suite green; `pytest -q -m integration` (real SigLIP-LoRA + real-tile forward) green or explicitly skipped-with-reason if Phase-1/Phase-2 artifacts unavailable.

### Wave 0 Gaps
- [ ] `tests/test_seg_backbones.py` — backbone shape/stride contract + Gemma-absence (PHASE-03, EVAL-03)
- [ ] `tests/test_seg_decoder.py` — decoder/head output shapes (PHASE-03)
- [ ] `tests/test_seg_recursive.py` — prior quadrant alignment vs pyramid.json (PHASE-03)
- [ ] `tests/test_seg_dataset.py` — split safety + weight surfacing + batch shapes (PHASE-03/D-06)
- [ ] `tests/test_seg_smoke.py` — synthetic mini-pyramid end-to-end forward (D-06 deliverable)
- [ ] `tests/integration/test_seg_online.py` — real adapter + real tile (gated)
- [ ] Shared seg fixtures (synthetic mini-pyramid builder; reuse `tests/test_tiling.py::_make_map_dir` pattern) — add to `tests/conftest.py` or a local fixture
- [ ] `requirements.txt`: add `timm` (currently missing) — Framework deps install: `pip install -r requirements.txt`

## Sources

### Primary (HIGH confidence)
- `scripts/finetune_paligemma.py` (repo) — Phase-1 PEFT adapter production: LoRA `target_modules=["q_proj","k_proj","v_proj","out_proj"]`, Gemma frozen, `model-id google/paligemma-3b-pt-224`, save path `checkpoints/paligemma-terrain/epochNN/`.
- `scripts/tiling.py` (repo) — `pyramid.json` schema: `tiles[].{id,x,y,size,children}`, 1×896+4×448+16×224 strict 2×2 nesting, stride 448, PIL zero-pad on edge pyramids.
- `tests/conftest.py`, `tests/test_tiling.py`, `pytest.ini` (repo) — test conventions, `pythonpath=scripts`, `integration` marker, `tmp_path` fixture idiom.
- transformers v5.8.1 `modeling_paligemma.py` (via WebFetch) — submodule layout `self.model.{vision_tower,multi_modal_projector,language_model}`; `get_image_features` body (vision_tower + projector, **no language_model**).
- https://huggingface.co/docs/transformers/model_doc/paligemma — PaliGemma/SigLIP architecture, processor, model API.
- https://huggingface.co/docs/transformers/model_doc/siglip — SigLIP vision model, `interpolate_pos_encoding` semantics, no-CLS design.
- PyPI JSON API (`pypi.org/pypi/<pkg>/json`) — verified current versions: transformers 5.8.1, peft 0.19.1, timm 1.0.27, torch 2.12.0.

### Secondary (MEDIUM confidence)
- WebSearch (HF timm `feature_extraction` docs; pytorch-image-models discussion #2068) — `get_intermediate_layers(reshape=True)` returns 2D maps; Swin `features_only`+`feature_info.reduction()/channels()`.
- WebSearch (ViT-UPerNet, Springer; CSAIL semantic-segmentation-pytorch) — UPerNet PPM+FPN-on-ViT is the established ViT→dense decoder pattern; multi-block taps for pseudo-pyramid.
- `.planning/PROJECT.md`, `.planning/REQUIREMENTS.md`, `.planning/phases/02-.../02-CONTEXT.md`, `notes/architectural_references.md` (repo) — locked constraints, Phase-2 dataset contract, recursive-prior thesis.

### Tertiary (LOW confidence — flagged for validation)
- SigLIP-So400m/14 numeric config for `paligemma-3b-pt-224` (hidden 1152 / patch 14 / 27 layers / no CLS): well-known published values but live `config.json` requires HF auth (HTTP 401 this session). **Verify at plan time** via `model.config.vision_config` (Assumption A1).

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — versions PyPI-verified; transformers/peft/timm roles confirmed against repo + official docs.
- Architecture (backbone/decoder/protocol): HIGH — APIs verified via transformers source + timm docs; UPerNet-on-ViT pattern well-established.
- Recursive prior mechanics: MEDIUM — D-04 alignment is deterministic from `pyramid.json` (HIGH); injection point/encoding is a justified recommendation within D-03 discretion (MEDIUM, flagged A3/A4).
- Pitfalls: HIGH — Gemma-exclusion, PEFT mismatch, 4.x/5.x path, Swin/ViT stride, split leakage are concrete and source-grounded.
- SigLIP numeric config: MEDIUM — published values, unverified live this session (A1).

**Research date:** 2026-05-15
**Valid until:** ~2026-06-15 (transformers/timm move fast — re-verify the PaliGemma submodule path and SigLIP config if the plan slips >30 days or the transformers pin changes).
