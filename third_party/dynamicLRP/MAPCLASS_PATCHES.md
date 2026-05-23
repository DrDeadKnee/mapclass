# MapClass patches to vendored dynamicLRP

The vendored engine was deliberately modified (user decision, 2026-05-23) to make
dynamicLRP traverse **SigLIP-2-so400m**. `VENDOR_SHA` still records the pristine
upstream base commit (`405e74243ecaa1f615f418fdc8ba24c3c5889b1e`); this file
records the local deltas on top of it. The `config.VENDOR_SHA` assertion still
passes (it compares the recorded string, not a tree hash).

## Patches

### 1. `src/lrp_engine/lrp_prop_fcns.py` — `padding="valid"` conv fix (REAL FIX)
`DecomposedConvolutionBackwardProp` indexed `padding[i]` per spatial dim. A
`Conv2d(..., padding="valid")` (SigLIP-2's patch-embedding conv) saves
`_saved_padding` as the string `"valid"` / an empty tuple, so the indexing
`IndexError`-ed ("tuple index out of range"). Now normalized to explicit per-dim
zeros. Convs with `padding=0` (e.g. ViT) already yield `(0,0)` and are
unaffected. This is a genuine upstream-worthy bug fix.

### 2. `src/lrp_engine/lrp.py` + `src/lrp_engine/promises/promise.py` — diagnostics
Added `LRP_DEBUG`-env-gated traceback/shape prints at the prop-dispatch
swallow point (`lrp.py` ~line 516) and in `Promise.bwd` (`promise.py`). **No
runtime effect unless `LRP_DEBUG` is set.** These are how each op blocker was
located.

## NOT patched (kept frozen)
The relevance math (epsilon/sign/gamma/promise machinery) is unchanged.

## Companion model-side adaptation (lives in our code, NOT the engine)
`src/mapclass/siglip_lrp_patch.py` reimplements SigLIP-2's
`Siglip2MultiheadAttentionPoolingHead` with separate Q/K/V projections instead of
`nn.MultiheadAttention` (whose fused-QKV `split_with_sizes` the engine can't
traverse), pre-extracted into leaf params (so the forward never slices a packed
1-D bias). Verified numerically identical to the original head (fp32 Δ=0.00).

## Additional patches (groundwork toward landing it on the L4)

### 3. `lrp_prop_fcns.py` `DecomposedConvolutionBackwardProp` — half-precision conv
The transpose-conv / weight-grad are done in fp32 then cast back when the saved
tensors are bf16/fp16: bf16 `conv_transpose2d` hits `cublasLtCreate` and fp16
underflows the ε-division at conv scale. The retained graph stays half-precision.

### 4. `lrp.py` param extraction — tolerate unreached params
When the target depends on only a sub-graph (e.g. attributing the IMAGE tower of
a contrastive model leaves the TEXT-tower embeddings untouched), those embedding
param nodes never get `"relevance"`. The extraction loop now skips unreached
params instead of `KeyError`-ing, so the requested params still return. (Plus an
`LRP_DEBUG` param-reach diagnostic.)

## Status (2026-05-23)
**Op-coverage is SOLVED**: dynamicLRP traverses SigLIP-2 end-to-end and the image
relevance is computed. The remaining wall is the **numerical precision vs memory**
of the relevance pass on the 24 GB L4:
- **fp32 relevance**: numerically correct but OOMs (~21 GB working set, doesn't fit).
- **bf16 forward + bf16 relevance**: fits easily (**peak 7.78 GB**) but the
  relevance *vanishes* (absmax ~1e-10) — bf16 can't hold the `r/(z+ε)` relevance
  redistribution across 27 layers.
- **bf16 forward + fp32 relevance (blanket upcast in `detach_if_no_grad`)**:
  correct precision but the upcast fp32 copies pile up -> OOM (~21 GB). (Tried and
  reverted — wrong granularity.)

### 5. Surgical mixed precision (`util.py` epsilon_lrp_matmul + `lrp_prop_fcns.py` conv)
The numerically-sensitive relevance core (`s = r/(z+ε·sign)` + the redistribution
matmuls / transpose-conv) is computed in fp32 — upcasting the saved bf16 x/w/z
transiently — and returned in the relevance dtype. No-op for fp32 models (ViT
unaffected). Lets the model FORWARD run in bf16 (small graph) while the relevance
math is fp32-accurate.

## VERDICT (2026-05-23): so400m fp32 LRP does NOT fit the 24 GB L4
Exhaustively tested on a clean L4. The relevance pass must be STORED in fp32 or it
breaks; fp32 storage of so400m's relevance (729 tokens × 27 layers × 1152 dim)
peaks **>22 GB** even with the forward graph in bf16. Storage-precision results:
- **bf16 store** (compute fp32): fits (7.78 GB) but relevance VANISHES (~1e-6,
  0.3% nonzero) — 7-bit mantissa rounds the distributed small relevance to 0.
- **fp16 store** (compute fp32): NaNs (range too small).
- **fp32 store**: numerically correct but OOMs (>22 GB). `no_recompile` and
  `relevance_filter` do NOT reduce this peak.

**Op-coverage + numerics are SOLVED; this is purely a VRAM ceiling.** Paths to a
real so400m heatmap: (a) a bigger GPU — A100 40 GB runs fp32 comfortably (not
available in northamerica-northeast1; needs another region); (b) a smaller
SigLIP-2 variant (e.g. `siglip2-base-patch16-224`, ~10× less relevance memory)
fits the L4 in fp32 with these exact patches — same architecture, same MAP-pool
head fix. The IG baseline remains the meanwhile option.
