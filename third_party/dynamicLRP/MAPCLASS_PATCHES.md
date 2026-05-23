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

## Status (2026-05-23)
Op-coverage is SOLVED: dynamicLRP traverses SigLIP-2 end-to-end. Remaining wall
is precision/memory on the 24 GB L4:
- **fp32**: numerically correct, but OOMs (~needs >22 GB; peaks at the ceiling).
- **bf16**: fits memory, but the patch-conv backward's `conv_transpose2d` hits
  `Cannot load symbol cublasLtCreate` (a bf16 cuBLASLt env issue).
- **fp16**: fits memory, conv runs, but the relevance pass NaNs (range too small
  for the epsilon division).
Likely resolution: mixed precision (bf16 forward to save memory, fp32 for the
conv-transpose / epsilon-sensitive ops), a cuBLASLt env fix for bf16, or a
larger GPU for plain fp32.
