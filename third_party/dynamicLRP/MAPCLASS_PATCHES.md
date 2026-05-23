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

**Precise remaining task**: surgical mixed precision — do ONLY the
numerically-sensitive relevance steps (`s = r / (z + sign*ε)` and the relevance
recombination) in fp32 inside each prop fcn, keeping the bulk/retained tensors in
bf16. That keeps the ~7.8 GB footprint while restoring relevance conservation.
Alternatives: a larger GPU (plain fp32 just works) or accept the IG baseline.
