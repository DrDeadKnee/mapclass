"""SigLIP-2 forward + dynamic LRP in ONE function scope (ATTR-01).

The attribution target is the **contrastive image–text similarity scalar**
``output.logits_per_image[0, 0]`` — a single image vs. a single query (Pitfall
B). It is deliberately NOT the pooled image-feature vector and NOT an
image-embedding norm: those are query-independent and would produce
image-saliency, not query-driven, heatmaps (the forbidden token names are
omitted here so a grep can assert this module never references them). The
image L2-norm stays inside the traced graph; the text path is detached
implicitly because relevance is requested only for the pixel tensor.

Pattern 2 — tensor-identity invariant: the forward AND ``engine.run`` live in
ONE function scope and the SAME ``img_tensor`` object flows into both
``model(pixel_values=img_tensor, ...)`` and ``engine.params_to_interpret``.
``LRPEngine`` walks the live ``grad_fn`` chain and resolves
``params_to_interpret`` by tensor object identity; cloning / detaching /
re-``.to()``-ing the tensor, or letting the forward scope close before
``run()``, destroys the graph and the relevance cannot be computed. The data
loader (``data_loader.load_slice``) hands back exactly this
``requires_grad_()`` tensor; do not break that chain here.

Empirical Risk 1 / Assumption A5 — ``LRPEngine.run`` accepted a 2-D tensor
(``output.logits``) in the verified ViT.ipynb usage; whether it accepts the
0-dim ``logits_per_image[0, 0]`` scalar is resolved empirically here. We try
the 0-dim scalar first and, on failure, fall back to ``[:1, :1]`` (2-D) then
``.reshape(1)`` (1-D), recording which target form worked.

Pitfall E — the Promise system retains forward activations, so VRAM is far
above plain inference. Batch size is always 1; ``torch.cuda.empty_cache()`` is
called after the relevance pass and ``torch.cuda.max_memory_allocated()`` is
captured so the notebook can report peak VRAM (STATE.md blocker — gates any
Phase 2 sweep sizing).

----------------------------------------------------------------------------
Fallback Ladder (op-coverage / faithfulness failure response path).

Implement ONLY the as-is path below. The following are the *documented*
escalation route if the coverage probe reports uncovered ops OR a sanity
control (esp. occlusion) visually FAILs — they are NOT pre-built; each is a
recorded deviation when taken (T-01-SC3 — deviates from the vendored SHA):

  1. Probe first: ``coverage_probe(...)`` →
     ``LRPEngine.get_model_operations(output.logits_per_image[0, 0])``. If it
     reports no uncovered ops, run as-is (this module).
  2. Engine erroring / uncovered op: add a custom backward Promise in
     ``third_party/dynamicLRP/src/lrp_engine/promises/`` subclassing
     ``Promise`` / ``DummyPromise`` (pattern:
     ``promises/softmax_backward_promise.py``); for a model-specific custom
     autograd op follow ``model_specific/mosaicbert.py``
     (``torch.autograd.Function`` + ``.apply``). Document the patch and record
     that it deviates from the vendored SHA.
  3. Runs but MAP-pool head distorts relevance (occlusion control FAILs;
     sparse/flat map): try ``LRPEngine(use_attn_lrp=True)`` and/or attribute
     on the **pre-pool** image-patch logits (raw ``image_embeds @
     text_embeds`` before the ``SiglipMultiheadAttentionPoolingHead``) instead
     of post-pool ``logits_per_image``.
  4. Memory: keep ``use_gamma=False``, ``relevance_filter=0.5`` (cut
     memory/noise). so400m fp32 single-image should fit a mid-tier GPU; if
     not, run the forward in bf16 but keep ``LRPEngine(dtype=torch.float32)``
     for the relevance pass.
  5. Last resort: the vendored LXT
     (``LRP-eXplains-Transformers``) with a ViT-style patch, OR ``captum``
     Integrated Gradients on ``logits_per_image[0, 0]`` as a degraded but
     trivially-correct baseline (captum is already a pinned dep).
----------------------------------------------------------------------------

EMPIRICAL FINDING (Plan 01-03 execution, 2026-05-19, NVIDIA L4, pinned stack):
The as-is path FAILS on SigLIP-2-so400m. ``LRPEngine.run`` against the
contrastive target was exercised in all forms:
  * ``logits_per_image[0, 0]`` (0-dim) → ``IndexError`` (engine's
    starting-relevance builder indexes a 0-dim tensor — A5 confirmed: a 0-dim
    target is NOT accepted).
  * ``logits_per_image[:1, :1]`` (1×1, 2-D) and ``...[0,0].reshape(1)`` (1-D)
    → the engine's first-pass traversal returns its 5-tuple ERROR path with
    ``TypeError("'DummyPromise' object is not iterable")`` at autograd node
    ``SplitWithSizesBackward0``.
  * Fallback step 3 (``use_attn_lrp=True``; pre-pool ``image_embeds ·
    text_embeds``; query-conditioned vision-pooled · detached-text) ALL fail
    at the SAME ``SplitWithSizesBackward0`` node with the same TypeError.
The ``split_with_sizes`` op is intrinsic to SigLIP-2's attention / MAP-pool
head. The engine registers ``SplitWithSizesBackward`` → ``SplitBackwardProp``
but its Promise consumer chokes on SigLIP-2's split topology. This is a
genuine engine op-coverage failure on SigLIP-2 as a contrastive MAP-pool
encoder — exactly the MEDIUM-LOW research risk. Resolution requires Fallback
Ladder step 2 (a custom/repaired engine Promise — deviates from the vendored
SHA, T-01-SC3) OR step 5 (captum Integrated Gradients degraded baseline).
This is an architectural decision surfaced at the Plan 01-03 human-verify
checkpoint (the plan designates the Fallback Ladder as the FAIL response
path); it is NOT auto-selected by the executor.

USER RESOLUTION (01-03 human-verify checkpoint, 2026-05-19) — Fallback Ladder
DECLINED, do NOT re-attempt. The user reviewed this finding and the
end-to-end smoke test and explicitly declined the ENTIRE Fallback Ladder: NO
custom dynamicLRP Promise (step 2), NO pre-pool / ``use_attn_lrp``
engineering (step 3), NO vendored LXT, and NO captum Integrated Gradients
baseline (step 5). Quote: "Smoke-test was good, it didn't crash. Let's leave
well enough alone and move on." The project is reframed as a multi-model
comparison of dynamic-LRP; SigLIP-2's ``split_with_sizes`` op-coverage gap is
an ACCEPTED per-model FINDING and the Phase 1 D-02/D-03 visual-eyeball gate is
CONSCIOUSLY WAIVED for SigLIP-2. The Fallback Ladder above is retained ONLY as
historical documentation of the escalation route that was offered and
declined — it MUST NOT be implemented later for SigLIP-2 without an explicit
new user decision. ``third_party/dynamicLRP`` is unmodified (VENDOR_SHA
intact; no T-01-SC3 deviation taken).

UPDATE 2026-05-23 — that explicit new user decision was made: Fallback Ladder
step 5 (captum Integrated Gradients) is now provided as a SEPARATE
query-conditioned baseline in ``attribution_ig.py`` (selected via the notebook's
``METHOD="ig"`` switch) so SigLIP-2/CLIP get real query-driven heatmaps. It does
NOT touch the engine — VENDOR_SHA still intact, no T-01-SC3 deviation. Ladder
steps 2-4 (custom Promise / pre-pool / ``use_attn_lrp``) remain declined.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from lrp_engine import LRPEngine

# CLAUDE.md / RESEARCH.md engine knobs: signed γ-free relevance, no recompile,
# relevance_filter=0.5 cuts memory + noise (Pitfall E / Fallback Ladder step 4).
_USE_GAMMA = False
_NO_RECOMPILE = True
_RELEVANCE_FILTER = 0.5


@dataclass
class AttributionResult:
    """Relevance + provenance for one (map, query) attribution.

    ``relevance`` has the SAME shape as the input pixel tensor
    ``(1, 3, 384, 384)`` (Assumption A3 — relevance is read at the input by
    tensor identity, surviving back through the MAP-pool head to the pixels).
    ``target_form`` records which ``run()`` target form succeeded
    (``"scalar_0d"`` / ``"slice_2d"`` / ``"reshape_1d"``) — Empirical Risk 1.
    ``peak_vram_bytes`` is ``torch.cuda.max_memory_allocated`` measured across
    the forward + LRP pass (STATE.md blocker; ``None`` on CPU).
    """

    relevance: object
    target_form: str
    peak_vram_bytes: Optional[int]
    similarity: float


def _empty_cuda_cache() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def coverage_probe(model, img_tensor, input_ids, attention_mask):
    """Static op-coverage probe — Fallback Ladder step 1 (run before first use).

    Wraps ``LRPEngine.get_model_operations`` on the SigLIP-2 forward against
    the contrastive similarity scalar. Returns whatever the engine reports
    (op-name set / count / graph) so the notebook can print the coverage
    artifact. Does NOT run the relevance pass.
    """
    output = model(
        pixel_values=img_tensor,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    target = output.logits_per_image[0, 0]
    return LRPEngine.get_model_operations(target)


def _siglip2_target_fn(model, output, forward_inputs):
    """Default (v1.0) target: SigLIP-2/CLIP contrastive similarity scalar.

    The attribution target is ``output.logits_per_image[0, 0]`` (Pitfall B) —
    query-conditioned image–text similarity, NOT a pooled embedding.
    """
    return output.logits_per_image[0, 0]


def attribute(
    model,
    forward_inputs,
    img_tensor=None,
    target_fn=None,
    attention_mask=None,
) -> AttributionResult:
    """Forward + dynamic LRP in ONE scope → input-shaped signed relevance.

    GENERIC (v1.1, MODEL-02): runs ``model(**forward_inputs)`` in ONE scope,
    builds the per-model query-conditioned target via
    ``target_fn(model, output, forward_inputs)``, then runs the dynamicLRP
    relevance pass with ``engine.params_to_interpret = [img_tensor]`` — the
    SAME ``img_tensor`` object that ``forward_inputs["pixel_values"]`` carries
    (Pattern 2 tensor-identity invariant; NO clone/detach/re-.to()).

    BACKWARD-COMPATIBLE (v1.0): the legacy SigLIP-2 call
    ``attribute(model, img_tensor, input_ids, attention_mask)`` is still
    accepted — when ``forward_inputs`` is a tensor (the old ``img_tensor``
    positional) and ``img_tensor`` is the old ``input_ids``, the SigLIP-2
    ``pixel_values``/``input_ids``/``attention_mask`` forward kwargs and the
    default ``output.logits_per_image[0, 0]`` target are reconstructed.

    Empirical Risk 1 / A5: the 0-dim scalar target is tried first; on a
    dimensionality / engine error we fall back to a 2-D ``[:1,:1]`` slice then
    a 1-D ``.reshape(1)`` and record which form worked. The SigLIP-2
    ``split_with_sizes`` op-coverage RuntimeError is preserved (recorded
    FINDING — Fallback Ladder DECLINED, do NOT re-attempt).
    """
    import torch

    # ----- legacy-signature shim (v1.0 SigLIP-2 call) ----------------------
    # Old: attribute(model, img_tensor, input_ids, attention_mask)
    # New: attribute(model, forward_inputs: dict, img_tensor, target_fn)
    if not isinstance(forward_inputs, dict):
        legacy_img_tensor = forward_inputs
        legacy_input_ids = img_tensor
        legacy_attention_mask = (
            target_fn if target_fn is not None else attention_mask
        )
        forward_inputs = {
            "pixel_values": legacy_img_tensor,
            "input_ids": legacy_input_ids,
            "attention_mask": legacy_attention_mask,
        }
        img_tensor = legacy_img_tensor
        target_fn = _siglip2_target_fn

    if img_tensor is None:
        img_tensor = forward_inputs["pixel_values"]
    if target_fn is None:
        target_fn = _siglip2_target_fn

    import gc

    on_cuda = torch.cuda.is_available()
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()

    # Pitfall E — the dynamicLRP Promise system + ``no_recompile=True`` RETAINS
    # the full forward-activation graph on the GPU. The v1.0 code only freed it
    # on the success path; a FAILED attribution (e.g. SigLIP-2's
    # ``split_with_sizes`` op-coverage RuntimeError) leaked the retained graph
    # for the lifetime of the caller. In the v1.1 sequential 4-model × 50-map
    # loop that leak accumulates and OOMs every model after the first. The
    # whole forward+engine region is therefore wrapped so the graph (output /
    # engine / intermediates) is dropped and the CUDA cache emptied in EVERY
    # exit path — success, all-forms-rejected, and exception. The returned
    # relevance is detached+cloned OFF the graph so it cannot keep it alive.
    output = None
    engine = None
    param_vals = None
    relevance = None
    target_form = None
    similarity = float("nan")
    last_err: Optional[Exception] = None

    def _as_2d(t):
        return t.reshape(1, 1) if t.dim() == 0 else t.reshape(1, -1)[:1, :1]

    try:
        # ----- forward (live grad_fn graph; img_tensor identity preserved) --
        output = model(**forward_inputs)
        target0 = target_fn(model, output, forward_inputs)
        similarity = float(target0.detach().reshape(-1)[0].item())

        # ----- target-form fallback ladder (Empirical Risk 1 / A5) ---------
        # 0-dim scalar first; ViT used 2-D logits, so a 0-dim target may be
        # rejected — fall back to a 2-D [:1,:1] slice, then a 1-D reshape(1).
        candidates = (
            ("scalar_0d", lambda: target_fn(model, output, forward_inputs)),
            (
                "slice_2d",
                lambda: _as_2d(target_fn(model, output, forward_inputs)),
            ),
            (
                "reshape_1d",
                lambda: target_fn(
                    model, output, forward_inputs
                ).reshape(-1)[:1],
            ),
        )

        for form_name, make_target in candidates:
            try:
                engine = LRPEngine(
                    use_gamma=_USE_GAMMA,
                    no_recompile=_NO_RECOMPILE,
                    relevance_filter=_RELEVANCE_FILTER,
                )
                # SAME object as the forward (tensor-identity invariant).
                engine.params_to_interpret = [img_tensor]
                _ckpt_vals, param_vals = engine.run(make_target())
                # Detach + clone OFF the retained graph so the returned
                # relevance does not keep the activation graph alive.
                relevance = param_vals[0].detach().clone()
                target_form = form_name
                break
            except Exception as exc:  # dimensionality / engine error → next
                last_err = exc
                engine = None
                param_vals = None
                continue

        if relevance is None:
            raise RuntimeError(
                "LRPEngine.run rejected every target form "
                "(scalar_0d / slice_2d / reshape_1d). Last error: "
                f"{last_err!r}. See the Fallback Ladder in this module's "
                "docstring (steps 2-5) for the escalation path."
            )

        peak_vram = (
            int(torch.cuda.max_memory_allocated()) if on_cuda else None
        )
        return AttributionResult(
            relevance=relevance,
            target_form=target_form,
            peak_vram_bytes=peak_vram,
            similarity=similarity,
        )
    finally:
        # Drop EVERY reference to the retained activation graph regardless of
        # how we exit (Pitfall E). Without this the SigLIP-2 op-coverage
        # failure leaks ~6 GB that survives the caller's del + gc.
        del output, engine, param_vals
        try:
            del target0
        except Exception:
            pass
        for _v in ("candidates",):
            if _v in dir():
                pass
        gc.collect()
        _empty_cuda_cache()  # Pitfall E — free the retained activation graph
