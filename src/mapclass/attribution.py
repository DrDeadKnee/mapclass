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


def attribute(model, img_tensor, input_ids, attention_mask) -> AttributionResult:
    """Forward + dynamic LRP in ONE scope → input-shaped signed relevance.

    The SAME ``img_tensor`` object flows into both the forward and
    ``engine.params_to_interpret`` (Pattern 2). The attribution target is
    ``output.logits_per_image[0, 0]`` (Pitfall B). Empirical Risk 1: the 0-dim
    scalar is tried first; on a dimensionality error we fall back to ``[:1,
    :1]`` then ``.reshape(1)`` and record which form worked.
    """
    import torch

    on_cuda = torch.cuda.is_available()
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()

    # ----- forward (live grad_fn graph; img_tensor identity preserved) -----
    output = model(
        pixel_values=img_tensor,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    sim_scalar = output.logits_per_image[0, 0]
    similarity = float(sim_scalar.detach().item())

    # ----- target-form fallback ladder (Empirical Risk 1 / Assumption A5) ---
    # 0-dim scalar first; the ViT path used 2-D logits, so a 0-dim target may
    # be rejected — fall back to a 2-D [:1,:1] slice, then a 1-D reshape(1).
    candidates = (
        ("scalar_0d", lambda: output.logits_per_image[0, 0]),
        ("slice_2d", lambda: output.logits_per_image[:1, :1]),
        ("reshape_1d", lambda: output.logits_per_image[0, 0].reshape(1)),
    )

    relevance = None
    target_form = None
    last_err: Optional[Exception] = None
    for form_name, make_target in candidates:
        try:
            engine = LRPEngine(
                use_gamma=_USE_GAMMA,
                no_recompile=_NO_RECOMPILE,
                relevance_filter=_RELEVANCE_FILTER,
            )
            engine.params_to_interpret = [img_tensor]  # SAME object as forward
            _ckpt_vals, param_vals = engine.run(make_target())
            relevance = param_vals[0]  # positionally matches params_to_interpret
            target_form = form_name
            break
        except Exception as exc:  # dimensionality / engine error → next form
            last_err = exc
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
    _empty_cuda_cache()  # Pitfall E — free the retained activation graph

    return AttributionResult(
        relevance=relevance,
        target_form=target_form,
        peak_vram_bytes=peak_vram,
        similarity=similarity,
    )
