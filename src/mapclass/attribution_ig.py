"""Captum Integrated Gradients baseline — query-driven attribution WITHOUT the
dynamicLRP engine.

WHY THIS EXISTS: for the text-conditioned encoders (SigLIP-2, CLIP) dynamicLRP
cannot traverse the autograd graph (the ``split_with_sizes`` / ``Expand``
op-coverage gaps recorded in ``attribution.py`` / STATE.md). Integrated
Gradients is a trivially-correct, *query-conditioned* attribution that needs
nothing from the engine: it integrates the gradient of the image-text
similarity logit w.r.t. the input pixels along a straight path from a zero
baseline to the real image. NO engine hacks — the vendored dynamicLRP
(``VENDOR_SHA`` frozen) is never touched.

The returned ``AttributionResult`` is shape-compatible with
``attribution.attribute`` (a signed ``(1,3,H,W)`` relevance), so the SAME
``overlay.to_patch_grid`` + ``overlay.composite`` render path is reused
unchanged. The attribution is signed (gradient × input integral), which the
D-09 zero-centered overlay already handles.

Target is supplied the SAME way as the dynamicLRP path: a ``target_fn(model,
output, forward_inputs) -> scalar`` (e.g. ``output.logits_per_image[0, 0]`` for
SigLIP-2 / CLIP). The query enters through the fixed text inputs in
``forward_inputs``, so the heatmap is genuinely query-driven.
"""

from __future__ import annotations

from mapclass.attribution import AttributionResult  # reuse the dataclass

# IG path resolution + memory knobs. n_steps trades fidelity for compute (raise
# it for a smoother/denser map) and barely affects peak VRAM. Keep
# internal_batch_size = 1: the bundled target_fns index sample 0 (e.g.
# logits_per_image[0, 0]) and PaliGemma is a fused VLM whose text holds exactly
# one image's worth of <image> tokens — so >1 either mis-attributes or raises
# "Number of images does not match ... special image tokens". ibs=1 is also what
# fits the L4 for every model (SigLIP-2 so400m ~6.6 GB; PaliGemma-3B OOMs at >1).
_DEFAULT_STEPS = 32
_DEFAULT_INTERNAL_BS = 1


def attribute_ig(
    model,
    forward_inputs,
    img_tensor=None,
    target_fn=None,
    n_steps: int = _DEFAULT_STEPS,
    internal_batch_size: int = _DEFAULT_INTERNAL_BS,
) -> AttributionResult:
    """Integrated Gradients of ``target_fn`` w.r.t. the input pixels.

    Signature mirrors ``attribution.attribute`` so the notebook can swap the two
    by a ``METHOD`` switch. ``forward_inputs`` is the model forward kwargs dict
    (``_``-prefixed keys are stripped from the actual forward); ``img_tensor`` is
    the ``pixel_values`` object IG attributes against; ``target_fn`` builds the
    query-conditioned scalar from the model output.
    """
    import gc

    import torch
    from captum.attr import IntegratedGradients

    if img_tensor is None:
        img_tensor = forward_inputs["pixel_values"]
    if target_fn is None:
        raise ValueError("attribute_ig requires a target_fn (query-conditioned).")

    # Forward kwargs minus the pixel tensor (IG supplies that) and any private
    # "_"-prefixed carriers (e.g. "_query") that are not real model kwargs.
    base_inputs = {
        k: v for k, v in forward_inputs.items()
        if not k.startswith("_") and k != "pixel_values"
    }

    on_cuda = torch.cuda.is_available()
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()

    def forward_fn(pixels):
        out = model(pixel_values=pixels, **base_inputs)
        # (1,) so captum treats it as a single-output, single-sample target.
        return target_fn(model, out, forward_inputs).reshape(1)

    # Read the similarity at the true input (no grad needed for the scalar).
    with torch.no_grad():
        similarity = float(forward_fn(img_tensor).reshape(-1)[0].item())

    relevance = None
    try:
        ig = IntegratedGradients(forward_fn)
        baseline = torch.zeros_like(img_tensor)
        attributions = ig.attribute(
            img_tensor,
            baselines=baseline,
            n_steps=n_steps,
            internal_batch_size=internal_batch_size,
        )
        # Detach + clone off the graph so nothing keeps activations alive.
        relevance = attributions.detach().clone()
    finally:
        gc.collect()
        if on_cuda:
            torch.cuda.empty_cache()

    peak = int(torch.cuda.max_memory_allocated()) if on_cuda else None
    return AttributionResult(
        relevance=relevance,
        target_form=f"ig_{n_steps}step",
        peak_vram_bytes=peak,
        similarity=similarity,
    )
