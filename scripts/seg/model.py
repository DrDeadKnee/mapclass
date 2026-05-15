"""
SegModel assembly: backbone + decoder + heads; Variant A and Variant B wiring (D-03a).

Strategy:
  - ``SegModelVariantA``: fully-frozen backbone; 12 prior channels injected via a
    small trainable prior-encoder concatenated into the decoder at working resolution
    (Pattern 4 / D-03a Variant A).  Backbone-agnostic.
  - ``SegModelVariantB``: all three backbones' patch-embed widened to 15 channels
    (RGB + 12 prior); widened patch-embed trainable; prior enters at the input
    (D-03a Variant B / D-06a construction-only).
  Both variants share the SAME shared conv decoder (D-01/D-02), recursive c2f
  orchestration wiring (D-04), and unified Backbone protocol (D-05).

Phase 3 constructs these models; Phase 4 trains + compares A vs B.
No training loop, no optimizer, no .backward() here (D-06/D-06a).

Requires:
  seg.backbones — Backbone protocol + SigLIP/DINOv2/Swin implementations
  seg.decoder   — UPerNet-style PPM+FPN conv decoder (plan 03-04)
  seg.heads     — two thin 1×1-conv task heads (plan 03-04)

Covered decisions: D-02, D-03a, D-05, D-06, D-06a.
"""

from __future__ import annotations

import torch.nn as nn

from seg.backbones import Backbone


# ---------------------------------------------------------------------------
# SegModelVariantA — fully-frozen backbone, decoder-level prior injection
# ---------------------------------------------------------------------------

class SegModelVariantA(nn.Module):
    """
    Variant A seg model stub (D-03a).

    Fully frozen backbone; the 12 prior channels enter via a small trainable
    prior-encoder that is concatenated/summed into the decoder at its working
    resolution.  Backbone-agnostic (any Backbone protocol implementation plugs
    in without changes to this class).

    Full decoder + heads implementation lands in plan 03-04.  This stub provides
    the class name that plan 03-02's Gemma-exclusion test gates on, so that
    ``test_no_gemma_variant_a`` can import and verify the backbone in isolation.
    """

    def __init__(self, backbone: Backbone) -> None:  # type: ignore[valid-type]
        super().__init__()
        # Backbone held as a submodule so named_parameters() / modules()
        # traversal reaches it (PHASE-03 SC#2 assertion).
        self.backbone = backbone  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# SegModelVariantB — widened patch-embed, input-level prior injection (stub)
# ---------------------------------------------------------------------------

class SegModelVariantB(nn.Module):
    """
    Variant B seg model stub (D-03a / D-06a).

    All three backbones' first conv/patch-embed widened to 15 channels (RGB + 12
    prior); the widened patch-embed is trainable; prior enters at the image input.
    Full decoder + heads land in plan 03-04.
    """

    def __init__(self, backbone: Backbone) -> None:  # type: ignore[valid-type]
        super().__init__()
        self.backbone = backbone  # type: ignore[assignment]
