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

Trust model:
  T-03-13: Per-variant parameter-graph assertion that no ``language_model`` param
           name and no ``gemma`` module-type is reachable from EITHER variant
           (PHASE-03 SC#2 — Gemma exclusion).
"""

from __future__ import annotations

import types
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from seg.backbones import Backbone, SiglipBackbone, Dinov2Backbone, SwinBackbone
from seg.decoder import SegDecoder
from seg.heads import LandCoverHead, TopographyHead


# ---------------------------------------------------------------------------
# Factory helper: build the backbone by name
# ---------------------------------------------------------------------------

def _build_backbone(
    backbone_name: str,
    variant_b: bool,
    stub_config: types.SimpleNamespace | None = None,
    model_id: str | None = None,
    adapter_dir: str | None = None,
) -> nn.Module:
    """
    Construct a backbone by name with the given variant_b flag.

    Parameters
    ----------
    backbone_name:
        One of ``"siglip"``, ``"dinov2"``, ``"swin"``.
    variant_b:
        If ``True``, widen patch-embed to 15 channels (D-03a Variant B).
    stub_config:
        Offline namespace for SigLIP (no checkpoint load).
    model_id:
        HuggingFace model ID for SigLIP (ignored for DINOv2/Swin).
    adapter_dir:
        Phase-1 PEFT adapter directory (SigLIP only).
    """
    name = backbone_name.lower()
    if name == "siglip":
        return SiglipBackbone(
            model_id=model_id,
            adapter_dir=adapter_dir,
            stub_config=stub_config,
            variant_b=variant_b,
        )
    elif name == "dinov2":
        return Dinov2Backbone(variant_b=variant_b)
    elif name == "swin":
        return SwinBackbone(variant_b=variant_b)
    else:
        raise ValueError(
            f"Unknown backbone_name '{backbone_name}'. "
            "Choose one of: 'siglip', 'dinov2', 'swin'."
        )


# ---------------------------------------------------------------------------
# PriorEncoder (Variant A) — small trainable module mapping the 12-ch prior to
# the decoder working resolution.  Defined here so it's accessible from model.py
# and importable from seg.model if needed; also re-exported from seg.recursive.
# ---------------------------------------------------------------------------

class PriorEncoder(nn.Module):
    """
    Variant A trainable prior-encoder (D-03a Variant A).

    Maps a 12-channel raw softmax-probability prior (B, 12, H, W) to a
    (B, out_channels, target_h, target_w) feature map that is summed/
    concatenated into the shared decoder at its working resolution.

    Design (D-01 Claude's Discretion):
      - Two 3×3 conv layers with GroupNorm(4)+ReLU (4 groups used because
        12 channels is divisible by 4 but not 32; GN avoids batch-size-1 issues).
      - F.interpolate to target_size (bilinear, align_corners=False) so the
        encoder adapts to any tile resolution without further changes.
      - The whole module is trainable; the backbone and decoder remain frozen
        for Variant A (D-06 / D-06a).

    Parameters
    ----------
    in_channels:
        Number of prior channels (always 12 = 9 LC + 3 topo).
    out_channels:
        Decoder working channel width (matches ``SegDecoder.out_channels``).
    target_size:
        (h, w) spatial size to resize to before injection.
    """

    _GN_GROUPS: int = 4   # 12 channels → 4 groups (12 % 4 == 0)

    def __init__(
        self,
        in_channels: int = 12,
        out_channels: int = 256,
        target_size: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        mid = max(out_channels // 2, in_channels)
        self._conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(self._GN_GROUPS, mid),
            nn.ReLU(inplace=True),
        )
        self._conv2 = nn.Sequential(
            nn.Conv2d(mid, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_channels), out_channels),
            nn.ReLU(inplace=True),
        )
        self._target_size = target_size

    def forward(self, prior: Tensor) -> Tensor:
        """
        Parameters
        ----------
        prior:
            (B, 12, H, W) raw softmax probabilities.

        Returns
        -------
        Tensor
            (B, out_channels, target_h, target_w) feature ready to be summed
            into the decoder at its working resolution.
        """
        x = self._conv1(prior)
        x = self._conv2(x)
        if self._target_size is not None:
            if x.shape[-2:] != torch.Size(self._target_size):
                x = F.interpolate(
                    x,
                    size=self._target_size,
                    mode="bilinear",
                    align_corners=False,
                )
        return x


# ---------------------------------------------------------------------------
# SegModelVariantA — fully-frozen backbone, decoder-level prior injection
# ---------------------------------------------------------------------------

class SegModelVariantA(nn.Module):
    """
    Variant A seg model (D-03a).

    Fully frozen backbone (SigLIP / DINOv2 / Swin); the 12 prior channels enter
    via a small trainable ``PriorEncoder`` that maps the prior to the decoder
    working resolution and is summed into the decoder output before the task
    heads.  Backbone-agnostic.

    ``forward(rgb, prior)`` accepts:
      - ``rgb``:   (B, 3, H, W) image tile (float32 in [0, 1]).
      - ``prior``: (B, 12, H, W) 12-channel raw softmax probabilities
                   (9 LC + 3 topo). All-zero is the valid cold-start.

    Returns ``(lc_logits, topo_logits)`` — raw logits at (B, 9, H, W) and
    (B, 3, H, W) at the full tile resolution H×W.  NO softmax (heads output raw).

    Construction-only (D-06 / D-06a): no optimizer, no .backward(), no training.

    Gemma exclusion (PHASE-03 SC#2 / T-03-13): the backbone submodule holds only
    the vision tower (SigLIP case) — no ``language_model`` parameter is reachable
    from this module.
    """

    def __init__(
        self,
        backbone_name: str = "siglip",
        stub_config: types.SimpleNamespace | None = None,
        model_id: str | None = None,
        adapter_dir: str | None = None,
        tile_size: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        backbone_name:
            ``"siglip"`` (primary), ``"dinov2"`` or ``"swin"`` (EVAL-03 benchmarks).
        stub_config:
            Offline namespace for SigLIP (no HF checkpoint required).
        model_id:
            HuggingFace model ID (SigLIP only; default ``google/paligemma-3b-pt-224``).
        adapter_dir:
            Phase-1 PEFT adapter directory (SigLIP only; None = base model).
        tile_size:
            Override default tile size for the decoder (inferred as 224 when
            ``None``; override for 448/896 if needed — the decoder adapts).
            NOTE: SegModelVariantA adapts to any input H×W at forward time,
            so ``tile_size`` only controls the default decoder construction;
            the progressive upsample handles other sizes.
        """
        super().__init__()

        # Build backbone (Variant A = NOT variant_b)
        self.backbone: nn.Module = _build_backbone(
            backbone_name,
            variant_b=False,
            stub_config=stub_config,
            model_id=model_id,
            adapter_dir=adapter_dir,
        )

        # Freeze backbone (D-06 Variant A: fully frozen)
        self.backbone.requires_grad_(False)

        # Build shared decoder and heads (D-02: shared trunk)
        _tile_size = tile_size if tile_size is not None else 224
        self.decoder = SegDecoder(
            feature_strides=self.backbone.feature_strides,   # type: ignore[union-attr]
            feature_channels=self.backbone.feature_channels,  # type: ignore[union-attr]
            tile_size=_tile_size,
        )
        self.lc_head = LandCoverHead(self.decoder.out_channels)
        self.topo_head = TopographyHead(self.decoder.out_channels)

        # Build trainable prior-encoder (D-03a Variant A: inject at decoder working res)
        # We will compute the actual target size at forward-time for flexibility,
        # so no target_size here; PriorEncoder will resize to decoder output if needed.
        self.prior_encoder = PriorEncoder(
            in_channels=12,
            out_channels=self.decoder.out_channels,
            target_size=None,   # resolved dynamically in forward
        )

        # Gemma-exclusion assertion at construction time (T-03-13)
        _assert_no_gemma(self)

    def forward(self, rgb: Tensor, prior: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        rgb:
            (B, 3, H, W) image tile.
        prior:
            (B, 12, H, W) raw softmax prior (zeros at 896 cold-start).

        Returns
        -------
        lc_logits:   (B, 9, H, W) land-cover raw logits.
        topo_logits: (B, 3, H, W) topography raw logits.
        """
        tile_h, tile_w = rgb.shape[-2], rgb.shape[-1]

        # (1) Backbone: frozen; RGB only (3 channels)
        features = self.backbone(rgb)   # list[(B, C, h, w)]

        # (2) Shared decoder: emits (B, C_dec, H, W) at decoder working resolution
        dec_feat = self.decoder(features)   # (B, out_channels, H_dec, W_dec)

        # (3) Prior-encoder (Variant A: decoder-level injection, D-03a):
        #     resize prior to decoder spatial resolution, then encode to same channels
        target_size = (dec_feat.shape[-2], dec_feat.shape[-1])
        prior_resized = (
            F.interpolate(prior, size=target_size, mode="bilinear", align_corners=False)
            if (prior.shape[-2] != target_size[0] or prior.shape[-1] != target_size[1])
            else prior
        )
        pe_out = self.prior_encoder(prior_resized)   # (B, out_channels, H_dec, W_dec)
        # Ensure spatial dims match after conv (convs are padding=1 so same-size)
        if pe_out.shape[-2:] != dec_feat.shape[-2:]:
            pe_out = F.interpolate(
                pe_out, size=(dec_feat.shape[-2], dec_feat.shape[-1]),
                mode="bilinear", align_corners=False,
            )

        # (4) Sum prior features into decoder features (decoder-level injection)
        fused = dec_feat + pe_out   # (B, out_channels, H_dec, W_dec)

        # (5) Upsample fused features to full tile resolution H×W if needed
        if fused.shape[-2] != tile_h or fused.shape[-1] != tile_w:
            fused = F.interpolate(
                fused, size=(tile_h, tile_w), mode="bilinear", align_corners=False
            )

        # (6) Two thin heads (D-02)
        lc_logits = self.lc_head(fused)      # (B, 9, H, W)
        topo_logits = self.topo_head(fused)  # (B, 3, H, W)

        return lc_logits, topo_logits


# ---------------------------------------------------------------------------
# SegModelVariantB — widened patch-embed, input-level prior injection (D-03a)
# ---------------------------------------------------------------------------

class SegModelVariantB(nn.Module):
    """
    Variant B seg model (D-03a / D-06a).

    All three backbones' first conv/patch-embed widened to 15 channels (RGB + 12
    prior); the widened patch-embed is the only trainable parameter (construction-
    only — D-06a); prior enters at the image input level.

    ``forward(x15)`` accepts:
      - ``x15``: (B, 15, H, W) = concat([rgb (3 ch), prior (12 ch)]).
        At 896 cold-start the prior 12 channels are all-zero.

    Returns ``(lc_logits, topo_logits)`` at (B, 9, H, W) and (B, 3, H, W).

    Both Variant A and Variant B SHARE the SAME decoder + heads topology (D-02 /
    EVAL-03 apples-to-apples requirement) — only the backbone input differs.

    Construction-only (D-06a): no optimizer, no .backward(), no training.

    Gemma exclusion (PHASE-03 SC#2 / T-03-13): backbone holds vision tower only.
    """

    def __init__(
        self,
        backbone_name: str = "siglip",
        stub_config: types.SimpleNamespace | None = None,
        model_id: str | None = None,
        adapter_dir: str | None = None,
        tile_size: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        backbone_name:
            ``"siglip"``, ``"dinov2"``, or ``"swin"``.
        stub_config:
            Offline namespace for SigLIP offline shape tests.
        model_id:
            HuggingFace model ID (SigLIP only).
        adapter_dir:
            Phase-1 PEFT adapter dir (SigLIP only; None = base model).
        tile_size:
            Default tile size for decoder construction.
        """
        super().__init__()

        # Build backbone (Variant B = variant_b=True → widened 15-ch patch-embed)
        self.backbone: nn.Module = _build_backbone(
            backbone_name,
            variant_b=True,
            stub_config=stub_config,
            model_id=model_id,
            adapter_dir=adapter_dir,
        )

        # For Variant B, backbone is NOT fully frozen: the widened patch-embed
        # is trainable; everything else is frozen (D-06a / Variant B construction-only).
        # The backbone's _widen_patch_embed already marks the new conv as trainable;
        # all other params were frozen when the base model was loaded.
        # (We do NOT call requires_grad_(False) here — let the backbone's own
        # construction handle the frozen/trainable split.)

        # Build shared decoder and heads (D-02: same topology as Variant A)
        _tile_size = tile_size if tile_size is not None else 224
        self.decoder = SegDecoder(
            feature_strides=self.backbone.feature_strides,   # type: ignore[union-attr]
            feature_channels=self.backbone.feature_channels,  # type: ignore[union-attr]
            tile_size=_tile_size,
        )
        self.lc_head = LandCoverHead(self.decoder.out_channels)
        self.topo_head = TopographyHead(self.decoder.out_channels)

        # No PriorEncoder in Variant B: prior enters at backbone input level
        # (15-ch input → widened patch-embed).

        # Gemma-exclusion assertion at construction time (T-03-13)
        _assert_no_gemma(self)

    def forward(self, x15: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        x15:
            (B, 15, H, W) concatenated input: [rgb (3 ch) | prior (12 ch)].
            At 896 cold-start, prior channels are all-zero.

        Returns
        -------
        lc_logits:   (B, 9, H, W) land-cover raw logits.
        topo_logits: (B, 3, H, W) topography raw logits.
        """
        tile_h, tile_w = x15.shape[-2], x15.shape[-1]

        # (1) Backbone: 15-ch input through widened patch-embed
        features = self.backbone(x15)   # list[(B, C, h, w)]

        # (2) Shared decoder: emits (B, C_dec, H, W) at tile resolution
        dec_feat = self.decoder(features)   # (B, out_channels, H, W)

        # (3) Upsample to full tile resolution H×W if needed
        if dec_feat.shape[-2] != tile_h or dec_feat.shape[-1] != tile_w:
            dec_feat = F.interpolate(
                dec_feat, size=(tile_h, tile_w), mode="bilinear", align_corners=False
            )

        # (4) Two thin heads (D-02)
        lc_logits = self.lc_head(dec_feat)      # (B, 9, H, W)
        topo_logits = self.topo_head(dec_feat)  # (B, 3, H, W)

        return lc_logits, topo_logits


# ---------------------------------------------------------------------------
# SegModel: factory wrapper providing a unified construction interface
# ---------------------------------------------------------------------------

class SegModel(nn.Module):
    """
    Unified factory wrapper for both D-03a variants.

    ``SegModel(variant="A", ...)`` → delegates to ``SegModelVariantA``.
    ``SegModel(variant="B", ...)`` → delegates to ``SegModelVariantB``.

    Provided for convenience; tests that import ``SegModelVariantA`` /
    ``SegModelVariantB`` directly also work (both classes are exported).
    """

    def __new__(  # type: ignore[override]
        cls,
        variant: str = "A",
        **kwargs,
    ) -> "SegModelVariantA | SegModelVariantB":
        """
        Returns a ``SegModelVariantA`` or ``SegModelVariantB`` instance
        (NOT a ``SegModel`` instance).
        """
        v = variant.upper()
        if v == "A":
            return SegModelVariantA(**kwargs)
        elif v == "B":
            return SegModelVariantB(**kwargs)
        else:
            raise ValueError(f"variant must be 'A' or 'B', got {variant!r}")


# ---------------------------------------------------------------------------
# Gemma-exclusion helper (PHASE-03 SC#2 / T-03-13)
# ---------------------------------------------------------------------------

def _assert_no_gemma(module: nn.Module) -> None:
    """
    Assert that no ``language_model`` parameter name or ``gemma`` module-type
    is reachable from ``module``.

    Called at SegModelVariantA / SegModelVariantB construction time.
    Also usable as a standalone test helper.

    Raises
    ------
    AssertionError
        If any such parameter or module is found.
    """
    # Check param names
    bad_params = [
        n for n, _ in module.named_parameters()
        if "language_model" in n
    ]
    assert not bad_params, (
        f"Gemma (language_model) parameters reachable from SegModel "
        f"(PHASE-03 SC#2 violation): {bad_params[:5]}"
    )
    # Check module types
    bad_modules = [
        type(m).__name__ for m in module.modules()
        if "gemma" in type(m).__name__.lower()
    ]
    assert not bad_modules, (
        f"Gemma module-types reachable from SegModel "
        f"(PHASE-03 SC#2 violation): {bad_modules[:5]}"
    )
