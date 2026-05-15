"""
Shared UPerNet-style PPM+FPN conv decoder (D-01) for the dense semantic
segmentation pipeline (Phase 3).

Strategy:
  - Implements D-01: a lightweight UPerNet-style convolutional decoder consuming
    the ``Backbone`` protocol's declared-stride feature list (D-05 seam).  NOT a
    linear-probe upsample, NOT DPT-reassemble (D-01 explicit exclusions).
  - Reads ``feature_strides`` / ``feature_channels`` dynamically from the
    backbone at construction time — never hard-codes level count or channel dims
    (RESEARCH Pitfall 4; T-03-10 mitigation).
  - PPM (Pyramid Pooling Module) on the deepest-stride feature adds global
    context; lateral 1×1 convs + FPN top-down fuses multi-scale information;
    3×3 smoothing convs sharpen boundaries (D-01 terrain-boundary driver).
  - ``F.interpolate`` brings every level to a common decoder working resolution
    before FPN addition so ViT k-pseudo-levels (all same stride) and Swin's 4
    native levels (strides 4/8/16/32) BOTH work with zero code change
    (Pitfall 4 guard, T-03-10).
  - Final progressive upsample (staged 2× bilinear + 3×3 conv) reaches tile
    H×W WITHOUT a single ×14 bilinear blow-up — preserves coastline/range
    sharpness (D-01 boundary-sharpness driver).

Topology decisions (D-01 Claude's Discretion — documented):
  - Decoder working channels: 256 (``_DECODER_CHANNELS``).
  - PPM bin sizes: (1, 2, 3, 6) — standard UPerNet setting.
  - PPM output is concatenated with the deepest-level lateral output and
    projected back to 256 channels before FPN fusion.
  - All lateral convs: (C_in → 256) 1×1 + GroupNorm(32) + ReLU.
  - FPN smoothing: 3×3 depthwise-separable-style conv (keep param count low).
  - Progressive upsample: repeated 2× bilinear + 3×3 conv blocks until the
    output spatial size reaches tile_size.  This avoids the single large-factor
    bilinear step that blurs terrain boundaries.
  - GroupNorm(32) throughout (batch-size-agnostic; research pipeline may use
    B=1 at inference).

Requires:
  pip install torch

Usage:
  from seg.decoder import SegDecoder
  decoder = SegDecoder(
      feature_strides=[14],          # e.g. from SiglipBackbone / Dinov2Backbone
      feature_channels=[1152],
      tile_size=224,
  )
  # or Swin:
  decoder = SegDecoder(
      feature_strides=[4, 8, 16, 32],
      feature_channels=[128, 256, 512, 1024],
      tile_size=224,
  )
  dense = decoder(backbone_features)   # (B, out_channels, tile_size, tile_size)
  # feed to LandCoverHead / TopographyHead (see seg.heads)

Covered decisions: D-01 (lightweight FPN/UPerNet conv decoder), D-05 (reads
                   backbone.feature_strides/feature_channels dynamically),
                   D-06 (construction-only; no training loop/optimizer/backward).

Trust model:
  T-03-10: decoder reads backbone strides/channels dynamically + F.interpolate to
           a common resolution — backbone swap cannot silently misalign features.
  T-03-11: decoder channel widths are modest (256 working channels); unbounded
           params on large maps are a research-pipeline accepted risk (D-11).
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Hyper-parameters (D-01 Claude's Discretion)
# ---------------------------------------------------------------------------

_DECODER_CHANNELS: int = 256   # decoder working width
_PPM_BINS: tuple[int, ...] = (1, 2, 3, 6)   # pyramid pooling bins (UPerNet default)
_GN_GROUPS: int = 32   # GroupNorm groups (batch-size-agnostic)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _gn_relu(num_channels: int) -> nn.Sequential:
    """GroupNorm(32) + ReLU block, used throughout the decoder."""
    return nn.Sequential(
        nn.GroupNorm(_GN_GROUPS, num_channels),
        nn.ReLU(inplace=True),
    )


def _conv1x1(in_ch: int, out_ch: int) -> nn.Sequential:
    """1×1 conv + GroupNorm + ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
        *_gn_relu(out_ch),
    )


def _conv3x3(in_ch: int, out_ch: int) -> nn.Sequential:
    """3×3 conv + GroupNorm + ReLU (smoothing conv)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        *_gn_relu(out_ch),
    )


class _PPMBlock(nn.Module):
    """
    Pyramid Pooling Module on a single feature map (D-01 / RESEARCH Pattern 3).

    Performs adaptive average pooling at each bin size, projects each bin to
    ``_DECODER_CHANNELS // len(bins)`` channels, bilinearly upsamples back to
    the original spatial size, and concatenates with the input feature map.
    The concatenated output is projected to ``out_channels`` via a 1×1 conv.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        inter_ch = _DECODER_CHANNELS // len(_PPM_BINS)
        self._pools = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(b),
                nn.Conv2d(in_channels, inter_ch, kernel_size=1, bias=False),
                nn.GroupNorm(_GN_GROUPS, inter_ch),
                nn.ReLU(inplace=True),
            )
            for b in _PPM_BINS
        ])
        fuse_in = in_channels + inter_ch * len(_PPM_BINS)
        self._fuse = _conv1x1(fuse_in, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.shape[-2:]
        parts = [x]
        for pool in self._pools:
            p = pool(x)
            parts.append(F.interpolate(p, size=(h, w), mode="bilinear", align_corners=False))
        return self._fuse(torch.cat(parts, dim=1))


class _UpsampleBlock(nn.Module):
    """2× bilinear upsample + 3×3 smoothing conv (progressive upsample stage)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self._smooth = _conv3x3(channels, channels)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self._smooth(x)


# ---------------------------------------------------------------------------
# SegDecoder
# ---------------------------------------------------------------------------

class SegDecoder(nn.Module):
    """
    Shared UPerNet-style PPM+FPN conv decoder (D-01).

    Accepts any backbone satisfying the D-05 declared-stride contract:
      ``feature_strides: list[int]``, ``feature_channels: list[int]``
      ``forward(x) -> list[Tensor]``  # (B, Ci, Hi, Wi) per level

    Works unchanged for:
      - ViT k-pseudo-level feature lists (all same stride, all same channel dim).
      - Swin 4-level hierarchical pyramid (strides 4/8/16/32, varying channels).

    Output: a single dense feature map of shape
      ``(B, out_channels, tile_size, tile_size)``
    consumed by ``LandCoverHead`` and ``TopographyHead`` in ``seg.heads``.

    Parameters
    ----------
    feature_strides:
        Declared strides from the backbone (matches ``backbone.feature_strides``).
    feature_channels:
        Channel dims per level (matches ``backbone.feature_channels``).
    tile_size:
        Target spatial resolution H = W of the output (must equal the input
        tile height/width).
    decoder_channels:
        Internal working width (default 256 — D-01 Claude's Discretion).
    """

    def __init__(
        self,
        feature_strides: Sequence[int],
        feature_channels: Sequence[int],
        tile_size: int,
        decoder_channels: int = _DECODER_CHANNELS,
    ) -> None:
        super().__init__()

        assert len(feature_strides) == len(feature_channels), (
            "feature_strides and feature_channels must have the same length"
        )
        assert len(feature_strides) >= 1, "At least one feature level required"

        self._feature_strides = list(feature_strides)
        self._feature_channels = list(feature_channels)
        self._tile_size = tile_size
        self._decoder_channels = decoder_channels

        n = len(feature_strides)

        # ------------------------------------------------------------------
        # PPM on the deepest level (highest stride index → largest stride →
        # smallest spatial size → most semantic)
        # ------------------------------------------------------------------
        deep_ch = feature_channels[-1]
        self._ppm = _PPMBlock(deep_ch, decoder_channels)

        # ------------------------------------------------------------------
        # Lateral 1×1 convs: one per level (shallowest → deepest)
        # The deepest level goes through PPM first; its lateral projects the
        # PPM *output* (already decoder_channels wide) as identity (1×1 on
        # decoder_channels → decoder_channels) to keep the FPN loop uniform.
        # ------------------------------------------------------------------
        laterals = []
        for i, ch in enumerate(feature_channels):
            if i < n - 1:
                laterals.append(_conv1x1(ch, decoder_channels))
            else:
                # Deepest level — the PPM output is already decoder_channels;
                # use a 1×1 identity-like proj to keep the code uniform.
                laterals.append(_conv1x1(decoder_channels, decoder_channels))
        self._laterals = nn.ModuleList(laterals)

        # ------------------------------------------------------------------
        # FPN smoothing: 3×3 conv per level after top-down add
        # ------------------------------------------------------------------
        self._smoothers = nn.ModuleList([_conv3x3(decoder_channels, decoder_channels) for _ in range(n)])

        # ------------------------------------------------------------------
        # FPN working resolution: all levels are interpolated to this spatial
        # size before the top-down add (Pitfall 4 / T-03-10 mitigation).
        # Use the shallowest (finest) level's spatial size as the working
        # resolution.  For ViT equal-stride levels this is also the only size.
        # For Swin it is tile_size // min(strides).
        # ------------------------------------------------------------------
        self._working_h = tile_size // min(feature_strides)
        self._working_w = tile_size // min(feature_strides)

        # ------------------------------------------------------------------
        # Progressive upsample: from working resolution to tile_size.
        # Number of 2× stages = ceil(log2(tile_size / working_h)).
        # If working_h == tile_size, num_stages = 0 (no upsampling needed).
        # ------------------------------------------------------------------
        if self._working_h < tile_size:
            scale = tile_size / self._working_h
            num_stages = math.ceil(math.log2(scale))
        else:
            num_stages = 0

        self._upsample_stages = nn.ModuleList(
            [_UpsampleBlock(decoder_channels) for _ in range(num_stages)]
        )

        # Exposed for head construction (D-02)
        self.out_channels: int = decoder_channels

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, features: list[Tensor]) -> Tensor:
        """
        Parameters
        ----------
        features:
            List of (B, Ci, Hi, Wi) tensors, one per backbone level, in
            ascending stride order (shallowest / finest first, deepest last) —
            matching the backbone's declared order.

        Returns
        -------
        Tensor
            (B, out_channels, tile_size, tile_size) shared dense feature map.
        """
        assert len(features) == len(self._feature_strides), (
            f"Expected {len(self._feature_strides)} feature maps, got {len(features)}"
        )

        wh, ww = self._working_h, self._working_w

        # ---- (1) Apply PPM to the deepest level ----
        deep_ppm = self._ppm(features[-1])   # (B, decoder_channels, H_deep, W_deep)

        # ---- (2) Build lateral projections, interpolating to working resolution ----
        # laterals[i] corresponds to feature level i; deepest last.
        level_feats: list[Tensor] = []
        for i, (feat, lat) in enumerate(zip(features, self._laterals)):
            if i == len(features) - 1:
                # Deepest: project PPM output
                proj = lat(deep_ppm)
            else:
                proj = lat(feat)
            # Interpolate to common working resolution (Pitfall 4 guard)
            if proj.shape[-2] != wh or proj.shape[-1] != ww:
                proj = F.interpolate(proj, size=(wh, ww), mode="bilinear", align_corners=False)
            level_feats.append(proj)

        # ---- (3) FPN top-down fusion ----
        # Start from the deepest and add toward the shallowest.
        fpn = level_feats[-1]                  # deepest; already at working res
        fpn_levels = [fpn]
        for i in range(len(level_feats) - 2, -1, -1):
            fpn = level_feats[i] + fpn         # element-wise add (same working res)
            fpn_levels.insert(0, fpn)

        # ---- (4) 3×3 smoothing on each FPN level, then sum-fuse ----
        fused = torch.zeros_like(fpn_levels[0])
        for lvl, smoother in zip(fpn_levels, self._smoothers):
            fused = fused + smoother(lvl)

        # ---- (5) Progressive upsample to tile_size ----
        out = fused
        for stage in self._upsample_stages:
            out = stage(out)

        # Final interpolate to exact tile_size (handles cases where 2^n > tile_size)
        if out.shape[-2] != self._tile_size or out.shape[-1] != self._tile_size:
            out = F.interpolate(
                out, size=(self._tile_size, self._tile_size),
                mode="bilinear", align_corners=False,
            )

        return out
