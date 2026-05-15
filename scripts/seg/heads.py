"""
Two thin task heads on the shared decoder trunk (D-02) for the dense semantic
segmentation pipeline (Phase 3).

Strategy:
  - Implements D-02: a ``LandCoverHead`` and a ``TopographyHead``, each a
    single 1×1 Conv2d projecting the shared decoder trunk (``SegDecoder``
    output) to class-logit channels.
  - Land cover: 9-class dense logits → (B, 9, H, W).
  - Topography: 3-class dense logits → (B, 3, H, W).
  - Spatial resolution is unchanged (heads do NOT upsample — the decoder
    (D-01) already outputs at full tile H×W; heads are thin projections only).
  - NO softmax inside the heads — raw logits only (Phase 4 / recursive
    orchestrator owns probability computation and NLL loss; D-06).
  - NOT two independent decoders — both heads share the single decoder trunk
    (D-02 explicit constraint).

Covered decisions: D-01 (decoder is the shared trunk), D-02 (two thin task
                   heads; NOT two decoders), D-06 (construction-only; no
                   training loop / optimizer / backward; no softmax in head).

Requires:
  pip install torch

Usage:
  from seg.decoder import SegDecoder
  from seg.heads import LandCoverHead, TopographyHead

  decoder = SegDecoder(feature_strides=[14], feature_channels=[1152], tile_size=224)
  lc_head   = LandCoverHead(decoder.out_channels)    # in_channels = decoder.out_channels
  topo_head = TopographyHead(decoder.out_channels)

  with torch.no_grad():
      dense      = decoder(backbone_features)        # (B, C, H, W) shared trunk
      lc_logits  = lc_head(dense)                    # (B, 9, H, W)  land-cover logits
      topo_logits = topo_head(dense)                 # (B, 3, H, W)  topography logits
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Class constants (D-02 / PHASE-03 SC#1)
# ---------------------------------------------------------------------------

_LC_CLASSES: int = 9    # 9-class land-cover output
_TOPO_CLASSES: int = 3  # 3-class topography output


# ---------------------------------------------------------------------------
# Thin task heads
# ---------------------------------------------------------------------------

class LandCoverHead(nn.Module):
    """
    9-class land-cover head (D-02 / PHASE-03 SC#1).

    A single 1×1 Conv2d projection on the shared decoder trunk feature map.
    Emits raw logits (B, 9, H, W) — NO softmax (D-06; Phase 4 owns NLL).
    Spatial resolution is preserved (no upsample; the decoder already
    outputs at full tile resolution).

    Parameters
    ----------
    in_channels:
        Number of channels of the shared decoder trunk output
        (``SegDecoder.out_channels``).
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self._proj = nn.Conv2d(in_channels, _LC_CLASSES, kernel_size=1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Shared decoder trunk (B, in_channels, H, W).

        Returns
        -------
        Tensor
            Land-cover logits (B, 9, H, W) — raw, no softmax.
        """
        return self._proj(x)


class TopographyHead(nn.Module):
    """
    3-class topography head (D-02 / PHASE-03 SC#1).

    A single 1×1 Conv2d projection on the shared decoder trunk feature map.
    Emits raw logits (B, 3, H, W) — NO softmax (D-06; Phase 4 owns NLL).
    Spatial resolution is preserved (no upsample).

    Parameters
    ----------
    in_channels:
        Number of channels of the shared decoder trunk output
        (``SegDecoder.out_channels``).
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self._proj = nn.Conv2d(in_channels, _TOPO_CLASSES, kernel_size=1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Shared decoder trunk (B, in_channels, H, W).

        Returns
        -------
        Tensor
            Topography logits (B, 3, H, W) — raw, no softmax.
        """
        return self._proj(x)
