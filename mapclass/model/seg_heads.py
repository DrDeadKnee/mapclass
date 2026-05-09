"""
Dense segmentation heads consuming Backbone.extract_features output.

LandCoverHead emits 9-class logits over LANDCOVER_CLASSES.
TopographyHead emits 3-class logits over TOPO_CLASSES.

The WATER_TOPO=255 sentinel from scripts/label.py is NOT predicted — TopographyHead
learns 3 classes only. mapclass.infer overlays WATER_TOPO=255 at every pixel where
land_cover_argmax == LANDCOVER_IDX["water"] (PATTERNS.md cross-cutting gotcha #4;
mirrors scripts/historical/dem.py:_classify lines 86-92).

Phase 1 consumes the single-key feature dict (D-02). Phase 2 will refactor to
consume multi-stage skip connections per ARCHITECTURE.md Pattern 3 — TODO comment
left below to surface this debt (PATTERNS.md cross-cutting gotcha #8).
"""

import torch
from torch import nn
from torch.nn import functional as F

from mapclass.data.taxonomy import LANDCOVER_CLASSES, TOPO_CLASSES


class LandCoverHead(nn.Module):
    """Dense seg head: dict[str, Tensor] -> Tensor [B, 9, H, W] (logits)."""

    def __init__(self, in_channels: int = 64, num_classes: int = len(LANDCOVER_CLASSES)):
        super().__init__()
        # TODO(phase-2): refactor for multi-stage feature dict per ARCHITECTURE.md Pattern 3
        self.proj = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, feats: dict[str, torch.Tensor], out_size: tuple[int, int] | None = None) -> torch.Tensor:
        x = feats["features"]                                  # Phase 1: single key per D-02
        logits = self.proj(x)
        if out_size is not None and logits.shape[-2:] != out_size:
            logits = F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
        return logits


class TopographyHead(nn.Module):
    """Dense seg head: dict[str, Tensor] -> Tensor [B, 3, H, W] (logits)."""

    def __init__(self, in_channels: int = 64, num_classes: int = len(TOPO_CLASSES)):
        super().__init__()
        # TODO(phase-2): refactor for multi-stage feature dict per ARCHITECTURE.md Pattern 3
        self.proj = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, feats: dict[str, torch.Tensor], out_size: tuple[int, int] | None = None) -> torch.Tensor:
        x = feats["features"]
        logits = self.proj(x)
        if out_size is not None and logits.shape[-2:] != out_size:
            logits = F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
        return logits
