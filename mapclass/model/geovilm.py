"""
GeoViLM composition module — wraps (backbone, lc_head, topo_head) as a single nn.Module.

Phase 1 holds the Mock backbone. Phase 2 swaps backbone for SmolVLM via the same
Backbone ABC. Forward pass returns a dict of logits — the trainer applies softmax+
cross-entropy externally so that label smoothing / per-pixel weighting (Phase 2+)
can hook in cleanly.

The module is the unit checkpointed via safetensors.torch.save_file(self.state_dict()).
"""

import torch
from torch import nn

from mapclass.model.backbone import Backbone
from mapclass.model.seg_heads import LandCoverHead, TopographyHead


class GeoViLM(nn.Module):
    def __init__(self, backbone: Backbone, lc_head: LandCoverHead, topo_head: TopographyHead):
        super().__init__()
        # Backbone is also an nn.Module in Phase 1 (MockBackbone subclasses nn.Module).
        # Phase 2 SmolVLMBackbone will wrap an nn.Module internally; either way state_dict() chains.
        self.backbone = backbone
        self.lc_head = lc_head
        self.topo_head = topo_head

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """image: (B, 3, H, W) — already preprocessed (call backbone.preprocess upstream).

        Returns: {"land_cover_logits": (B, 9, H, W), "topography_logits": (B, 3, H, W)}.
        Spatial size matches the input H,W (heads upsample if backbone downsamples).
        """
        feats = self.backbone.extract_features(image)
        H, W = image.shape[-2:]
        return {
            "land_cover_logits": self.lc_head(feats, out_size=(H, W)),
            "topography_logits": self.topo_head(feats, out_size=(H, W)),
        }
