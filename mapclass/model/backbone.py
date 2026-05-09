"""
Backbone interface — swap surface for the small VL backbone.

The Backbone ABC is the only place that knows about the specific VL model.
Concrete subclasses (MockBackbone in Phase 1; SmolVLMBackbone in Phase 2;
PaliGemma2Backbone in Phase 5) wrap one specific model and expose a uniform
extract_features + preprocess interface (research/ARCHITECTURE.md Pattern 3).

PITFALL 4 (CRITICAL): the Backbone owns preprocess() — never re-implement
torchvision.transforms.Compose at the training-loop or inference-path level.
The processor identity hash is embedded in checkpoint metadata; load_model
refuses on taxonomy_hash mismatch.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from PIL.Image import Image


class Backbone(ABC):
    """Swap surface for the small VL backbone.

    Subclasses MUST set image_size, feature_channels, processor_identity as
    class attributes, and MUST implement preprocess() and extract_features().
    """

    image_size: int                        # input resolution (384 for SmolVLM, 224 for PaliGemma)
    feature_channels: dict[str, int]       # stage_name -> channel count, e.g. {"features": 64}
    processor_identity: str                # short string embedded in checkpoint metadata (TS-8)

    @abstractmethod
    def preprocess(self, image: "Image | torch.Tensor") -> "torch.Tensor":
        """Convert PIL.Image or HxWx3 uint8 tensor to the backbone's expected input.

        Owns resize, normalize, padding, dtype. Train + infer both call this —
        never re-implement preprocessing externally (PITFALL 4 prevention #1).
        """
        ...

    @abstractmethod
    def extract_features(self, image: "torch.Tensor") -> "dict[str, torch.Tensor]":
        """Returns a dict of feature maps. Caller is the seg-head / OCR-head.

        Phase 1 (MockBackbone): single key "features", single tensor (D-02).
        Phase 2+ (SmolVLMBackbone): same shape; multi-stage punted to Phase 2 plan.
        """
        ...
