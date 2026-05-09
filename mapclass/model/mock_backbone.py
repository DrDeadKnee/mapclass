"""
Mock backbone — a tiny learnable conv stub for the Phase 1 walking skeleton.

~10–100k parameters, ~2 conv layers RGB → features. Training actually backprops
through it; the gradient path is real (D-01). Eval NLL may improve modestly over
the class-frequency prior — that is a positive diagnostic, NOT a ship metric.

Phase 2 swaps this out for SmolVLMBackbone via the same Backbone interface;
Phase 5 swaps it for PaliGemma2Backbone. The single-key feature dict ({"features": ...})
is the D-02 simplification — Phase 2 must refactor seg heads to consume multi-stage
features (PATTERNS.md cross-cutting gotcha #8).

CONTEXT.md D-01..D-04 anchor this implementation:
  - D-01: learnable, not random (real gradient path tested)
  - D-02: single-key feature dict (multi-stage deferred to Phase 2)
  - D-03: init seeded from training_seed (PITFALL 15 prevention)
  - D-04: backbone='mock' tag in safetensors metadata (discipline-only)
"""

import torch
from torch import nn
from torchvision import transforms as T

from mapclass.model.backbone import Backbone
from mapclass.seeding import set_global_seed


class MockBackbone(Backbone, nn.Module):
    """Learnable conv stub. ~30k params total (Conv2d(3→32) + Conv2d(32→64))."""

    image_size: int = 384                                 # match SmolVLM's input (Phase 2 swap parity)
    feature_channels: dict[str, int] = {"features": 64}   # single-key per D-02
    processor_identity: str = "mock_passthrough"          # embedded in checkpoint metadata

    def __init__(self, training_seed: int = 42):
        nn.Module.__init__(self)
        # D-03: seed before parameter init so two runs with the same seed produce
        # byte-identical weights.
        set_global_seed(training_seed)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Mean/std picked to roughly match ImageNet stats so Phase-2 SmolVLM swap is trivial.
        self._normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def preprocess(self, image) -> torch.Tensor:
        """Pass-through normalize. PIL.Image or (3, H, W)/(B, 3, H, W) tensor accepted."""
        if not isinstance(image, torch.Tensor):
            # PIL.Image → tensor in [0, 1]
            arr = T.functional.to_tensor(image)            # (3, H, W) float32 [0, 1]
            return self._normalize(arr).unsqueeze(0)       # (1, 3, H, W)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        # Already a (B, 3, H, W) float tensor in [0, 1] (e.g. from MapClassDataset)
        return self._normalize(image)

    def extract_features(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        x = torch.relu(self.conv1(image))
        x = torch.relu(self.conv2(x))
        return {"features": x}
