"""
Inference module — the stable contract surface to the downstream hex-grid app.

Usage:
    python -m mapclass.infer <checkpoint> <image>

    or as a Python API (Phase 1 minimal; Phase 6 finalises):
        from mapclass.infer import load_model
        predictor = load_model("models/geovilm_phase1_mock.pt")
        result = predictor.predict(image)

Returns a dict:
    {
      "land_cover": Tensor [9, H, W]  — softmax probabilities, sums to 1.0 ± fp16 epsilon
      "topography": Tensor [3, H, W]  — softmax probabilities; the integer-label form
                                        overlays WATER_TOPO=255 at pixels where
                                        land_cover argmax == LANDCOVER_IDX["water"]
                                        (the float-prob form keeps its 3-class softmax).
      "model_version": str,
      "taxonomy": {"land_cover": [9 names], "topography": [3 names]},
      "metadata": dict[str, Any]      — round-tripped + JSON-decoded checkpoint metadata
    }

CRITICAL (ARCHITECTURE.md Anti-Pattern 4 / PATTERNS.md gotcha #3): mapclass.infer
MUST NOT import anything from the dataset, contract, loss_weights, or splits
submodules of the data layer. The inference module's deps are model + numpy +
PIL + safetensors only (no torch.utils.data, no rasterio).

Phase 6 will add a startup-import test enforcing this discipline; Phase 1 honors
it by hand-discipline (PATTERNS.md cross-cutting gotcha #3).
"""

# Block 1: stdlib
import argparse
import json
import sys
from pathlib import Path

# Block 2: third-party
import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file

# Block 3: first-party (taxonomy is the ONLY mapclass.data import allowed; see Anti-Pattern 4)
from mapclass.data.taxonomy import (
    LANDCOVER_CLASSES,
    LANDCOVER_IDX,
    TOPO_CLASSES,
    WATER_TOPO,
    TaxonomyHashMismatchError,
    taxonomy_hash,
)
from mapclass.model.geovilm import GeoViLM
from mapclass.model.mock_backbone import MockBackbone
from mapclass.model.seg_heads import LandCoverHead, TopographyHead


# ---------------------------------------------------------------------------
# Module-private constants
# ---------------------------------------------------------------------------
_WATER_CLASS_IDX = LANDCOVER_IDX["water"]   # 0; used for WATER_TOPO overlay


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
class _Predictor:
    """Bundles a GeoViLM model + its decoded metadata. Returned by load_model."""

    def __init__(self, model: GeoViLM, metadata: dict, device: torch.device):
        self.model = model.to(device).eval()
        self.metadata = metadata
        self.device = device

    @torch.no_grad()
    def predict(self, image) -> dict:
        """See module docstring for the contract."""
        # Accept PIL.Image, numpy ndarray (H, W, 3), or tensor (3, H, W) / (B, 3, H, W)
        if isinstance(image, Image.Image):
            image_input = image
            H_in, W_in = image.height, image.width
        elif isinstance(image, np.ndarray):
            image_input = Image.fromarray(image).convert("RGB")
            H_in, W_in = image_input.height, image_input.width
        elif isinstance(image, torch.Tensor):
            t = image if image.dim() == 4 else image.unsqueeze(0)
            H_in, W_in = t.shape[-2], t.shape[-1]
            image_input = t.to(self.device).float()
            if image_input.max() > 1.5:           # uint8 path
                image_input = image_input / 255.0
            preprocessed = self.model.backbone.preprocess(image_input)
            return _predict_from_preprocessed(self.model, preprocessed, self.metadata,
                                              H_in=H_in, W_in=W_in)
        else:
            raise TypeError(f"unsupported image type: {type(image)}")
        print(f"  inference: image {W_in}x{H_in} (T-02-02: log dims at infer)")
        preprocessed = self.model.backbone.preprocess(image_input).to(self.device)
        return _predict_from_preprocessed(self.model, preprocessed, self.metadata,
                                          H_in=H_in, W_in=W_in)


def _predict_from_preprocessed(model: GeoViLM, image: torch.Tensor, metadata: dict,
                               *, H_in: int, W_in: int) -> dict:
    out = model(image)                                          # (1, 9, H, W), (1, 3, H, W)
    lc_logits = out["land_cover_logits"][0]                     # (9, H, W)
    topo_logits = out["topography_logits"][0]                   # (3, H, W)
    # TS-4: probabilities, not logits.
    lc_probs = torch.softmax(lc_logits, dim=0)                  # sum_c == 1.0 per pixel
    topo_probs = torch.softmax(topo_logits, dim=0)
    return {
        "land_cover": lc_probs,
        "topography": topo_probs,
        "land_cover_argmax": lc_probs.argmax(dim=0),            # (H, W) int64
        "topography_label_with_overlay": _overlay_water_topo(
            topo_probs.argmax(dim=0), lc_probs.argmax(dim=0)
        ),                                                      # (H, W) uint8 with WATER_TOPO=255
        "model_version": metadata.get("model_version", "unknown"),
        "taxonomy": {"land_cover": list(LANDCOVER_CLASSES), "topography": list(TOPO_CLASSES)},
        "metadata": metadata,
    }


def _overlay_water_topo(topo_argmax: torch.Tensor, lc_argmax: torch.Tensor) -> torch.Tensor:
    """PATTERNS.md gotcha #4: where lc_argmax==water, set topo label to WATER_TOPO=255.

    Mirrors scripts/historical/dem.py:_classify (lines 86-92). Returns uint8 tensor.
    """
    overlay = topo_argmax.to(torch.uint8).clone()
    water_mask = (lc_argmax == _WATER_CLASS_IDX)
    overlay[water_mask] = WATER_TOPO                            # 255
    return overlay


def load_model(checkpoint: str | Path, device: str = "auto") -> _Predictor:
    """Load a trained GeoViLM checkpoint (Phase 1: Mock backbone only).

    Reads metadata from the safetensors header; raises TaxonomyHashMismatchError
    if the checkpoint's taxonomy_hash does not match the live taxonomy_hash().
    Phase 1 only supports backbone='mock'; Phase 2 swap will dispatch on this field.
    """
    ckpt_path = Path(checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    # Round-trip metadata: every value JSON-decoded (PATTERNS.md gotcha #2).
    with safe_open(str(ckpt_path), framework="pt") as f:
        raw_meta = f.metadata() or {}
    metadata: dict = {}
    for k, v in raw_meta.items():
        try:
            metadata[k] = json.loads(v)
        except (TypeError, json.JSONDecodeError):
            metadata[k] = v

    # PITFALL 4 prevention #3: refuse mismatched taxonomy.
    ckpt_tax = metadata.get("taxonomy_hash")
    live_tax = taxonomy_hash()
    if ckpt_tax != live_tax:
        raise TaxonomyHashMismatchError(
            f"checkpoint taxonomy_hash {ckpt_tax!r} != live taxonomy_hash {live_tax!r}; "
            "label-class mapping has drifted (PITFALL 4 prevention #3). Aborting load."
        )

    backbone_tag = metadata.get("backbone", "mock")
    if backbone_tag != "mock":
        raise ValueError(
            f"Phase 1 mapclass.infer only supports backbone='mock'; got {backbone_tag!r}. "
            "SmolVLM lands in Phase 2."
        )

    # Resolve device.
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # Build the same architecture as the trainer; load weights via safetensors.
    seed = int(metadata.get("training_seed", 42))
    backbone = MockBackbone(training_seed=seed)
    n_feat = backbone.feature_channels["features"]
    model = GeoViLM(backbone, LandCoverHead(in_channels=n_feat), TopographyHead(in_channels=n_feat))
    state = load_file(str(ckpt_path))
    model.load_state_dict(state)
    return _Predictor(model, metadata, dev)


def predict(predictor: _Predictor, image) -> dict:
    """Functional alias for predictor.predict(image)."""
    return predictor.predict(image)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
def _epsilon_clamp(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Numerical stability for log(prob) downstream (T-02-03 mitigation)."""
    return t.clamp_min(eps)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a GeoViLM checkpoint on a single image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    print("=== Infer: Phase 1 (Mock backbone) ===")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Image:      {args.image}")
    predictor = load_model(args.checkpoint, device=args.device)
    img = Image.open(args.image).convert("RGB")
    result = predictor.predict(img)

    lc = result["land_cover"]
    topo = result["topography"]
    # TS-4 contract assertion: probabilities sum to 1.0 ± fp16 epsilon (1e-3) per pixel.
    lc_sum = lc.sum(dim=0)
    topo_sum = topo.sum(dim=0)
    assert (lc_sum - 1.0).abs().max().item() < 1e-3, "land_cover probabilities do not sum to 1.0"
    assert (topo_sum - 1.0).abs().max().item() < 1e-3, "topography probabilities do not sum to 1.0"
    print(f"  land_cover: shape={tuple(lc.shape)}, sum-deviation={(lc_sum - 1.0).abs().max().item():.2e}")
    print(f"  topography: shape={tuple(topo.shape)}, sum-deviation={(topo_sum - 1.0).abs().max().item():.2e}")
    print(f"  WATER_TOPO overlay applied at {(result['topography_label_with_overlay'] == WATER_TOPO).sum().item()} pixels")


if __name__ == "__main__":
    main()
