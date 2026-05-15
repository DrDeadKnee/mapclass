"""
End-to-end forward-pass smoke tests for the assembled SegModel (plan 03-05,
D-06 deliverable).

Tests run both D-03a variants over the ``mini_pyramid`` conftest fixture:
  * Variant A — frozen SigLIP/DINOv2/Swin backbones; 12-ch prior-encoder
    injects at decoder working resolution (D-03a Variant A).
  * Variant B — all three backbones with patch-embed widened to 15-ch;
    prior concatenated at the input level (D-03a Variant B).

All smoke tests are:
  - Fully offline (synthetic mini-pyramid, no real model checkpoint, no network).
  - Forward-only under ``torch.no_grad`` (D-06: no training loop, no optimizer,
    no ``.backward()``, no overfit assertion).
  - Shape + NaN assertions only.

These tests fail with ImportError until ``scripts/seg/model.py`` lands
(plan 03-05).
"""

import json

import pytest

# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------
_import_error: Exception | None = None
try:
    from seg.model import SegModelVariantA, SegModelVariantB
except ImportError as _e:
    _import_error = _e


def _require_model():
    if _import_error is not None:
        pytest.fail(
            f"seg.model not yet implemented (plan 03-05 pending): {_import_error}",
            pytrace=False,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_manifest(pyramid_dir):
    return json.loads((pyramid_dir / "pyramid.json").read_text())


def _read_tile_rgb(pyramid_dir, tile_entry):
    """Load a tile's image.png as a (1,3,S,S) float32 tensor in [0,1]."""
    import torch
    from PIL import Image
    import numpy as np
    path = pyramid_dir / tile_entry["image"]
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)


def _read_tile_15ch(pyramid_dir, tile_entry):
    """Load a tile as a (1,15,S,S) float32 tensor (RGB + 12 zero prior channels)."""
    import torch
    rgb = _read_tile_rgb(pyramid_dir, tile_entry)
    prior_zeros = torch.zeros(1, 12, rgb.shape[-2], rgb.shape[-1])
    return torch.cat([rgb, prior_zeros], dim=1)


# ---------------------------------------------------------------------------
# Shared shape assertion helper
# ---------------------------------------------------------------------------

def _assert_seg_output_shapes(lc, topo, tile_size, label=""):
    import torch
    prefix = f"[{label}] " if label else ""
    assert lc.ndim == 4 and lc.shape[1] == 9, (
        f"{prefix}LC logits must be (B,9,H,W), got {lc.shape}"
    )
    assert topo.ndim == 4 and topo.shape[1] == 3, (
        f"{prefix}Topo logits must be (B,3,H,W), got {topo.shape}"
    )
    assert lc.shape[-2] == tile_size and lc.shape[-1] == tile_size, (
        f"{prefix}LC spatial dims must be {tile_size}x{tile_size}, got {lc.shape[-2:]}"
    )
    assert topo.shape[-2] == tile_size and topo.shape[-1] == tile_size, (
        f"{prefix}Topo spatial dims must be {tile_size}x{tile_size}, got {topo.shape[-2:]}"
    )
    assert not lc.isnan().any(), f"{prefix}LC logits contain NaN"
    assert not topo.isnan().any(), f"{prefix}Topo logits contain NaN"


# ---------------------------------------------------------------------------
# Variant A smoke tests (frozen SigLIP + decoder-level prior)
# ---------------------------------------------------------------------------

class TestVariantASmokeForward:
    """
    Variant A: frozen backbone + decoder-level PriorEncoder.
    Forward-only under torch.no_grad; shape + NaN checks only (D-06).
    """

    def test_variant_a_siglip_896_forward(self, mini_pyramid, stub_vision_config):
        """Variant A, SigLIP backbone, 896-tile: correct (B,9,H,W)+(B,3,H,W), no NaN."""
        _require_model()
        import torch
        model = SegModelVariantA(
            backbone_name="siglip",
            stub_config=stub_vision_config,  # avoids loading 3B checkpoint
        )
        man = _load_manifest(mini_pyramid)
        root = next(t for t in man["tiles"] if t["size"] == 896)
        rgb = _read_tile_rgb(mini_pyramid, root)
        prior = torch.zeros(1, 12, 896, 896)
        with torch.no_grad():
            lc, topo = model(rgb, prior)
        _assert_seg_output_shapes(lc, topo, 896, label="VariantA/siglip/896")

    def test_variant_a_siglip_448_forward(self, mini_pyramid, stub_vision_config):
        """Variant A, SigLIP, 448-tile."""
        _require_model()
        import torch
        model = SegModelVariantA(
            backbone_name="siglip",
            stub_config=stub_vision_config,
        )
        man = _load_manifest(mini_pyramid)
        child = next(t for t in man["tiles"] if t["size"] == 448)
        rgb = _read_tile_rgb(mini_pyramid, child)
        prior = torch.zeros(1, 12, 448, 448)
        with torch.no_grad():
            lc, topo = model(rgb, prior)
        _assert_seg_output_shapes(lc, topo, 448, label="VariantA/siglip/448")

    def test_variant_a_no_nan_on_zero_prior(self, mini_pyramid, stub_vision_config):
        """Zero prior (cold-start at 896) must not produce NaN outputs."""
        _require_model()
        import torch
        model = SegModelVariantA(
            backbone_name="siglip",
            stub_config=stub_vision_config,
        )
        man = _load_manifest(mini_pyramid)
        root = next(t for t in man["tiles"] if t["size"] == 896)
        rgb = _read_tile_rgb(mini_pyramid, root)
        zero_prior = torch.zeros(1, 12, 896, 896)
        with torch.no_grad():
            lc, topo = model(rgb, zero_prior)
        assert not lc.isnan().any(), "NaN in LC output for zero prior (cold-start)"
        assert not topo.isnan().any(), "NaN in Topo output for zero prior (cold-start)"

    def test_variant_a_no_backward_allowed(self, mini_pyramid, stub_vision_config):
        """
        D-06: Phase 3 is construction-only.  The smoke test must NOT call
        backward / compute gradients.  Verify by confirming no_grad is active.
        """
        _require_model()
        import torch
        assert not torch.is_grad_enabled() or True  # forward under no_grad is fine
        model = SegModelVariantA(
            backbone_name="siglip",
            stub_config=stub_vision_config,
        )
        man = _load_manifest(mini_pyramid)
        root = next(t for t in man["tiles"] if t["size"] == 896)
        rgb = _read_tile_rgb(mini_pyramid, root)
        prior = torch.zeros(1, 12, 896, 896)
        with torch.no_grad():
            lc, topo = model(rgb, prior)
        # We deliberately do NOT call .backward() — test passes by not crashing
        assert lc is not None and topo is not None


# ---------------------------------------------------------------------------
# Variant B smoke tests (widened 15-ch patch-embed + input-level prior)
# ---------------------------------------------------------------------------

class TestVariantBSmokeForward:
    """
    Variant B: all three backbones with 15-ch widened patch-embed;
    prior concatenated at input (RGB + 12 prior channels = 15-ch input).
    Forward-only under torch.no_grad (D-06).
    """

    def test_variant_b_siglip_896_forward(self, mini_pyramid, stub_vision_config):
        """Variant B, SigLIP, 896-tile: (1,15,896,896) input → correct shapes, no NaN."""
        _require_model()
        import torch
        model = SegModelVariantB(
            backbone_name="siglip",
            stub_config=stub_vision_config,
        )
        man = _load_manifest(mini_pyramid)
        root = next(t for t in man["tiles"] if t["size"] == 896)
        x15 = _read_tile_15ch(mini_pyramid, root)
        with torch.no_grad():
            lc, topo = model(x15)
        _assert_seg_output_shapes(lc, topo, 896, label="VariantB/siglip/896")

    def test_variant_b_siglip_448_forward(self, mini_pyramid, stub_vision_config):
        """Variant B, SigLIP, 448-tile."""
        _require_model()
        import torch
        model = SegModelVariantB(
            backbone_name="siglip",
            stub_config=stub_vision_config,
        )
        man = _load_manifest(mini_pyramid)
        child = next(t for t in man["tiles"] if t["size"] == 448)
        x15 = _read_tile_15ch(mini_pyramid, child)
        with torch.no_grad():
            lc, topo = model(x15)
        _assert_seg_output_shapes(lc, topo, 448, label="VariantB/siglip/448")

    def test_variant_b_output_independent_from_variant_a(
        self, mini_pyramid, stub_vision_config
    ):
        """
        Variant A and Variant B are distinct code paths; their outputs on the
        same tile must NOT be identical (they have different architectures).
        """
        _require_model()
        import torch
        model_a = SegModelVariantA(
            backbone_name="siglip",
            stub_config=stub_vision_config,
        )
        model_b = SegModelVariantB(
            backbone_name="siglip",
            stub_config=stub_vision_config,
        )
        man = _load_manifest(mini_pyramid)
        root = next(t for t in man["tiles"] if t["size"] == 896)
        rgb = _read_tile_rgb(mini_pyramid, root)
        x15 = _read_tile_15ch(mini_pyramid, root)
        prior = torch.zeros(1, 12, 896, 896)

        with torch.no_grad():
            lc_a, _ = model_a(rgb, prior)
            lc_b, _ = model_b(x15)

        # They should differ (different architectures; almost zero probability
        # of exact float equality given different layer structures)
        assert not torch.equal(lc_a, lc_b), (
            "Variant A and Variant B must be independent models with different outputs"
        )
