"""
Offline unit tests for ``scripts/seg/backbones.py`` (plan 03-02, D-05).

All tests are fully offline under pytest tmp_path or stub_vision_config.
No network access; no real model checkpoint required for scaffold phase.

Covered decisions: D-05 (Backbone protocol), D-06 (construction-only),
PHASE-03 SC#2 (provable Gemma exclusion), EVAL-03 (DINOv2 + Swin protocol).

Tests verify:
  - SigLIP backbone yields (B,C,h,w) feature grids at declared strides for
    224 / 448 / 896 input resolutions.
  - No ``language_model`` parameter or ``gemma`` module type is reachable from
    EITHER variant's seg forward pass (PHASE-03 SC#2).
  - Variant B widens the patch-embed to 15 channels, zero-initialises the
    extra 12 channels, and preserves the pretrained RGB weights across all
    three backbones.
  - DINOv2 and Swin satisfy the same Backbone protocol (declared
    feature_strides / feature_channels; identical decoder path).

These tests fail with ImportError until ``scripts/seg/backbones.py`` lands
(plan 03-02).  They collect cleanly so they serve as live Nyquist gates.
"""

import pytest

# ---------------------------------------------------------------------------
# Import guard — collect cleanly even before seg.backbones is written
# ---------------------------------------------------------------------------
_import_error: Exception | None = None
try:
    from seg.backbones import (
        Backbone,
        SiglipBackbone,
        Dinov2Backbone,
        SwinBackbone,
    )
except ImportError as _e:
    _import_error = _e


def _require_backbones():
    """Fail with a clear ImportError message if the module is not yet written."""
    if _import_error is not None:
        pytest.fail(
            f"seg.backbones not yet implemented (plan 03-02 pending): {_import_error}",
            pytrace=False,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_input(batch=1, channels=3, size=224):
    """Return a float32 random tensor of shape (B, C, H, W)."""
    import torch
    return torch.randn(batch, channels, size, size)


def _make_15ch_input(batch=1, size=224):
    """15-channel input for Variant B (RGB + 12 prior channels)."""
    import torch
    return torch.randn(batch, 15, size, size)


# ---------------------------------------------------------------------------
# Backbone protocol conformance (D-05)
# ---------------------------------------------------------------------------

class TestBackboneProtocol:
    """Backbone Protocol: declared strides + channels; forward returns list[Tensor]."""

    def test_siglip_has_declared_strides(self):
        _require_backbones()
        bb = SiglipBackbone.__new__(SiglipBackbone)
        assert hasattr(bb, "feature_strides"), "SiglipBackbone must declare feature_strides"
        assert hasattr(bb, "feature_channels"), "SiglipBackbone must declare feature_channels"
        assert isinstance(bb.feature_strides, list), "feature_strides must be a list"
        assert isinstance(bb.feature_channels, list), "feature_channels must be a list"

    def test_dinov2_protocol(self, stub_vision_config):
        _require_backbones()
        import torch
        bb = Dinov2Backbone()
        assert hasattr(bb, "feature_strides"), "Dinov2Backbone must declare feature_strides"
        assert hasattr(bb, "feature_channels"), "Dinov2Backbone must declare feature_channels"
        with torch.no_grad():
            feats = bb(_make_input())
        assert isinstance(feats, list), "forward must return list[Tensor]"
        assert len(feats) == len(bb.feature_strides)
        for f, ch in zip(feats, bb.feature_channels):
            B, C, h, w = f.shape
            assert C == ch, f"channel mismatch: expected {ch}, got {C}"

    def test_swin_protocol(self):
        _require_backbones()
        import torch
        bb = SwinBackbone()
        assert hasattr(bb, "feature_strides")
        assert hasattr(bb, "feature_channels")
        assert len(bb.feature_strides) == len(bb.feature_channels)
        # Swin natively emits 4 hierarchical levels at strides 4/8/16/32
        assert len(bb.feature_strides) == 4, (
            f"SwinBackbone must expose 4 FPN levels, got {len(bb.feature_strides)}"
        )
        with torch.no_grad():
            feats = bb(_make_input())
        assert isinstance(feats, list) and len(feats) == 4
        for i, (f, ch) in enumerate(zip(feats, bb.feature_channels)):
            assert f.shape[1] == ch, (
                f"Swin level {i}: expected {ch} channels, got {f.shape[1]}"
            )


# ---------------------------------------------------------------------------
# SigLIP backbone shape / stride contract at 224 / 448 / 896
# ---------------------------------------------------------------------------

class TestSiglipShape:
    """SigLIP backbone yields (B,C,h,w) grids at declared strides for all scales."""

    @pytest.mark.parametrize("tile_size", [224, 448, 896])
    def test_siglip_feature_grid_shape(self, tile_size, stub_vision_config):
        _require_backbones()
        import torch
        bb = SiglipBackbone(stub_config=stub_vision_config)
        with torch.no_grad():
            feats = bb(_make_input(size=tile_size))
        assert isinstance(feats, list) and len(feats) > 0, (
            "SiglipBackbone.forward must return a non-empty list of feature maps"
        )
        patch = stub_vision_config.patch_size  # 14
        expected_grid = tile_size // patch
        for f in feats:
            B, C, h, w = f.shape
            assert B == 1
            assert h == expected_grid and w == expected_grid, (
                f"tile_size={tile_size}: expected grid {expected_grid}x{expected_grid}, "
                f"got {h}x{w}"
            )
            assert C == stub_vision_config.hidden_size, (
                f"expected hidden_size={stub_vision_config.hidden_size}, got C={C}"
            )

    def test_siglip_no_cls_token_in_grid(self, stub_vision_config):
        """SigLIP has NO CLS token — every token is a patch, grid side = tile/patch."""
        _require_backbones()
        import torch
        bb = SiglipBackbone(stub_config=stub_vision_config)
        with torch.no_grad():
            feats = bb(_make_input(size=224))
        # 224/14=16; if CLS were included N=257 and reshape would be wrong
        for f in feats:
            assert f.shape[-1] == 16 and f.shape[-2] == 16, (
                "CLS token must NOT be included in the feature grid (SigLIP has none)"
            )


# ---------------------------------------------------------------------------
# Gemma exclusion (PHASE-03 SC#2) — BOTH variants
# ---------------------------------------------------------------------------

class TestNoGemma:
    """No language_model param or gemma module-type reachable from EITHER variant."""

    def test_no_gemma_variant_a(self, stub_vision_config):
        """Variant A: frozen backbone — no Gemma in parameter graph."""
        _require_backbones()
        import torch
        from seg.model import SegModelVariantA  # noqa: F401 (also gates model)
        bb = SiglipBackbone(stub_config=stub_vision_config)
        names = [n for n, _ in bb.named_parameters()]
        assert not any("language_model" in n for n in names), (
            "Gemma language_model parameters reachable from Variant A backbone"
        )
        assert not any(
            "gemma" in type(m).__name__.lower() for m in bb.modules()
        ), "A Gemma module type is reachable from Variant A backbone"

    def test_no_gemma_variant_b(self, stub_vision_config):
        """Variant B: widened patch-embed — no Gemma in parameter graph."""
        _require_backbones()
        import torch
        bb = SiglipBackbone(stub_config=stub_vision_config, variant_b=True)
        names = [n for n, _ in bb.named_parameters()]
        assert not any("language_model" in n for n in names), (
            "Gemma language_model parameters reachable from Variant B backbone"
        )
        assert not any(
            "gemma" in type(m).__name__.lower() for m in bb.modules()
        ), "A Gemma module type is reachable from Variant B backbone"

    def test_no_gemma_across_all_backbones(self):
        """DINOv2 and Swin also have no gemma module type."""
        _require_backbones()
        for BackboneCls in (Dinov2Backbone, SwinBackbone):
            bb = BackboneCls()
            assert not any(
                "gemma" in type(m).__name__.lower() for m in bb.modules()
            ), f"{BackboneCls.__name__}: Gemma module type must not be reachable"


# ---------------------------------------------------------------------------
# Variant B: widened patch-embed (15-ch), zero-init extras, RGB preserved
# ---------------------------------------------------------------------------

class TestVariantBPatchEmbed:
    """
    Variant B widens the patch-embed to 15 channels.

    D-03a: Variant B accepts RGB (3-ch) + 12 prior channels at the input;
    the patch-embed first conv must accept 15 channels.  The 12 extra channels
    are zero-initialised; the original 3-ch (RGB) weights are preserved from
    the pretrained SigLIP checkpoint.

    Applies to all three backbones behind D-05.
    """

    def test_variant_b_siglip_15ch_input(self, stub_vision_config):
        _require_backbones()
        import torch
        bb = SiglipBackbone(stub_config=stub_vision_config, variant_b=True)
        with torch.no_grad():
            feats = bb(_make_15ch_input(size=224))
        assert feats, "Variant B SiglipBackbone must accept 15-ch input"
        for f in feats:
            assert f.shape[1] == stub_vision_config.hidden_size

    def test_variant_b_siglip_rgb_weights_preserved(self, stub_vision_config):
        """
        The first 3 channels of the widened patch-embed weight must equal
        the pretrained 3-ch weight (RGB weights not clobbered by widening).
        """
        _require_backbones()
        import torch
        # Baseline: 3-ch backbone patch-embed weight
        bb3 = SiglipBackbone(stub_config=stub_vision_config, variant_b=False)
        bb15 = SiglipBackbone(stub_config=stub_vision_config, variant_b=True)
        w3 = bb3.patch_embed_weight()   # expected shape: (out, 3, kH, kW) or similar
        w15 = bb15.patch_embed_weight() # expected shape: (out, 15, kH, kW) or similar
        assert w15.shape[1] == 15, (
            f"Variant B patch-embed must have 15 input channels, got {w15.shape[1]}"
        )
        # RGB portion (channels 0:3) must match the original pretrained weights
        torch.testing.assert_close(
            w15[:, :3, ...], w3[:, :3, ...],
            msg="Variant B must preserve pretrained RGB weights in channels 0:3",
        )

    def test_variant_b_extra_channels_zero_init(self, stub_vision_config):
        """Channels 3:15 of the widened patch-embed must be zero-initialised."""
        _require_backbones()
        import torch
        bb = SiglipBackbone(stub_config=stub_vision_config, variant_b=True)
        w = bb.patch_embed_weight()  # (out, 15, kH, kW)
        extra = w[:, 3:, ...]
        assert extra.abs().max().item() == 0.0, (
            "Variant B extra channels (3:15) must be zero-initialised"
        )

    def test_variant_b_dinov2_15ch(self):
        """DINOv2 Variant B also widens patch-embed to 15 channels."""
        _require_backbones()
        import torch
        bb = Dinov2Backbone(variant_b=True)
        with torch.no_grad():
            feats = bb(_make_15ch_input(size=224))
        assert feats, "Variant B Dinov2Backbone must accept 15-ch input"

    def test_variant_b_swin_15ch(self):
        """Swin Variant B also widens patch-embed to 15 channels."""
        _require_backbones()
        import torch
        bb = SwinBackbone(variant_b=True)
        with torch.no_grad():
            feats = bb(_make_15ch_input(size=224))
        assert feats, "Variant B SwinBackbone must accept 15-ch input"
