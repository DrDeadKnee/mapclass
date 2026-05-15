"""
Offline unit tests for ``scripts/seg/decoder.py`` and ``scripts/seg/heads.py``
(plan 03-03, D-01/D-02).

All tests are fully offline and require no model checkpoint.  Feature maps are
synthesised as random tensors matching the declared-stride contract from
``seg.backbones`` (D-05).

Covered decisions: D-01 (lightweight conv decoder), D-02 (shared trunk + 2
thin heads), D-05 (decoder reads backbone.feature_strides/feature_channels
dynamically), D-06 (construction-only, no training).

Tests verify:
  - Shared conv decoder + land-cover head emits (B, 9, H, W) at full tile res.
  - Shared conv decoder + topography head emits (B, 3, H, W) at full tile res.
  - Decoder adapts to single-scale ViT (1 feature map, stride 14/16) and
    multi-scale Swin (4 feature maps at strides 4/8/16/32) via the same code
    path (Pitfall 4 guard).
  - Output spatial dimensions exactly match the input tile size H, W.

These tests fail with ImportError until ``scripts/seg/decoder.py`` and
``scripts/seg/heads.py`` land (plan 03-03).
"""

import pytest

# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------
_import_error: Exception | None = None
try:
    from seg.decoder import SegDecoder
    from seg.heads import LandCoverHead, TopographyHead
except ImportError as _e:
    _import_error = _e


def _require_decoder():
    if _import_error is not None:
        pytest.fail(
            f"seg.decoder / seg.heads not yet implemented (plan 03-03 pending): "
            f"{_import_error}",
            pytrace=False,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_feature_maps(strides, channels, tile_size=224):
    """
    Synthesise a list of feature-map tensors matching the declared D-05
    contract: (B=1, C, H/stride, W/stride).
    """
    import torch
    return [
        torch.randn(1, ch, tile_size // s, tile_size // s)
        for s, ch in zip(strides, channels)
    ]


# ViT-like single-scale config (SigLIP or DINOv2 style: one level at stride 14)
_VIT_STRIDES = [14]
_VIT_CHANNELS = [1152]

# Swin-like multi-scale config: 4 levels at strides 4/8/16/32
_SWIN_STRIDES = [4, 8, 16, 32]
_SWIN_CHANNELS = [128, 256, 512, 1024]


# ---------------------------------------------------------------------------
# Decoder + heads output-shape tests
# ---------------------------------------------------------------------------

class TestDecoderHeadShapes:
    """Decoder + 2 heads emit (B,9,H,W) and (B,3,H,W) at full tile resolution."""

    @pytest.mark.parametrize("tile_size", [224, 448, 896])
    def test_vit_decoder_land_cover_shape(self, tile_size):
        _require_decoder()
        import torch
        feats = _make_feature_maps(_VIT_STRIDES, _VIT_CHANNELS, tile_size)
        decoder = SegDecoder(
            feature_strides=_VIT_STRIDES,
            feature_channels=_VIT_CHANNELS,
            tile_size=tile_size,
        )
        lc_head = LandCoverHead(decoder.out_channels)
        topo_head = TopographyHead(decoder.out_channels)
        with torch.no_grad():
            dense = decoder(feats)
            lc_logits = lc_head(dense)
            topo_logits = topo_head(dense)
        assert lc_logits.shape == (1, 9, tile_size, tile_size), (
            f"LC head shape mismatch: expected (1,9,{tile_size},{tile_size}), "
            f"got {lc_logits.shape}"
        )
        assert topo_logits.shape == (1, 3, tile_size, tile_size), (
            f"Topo head shape mismatch: expected (1,3,{tile_size},{tile_size}), "
            f"got {topo_logits.shape}"
        )

    @pytest.mark.parametrize("tile_size", [224, 448, 896])
    def test_swin_decoder_shapes(self, tile_size):
        """Same decoder runs on Swin's 4-level feature pyramid (Pitfall 4 guard)."""
        _require_decoder()
        import torch
        feats = _make_feature_maps(_SWIN_STRIDES, _SWIN_CHANNELS, tile_size)
        decoder = SegDecoder(
            feature_strides=_SWIN_STRIDES,
            feature_channels=_SWIN_CHANNELS,
            tile_size=tile_size,
        )
        lc_head = LandCoverHead(decoder.out_channels)
        topo_head = TopographyHead(decoder.out_channels)
        with torch.no_grad():
            dense = decoder(feats)
            lc_out = lc_head(dense)
            topo_out = topo_head(dense)
        assert lc_out.shape == (1, 9, tile_size, tile_size)
        assert topo_out.shape == (1, 3, tile_size, tile_size)

    def test_decoder_out_channels_attribute(self):
        """SegDecoder exposes out_channels so heads can be built without magic numbers."""
        _require_decoder()
        decoder = SegDecoder(
            feature_strides=_VIT_STRIDES,
            feature_channels=_VIT_CHANNELS,
            tile_size=224,
        )
        assert hasattr(decoder, "out_channels"), (
            "SegDecoder must expose out_channels for head construction"
        )
        assert decoder.out_channels > 0

    def test_batch_dimension_propagates(self):
        """Batch size B > 1 flows through decoder + heads correctly."""
        _require_decoder()
        import torch
        B = 3
        feats = [torch.randn(B, 1152, 16, 16)]  # ViT at 224 / stride 14
        decoder = SegDecoder(
            feature_strides=[14],
            feature_channels=[1152],
            tile_size=224,
        )
        lc_head = LandCoverHead(decoder.out_channels)
        with torch.no_grad():
            dense = decoder(feats)
            lc_out = lc_head(dense)
        assert lc_out.shape[0] == B, f"batch size must propagate: expected {B}, got {lc_out.shape[0]}"


# ---------------------------------------------------------------------------
# Head independence (D-02)
# ---------------------------------------------------------------------------

class TestHeadIndependence:
    """Two heads share one decoder trunk (D-02) — thin 1x1-conv wrappers."""

    def test_heads_are_distinct_modules(self):
        _require_decoder()
        decoder = SegDecoder(
            feature_strides=_VIT_STRIDES,
            feature_channels=_VIT_CHANNELS,
            tile_size=224,
        )
        lc = LandCoverHead(decoder.out_channels)
        topo = TopographyHead(decoder.out_channels)
        assert lc is not topo, "LandCoverHead and TopographyHead must be distinct instances"
        # A single forward produces both outputs from the same dense features
        import torch
        feats = _make_feature_maps(_VIT_STRIDES, _VIT_CHANNELS, 224)
        with torch.no_grad():
            dense = decoder(feats)
            lc_out = lc(dense)
            topo_out = topo(dense)
        assert lc_out.shape[1] == 9
        assert topo_out.shape[1] == 3
