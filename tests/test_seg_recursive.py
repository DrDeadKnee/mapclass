"""
Offline unit tests for ``scripts/seg/recursive.py`` (plan 03-04, D-03/D-04).

All tests are fully offline; geometry uses the ``mini_pyramid`` conftest
fixture which calls the real ``tiling.tile()`` producer (D-04: never
hand-write geometry).

Covered decisions: D-03 (explicit recursive prior as extra input channels),
D-03a Variant A (frozen backbones + decoder-level prior-encoder injection),
D-04 (inference walks pyramid.json parent→child literally), D-06
(construction-only, no training or backward).

Tests verify:
  - A known one-hot blob placed in a parent-896 quadrant lands in the correct
    child-448 pyramid tile after the pyramid.json-box crop+resize (extends
    tests/test_tiling.py::test_nested_alignment, guards Pitfall 5).
  - The 896 cold-start prior is all-zeros (D-03 spec: coarse tile has no
    parent, prior = zero).
  - Variant A's 12-channel prior-encoder injects at decoder working resolution
    and produces a tensor at the expected spatial size.

These tests fail with ImportError until ``scripts/seg/recursive.py`` lands
(plan 03-04).
"""

import json

import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------
_import_error: Exception | None = None
try:
    from seg.recursive import (
        crop_prior_to_child,
        cold_start_prior,
        PriorEncoder,
    )
except ImportError as _e:
    _import_error = _e


def _require_recursive():
    if _import_error is not None:
        pytest.fail(
            f"seg.recursive not yet implemented (plan 03-04 pending): {_import_error}",
            pytrace=False,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_manifest(pyramid_dir):
    return json.loads((pyramid_dir / "pyramid.json").read_text())


# ---------------------------------------------------------------------------
# Quadrant-alignment: known blob in parent lands in correct child (Pitfall 5)
# ---------------------------------------------------------------------------

class TestPriorCropAlignment:
    """
    Prior crop must select the correct quadrant, not a rescaled whole-parent
    (Pitfall 5: prior crop must crop-then-resize, not resize-the-whole-parent).
    """

    def test_known_blob_lands_in_correct_child(self, mini_pyramid):
        """
        Place a one-hot blob in the top-left quadrant of a synthetic parent-896
        probability tensor.  After crop_prior_to_child the blob must appear
        in child tile 448_0 (quadrant (0,0)) and NOT in 448_1/448_2/448_3.

        Extends test_tiling.py::test_nested_alignment with a probabilistic
        content-alignment assertion (RESEARCH Pitfall 5 guard).
        """
        _require_recursive()
        import torch

        man = _load_manifest(mini_pyramid)
        by_id = {t["id"]: t for t in man["tiles"]}
        parent = by_id["896"]
        children = [by_id[cid] for cid in parent["children"]]

        # Synthesise a (1, 12, 896, 896) parent probability map
        B, C, S = 1, 12, 896
        parent_prob = torch.zeros(B, C, S, S)

        # Place a unit-value blob in the top-left quadrant (0,0) = child 448_0
        parent_prob[:, 0, 0:10, 0:10] = 1.0  # channel 0 hot in (0,0) corner

        # The four children are ordered (0,0),(1,0),(0,1),(1,1) per tiling.py
        # Child index 0 = quadrant (0,0) = top-left
        child_0 = children[0]  # 448_0: top-left
        child_1 = children[1]  # 448_1: top-right
        child_2 = children[2]  # 448_2: bottom-left
        child_3 = children[3]  # 448_3: bottom-right

        ox, oy = parent["x"], parent["y"]

        crop_0 = crop_prior_to_child(parent_prob, parent, child_0)
        crop_1 = crop_prior_to_child(parent_prob, parent, child_1)
        crop_2 = crop_prior_to_child(parent_prob, parent, child_2)
        crop_3 = crop_prior_to_child(parent_prob, parent, child_3)

        # The blob must be in child_0 (top-left) and NOT in the other three
        assert crop_0[:, 0].max().item() > 0.0, (
            "Known blob in parent (0,0) quadrant must appear in child 448_0"
        )
        assert crop_1[:, 0].max().item() == 0.0, (
            "Blob must NOT leak into child 448_1 (top-right)"
        )
        assert crop_2[:, 0].max().item() == 0.0, (
            "Blob must NOT leak into child 448_2 (bottom-left)"
        )
        assert crop_3[:, 0].max().item() == 0.0, (
            "Blob must NOT leak into child 448_3 (bottom-right)"
        )

    def test_crop_output_shape(self, mini_pyramid):
        """crop_prior_to_child must return a tensor at the child's tile size."""
        _require_recursive()
        import torch

        man = _load_manifest(mini_pyramid)
        by_id = {t["id"]: t for t in man["tiles"]}
        parent = by_id["896"]
        child = by_id[parent["children"][0]]

        parent_prob = torch.zeros(1, 12, 896, 896)
        crop = crop_prior_to_child(parent_prob, parent, child)
        # Must be resized to the child's size (448 x 448) — align to child grid
        child_s = child["size"]  # 448
        assert crop.shape[-2] == child_s and crop.shape[-1] == child_s, (
            f"crop_prior_to_child must resize to child size {child_s}x{child_s}, "
            f"got {crop.shape[-2]}x{crop.shape[-1]}"
        )

    def test_crop_uses_manifest_boxes_not_recomputed(self, mini_pyramid):
        """
        The crop coordinates must come from pyramid.json x/y/size literals
        (D-04 mandate), not re-derived geometry.  We test this indirectly by
        checking that two different children with the same size but different
        positions yield distinct crops on a spatially-varying prior.
        """
        _require_recursive()
        import torch

        man = _load_manifest(mini_pyramid)
        by_id = {t["id"]: t for t in man["tiles"]}
        parent = by_id["896"]
        children = [by_id[cid] for cid in parent["children"]]

        # Gradient ramp so spatial position matters
        prob = torch.zeros(1, 12, 896, 896)
        for i in range(896):
            prob[0, 0, i, :] = i / 896.0  # row-gradient

        crop_0 = crop_prior_to_child(prob, parent, children[0])  # top-left
        crop_2 = crop_prior_to_child(prob, parent, children[2])  # bottom-left
        # Bottom child has larger row-index values → higher mean
        assert crop_2.mean() > crop_0.mean(), (
            "Bottom child crop must have higher mean (larger row values) "
            "than top child; manifest-box crop must use correct y offsets"
        )


# ---------------------------------------------------------------------------
# Cold-start prior at 896 (D-03)
# ---------------------------------------------------------------------------

class TestColdStartPrior:
    """At 896 coarse level there is no parent; prior must be all-zeros."""

    def test_cold_start_is_zeros(self):
        _require_recursive()
        import torch
        prior = cold_start_prior(batch=1, tile_size=896)
        assert isinstance(prior, torch.Tensor), "cold_start_prior must return a Tensor"
        # 12 channels = 9 LC + 3 topo softmax probs (RESEARCH Pattern 4)
        assert prior.shape == (1, 12, 896, 896), (
            f"cold_start_prior shape must be (1,12,896,896), got {prior.shape}"
        )
        assert prior.abs().max().item() == 0.0, (
            "cold_start_prior must be all-zeros (D-03 spec)"
        )

    @pytest.mark.parametrize("batch,tile_size", [(1, 448), (2, 224)])
    def test_cold_start_shape_variants(self, batch, tile_size):
        _require_recursive()
        prior = cold_start_prior(batch=batch, tile_size=tile_size)
        assert prior.shape == (batch, 12, tile_size, tile_size)


# ---------------------------------------------------------------------------
# Variant A: prior-encoder injects at decoder working resolution (D-03a)
# ---------------------------------------------------------------------------

class TestVariantAPriorEncoder:
    """
    Variant A: frozen backbones; a small trainable PriorEncoder maps the
    12-ch softmax prior to the decoder's working resolution and concatenates/
    sums there (D-03a: decoder-level injection, backbone-agnostic).
    """

    def test_prior_encoder_output_shape(self):
        """PriorEncoder maps (B,12,H,W) prior → (B,C_dec,h_dec,w_dec)."""
        _require_recursive()
        import torch
        # Decoder working resolution: 1/4 of tile (e.g., for a 14-stride ViT
        # on 224-tile → 16x16 grid; prior-encoder aligns to it)
        prior = torch.zeros(1, 12, 448, 448)
        # typical decoder working grid for a 448-tile: 32x32 (448/14 = 32)
        target_h = target_w = 32
        encoder = PriorEncoder(
            in_channels=12,
            out_channels=256,  # arbitrary decoder channel width
            target_size=(target_h, target_w),
        )
        with torch.no_grad():
            out = encoder(prior)
        assert out.shape == (1, 256, target_h, target_w), (
            f"PriorEncoder output must be (1,256,{target_h},{target_w}), got {out.shape}"
        )

    def test_prior_encoder_trainable(self):
        """PriorEncoder must have trainable parameters (it is the small learned piece)."""
        _require_recursive()
        encoder = PriorEncoder(
            in_channels=12,
            out_channels=256,
            target_size=(32, 32),
        )
        params = list(encoder.parameters())
        assert params, "PriorEncoder must have at least one trainable parameter"
        trainable = sum(p.numel() for p in params if p.requires_grad)
        assert trainable > 0, "PriorEncoder must have trainable parameters"
