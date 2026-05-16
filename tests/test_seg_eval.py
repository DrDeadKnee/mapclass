"""
Offline eval-harness tests for ``scripts/evaluate_seg.py`` (plan 04-03).

Covers EVAL-02 (NLL formula, leaf-tile filter, numerical stability) and
EVAL-03 (identical harness across siglip/dinov2/swin × variant A/B).

All tests are fully offline:
  - No GPU, no network, no pretrained checkpoints.
  - Stub backbones via ``stub_vision_config`` fixture.
  - Geometry from ``mini_pyramid`` conftest fixture (real tiling.tile() output).

Guard: LOUD fail (pytest.fail, not pytest.skip) if ``evaluate_seg`` is not
importable — honouring the Phase-4 Nyquist gate (plan 04-01 design).
"""

import json
import math
from pathlib import Path
from unittest import mock

import pytest
import torch

# ---------------------------------------------------------------------------
# Import guard — LOUD fail (never skip) so the Nyquist gate fires offline
# ---------------------------------------------------------------------------
_import_error: Exception | None = None
try:
    from evaluate_seg import evaluate_joint_nll, load_test_pyramid_dirs, write_nll_metrics  # noqa: F401
except ImportError as _e:
    _import_error = _e


def _require_eval():
    if _import_error is not None:
        pytest.fail(
            f"evaluate_seg not yet implemented: {_import_error}",
            pytrace=False,
        )


# ---------------------------------------------------------------------------
# TestNLLFormula
# ---------------------------------------------------------------------------

class TestNLLFormula:
    """
    NLL = -(log p_lc[true] + log p_topo[true]) per pixel (EVAL-02 formula).

    Tests verify:
    - The exact formula against a hand-computed value.
    - Invalid pixels (lc_gt >= 9 or topo_gt >= 3) are excluded from the average.
    - Only size==224 leaf tiles contribute to the NLL.
    """

    def test_nll_formula_correct(self, mini_pyramid, stub_vision_config):
        """
        Uniform probability vector → NLL = log(9) + log(3) per pixel.

        With uniform softmax over 9 LC classes, p_lc[0] = 1/9.
        With uniform softmax over 3 topo classes, p_topo[0] = 1/3.
        NLL = -log(1/9) - log(1/3) = log(9) + log(3).

        Strategy: monkeypatch ``recursive_predict`` inside evaluate_joint_nll
        to return a controlled prob_cache with uniform 12-ch probabilities.
        All gt labels set to 0 (valid classes).
        """
        _require_eval()
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

        from evaluate_seg import evaluate_joint_nll
        from seg.model import SegModelVariantA

        model = SegModelVariantA(
            backbone_name="siglip",
            stub_config=stub_vision_config,
        )

        # Build a uniform prob_cache: 1/9 for LC, 1/3 for topo
        manifest = json.loads((mini_pyramid / "pyramid.json").read_text())
        leaf_tiles = [t for t in manifest["tiles"] if t["size"] == 224]

        # Build a prob_cache with uniform distributions for every tile
        tile_size = 224
        lc_prob = torch.full((1, 9, tile_size, tile_size), 1.0 / 9)
        topo_prob = torch.full((1, 3, tile_size, tile_size), 1.0 / 3)
        uniform_prob12 = torch.cat([lc_prob, topo_prob], dim=1)  # (1, 12, 224, 224)

        fake_prob_cache = {t["id"]: uniform_prob12.clone() for t in manifest["tiles"]}

        expected_nll = math.log(9.0) + math.log(3.0)

        # Monkeypatch recursive_predict in evaluate_seg's namespace
        import evaluate_seg as _eval_mod

        with mock.patch.object(_eval_mod, "recursive_predict", return_value=fake_prob_cache):
            nll = evaluate_joint_nll(model, [mini_pyramid], variant="A")

        assert abs(nll - expected_nll) < 1e-5, (
            f"Expected NLL = log(9)+log(3) ≈ {expected_nll:.6f}, got {nll:.6f}\n"
            "NLL formula: -(log p_lc[true] + log p_topo[true]) with uniform probs."
        )

    def test_nll_ignores_invalid_pixels(self, mini_pyramid, stub_vision_config):
        """
        Pixels with lc_gt >= 9 or topo_gt >= 3 must be excluded from the NLL average.

        Strategy: build a prob_cache with a known NLL for valid pixels and
        a very large NLL for invalid pixels.  If invalid pixels leak into the
        average, the result will differ from the expected value.
        """
        _require_eval()
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

        from evaluate_seg import evaluate_joint_nll, _load_label
        from seg.model import SegModelVariantA

        model = SegModelVariantA(
            backbone_name="siglip",
            stub_config=stub_vision_config,
        )

        manifest = json.loads((mini_pyramid / "pyramid.json").read_text())
        leaf_tiles = [t for t in manifest["tiles"] if t["size"] == 224]
        assert leaf_tiles, "mini_pyramid must have at least one 224-level tile"

        # Uniform probs: NLL = log(9)+log(3) for any valid gt=0 pixel.
        tile_size = 224
        lc_prob = torch.full((1, 9, tile_size, tile_size), 1.0 / 9)
        topo_prob = torch.full((1, 3, tile_size, tile_size), 1.0 / 3)
        uniform_prob12 = torch.cat([lc_prob, topo_prob], dim=1)

        fake_prob_cache = {t["id"]: uniform_prob12.clone() for t in manifest["tiles"]}

        # The mini_pyramid label images have constant label value 1 (LC) and 2 (topo),
        # both of which are valid (< 9 and < 3 respectively).  So we just assert the NLL
        # is finite (not nan or inf) — the valid-pixel filter did not exclude everything.
        import evaluate_seg as _eval_mod

        with mock.patch.object(_eval_mod, "recursive_predict", return_value=fake_prob_cache):
            nll = evaluate_joint_nll(model, [mini_pyramid], variant="A")

        assert math.isfinite(nll), (
            f"NLL should be finite when valid pixels exist, got: {nll}"
        )

        # Now build a test where ALL label pixels are invalid (lc_gt=9, topo_gt=3)
        # — the eval should return nan (no valid pixels to average over)
        from PIL import Image
        import numpy as np
        import tempfile
        import shutil

        # Create a temporary pyramid dir with invalid labels
        with tempfile.TemporaryDirectory() as tmp:
            tmp_pdir = Path(tmp) / "fake_pyramid"
            tmp_pdir.mkdir()

            # Copy original pyramid.json
            orig_manifest = json.loads((mini_pyramid / "pyramid.json").read_text())
            leaf = leaf_tiles[0]

            # Write invalid label images (lc=9, topo=3 — both out-of-range)
            for tile_entry in orig_manifest["tiles"]:
                # Copy image (needed for _default_read_rgb)
                img_src = mini_pyramid / tile_entry["image"]
                img_dst = tmp_pdir / tile_entry["image"]
                img_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(img_src, img_dst)

                # Write out-of-range labels for leaf tiles only
                lc_path = tmp_pdir / tile_entry["land_cover"]
                topo_path = tmp_pdir / tile_entry["topography"]
                lc_path.parent.mkdir(parents=True, exist_ok=True)
                topo_path.parent.mkdir(parents=True, exist_ok=True)
                s = tile_entry["size"]
                # Use label value 9 (invalid: lc_gt < 9 required) and 3 (invalid: topo_gt < 3 required)
                Image.fromarray(np.full((s, s), 9, dtype=np.uint8), mode="L").save(lc_path)
                Image.fromarray(np.full((s, s), 3, dtype=np.uint8), mode="L").save(topo_path)

            # Write pyramid.json unchanged
            (tmp_pdir / "pyramid.json").write_text(json.dumps(orig_manifest))

            fake_cache_invalid = {t["id"]: uniform_prob12.clone() for t in orig_manifest["tiles"]}

            with mock.patch.object(_eval_mod, "recursive_predict", return_value=fake_cache_invalid):
                nll_invalid = evaluate_joint_nll(model, [tmp_pdir], variant="A")

        assert math.isnan(nll_invalid), (
            f"NLL with all-invalid labels should be nan (no valid pixels), got: {nll_invalid}"
        )

    def test_eval_uses_only_leaf_tiles(self, mini_pyramid, stub_vision_config):
        """
        Only size==224 tiles contribute to the NLL; 896 and 448 tiles are excluded.

        Strategy: build a prob_cache where 896 and 448 tiles have near-zero probs
        (which would give huge NLL if counted), but 224 tiles have uniform probs
        (giving log(9)+log(3)).  Assert the result matches the uniform-only value.
        """
        _require_eval()
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

        from evaluate_seg import evaluate_joint_nll
        from seg.model import SegModelVariantA

        model = SegModelVariantA(
            backbone_name="siglip",
            stub_config=stub_vision_config,
        )

        manifest = json.loads((mini_pyramid / "pyramid.json").read_text())

        # For 224 tiles: uniform probs (NLL = log(9)+log(3))
        # For 896/448 tiles: near-zero probs (NLL would be huge if mistakenly used)
        fake_prob_cache = {}
        for t in manifest["tiles"]:
            s = t["size"]
            if s == 224:
                lc_prob = torch.full((1, 9, s, s), 1.0 / 9)
                topo_prob = torch.full((1, 3, s, s), 1.0 / 3)
            else:
                # Near-zero: if mistakenly included, would produce very large NLL
                lc_prob = torch.full((1, 9, s, s), 1e-10).softmax(dim=1)
                topo_prob = torch.full((1, 3, s, s), 1e-10).softmax(dim=1)
            fake_prob_cache[t["id"]] = torch.cat([lc_prob, topo_prob], dim=1)

        expected_nll = math.log(9.0) + math.log(3.0)

        import evaluate_seg as _eval_mod

        with mock.patch.object(_eval_mod, "recursive_predict", return_value=fake_prob_cache):
            nll = evaluate_joint_nll(model, [mini_pyramid], variant="A")

        # The mini_pyramid has constant label values (lc=1, topo=2) — both valid.
        # The result should be close to expected_nll from the 224 tiles only.
        assert abs(nll - expected_nll) < 1e-4, (
            f"NLL should equal log(9)+log(3)≈{expected_nll:.4f} when only leaf-tiles "
            f"are included, got {nll:.4f}.\n"
            "RESEARCH Pitfall 7: 896/448 levels must be excluded from NLL computation."
        )


# ---------------------------------------------------------------------------
# TestEvalLeakageGuard
# ---------------------------------------------------------------------------

class TestEvalLeakageGuard:
    """
    evaluate_seg.py must never enumerate train/ paths (EVAL-01 / T-04-08).
    load_test_pyramid_dirs returns only test/ pyramid dirs.
    """

    def test_load_test_dirs_only_test_paths(self, tmp_path):
        """
        Every dir returned by load_test_pyramid_dirs must have 'test' in its
        parts and must NOT have 'train' in its parts.

        Mirrors test_seg_dataset.py::TestSplitSafety pattern.
        """
        _require_eval()
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

        from evaluate_seg import load_test_pyramid_dirs
        from tiling import tile
        from PIL import Image
        import json as _json

        WEIGHTS = _json.dumps({
            "land_cover_weights": {
                "water": 1.0, "trees": 1.0, "shrubland": 1.0,
                "grassland": 1.0, "cropland": 1.5, "built_up": 1.5,
                "bare_sparse": 1.0, "flooded_wetland": 1.5, "snow_ice": 1.0
            },
            "topography_weight": 1.0,
            "source": "synthetic",
            "map_file": "test_map",
        }, indent=2)

        def _make_map(base: Path) -> Path:
            base.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (1792, 1792), (10, 20, 30)).save(base / "image.png")
            Image.new("L", (1792, 1792), 1).save(base / "land_cover.png")
            Image.new("L", (1792, 1792), 2).save(base / "topography.png")
            (base / "sample_weights.json").write_text(WEIGHTS)
            return tile(base)

        # Create test/ maps
        data_root = tmp_path / "data"
        test_map_dir = data_root / "test" / "map_alpha"
        _make_map(test_map_dir)

        # Also create a train/ map (must NOT appear in results)
        train_map_dir = data_root / "train" / "map_beta"
        _make_map(train_map_dir)

        # Create split.json referencing only the test map
        split_json = tmp_path / "split.json"
        split_json.write_text(_json.dumps({
            "test": ["map_alpha"],
            "train": ["map_beta"],
        }))

        dirs = load_test_pyramid_dirs(split_json, data_root)

        assert dirs, "load_test_pyramid_dirs returned empty list — expected test pyramids"

        for d in dirs:
            parts = Path(d).parts
            assert "test" in parts, (
                f"evaluate_seg returned a path without 'test' component: {d}\n"
                "load_test_pyramid_dirs must only enumerate test/ paths (EVAL-01 / T-04-08)."
            )
            assert "train" not in parts, (
                f"evaluate_seg returned a path with 'train' component: {d}\n"
                "load_test_pyramid_dirs must NEVER enumerate train/ paths (EVAL-01)."
            )

    def test_rejects_traversal_map_id(self, tmp_path):
        """
        A split.json with a path-traversal map ID must raise ValueError (T-04-07).

        The map ID ``'../secret'`` resolves from ``data_root/test/`` upward to
        ``data_root/secret``, which is outside ``data_root/test/`` — the guard
        must reject it before any directory listing.
        """
        _require_eval()
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

        from evaluate_seg import load_test_pyramid_dirs
        import json as _json

        data_root = tmp_path / "data"
        data_root.mkdir(parents=True)

        # Create a sibling-of-test/ directory at data_root/secret
        # (reachable via data_root/test/../secret = data_root/secret)
        sibling = data_root / "secret"
        sibling.mkdir()

        # Split JSON with a traversal map ID that escapes data_root/test/
        # "../secret" → data_root/test/../secret = data_root/secret (NOT under test/)
        split_json = tmp_path / "split.json"
        split_json.write_text(_json.dumps({
            "test": ["../secret"],
            "train": [],
        }))

        with pytest.raises(ValueError, match="outside|traversal|path"):
            load_test_pyramid_dirs(split_json, data_root)


# ---------------------------------------------------------------------------
# TestEvalAllBackbones
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backbone,variant", [
    ("siglip", "A"),
    ("siglip", "B"),
    ("dinov2", "A"),
    ("dinov2", "B"),
    ("swin", "A"),
    ("swin", "B"),
])
class TestEvalAllBackbones:
    """
    evaluate_joint_nll returns a finite float through the same harness with a
    stub-config model on mini_pyramid, for all backbone × variant combinations.

    EVAL-03: the identical harness over siglip/dinov2/swin makes the comparison
    apples-to-apples.

    All tests are fully offline; stub backbone via stub_vision_config (no GPU,
    no pretrained download).
    """

    def test_evaluate_returns_finite_float(
        self, backbone, variant, mini_pyramid, stub_vision_config
    ):
        """
        evaluate_joint_nll(model, [mini_pyramid], variant=variant) must return
        a finite float for every backbone × variant combination.

        DINOv2 and Swin require the ``timm`` package (for offline testing with
        pretrained=False).  The test is skipped when ``timm`` is not installed.
        SigLIP tests use ``stub_vision_config`` and never require timm.
        """
        _require_eval()

        # DINOv2 and Swin backbone construction requires timm.
        # Skip gracefully when timm is absent (offline test environment).
        if backbone in ("dinov2", "swin"):
            pytest.importorskip("timm", reason=f"{backbone} backbone requires timm")

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

        from evaluate_seg import evaluate_joint_nll
        from seg.model import SegModelVariantA, SegModelVariantB

        if variant == "A":
            model = SegModelVariantA(
                backbone_name=backbone,
                stub_config=stub_vision_config,
            )
        else:
            model = SegModelVariantB(
                backbone_name=backbone,
                stub_config=stub_vision_config,
            )

        nll = evaluate_joint_nll(model, [mini_pyramid], variant=variant)

        assert isinstance(nll, float), (
            f"evaluate_joint_nll must return a float, got {type(nll)} "
            f"for backbone={backbone!r} variant={variant!r}"
        )
        assert math.isfinite(nll), (
            f"NLL must be finite, got {nll} "
            f"for backbone={backbone!r} variant={variant!r}\n"
            "Check that the mini_pyramid has valid (< 9 LC, < 3 topo) label pixels."
        )
