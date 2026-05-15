"""
Offline unit tests for ``scripts/seg/dataset.py`` (plan 03-04, D-06).

All tests use the ``mini_pyramid`` conftest fixture — a real Phase-2 pyramid
produced by ``tiling.tile()`` under ``tmp_path``.  No network, no S3, no model
checkpoint required.

Covered decisions: D-06 (Phase-2 pyramid dataset, train-only), EVAL-01
(frozen split leakage guard), RESEARCH Pitfall 6 (no glob of test/).

Tests verify:
  - PyramidDataset surfaces ``sample_weights.json`` contents unchanged in each
    sample (D-claude-discretion: per-source weights propagate byte-identically).
  - Dataset yields correctly-shaped tensor batches:
      image  → (3, H, W) float32 in [0, 1]
      land_cover  → (H, W) long or (1, H, W) long
      topography  → (H, W) long or (1, H, W) long
  - No enumerated path contains a ``test/`` component (EVAL-01 split-safety;
    mirrors tests/test_tiling.py:213-215 and tests/test_split.py).
  - ``len(dataset) >= 1`` for a non-empty pyramid tree.
  - Raises ``ValueError`` (or similar) when pointed at an empty/missing tree
    (project idiom from ToonDataset, finetune_paligemma.py lines 93-94).

These tests fail with ImportError until ``scripts/seg/dataset.py`` lands
(plan 03-04).
"""

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------
_import_error: Exception | None = None
try:
    from seg.dataset import PyramidDataset
except ImportError as _e:
    _import_error = _e


def _require_dataset():
    if _import_error is not None:
        pytest.fail(
            f"seg.dataset not yet implemented (plan 03-04 pending): {_import_error}",
            pytrace=False,
        )


# ---------------------------------------------------------------------------
# Dataset construction and length
# ---------------------------------------------------------------------------

class TestPyramidDataset:
    """PyramidDataset over a real mini pyramid reads tiles and surfaces weights."""

    def test_dataset_nonempty(self, mini_pyramid):
        _require_dataset()
        ds = PyramidDataset(mini_pyramid)
        assert len(ds) >= 1, (
            "PyramidDataset must have at least one sample for a non-empty pyramid"
        )

    def test_dataset_getitem_keys(self, mini_pyramid):
        """Each sample must contain 'image', 'land_cover', 'topography', 'sample_weights'."""
        _require_dataset()
        ds = PyramidDataset(mini_pyramid)
        sample = ds[0]
        for key in ("image", "land_cover", "topography", "sample_weights"):
            assert key in sample, f"dataset sample missing key '{key}'"

    def test_image_shape_and_dtype(self, mini_pyramid):
        """Image tensor: (3, H, W) float32, normalised to [0, 1]."""
        _require_dataset()
        import torch
        ds = PyramidDataset(mini_pyramid)
        sample = ds[0]
        img = sample["image"]
        assert isinstance(img, torch.Tensor), "image must be a Tensor"
        assert img.ndim == 3 and img.shape[0] == 3, (
            f"image must be (3, H, W), got {img.shape}"
        )
        assert img.dtype == torch.float32, f"image must be float32, got {img.dtype}"
        assert img.min() >= 0.0 and img.max() <= 1.0, (
            "image must be normalised to [0, 1]"
        )

    def test_label_shapes(self, mini_pyramid):
        """land_cover and topography tensors must have spatial dims matching image."""
        _require_dataset()
        import torch
        ds = PyramidDataset(mini_pyramid)
        sample = ds[0]
        img = sample["image"]
        lc = sample["land_cover"]
        topo = sample["topography"]
        H, W = img.shape[-2], img.shape[-1]
        # Accept either (H, W) or (1, H, W)
        assert lc.shape[-2:] == (H, W), (
            f"land_cover spatial dims must match image ({H},{W}), got {lc.shape}"
        )
        assert topo.shape[-2:] == (H, W), (
            f"topography spatial dims must match image ({H},{W}), got {topo.shape}"
        )

    def test_sample_weights_surfaced_unchanged(self, mini_pyramid):
        """
        ``sample_weights`` must be surfaced byte-identically from
        ``sample_weights.json`` in the pyramid directory (D-claude-discretion).
        """
        _require_dataset()
        ds = PyramidDataset(mini_pyramid)
        sample = ds[0]
        # Load the ground-truth weights blob from the pyramid directory
        weights_path = mini_pyramid / "sample_weights.json"
        expected = json.loads(weights_path.read_text())
        surfaced = sample["sample_weights"]
        # Accept either the raw dict or a JSON-serialised string round-trip
        if isinstance(surfaced, dict):
            assert surfaced == expected, (
                "surfaced sample_weights dict must equal the pyramid's "
                "sample_weights.json contents"
            )
        elif isinstance(surfaced, str):
            assert json.loads(surfaced) == expected, (
                "surfaced sample_weights (string) must JSON-decode to the "
                "pyramid's sample_weights.json contents"
            )
        else:
            pytest.fail(
                f"sample_weights must be a dict or JSON string, got {type(surfaced)}"
            )


# ---------------------------------------------------------------------------
# Split safety (EVAL-01 — no test/ leakage)
# ---------------------------------------------------------------------------

class TestSplitSafety:
    """
    Dataset must never enumerate a path containing a ``test/`` component.
    Mirrors tests/test_tiling.py:213-215 and tests/test_split.py leakage guard.
    """

    def test_no_test_path_in_dataset(self, mini_pyramid):
        """
        The mini_pyramid is under tmp_path (no ``test/`` component). Verify
        that PyramidDataset does not accidentally walk outside the given root
        and that no path in the dataset contains ``/test/``.
        """
        _require_dataset()
        ds = PyramidDataset(mini_pyramid)
        for i in range(len(ds)):
            path = ds.tile_path(i)  # returns a Path or str for the tile directory
            parts = Path(path).parts
            assert "test" not in parts, (
                f"Dataset sample {i} has a path containing 'test/' component: {path}\n"
                "Dataset must only enumerate paths under the train/ subtree (EVAL-01)."
            )

    def test_dataset_pointed_at_train_root(self, tmp_path):
        """
        Dataset pointed at a synthetic ``train/`` subtree must only enumerate
        pyramids under that subtree.  Sibling ``test/`` pyramids must NOT appear.
        """
        _require_dataset()
        from tiling import tile
        from PIL import Image
        import json as _json

        WEIGHTS = _json.dumps({"land_cover_weights": {}, "topography_weight": 1.0,
                               "source": "synthetic", "map_file": "f"}, indent=2)

        def _make(base):
            base.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (896, 896), (10, 20, 30)).save(base / "image.png")
            Image.new("L", (896, 896), 1).save(base / "land_cover.png")
            Image.new("L", (896, 896), 2).save(base / "topography.png")
            (base / "sample_weights.json").write_text(WEIGHTS)
            return tile(base)

        train_pyr = _make(tmp_path / "synthetic" / "train" / "map_a")
        test_pyr = _make(tmp_path / "synthetic" / "test" / "map_b")

        train_dirs = [p for p in train_pyr.iterdir() if p.is_dir()]
        assert train_dirs, "No pyramids produced for train map"

        ds = PyramidDataset(train_dirs[0])
        for i in range(len(ds)):
            path = Path(ds.tile_path(i))
            assert "test" not in path.parts, (
                f"Train-only dataset contains a path with 'test/' component: {path}"
            )


# ---------------------------------------------------------------------------
# Empty-tree guard
# ---------------------------------------------------------------------------

class TestEmptyDataset:
    """Raises ValueError (project idiom) for an empty pyramid directory."""

    def test_raises_on_empty_dir(self, tmp_path):
        _require_dataset()
        # An empty directory with no tiles
        empty = tmp_path / "empty_pyramid"
        empty.mkdir()
        with pytest.raises((ValueError, FileNotFoundError)):
            PyramidDataset(empty)
