"""Unit checks for src/mapclass/overlay.py (ATTR-02, Pitfall D, D-07, D-09).

``to_patch_grid`` is now GEOMETRY-PARAMETERIZED — ``to_patch_grid(relevance,
patch_size, img_dim, has_cls=False)`` — so each model reconstructs its own
patch grid from its own patch size (NOT a hardcoded 27x27):

  * SigLIP-2  (14 / 384 / no-cls) -> 27x27
  * CLIP-L/14 (14 / 224 / cls)    -> 16x16
  * ViT-b-16  (16 / 224 / cls)    -> 14x14

D-09 invariant preserved EXACTLY: signed grid == channel-sum with NO
``.abs()`` and NO min-max to ``[0,1]``; magnitude grid == ``.abs()``
channel-sum (``>= 0``). ``has_cls=True`` strips token 0 ONLY when relevance
arrives token-shaped ``(1, ntok, ...)``; the pixel-space ``(1,3,H,W)`` path
(the real path for all four image models) is unchanged.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (
    _REPO_ROOT / "third_party" / "dynamicLRP" / "src",
    _REPO_ROOT / "src",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

torch = pytest.importorskip("torch")

from mapclass import overlay  # noqa: E402

_OVL_SRC = (_REPO_ROOT / "src" / "mapclass" / "overlay.py").read_text()


def test_grid_geometry_constants_default_siglip2():
    """The module-level SigLIP-2 defaults are unchanged (v1.0 back-compat)."""
    assert overlay.GRID == 27
    assert overlay.VALID_DIM == 378
    assert overlay.EDGE_DISCARD == 6


def test_to_patch_grid_default_is_siglip2_27x27():
    """Called WITHOUT geometry args -> SigLIP-2 14/384 -> 27x27 (back-compat)."""
    rel = torch.randn(1, 3, 384, 384)
    grid_signed, grid_mag = overlay.to_patch_grid(rel)
    assert grid_signed.shape == (27, 27)
    assert grid_mag.shape == (27, 27)


@pytest.mark.parametrize(
    "patch_size,img_dim,grid",
    [
        (14, 384, 27),  # SigLIP-2
        (14, 224, 16),  # CLIP-L/14
        (16, 224, 14),  # ViT-b-16
    ],
)
def test_to_patch_grid_parameterized_geometry(patch_size, img_dim, grid):
    rel = torch.randn(1, 3, img_dim, img_dim)
    gs, gm = overlay.to_patch_grid(
        rel, patch_size=patch_size, img_dim=img_dim, has_cls=False
    )
    assert gs.shape == (grid, grid)
    assert gm.shape == (grid, grid)


def test_signed_grid_keeps_sign_no_minmax():
    """D-09: known-negative relevance survives as negative; mag is >= 0."""
    rel = torch.full((1, 3, 224, 224), -2.0)
    gs, gm = overlay.to_patch_grid(rel, patch_size=14, img_dim=224)
    assert (gs < 0).all()
    assert np.isclose(gs.mean(), -6.0)  # 3 channels x -2
    assert gs.min() < 0.0  # NOT min-maxed into [0,1]
    assert (gm >= 0).all()
    assert np.isclose(gm.mean(), 6.0)


def test_mixed_sign_grid_straddles_zero():
    rel = torch.zeros(1, 3, 384, 384)
    rel[:, :, :192, :] = 5.0
    rel[:, :, 192:, :] = -5.0
    gs, _ = overlay.to_patch_grid(rel, patch_size=14, img_dim=384)
    assert gs.max() > 0.0
    assert gs.min() < 0.0


def test_valid_slice_discards_edge_band():
    """The right/bottom discarded band must NOT affect the grid."""
    rel = torch.zeros(1, 3, 384, 384)
    rel[:, :, 378:, :] = 1000.0
    rel[:, :, :, 378:] = 1000.0
    gs, gm = overlay.to_patch_grid(rel, patch_size=14, img_dim=384)
    assert np.allclose(gs, 0.0)
    assert np.allclose(gm, 0.0)


def test_has_cls_strips_token0_when_token_shaped():
    """has_cls=True on a token-shaped (1, ntok, ...) relevance drops index 0.

    14x14 + 1 CLS = 197 tokens for a ViT-b-16-like relevance carrying a
    per-channel feature dim; token 0 is the CLS and must be discarded BEFORE
    the spatial reshape.
    """
    grid = 14
    ntok = grid * grid + 1  # 197
    feat = 3
    rel = torch.zeros(1, ntok, feat)
    rel[:, 0, :] = 999.0  # CLS token — must be stripped, must NOT leak
    rel[:, 1:, :] = 1.0
    gs, gm = overlay.to_patch_grid(
        rel, patch_size=16, img_dim=224, has_cls=True
    )
    assert gs.shape == (grid, grid)
    # CLS (999) stripped -> every spatial cell == feat * 1.0 == 3.0
    assert np.allclose(gs, float(feat))
    assert gm.shape == (grid, grid)


def test_no_abs_on_signed_path_in_source():
    assert "r_signed = r.sum(0)" in _OVL_SRC
    assert "r_mag = r.abs().sum(0)" in _OVL_SRC


def test_no_minmax_normalization_in_source():
    assert "grid.min()" not in _OVL_SRC
    assert "- grid.min()" not in _OVL_SRC


def test_draw_patch_grid_marks_excluded_band():
    assert hasattr(overlay, "draw_patch_grid")
    assert "excluded" in _OVL_SRC and "EDGE_DISCARD" in _OVL_SRC


def test_to_patch_grid_is_geometry_parameterized_in_source():
    assert "def to_patch_grid(relevance, patch_size" in _OVL_SRC or (
        "def to_patch_grid(\n    relevance, patch_size" in _OVL_SRC
    )
