"""Unit checks for src/mapclass/overlay.py (ATTR-02, Pitfall D, D-07, D-09).

  * ``to_patch_grid`` on a ``(1,3,384,384)`` tensor returns a signed grid AND
    a magnitude grid, each EXACTLY ``(27,27)``.
  * The valid region is ``[:378,:378]`` — the right/bottom 6-px band is
    discarded (Pitfall D / ``padding="valid"``).
  * D-09: NO ``.abs()`` on the signed path and NO min-max to ``[0,1]`` — a
    fixture with known-negative relevance yields a signed grid that CONTAINS
    negative values; the magnitude grid is ``>= 0``.
  * ``draw_patch_grid`` exists and (by source) marks the excluded 6-px band.
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


def test_grid_geometry_constants():
    assert overlay.GRID == 27
    assert overlay.VALID_DIM == 378
    assert overlay.EDGE_DISCARD == 6


def test_to_patch_grid_shapes():
    rel = torch.randn(1, 3, 384, 384)
    grid_signed, grid_mag = overlay.to_patch_grid(rel)
    assert grid_signed.shape == (27, 27)
    assert grid_mag.shape == (27, 27)


def test_signed_grid_keeps_sign_no_minmax():
    """D-09: known-negative relevance must survive as negative in the signed
    grid (no .abs(), no min-max [0,1]); the magnitude grid is >= 0."""
    rel = torch.full((1, 3, 384, 384), -2.0)  # strictly negative everywhere
    grid_signed, grid_mag = overlay.to_patch_grid(rel)

    # Signed path preserves the negative sign (3 channels × -2 = -6 per pixel).
    assert (grid_signed < 0).all()
    assert np.isclose(grid_signed.mean(), -6.0)
    # NOT min-maxed into [0,1].
    assert grid_signed.min() < 0.0
    # Magnitude path is non-negative.
    assert (grid_mag >= 0).all()
    assert np.isclose(grid_mag.mean(), 6.0)


def test_mixed_sign_grid_straddles_zero():
    rel = torch.zeros(1, 3, 384, 384)
    rel[:, :, :192, :] = 5.0    # top half positive
    rel[:, :, 192:, :] = -5.0   # bottom half negative
    grid_signed, _ = overlay.to_patch_grid(rel)
    assert grid_signed.max() > 0.0
    assert grid_signed.min() < 0.0


def test_valid_slice_is_378():
    """The 6-px-discarded valid region is [:378,:378]: a relevance tensor that
    is large only in the discarded right/bottom band must NOT affect the
    grid."""
    rel = torch.zeros(1, 3, 384, 384)
    rel[:, :, 378:, :] = 1000.0   # bottom 6-px band only
    rel[:, :, :, 378:] = 1000.0   # right 6-px band only
    grid_signed, grid_mag = overlay.to_patch_grid(rel)
    assert np.allclose(grid_signed, 0.0)
    assert np.allclose(grid_mag, 0.0)


def test_no_abs_on_signed_path_in_source():
    # The signed reduction must be a plain channel sum (no abs on that line).
    assert "r_signed = r.sum(0)" in _OVL_SRC
    assert "r_mag = r.abs().sum(0)" in _OVL_SRC


def test_no_minmax_normalization_in_source():
    # Guard against a reintroduced min-max-to-[0,1] (the D-09 defect).
    assert "grid.min()" not in _OVL_SRC
    assert "- grid.min()" not in _OVL_SRC


def test_draw_patch_grid_marks_excluded_band():
    assert hasattr(overlay, "draw_patch_grid")
    assert "excluded" in _OVL_SRC and "EDGE_DISCARD" in _OVL_SRC


def test_reshape_27_present_in_source():
    # must_haves artifact contract: overlay.py contains "reshape(27".
    assert "reshape(27" in _OVL_SRC
