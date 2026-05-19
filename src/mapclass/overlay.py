"""Patch-relevance → 27×27 grid → un-squashed alpha-blended composite (ATTR-02,
VIZ-01, Pitfall D, D-07, D-09).

Geometry (Pitfall D): the SigLIP-2 vision tower uses ``patch_size=14`` on a
``384×384`` input. ``floor(384/14) = 27`` → ``27×27 = 729`` patches; the
convolutional patch embed uses ``padding="valid"``, so the **last 6 px of the
right and bottom edges** (``384 − 27·14 = 6``) are discarded. The processor
*squashes* (not letterboxes) the non-square Rumsey scan to the 384 square, so
the composite must invert that square-squash back to the original PIL aspect
ratio before overlaying.

D-09 (signed-relevance — user-flagged 2026-05-18, established + verified in
``notebooks/00_vit_repro.ipynb`` cells 8-9): LRP relevance is SIGNED. The
signed reduction sums over channels WITHOUT ``.abs()`` and does NOT min-max to
``[0, 1]`` — physical units are preserved; scaling is purely a render concern.
The composite renders on a ZERO-CENTERED diverging colormap (``bwr`` with a
symmetric ``TwoSlopeNorm`` so white == 0) so negative relevance (evidence
AGAINST the query) is visible. A separate non-negative ``|relevance|``
magnitude grid is ALSO produced for the "where it concentrates" read and as
the optional alpha channel.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# Verified SigLIP-2 so400m-patch14-384 geometry.
PATCH_SIZE = 14
IMG_DIM = 384
GRID = IMG_DIM // PATCH_SIZE          # = 27
VALID_DIM = GRID * PATCH_SIZE         # = 378  (discard last 6 px R/B)
EDGE_DISCARD = IMG_DIM - VALID_DIM    # = 6


def to_patch_grid(relevance) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce ``(1,3,384,384)`` relevance → a SIGNED and a MAGNITUDE 27×27 grid.

    D-09: the signed path is ``relevance.sum(1)`` over channels with **NO
    ``.abs()``** and **NO min-max to ``[0, 1]``** — the returned signed grid
    keeps physical units (it can and should contain negative values for a
    meaningful LRP map). The magnitude path is ``relevance.abs().sum(1)`` and
    is ``>= 0`` by construction.

    Pitfall D: only the valid ``[:378, :378]`` region is used (the right /
    bottom 6-px band is discarded, ``padding="valid"``); each ``378×378``
    region is block-reduced ``reshape(27,14,27,14).mean((1,3))`` → ``(27,27)``.

    Returns ``(grid_signed, grid_mag)`` as float ndarrays, each ``(27, 27)``.
    """
    r = relevance.detach()[0]                       # (3, 384, 384)

    r_signed = r.sum(0)                             # (384,384) SIGNED — no abs
    r_mag = r.abs().sum(0)                          # (384,384) >= 0 magnitude

    # Discard the last 6 px of the right and bottom edges (padding="valid").
    # Concretely, with GRID=27 / PATCH_SIZE=14 this is:
    #   vs[:378,:378].reshape(27,14,27,14).mean((1,3))  -> (27,27) per-patch
    vs = r_signed[:VALID_DIM, :VALID_DIM]           # (378,378)
    vm = r_mag[:VALID_DIM, :VALID_DIM]              # (378,378)

    grid_signed = vs.reshape(
        GRID, PATCH_SIZE, GRID, PATCH_SIZE
    ).mean((1, 3))                                  # reshape(27,14,27,14) SIGNED
    grid_mag = vm.reshape(
        GRID, PATCH_SIZE, GRID, PATCH_SIZE
    ).mean((1, 3))                                  # reshape(27,14,27,14) >= 0

    # NO min-max normalization (D-09). float ndarrays only.
    return (
        grid_signed.cpu().numpy().astype(np.float64),
        grid_mag.cpu().numpy().astype(np.float64),
    )


def _sym_norm(arr: np.ndarray):
    """Zero-centered symmetric norm so white == 0 on the bwr diverging map.

    ``M = max(|arr|)`` → ``TwoSlopeNorm(vmin=-M, vcenter=0, vmax=M)`` (D-09).
    """
    from matplotlib.colors import TwoSlopeNorm

    m = float(np.nanmax(np.abs(arr))) if arr.size else 0.0
    if not np.isfinite(m) or m == 0.0:
        m = 1e-12
    return TwoSlopeNorm(vmin=-m, vcenter=0.0, vmax=m)


def composite(
    ax,
    pil,
    grid_signed: np.ndarray,
    grid_mag: Optional[np.ndarray] = None,
    *,
    title: Optional[str] = None,
    interpolation: str = "nearest",
):
    """Alpha-blend the SIGNED 27×27 grid onto the source map (un-squashed).

    The 27×27 grid is drawn over the original ``pil`` via ``imshow`` with
    ``extent`` spanning the full original image size, so matplotlib resamples
    the square 27×27 grid back onto the original (non-square) PIL aspect ratio
    — i.e. the square-squash done by the processor is inverted for display.

    D-09: ``cmap="bwr"`` with the zero-centered symmetric norm so white == 0
    and negative relevance is visible (NO min-max to ``[0,1]`` anywhere). If
    ``grid_mag`` is given it is used (normalized to ``[0,1]``) as the per-cell
    alpha so the overlay fades where there is little attribution.

    ``interpolation="nearest"`` is the HONEST patch-resolution primary view;
    pass ``interpolation="bilinear"`` for an explicitly-labeled smoothed
    secondary view.
    """
    w, h = pil.size
    ax.imshow(pil, extent=(0, w, h, 0))

    if grid_mag is not None:
        mag = np.asarray(grid_mag, dtype=np.float64)
        mmax = float(np.nanmax(mag)) if mag.size else 0.0
        alpha = (mag / mmax) if mmax > 0 else np.full_like(mag, 0.5)
        # Scale the per-cell alpha into a readable band.
        alpha = 0.15 + 0.70 * alpha
    else:
        alpha = 0.5

    im = ax.imshow(
        grid_signed,
        cmap="bwr",
        norm=_sym_norm(grid_signed),
        alpha=alpha,
        interpolation=interpolation,
        extent=(0, w, h, 0),
    )
    if title:
        ax.set_title(title)
    ax.set_axis_off()
    return im


def draw_patch_grid(ax, pil, *, title: Optional[str] = None):
    """Overlay the bare 27×27 cell boundaries + mark the excluded 6-px band.

    D-07: with no guaranteed sharp landmark on the locked slice, alignment
    correctness is demonstrated *explicitly* — the 27 cell lines are drawn at
    the correct pixel positions (the 384 square's first 378 px, un-squashed to
    the original PIL aspect), and the right / bottom 6-px band that
    ``padding="valid"`` discards is shaded so the edge exclusion is visible.
    """
    from matplotlib.patches import Rectangle

    w, h = pil.size
    ax.imshow(pil, extent=(0, w, h, 0))

    # The valid region is the first 378/384 of each axis (square space),
    # mapped onto the original PIL extent.
    valid_frac = VALID_DIM / IMG_DIM            # 378/384
    vx = w * valid_frac
    vy = h * valid_frac

    # 27 evenly-spaced cell boundaries across the valid region.
    for k in range(GRID + 1):
        x = vx * k / GRID
        y = vy * k / GRID
        ax.plot([x, x], [0, vy], color="lime", linewidth=0.4, alpha=0.7)
        ax.plot([0, vx], [y, y], color="lime", linewidth=0.4, alpha=0.7)

    # Shade the excluded right / bottom 6-px band (in original-PIL units).
    ax.add_patch(
        Rectangle(
            (vx, 0), w - vx, h, facecolor="black", alpha=0.45, edgecolor="none"
        )
    )
    ax.add_patch(
        Rectangle(
            (0, vy), w, h - vy, facecolor="black", alpha=0.45, edgecolor="none"
        )
    )
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_title(
        title
        or f"27x27 patch grid + excluded {EDGE_DISCARD}px right/bottom band (D-07)"
    )
    ax.set_axis_off()
