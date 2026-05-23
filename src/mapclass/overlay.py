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


def to_patch_grid(
    relevance, patch_size: int = PATCH_SIZE, img_dim: int = IMG_DIM,
    has_cls: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce relevance → a SIGNED and a MAGNITUDE per-patch grid.

    GEOMETRY-PARAMETERIZED (ATTR-02): the grid is reconstructed from the
    MODEL's own ``patch_size`` / ``img_dim`` — NOT a hardcoded 27×27. The
    defaults are the SigLIP-2 so400m-patch14-384 geometry (14 / 384 → 27×27)
    so v1.0 callers that pass only ``relevance`` are unchanged.

      * SigLIP-2  : 14 / 384 → 27×27
      * CLIP-L/14 : 14 / 224 → 16×16
      * ViT-b-16  : 16 / 224 → 14×14

    D-09: the signed path is the channel-sum with **NO ``.abs()``** and **NO
    min-max to ``[0, 1]``** — physical units preserved (negatives kept). The
    magnitude path is the ``.abs()`` channel-sum (``>= 0``).

    The real relevance for all four image models is at the PIXEL tensor
    ``(1,3,H,W)`` (relevance is read at the input by tensor identity), so the
    pixel-space reshape is the primary path and ``has_cls`` is a no-op there.
    The ``has_cls`` strip ONLY applies if the relevance instead arrives
    token-shaped ``(1, ntok, ...)`` (``ntok == grid*grid + 1``) — then index 0
    (the CLS token) is discarded BEFORE the spatial reshape.

    ``grid = img_dim // patch_size``; ``valid_dim = grid * patch_size``; the
    right/bottom edge band (``img_dim - valid_dim``, ``padding="valid"``) is
    discarded — exactly as the SigLIP-2 v1.0 path did.

    Returns ``(grid_signed, grid_mag)`` as float ndarrays, each ``(grid, grid)``.
    """
    grid = img_dim // patch_size
    valid_dim = grid * patch_size

    r = relevance.detach()

    # --- token-shaped relevance branch (only if a model emits (1,ntok,...)) --
    # Pixel relevance is (1, 3, H, W) -> 4-D; a token-shaped relevance is
    # (1, ntok, feat) -> 3-D. Only then is CLS-strip meaningful.
    if r.dim() == 3:
        toks = r[0]                                 # (ntok, feat)
        if has_cls and toks.shape[0] == grid * grid + 1:
            toks = toks[1:]                         # drop CLS (token 0)
        # (grid*grid, feat) -> per-token signed/mag over the feature dim,
        # reshaped to the (grid, grid) spatial layout (row-major patch order).
        signed = toks.sum(1).reshape(grid, grid)    # SIGNED — no abs
        mag = toks.abs().sum(1).reshape(grid, grid) # >= 0
        return (
            signed.cpu().numpy().astype(np.float64),
            mag.cpu().numpy().astype(np.float64),
        )

    # --- pixel-space path (the real path for all four image models) ---------
    r = r[0]                                        # (3, H, W)

    r_signed = r.sum(0)                             # (H,W) SIGNED — no abs
    r_mag = r.abs().sum(0)                          # (H,W) >= 0 magnitude

    # Discard the last (img_dim - valid_dim) px of the right/bottom edges
    # (padding="valid"); block-reduce reshape(grid,patch,grid,patch).mean.
    vs = r_signed[:valid_dim, :valid_dim]
    vm = r_mag[:valid_dim, :valid_dim]

    grid_signed = vs.reshape(
        grid, patch_size, grid, patch_size
    ).mean((1, 3))                                  # reshape(27,14,27,14) SIGNED
    grid_mag = vm.reshape(
        grid, patch_size, grid, patch_size
    ).mean((1, 3))                                  # >= 0

    # NO min-max normalization (D-09). float ndarrays only.
    return (
        grid_signed.cpu().numpy().astype(np.float64),
        grid_mag.cpu().numpy().astype(np.float64),
    )


def to_pixel_grid(relevance, has_cls: bool = False):
    """Full-resolution per-pixel relevance — NO patch block-averaging (D-09).

    For pixel-shaped ``(1, 3, H, W)`` relevance (the real path for all four image
    models, IG and dynamicLRP alike), this returns the channel-sum at the native
    input resolution: a fine-grained ``(H, W)`` heatmap instead of the coarse
    27×27 / 14×14 patch blocks ``to_patch_grid`` produces. ``has_cls`` is a no-op
    here (pixel relevance has no CLS token). Token-shaped relevance has no pixel
    resolution — use ``to_patch_grid`` for that.

    Returns ``(signed_HxW, mag_HxW)`` as float64 ndarrays. ``.float()`` first so
    bf16 relevance (PaliGemma) survives the numpy conversion.
    """
    r = relevance.detach().float()
    if r.dim() != 4:
        raise ValueError(
            "to_pixel_grid expects pixel-shaped (1,3,H,W) relevance; for "
            "token-shaped relevance use to_patch_grid."
        )
    r = r[0]                       # (3, H, W)
    signed = r.sum(0)              # (H, W) SIGNED — no abs
    mag = r.abs().sum(0)           # (H, W) >= 0 magnitude
    return (
        signed.cpu().numpy().astype(np.float64),
        mag.cpu().numpy().astype(np.float64),
    )


def _sym_norm(arr: np.ndarray, pct: float = 99.0):
    """Zero-centered symmetric norm so white == 0 on the bwr diverging map.

    D-09 keeps the map signed and physical-unit. The scale reference is the
    ``pct``-th percentile of ``|arr|`` (default 99) rather than the raw max, so a
    few outlier pixels do not saturate the colormap and wash everything else to
    white — the dominant cause of "the heatmap looks too sparse". Values beyond
    the reference clip to full red/blue. ``M = percentile(|arr|, pct)`` →
    ``TwoSlopeNorm(vmin=-M, vcenter=0, vmax=M)``.
    """
    from matplotlib.colors import TwoSlopeNorm

    a = np.abs(np.asarray(arr, dtype=np.float64))
    finite = a[np.isfinite(a)]
    m = float(np.percentile(finite, pct)) if finite.size else 0.0
    if not np.isfinite(m) or m == 0.0:                     # flat/degenerate
        m = float(np.nanmax(a)) if a.size else 0.0
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
    pct: float = 99.0,
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
        finite = mag[np.isfinite(mag)]
        # Percentile reference (not raw max) so outlier cells don't push every
        # other cell's alpha to ~0 and make the overlay look empty.
        mref = float(np.percentile(finite, pct)) if finite.size else 0.0
        if mref <= 0:
            mref = float(np.nanmax(mag)) if mag.size else 0.0
        alpha = (
            np.clip(mag / mref, 0.0, 1.0) if mref > 0
            else np.full_like(mag, 0.5)
        )
        # Scale the per-cell alpha into a readable band.
        alpha = 0.15 + 0.70 * alpha
    else:
        alpha = 0.5

    im = ax.imshow(
        grid_signed,
        cmap="bwr",
        norm=_sym_norm(grid_signed, pct=pct),
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
