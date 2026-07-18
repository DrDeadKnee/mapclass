"""
Window sampling and contiguous-blob masking over H3 hex regions (Phase B).

A training sample is a *window*: a connected set of hexes BFS-grown from a
seed cell within one region. Token positions are window-relative cartesian
coordinates derived from H3 local IJ — deliberately no absolute geography,
so the trained prior transfers to invented (hand-drawn) maps.

Masking follows the design in the reboot notes: high variable mask ratios
(50–95%) as *contiguous blobs*, so the model learns to diffuse terrain from
sparse far-away evidence rather than interpolate pinholes.

Pure numpy + h3 — no torch here (torch enters in dataset.py / model.py).
"""

import io
import json
from pathlib import Path

import numpy as np

try:  # h3-py v4: int-based API, matching aggregate.py
    from h3.api import basic_int as h3
except ImportError:  # pragma: no cover
    import h3  # type: ignore[no-redef]

# Basis for H3 local IJ → cartesian: axes are 120° apart with unit hex
# spacing, so e_i = (1, 0), e_j = (-1/2, √3/2). Neighbour cells then sit at
# distance exactly 1.
_SQRT3_2 = np.sqrt(3.0) / 2.0


def load_region(path: str) -> dict:
    """
    Load one region npz (+ sibling meta.json) written by build_hex_dataset.

    Hexes with zero valid pixels (all-nodata) are dropped. Returns
    ``{"cells": (n,) uint64, "soft": (n, C) float32, "meta": dict}``
    where soft rows sum to 1.
    """
    if path.startswith("gs://"):
        import gcsfs
        fs = gcsfs.GCSFileSystem()
        raw = fs.cat_file(path[len("gs://"):])
        meta_raw = fs.cat_file(path[len("gs://"):].rsplit(".npz", 1)[0] + ".meta.json")
    else:
        raw = Path(path).read_bytes()
        meta_raw = Path(str(path).rsplit(".npz", 1)[0] + ".meta.json").read_bytes()

    data = np.load(io.BytesIO(raw))
    cells = data["cells"].astype(np.uint64)
    counts = data["counts"].astype(np.float64)
    total = counts.sum(axis=1)
    keep = total > 0
    soft = (counts[keep] / total[keep, None]).astype(np.float32)
    return {"cells": cells[keep], "soft": soft, "meta": json.loads(meta_raw)}


def build_adjacency(cells: np.ndarray) -> dict[int, list[int]]:
    """Adjacency over the region's own cells: cell -> present neighbours."""
    present = set(int(c) for c in cells)
    adj: dict[int, list[int]] = {}
    for c in present:
        adj[c] = [n for n in h3.grid_ring(c, 1) if n in present]
    return adj


def sample_window(
    adjacency: dict[int, list[int]],
    size: int,
    rng: np.random.Generator,
    seed_cell: int | None = None,
) -> list[int]:
    """
    BFS-grow a connected window of up to ``size`` cells from ``seed_cell``
    (random cell if None). Neighbour order is shuffled per step so windows
    from one seed vary in shape. Returns fewer than ``size`` cells only when
    the seed's connected component is smaller.
    """
    cells = list(adjacency)
    if seed_cell is None:
        seed_cell = cells[int(rng.integers(len(cells)))]
    window: list[int] = []
    seen = {seed_cell}
    frontier = [seed_cell]
    while frontier and len(window) < size:
        idx = int(rng.integers(len(frontier)))
        cell = frontier.pop(idx)
        window.append(cell)
        for n in adjacency[cell]:
            if n not in seen:
                seen.add(n)
                frontier.append(n)
    return window


def local_xy(window: list[int], origin: int | None = None) -> np.ndarray:
    """
    Window-relative cartesian coordinates (unit hex spacing), centred on the
    window mean. Cells whose local IJ is undefined w.r.t. the origin (rare
    pentagon-distortion cases) get the coordinate of the window centroid —
    callers should keep windows local enough that this stays negligible.
    """
    if origin is None:
        origin = window[0]
    xy = np.zeros((len(window), 2), dtype=np.float32)
    ok = np.zeros(len(window), dtype=bool)
    for k, cell in enumerate(window):
        try:
            i, j = h3.cell_to_local_ij(origin, cell)
        except Exception:
            continue
        xy[k, 0] = i - 0.5 * j
        xy[k, 1] = _SQRT3_2 * j
        ok[k] = True
    if ok.any():
        center = xy[ok].mean(axis=0)
        xy[ok] -= center
        xy[~ok] = 0.0
    return xy


def blob_mask(
    window: list[int],
    adjacency: dict[int, list[int]],
    ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Boolean mask (True = hidden) covering ~``ratio`` of the window as
    contiguous blobs. Guarantees at least 1 masked and at least 1 visible
    cell. Blobs are grown by randomized BFS restricted to the window; a new
    blob is seeded whenever growth stalls or a random blob budget runs out.
    """
    k = len(window)
    if k < 2:
        raise ValueError("window must have at least 2 cells to mask")
    target = int(np.clip(round(ratio * k), 1, k - 1))
    in_window = {c: idx for idx, c in enumerate(window)}
    masked: set[int] = set()

    while len(masked) < target:
        unmasked = [c for c in window if c not in masked]
        seed = unmasked[int(rng.integers(len(unmasked)))]
        # One blob: random budget between ~k/8 and the remaining deficit.
        deficit = target - len(masked)
        lo = min(max(1, k // 8), deficit)
        budget = int(rng.integers(lo, deficit + 1))
        frontier = [seed]
        while frontier and budget > 0 and len(masked) < target:
            idx = int(rng.integers(len(frontier)))
            cell = frontier.pop(idx)
            if cell in masked:
                continue
            masked.add(cell)
            budget -= 1
            for n in adjacency[cell]:
                if n in in_window and n not in masked:
                    frontier.append(n)

    out = np.zeros(k, dtype=bool)
    for c in masked:
        out[in_window[c]] = True
    return out
