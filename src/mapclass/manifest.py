"""Ordered-index reader for the Rumsey manifest.

``metadata/rumsey_manifest.json`` is a JSON **list** of 1,544 dicts; the list
position IS the manifest index (verified: ``type=list``, ``len=1544``, each
entry has ``id`` and ``image_url``).

IMPORTANT — ``richness_score`` is NOT monotonic with the manifest index
(Pitfall A). Verified directly against the data: ``manifest[0].richness_score
== 18`` while ``manifest[1543].richness_score == 0``. The "higher index ≈
richer map" framing in PROJECT.md / CONTEXT.md does **not** hold in the actual
data. The locked slice (D-05) is the **highest LIST index** — i.e.
``manifest[-1]`` (id ``RUMSEY~8~1~344476~90112460``) — a purely positional,
programmatic choice. Do NOT sort or filter the manifest by ``richness_score``
anywhere; doing so would silently change the locked slice. There is
deliberately no ``richness_score`` reference in this module.
"""

from __future__ import annotations

import json
from typing import Iterator, List, Dict, Tuple

from mapclass import config


def load_manifest(path: str = config.MANIFEST_PATH) -> List[Dict]:
    """Load the manifest as the raw JSON list (no sorting, no filtering).

    The list is returned exactly as stored so that list position == manifest
    index. ``len(load_manifest()) == 1544`` and
    ``load_manifest()[-1]['id'] == 'RUMSEY~8~1~344476~90112460'`` (D-05).
    """
    with open(path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if not isinstance(manifest, list):
        raise ValueError(
            f"Expected the manifest to be a JSON list, got {type(manifest)!r}"
        )
    return manifest


def entry_by_index(manifest: List[Dict], idx: int) -> Dict:
    """Return ``manifest[idx]``.

    Plain list indexing — ``idx=-1`` resolves to the highest manifest index,
    which is the locked slice per D-05.
    """
    return manifest[idx]


def count_down_from(
    manifest: List[Dict], start_idx: int, count: int
) -> Iterator[Tuple[int, Dict]]:
    """Yield ``(i, manifest[i])`` for ``i`` descending from ``start_idx``.

    Yields at most ``count`` pairs and never indexes below 0. Phase 2's sweep
    consumes this to count *down from a high index N* (PROJECT.md index
    semantics). E.g. ``list(count_down_from(m, 1543, 3))`` yields indices
    ``[1543, 1542, 1541]``.
    """
    stop = max(start_idx - count, -1)
    for i in range(start_idx, stop, -1):
        yield i, manifest[i]
