"""
Deterministic sample-id-hash splits (PITFALL 5 prevention).

Splits are defined ONCE at v0 build time and committed to mapclass/configs/splits.json.
Future bootstrap-expansion (Phase 4) MUST extend the train split via the same bucket
policy; the eval-time overlap assertion (in mapclass.eval) refuses any overlap.
"""

import hashlib
import json
from pathlib import Path


class SplitsContaminationError(ValueError):
    """Raised by mapclass.eval at startup if test split overlaps train manifest."""


# Bucket policy: 0-7 → train, 8 → val, 9 → test. (80/10/10 by hash bucket.)
_TRAIN_BUCKETS = frozenset({0, 1, 2, 3, 4, 5, 6, 7})
_VAL_BUCKETS = frozenset({8})
_TEST_BUCKETS = frozenset({9})


def _bucket_for_sample_id(sample_id: str) -> int:
    """sha256(sample_id)[:8] → bucket 0..9. Pure function of ID, immune to glob order."""
    h = hashlib.sha256(sample_id.encode()).hexdigest()
    return int(h[:8], 16) % 10


def build_splits(sample_ids: list[str]) -> dict:
    """Build the canonical splits.json structure from a list of sample IDs."""
    by_bucket: dict[str, int] = {}
    by_split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for sid in sample_ids:
        b = _bucket_for_sample_id(sid)
        by_bucket[sid] = b
        if b in _TRAIN_BUCKETS:
            by_split["train"].append(sid)
        elif b in _VAL_BUCKETS:
            by_split["val"].append(sid)
        elif b in _TEST_BUCKETS:
            by_split["test"].append(sid)
    return {"version": 1, "by_split": by_split, "by_id_hash_bucket": by_bucket}


def write_splits(splits: dict, path: str | Path) -> None:
    """Write splits.json with indent=2 (Pattern H)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(splits, f, indent=2)


def load_splits(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)
