"""
GCS I/O abstraction for the MapClass Phase-2 dataset pipeline.

Provides a single, tested GCS I/O surface shared by every Phase-2 build script
(build_dataset.py, build_historical_dataset.py, build_satellite_dataset.py,
tiling.py) and the Phase-4 finetune/evaluate pull-once helpers.

Design notes
------------
- Lazy gcsfs import: the module can be imported on the planning VM where gcsfs
  is absent.  Tests patch ``gcs_io.gcsfs`` the same way tests patch
  ``seg.gcs_checkpoint.gcsfs``.
- _GCSWriter instances are NOT thread-safe; create one per-pyramid in the pool.
- validate_manifest is the ONLY gate before split.json is frozen (RW-02).  It
  calls sys.exit(1) on any mismatch — never warns or raises ValueError.
- pull_dataset_from_gcs + verify_pull implement the RW-03 pull-once pattern.

GCS Layout Decision (OQ2, LOCKED 2026-05-16, 02-05):
-----------------------------------------------------
Family-rooted GCS layout — each build script owns its family prefix:
  gs://.../data/synthetic/{train,test}/<map_id>/
  gs://.../data/historical/dataset/<map_id>/
  gs://.../data/satellite/dataset/<map_id>/

At pull-once time (RW-03), finetune_seg pulls ALL three families into a
MERGED local scratch root so carve_train_val / load_test_pyramid_dirs see
a flat tree.  evaluate_seg pulls ONLY the synthetic test subset because
EVAL-01 hold-out is synthetic-only (split.json stays at
gs://.../data/synthetic/split.json and is never relocated).

Rationale: each build script is decoupled; merging happens only at pull time;
no cross-family path contamination; historical/satellite pyramids add to train
without polluting the EVAL-01 split logic.

_BUILD_COMPLETE sentinel (OQ1, LOCKED 2026-05-16, 02-05):
----------------------------------------------------------
After all pyramid objects for a map-dir are written, a zero-byte sentinel
``_BUILD_COMPLETE`` is written as the LAST write.  The skip guard in every
build script checks ``is_build_complete()`` — a prefix whose objects exist
but whose sentinel is absent is treated as a partial/aborted build and
rebuilt from scratch (Pitfall R-2: preemption leaves partial dirs).

Trust boundary: user-authored manifest.json filenames/templates cross a trust
boundary here.  The _FILENAME_RE fullmatch + 4-way cross-check in
validate_manifest is the mitigation for T-02-02 (bucket-path injection via
manifest filenames).  validate_manifest must be called BEFORE load_or_create_split.

ADC auth: gcsfs uses Application Default Credentials automatically.  No key or
token literal appears in this file (T-02-01 mitigation).
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

# ---------------------------------------------------------------------------
# Lazy gcsfs import — same pattern as scripts/seg/gcs_checkpoint.py lines 36-39
# ---------------------------------------------------------------------------

try:
    import gcsfs  # type: ignore[import]
except ModuleNotFoundError:
    gcsfs = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Constants — copy GCS_PROJECT verbatim from gcs_checkpoint.py line 45
# ---------------------------------------------------------------------------

GCS_PROJECT = "narrative-campaign"
BUCKET = "mapclass-training-northeast1"
DATA_PREFIX = "mapclass-training-northeast1/data"

# ---------------------------------------------------------------------------
# _BUILD_COMPLETE sentinel (OQ1)
# ---------------------------------------------------------------------------

#: Sentinel filename written as the LAST object after a map-dir build succeeds.
#: The skip guard checks is_build_complete() — a dir without this sentinel is
#: rebuilt from scratch (Pitfall R-2 / T-02-40 mitigation).
_BUILD_COMPLETE = "_BUILD_COMPLETE"


def mark_build_complete(fs: "Any", map_dir_prefix: str) -> None:
    """Write the _BUILD_COMPLETE sentinel as the LAST step of a map-dir build.

    Must be called AFTER ``tiling.tile(...)`` completes and ALL pyramid objects
    have been written.  The sentinel presence is the only reliable indicator
    that the build for ``map_dir_prefix`` finished without preemption (Pitfall
    R-2 / T-02-40).

    Args:
        fs:             fsspec-compatible filesystem (gcsfs.GCSFileSystem or
                        fsspec.filesystem("file") for tests).
        map_dir_prefix: bare GCS prefix of the map directory, e.g.
                        ``"mapclass-training-northeast1/data/synthetic/train/europe_01__flat"``.
    """
    fs.pipe_file(f"{map_dir_prefix}/{_BUILD_COMPLETE}", b"1")


def is_build_complete(fs: "Any", map_dir_prefix: str) -> bool:
    """Return True iff the _BUILD_COMPLETE sentinel exists for ``map_dir_prefix``.

    A map-dir prefix whose objects exist but whose sentinel is absent is
    treated as a partial/aborted build and must be rebuilt from scratch
    (Pitfall R-2 / T-02-40).

    Args:
        fs:             fsspec-compatible filesystem.
        map_dir_prefix: bare GCS prefix of the map directory.

    Returns:
        True if the sentinel object exists; False otherwise.
    """
    return bool(fs.exists(f"{map_dir_prefix}/{_BUILD_COMPLETE}"))


# ---------------------------------------------------------------------------
# family_subset_prefix (OQ2 — family-rooted GCS layout)
# ---------------------------------------------------------------------------

def family_subset_prefix(family: str, subset: str) -> str:
    """Return the canonical bare GCS prefix for a (family, subset) pair.

    Encodes the OQ2 family-rooted layout decision (LOCKED 2026-05-16, 02-05):
    each build script writes into its own family subdirectory; pull-once merges
    them into a flat local scratch root.

    Args:
        family: one of ``"synthetic"``, ``"historical"``, ``"satellite"``.
        subset: e.g. ``"train"``, ``"test"``, ``"dataset"``.

    Returns:
        Bare GCS prefix string (no ``gs://`` scheme), e.g.
        ``"mapclass-training-northeast1/data/synthetic/train"``.
    """
    return f"{DATA_PREFIX}/{family}/{subset}"


# ---------------------------------------------------------------------------
# _FILENAME_RE — RW-02 filename convention
# ---------------------------------------------------------------------------

_FILENAME_RE = re.compile(r"^[a-z][a-z0-9_]*_[0-9]{2}\.geojson$")


# ---------------------------------------------------------------------------
# _GCSWriter
# ---------------------------------------------------------------------------

class _GCSWriter:
    """Thin write-only abstraction over a GCS (or local-fs) prefix.

    Replicates the handful of pathlib.Path write methods that the Phase-2
    pipeline needs for the output side only.  Reads (map_dir/) stay local
    Path-based.

    The fs argument MUST be an already-instantiated fsspec AbstractFileSystem
    (e.g. gcsfs.GCSFileSystem or fsspec.filesystem("file")).  NEVER instantiate
    GCSFileSystem inside this class — callers pass it in, mirroring the
    gcs_checkpoint.py single-instance discipline.

    Instances are NOT thread-safe; create one per-pyramid in the thread pool.

    Usage::

        fs = gcsfs.GCSFileSystem(project=GCS_PROJECT)
        root = _GCSWriter(fs, "mapclass-training-northeast1/data/synthetic/train")
        map_writer = root / "europe_01__flat" / "pyramids" / "py_r0"
        map_writer.mkdir()
        map_writer.write_bytes(b"...")
        map_writer.write_text("{...}")
        with map_writer.open("wb") as fh:
            img.save(fh, format="PNG")
    """

    def __init__(self, fs: "Any", prefix: str) -> None:
        self._fs = fs
        self._prefix = prefix.rstrip("/")

    def __truediv__(self, name: str) -> "_GCSWriter":
        """Return a child writer at ``self._prefix/name``."""
        return _GCSWriter(self._fs, f"{self._prefix}/{name}")

    def mkdir(self, parents: bool = True, exist_ok: bool = True) -> None:
        """Create the prefix as a directory on the filesystem.

        For GCS this is a no-op (GCS has no real directories); for a local
        filesystem it delegates to ``fs.mkdirs`` which creates the directory
        hierarchy.
        """
        self._fs.mkdirs(self._prefix, exist_ok=True)

    @property
    def name(self) -> str:
        """Last path segment of the prefix (mirrors pathlib.Path.name)."""
        return self._prefix.rsplit("/", 1)[-1]

    def write_bytes(self, data: bytes) -> None:
        """Write *data* to the prefix path atomically via pipe_file."""
        self._fs.pipe_file(self._prefix, data)

    def write_text(self, text: str, encoding: str = "utf-8") -> None:
        """Encode *text* as UTF-8 (or *encoding*) and write via pipe_file."""
        self._fs.pipe_file(self._prefix, text.encode(encoding))

    def open(self, mode: str = "wb") -> "Any":
        """Return a writable file-object for Pillow .save() calls.

        ``gcsfs.GCSFile`` (returned by gcsfs.GCSFileSystem.open) implements
        write(), seek(), tell(), flush(), and close() — the full seekable/
        writable buffer protocol that Pillow's PNG encoder requires.
        ``fsspec.implementations.local.LocalFileOpener`` implements the same
        protocol for local filesystem tests.
        """
        return self._fs.open(self._prefix, mode)


# ---------------------------------------------------------------------------
# validate_manifest
# ---------------------------------------------------------------------------

def _sanitize_stem(stem: str) -> str:
    """Sanitize a GeoJSON stem for safe use as a source ID.

    Mirrors build_dataset._sanitize_stem (re.sub(r'[^\\w-]', '_', stem)).
    Replicated here so gcs_io has no circular import dependency on build_dataset.
    """
    return re.sub(r"[^\w-]", "_", stem)


def validate_manifest(
    gcs_filenames: "list[str]",
    manifest: "dict[str, Any]",
) -> "dict[str, str]":
    """Validate GCS raw filenames against the user-authored manifest.

    Returns ``{sanitized_id: template}`` on success.  Calls ``sys.exit(1)`` on
    ANY validation error — this is the ONLY gate before split.json is frozen
    (RW-02 + T-02-02 + T-02-03).

    Four error classes checked:
      (a) filename fails _FILENAME_RE.fullmatch (regex convention)
      (b) filename not in manifest["entries"] (unlisted)
      (c) manifest entry not in gcs_filenames (phantom)
      (d) entry template != fn.rsplit("_", 1)[0] (template mismatch)

    Args:
        gcs_filenames: list of .geojson basenames from fs.ls(raw_gcs_prefix).
        manifest:      parsed manifest.json dict ({"version": "1", "entries": {...}}).

    Returns:
        {_sanitize_stem(fn[:-8]): entries[fn]["template"] for fn in gcs_filenames}
        where fn[:-8] strips the ".geojson" suffix (8 chars).

    Raises:
        SystemExit(1) on any validation error (never ValueError, never warning).
    """
    errors: list[str] = []
    entries: dict[str, Any] = manifest.get("entries", {})

    for fn in gcs_filenames:
        if not _FILENAME_RE.fullmatch(fn):
            errors.append(
                f"  filename convention violation: {fn!r} does not match "
                f"<template>_<NN>.geojson"
            )
        if fn not in entries:
            errors.append(f"  unlisted in manifest: {fn!r}")
        else:
            expected_tmpl = fn.rsplit("_", 1)[0]  # everything before _NN
            actual_tmpl = entries[fn].get("template", "")
            if actual_tmpl != expected_tmpl:
                errors.append(
                    f"  template mismatch: {fn!r} has template "
                    f"{actual_tmpl!r}, expected {expected_tmpl!r}"
                )

    for fn in entries:
        if fn not in gcs_filenames:
            errors.append(f"  manifest entry missing from GCS raw/: {fn!r}")

    if errors:
        print(
            "FATAL: manifest validation failed — aborting before split is "
            "computed (RW-02):"
        )
        for e in errors:
            print(e)
        sys.exit(1)

    return {
        _sanitize_stem(fn[:-8]): entries[fn]["template"]
        for fn in gcs_filenames
    }


# ---------------------------------------------------------------------------
# pull_dataset_from_gcs
# ---------------------------------------------------------------------------

def pull_dataset_from_gcs(
    subset: str,
    local_scratch: "Path | str",
    gcs_prefix: str = "gs://mapclass-training-northeast1/data/synthetic",
) -> Path:
    """Bulk-fetch a train or test subset from GCS to local scratch.

    Idempotent: safe to re-run after preemption (fs.get overwrites local files).

    Args:
        subset:        "train" or "test" (or any GCS subdirectory name).
        local_scratch: local directory path for the downloaded data.
        gcs_prefix:    GCS prefix without trailing slash; defaults to the
                       canonical synthetic data prefix.

    Returns:
        Path to the local root for the subset (``local_scratch / subset``).

    Raises:
        ImportError if gcsfs is not installed.
    """
    if gcsfs is None:
        raise ImportError(
            "gcsfs is not installed.  Install it on the training host with: "
            "pip install gcsfs"
        )

    local_scratch = Path(local_scratch)
    fs = gcsfs.GCSFileSystem(project=GCS_PROJECT)

    # Strip gs:// for the bare-prefix convention (mirrors gcs_checkpoint.py
    # bare_prefix pattern at lines 174-177)
    bare_prefix = gcs_prefix.lstrip("gs://")
    remote = f"{bare_prefix}/{subset}/"
    local_root = local_scratch / subset
    local_root.mkdir(parents=True, exist_ok=True)

    # recursive=True downloads the entire subtree; idempotent (overwrites on
    # re-run, safe after preemption) — matches gcsfs.GCSFileSystem.get API
    # (verified 2026-05-16).
    fs.get(remote, str(local_root), recursive=True)
    return local_root


# ---------------------------------------------------------------------------
# verify_pull
# ---------------------------------------------------------------------------

def verify_pull(
    local_root: "Path | str",
    split_json: "dict[str, Any]",
    subset: str,
) -> None:
    """Verify a pulled subset has the expected map directories and pyramids.

    Implements the RW-03 verification design:
      1. For each map_id in split_json[subset]: assert local dir exists.
      2. For a 10% sample of pyramid dirs: assert pyramid.json + >= 60 PNGs.

    Args:
        local_root:  local directory containing the pulled subset (e.g.
                     ``/tmp/mapclass_data/train``).
        split_json:  parsed split.json dict ({"test": [...], "train": [...]}).
        subset:      "train" or "test" — selects the key from split_json.

    Raises:
        RuntimeError on any verification failure (NOT sys.exit — callers
        handle the error appropriately).
    """
    local_root = Path(local_root)
    map_ids: list[str] = split_json.get(subset, [])

    # Step 1: membership + presence check
    for map_id in map_ids:
        map_dir = local_root / map_id
        if not map_dir.is_dir():
            raise RuntimeError(
                f"verify_pull: missing map directory for {map_id!r} "
                f"under {local_root}"
            )

    # Step 2: 10% pyramid spot-check
    # Collect all pyramid dirs across all map dirs for the subset
    all_pyramid_dirs: list[Path] = []
    for map_id in map_ids:
        map_dir = local_root / map_id
        pyramids_root = map_dir / "pyramids"
        if pyramids_root.is_dir():
            all_pyramid_dirs.extend(
                p for p in pyramids_root.iterdir() if p.is_dir()
            )

    if all_pyramid_dirs:
        # Sample ~10% (at least 1 dir)
        k = max(1, len(all_pyramid_dirs) // 10)
        rng = random.Random(42)
        sample = rng.sample(all_pyramid_dirs, min(k, len(all_pyramid_dirs)))

        for pdir in sample:
            if not (pdir / "pyramid.json").exists():
                raise RuntimeError(
                    f"verify_pull: missing pyramid.json in {pdir}"
                )
            png_count = len(list(pdir.glob("*.png")))
            if png_count < 60:
                raise RuntimeError(
                    f"verify_pull: truncated pyramid at {pdir}: "
                    f"found {png_count} PNGs, expected >= 60"
                )
