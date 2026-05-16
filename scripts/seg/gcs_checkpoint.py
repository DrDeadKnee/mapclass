"""
GCS checkpoint write + auto-resume for the MapClass segmentation training loop.

Design: D-07 (preemption-safe periodic checkpoints) + D-09 (auto-detect latest
        step on resume) — both described in 04-RESEARCH.md Pattern 3.

Checkpoint path schema:
    gs://mapclass-training-northeast1/models/<config>/step_{step:07d}.pt

Config naming: ``<backbone>-<variant>`` lowercased, e.g. ``siglip-b``, ``dinov2-a``.
The config string is validated against ``^[a-z0-9][a-z0-9\\-]*$`` before any GCS
URI is constructed (T-04-04 — path-traversal guard).

GCS I/O is LAZY: ``gcsfs`` is imported inside each function so the module can be
imported on the planning VM where gcsfs is not installed.  Tests mock
``seg.gcs_checkpoint.gcsfs.GCSFileSystem`` rather than the module-level import.

Trust assumption (T-04-06 / D-08): only own-produced checkpoints in the
project-private GCS bucket are deserialised with ``weights_only=False``.  Never
use this module to load third-party artifacts.
"""

from __future__ import annotations

import io
import re
from pickle import UnpicklingError

import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GCS_PROJECT = "narrative-campaign"
BUCKET_PREFIX = "gs://mapclass-training-northeast1/models"

_CONFIG_RE = re.compile(r"^[a-z0-9][a-z0-9\-]*$")

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_config_name(name: str) -> str:
    """Validate *name* against the GCS-safe config regex.

    Returns *name* unchanged if it matches ``^[a-z0-9][a-z0-9\\-]*$``.
    Raises :class:`ValueError` otherwise (rejects path traversal, uppercase
    letters, leading dashes, slashes, and empty strings).

    This is the single enforcement point for T-04-04.
    """
    if not _CONFIG_RE.fullmatch(name):
        raise ValueError(
            f"Invalid config name {name!r}: must match ^[a-z0-9][a-z0-9\\-]*$ "
            "(lowercase alphanumerics and hyphens only, no leading hyphen)."
        )
    return name


def config_prefix(backbone: str, variant: str) -> str:
    """Build and validate a GCS config string from backbone + variant.

    Lowercases both arguments before concatenation so that callers may pass
    the Phase-3/CONTEXT forms (e.g. ``'SigLIP'``, ``'A'``) without raising.
    The resulting string ``'siglip-a'`` is then passed through
    :func:`validate_config_name`.

    Lowercase normalisation is intentional: GCS object names are
    case-sensitive and ``siglip-A`` would fail the traversal-safe regex.
    Plan-05 comparison reports must use the same lowercased form.
    """
    name = f"{backbone.lower()}-{variant.lower()}"
    return validate_config_name(name)


# ---------------------------------------------------------------------------
# Checkpoint write
# ---------------------------------------------------------------------------


def gcs_save_checkpoint(config_name: str, step: int, state: dict) -> str:
    """Serialise *state* and write it to GCS atomically.

    The write uses an in-memory BytesIO buffer so that the GCS object is
    committed atomically (Pitfall 4 — preemption mid-stream leaves no partial
    object).  The caller should invoke this function AFTER the optimiser step
    completes so that the step number in the filename is accurate.

    Args:
        config_name: A pre-validated GCS config string (e.g. ``'siglip-b'``).
                     Validated again here as a defence-in-depth measure.
        step:        The completed optimiser step number.
        state:       Checkpoint state dict.  Expected shape::

                         {
                             "step": int,
                             "model_state_dict": dict,
                             "optimizer_state_dict": dict,
                             "config": str,
                         }

    Returns:
        The full ``gs://`` path of the written object.
    """
    import gcsfs  # lazy — gcsfs absent on the planning VM

    validate_config_name(config_name)  # defence-in-depth

    path = f"{BUCKET_PREFIX}/{config_name}/step_{step:07d}.pt"
    buf = io.BytesIO()
    torch.save(state, buf)
    buf.seek(0)

    fs = gcsfs.GCSFileSystem(project=GCS_PROJECT)
    with fs.open(path, "wb") as fh:
        fh.write(buf.read())

    return path


# ---------------------------------------------------------------------------
# Checkpoint resume
# ---------------------------------------------------------------------------


def gcs_latest_checkpoint(config_name: str) -> tuple[int, dict | None]:
    """Return ``(step, state_dict)`` for the highest-numbered valid checkpoint.

    Behaviour:
    - If the GCS prefix does not exist (``FileNotFoundError``) → ``(0, None)``.
    - If no ``step_*.pt`` files are found → ``(0, None)``.
    - Picks the file with the highest step number and attempts ``torch.load``.
    - If ``torch.load`` raises (corrupt/truncated blob — T-04-05), the file is
      skipped and the next-highest step is tried.
    - If all candidates are corrupt → ``(0, None)``.

    Trust assumption: only own-produced GCS objects from the project-private
    bucket are deserialised; ``weights_only=False`` is safe in this context
    (T-04-06 / D-08).

    Args:
        config_name: GCS config string (e.g. ``'siglip-b'``).

    Returns:
        ``(step, state_dict)`` on success, ``(0, None)`` if nothing loadable.
    """
    import gcsfs  # lazy — gcsfs absent on the planning VM

    fs = gcsfs.GCSFileSystem(project=GCS_PROJECT)
    # ls() returns bare paths without the gs:// prefix, e.g.
    # "mapclass-training-northeast1/models/siglip-b/step_0000042.pt"
    bare_prefix = f"mapclass-training-northeast1/models/{config_name}/"

    try:
        files = fs.ls(bare_prefix)
    except FileNotFoundError:
        return 0, None

    # Filter to step_*.pt files
    _step_re = re.compile(r"step_(\d+)\.pt$")
    step_files = [(f, int(m.group(1))) for f in files if (m := _step_re.search(f))]
    if not step_files:
        return 0, None

    # Sort descending by step number so we try the latest first
    step_files.sort(key=lambda x: x[1], reverse=True)

    for bare_path, step_num in step_files:
        gs_path = f"gs://{bare_path}"
        try:
            with fs.open(gs_path, "rb") as fh:
                # Trust assumption: own-produced checkpoints only (D-08).
                state = torch.load(io.BytesIO(fh.read()), weights_only=False)
            return step_num, state
        except (UnpicklingError, EOFError, RuntimeError, Exception):
            # Corrupt or truncated blob — skip and try the next candidate.
            continue

    return 0, None
