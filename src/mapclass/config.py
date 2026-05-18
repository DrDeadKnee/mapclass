"""Central configuration constants for mapclass.

Single source of truth for the GCS bucket/layout, the SigLIP-2 checkpoint id,
the local cache locations, the compute device, and the vendored dynamicLRP
source SHA (D-08).

NOTE (parallel-wave provenance): the canonical owner of this file is Plan
01-01. It is reproduced here verbatim to its documented contract so the two
wave-1 worktrees converge to a byte-identical file on merge. The VENDOR_SHA
value mirrors third_party/dynamicLRP/VENDOR_SHA (D-08).
"""

from __future__ import annotations

import os

# --- GCS bucket + layout ---------------------------------------------------
GCS_BUCKET = "mapclass-training-northeast1"
GCS_DATA_PREFIX = "data/"
GCS_MODELS_PREFIX = "models/"

# --- Model -----------------------------------------------------------------
MODEL_REPO_ID = "google/siglip2-so400m-patch14-384"
MODEL_GCS_DIR = "models/siglip2-so400m-patch14-384"

# --- Manifest --------------------------------------------------------------
MANIFEST_PATH = "metadata/rumsey_manifest.json"

# --- Local caches (gitignored) ---------------------------------------------
LOCAL_IMAGE_CACHE_DIR = os.environ.get(
    "MAPCLASS_IMAGE_CACHE_DIR", os.path.expanduser("~/.cache/mapclass/images")
)
LOCAL_MODEL_CACHE_DIR = os.environ.get(
    "MAPCLASS_MODEL_CACHE_DIR", os.path.expanduser("~/.cache/mapclass/model")
)

# --- Vendored dynamicLRP provenance (D-08) ---------------------------------
VENDOR_SHA = "405e74243ecaa1f615f418fdc8ba24c3c5889b1e"


def _resolve_device() -> str:
    """Resolve the compute device.

    Defers the torch import so that pure-stdlib consumers (e.g. the manifest
    reader and its unit tests) do not require torch to be installed.
    """
    try:
        import torch  # noqa: WPS433 (intentional local import)

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # torch not installed yet (Plan 01-01 builds the venv)
        return "cpu"


DEVICE = _resolve_device()
