"""Central configuration constants for mapclass.

Single source of truth for the GCS bucket/layout, the SigLIP-2 checkpoint id,
the manifest path, the local cache locations, the compute device, and the
vendored dynamicLRP source provenance.

This file is the canonical contract of Plan 01-01 (pinned environment +
package scaffold). The Plan 01-02 persistence/data-access layer consumes the
GCS/cache/manifest/device names; Plan 01-01's notebook and the D-08 vendor
check consume ``REPO_ROOT``, ``VENDOR_SHA``, ``VENDOR_SHA_FILE`` and
``VENDORED_LRP_SRC``. The two wave-1 worktrees authored divergent drafts of
this file; this is the reconciled superset (post-merge integration fix).

The vendored engine's provenance is carried by ``VENDOR_SHA`` here, which MUST
equal the single line in ``third_party/dynamicLRP/VENDOR_SHA`` (D-08). Updating
the engine is a deliberate manual re-vendor that changes BOTH.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repo root = two parents up from this file (src/mapclass/config.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]

# --- GCS bucket + layout (bucket already exists; VM has ADC) ----------------
GCS_BUCKET = "mapclass-training-northeast1"
GCS_DATA_PREFIX = "data/"
GCS_MODELS_PREFIX = "models/"

# --- Model under attribution -----------------------------------------------
# Fixed-resolution SigLIP-2 variant (model_type "siglip", SiglipImageProcessor) —
# NOT the NaFlex Siglip2 path (RESEARCH.md State of the Art).
MODEL_REPO_ID = "google/siglip2-so400m-patch14-384"
MODEL_GCS_DIR = "models/siglip2-so400m-patch14-384"

# --- Manifest --------------------------------------------------------------
MANIFEST_PATH = "metadata/rumsey_manifest.json"

# --- Local caches (gitignored), env-overridable ----------------------------
LOCAL_IMAGE_CACHE_DIR = os.environ.get(
    "MAPCLASS_IMAGE_CACHE_DIR", os.path.expanduser("~/.cache/mapclass/images")
)
LOCAL_MODEL_CACHE_DIR = os.environ.get(
    "MAPCLASS_MODEL_CACHE_DIR", os.path.expanduser("~/.cache/mapclass/model")
)

# --- Vendored dynamicLRP provenance (D-08) ---------------------------------
# VENDOR_SHA MUST equal third_party/dynamicLRP/VENDOR_SHA. A deliberate
# re-vendor changes both. VENDORED_LRP_SRC is prepended to sys.path by the
# notebook/conftest so ``from lrp_engine import LRPEngine`` resolves without
# packaging metadata (dynamicLRP has none).
VENDOR_SHA = "405e74243ecaa1f615f418fdc8ba24c3c5889b1e"
VENDOR_SHA_FILE = REPO_ROOT / "third_party" / "dynamicLRP" / "VENDOR_SHA"
VENDORED_LRP_SRC = REPO_ROOT / "third_party" / "dynamicLRP" / "src"


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
