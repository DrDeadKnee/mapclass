"""Central project configuration.

Single source of truth for the GCS bucket/prefixes, the SigLIP-2 checkpoint id,
the resolved compute device, and the vendored dynamicLRP source SHA.

The vendored engine's provenance is carried by ``VENDOR_SHA`` here, which MUST
equal the single line in ``third_party/dynamicLRP/VENDOR_SHA`` (D-08). Updating
the engine is a deliberate manual re-vendor that changes BOTH.
"""

from pathlib import Path

import torch

# Repo root = two parents up from this file (src/mapclass/config.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]

# --- GCS layout (bucket already exists; VM has ADC) ---------------------------
GCS_BUCKET = "mapclass-training-northeast1"
GCS_DATA_PREFIX = "data/"
GCS_MODELS_PREFIX = "models/"

# --- Model under attribution --------------------------------------------------
# Fixed-resolution SigLIP-2 variant (model_type "siglip", SiglipImageProcessor) —
# NOT the NaFlex Siglip2 path (RESEARCH.md State of the Art).
MODEL_REPO_ID = "google/siglip2-so400m-patch14-384"
MODEL_GCS_DIR = "models/siglip2-so400m-patch14-384"

# --- Compute device -----------------------------------------------------------
# LRP on so400m on CPU is impractical; verify CUDA fail-fast at kernel start.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Vendored dynamicLRP provenance (D-08) ------------------------------------
# MUST equal third_party/dynamicLRP/VENDOR_SHA. A deliberate re-vendor changes both.
VENDOR_SHA = "405e74243ecaa1f615f418fdc8ba24c3c5889b1e"
VENDOR_SHA_FILE = REPO_ROOT / "third_party" / "dynamicLRP" / "VENDOR_SHA"
VENDORED_LRP_SRC = REPO_ROOT / "third_party" / "dynamicLRP" / "src"

# --- Local (gitignored) caches + manifest -------------------------------------
LOCAL_IMAGE_CACHE_DIR = REPO_ROOT / ".cache" / "rumsey_images"
MANIFEST_PATH = "metadata/rumsey_manifest.json"
