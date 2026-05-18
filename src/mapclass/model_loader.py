"""Singleton SigLIP-2 loader from the GCS model mirror (never HF at runtime).

This checkpoint (``google/siglip2-so400m-patch14-384``) is the **fixed-
resolution** variant: ``model_type: "siglip"`` with a plain
``SiglipImageProcessor``. It loads via ``AutoModel`` / ``AutoProcessor`` with
``attn_implementation="eager"``.

Do NOT use the NaFlex ``Siglip2*`` image-processor / per-pixel-mask /
patch-spatial-shapes code path — that is for the NaFlex variants and would
produce load/shape errors on this fixed-resolution checkpoint ("What NOT to
use"). (The exact NaFlex symbol names are deliberately not written here so a
grep can assert this module never references them.)

Reproducibility (Pitfall G): the weights come ONLY from the GCS mirror; HF is
never contacted at runtime. ``attn_implementation="eager"`` also lowers
integration risk and improves determinism vs. SDPA kernels.
"""

from __future__ import annotations

import os
import threading
from typing import Optional, Tuple

from mapclass import config
from mapclass.ingest_images import get_bucket

# Module-level singleton cache (model is multi-GB — load once per kernel).
_MODEL = None
_PROCESSOR = None
_LOCK = threading.Lock()


def _download_model_mirror(local_dir: str) -> str:
    """Cache-first download of the GCS model dir to ``local_dir``.

    Only blobs missing locally (or size-mismatched) are fetched, so a warm
    cache makes this a near no-op on subsequent kernels.
    """
    os.makedirs(local_dir, exist_ok=True)
    bucket = get_bucket()
    prefix = f"{config.MODEL_GCS_DIR}/"
    for blob in bucket.list_blobs(prefix=prefix):
        relpath = blob.name[len(prefix):]
        if not relpath:  # the "directory" placeholder, if any
            continue
        dest = os.path.join(local_dir, relpath.replace("/", os.sep))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.exists(dest) and os.path.getsize(dest) == (blob.size or -1):
            continue
        blob.download_to_filename(dest)
    return local_dir


def _verify_gpu() -> None:
    """Fail-fast with a clear message if no CUDA device is available.

    LRP on so400m on CPU is impractical (research: Environment Availability).
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA device available. SigLIP-2 + dynamic-LRP on so400m is "
            "impractical on CPU — this pipeline requires the GPU VM. Verify "
            "`torch.cuda.is_available()` on the runtime VM."
        )


def get_model_and_processor(
    local_dir: Optional[str] = None,
) -> Tuple[object, object]:
    """Return the cached ``(model, processor)`` singleton.

    First call: cache-first download from GCS, then
    ``AutoModel.from_pretrained(local_dir, attn_implementation="eager")
    .to(DEVICE).eval()`` and ``AutoProcessor.from_pretrained(local_dir)``.
    Subsequent calls return the SAME objects (never re-loaded, never HF).
    """
    global _MODEL, _PROCESSOR

    if _MODEL is not None and _PROCESSOR is not None:
        return _MODEL, _PROCESSOR

    with _LOCK:
        if _MODEL is not None and _PROCESSOR is not None:
            return _MODEL, _PROCESSOR

        from transformers import AutoModel, AutoProcessor

        _verify_gpu()
        target_dir = local_dir or config.LOCAL_MODEL_CACHE_DIR
        _download_model_mirror(target_dir)

        model = (
            AutoModel.from_pretrained(
                target_dir, attn_implementation="eager"
            )
            .to(config.DEVICE)
            .eval()
        )
        processor = AutoProcessor.from_pretrained(target_dir)

        _MODEL, _PROCESSOR = model, processor
        return _MODEL, _PROCESSOR


def reset_singleton() -> None:
    """Test hook: clear the cached singleton."""
    global _MODEL, _PROCESSOR
    with _LOCK:
        _MODEL = None
        _PROCESSOR = None
