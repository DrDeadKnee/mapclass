"""Image data loader: manifest entry → GCS object → (requires_grad tensor, PIL).

Local-disk-cache-first (DATA-04): the image bytes are cached under
``config.LOCAL_IMAGE_CACHE_DIR`` keyed by id; GCS is contacted ONLY on a cache
miss. The returned ``img_tensor`` is the EXACT object Plan 03's attribution
passes to both the forward AND ``engine.params_to_interpret`` — it is
``inputs["pixel_values"].requires_grad_()`` with NO ``.clone()`` /
``.detach()`` / re-``.to()`` applied afterward (Pattern 2 tensor-identity
invariant; the live ``grad_fn`` graph must not be broken).

The processor is the shared one from the GCS-mirrored model dir (key link to
model_loader). SigLIP text encoding REQUIRES ``padding="max_length",
max_length=64`` (the SigLIP tokenizer's ``model_max_length`` is 64,
Assumption A2 — confirmed at runtime).
"""

from __future__ import annotations

import os
from typing import Tuple

from mapclass import config
from mapclass.ingest_images import get_bucket

SIGLIP_MAX_LENGTH = 64


def _cached_image_path(entry_id: str) -> str:
    """Return the local cache path for an id; download from GCS on miss."""
    os.makedirs(config.LOCAL_IMAGE_CACHE_DIR, exist_ok=True)
    local_path = os.path.join(config.LOCAL_IMAGE_CACHE_DIR, f"{entry_id}.jpg")
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path  # cache hit — no GCS call (DATA-04)

    bucket = get_bucket()
    blob = bucket.blob(f"{config.GCS_DATA_PREFIX}{entry_id}.jpg")
    blob.download_to_filename(local_path)
    return local_path


def load_slice(
    entry: dict, query: str, device: str = None, processor=None
) -> Tuple[object, object, object, object]:
    """Load one (map, query) slice for attribution.

    Returns ``(img_tensor, input_ids, attention_mask, pil)`` where:
      * ``img_tensor`` == ``inputs["pixel_values"].requires_grad_()`` — the
        SAME object that must flow into both the forward and
        ``params_to_interpret`` downstream. NOT cloned/detached/re-.to()'d.
      * ``pil`` is the original ``PIL.Image`` (RGB) for the overlay.

    The processor defaults to the shared GCS-mirrored singleton processor.
    Confirms the tokenizer ``model_max_length`` is 64 (Assumption A2).
    """
    from PIL import Image

    if device is None:
        device = config.DEVICE
    if processor is None:
        # Lazy import to avoid a hard model_loader dependency for callers
        # that pass an explicit processor (e.g. unit tests).
        from mapclass.model_loader import get_model_and_processor

        _model, processor = get_model_and_processor()

    # Assumption A2: SigLIP text encoding REQUIRES padding="max_length",
    # max_length=64 — which is passed explicitly to the processor below, so
    # encoding is correct regardless of the tokenizer's reported
    # model_max_length. The real loaded SigLIP-2 processor reports the
    # transformers "no limit" sentinel (a VERY_LARGE_INT), NOT 64; the older
    # exact-equality assertion (Plan 01-02) wrongly hard-failed on that
    # sentinel. Only fail if the tokenizer advertises a concrete SMALLER
    # window than 64, which would silently truncate the query (Plan 01-03
    # Rule 1 bug fix).
    _HF_NO_LIMIT_SENTINEL = int(1e30)  # transformers' "unset" model_max_length
    tok = getattr(processor, "tokenizer", None)
    _mml = getattr(tok, "model_max_length", None) if tok is not None else None
    if (
        _mml is not None
        and _mml < SIGLIP_MAX_LENGTH
        and _mml < _HF_NO_LIMIT_SENTINEL
    ):
        raise AssertionError(
            "SigLIP tokenizer model_max_length "
            f"({_mml}) is smaller than the required {SIGLIP_MAX_LENGTH} — "
            "the query would be silently truncated (Assumption A2)."
        )

    local_path = _cached_image_path(entry["id"])
    pil = Image.open(local_path).convert("RGB")

    inputs = processor(
        text=[query],
        images=[pil],
        padding="max_length",
        max_length=SIGLIP_MAX_LENGTH,  # REQUIRED for SigLIP tokenizer
        return_tensors="pt",
    ).to(device)

    # SAME object flows downstream — no clone/detach/re-.to() after this line.
    img_tensor = inputs["pixel_values"].requires_grad_()
    # The fixed-resolution SigLIP-2 processor pads text to a fixed
    # max_length and does NOT emit an attention_mask (the SigLIP text model
    # forward accepts attention_mask=None). Return None rather than
    # KeyError-ing when it is absent (Plan 01-03 Rule 1 bug fix).
    attention_mask = inputs.get("attention_mask", None)
    return img_tensor, inputs["input_ids"], attention_mask, pil
