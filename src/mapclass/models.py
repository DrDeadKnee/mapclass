"""Plain per-model registry for the v1.1 multi-model dynamic-LRP comparison.

KEEP IT SIMPLE (REQUIREMENTS.md Out of Scope): this is a plain
``dict[str, dict]`` with no typing-scaffold, no geometry class, no
conformance-test framework, no pre-flight-guard scaffold (all explicitly
rejected by the user as over-complicated). Each of the four entries is a
plain dict of three callables + a geometry dict:

    MODEL_REGISTRY[name] = {
        "load_fn":         () -> (model, processor)   # GCS-cache-first, no HF runtime
        "build_inputs_fn": (model, processor, pil, query, device)
                               -> (forward_inputs: dict, img_tensor)
        "target_fn":       (model, output, forward_inputs) -> scalar-ish tensor
        "patch_geom":      {"patch_size": int, "img_dim": int, "has_cls": bool}
    }

ATTR-01 tensor-identity invariant: ``build_inputs_fn`` returns the SAME
``requires_grad`` pixel tensor object that ``forward_inputs["pixel_values"]``
carries — ``attribution.attribute`` runs ``model(**forward_inputs)`` and
``engine.params_to_interpret = [img_tensor]`` on that one object (NO
clone/detach/re-.to() after ``requires_grad_()``).

Per-model attribution target (research/STACK.md "Per-Model Attribution
Target"; CLAUDE.md forbids pooled-embedding / image-feature targets — they
are query-INDEPENDENT; the exact forbidden symbol name is deliberately not
written here so a grep can assert this module never references it):

  * siglip2 / clip : ``output.logits_per_image[0, 0]`` — query-conditioned
    image-text similarity (verbatim v1.0 target; CLIP is the SAME wiring).
  * vit_b16        : ``output.logits[0, class_id]`` — an ImageNet-1k CLASS
    logit. **NOT text-conditioned**: a plain ViT has no text tower, so the
    free-text query CANNOT condition it. We map the fixed query string to the
    nearest ImageNet label via ``model.config.label2id`` (substring match);
    if no label matches we fall back to a FIXED, documented class id. This
    asymmetry is surfaced in the panel title by Task 3 (it must NOT be
    misread as apples-to-apples with the text-conditioned models).
  * paligemma      : ``output.logits[0, answer_pos, answer_token_id]`` — the
    logit of a teacher-forced single-token answer. DECISION (recorded here +
    in the panel label): a "<image> {query}" presence prompt; ``answer_pos``
    = the LAST prompt position (``logits[0, -1, :]``); ``answer_token_id`` =
    the FIRST sub-token the processor tokenizer assigns to the fixed answer
    word ``"yes"``. PaliGemma-3B may OOM the L4 — that is an EXPECTED,
    recorded "no heatmap" result, not a bug to engineer around.

GCS loading (config.GCS_MODELS_PREFIX == "models/"): each ``load_fn``
cache-first downloads its ``models/<dir>/`` mirror to a per-model local cache
then ``<HFClass>.from_pretrained(local_dir, attn_implementation="eager")
.to(DEVICE).eval()``. NO Hugging Face call at runtime. PaliGemma is loaded
``torch_dtype=bfloat16`` (3B). PaliGemma weights are ALREADY at
``models/paligemma-3b-mix-224/`` (PaliGemma-1 mix variant, NOT gated — no HF
token / Gemma-license step anywhere). CLIP + ViT-b-16 are mirrored by Task 2
(``openai/clip-vit-large-patch14`` / ``google/vit-base-patch16-224``, both
non-gated).
"""

from __future__ import annotations

import os
from typing import Tuple

from mapclass import config
from mapclass.ingest_images import get_bucket

# SigLIP text encoding requires this fixed max_length (v1.0 data_loader pin).
SIGLIP_MAX_LENGTH = 64

# Per-model GCS mirror dirs (under config.GCS_MODELS_PREFIX == "models/").
_GCS_DIRS = {
    "siglip2": config.MODEL_GCS_DIR,            # models/siglip2-so400m-patch14-384
    "siglip2_base": "models/siglip2-base-patch16-224",
    "clip": "models/clip-vit-large-patch14",
    "vit_b16": "models/vit-base-patch16-224",
    "paligemma": "models/paligemma-3b-mix-224",
}

# Fixed ViT class fallback if no ImageNet label substring-matches the query.
# 'lakeside, lakeshore' is the closest ImageNet-1k class to a map "river"
# (id 975 in the standard torchvision/HF ImageNet-1k ordering); used only as a
# documented fallback — the primary path is the label2id substring match.
_VIT_FALLBACK_CLASS_ID = 975

# PaliGemma teacher-forced single-token answer word (presence prompt).
_PALIGEMMA_ANSWER_WORD = "yes"


def _download_model_mirror(gcs_dir: str, local_dir: str) -> str:
    """Cache-first download of a GCS model dir to ``local_dir`` (no HF).

    Generalized from model_loader._download_model_mirror: only blobs missing
    locally (or size-mismatched) are fetched, so a warm cache is a near no-op.
    """
    os.makedirs(local_dir, exist_ok=True)
    bucket = get_bucket()
    prefix = f"{gcs_dir}/"
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


def _local_cache_dir(name: str) -> str:
    """Per-model local cache dir (sibling of the v1.0 SigLIP-2 cache)."""
    return os.path.join(config.LOCAL_MODEL_CACHE_DIR, name)


# --------------------------------------------------------------------------
# load_fn — cache-first GCS download then from_pretrained (no HF at runtime).
# --------------------------------------------------------------------------
def _load_siglip2() -> Tuple[object, object]:
    from transformers import AutoModel, AutoProcessor

    d = _download_model_mirror(_GCS_DIRS["siglip2"], _local_cache_dir("siglip2"))
    model = (
        AutoModel.from_pretrained(d, attn_implementation="eager")
        .to(config.DEVICE)
        .eval()
    )
    return model, AutoProcessor.from_pretrained(d)


def _load_siglip2_base() -> Tuple[object, object]:
    """SigLIP-2 base (patch16-224), loaded fp32 with the dynamicLRP-traversable
    pooling head patched in. ~10x less relevance memory than so400m, so the LRP
    relevance pass fits the L4 in fp32 (~3 GB). The split-free pooling head
    (siglip_lrp_patch) is required for the engine to traverse it; it is forward-
    identical so it is harmless for the IG path too.
    """
    from transformers import AutoModel, AutoProcessor

    from mapclass.siglip_lrp_patch import attach_lrp_pooling_head

    d = _download_model_mirror(
        _GCS_DIRS["siglip2_base"], _local_cache_dir("siglip2_base")
    )
    model = (
        AutoModel.from_pretrained(d, attn_implementation="eager")
        .to(config.DEVICE)
        .eval()
    )
    attach_lrp_pooling_head(model)
    return model, AutoProcessor.from_pretrained(d)


def _load_clip() -> Tuple[object, object]:
    from transformers import CLIPModel, CLIPProcessor

    d = _download_model_mirror(_GCS_DIRS["clip"], _local_cache_dir("clip"))
    model = (
        CLIPModel.from_pretrained(d, attn_implementation="eager")
        .to(config.DEVICE)
        .eval()
    )
    return model, CLIPProcessor.from_pretrained(d)


def _load_vit_b16() -> Tuple[object, object]:
    from transformers import ViTForImageClassification, ViTImageProcessor

    d = _download_model_mirror(_GCS_DIRS["vit_b16"], _local_cache_dir("vit_b16"))
    model = (
        ViTForImageClassification.from_pretrained(d, attn_implementation="eager")
        .to(config.DEVICE)
        .eval()
    )
    return model, ViTImageProcessor.from_pretrained(d)


def _load_paligemma() -> Tuple[object, object]:
    import torch
    from transformers import (
        PaliGemmaForConditionalGeneration,
        PaliGemmaProcessor,
    )

    d = _download_model_mirror(
        _GCS_DIRS["paligemma"], _local_cache_dir("paligemma")
    )
    model = (
        PaliGemmaForConditionalGeneration.from_pretrained(
            d, attn_implementation="eager", torch_dtype=torch.bfloat16
        )
        .to(config.DEVICE)
        .eval()
    )
    return model, PaliGemmaProcessor.from_pretrained(d)


# --------------------------------------------------------------------------
# build_inputs_fn — returns (forward_inputs: dict, img_tensor) with the SAME
# requires_grad pixel tensor object reused in both (ATTR-01 identity).
# --------------------------------------------------------------------------
def _grad_pixels(inputs):
    """Mark pixel_values requires_grad IN PLACE and return that one object.

    NO clone/detach/re-.to() afterward — the returned object IS what the
    forward and engine.params_to_interpret both consume (Pattern 2).
    """
    img_tensor = inputs["pixel_values"].requires_grad_()
    inputs["pixel_values"] = img_tensor
    return img_tensor


def _build_inputs_siglip2(model, processor, pil, query, device):
    inputs = processor(
        text=[query],
        images=[pil],
        padding="max_length",
        max_length=SIGLIP_MAX_LENGTH,  # REQUIRED for the SigLIP tokenizer
        return_tensors="pt",
    ).to(device)
    img_tensor = _grad_pixels(inputs)
    return dict(inputs), img_tensor


def _build_inputs_clip(model, processor, pil, query, device):
    inputs = processor(
        text=[query], images=[pil], return_tensors="pt", padding=True
    ).to(device)
    img_tensor = _grad_pixels(inputs)
    return dict(inputs), img_tensor


def _build_inputs_vit_b16(model, processor, pil, query, device):
    # NO text tower — image-only. The query does not enter the forward; it is
    # mapped to a class id in target_fn (documented asymmetry).
    inputs = processor(images=[pil], return_tensors="pt").to(device)
    img_tensor = _grad_pixels(inputs)
    return dict(inputs), img_tensor


def _build_inputs_paligemma(model, processor, pil, query, device):
    import torch

    inputs = processor(
        text="<image> " + query,
        images=[pil],
        return_tensors="pt",
    ).to(device)
    # bf16 weights -> cast the pixel tensor to match before requires_grad so
    # the SAME object (correct dtype) flows into forward + the engine.
    if inputs["pixel_values"].dtype != torch.bfloat16:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    img_tensor = _grad_pixels(inputs)
    return dict(inputs), img_tensor


# --------------------------------------------------------------------------
# target_fn — per-model query-conditioned scalar (NO pooled embedding).
# --------------------------------------------------------------------------
def _target_logits_per_image(model, output, forward_inputs):
    # siglip2 / clip: query-conditioned image-text similarity (Pitfall B).
    return output.logits_per_image[0, 0]


def _target_siglip_cosine(model, output, forward_inputs):
    # siglip2_base + dynamicLRP: attribute the COSINE similarity (normalized
    # image·text), BEFORE logit_scale/logit_bias. SigLIP's logit_bias is a large
    # additive constant that acts as an LRP relevance SINK — attributing on
    # logits_per_image drains ~all relevance into the bias and the heatmap
    # vanishes. The raw cosine sim is the same query-conditioned quantity and
    # keeps relevance flowing to the pixels (verified non-vanishing on the L4).
    #
    # Return the (1,1) matmul (NOT a 0-dim [0,0] scalar): the engine accepts the
    # 2-D form on the first try, so attribute()'s scalar->2d->1d fallback ladder
    # never re-runs LRPEngine.run on the same (already-mutated) forward graph —
    # that re-run intermittently dead-ends ("No valid curnode candidate").
    return output.image_embeds @ output.text_embeds.t()


def _resolve_vit_class_id(model, query):
    """Map the free-text query -> nearest ImageNet-1k class id (documented).

    Primary: substring match against model.config.label2id label strings.
    Fallback: a FIXED documented class id (_VIT_FALLBACK_CLASS_ID).
    """
    label2id = getattr(model.config, "label2id", None) or {}
    q = query.strip().lower()
    q_words = [w for w in q.replace("-", " ").split() if len(w) > 2]
    for label, idx in label2id.items():
        ll = str(label).lower()
        if any(w in ll for w in q_words):
            return int(idx), str(label)
    id2label = getattr(model.config, "id2label", None) or {}
    return _VIT_FALLBACK_CLASS_ID, str(
        id2label.get(_VIT_FALLBACK_CLASS_ID, _VIT_FALLBACK_CLASS_ID)
    )


def _target_vit_class_logit(model, output, forward_inputs):
    # CLASS-conditioned (NOT text-conditioned). The query->class id is
    # resolved once here; Task 3 surfaces id:label in the panel title.
    query = forward_inputs.get("_query", "a river")
    class_id, _label = _resolve_vit_class_id(model, query)
    return output.logits[0, class_id]


def _target_paligemma_answer_logit(model, output, forward_inputs):
    # answer_pos = LAST prompt position; answer_token_id = first sub-token of
    # the fixed answer word "yes" (teacher-forced single-token answer).
    tok = getattr(model, "_mapclass_tok", None)
    answer_token_id = getattr(model, "_mapclass_answer_token_id", None)
    if answer_token_id is None:
        # Resolve lazily/robustly from the processor stashed on the model.
        proc = getattr(model, "_mapclass_processor", None)
        if proc is not None and hasattr(proc, "tokenizer"):
            ids = proc.tokenizer(
                _PALIGEMMA_ANSWER_WORD, add_special_tokens=False
            )["input_ids"]
            answer_token_id = int(ids[0])
        else:  # last-resort: a stable id; documented in the panel label
            answer_token_id = 0
    return output.logits[0, -1, answer_token_id]


# --------------------------------------------------------------------------
# The registry. Plain dict. Four entries. No framework.
# --------------------------------------------------------------------------
MODEL_REGISTRY = {
    "siglip2": {
        "load_fn": _load_siglip2,
        "build_inputs_fn": _build_inputs_siglip2,
        "target_fn": _target_logits_per_image,
        "patch_geom": {"patch_size": 14, "img_dim": 384, "has_cls": False},
    },
    "siglip2_base": {
        "load_fn": _load_siglip2_base,
        "build_inputs_fn": _build_inputs_siglip2,
        "target_fn": _target_siglip_cosine,
        "patch_geom": {"patch_size": 16, "img_dim": 224, "has_cls": False},
    },
    "clip": {
        "load_fn": _load_clip,
        "build_inputs_fn": _build_inputs_clip,
        "target_fn": _target_logits_per_image,
        "patch_geom": {"patch_size": 14, "img_dim": 224, "has_cls": True},
    },
    "vit_b16": {
        "load_fn": _load_vit_b16,
        "build_inputs_fn": _build_inputs_vit_b16,
        "target_fn": _target_vit_class_logit,
        "patch_geom": {"patch_size": 16, "img_dim": 224, "has_cls": True},
    },
    "paligemma": {
        "load_fn": _load_paligemma,
        "build_inputs_fn": _build_inputs_paligemma,
        "target_fn": _target_paligemma_answer_logit,
        "patch_geom": {"patch_size": 14, "img_dim": 224, "has_cls": False},
    },
}
