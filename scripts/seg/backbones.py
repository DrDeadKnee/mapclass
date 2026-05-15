"""
Backbone Protocol (D-05) and its three implementations for the dense semantic
segmentation pipeline (Phase 3).

Strategy:
  - One unified ``Backbone`` typing.Protocol drives the D-05 seam: every backbone
    exposes ``feature_strides``, ``feature_channels``, and a ``forward`` that
    returns a ``list[Tensor]`` (one (B,C,h,w) per declared stride).
  - ``SiglipBackbone`` (primary): loads the Phase-1 PEFT LoRA adapter onto
    PaliGemmaForConditionalGeneration, reaches the SigLIP vision tower via a
    defensive accessor (transformers 4.x/.vision_tower vs 5.x/.model.vision_tower),
    and holds a reference to the vision tower ONLY — never the projector or
    language_model (PHASE-03 SC#2 Gemma-exclusion guarantee).  Mirrors
    ``finetune_paligemma.py::apply_lora`` load order exactly.
  - ``Dinov2Backbone`` / ``SwinBackbone``: timm-backed benchmarks satisfying the
    same protocol (EVAL-03 apples-to-apples seam; D-05).
  - Variant B (D-03a / D-06a): widens the first conv/patch-embed of ALL THREE
    backbones from 3 to 15 input channels (3 RGB + 12 prior-probability channels).
    Extra channels zero-initialised; pretrained RGB weights preserved verbatim;
    widened layer marked trainable.  Construction-only — NOT trained (D-06a).

Requires:
  pip install torch transformers>=5.8 peft>=0.10 timm>=1.0.27

Usage:
  # Offline tests (no checkpoint):
  from seg.backbones import SiglipBackbone, Dinov2Backbone, SwinBackbone
  bb = SiglipBackbone(stub_config=cfg)          # offline / unit-test mode
  bb = SiglipBackbone("google/paligemma-3b-pt-224", "path/to/adapter")  # real mode

  # Variant B (widened 15-ch patch-embed, trainable, construction-only):
  bb_b = SiglipBackbone(stub_config=cfg, variant_b=True)
  feats = bb_b(torch.randn(1, 15, 224, 224))    # 15-ch input

  # timm benchmarks (offline, pretrained=False):
  dino = Dinov2Backbone()
  swin = SwinBackbone()

Trust model:
  T-03-04: timm/HF checkpoints deserialized into process.  Prefer safetensors
           where available; model_id / adapter_dir are explicit user-supplied
           parameters, never auto-fetched from untrusted URLs.
  T-03-05: Phase-1 PEFT adapter loaded via PeftModel.from_pretrained; resolved
           vision tower asserted to be a SigLIP type (Pitfall 3).
  T-03-06: No language_model parameter or gemma module-type is reachable from
           any backbone (PHASE-03 SC#2).

Covered decisions: D-03a (Variant B), D-05 (unified protocol), D-06 (frozen),
                   D-06a (Variant B construction-only).
"""

from __future__ import annotations

import math
import types
from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Backbone Protocol (D-05)
# ---------------------------------------------------------------------------

@runtime_checkable
class Backbone(Protocol):
    """Unified backbone interface (D-05).

    All backbones expose declared strides/channels and return a list of
    (B, C, h, w) feature maps — one per declared stride — from ``forward``.
    The shared conv decoder (03-04) reads ``feature_strides`` and
    ``feature_channels`` dynamically; never hard-code level count.
    """

    feature_strides: list[int]
    """Declared spatial strides relative to the input (e.g. [14]*k or [4,8,16,32])."""

    feature_channels: list[int]
    """Channel count for each returned feature map."""

    def forward(self, x: Tensor) -> list[Tensor]:
        """Run the backbone on ``x`` (B, C, H, W) → list[(B, Ci, hi, wi)]."""
        ...


# ---------------------------------------------------------------------------
# Stub vision tower (for offline/unit-test mode — no HF checkpoint required)
# ---------------------------------------------------------------------------

class _StubPatchEmbed(nn.Module):
    """Minimal patch-embed conv matching the stub config's patch_size and hidden_size.

    The conv weights are initialised with a fixed deterministic seed so that two
    ``_StubVisionTower`` instances built from the same config produce identical
    weights.  This is required by Variant-B tests that compare a 3-ch baseline
    against the widened 15-ch version and expect channels 0:3 to be equal.
    """

    def __init__(self, in_channels: int, hidden_size: int, patch_size: int) -> None:
        super().__init__()
        # Use a fixed seed derived from config dims so any two stubs with the
        # same shape produce the same pretrained-like weights.
        with torch.random.fork_rng():
            torch.manual_seed(hidden_size * 1000 + patch_size * 100 + in_channels)
            self.proj = nn.Conv2d(
                in_channels, hidden_size, kernel_size=patch_size, stride=patch_size
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class _StubVisionTower(nn.Module):
    """
    Lightweight stand-in for a SigLIP vision tower (no SigLIP weights).
    Produces the correct (B, N, hidden_size) token layout that matches
    the real SigLIP output so offline shape tests run without a checkpoint.

    Mimics the minimal surface called by SiglipBackbone.forward:
      tower(pixel_values, interpolate_pos_encoding=True, output_hidden_states=True)
    Returns an object with .hidden_states[i] tensors of shape (B, N, hidden_size).
    """

    def __init__(self, cfg: types.SimpleNamespace) -> None:
        super().__init__()
        self.config = cfg
        # Patch embed: produces (B, hidden_size, h, w) then flattened to (B, N, C)
        self.patch_embed = _StubPatchEmbed(
            in_channels=getattr(cfg, "num_channels", 3),
            hidden_size=cfg.hidden_size,
            patch_size=cfg.patch_size,
        )
        # Single linear layer per "transformer layer" (just identity-scale for shape)
        # We need num_hidden_layers+1 states (0..num_hidden_layers) to match indexing.
        self._depth = cfg.num_hidden_layers

    def forward(
        self,
        pixel_values: Tensor,
        interpolate_pos_encoding: bool = True,
        output_hidden_states: bool = True,
    ) -> types.SimpleNamespace:
        B = pixel_values.shape[0]
        # Patch embed → (B, hidden_size, h, w) → (B, N, hidden_size)
        feat = self.patch_embed(pixel_values)   # (B, C, h, w)
        h, w = feat.shape[2], feat.shape[3]
        N = h * w
        # Flatten spatial dims → token sequence
        tokens = feat.flatten(2).transpose(1, 2)  # (B, N, hidden_size)

        # Build num_hidden_layers+1 hidden states (all equal in the stub,
        # no actual transformer computation — shape correctness only).
        hidden_states = tuple(tokens for _ in range(self._depth + 1))

        result = types.SimpleNamespace()
        result.last_hidden_state = tokens
        result.hidden_states = hidden_states
        return result


# ---------------------------------------------------------------------------
# Variant B: widen patch-embed from 3 to 15 channels (D-03a / D-06a)
# ---------------------------------------------------------------------------

def _widen_patch_embed(conv: nn.Conv2d, extra_in: int = 12) -> nn.Conv2d:
    """
    Return a new Conv2d with in_channels widened by ``extra_in``.

    D-03a Variant B:
      - Original RGB weights (channels 0:in_channels) copied verbatim.
      - Extra channels zero-initialised.
      - New conv returned with requires_grad=True (trainable under Variant B).
      - Construction-only — not trained (D-06a).
    """
    orig_in = conv.in_channels
    new_in = orig_in + extra_in
    # Clone key conv attributes
    new_conv = nn.Conv2d(
        new_in,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        # Zero-init the full weight
        new_conv.weight.zero_()
        # Copy pretrained RGB weights into channels 0:orig_in
        new_conv.weight[:, :orig_in, ...].copy_(conv.weight)
        # Copy bias if present
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)  # type: ignore[union-attr]
    # Widened patch-embed is the ONLY trainable parameter under Variant B (D-06a)
    new_conv.requires_grad_(True)
    return new_conv


# ---------------------------------------------------------------------------
# SiglipBackbone
# ---------------------------------------------------------------------------

class SiglipBackbone(nn.Module):
    """
    SigLIP-So400m/14 backbone wrapping the Phase-1 PEFT LoRA adapter.

    Load order mirrors ``finetune_paligemma.py::apply_lora`` exactly:
      1. PaliGemmaForConditionalGeneration.from_pretrained(model_id)
      2. PeftModel.from_pretrained(base, adapter_dir)        # binds Phase-1 LoRA
      3. .eval().requires_grad_(False)                       # D-06: frozen
      4. Reach vision tower via defensive accessor (Pitfall 3: 4.x/.vision_tower
         vs 5.x/.model.vision_tower); assert SigLIP type (T-03-05).

    In offline/stub mode (``stub_config`` provided, no checkpoint), a minimal
    ``_StubVisionTower`` is used instead so shape tests run without HF auth.

    Variant B (``variant_b=True``): widens the patch-embed conv from 3 to 15
    input channels (D-03a); RGB weights preserved, extras zero-init, widened
    layer trainable.  Construction-only (D-06a).

    Gemma exclusion (PHASE-03 SC#2 / T-03-06): the backbone holds a reference
    to the vision tower ONLY.  No ``language_model`` parameter or ``gemma``
    module-type is reachable from this object.
    """

    # Class-level defaults so __new__-only access (test_siglip_has_declared_strides)
    # finds the attributes without calling __init__.
    feature_strides: list[int] = [14, 14, 14]   # 3 taps; all same stride (patch 14)
    feature_channels: list[int] = [1152, 1152, 1152]

    def __init__(
        self,
        model_id: str | None = None,
        adapter_dir: str | None = None,
        stub_config: types.SimpleNamespace | None = None,
        variant_b: bool = False,
        tap_layers: tuple[int, ...] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        model_id:
            HuggingFace model ID for PaliGemma (default: ``google/paligemma-3b-pt-224``).
            Ignored when ``stub_config`` is provided.
        adapter_dir:
            Path to the Phase-1 PEFT adapter directory.  If ``None`` and
            ``stub_config`` is also ``None``, the base model is used without an
            adapter (offline shape testing against a real base model).
        stub_config:
            Offline namespace with ``hidden_size``, ``patch_size``,
            ``num_hidden_layers``, [``image_size``], [``num_channels``].
            When provided, a ``_StubVisionTower`` is used instead of loading
            the real checkpoint — no network access required.
        variant_b:
            If ``True``, widen the patch-embed to 15 input channels (D-03a).
        tap_layers:
            Indices into ``output_hidden_states`` to tap as feature grids.
            Default: evenly-spaced 3 taps ~ (depth//3, depth*2//3, depth-1)
            targeting the documented (8, 17, 26) for 27-layer SigLIP.
        """
        super().__init__()

        if stub_config is None and model_id is None:
            model_id = "google/paligemma-3b-pt-224"

        if stub_config is not None:
            # Offline/stub mode: synthesise a minimal vision tower
            self._vision_tower = _StubVisionTower(stub_config)
            hidden = stub_config.hidden_size
            depth = stub_config.num_hidden_layers
        else:
            # Real model mode: mirror apply_lora load order (finetune_paligemma.py)
            from transformers import PaliGemmaForConditionalGeneration

            base = PaliGemmaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
            )

            if adapter_dir is not None:
                from peft import PeftModel
                model = PeftModel.from_pretrained(base, adapter_dir)
            else:
                model = base

            model.eval().requires_grad_(False)

            # Defensive accessor (Pitfall 3): transformers 4.x → .vision_tower,
            # transformers 5.x → .model.vision_tower.
            inner = getattr(model, "base_model", model)
            pg = getattr(inner, "model", inner)
            vt = getattr(pg, "vision_tower", None)
            if vt is None:
                vt = pg.model.vision_tower

            # Assert resolved tower is a SigLIP type (T-03-05 — fail loudly on mismatch)
            vt_type = type(vt).__name__.lower()
            assert "siglip" in vt_type or "vision" in vt_type, (
                f"Resolved vision tower is '{type(vt).__name__}', expected SigLIP type.  "
                "Check model_id and adapter_dir (RESEARCH Pitfall 3)."
            )

            self._vision_tower = vt
            cfg = model.config.vision_config
            hidden = cfg.hidden_size
            depth = cfg.num_hidden_layers

        # Compute default tap layers: ~(depth/3, 2*depth/3, depth-1)
        # For 27-layer SigLIP this yields (8, 17, 26) as documented.
        if tap_layers is not None:
            self._tap_layers: tuple[int, ...] = tap_layers
        else:
            d = depth
            self._tap_layers = (d // 3, (2 * d) // 3, d - 1)

        k = len(self._tap_layers)
        self.feature_strides = [14] * k
        self.feature_channels = [hidden] * k

        # Variant B: widen patch-embed (D-03a)
        self._variant_b = variant_b
        if variant_b:
            pe_conv = self._vision_tower.patch_embed.proj  # type: ignore[attr-defined]
            new_conv = _widen_patch_embed(pe_conv, extra_in=12)
            self._vision_tower.patch_embed.proj = new_conv  # type: ignore[attr-defined]

    def patch_embed_weight(self) -> Tensor:
        """Return the patch-embed projection weight (out, in_ch, kH, kW).

        Used by Variant-B tests to verify RGB preservation and zero-init extras.
        """
        return self._vision_tower.patch_embed.proj.weight  # type: ignore[attr-defined]

    def forward(self, x: Tensor) -> list[Tensor]:
        """Run SigLIP vision tower → list of (B, hidden, h, w) feature grids.

        ``x`` is (B, C, H, W) where C=3 (Variant A) or C=15 (Variant B).
        Each returned map has h = w = H // patch_size.
        """
        B = x.shape[0]
        out = self._vision_tower(
            x,
            interpolate_pos_encoding=True,
            output_hidden_states=True,
        )
        grids: list[Tensor] = []
        for li in self._tap_layers:
            tok = out.hidden_states[li]           # (B, N, hidden_size)
            N = tok.shape[1]
            h = w = int(round(math.sqrt(N)))
            grids.append(tok.transpose(1, 2).reshape(B, -1, h, w))  # (B, C, h, w)
        return grids


# ---------------------------------------------------------------------------
# Dinov2Backbone
# ---------------------------------------------------------------------------

class Dinov2Backbone(nn.Module):
    """
    DINOv2 ViT benchmark backbone via timm (EVAL-03 / D-05).

    Uses ``timm.create_model`` with ``dynamic_img_size=True`` so the same
    backbone handles 224 / 448 / 896 input resolutions without reloading.
    ``get_intermediate_layers(x, n=k, reshape=True)`` returns a list of
    (B, C, h, w) maps directly, matching the Backbone protocol.

    Variant B: widens ``patch_embed.proj`` to 15 input channels (D-03a).
    Construction-only — NOT trained (D-06a).
    """

    feature_strides: list[int] = [14, 14, 14]
    feature_channels: list[int] = [384, 384, 384]

    def __init__(
        self,
        model_name: str = "vit_small_patch14_dinov2",
        pretrained: bool = False,
        num_taps: int = 3,
        variant_b: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        model_name:
            timm model tag.  Default ``vit_small_patch14_dinov2`` (patch14,
            embed_dim 384, no CLS stripping needed — timm handles it).
            Use ``timm.list_models("*dinov2*")`` to enumerate available tags.
        pretrained:
            Download pretrained weights (requires network). Default ``False``
            so offline tests pass.
        num_taps:
            Number of intermediate-layer taps returned by forward (D-05 declares
            this many strides/channels).
        variant_b:
            Widen patch-embed to 15 channels (D-03a).
        """
        super().__init__()
        import timm as _timm

        self._model = _timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            dynamic_img_size=True,
        ).eval().requires_grad_(False)

        self._num_taps = num_taps
        embed_dim = self._model.embed_dim

        self.feature_strides = [14] * num_taps
        self.feature_channels = [embed_dim] * num_taps

        self._variant_b = variant_b
        if variant_b:
            pe_conv = self._model.patch_embed.proj
            new_conv = _widen_patch_embed(pe_conv, extra_in=12)
            self._model.patch_embed.proj = new_conv

    def patch_embed_weight(self) -> Tensor:
        """Return the patch-embed projection weight for Variant-B checks."""
        return self._model.patch_embed.proj.weight

    def forward(self, x: Tensor) -> list[Tensor]:
        """timm get_intermediate_layers(reshape=True) → list[(B,C,h,w)]."""
        return list(self._model.get_intermediate_layers(x, n=self._num_taps, reshape=True))


# ---------------------------------------------------------------------------
# SwinBackbone
# ---------------------------------------------------------------------------

class SwinBackbone(nn.Module):
    """
    Swin Transformer benchmark backbone via timm ``features_only`` (EVAL-03 / D-05).

    Natively hierarchical: 4 levels at strides [4, 8, 16, 32].  The decoder reads
    ``feature_strides`` / ``feature_channels`` dynamically (Pitfall 4: never hard-code
    level count).

    Swin timm outputs are (B, H, W, C) — transposed to (B, C, H, W) before return
    so the protocol's (B, C, h, w) contract is satisfied uniformly.

    Variant B: widens ``patch_embed.proj`` to 15 channels (D-03a).
    Construction-only — NOT trained (D-06a).
    """

    feature_strides: list[int] = [4, 8, 16, 32]
    feature_channels: list[int] = [128, 256, 512, 1024]

    def __init__(
        self,
        model_name: str = "swin_base_patch4_window7_224",
        pretrained: bool = False,
        variant_b: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        model_name:
            timm Swin tag.  Default ``swin_base_patch4_window7_224``.
            Use ``timm.list_models("swin_base*")`` for alternatives.
        pretrained:
            Download pretrained weights. Default ``False`` for offline tests.
        variant_b:
            Widen patch-embed to 15 channels (D-03a).
        """
        super().__init__()
        import timm as _timm

        self._model = _timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        ).eval().requires_grad_(False)

        # Declare strides/channels from timm feature_info (Pitfall 4: dynamic)
        self.feature_strides = list(self._model.feature_info.reduction())
        self.feature_channels = list(self._model.feature_info.channels())

        self._variant_b = variant_b
        if variant_b:
            pe_conv = self._model.patch_embed.proj
            new_conv = _widen_patch_embed(pe_conv, extra_in=12)
            self._model.patch_embed.proj = new_conv

    def patch_embed_weight(self) -> Tensor:
        """Return the patch-embed projection weight for Variant-B checks."""
        return self._model.patch_embed.proj.weight

    def forward(self, x: Tensor) -> list[Tensor]:
        """timm Swin features_only → list[(B,C,h,w)].

        timm Swin returns (B,H,W,C) per level — permuted to (B,C,H,W) here
        so the protocol's spatial-last convention is satisfied uniformly.
        """
        raw = self._model(x)   # list of (B, H, W, C) tensors
        return [f.permute(0, 3, 1, 2).contiguous() for f in raw]
