"""Make SigLIP-2 traversable by dynamicLRP WITHOUT touching the frozen engine.

SigLIP-2's vision encoder already uses eager separate q/k/v projections
(``Siglip2Attention`` with ``q_proj``/``k_proj``/``v_proj``) — fully covered by
the dynamicLRP engine. The ONE op the engine chokes on is in the attention
*pooling* head (``Siglip2MultiheadAttentionPoolingHead``), which uses
``torch.nn.MultiheadAttention``: its forward fuses Q/K/V into one
``in_proj_weight`` and ``split_with_sizes``-es it, and the engine's split handler
raises ``'DummyPromise' object is not iterable`` on SigLIP-2's split topology.

This module replaces ONLY that head's ``forward`` with a mathematically identical
computation expressed as three separate ``F.linear`` projections + a manual
softmax attention — exactly the ops the eager encoder already uses, all
engine-covered. No fused ``split_with_sizes`` on the relevance-carrying path.

This is a *model-side* adaptation (the dynamicLRP repo's own extension pattern;
cf. ``src/lrp_engine/model_specific/``). The vendored engine and ``VENDOR_SHA``
are untouched. ``attach_lrp_pooling_head`` asserts numerical equivalence against
the original head before swapping, so a botched reimplementation fails loudly
rather than silently changing the model's output.
"""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F


def _pooling_head_forward(self, hidden_state, attention_mask=None):
    """Drop-in for Siglip2MultiheadAttentionPoolingHead.forward, split-free.

    Reads the existing ``nn.MultiheadAttention`` weights so it is bit-for-bit the
    same linear algebra — only the qkv projection is done as three slices of
    ``in_proj_weight`` (separate matmuls) instead of one fused split, and the
    attention is a manual ``softmax(qk^T / sqrt(d)) v`` (engine-covered ops).
    """
    mha = self.attention
    embed_dim = mha.embed_dim
    num_heads = mha.num_heads
    head_dim = embed_dim // num_heads

    batch_size = hidden_state.shape[0]
    src_len = hidden_state.shape[1]
    # self.probe is (1, 1, E). For a single image (batch 1) use it directly to
    # avoid a repeat/expand node the dynamicLRP engine can't traverse; only
    # materialize the repeat for true batches.
    probe = self.probe if batch_size == 1 else self.probe.repeat(batch_size, 1, 1)

    # Use the PRE-EXTRACTED separate q/k/v projections (set up in
    # attach_lrp_pooling_head). Slicing the packed in_proj_weight/in_proj_bias
    # inside the forward creates Slice nodes the engine can't traverse (a 1-D
    # bias slice vs 2-D relevance) — exactly the failure this patch removes.
    q = F.linear(probe, self._lrp_wq, self._lrp_bq)        # (B, 1, E)
    k = F.linear(hidden_state, self._lrp_wk, self._lrp_bk)  # (B, N, E)
    v = F.linear(hidden_state, self._lrp_wv, self._lrp_bv)  # (B, N, E)

    # (B, L, E) -> (B, H, L, hd) with contiguous head split (matches nn.MHA).
    q = q.view(batch_size, 1, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, src_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, src_len, num_heads, head_dim).transpose(1, 2)

    scores = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)  # (B, H, 1, N)
    if attention_mask is not None:
        # transformers reshapes to (B*H, tgt, src); fold to (B, H, 1, N).
        scores = scores + attention_mask.view(
            batch_size, num_heads, 1, src_len
        )
    attn = scores.softmax(dim=-1)
    out = attn @ v                                          # (B, H, 1, hd)
    out = out.transpose(1, 2).reshape(batch_size, 1, embed_dim)
    out = F.linear(out, mha.out_proj.weight, mha.out_proj.bias)

    residual = out
    out = self.layernorm(out)
    out = residual + self.mlp(out)
    return out[:, 0]


def attach_lrp_pooling_head(model, atol: float = None):
    """Swap the SigLIP-2 pooling head forward for the split-free version.

    Asserts the replacement matches the original head on random input (so a
    reimplementation error fails loudly), then patches in place. Idempotent.
    Returns ``model``.
    """
    head = model.vision_model.head
    if getattr(head, "_mapclass_lrp_patched", False):
        return model

    mha = head.attention
    embed_dim = mha.embed_dim
    dev = head.probe.device
    dtype = head.probe.dtype

    # Pre-extract the packed QKV into separate leaf parameters so the patched
    # forward never slices in_proj_weight/in_proj_bias (those slices are the
    # ops the engine can't traverse). Identical values -> identical math.
    w = mha.in_proj_weight.detach()
    b = mha.in_proj_bias.detach()
    head._lrp_wq = torch.nn.Parameter(w[:embed_dim].clone())
    head._lrp_wk = torch.nn.Parameter(w[embed_dim:2 * embed_dim].clone())
    head._lrp_wv = torch.nn.Parameter(w[2 * embed_dim:].clone())
    head._lrp_bq = torch.nn.Parameter(b[:embed_dim].clone())
    head._lrp_bk = torch.nn.Parameter(b[embed_dim:2 * embed_dim].clone())
    head._lrp_bv = torch.nn.Parameter(b[2 * embed_dim:].clone())

    # Tolerance scales with precision: fp32 is exact-ish; bf16/fp16 carry ~2-3
    # significant digits, so a multi-op attention reformulation differs at ~1e-2.
    if atol is None:
        atol = 1e-3 if dtype in (torch.float32, torch.float64) else 5e-2
    x = torch.randn(2, 17, embed_dim, device=dev, dtype=dtype)
    with torch.no_grad():
        ref = head.forward(x)                              # original nn.MHA path
        new = _pooling_head_forward(head, x)               # split-free path
    max_abs = (ref - new).abs().max().item()
    if not torch.allclose(ref, new, atol=atol):
        raise AssertionError(
            "split-free SigLIP-2 pooling head diverges from the original "
            f"(max|delta|={max_abs:.3e} > atol={atol}). Refusing to patch."
        )

    head.forward = types.MethodType(_pooling_head_forward, head)
    head._mapclass_lrp_patched = True
    head._mapclass_lrp_equiv_maxabs = max_abs
    return model
