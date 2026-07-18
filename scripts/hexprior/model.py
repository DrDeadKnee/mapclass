"""
Masked-hex terrain prior model (Phase B).

A permutation-invariant transformer encoder over hex tokens: each token is a
hex's soft terrain label (25-dim distribution) plus a Fourier encoding of its
window-relative position. Masked tokens have their label replaced by a
learned mask embedding; the model predicts the full 25-class distribution
everywhere and is trained with KL(target ‖ predicted) on the masked tokens
only.

Positions are relative and unnormalised (unit = one hex spacing), so the
same weights apply to any window size and to invented geography.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hexprior.labels import N_COMPOSITE

DEFAULT_N_FREQS = 8
# Wavelengths spanning one hex spacing up to ~2 window diameters.
_MIN_WAVELENGTH = 2.0
_MAX_WAVELENGTH = 128.0


def fourier_features(xy: torch.Tensor, n_freqs: int = DEFAULT_N_FREQS) -> torch.Tensor:
    """
    Sin/cos features over geometrically spaced wavelengths for each of x, y.

    xy: (..., 2) → (..., 4 * n_freqs)
    """
    wavelengths = torch.tensor(
        np.geomspace(_MIN_WAVELENGTH, _MAX_WAVELENGTH, n_freqs, dtype=np.float32),
        device=xy.device,
    )
    ang = 2.0 * torch.pi * xy.unsqueeze(-1) / wavelengths  # (..., 2, F)
    feats = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # (..., 2, 2F)
    return feats.flatten(-2)


class HexPriorModel(nn.Module):
    def __init__(
        self,
        n_classes: int = N_COMPOSITE,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.0,
        n_freqs: int = DEFAULT_N_FREQS,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_freqs = n_freqs
        self.label_proj = nn.Linear(n_classes, d_model)
        self.pos_proj = nn.Linear(4 * n_freqs, d_model)
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes),
        )

    def forward(
        self,
        soft: torch.Tensor,   # (B, K, C) target distributions (any values at masked/pad)
        xy: torch.Tensor,     # (B, K, 2) window-relative positions
        mask: torch.Tensor,   # (B, K) bool — True = label hidden from the model
        pad: torch.Tensor | None = None,  # (B, K) bool — True = padding token
    ) -> torch.Tensor:
        """Return (B, K, C) class logits."""
        x = self.label_proj(soft)
        x = torch.where(mask.unsqueeze(-1), self.mask_token.to(x.dtype), x)
        x = x + self.pos_proj(fourier_features(xy, self.n_freqs))
        x = self.encoder(x, src_key_padding_mask=pad)
        return self.head(x)


def masked_kl_loss(
    logits: torch.Tensor,   # (B, K, C)
    target: torch.Tensor,   # (B, K, C) rows sum to 1
    mask: torch.Tensor,     # (B, K) bool — score these tokens only
) -> torch.Tensor:
    """Mean KL(target ‖ softmax(logits)) over masked tokens."""
    logp = F.log_softmax(logits, dim=-1)
    # Σ t·log t − Σ t·log p, with 0·log 0 = 0 via xlogy.
    kl = torch.xlogy(target, target).sum(-1) - (target * logp).sum(-1)
    denom = mask.sum().clamp(min=1)
    return (kl * mask).sum() / denom


@torch.no_grad()
def masked_top1_accuracy(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Fraction of masked tokens whose argmax class matches the target's."""
    hit = logits.argmax(-1) == target.argmax(-1)
    denom = mask.sum().clamp(min=1)
    return (hit & mask).sum() / denom
