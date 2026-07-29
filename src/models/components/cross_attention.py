"""Cross-attention refinement block (paper Section 5.5, Eq. 13-14).

The seed-type-aware feature ``h' = h + P(p_s)`` is the query; the raw MoE feature
``h`` is both key and value:

    a   = softmax(Q K^T / sqrt(d)) V
    h'' = LayerNorm(a + Q)

Note on sequence length: the paper operates on a pooled feature *vector*, so the
key/value sequence has length 1 and the softmax is over a single key (weight 1.0).
Cross-attention therefore reduces to a learned projection of ``h`` plus a residual
from ``h'``. That is a faithful reading of the paper; the machinery generalises
unchanged to multi-token inputs if a future variant keeps the patch tokens.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn


class CrossAttentionOutput(NamedTuple):
    features: torch.Tensor
    """``h''`` from Eq. 14, same shape as the query."""

    attn_weights: torch.Tensor | None
    """Attention map ``[batch, query_len, key_len]``, or ``None`` when not requested."""


class AdaptiveGating(nn.Module):
    """Scalar gate in ``[0, 1]`` per token, used by the ``gated`` variant."""

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(x)


class CrossAttention(nn.Module):
    """Query-key-value cross-attention with a post-attention LayerNorm residual.

    Args:
        dim: Feature width.
        num_heads: Attention heads.
        dropout: Dropout inside the attention (and the ``gated`` feed-forward).
        variant: ``"paper"`` implements Eq. 14 exactly. ``"gated"`` keeps the
            earlier pre-norm block with an adaptive gate and a feed-forward
            branch; it is retained for ablation and is not the paper's design.
        mlp_ratio: Feed-forward expansion used only by the ``gated`` variant.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        variant: str = "paper",
        mlp_ratio: int = 4,
    ):
        super().__init__()
        if variant not in {"paper", "gated"}:
            raise ValueError(f"variant must be 'paper' or 'gated', got {variant!r}")
        self.variant = variant
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)

        if variant == "gated":
            self.query_norm = nn.LayerNorm(dim)
            self.mlp_norm = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim * mlp_ratio),
                nn.GELU(),
                nn.Linear(dim * mlp_ratio, dim),
                nn.Dropout(dropout),
            )
            self.gating = AdaptiveGating(dim)
            self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> CrossAttentionOutput:
        if self.variant == "gated":
            query = self.query_norm(query)

        attn_output, attn_weights = self.attn(
            query,
            key,
            value,
            attn_mask=attn_mask,
            need_weights=need_weights,
        )

        if self.variant == "paper":
            return CrossAttentionOutput(self.norm(attn_output + query), attn_weights)

        gated = query + self.gating(query) * self.dropout(attn_output)
        return CrossAttentionOutput(gated + self.mlp(self.mlp_norm(gated)), attn_weights)
