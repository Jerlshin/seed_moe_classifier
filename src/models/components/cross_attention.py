"""Cross-attention refinement block (paper Section 5.5, Eqs. 11-12).

The seed-type-aware feature ``h' = h + P(p_s)`` is the query; the raw MoE feature
``h`` is both key and value:

    a   = softmax(Q K^T / sqrt(d)) V
    h'' = LayerNorm(a + Q)

Sequence length is load-bearing
-------------------------------

With a key/value sequence of length 1 -- which is what a pooled feature *vector*
gives you -- the softmax is over a single key and evaluates to ``1`` for any
``Q`` and ``K``:

    softmax(QK^T/sqrt(d)) = softmax([s]) = [1]
    => a = 1 * V = W_O(W_V x)          -- Q and K appear nowhere in the output
    => da/dW_Q = da/dW_K = 0           -- exactly zero gradient, forever

So under a pooled input this block is an affine map, ``num_heads`` is inert, and
the returned ``attn_weights`` is identically 1.0 (any attention-map figure drawn
from it is a constant image). ``mode="affine"`` therefore does not allocate a
``MultiheadAttention`` at all -- it uses the single ``nn.Linear`` that spans the
identical function class, so no parameter in the block is unreachable by
gradient and none is counted as "active" in the efficiency table.

``mode="attention"`` builds the real module and is correct exactly when the head
runs on SwinV2's ``8x8`` token grid (``token_mode="grid"``), where the softmax is
over 64 keys, the heads do something, and ``attn_weights`` is a genuine spatial
map worth plotting.

:func:`resolve_attention_mode` picks between them from the head's ``token_mode``
so the two cannot be configured inconsistently.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

ATTENTION_MODES = ("attention", "affine")


def resolve_attention_mode(token_mode: str) -> str:
    """Map the head's ``token_mode`` onto the only honest attention mode for it."""
    return "attention" if str(token_mode) == "grid" else "affine"


class CrossAttentionOutput(NamedTuple):
    features: torch.Tensor
    """``h''`` from Eq. 12, same shape as the query."""

    attn_weights: torch.Tensor | None
    """Attention map ``[batch, query_len, key_len]``, or ``None`` when not
    requested or when ``mode="affine"`` (where it would be a constant 1.0 and
    plotting it would be misleading)."""


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
        num_heads: Attention heads. Only meaningful under ``mode="attention"``.
        dropout: Dropout inside the attention (and the ``gated`` feed-forward).
        variant: ``"paper"`` implements Eq. 12 exactly. ``"gated"`` keeps the
            earlier pre-norm block with an adaptive gate and a feed-forward
            branch; it is retained for ablation and is not the paper's design.
        mlp_ratio: Feed-forward expansion used only by the ``gated`` variant.
        mode: ``"attention"`` or ``"affine"``; see the module docstring. Use
            :func:`resolve_attention_mode` rather than setting it by hand.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        variant: str = "paper",
        mlp_ratio: int = 4,
        mode: str = "affine",
    ):
        super().__init__()
        if variant not in {"paper", "gated"}:
            raise ValueError(f"variant must be 'paper' or 'gated', got {variant!r}")
        if mode not in ATTENTION_MODES:
            raise ValueError(f"mode must be one of {ATTENTION_MODES}, got {mode!r}")
        self.variant = variant
        self.mode = mode

        if mode == "attention":
            self.attn: nn.Module | None = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.value_projection: nn.Module | None = None
        else:
            self.attn = None
            self.value_projection = nn.Linear(dim, dim)

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

        if self.attn is not None:
            attn_output, attn_weights = self.attn(
                query,
                key,
                value,
                attn_mask=attn_mask,
                need_weights=need_weights,
            )
        else:
            if value.shape[1] != 1:
                raise ValueError(
                    "mode='affine' collapses attention to a length-1 identity and is only valid "
                    f"for a single key/value token, got {value.shape[1]}. Use mode='attention' "
                    "(token_mode='grid') for multi-token inputs."
                )
            attn_output = self.value_projection(value).expand_as(query)
            attn_weights = None

        if self.variant == "paper":
            return CrossAttentionOutput(self.norm(attn_output + query), attn_weights)

        gated = query + self.gating(query) * self.dropout(attn_output)
        return CrossAttentionOutput(gated + self.mlp(self.mlp_norm(gated)), attn_weights)

    def extra_repr(self) -> str:
        return f"variant={self.variant}, mode={self.mode}"
