"""Sparse Mixture-of-Experts block (paper Section 5.2, Eq. 8).

Six transformer experts with a Top-K gate that activates the K most relevant
experts per sample:

    h = sum_{i in Top-K} G_i * E_i(z),    h in R^d

``z`` is the pooled DINO embedding, so every sample is a single token. The gate
is a linear projection followed by a softmax over experts; the Top-K weights are
renormalised so that the selected experts form a convex combination over the
selection.

**K = 2, revised.** The submitted manuscript specifies ``K = 4`` of ``E = 6``.
The revision routes ``K = 2``, which halves the experts evaluated per sample and
therefore the active parameter count, at unchanged total capacity. ``DEFAULT_TOP_K``
below is the single place that value is defined; :mod:`src.utils.efficiency`
quantifies the resulting saving for the revision's efficiency table.

:class:`DenseExpertBlock` is the ``use_moe=False`` ablation. It exposes the same
:class:`MoEOutput` contract with a degenerate single-expert gate, so every
downstream consumer -- losses, metrics, trackers -- runs unmodified.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_NUM_EXPERTS = 6
DEFAULT_TOP_K = 2


class MoEOutput(NamedTuple):
    """Everything downstream code and the regularisers need from one MoE pass."""

    features: torch.Tensor
    """``h`` from Eq. 8, shape ``[batch, embed_dim]``."""

    gate_probs: torch.Tensor
    """Full gate distribution ``G``, shape ``[batch, num_experts]``, rows sum to 1."""

    top_k_indices: torch.Tensor
    """Indices of the selected experts, shape ``[batch, top_k]``."""

    top_k_weights: torch.Tensor
    """Renormalised weights of the selected experts, shape ``[batch, top_k]``."""

    dispatch_weights: torch.Tensor
    """``top_k_weights`` scattered back over all experts, zero elsewhere."""


class TransformerExpert(nn.Module):
    """A single expert: post-norm transformer block over a length-1 token sequence."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_output)
        return self.norm2(x + self.mlp(x))


class MixtureOfExperts(nn.Module):
    """Top-K sparse MoE over ``num_experts`` transformer experts.

    Args:
        embed_dim: Width of the token the experts operate on.
        num_experts: Expert count (6).
        mlp_dim: Hidden width inside each expert's feed-forward block.
        top_k: Experts activated per sample (2 after the revision; the submitted
            manuscript used 4).
        num_heads: Attention heads inside each expert.
        dropout: Dropout applied inside the experts.
        renormalize_top_k: Rescale the selected gate weights to sum to 1.
        sparse_dispatch: Run each expert only on the samples routed to it. Set
            ``False`` to evaluate every expert densely, which is slower but keeps
            gradients flowing to unselected experts (useful for debugging).
    """

    def __init__(
        self,
        embed_dim: int,
        num_experts: int = DEFAULT_NUM_EXPERTS,
        mlp_dim: int = 512,
        top_k: int = DEFAULT_TOP_K,
        num_heads: int = 8,
        dropout: float = 0.0,
        renormalize_top_k: bool = True,
        sparse_dispatch: bool = True,
    ):
        super().__init__()
        if num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {num_experts}")
        if not 1 <= top_k <= num_experts:
            raise ValueError(f"top_k must be in [1, {num_experts}], got {top_k}")

        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.renormalize_top_k = renormalize_top_k
        self.sparse_dispatch = sparse_dispatch
        self.experts = nn.ModuleList(
            TransformerExpert(embed_dim, num_heads, mlp_dim, dropout) for _ in range(num_experts)
        )
        self.gate = nn.Linear(embed_dim, num_experts)

    def forward(self, x: torch.Tensor) -> MoEOutput:
        if x.ndim != 2:
            raise ValueError(f"MoE expects [batch, embed_dim] input, got shape {tuple(x.shape)}")
        if x.shape[-1] != self.embed_dim:
            raise ValueError(f"MoE expects embed_dim={self.embed_dim}, got {x.shape[-1]}")

        gate_probs = F.softmax(self.gate(x), dim=-1)
        top_k_weights, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        if self.renormalize_top_k:
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        dispatch_weights = torch.zeros_like(gate_probs).scatter(1, top_k_indices, top_k_weights)
        features = (
            self._sparse_forward(x, top_k_indices, dispatch_weights)
            if self.sparse_dispatch
            else self._dense_forward(x, dispatch_weights)
        )
        return MoEOutput(
            features=features,
            gate_probs=gate_probs,
            top_k_indices=top_k_indices,
            top_k_weights=top_k_weights,
            dispatch_weights=dispatch_weights,
        )

    def _sparse_forward(
        self,
        x: torch.Tensor,
        top_k_indices: torch.Tensor,
        dispatch_weights: torch.Tensor,
    ) -> torch.Tensor:
        selection = torch.zeros_like(dispatch_weights, dtype=torch.bool)
        selection.scatter_(1, top_k_indices, True)

        output = torch.zeros_like(x)
        for expert_index, expert in enumerate(self.experts):
            rows = selection[:, expert_index].nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            expert_output = expert(x.index_select(0, rows).unsqueeze(1)).squeeze(1)
            weights = dispatch_weights.index_select(0, rows)[:, expert_index].unsqueeze(-1)
            output = output.index_add(0, rows, expert_output * weights)
        return output

    def _dense_forward(self, x: torch.Tensor, dispatch_weights: torch.Tensor) -> torch.Tensor:
        token = x.unsqueeze(1)
        expert_outputs = torch.stack([expert(token).squeeze(1) for expert in self.experts], dim=1)
        return torch.sum(expert_outputs * dispatch_weights.unsqueeze(-1), dim=1)

    @torch.no_grad()
    def expert_utilization(self, top_k_indices: torch.Tensor) -> torch.Tensor:
        """Fraction of samples in the batch routed to each expert, shape ``[num_experts]``."""
        counts = torch.zeros(self.num_experts, device=top_k_indices.device)
        counts.scatter_add_(
            0,
            top_k_indices.reshape(-1),
            torch.ones_like(top_k_indices.reshape(-1), dtype=counts.dtype),
        )
        return counts / max(top_k_indices.shape[0], 1)

    # ------------------------------------------------------- efficiency support

    def parameters_per_expert(self) -> int:
        """Parameter count of a single expert.

        All experts share one architecture, so counting the first is exact. This
        is what makes the active-parameter arithmetic in
        :mod:`src.utils.efficiency` a closed form rather than a measurement.
        """
        return sum(parameter.numel() for parameter in self.experts[0].parameters())

    def dormant_parameters(self) -> int:
        """Parameters that a single forward pass never touches.

        ``(E - K)`` experts sit out every sample, so their weights contribute to
        the total parameter count but not to the per-sample compute. Dropping
        from ``K = 4`` to ``K = 2`` doubles this number.
        """
        return (self.num_experts - self.top_k) * self.parameters_per_expert()

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, num_experts={self.num_experts}, "
            f"top_k={self.top_k}, sparse_dispatch={self.sparse_dispatch}"
        )


class DenseExpertBlock(nn.Module):
    """Single always-on transformer block replacing the MoE (``use_moe=False``).

    The ``wo_moe`` ablation asks what sparse routing contributes. Deleting the
    experts outright would also delete a transformer block's worth of capacity,
    so the resulting gap would confound *routing* with *depth*. This keeps one
    dense block -- identical in architecture to a single expert -- so the only
    removed ingredient is the gate.

    It returns a :class:`MoEOutput` describing a one-expert router that always
    selects its single expert with weight 1. Both MoE regularisers evaluate to
    exactly zero on that degenerate gate (the entropy term because a
    one-dimensional distribution has no entropy to spread, the sparsity term
    because no mass falls outside the selection), so no caller needs a special
    case.
    """

    def __init__(
        self,
        embed_dim: int,
        mlp_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = 1
        self.top_k = 1
        self.expert = TransformerExpert(embed_dim, num_heads, mlp_dim, dropout)

    def forward(self, x: torch.Tensor) -> MoEOutput:
        if x.ndim != 2:
            raise ValueError(f"DenseExpertBlock expects [batch, embed_dim] input, got {tuple(x.shape)}")
        if x.shape[-1] != self.embed_dim:
            raise ValueError(f"DenseExpertBlock expects embed_dim={self.embed_dim}, got {x.shape[-1]}")

        features = self.expert(x.unsqueeze(1)).squeeze(1)
        ones = x.new_ones(x.shape[0], 1)
        indices = torch.zeros(x.shape[0], 1, dtype=torch.long, device=x.device)
        return MoEOutput(
            features=features,
            gate_probs=ones,
            top_k_indices=indices,
            top_k_weights=ones,
            dispatch_weights=ones,
        )

    def parameters_per_expert(self) -> int:
        return sum(parameter.numel() for parameter in self.expert.parameters())

    def dormant_parameters(self) -> int:
        """Always zero: a dense block activates everything it owns."""
        return 0

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, dense=True"
