"""Mixture-of-Experts regularisers (paper Section 5.2).

The paper introduces two terms "to prevent expert collapse and encourage
balanced expert utilization":

*Load Balancing Loss*, "implemented through entropy regularization, ensures an
even distribution of expert selection across batches". Averaging the gate
distribution over the batch gives a utilisation vector ``u``; maximising its
entropy makes that vector uniform. Expressed as something to *minimise*::

    L_load = -H(u) / log(E),      u_i = mean_batch(G_i)

The ``log(E)`` normalisation bounds the term in ``[-1, 0]`` (``-1`` = perfectly
uniform, ``0`` = total collapse onto one expert) so a chosen ``lambda`` keeps
its meaning when the expert count changes.

*Sparsity Loss*, "enforced via L1 regularization, restricts the selection to
only the top-K most relevant experts". The gate output is a probability
simplex, so its total L1 norm is identically 1 and penalising that would do
nothing. What the sentence describes is penalising the routing mass that lands
*outside* the Top-K selection::

    L_sparsity = mean_batch( sum_{i not in Top-K} G_i )   in [0, 1]

Driving that to zero concentrates all routing mass on the K selected experts.

The two terms deliberately pull against each other -- one wants each *sample*
routed decisively, the other wants the *batch* spread evenly across experts.
That tension is the point: it yields specialised but fully-utilised experts.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

SPARSITY_MODES = ("off_topk", "topk")


class MoERegularizationOutput(NamedTuple):
    """Weighted total plus the individual, unweighted terms for logging."""

    total: torch.Tensor
    load_balancing: torch.Tensor
    sparsity: torch.Tensor
    utilization: torch.Tensor
    """Batch-averaged gate distribution ``u``, shape ``[num_experts]``."""


def expert_utilization(gate_probs: torch.Tensor) -> torch.Tensor:
    """Batch-averaged gate distribution ``u``, shape ``[num_experts]``, summing to 1."""
    return gate_probs.mean(dim=0)


def load_balancing_loss(gate_probs: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Negative entropy of the batch-averaged gate distribution.

    Minimising this maximises entropy, which spreads utilisation evenly.

    Args:
        gate_probs: Full gate distribution, shape ``[batch, num_experts]``.
        normalize: Divide by ``log(num_experts)`` to bound the result in
            ``[-1, 0]``.
    """
    utilization = expert_utilization(gate_probs)
    negative_entropy = torch.sum(utilization * torch.log(utilization.clamp_min(1e-8)))
    if not normalize:
        return negative_entropy
    num_experts = gate_probs.shape[-1]
    if num_experts <= 1:
        return negative_entropy
    return negative_entropy / torch.log(torch.tensor(float(num_experts), device=gate_probs.device))


def l1_sparsity_loss(
    gate_probs: torch.Tensor,
    top_k_indices: torch.Tensor,
    mode: str = "off_topk",
) -> torch.Tensor:
    """L1 penalty on routing mass, restricting activation to the Top-K experts.

    Args:
        gate_probs: Full gate distribution, shape ``[batch, num_experts]``.
        top_k_indices: Selected expert indices, shape ``[batch, top_k]``.
        mode: ``"off_topk"`` penalises the mass outside the selection (the
            paper's reading, bounded in ``[0, 1]``). ``"topk"`` reproduces this
            repository's earlier behaviour of penalising the selected gate
            weights themselves; kept for ablation only, since it fights the
            load-balancing term rather than complementing it.
    """
    if mode not in SPARSITY_MODES:
        raise ValueError(f"mode must be one of {SPARSITY_MODES}, got {mode!r}")

    selected_mass = torch.gather(gate_probs, 1, top_k_indices).sum(dim=-1)
    if mode == "topk":
        return torch.gather(gate_probs, 1, top_k_indices).abs().mean()
    # Gate probabilities are non-negative, so the L1 norm of the discarded mass
    # is just its sum, and total mass is 1 by construction.
    return (1.0 - selected_mass).clamp_min(0.0).mean()


class MoERegularization(nn.Module):
    """Weighted sum of the load-balancing and sparsity terms.

    Args:
        lambda_load: Weight on the entropy load-balancing term.
        lambda_sparsity: Weight on the L1 sparsity term.
        normalize_entropy: Bound the load-balancing term in ``[-1, 0]``.
        sparsity_mode: See :func:`l1_sparsity_loss`.
    """

    def __init__(
        self,
        lambda_load: float = 0.01,
        lambda_sparsity: float = 0.01,
        normalize_entropy: bool = True,
        sparsity_mode: str = "off_topk",
    ):
        super().__init__()
        if sparsity_mode not in SPARSITY_MODES:
            raise ValueError(f"sparsity_mode must be one of {SPARSITY_MODES}, got {sparsity_mode!r}")
        self.lambda_load = float(lambda_load)
        self.lambda_sparsity = float(lambda_sparsity)
        self.normalize_entropy = bool(normalize_entropy)
        self.sparsity_mode = sparsity_mode

    def forward(
        self,
        gate_probs: torch.Tensor,
        top_k_indices: torch.Tensor,
    ) -> MoERegularizationOutput:
        load = load_balancing_loss(gate_probs, normalize=self.normalize_entropy)
        sparsity = l1_sparsity_loss(gate_probs, top_k_indices, mode=self.sparsity_mode)
        total = self.lambda_load * load + self.lambda_sparsity * sparsity
        return MoERegularizationOutput(
            total=total,
            load_balancing=load,
            sparsity=sparsity,
            utilization=expert_utilization(gate_probs).detach(),
        )

    def extra_repr(self) -> str:
        return (
            f"lambda_load={self.lambda_load}, lambda_sparsity={self.lambda_sparsity}, "
            f"sparsity_mode={self.sparsity_mode}"
        )
