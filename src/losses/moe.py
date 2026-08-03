"""Mixture-of-Experts regularisers (paper Section 5.2).

The paper introduces two terms "to prevent expert collapse and encourage
balanced expert utilization". The revision keeps that goal and changes the
mathematics, because the submitted formulation could not reach it.

Load balancing
--------------

The submitted form (``load_mode="entropy"``) is the negative entropy of the
batch-mean **soft** gate::

    L_load = sum_i u_i log u_i / log E,     u_i = mean_batch(G_i)      in [-1, 0]

The model's behaviour, however, is decided by ``topk(G, K)``, which ``u`` never
observes. The two come apart badly. A router emitting the same
``G = (0.30, 0.30, 0.10, 0.10, 0.10, 0.10)`` for every sample scores::

    -sum u_i ln u_i = 1.64342,   ln 6 = 1.79176,   L_load = -0.9172

i.e. 92 % of the way to "perfect balance" -- while Top-2 sends *every* sample to
experts {0, 1} and the other four receive no gradient for the entire run. Worse,
at the entropy term's global optimum ``G = (1/6, ..., 1/6)`` exactly, and
``torch.topk`` breaks ties toward the lowest indices, so the **global minimum of
the load-balancing loss produces maximally imbalanced hard routing**.
:func:`switch_load_balancing_loss` is tested against exactly this counterexample.

The revision default (``load_mode="switch"``) is the Shazeer / GShard / Switch
auxiliary loss, which couples the hard dispatch fraction to the differentiable
router probability::

    L_load = E * sum_i f_i * P_i,
        f_i = (1/T) sum_x 1[i in TopK(x)]      (hard, no gradient)
        P_i = (1/T) sum_x G_i(x)               (soft, carries the gradient)

``f`` acts as a per-expert coefficient on ``P``, so an *over-dispatched* expert
gets its router probability pushed down. The minimum is ``1`` at uniform
routing; the reported value is ``L_load - 1`` so a balanced router still scores
``0`` and ``lambda_load`` keeps its meaning across expert counts.

Router z-loss
-------------

``L_z = mean_x (logsumexp_i x_i)^2`` over the pre-softmax router logits (Zoph et
al., ST-MoE 2022) at the standard ``beta = 1e-3``. It prevents router logit
growth, which is the failure mode that makes Top-K selection brittle.

Sparsity
--------

``L_sparsity = mean_batch(sum_{i not in TopK} G_i)`` penalises routing mass that
lands outside the selection. **Under ``renormalize_top_k=True`` this term cannot
change the model's function**: the renormalised weights are invariant to any
rescaling of ``G`` restricted to the Top-K set, so there is a descent direction
along which the output is unchanged. Its only reliable effect is to reduce
router entropy -- which directly fights the load-balancing term. The two are
mutually redundant; ``lambda_moe_sparsity`` therefore defaults to ``0.0`` and
the term is retained only as an ablation axis.

Estimator noise
---------------

``f`` and ``P`` are estimated from one batch. At ``batch_size=16`` and ``K=2``
that is 32 routing slots over 6 experts -- far too noisy to steer a
self-reinforcing process. ``utilization_momentum`` EMA-smooths both statistics
across steps. Under ``token_mode="grid"`` each image contributes 64 routing
tokens instead of 1, which fixes the same problem at the source.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

SPARSITY_MODES = ("off_topk", "topk")
LOAD_MODES = ("switch", "entropy")


class MoERegularizationOutput(NamedTuple):
    """Weighted total plus the individual, unweighted terms for logging."""

    total: torch.Tensor
    load_balancing: torch.Tensor
    sparsity: torch.Tensor
    z_loss: torch.Tensor
    utilization: torch.Tensor
    """Batch-averaged **soft** gate distribution ``P``, shape ``[num_experts]``."""

    hard_utilization: torch.Tensor
    """**Hard** dispatch fraction ``f``, shape ``[num_experts]``. This is what the
    utilisation figure draws and what "balanced" actually means."""

    dead_experts: int
    """``#{i : f_i = 0}`` for this batch. Logged every epoch as a first-class
    metric, because the load term alone can look healthy while this is 4."""


def expert_utilization(gate_probs: torch.Tensor) -> torch.Tensor:
    """Batch-averaged soft gate distribution ``P``, shape ``[num_experts]``."""
    return gate_probs.mean(dim=0)


def dispatch_fraction(top_k_indices: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Hard dispatch fraction ``f``, shape ``[num_experts]``, summing to 1.

    Non-differentiable by construction: ``top_k_indices`` comes from ``topk``,
    which has zero gradient almost everywhere. In the Switch loss ``f`` is a
    coefficient and ``P`` carries the gradient.
    """
    flat = top_k_indices.reshape(-1)
    counts = torch.zeros(num_experts, device=top_k_indices.device, dtype=torch.float32)
    counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=counts.dtype))
    return counts / counts.sum().clamp_min(1.0)


def switch_load_balancing_loss(
    gate_probs: torch.Tensor,
    top_k_indices: torch.Tensor,
    zero_floor: bool = True,
) -> torch.Tensor:
    """``E * sum_i f_i * P_i`` (Shazeer 2017; GShard; Switch Transformer).

    Args:
        gate_probs: Full gate distribution, shape ``[tokens, num_experts]``.
        top_k_indices: Selected expert indices, shape ``[tokens, top_k]``.
        zero_floor: Subtract the uniform-routing minimum of ``1`` so a balanced
            router scores ``0`` rather than ``1``, keeping ``lambda`` comparable
            with the entropy form and across expert counts.
    """
    num_experts = gate_probs.shape[-1]
    soft = expert_utilization(gate_probs)
    hard = dispatch_fraction(top_k_indices, num_experts).to(soft.dtype).detach()
    loss = num_experts * torch.sum(hard * soft)
    return loss - 1.0 if zero_floor else loss


def entropy_load_balancing_loss(gate_probs: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Negative entropy of the batch-averaged soft gate (the submitted form).

    Retained so ``L_load(entropy)`` and ``L_load(switch)`` can be compared on the
    same split. It is **not** the default: see the module docstring for the
    counterexample where this scores -0.917 while four of six experts are dead.
    """
    utilization = expert_utilization(gate_probs)
    negative_entropy = torch.sum(utilization * torch.log(utilization.clamp_min(1e-8)))
    if not normalize:
        return negative_entropy
    num_experts = gate_probs.shape[-1]
    if num_experts <= 1:
        return negative_entropy
    return negative_entropy / torch.log(torch.tensor(float(num_experts), device=gate_probs.device))


def load_balancing_loss(
    gate_probs: torch.Tensor,
    top_k_indices: torch.Tensor | None = None,
    mode: str = "switch",
    normalize: bool = True,
) -> torch.Tensor:
    """Dispatch-aware (``switch``) or soft-only (``entropy``) load balancing."""
    if mode not in LOAD_MODES:
        raise ValueError(f"mode must be one of {LOAD_MODES}, got {mode!r}")
    if mode == "entropy":
        return entropy_load_balancing_loss(gate_probs, normalize=normalize)
    if top_k_indices is None:
        raise ValueError("load_mode='switch' requires top_k_indices; it is the hard dispatch it balances")
    return switch_load_balancing_loss(gate_probs, top_k_indices, zero_floor=normalize)


def router_z_loss(gate_logits: torch.Tensor) -> torch.Tensor:
    """``mean_x (logsumexp_i x_i)^2`` over pre-softmax router logits (ST-MoE).

    Must be given *logits*: applied to probabilities it is a constant.
    """
    return torch.logsumexp(gate_logits, dim=-1).pow(2).mean()


def l1_sparsity_loss(
    gate_probs: torch.Tensor,
    top_k_indices: torch.Tensor,
    mode: str = "off_topk",
) -> torch.Tensor:
    """L1 penalty on routing mass outside the Top-K selection.

    See the module docstring: with ``renormalize_top_k=True`` this term has a
    null space with respect to the module's output, so it is off by default.

    Args:
        gate_probs: Full gate distribution, shape ``[tokens, num_experts]``.
        top_k_indices: Selected expert indices, shape ``[tokens, top_k]``.
        mode: ``"off_topk"`` penalises the mass outside the selection (the
            paper's reading, bounded in ``[0, 1]``). ``"topk"`` reproduces this
            repository's earlier behaviour of penalising the selected gate
            weights themselves; kept for ablation only.
    """
    if mode not in SPARSITY_MODES:
        raise ValueError(f"mode must be one of {SPARSITY_MODES}, got {mode!r}")

    if mode == "topk":
        return torch.gather(gate_probs, 1, top_k_indices).abs().mean()
    # Gate probabilities are non-negative, so the L1 norm of the discarded mass
    # is just its sum, and total mass is 1 by construction.
    selected_mass = torch.gather(gate_probs, 1, top_k_indices).sum(dim=-1)
    return (1.0 - selected_mass).clamp_min(0.0).mean()


class MoERegularization(nn.Module):
    """Weighted sum of the load-balancing, sparsity and router z-loss terms.

    Args:
        lambda_load: Weight on the load-balancing term.
        lambda_sparsity: Weight on the L1 sparsity term (0 by default; see the
            module docstring).
        lambda_z: Weight on the router z-loss (ST-MoE standard: 1e-3).
        normalize_entropy: Bound the entropy form in ``[-1, 0]``, and give the
            switch form a zero floor at uniform routing.
        sparsity_mode: See :func:`l1_sparsity_loss`.
        load_mode: ``"switch"`` (dispatch-aware, default) or ``"entropy"``.
        utilization_momentum: EMA factor for the ``f`` and ``P`` statistics fed
            to the load term. ``0`` uses the raw per-batch estimate. The buffers
            are registered non-persistently, so they move with ``.to(device)``
            and are never handed to the optimizer.
        num_experts: Expert count, needed to size the EMA buffers up front.
    """

    def __init__(
        self,
        lambda_load: float = 0.01,
        lambda_sparsity: float = 0.0,
        lambda_z: float = 1e-3,
        normalize_entropy: bool = True,
        sparsity_mode: str = "off_topk",
        load_mode: str = "switch",
        utilization_momentum: float = 0.9,
        num_experts: int = 6,
    ):
        super().__init__()
        if sparsity_mode not in SPARSITY_MODES:
            raise ValueError(f"sparsity_mode must be one of {SPARSITY_MODES}, got {sparsity_mode!r}")
        if load_mode not in LOAD_MODES:
            raise ValueError(f"load_mode must be one of {LOAD_MODES}, got {load_mode!r}")

        self.lambda_load = float(lambda_load)
        self.lambda_sparsity = float(lambda_sparsity)
        self.lambda_z = float(lambda_z)
        self.normalize_entropy = bool(normalize_entropy)
        self.sparsity_mode = sparsity_mode
        self.load_mode = load_mode
        self.utilization_momentum = float(utilization_momentum)

        uniform = torch.full((int(num_experts),), 1.0 / max(int(num_experts), 1))
        self.register_buffer("soft_utilization_ema", uniform.clone(), persistent=False)
        self.register_buffer("hard_utilization_ema", uniform.clone(), persistent=False)

    def _smoothed(self, soft: torch.Tensor, hard: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Blend this batch's statistics into the EMA and return the result.

        The gradient path stays on the *current* batch's ``P``; the EMA history
        contributes only its (detached) value. That keeps the term a function of
        the parameters being optimised while smoothing the estimate.
        """
        momentum = self.utilization_momentum
        if momentum <= 0.0 or soft.shape != self.soft_utilization_ema.shape:
            return soft, hard

        with torch.no_grad():
            self.soft_utilization_ema.mul_(momentum).add_(soft.detach(), alpha=1.0 - momentum)
            self.hard_utilization_ema.mul_(momentum).add_(hard, alpha=1.0 - momentum)

        smoothed_soft = momentum * self.soft_utilization_ema.detach() + (1.0 - momentum) * soft
        return smoothed_soft, self.hard_utilization_ema.detach().clone()

    def forward(
        self,
        gate_probs: torch.Tensor,
        top_k_indices: torch.Tensor,
        gate_logits: torch.Tensor | None = None,
    ) -> MoERegularizationOutput:
        num_experts = gate_probs.shape[-1]
        soft = expert_utilization(gate_probs)
        hard = dispatch_fraction(top_k_indices, num_experts).to(soft.dtype)
        smoothed_soft, smoothed_hard = self._smoothed(soft, hard)

        if self.load_mode == "switch":
            load = num_experts * torch.sum(smoothed_hard * smoothed_soft)
            if self.normalize_entropy:
                load = load - 1.0
        else:
            negative_entropy = torch.sum(smoothed_soft * torch.log(smoothed_soft.clamp_min(1e-8)))
            load = (
                negative_entropy / torch.log(torch.tensor(float(num_experts), device=gate_probs.device))
                if self.normalize_entropy and num_experts > 1
                else negative_entropy
            )

        sparsity = l1_sparsity_loss(gate_probs, top_k_indices, mode=self.sparsity_mode)
        z_loss = (
            router_z_loss(gate_logits)
            if gate_logits is not None and self.lambda_z != 0.0
            else gate_probs.new_zeros(())
        )

        total = (
            self.lambda_load * load
            + self.lambda_sparsity * sparsity
            + self.lambda_z * z_loss
        )
        return MoERegularizationOutput(
            total=total,
            load_balancing=load,
            sparsity=sparsity,
            z_loss=z_loss,
            utilization=soft.detach(),
            hard_utilization=hard.detach(),
            dead_experts=int((hard == 0).sum()),
        )

    def extra_repr(self) -> str:
        return (
            f"load_mode={self.load_mode}, lambda_load={self.lambda_load}, "
            f"lambda_sparsity={self.lambda_sparsity}, lambda_z={self.lambda_z}, "
            f"sparsity_mode={self.sparsity_mode}"
        )
