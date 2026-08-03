"""The combined hierarchical objective (paper Section 5).

Total objective, uniting every weighted component the paper describes::

    L = w_seed        * L_seed        (Eq. 7,  categorical cross-entropy)
      + w_arcface     * L_ArcFace     (Eq. 13, angular margin)
      + w_kl          * L_KL          (Eq. 10, hierarchy consistency)
      + lambda_load     * L_load      (Sec 5.2, dispatch-aware load balancing)
      + lambda_sparsity * L_sparsity  (Sec 5.2, L1 Top-K sparsity; 0 by default)
      + lambda_z        * L_z         (router z-loss, ST-MoE)
      + lambda_cosine   * L_cos       (Sec 1,  class compactness)
      + lambda_residual * L_res       (Eq. 9 residual magnitude hinge)
      + lambda_sub_ce   * L_sub_CE    (auxiliary, off by default)

Hierarchy consistency, in log space
-----------------------------------

Eq. 10 is ``D_KL(P_seed || P_sub-variety)``. Those distributions live over
different label sets -- 4 seed types versus 27 sub-varieties -- so the
sub-variety distribution is first aggregated to seed-type granularity.

The submitted implementation aggregated in probability space and then took a
logarithm::

    aggregated = sub_probs @ mapping_matrix
    F.kl_div(torch.log(aggregated.clamp_min(1e-8)), seed_probs, ...)

``clamp_min`` has **zero gradient in the clamped region**, so whenever an
aggregated seed-type probability fell below ``1e-8`` the term contributed
nothing -- silently, with no NaN and no warning. That was not an edge case. With
``sub_logits = s cos(theta)`` at ``s = 30``, a cosine gap of 1.0 between the
argmax sub-variety and a competitor is a logit gap of 30, i.e. a probability
ratio of ``e^30 ~ 1e13``; aggregating 27 such probabilities into 4 bins leaves
the non-argmax bins around ``1e-13``--``1e-10``, comfortably clamped. The term
was therefore live when the two heads agreed (where it has nothing to do) and
dead when they disagreed confidently (which is the entire point of it).

:func:`hierarchical_kl_loss` aggregates with ``logsumexp`` over each parent's
children instead. That is exact rather than a tolerance tweak: ``logsumexp`` is
numerically stable by construction, needs no epsilon, and has a well-conditioned
gradient across the whole probability range.

``tau_kl`` decouples this term from the ArcFace scale. ``P_sub =
softmax(s cos theta)`` is near-one-hot by construction, so ``lambda_kl`` and
``arcface_scale`` were secretly one hyperparameter; dividing by ``tau_kl`` before
the softmax separates them again.

Direction and detachment
------------------------

``detach_kl_seed_target`` defaults to **True**. The coarse head is already
supervised by hard labels through Eq. 7; letting ``L_KL`` also push ``P_seed``
means it can reduce the term by becoming *less* accurate -- agreeing with a
confidently wrong fine prediction. And ``KL(p||q)`` is zero-avoiding in ``q``, so
an uncertain coarse head forces the fine head to hedge across seed types, which
is the opposite of what a 27-way fine-grained task needs.

``kl_mode="jsd"`` offers the symmetric alternative the hierarchy-consistency
literature has converged on (HAF, Garg et al. 2022): bounded by ``log 2``,
symmetric, and without the zero-avoidance.

Loss weighting
--------------

The submitted objective used seven fixed lambdas over terms whose magnitudes at
initialisation differ by ~13x (``L_ArcFace = 17.64`` against ``L_seed = 1.386``
at ``s = 30``), so one term carried ~92 % of the initial gradient. Lowering the
ArcFace scale to the AdaCos value removes most of that spread; ``weighting_mode
= "uncertainty"`` removes the rest by learning the three **task** weights via
Kendall et al.'s homoscedastic formulation::

    L = sum_t ( L_t / (2 sigma_t^2) + log sigma_t / 2 ) + sum_r lambda_r L_r

The three regularisers keep fixed lambdas -- they are genuinely auxiliary and
small. The learned ``sigma_t`` are also diagnostic: if ``sigma_arcface``
collapses while ``sigma_kl`` explodes, the dominance hypothesis is confirmed
empirically rather than argued.

**The weighting mode must be identical across every variant in a suite**, or the
ablation gaps become gaps in loss-weighting policy.

Statefulness
------------

The criterion holds learnable parameters only under
``weighting_mode="uncertainty"`` (three scalars) and buffers only for the EMA
class centroids. The ArcFace class centres stay in the model, so
``model_state_dict`` alone still reproduces the head; the trainer adds the
criterion to the optimizer explicitly and saves ``criterion_state_dict``
alongside.

Ablation switches
-----------------

``use_kl_loss=False`` (the ``wo_kl`` variant) skips Eq. 10 outright rather than
weighting it to zero, so no gradient is computed for it at all.

The sub-variety head variant is chosen in the model, not here: ``ArcFaceHead``,
``NormFaceHead`` and ``LinearSubVarietyHead`` all return
``(logits, margin_logits)``, and ``L_ArcFace`` is a cross-entropy over
``margin_logits``, so a head with no margin degrades it to plain CE with no
loss-side branch. Keeping one code path here means an ablation cannot drift from
the full model by way of a second loss implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.arcface import arcface_loss
from src.losses.cosine import CosineSimilarityLoss, residual_magnitude_loss
from src.losses.moe import MoERegularization
from src.models.builder import HierarchicalOutput

KL_MODES = ("forward", "jsd")
WEIGHTING_MODES = ("fixed", "uncertainty")

#: Task terms whose weights uncertainty weighting learns. The regularisers keep
#: fixed lambdas on purpose -- they are auxiliary, small, and their scale is
#: meaningful (``L_load`` has a zero floor by construction).
TASK_TERMS = ("seed", "arcface", "kl")


class LossBreakdown(NamedTuple):
    """Weighted total plus every unweighted component, for logging."""

    total: torch.Tensor
    seed: torch.Tensor
    arcface: torch.Tensor
    sub_ce: torch.Tensor
    kl: torch.Tensor
    moe_load: torch.Tensor
    moe_sparsity: torch.Tensor
    moe_z: torch.Tensor
    cosine: torch.Tensor
    residual: torch.Tensor
    dead_experts: int = 0
    task_weights: dict[str, float] | None = None

    def as_dict(self) -> dict[str, float]:
        """Detached Python floats keyed for the experiment tracker."""
        payload = {
            "total_loss": float(self.total.detach()),
            "seed_type_loss": float(self.seed.detach()),
            "arcface_loss": float(self.arcface.detach()),
            "sub_variety_ce_loss": float(self.sub_ce.detach()),
            "kl_loss": float(self.kl.detach()),
            "moe_load_balancing_loss": float(self.moe_load.detach()),
            "moe_sparsity_loss": float(self.moe_sparsity.detach()),
            "moe_router_z_loss": float(self.moe_z.detach()),
            "cosine_loss": float(self.cosine.detach()),
            "residual_magnitude_loss": float(self.residual.detach()),
            "moe_dead_experts": float(self.dead_experts),
        }
        for name, value in (self.task_weights or {}).items():
            payload[f"task_weight/{name}"] = float(value)
        return payload


def build_subvariety_seed_mapping(
    num_sub_varieties: int,
    num_seed_types: int,
    subvariety_to_seed_type: Sequence[int] | None = None,
    subvarieties_per_seed_type: Sequence[int] | None = None,
) -> torch.Tensor:
    """Build the one-hot ``[num_sub_varieties, num_seed_types]`` mapping ``M``.

    Supply either an explicit per-sub-variety parent list (preferred; this is
    what the dataset produces) or the count of sub-varieties per seed type,
    which assumes sub-variety indices are contiguous within each seed type.

    The log-space aggregation reads this as a boolean children mask; the one-hot
    float form is kept so the buffer, the checkpoint layout and every existing
    consumer stay unchanged.
    """
    if subvariety_to_seed_type is not None:
        if len(subvariety_to_seed_type) != num_sub_varieties:
            raise ValueError(
                f"subvariety_to_seed_type has {len(subvariety_to_seed_type)} entries, "
                f"expected num_sub_varieties={num_sub_varieties}"
            )
        mapping = torch.zeros(num_sub_varieties, num_seed_types)
        for sub_index, seed_index in enumerate(subvariety_to_seed_type):
            seed_index = int(seed_index)
            if not 0 <= seed_index < num_seed_types:
                raise ValueError(
                    f"sub-variety {sub_index} maps to seed type {seed_index}, "
                    f"outside [0, {num_seed_types})"
                )
            mapping[sub_index, seed_index] = 1.0
        return mapping

    if subvarieties_per_seed_type is None:
        raise ValueError("Provide subvariety_to_seed_type or subvarieties_per_seed_type")
    if len(subvarieties_per_seed_type) != num_seed_types:
        raise ValueError("subvarieties_per_seed_type length must equal num_seed_types")
    if sum(subvarieties_per_seed_type) != num_sub_varieties:
        raise ValueError("subvarieties_per_seed_type must sum to num_sub_varieties")

    mapping = torch.zeros(num_sub_varieties, num_seed_types)
    start = 0
    for seed_index, count in enumerate(subvarieties_per_seed_type):
        end = start + int(count)
        mapping[start:end, seed_index] = 1.0
        start = end
    return mapping


def seed_type_loss(
    seed_type_logits: torch.Tensor,
    labels: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Categorical cross-entropy over the four seed types (Eq. 7)."""
    return F.cross_entropy(seed_type_logits, labels, label_smoothing=label_smoothing)


def aggregate_sub_log_probs(
    sub_variety_logits: torch.Tensor,
    mapping_matrix: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """Exact log ``P(seed type)`` marginalised from the sub-variety distribution.

        log P_agg[b, c] = logsumexp_{j in children(c)} log P_sub[b, j]

    Computed with ``logsumexp`` over a ``-inf``-masked log-probability tensor, so
    it is stable at any probability and differentiable everywhere -- unlike
    aggregating in probability space and clamping before the log.

    Args:
        sub_variety_logits: Un-margined logits, ``[batch, num_sub_varieties]``.
        mapping_matrix: ``M``, ``[num_sub_varieties, num_seed_types]``, one-hot.
        tau: Temperature applied before the softmax. Decouples this branch from
            the ArcFace feature scale.
    """
    log_p_sub = F.log_softmax(sub_variety_logits / float(tau), dim=-1)  # [B, S]
    children = mapping_matrix.to(log_p_sub.device).t().bool()  # [C, S]
    masked = log_p_sub.unsqueeze(1).masked_fill(~children.unsqueeze(0), float("-inf"))
    return torch.logsumexp(masked, dim=-1)  # [B, C]


def hierarchical_kl_loss(
    seed_type_logits: torch.Tensor,
    sub_variety_logits: torch.Tensor,
    mapping_matrix: torch.Tensor,
    detach_seed_target: bool = True,
    tau: float = 1.0,
    mode: str = "forward",
) -> torch.Tensor:
    """Hierarchy consistency between the coarse head and the marginalised fine head.

    Args:
        seed_type_logits: ``s``, shape ``[batch, num_seed_types]``.
        sub_variety_logits: Un-margined logits, ``[batch, num_sub_varieties]``.
            Margin logits must **not** be used here: the margin is a training
            device for the ArcFace term, not part of the predicted distribution.
        mapping_matrix: ``M``, shape ``[num_sub_varieties, num_seed_types]``.
        detach_seed_target: Treat the seed-type distribution as a fixed target,
            so the KL gradient only reshapes the sub-variety branch. Default
            ``True``; see the module docstring.
        tau: Temperature for the sub-variety branch.
        mode: ``"forward"`` for ``D_KL(P_seed || P_sub-agg)`` as Eq. 10 writes it,
            or ``"jsd"`` for the symmetric Jensen-Shannon divergence.

    ``F.kl_div(input=log q, target=p)`` computes ``KL(p || q)``, so passing the
    aggregated sub-variety log-probabilities as ``input`` and the seed-type
    probabilities as ``target`` gives the direction Eq. 10 asks for.
    """
    if mode not in KL_MODES:
        raise ValueError(f"mode must be one of {KL_MODES}, got {mode!r}")

    log_p_agg = aggregate_sub_log_probs(sub_variety_logits, mapping_matrix, tau=tau)
    log_p_seed = F.log_softmax(seed_type_logits, dim=-1)
    if detach_seed_target:
        log_p_seed = log_p_seed.detach()

    if mode == "forward":
        return F.kl_div(log_p_agg, log_p_seed, reduction="batchmean", log_target=True)

    # Jensen-Shannon: both branches move toward their mixture, and the term is
    # bounded by log 2 so a confident disagreement cannot dominate the objective.
    log_mixture = torch.logsumexp(
        torch.stack([log_p_seed, log_p_agg], dim=0), dim=0
    ) - torch.log(torch.tensor(2.0, device=log_p_agg.device))
    forward = F.kl_div(log_mixture, log_p_seed, reduction="batchmean", log_target=True)
    reverse = F.kl_div(log_mixture, log_p_agg, reduction="batchmean", log_target=True)
    return 0.5 * (forward + reverse)


class UncertaintyWeighting(nn.Module):
    """Homoscedastic task weighting (Kendall et al., 2018).

    Optimises ``log sigma^2`` per task directly, which keeps the weights positive
    without a constraint and makes them readable as a diagnostic. Clamped for
    stability: an unbounded ``log sigma^2`` can run away and silently switch a
    task off.
    """

    def __init__(self, terms: Sequence[str] = TASK_TERMS, clamp: float = 5.0):
        super().__init__()
        self.terms = tuple(terms)
        self.clamp = float(clamp)
        self.log_variance = nn.Parameter(torch.zeros(len(self.terms)))

    def weights(self) -> dict[str, float]:
        """Current ``1 / (2 sigma^2)`` per task, for logging."""
        with torch.no_grad():
            values = torch.exp(-self.log_variance.clamp(-self.clamp, self.clamp)) * 0.5
        return {name: float(value) for name, value in zip(self.terms, values)}

    def forward(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        total = None
        for index, name in enumerate(self.terms):
            if name not in losses:
                continue
            log_variance = self.log_variance[index].clamp(-self.clamp, self.clamp)
            term = torch.exp(-log_variance) * losses[name] + 0.5 * log_variance
            total = term if total is None else total + term
        if total is None:
            raise ValueError(f"No known task terms in {sorted(losses)}; expected any of {self.terms}")
        return total


class CombinedHierarchicalLoss(nn.Module):
    """Weighted sum of every loss component described in the paper.

    Args:
        num_seed_types: Coarse classes (paper: 4).
        num_sub_varieties: Fine classes (paper: 27).
        subvariety_to_seed_type: Parent seed type of each sub-variety index.
        subvarieties_per_seed_type: Alternative to the above; see
            :func:`build_subvariety_seed_mapping`.
        embed_dim: Embedding width, needed to size the EMA class centroids.
        lambda_seed / lambda_arcface / lambda_kl: Task weights, used when
            ``weighting_mode="fixed"``.
        lambda_moe_load / lambda_moe_sparsity / lambda_moe_z: MoE regulariser
            weights.
        lambda_cosine: Weight on the class-compactness term.
        lambda_residual: Weight on the Eq. 9 residual magnitude hinge.
        lambda_sub_ce: Weight on the auxiliary plain cross-entropy (default 0).
        seed_label_smoothing / arcface_label_smoothing: Label smoothing.
        detach_kl_seed_target: See :func:`hierarchical_kl_loss`.
        kl_mode: ``"forward"`` or ``"jsd"``.
        tau_kl: Temperature decoupling the KL branch from the ArcFace scale.
        moe_sparsity_mode / moe_load_mode / normalize_moe_entropy: See
            :mod:`src.losses.moe`.
        moe_utilization_momentum: EMA factor for the routing statistics.
        num_experts: Expert count, for the EMA buffers.
        cosine_mode: ``"intra_class"`` (default) or ``"residual"``.
        centroid_momentum: EMA factor for the class centroids.
        residual_tau: Hinge threshold on ``||P(p_s)|| / ||h||``.
        weighting_mode: ``"fixed"`` or ``"uncertainty"``.
        use_kl_loss: The ``wo_kl`` ablation switch. ``False`` skips Eq. 10
            entirely, so no gradient flows through it and the reported ``kl``
            component is a constant zero rather than an unweighted measurement.
    """

    def __init__(
        self,
        num_seed_types: int,
        num_sub_varieties: int,
        subvariety_to_seed_type: Sequence[int] | None = None,
        subvarieties_per_seed_type: Sequence[int] | None = None,
        embed_dim: int = 384,
        lambda_seed: float = 1.0,
        lambda_arcface: float = 1.0,
        lambda_kl: float = 1.0,
        lambda_moe_load: float = 0.01,
        lambda_moe_sparsity: float = 0.0,
        lambda_moe_z: float = 1e-3,
        lambda_cosine: float = 0.1,
        lambda_residual: float = 0.01,
        lambda_sub_ce: float = 0.0,
        seed_label_smoothing: float = 0.0,
        arcface_label_smoothing: float = 0.0,
        detach_kl_seed_target: bool = True,
        kl_mode: str = "forward",
        tau_kl: float = 1.0,
        moe_sparsity_mode: str = "off_topk",
        moe_load_mode: str = "switch",
        normalize_moe_entropy: bool = True,
        moe_utilization_momentum: float = 0.9,
        num_experts: int = 6,
        cosine_mode: str = "intra_class",
        centroid_momentum: float = 0.9,
        residual_tau: float = 0.5,
        weighting_mode: str = "fixed",
        use_kl_loss: bool = True,
    ):
        super().__init__()
        if kl_mode not in KL_MODES:
            raise ValueError(f"kl_mode must be one of {KL_MODES}, got {kl_mode!r}")
        if weighting_mode not in WEIGHTING_MODES:
            raise ValueError(f"weighting_mode must be one of {WEIGHTING_MODES}, got {weighting_mode!r}")

        mapping = build_subvariety_seed_mapping(
            num_sub_varieties=num_sub_varieties,
            num_seed_types=num_seed_types,
            subvariety_to_seed_type=subvariety_to_seed_type,
            subvarieties_per_seed_type=subvarieties_per_seed_type,
        )
        self.register_buffer("mapping_matrix", mapping)

        self.moe_regularization = MoERegularization(
            lambda_load=lambda_moe_load,
            lambda_sparsity=lambda_moe_sparsity,
            lambda_z=lambda_moe_z,
            normalize_entropy=normalize_moe_entropy,
            sparsity_mode=moe_sparsity_mode,
            load_mode=moe_load_mode,
            utilization_momentum=moe_utilization_momentum,
            num_experts=num_experts,
        )
        self.cosine_loss = CosineSimilarityLoss(
            mode=cosine_mode,
            num_classes=num_sub_varieties,
            embed_dim=embed_dim,
            centroid_momentum=centroid_momentum,
        )

        self.use_kl_loss = bool(use_kl_loss)
        self.weighting_mode = weighting_mode
        self.uncertainty = (
            UncertaintyWeighting(TASK_TERMS) if weighting_mode == "uncertainty" else None
        )

        self.lambda_seed = float(lambda_seed)
        self.lambda_arcface = float(lambda_arcface)
        self.lambda_kl = float(lambda_kl) if self.use_kl_loss else 0.0
        self.lambda_cosine = float(lambda_cosine)
        self.lambda_residual = float(lambda_residual)
        self.lambda_sub_ce = float(lambda_sub_ce)
        self.seed_label_smoothing = float(seed_label_smoothing)
        self.arcface_label_smoothing = float(arcface_label_smoothing)
        self.detach_kl_seed_target = bool(detach_kl_seed_target)
        self.kl_mode = kl_mode
        self.tau_kl = float(tau_kl)
        self.residual_tau = float(residual_tau)

    # ------------------------------------------------------------------- flags

    def loss_flags(self) -> dict[str, Any]:
        """Every loss-side setting a variant can move, for ``summary.json``.

        Without this, a ``wo_kl`` run's machine-readable trace was byte-identical
        to ``full_model``'s -- only the variant *name* distinguished them.
        """
        return {
            "use_kl_loss": self.use_kl_loss,
            "kl_mode": self.kl_mode,
            "tau_kl": self.tau_kl,
            "detach_kl_seed_target": self.detach_kl_seed_target,
            "weighting_mode": self.weighting_mode,
            "cosine_mode": self.cosine_loss.mode,
            "moe_load_mode": self.moe_regularization.load_mode,
            "moe_sparsity_mode": self.moe_regularization.sparsity_mode,
            "lambda_seed": self.lambda_seed,
            "lambda_arcface": self.lambda_arcface,
            "lambda_kl": self.lambda_kl,
            "lambda_moe_load": self.moe_regularization.lambda_load,
            "lambda_moe_sparsity": self.moe_regularization.lambda_sparsity,
            "lambda_moe_z": self.moe_regularization.lambda_z,
            "lambda_cosine": self.lambda_cosine,
            "lambda_residual": self.lambda_residual,
            "lambda_sub_ce": self.lambda_sub_ce,
        }

    # ----------------------------------------------------------------- forward

    def component_losses(
        self,
        output: HierarchicalOutput,
        seed_type_labels: torch.Tensor,
        sub_variety_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Every unweighted term, keyed by name.

        Exposed separately from :meth:`forward` so the trainer's gradient
        telemetry can backprop one term at a time through the same code the
        objective uses -- a second implementation would measure a different
        thing.
        """
        seed = seed_type_loss(
            output.seed_type_logits,
            seed_type_labels,
            label_smoothing=self.seed_label_smoothing,
        )
        arcface = arcface_loss(
            output.sub_margin_logits,
            sub_variety_labels,
            label_smoothing=self.arcface_label_smoothing,
        )
        sub_ce = (
            F.cross_entropy(output.sub_logits, sub_variety_labels)
            if self.lambda_sub_ce != 0.0
            else output.sub_logits.new_zeros(())
        )
        kl = (
            hierarchical_kl_loss(
                output.seed_type_logits,
                output.sub_logits,
                self.mapping_matrix,
                detach_seed_target=self.detach_kl_seed_target,
                tau=self.tau_kl,
                mode=self.kl_mode,
            )
            if self.use_kl_loss
            else output.sub_logits.new_zeros(())
        )

        if self.cosine_loss.mode == "residual":
            cosine = self.cosine_loss(output.refined_features, output.moe_features)
        else:
            # Class compactness is measured on the ArcFace embedding.
            cosine = self.cosine_loss(output.sub_embeddings, output.sub_embeddings, sub_variety_labels)

        residual = (
            residual_magnitude_loss(output.projected_seed, output.moe_features, tau=self.residual_tau)
            if self.lambda_residual != 0.0
            else output.moe_features.new_zeros(())
        )
        return {
            "seed": seed,
            "arcface": arcface,
            "sub_ce": sub_ce,
            "kl": kl,
            "cosine": cosine,
            "residual": residual,
        }

    def forward(
        self,
        output: HierarchicalOutput,
        seed_type_labels: torch.Tensor,
        sub_variety_labels: torch.Tensor,
    ) -> LossBreakdown:
        """Compute the combined objective from one forward pass.

        Args:
            output: The model's :class:`~src.models.builder.HierarchicalOutput`.
            seed_type_labels: Coarse targets, shape ``[batch]``.
            sub_variety_labels: Fine targets, shape ``[batch]``.
        """
        parts = self.component_losses(output, seed_type_labels, sub_variety_labels)
        moe = self.moe_regularization(
            output.gate_probs,
            output.top_k_indices,
            gate_logits=output.gate_logits,
        )

        if self.uncertainty is not None:
            task_total = self.uncertainty(
                {name: parts[name] for name in TASK_TERMS if name in parts}
            )
            task_weights = self.uncertainty.weights()
        else:
            task_total = (
                self.lambda_seed * parts["seed"]
                + self.lambda_arcface * parts["arcface"]
                + self.lambda_kl * parts["kl"]
            )
            task_weights = {
                "seed": self.lambda_seed,
                "arcface": self.lambda_arcface,
                "kl": self.lambda_kl,
            }

        total = (
            task_total
            + self.lambda_sub_ce * parts["sub_ce"]
            + moe.total
            + self.lambda_cosine * parts["cosine"]
            + self.lambda_residual * parts["residual"]
        )
        return LossBreakdown(
            total=total,
            seed=parts["seed"],
            arcface=parts["arcface"],
            sub_ce=parts["sub_ce"],
            kl=parts["kl"],
            moe_load=moe.load_balancing,
            moe_sparsity=moe.sparsity,
            moe_z=moe.z_loss,
            cosine=parts["cosine"],
            residual=parts["residual"],
            dead_experts=moe.dead_experts,
            task_weights=task_weights,
        )

    def extra_repr(self) -> str:
        return (
            f"weighting_mode={self.weighting_mode}, kl_mode={self.kl_mode}, "
            f"tau_kl={self.tau_kl}, detach_kl_seed_target={self.detach_kl_seed_target}, "
            f"use_kl_loss={self.use_kl_loss}"
        )


def build_combined_loss(
    loss_cfg,
    num_seed_types: int,
    num_sub_varieties: int,
    subvariety_to_seed_type: Sequence[int] | None = None,
    embed_dim: int = 384,
    num_experts: int = 6,
) -> CombinedHierarchicalLoss:
    """Instantiate :class:`CombinedHierarchicalLoss` from a loss config node."""

    def get(key: str, default):
        value = getattr(loss_cfg, key, default)
        return default if value is None else value

    return CombinedHierarchicalLoss(
        num_seed_types=num_seed_types,
        num_sub_varieties=num_sub_varieties,
        subvariety_to_seed_type=subvariety_to_seed_type,
        embed_dim=int(embed_dim),
        lambda_seed=float(get("lambda_seed", 1.0)),
        lambda_arcface=float(get("lambda_arcface", 1.0)),
        lambda_kl=float(get("lambda_kl", 1.0)),
        lambda_moe_load=float(get("lambda_moe_load", 0.01)),
        lambda_moe_sparsity=float(get("lambda_moe_sparsity", 0.0)),
        lambda_moe_z=float(get("lambda_moe_z", 1e-3)),
        lambda_cosine=float(get("lambda_cosine", 0.1)),
        lambda_residual=float(get("lambda_residual", 0.01)),
        lambda_sub_ce=float(get("lambda_sub_ce", 0.0)),
        seed_label_smoothing=float(get("seed_label_smoothing", 0.0)),
        arcface_label_smoothing=float(get("arcface_label_smoothing", 0.0)),
        detach_kl_seed_target=bool(get("detach_kl_seed_target", True)),
        kl_mode=str(get("kl_mode", "forward")),
        tau_kl=float(get("tau_kl", 1.0)),
        moe_sparsity_mode=str(get("moe_sparsity_mode", "off_topk")),
        moe_load_mode=str(get("moe_load_mode", "switch")),
        normalize_moe_entropy=bool(get("normalize_moe_entropy", True)),
        moe_utilization_momentum=float(get("moe_utilization_momentum", 0.9)),
        num_experts=int(num_experts),
        cosine_mode=str(get("cosine_mode", "intra_class")),
        centroid_momentum=float(get("centroid_momentum", 0.9)),
        residual_tau=float(get("residual_tau", 0.5)),
        weighting_mode=str(get("weighting_mode", "fixed")),
        use_kl_loss=bool(get("use_kl_loss", True)),
    )
