"""The combined hierarchical objective (paper Section 5).

Total objective, uniting every weighted component the paper describes::

    L = lambda_seed     * L_seed        (Eq. 7,  categorical cross-entropy)
      + lambda_arcface  * L_ArcFace     (Eq. 13, angular margin)
      + lambda_kl       * L_KL          (Eq. 10, hierarchy consistency)
      + lambda_load     * L_load        (Sec 5.2, entropy load balancing)
      + lambda_sparsity * L_sparsity    (Sec 5.2, L1 Top-K sparsity)
      + lambda_cosine   * L_cos         (Sec 1,  residual compactness)
      + lambda_sub_ce   * L_sub_CE      (auxiliary, off by default)

Hierarchy consistency (Eq. 10) is ``D_KL(P_seed || P_sub-variety)``. Those two
distributions live over different label sets -- 4 seed types versus 27
sub-varieties -- so the sub-variety distribution is first aggregated to seed-type
granularity through a fixed ``[num_sub_varieties, num_seed_types]`` one-hot
mapping ``M``::

    P_sub_agg = P_sub @ M
    L_KL      = D_KL(P_seed || P_sub_agg)

``M`` comes from ``HierarchicalSeedDataset.get_subvariety_to_seed_type()``, i.e.
it is derived from the directory layout at runtime rather than hardcoded.

``L_sub_CE`` is a plain cross-entropy over the un-margined ArcFace logits. The
paper uses ArcFace alone for sub-variety classification, so its weight defaults
to ``0.0``; it is available because a small amount of plain CE sometimes helps
ArcFace converge early in training.

The criterion holds **no learnable parameters** -- the ArcFace class centres live
in the model. That means ``model.parameters()`` is the complete optimisation
set and a saved ``model_state_dict`` is sufficient for inference.

Ablation switches
-----------------

``use_kl_loss=False`` (the ``wo_kl`` variant) skips Eq. 10 outright rather than
weighting it to zero, so no gradient is computed for it at all.

``use_arcface=False`` (the ``wo_arcface`` variant) needs **no switch here**. That
flag swaps the model's ArcFace head for
:class:`~src.models.components.classifiers.LinearSubVarietyHead`, which returns
its logits unchanged as ``sub_margin_logits``. ``L_ArcFace`` is a cross-entropy
over ``sub_margin_logits``, so with no margin present it *is* the categorical
cross-entropy the ablation calls for. Keeping one code path here means the
ablation cannot drift from the full model by way of a second loss
implementation; the reported ``arcface_loss`` component simply becomes a plain
CE, which the summary table records under the same column.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.arcface import arcface_loss
from src.losses.cosine import CosineSimilarityLoss
from src.losses.moe import MoERegularization
from src.models.builder import HierarchicalOutput


class LossBreakdown(NamedTuple):
    """Weighted total plus every unweighted component, for logging."""

    total: torch.Tensor
    seed: torch.Tensor
    arcface: torch.Tensor
    sub_ce: torch.Tensor
    kl: torch.Tensor
    moe_load: torch.Tensor
    moe_sparsity: torch.Tensor
    cosine: torch.Tensor

    def as_dict(self) -> dict[str, float]:
        """Detached Python floats keyed for the experiment tracker."""
        return {
            "total_loss": float(self.total.detach()),
            "seed_type_loss": float(self.seed.detach()),
            "arcface_loss": float(self.arcface.detach()),
            "sub_variety_ce_loss": float(self.sub_ce.detach()),
            "kl_loss": float(self.kl.detach()),
            "moe_load_balancing_loss": float(self.moe_load.detach()),
            "moe_sparsity_loss": float(self.moe_sparsity.detach()),
            "cosine_loss": float(self.cosine.detach()),
        }


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


def hierarchical_kl_loss(
    seed_type_logits: torch.Tensor,
    sub_variety_logits: torch.Tensor,
    mapping_matrix: torch.Tensor,
    detach_seed_target: bool = False,
) -> torch.Tensor:
    """``D_KL(P_seed || P_sub-variety)`` after aggregating sub-varieties (Eq. 10).

    Args:
        seed_type_logits: ``s``, shape ``[batch, num_seed_types]``.
        sub_variety_logits: Un-margined ArcFace logits, shape
            ``[batch, num_sub_varieties]``. Margin logits must **not** be used
            here: the margin is a training device for the ArcFace term, not part
            of the predicted distribution.
        mapping_matrix: ``M``, shape ``[num_sub_varieties, num_seed_types]``.
        detach_seed_target: Treat the seed-type distribution as a fixed target,
            so the KL gradient only reshapes the sub-variety branch.

    ``F.kl_div(input=log q, target=p)`` computes ``KL(p || q)``, so passing the
    aggregated sub-variety log-probabilities as ``input`` and the seed-type
    probabilities as ``target`` gives the direction Eq. 10 asks for.
    """
    seed_probs = F.softmax(seed_type_logits, dim=-1)
    if detach_seed_target:
        seed_probs = seed_probs.detach()

    sub_probs = F.softmax(sub_variety_logits, dim=-1)
    mapping_matrix = mapping_matrix.to(device=sub_probs.device, dtype=sub_probs.dtype)
    aggregated = torch.matmul(sub_probs, mapping_matrix)
    return F.kl_div(torch.log(aggregated.clamp_min(1e-8)), seed_probs, reduction="batchmean")


class CombinedHierarchicalLoss(nn.Module):
    """Weighted sum of every loss component described in the paper.

    Args:
        num_seed_types: Coarse classes (paper: 4).
        num_sub_varieties: Fine classes (paper: 27).
        subvariety_to_seed_type: Parent seed type of each sub-variety index.
        subvarieties_per_seed_type: Alternative to the above; see
            :func:`build_subvariety_seed_mapping`.
        lambda_seed: Weight on Eq. 7.
        lambda_arcface: Weight on Eq. 13.
        lambda_kl: Weight on Eq. 10.
        lambda_moe_load: Weight on the entropy load-balancing term.
        lambda_moe_sparsity: Weight on the L1 sparsity term.
        lambda_cosine: Weight on the residual compactness term.
        lambda_sub_ce: Weight on the auxiliary plain cross-entropy (default 0).
        seed_label_smoothing: Label smoothing for Eq. 7.
        arcface_label_smoothing: Label smoothing for Eq. 13.
        detach_kl_seed_target: See :func:`hierarchical_kl_loss`.
        moe_sparsity_mode: See :func:`~src.losses.moe.l1_sparsity_loss`.
        normalize_moe_entropy: Bound the load-balancing term in ``[-1, 0]``.
        cosine_mode: ``"residual"`` or ``"intra_class"``.
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
        lambda_seed: float = 1.0,
        lambda_arcface: float = 1.0,
        lambda_kl: float = 1.0,
        lambda_moe_load: float = 0.01,
        lambda_moe_sparsity: float = 0.01,
        lambda_cosine: float = 0.1,
        lambda_sub_ce: float = 0.0,
        seed_label_smoothing: float = 0.0,
        arcface_label_smoothing: float = 0.0,
        detach_kl_seed_target: bool = False,
        moe_sparsity_mode: str = "off_topk",
        normalize_moe_entropy: bool = True,
        cosine_mode: str = "residual",
        use_kl_loss: bool = True,
    ):
        super().__init__()
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
            normalize_entropy=normalize_moe_entropy,
            sparsity_mode=moe_sparsity_mode,
        )
        self.cosine_loss = CosineSimilarityLoss(mode=cosine_mode)

        self.use_kl_loss = bool(use_kl_loss)
        self.lambda_seed = float(lambda_seed)
        self.lambda_arcface = float(lambda_arcface)
        self.lambda_kl = float(lambda_kl) if self.use_kl_loss else 0.0
        self.lambda_cosine = float(lambda_cosine)
        self.lambda_sub_ce = float(lambda_sub_ce)
        self.seed_label_smoothing = float(seed_label_smoothing)
        self.arcface_label_smoothing = float(arcface_label_smoothing)
        self.detach_kl_seed_target = bool(detach_kl_seed_target)

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
            )
            if self.use_kl_loss
            else output.sub_logits.new_zeros(())
        )
        moe = self.moe_regularization(output.gate_probs, output.top_k_indices)
        if self.cosine_loss.mode == "residual":
            # Eq. 9's residual: keep h' aligned with h.
            cosine = self.cosine_loss(output.refined_features, output.moe_features)
        else:
            # Class compactness is measured on the ArcFace embedding.
            cosine = self.cosine_loss(output.sub_embeddings, output.sub_embeddings, sub_variety_labels)

        total = (
            self.lambda_seed * seed
            + self.lambda_arcface * arcface
            + self.lambda_sub_ce * sub_ce
            + self.lambda_kl * kl
            + moe.total
            + self.lambda_cosine * cosine
        )
        return LossBreakdown(
            total=total,
            seed=seed,
            arcface=arcface,
            sub_ce=sub_ce,
            kl=kl,
            moe_load=moe.load_balancing,
            moe_sparsity=moe.sparsity,
            cosine=cosine,
        )

    def extra_repr(self) -> str:
        return (
            f"lambda_seed={self.lambda_seed}, lambda_arcface={self.lambda_arcface}, "
            f"lambda_kl={self.lambda_kl}, lambda_cosine={self.lambda_cosine}, "
            f"lambda_sub_ce={self.lambda_sub_ce}, use_kl_loss={self.use_kl_loss}"
        )


def build_combined_loss(
    loss_cfg,
    num_seed_types: int,
    num_sub_varieties: int,
    subvariety_to_seed_type: Sequence[int] | None = None,
) -> CombinedHierarchicalLoss:
    """Instantiate :class:`CombinedHierarchicalLoss` from a loss config node."""

    def get(key: str, default):
        value = getattr(loss_cfg, key, default)
        return default if value is None else value

    return CombinedHierarchicalLoss(
        num_seed_types=num_seed_types,
        num_sub_varieties=num_sub_varieties,
        subvariety_to_seed_type=subvariety_to_seed_type,
        lambda_seed=float(get("lambda_seed", 1.0)),
        lambda_arcface=float(get("lambda_arcface", 1.0)),
        lambda_kl=float(get("lambda_kl", 1.0)),
        lambda_moe_load=float(get("lambda_moe_load", 0.01)),
        lambda_moe_sparsity=float(get("lambda_moe_sparsity", 0.01)),
        lambda_cosine=float(get("lambda_cosine", 0.1)),
        lambda_sub_ce=float(get("lambda_sub_ce", 0.0)),
        seed_label_smoothing=float(get("seed_label_smoothing", 0.0)),
        arcface_label_smoothing=float(get("arcface_label_smoothing", 0.0)),
        detach_kl_seed_target=bool(get("detach_kl_seed_target", False)),
        moe_sparsity_mode=str(get("moe_sparsity_mode", "off_topk")),
        normalize_moe_entropy=bool(get("normalize_moe_entropy", True)),
        cosine_mode=str(get("cosine_mode", "residual")),
        use_kl_loss=bool(get("use_kl_loss", True)),
    )
