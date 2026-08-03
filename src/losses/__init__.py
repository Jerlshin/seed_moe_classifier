"""Loss functions for both training stages (paper Sections 4-5)."""

from src.losses.arcface import ArcFaceLoss, arcface_loss
from src.losses.cosine import (
    CosineSimilarityLoss,
    intra_class_cosine_loss,
    residual_cosine_loss,
    residual_magnitude_loss,
)
from src.losses.dino import CustomDINOLoss, koleo_regularizer, sinkhorn_knopp
from src.losses.hierarchical import (
    CombinedHierarchicalLoss,
    LossBreakdown,
    UncertaintyWeighting,
    aggregate_sub_log_probs,
    build_combined_loss,
    build_subvariety_seed_mapping,
    hierarchical_kl_loss,
    seed_type_loss,
)
from src.losses.moe import (
    MoERegularization,
    MoERegularizationOutput,
    dispatch_fraction,
    entropy_load_balancing_loss,
    expert_utilization,
    l1_sparsity_loss,
    load_balancing_loss,
    router_z_loss,
    switch_load_balancing_loss,
)

__all__ = [
    "ArcFaceLoss",
    "CombinedHierarchicalLoss",
    "CosineSimilarityLoss",
    "CustomDINOLoss",
    "LossBreakdown",
    "MoERegularization",
    "MoERegularizationOutput",
    "UncertaintyWeighting",
    "aggregate_sub_log_probs",
    "arcface_loss",
    "build_combined_loss",
    "build_subvariety_seed_mapping",
    "dispatch_fraction",
    "entropy_load_balancing_loss",
    "expert_utilization",
    "hierarchical_kl_loss",
    "intra_class_cosine_loss",
    "koleo_regularizer",
    "l1_sparsity_loss",
    "load_balancing_loss",
    "residual_cosine_loss",
    "residual_magnitude_loss",
    "router_z_loss",
    "seed_type_loss",
    "sinkhorn_knopp",
    "switch_load_balancing_loss",
]
