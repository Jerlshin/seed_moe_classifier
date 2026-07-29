"""Loss functions for both training stages (paper Sections 4-5)."""

from src.losses.arcface import ArcFaceLoss, arcface_loss
from src.losses.cosine import (
    CosineSimilarityLoss,
    intra_class_cosine_loss,
    residual_cosine_loss,
)
from src.losses.dino import CustomDINOLoss
from src.losses.hierarchical import (
    CombinedHierarchicalLoss,
    LossBreakdown,
    build_combined_loss,
    build_subvariety_seed_mapping,
    hierarchical_kl_loss,
    seed_type_loss,
)
from src.losses.moe import (
    MoERegularization,
    MoERegularizationOutput,
    expert_utilization,
    l1_sparsity_loss,
    load_balancing_loss,
)

__all__ = [
    "ArcFaceLoss",
    "CombinedHierarchicalLoss",
    "CosineSimilarityLoss",
    "CustomDINOLoss",
    "LossBreakdown",
    "MoERegularization",
    "MoERegularizationOutput",
    "arcface_loss",
    "build_combined_loss",
    "build_subvariety_seed_mapping",
    "expert_utilization",
    "hierarchical_kl_loss",
    "intra_class_cosine_loss",
    "l1_sparsity_loss",
    "load_balancing_loss",
    "residual_cosine_loss",
    "seed_type_loss",
]
