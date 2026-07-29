"""Model definitions for the hierarchical seed classifier (paper Sections 4-5)."""

from src.models.backbones.swinv2_dino import DINO, DINOHead
from src.models.baselines import (
    BASELINE_MODELS,
    FlatSupervisedBaseline,
    IdentityEncoder,
    build_baseline,
)
from src.models.builder import (
    PAPER_EMBED_DIM,
    BackboneFeatureExtractor,
    DinoV2SwinV2Encoder,
    HierarchicalOutput,
    HierarchicalSeedClassifier,
    build_encoder,
    build_feature_extractor,
    build_hierarchical_moe,
    validate_swinv2_name,
)
from src.models.components import (
    ArcFaceHead,
    CrossAttention,
    DenseExpertBlock,
    EmbeddingProjection,
    LinearSubVarietyHead,
    MixtureOfExperts,
    MoEOutput,
    SeedTypeClassifier,
    SeedTypeProjection,
    SubVarietyEmbedding,
    TransformerExpert,
)

__all__ = [
    "BASELINE_MODELS",
    "DINO",
    "PAPER_EMBED_DIM",
    "ArcFaceHead",
    "BackboneFeatureExtractor",
    "CrossAttention",
    "DINOHead",
    "DenseExpertBlock",
    "DinoV2SwinV2Encoder",
    "EmbeddingProjection",
    "FlatSupervisedBaseline",
    "HierarchicalOutput",
    "HierarchicalSeedClassifier",
    "IdentityEncoder",
    "LinearSubVarietyHead",
    "MixtureOfExperts",
    "MoEOutput",
    "SeedTypeClassifier",
    "SeedTypeProjection",
    "SubVarietyEmbedding",
    "TransformerExpert",
    "build_baseline",
    "build_encoder",
    "build_feature_extractor",
    "build_hierarchical_moe",
    "validate_swinv2_name",
]
