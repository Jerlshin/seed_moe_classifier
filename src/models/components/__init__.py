"""Reusable building blocks of the hierarchical head (paper Section 5)."""

from src.models.components.arcface_head import (
    ArcFaceHead,
    NormFaceHead,
    adacos_scale,
    resolve_scale,
)
from src.models.components.classifiers import (
    LinearSubVarietyHead,
    SeedTypeClassifier,
    SubVarietyEmbedding,
)
from src.models.components.cross_attention import (
    AdaptiveGating,
    CrossAttention,
    CrossAttentionOutput,
    resolve_attention_mode,
)
from src.models.components.moe_layer import (
    DEFAULT_NUM_EXPERTS,
    DEFAULT_TOP_K,
    ROUTER_MODES,
    TOKEN_MIXING_MODES,
    DenseExpertBlock,
    MixtureOfExperts,
    MoEOutput,
    TransformerExpert,
)
from src.models.components.projections import (
    FUSION_MODES,
    TOKEN_POOLING_MODES,
    EmbeddingProjection,
    FiLMFusion,
    LayerScale,
    SeedTypeProjection,
    TokenPooling,
)

__all__ = [
    "DEFAULT_NUM_EXPERTS",
    "DEFAULT_TOP_K",
    "FUSION_MODES",
    "ROUTER_MODES",
    "TOKEN_MIXING_MODES",
    "TOKEN_POOLING_MODES",
    "AdaptiveGating",
    "ArcFaceHead",
    "CrossAttention",
    "CrossAttentionOutput",
    "DenseExpertBlock",
    "EmbeddingProjection",
    "FiLMFusion",
    "LayerScale",
    "LinearSubVarietyHead",
    "MixtureOfExperts",
    "MoEOutput",
    "NormFaceHead",
    "SeedTypeClassifier",
    "SeedTypeProjection",
    "SubVarietyEmbedding",
    "TokenPooling",
    "TransformerExpert",
    "adacos_scale",
    "resolve_attention_mode",
    "resolve_scale",
]
