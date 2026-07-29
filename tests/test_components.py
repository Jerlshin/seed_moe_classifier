"""Cross-attention, ArcFace, projections and classifier heads (Eqs. 5-13)."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from src.models.components.arcface_head import ArcFaceHead
from src.models.components.classifiers import SeedTypeClassifier, SubVarietyEmbedding
from src.models.components.cross_attention import CrossAttention
from src.models.components.projections import EmbeddingProjection, SeedTypeProjection
from tests.conftest import PAPER_EMBED_DIM, PAPER_NUM_SEED_TYPES, PAPER_NUM_SUB_VARIETIES


# ------------------------------------------------------------ cross-attention


def test_cross_attention_preserves_query_shape():
    block = CrossAttention(dim=PAPER_EMBED_DIM, num_heads=4, dropout=0.0)
    query = torch.randn(8, 1, PAPER_EMBED_DIM)
    key_value = torch.randn(8, 1, PAPER_EMBED_DIM)
    output = block(query, key_value, key_value)
    assert output.features.shape == query.shape


def test_cross_attention_applies_layernorm_to_the_residual():
    """Eq. 12: h'' = LayerNorm(a + Q), so each row must be standardised."""
    block = CrossAttention(dim=64, num_heads=4, dropout=0.0)
    block.eval()
    query = torch.randn(8, 1, 64)
    key_value = torch.randn(8, 1, 64)
    with torch.no_grad():
        features = block(query, key_value, key_value).features.squeeze(1)
    assert torch.allclose(features.mean(dim=-1), torch.zeros(8), atol=1e-5)
    assert torch.allclose(features.std(dim=-1, unbiased=False), torch.ones(8), atol=1e-2)


def test_cross_attention_returns_weights_only_when_requested():
    block = CrossAttention(dim=32, num_heads=4, dropout=0.0)
    query = torch.randn(4, 1, 32)
    assert block(query, query, query, need_weights=False).attn_weights is None
    weights = block(query, query, query, need_weights=True).attn_weights
    assert weights is not None
    # Single key/value token: attention over one key is trivially 1.0.
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-5)


def test_cross_attention_rejects_unknown_variant():
    with pytest.raises(ValueError, match="variant"):
        CrossAttention(dim=32, variant="nonsense")


def test_cross_attention_gated_variant_runs():
    block = CrossAttention(dim=32, num_heads=4, dropout=0.0, variant="gated")
    query = torch.randn(4, 1, 32)
    assert block(query, query, query).features.shape == query.shape


# -------------------------------------------------------------------- ArcFace


@pytest.fixture
def arcface() -> ArcFaceHead:
    torch.manual_seed(0)
    return ArcFaceHead(
        feature_dim=PAPER_EMBED_DIM,
        num_classes=PAPER_NUM_SUB_VARIETIES,
        scale=30.0,
        margin=0.5,
    )


def test_arcface_shapes(arcface):
    embeddings = torch.randn(8, PAPER_EMBED_DIM)
    labels = torch.randint(0, PAPER_NUM_SUB_VARIETIES, (8,))
    logits, margin_logits = arcface(embeddings, labels)
    assert logits.shape == (8, PAPER_NUM_SUB_VARIETIES)
    assert margin_logits.shape == (8, PAPER_NUM_SUB_VARIETIES)


def test_arcface_logits_are_bounded_by_the_scale(arcface):
    """Logits are ``s * cos(theta)``, so they live in ``[-s, s]``."""
    logits, _ = arcface(torch.randn(32, PAPER_EMBED_DIM))
    assert logits.abs().max().item() <= arcface.scale + 1e-4


def test_arcface_without_labels_applies_no_margin(arcface):
    """At inference the two outputs must be identical."""
    logits, margin_logits = arcface(torch.randn(8, PAPER_EMBED_DIM), None)
    assert torch.equal(logits, margin_logits)


def test_arcface_margin_only_touches_the_target_class(arcface):
    embeddings = torch.randn(8, PAPER_EMBED_DIM)
    labels = torch.randint(0, PAPER_NUM_SUB_VARIETIES, (8,))
    logits, margin_logits = arcface(embeddings, labels)

    difference = margin_logits - logits
    for row, label in enumerate(labels):
        others = torch.ones(PAPER_NUM_SUB_VARIETIES, dtype=torch.bool)
        others[label] = False
        assert torch.allclose(difference[row][others], torch.zeros(others.sum()), atol=1e-4)


def test_arcface_margin_lowers_the_target_logit(arcface):
    """Adding m to theta reduces cos(theta + m), making the target harder."""
    embeddings = torch.randn(64, PAPER_EMBED_DIM)
    labels = torch.randint(0, PAPER_NUM_SUB_VARIETIES, (64,))
    logits, margin_logits = arcface(embeddings, labels)
    target = torch.arange(64), labels
    assert (margin_logits[target] <= logits[target] + 1e-4).all()


def test_arcface_margin_matches_the_closed_form(arcface):
    """cos(theta + m) computed by expansion must equal the direct evaluation."""
    embeddings = torch.randn(16, PAPER_EMBED_DIM)
    labels = torch.randint(0, PAPER_NUM_SUB_VARIETIES, (16,))
    cosine = arcface.cosine_similarity(embeddings)
    _, margin_logits = arcface(embeddings, labels)

    rows = torch.arange(16)
    target_cosine = cosine[rows, labels]
    theta = torch.acos(target_cosine.clamp(-1 + 1e-7, 1 - 1e-7))
    expected = torch.cos(theta + arcface.margin)
    # Only compare where theta + m stays below pi; beyond that ArcFace switches
    # to its linear fallback by design.
    in_range = (theta + arcface.margin) < math.pi
    actual = margin_logits[rows, labels] / arcface.scale
    assert torch.allclose(actual[in_range], expected[in_range], atol=1e-4)


def test_arcface_gradients_are_finite_at_the_cosine_boundary():
    """The expansion form must not blow up where an ``acos`` implementation would."""
    head = ArcFaceHead(feature_dim=8, num_classes=3, scale=30.0, margin=0.5)
    with torch.no_grad():
        head.weight.copy_(torch.eye(3, 8))
    # Embedding exactly parallel to class 0's centre => cos(theta) = 1.
    embeddings = torch.zeros(1, 8, requires_grad=True)
    with torch.no_grad():
        embeddings[0, 0] = 1.0
    embeddings.requires_grad_(True)

    _, margin_logits = head(embeddings, torch.tensor([0]))
    F.cross_entropy(margin_logits, torch.tensor([0])).backward()
    assert torch.isfinite(embeddings.grad).all()


def test_arcface_cosine_is_within_unit_range(arcface):
    cosine = arcface.cosine_similarity(torch.randn(32, PAPER_EMBED_DIM))
    assert cosine.min().item() >= -1.0
    assert cosine.max().item() <= 1.0


# ----------------------------------------------------------------- projections


def test_embedding_projection_maps_backbone_width_to_paper_dim():
    """Eq. 4: SwinV2-Base's 1024 channels must reach z in R^384."""
    projection = EmbeddingProjection(in_dim=1024, out_dim=PAPER_EMBED_DIM)
    assert projection(torch.randn(6, 1024)).shape == (6, PAPER_EMBED_DIM)


def test_embedding_projection_is_a_no_op_when_widths_match():
    projection = EmbeddingProjection(in_dim=384, out_dim=384, use_norm=False)
    x = torch.randn(4, 384)
    assert torch.equal(projection(x), x)


def test_embedding_projection_rejects_wrong_width():
    projection = EmbeddingProjection(in_dim=1024, out_dim=PAPER_EMBED_DIM)
    with pytest.raises(ValueError, match="in_dim"):
        projection(torch.randn(2, 512))


def test_seed_type_projection_lifts_probabilities_to_feature_space():
    """Eq. 9: P maps R^4 -> R^384."""
    projection = SeedTypeProjection(num_seed_types=PAPER_NUM_SEED_TYPES, embed_dim=PAPER_EMBED_DIM)
    probabilities = F.softmax(torch.randn(5, PAPER_NUM_SEED_TYPES), dim=-1)
    assert projection(probabilities).shape == (5, PAPER_EMBED_DIM)


def test_seed_type_projection_rejects_wrong_class_count():
    projection = SeedTypeProjection(num_seed_types=4, embed_dim=64)
    with pytest.raises(ValueError, match="seed types"):
        projection(torch.randn(3, 5))


# ------------------------------------------------------------------ classifiers


@pytest.mark.parametrize("variant", ["mlp", "se_gated"])
def test_seed_type_classifier_emits_four_logits(variant):
    """Eq. 5: s in R^4."""
    classifier = SeedTypeClassifier(
        feature_dim=PAPER_EMBED_DIM,
        num_seed_types=PAPER_NUM_SEED_TYPES,
        dropout_rate=0.0,
        variant=variant,
    )
    assert classifier(torch.randn(7, PAPER_EMBED_DIM)).shape == (7, PAPER_NUM_SEED_TYPES)


def test_seed_type_classifier_rejects_unknown_variant():
    with pytest.raises(ValueError, match="variant"):
        SeedTypeClassifier(feature_dim=32, num_seed_types=4, variant="nonsense")


@pytest.mark.parametrize("variant", ["mlp", "identity"])
def test_sub_variety_embedding_preserves_width(variant):
    embedding = SubVarietyEmbedding(feature_dim=PAPER_EMBED_DIM, dropout_rate=0.0, variant=variant)
    x = torch.randn(6, PAPER_EMBED_DIM)
    assert embedding(x).shape == (6, PAPER_EMBED_DIM)


def test_sub_variety_identity_variant_is_a_pass_through():
    embedding = SubVarietyEmbedding(feature_dim=32, variant="identity")
    x = torch.randn(4, 32)
    assert torch.equal(embedding(x), x)
