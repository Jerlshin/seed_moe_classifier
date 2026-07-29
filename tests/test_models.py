"""End-to-end shapes, dataflow, gradient flow and component toggles.

These tests encode the paper's Section 5 dataflow, in particular the two places
an implementation can silently diverge: what the MoE is routed on (``z``, not the
seed projection) and what the residual adds (``P(p_s)``, not ``P(s)``).

They also pin the revision's contracts: the encoder emits ``z in R^384`` for any
SwinV2 width, the router selects exactly ``K = 2`` experts, and each of the four
architectural toggles removes its block without breaking the output contract.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.builder import (
    HierarchicalSeedClassifier,
    build_hierarchical_moe,
    validate_swinv2_name,
)
from tests.conftest import (
    PAPER_EMBED_DIM,
    PAPER_NUM_EXPERTS,
    PAPER_NUM_SEED_TYPES,
    PAPER_NUM_SUB_VARIETIES,
    REVISED_TOP_K,
)


def make_model(**overrides) -> HierarchicalSeedClassifier:
    """The paper's head at test scale, with any keyword overridden."""
    torch.manual_seed(0)
    kwargs = {
        "feature_dim": PAPER_EMBED_DIM,
        "embed_dim": PAPER_EMBED_DIM,
        "num_seed_types": PAPER_NUM_SEED_TYPES,
        "num_sub_varieties": PAPER_NUM_SUB_VARIETIES,
        "num_experts": PAPER_NUM_EXPERTS,
        "top_k": REVISED_TOP_K,
        "moe_hidden_dim": 32,
        "num_heads": 4,
        "dropout_rate": 0.0,
    }
    kwargs.update(overrides)
    return HierarchicalSeedClassifier(**kwargs)


def test_forward_produces_every_paper_intermediate(hierarchical_model, batch):
    output = hierarchical_model(batch["features"], batch["sub_labels"])
    size = batch["size"]

    assert output.embedding.shape == (size, PAPER_EMBED_DIM)                  # Eq. 4
    assert output.seed_type_logits.shape == (size, PAPER_NUM_SEED_TYPES)      # Eq. 5
    assert output.seed_type_probs.shape == (size, PAPER_NUM_SEED_TYPES)       # Eq. 6
    assert output.moe_features.shape == (size, PAPER_EMBED_DIM)               # Eq. 8
    assert output.projected_seed.shape == (size, PAPER_EMBED_DIM)             # Eq. 9
    assert output.refined_features.shape == (size, PAPER_EMBED_DIM)           # Eq. 9
    assert output.attended_features.shape == (size, PAPER_EMBED_DIM)          # Eq. 12
    assert output.sub_embeddings.shape == (size, PAPER_EMBED_DIM)
    assert output.sub_logits.shape == (size, PAPER_NUM_SUB_VARIETIES)         # Eq. 13
    assert output.sub_margin_logits.shape == (size, PAPER_NUM_SUB_VARIETIES)
    assert output.gate_probs.shape == (size, PAPER_NUM_EXPERTS)
    assert output.top_k_indices.shape == (size, REVISED_TOP_K)


def test_seed_probabilities_are_a_softmax_of_the_logits(hierarchical_model, batch):
    """Eq. 6: p_s = softmax(s)."""
    output = hierarchical_model(batch["features"])
    assert torch.allclose(
        output.seed_type_probs, F.softmax(output.seed_type_logits, dim=-1), atol=1e-6
    )
    assert torch.allclose(output.seed_type_probs.sum(-1), torch.ones(batch["size"]), atol=1e-5)


def test_residual_is_exactly_moe_plus_projected_probabilities(hierarchical_model, batch):
    """Eq. 9: h' = h + P(p_s), with no extra scaling or normalisation."""
    output = hierarchical_model(batch["features"])
    assert torch.allclose(
        output.refined_features, output.moe_features + output.projected_seed, atol=1e-6
    )


def test_seed_projection_consumes_probabilities_not_logits(hierarchical_model, batch):
    """P must be applied to p_s; feeding it s would give a different vector."""
    output = hierarchical_model(batch["features"])
    from_probs = hierarchical_model.seed_projection(output.seed_type_probs)
    assert torch.allclose(output.projected_seed, from_probs, atol=1e-6)

    from_logits = hierarchical_model.seed_projection(output.seed_type_logits)
    assert not torch.allclose(output.projected_seed, from_logits, atol=1e-4)


def test_moe_is_routed_on_the_dino_embedding(hierarchical_model, batch):
    """Eq. 8 routes on z. Routing on P(p_s) would starve the experts of image content."""
    hierarchical_model.eval()
    with torch.no_grad():
        output = hierarchical_model(batch["features"])
        direct = hierarchical_model.moe(output.embedding)
    assert torch.allclose(output.gate_probs, direct.gate_probs, atol=1e-6)
    assert torch.allclose(output.moe_features, direct.features, atol=1e-6)


def test_cross_attention_uses_refined_query_and_raw_moe_key_value(hierarchical_model, batch):
    """Eq. 11: Q = h', K = V = h."""
    hierarchical_model.eval()
    with torch.no_grad():
        output = hierarchical_model(batch["features"])
        expected = hierarchical_model.cross_attention(
            query=output.refined_features.unsqueeze(1),
            key=output.moe_features.unsqueeze(1),
            value=output.moe_features.unsqueeze(1),
        ).features.squeeze(1)
    assert torch.allclose(output.attended_features, expected, atol=1e-5)


def test_margin_logits_equal_plain_logits_without_labels(hierarchical_model, batch):
    output = hierarchical_model(batch["features"], sub_variety_labels=None)
    assert torch.equal(output.sub_logits, output.sub_margin_logits)


def test_margin_is_applied_when_labels_are_supplied(hierarchical_model, batch):
    output = hierarchical_model(batch["features"], batch["sub_labels"])
    assert not torch.equal(output.sub_logits, output.sub_margin_logits)


def test_gradients_reach_every_parameter_outside_the_experts(
    hierarchical_model, batch, combined_loss
):
    """A dead branch in the cascade shows up here as a parameter with no gradient.

    Experts are excluded deliberately -- see
    :func:`test_only_the_routed_experts_receive_gradient` for why an unrouted
    expert having no gradient is correct rather than a defect.
    """
    output = hierarchical_model(batch["features"], batch["sub_labels"])
    combined_loss(output, batch["seed_labels"], batch["sub_labels"]).total.backward()

    missing = [
        name
        for name, parameter in hierarchical_model.named_parameters()
        if parameter.requires_grad
        and not name.startswith("moe.experts.")
        and (parameter.grad is None or parameter.grad.abs().sum() == 0)
    ]
    assert not missing, f"parameters received no gradient: {missing}"


def test_only_the_routed_experts_receive_gradient(hierarchical_model, batch, combined_loss):
    """Sparse dispatch means an expert no sample selected gets no gradient this step.

    That is the defining property of a sparse MoE, not a bug -- but it becomes
    much more visible at ``K = 2`` than at the submitted ``K = 4``. A batch of
    12 fills only 24 routing slots across 6 experts, so an expert sitting out an
    entire batch is common. Over an epoch the load-balancing term keeps pulling
    utilisation back toward uniform, so no expert stays unrouted for long.

    The assertion is therefore the precise one: every expert that *was* selected
    must have a gradient, and every expert that was *not* must not.
    """
    output = hierarchical_model(batch["features"], batch["sub_labels"])
    combined_loss(output, batch["seed_labels"], batch["sub_labels"]).total.backward()

    routed = set(output.top_k_indices.flatten().tolist())
    assert routed, "no expert was selected at all"

    for index, expert in enumerate(hierarchical_model.moe.experts):
        received = any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in expert.parameters()
        )
        assert received == (index in routed), (
            f"expert {index} was {'routed' if index in routed else 'not routed'} "
            f"but {'received' if received else 'received no'} gradient"
        )


def test_gradients_are_finite(hierarchical_model, batch, combined_loss):
    output = hierarchical_model(batch["features"], batch["sub_labels"])
    combined_loss(output, batch["seed_labels"], batch["sub_labels"]).total.backward()
    for name, parameter in hierarchical_model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), f"non-finite gradient in {name}"


def test_backbone_width_is_projected_to_the_paper_embedding_dim():
    """SwinV2-Base emits 1024 channels; Eq. 4 still requires z in R^384."""
    model = HierarchicalSeedClassifier(
        feature_dim=1024,
        embed_dim=PAPER_EMBED_DIM,
        num_seed_types=PAPER_NUM_SEED_TYPES,
        num_sub_varieties=PAPER_NUM_SUB_VARIETIES,
        moe_hidden_dim=32,
        num_heads=4,
        dropout_rate=0.0,
    )
    output = model(torch.randn(4, 1024))
    assert output.embedding.shape == (4, PAPER_EMBED_DIM)
    assert output.sub_logits.shape == (4, PAPER_NUM_SUB_VARIETIES)


def test_predict_returns_argmax_of_both_heads(hierarchical_model, batch):
    hierarchical_model.eval()
    seed_pred, sub_pred = hierarchical_model.predict(batch["features"])
    assert seed_pred.shape == (batch["size"],)
    assert sub_pred.shape == (batch["size"],)
    assert seed_pred.max() < PAPER_NUM_SEED_TYPES
    assert sub_pred.max() < PAPER_NUM_SUB_VARIETIES


def test_rejects_non_2d_features(hierarchical_model):
    with pytest.raises(ValueError, match=r"\[batch, feature_dim\]"):
        hierarchical_model(torch.randn(4, 7, PAPER_EMBED_DIM))


def test_eval_mode_is_deterministic(hierarchical_model, batch):
    hierarchical_model.eval()
    with torch.no_grad():
        first = hierarchical_model(batch["features"]).sub_logits
        second = hierarchical_model(batch["features"]).sub_logits
    assert torch.allclose(first, second, atol=1e-6)


def test_single_sample_batch_works(hierarchical_model):
    """Batch size 1 is a common inference path and must not break the MoE dispatch."""
    hierarchical_model.eval()
    with torch.no_grad():
        output = hierarchical_model(torch.randn(1, PAPER_EMBED_DIM))
    assert output.sub_logits.shape == (1, PAPER_NUM_SUB_VARIETIES)
    assert output.top_k_indices.shape == (1, REVISED_TOP_K)


def test_loss_breakdown_reports_every_paper_component(hierarchical_model, batch, combined_loss):
    output = hierarchical_model(batch["features"], batch["sub_labels"])
    parts = combined_loss(output, batch["seed_labels"], batch["sub_labels"]).as_dict()
    assert set(parts) == {
        "total_loss",
        "seed_type_loss",
        "arcface_loss",
        "sub_variety_ce_loss",
        "kl_loss",
        "moe_load_balancing_loss",
        "moe_sparsity_loss",
        "cosine_loss",
    }
    assert all(isinstance(value, float) for value in parts.values())


def test_criterion_holds_no_parameters(combined_loss):
    """ArcFace centres live in the model, so model_state_dict alone is enough."""
    assert list(combined_loss.parameters()) == []


def test_loss_weights_scale_the_total(hierarchical_model, batch, subvariety_to_seed_type):
    """Zeroing every weight but one must leave exactly that component."""
    from src.losses.hierarchical import CombinedHierarchicalLoss

    output = hierarchical_model(batch["features"], batch["sub_labels"])
    seed_only = CombinedHierarchicalLoss(
        num_seed_types=PAPER_NUM_SEED_TYPES,
        num_sub_varieties=PAPER_NUM_SUB_VARIETIES,
        subvariety_to_seed_type=subvariety_to_seed_type,
        lambda_seed=1.0, lambda_arcface=0.0, lambda_kl=0.0,
        lambda_moe_load=0.0, lambda_moe_sparsity=0.0, lambda_cosine=0.0, lambda_sub_ce=0.0,
    )
    breakdown = seed_only(output, batch["seed_labels"], batch["sub_labels"])
    assert breakdown.total.item() == pytest.approx(breakdown.seed.item(), abs=1e-6)


# ------------------------------------------------------------- Top-2 routing


def test_router_activates_exactly_two_of_six_experts(hierarchical_model, batch):
    """The revision's headline architectural change."""
    output = hierarchical_model(batch["features"])
    assert hierarchical_model.num_experts == PAPER_NUM_EXPERTS
    assert hierarchical_model.top_k == REVISED_TOP_K == 2
    assert output.top_k_indices.shape == (batch["size"], 2)
    # Two experts per sample, and no expert selected twice.
    assert (output.dispatch_weights > 0).sum(dim=-1).eq(2).all()
    for row in output.top_k_indices:
        assert len(set(row.tolist())) == 2


def test_router_selects_the_two_highest_gate_values(hierarchical_model, batch):
    output = hierarchical_model(batch["features"])
    expected = output.gate_probs.argsort(dim=-1, descending=True)[:, :2]
    assert torch.equal(
        output.top_k_indices.sort(dim=-1).values, expected.sort(dim=-1).values
    )


def test_selected_expert_weights_form_a_convex_combination(hierarchical_model, batch):
    """Renormalisation over the selection: the two weights must sum to 1."""
    output = hierarchical_model(batch["features"])
    assert torch.allclose(
        output.top_k_weights.sum(dim=-1), torch.ones(batch["size"]), atol=1e-5
    )
    assert (output.top_k_weights >= 0).all()


def test_gate_distribution_still_covers_all_six_experts(hierarchical_model, batch):
    """Routing 2 must not shrink the gate: the discarded mass is what L1 penalises."""
    output = hierarchical_model(batch["features"])
    assert output.gate_probs.shape == (batch["size"], PAPER_NUM_EXPERTS)
    assert torch.allclose(
        output.gate_probs.sum(dim=-1), torch.ones(batch["size"]), atol=1e-5
    )


@pytest.mark.parametrize("top_k", [1, 2, 4, 6])
def test_routing_width_is_configurable(top_k):
    """Top-4 must stay reachable so the submitted manuscript can be reproduced."""
    model = make_model(top_k=top_k)
    output = model(torch.randn(5, PAPER_EMBED_DIM))
    assert output.top_k_indices.shape == (5, top_k)
    assert output.sub_logits.shape == (5, PAPER_NUM_SUB_VARIETIES)


# ---------------------------------------------------------- component toggles


ABLATION_FLAGS = ["use_moe", "use_arcface", "use_residual", "use_cross_attention"]


@pytest.mark.parametrize("flag", ABLATION_FLAGS)
def test_each_toggle_preserves_the_output_contract(flag, batch):
    """Every ablation must still fill HierarchicalOutput with correctly-shaped tensors."""
    model = make_model(**{flag: False})
    output = model(batch["features"], batch["sub_labels"])
    size = batch["size"]

    assert output.embedding.shape == (size, PAPER_EMBED_DIM)
    assert output.seed_type_logits.shape == (size, PAPER_NUM_SEED_TYPES)
    assert output.sub_logits.shape == (size, PAPER_NUM_SUB_VARIETIES)
    assert output.moe_features.shape == (size, PAPER_EMBED_DIM)
    assert output.refined_features.shape == (size, PAPER_EMBED_DIM)
    assert output.attended_features.shape == (size, PAPER_EMBED_DIM)
    assert output.gate_probs.ndim == 2
    assert output.top_k_indices.ndim == 2
    assert model.component_flags()[flag] is False


@pytest.mark.parametrize("flag", ABLATION_FLAGS)
def test_each_toggle_keeps_gradients_flowing(flag, batch, combined_loss):
    """Removing one block must not orphan any of the blocks that remain.

    Experts are excluded for the sparse-routing reason documented in
    :func:`test_only_the_routed_experts_receive_gradient`.
    """
    model = make_model(**{flag: False})
    output = model(batch["features"], batch["sub_labels"])
    combined_loss(output, batch["seed_labels"], batch["sub_labels"]).total.backward()

    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not name.startswith("moe.experts.")
        and (parameter.grad is None or parameter.grad.abs().sum() == 0)
    ]
    assert not missing, f"{flag}=False left parameters without gradient: {missing}"


@pytest.mark.parametrize("flag", ABLATION_FLAGS)
def test_disabling_a_component_never_grows_the_model(flag):
    """A disabled block must be *absent*, not merely bypassed."""
    full = sum(p.numel() for p in make_model().parameters())
    ablated = sum(p.numel() for p in make_model(**{flag: False}).parameters())
    # use_arcface swaps an ArcFace centre matrix for a Linear of the same shape
    # plus a bias, so that one variant is allowed to differ by the bias term.
    tolerance = PAPER_NUM_SUB_VARIETIES if flag == "use_arcface" else 0
    assert ablated <= full + tolerance


def test_wo_moe_uses_one_dense_always_on_block():
    """The dense replacement must report a degenerate router, not a broken one."""
    model = make_model(use_moe=False)
    output = model(torch.randn(6, PAPER_EMBED_DIM))
    assert model.num_experts == 1
    assert model.top_k == 1
    assert output.gate_probs.shape == (6, 1)
    assert torch.allclose(output.gate_probs, torch.ones(6, 1))
    assert torch.equal(output.top_k_indices, torch.zeros(6, 1, dtype=torch.long))


def test_wo_moe_regularisers_are_exactly_zero(subvariety_to_seed_type):
    """A one-expert gate has no entropy to spread and no mass outside the selection."""
    from src.losses.hierarchical import CombinedHierarchicalLoss

    model = make_model(use_moe=False)
    output = model(torch.randn(8, PAPER_EMBED_DIM))
    criterion = CombinedHierarchicalLoss(
        num_seed_types=PAPER_NUM_SEED_TYPES,
        num_sub_varieties=PAPER_NUM_SUB_VARIETIES,
        subvariety_to_seed_type=subvariety_to_seed_type,
    )
    breakdown = criterion(output, torch.randint(0, 4, (8,)), torch.randint(0, 27, (8,)))
    assert breakdown.moe_load.item() == pytest.approx(0.0, abs=1e-6)
    assert breakdown.moe_sparsity.item() == pytest.approx(0.0, abs=1e-6)


def test_wo_arcface_produces_margin_free_logits(batch):
    """Without a margin the two logit tensors coincide, so the CE term is plain CE."""
    model = make_model(use_arcface=False)
    output = model(batch["features"], batch["sub_labels"])
    assert torch.equal(output.sub_logits, output.sub_margin_logits)


def test_wo_residual_zeroes_the_projection_and_skips_equation_9(batch):
    model = make_model(use_residual=False)
    output = model(batch["features"])
    assert model.seed_projection is None
    assert torch.count_nonzero(output.projected_seed) == 0
    assert torch.equal(output.refined_features, output.moe_features)


def test_wo_residual_makes_the_cosine_term_vanish(subvariety_to_seed_type):
    """cos(h, h) = 1, so 1 - cos = 0: there is no residual left to regularise."""
    from src.losses.hierarchical import CombinedHierarchicalLoss

    model = make_model(use_residual=False)
    output = model(torch.randn(8, PAPER_EMBED_DIM))
    criterion = CombinedHierarchicalLoss(
        num_seed_types=PAPER_NUM_SEED_TYPES,
        num_sub_varieties=PAPER_NUM_SUB_VARIETIES,
        subvariety_to_seed_type=subvariety_to_seed_type,
    )
    breakdown = criterion(output, torch.randint(0, 4, (8,)), torch.randint(0, 27, (8,)))
    assert breakdown.cosine.item() == pytest.approx(0.0, abs=1e-5)


def test_wo_cross_attention_passes_the_refined_feature_through(batch):
    model = make_model(use_cross_attention=False)
    output = model(batch["features"])
    assert model.cross_attention is None
    assert torch.equal(output.attended_features, output.refined_features)
    assert output.attn_weights is None


def test_all_toggles_off_still_runs(batch):
    """The most-reduced configuration must remain a working classifier."""
    model = make_model(
        use_moe=False, use_arcface=False, use_residual=False, use_cross_attention=False
    )
    output = model(batch["features"], batch["sub_labels"])
    assert output.sub_logits.shape == (batch["size"], PAPER_NUM_SUB_VARIETIES)
    assert output.seed_type_logits.shape == (batch["size"], PAPER_NUM_SEED_TYPES)


def test_builder_reads_the_toggles_from_a_config_node():
    from omegaconf import OmegaConf

    model = build_hierarchical_moe(
        OmegaConf.create(
            {
                "feature_dim": PAPER_EMBED_DIM,
                "embed_dim": PAPER_EMBED_DIM,
                "num_seed_types": PAPER_NUM_SEED_TYPES,
                "num_sub_varieties": PAPER_NUM_SUB_VARIETIES,
                "num_experts": PAPER_NUM_EXPERTS,
                "top_k": REVISED_TOP_K,
                "moe_hidden_dim": 32,
                "num_heads": 4,
                "dropout_rate": 0.0,
                "use_moe": False,
                "use_arcface": False,
                "use_residual": True,
                "use_cross_attention": False,
            }
        )
    )
    assert model.component_flags() == {
        "use_moe": False,
        "use_arcface": False,
        "use_residual": True,
        "use_cross_attention": False,
    }


def test_builder_defaults_to_top_2_when_the_config_omits_it():
    from omegaconf import OmegaConf

    model = build_hierarchical_moe(
        OmegaConf.create(
            {
                "feature_dim": PAPER_EMBED_DIM,
                "num_seed_types": PAPER_NUM_SEED_TYPES,
                "num_sub_varieties": PAPER_NUM_SUB_VARIETIES,
                "moe_hidden_dim": 32,
                "num_heads": 4,
            }
        )
    )
    assert model.top_k == REVISED_TOP_K
    assert model.num_experts == PAPER_NUM_EXPERTS


# ---------------------------------------------------- encoder standardisation


def test_input_projection_is_an_identity_when_widths_match():
    """The encoder already produces z, so the head must not project it again."""
    model = make_model(feature_dim=PAPER_EMBED_DIM, embed_dim=PAPER_EMBED_DIM)
    assert isinstance(model.input_projection, nn.Identity)


@pytest.mark.parametrize("name", ["swinv2_base_window16_256", "swinv2_tiny_window16_256"])
def test_swinv2_names_are_accepted(name):
    assert validate_swinv2_name(name) == name


@pytest.mark.parametrize(
    "name", ["dinov2_vits14", "vit_small_patch16_224", "resnet50", "swin_tiny_patch4_window7_224"]
)
def test_non_swinv2_backbones_are_rejected(name):
    """Including Swin V1, which is close enough to slip through an eyeball check."""
    with pytest.raises(ValueError, match="Swin Transformer V2"):
        validate_swinv2_name(name)


# -------------------------------------------------------------- baselines


def test_flat_baseline_emits_a_hierarchical_output():
    """Baselines must satisfy the same contract so evaluation code is shared."""
    from src.models.baselines import FlatSupervisedBaseline

    torch.manual_seed(0)
    model = FlatSupervisedBaseline(
        model_name="resnet18",  # smallest timm CNN that builds without a download
        num_seed_types=PAPER_NUM_SEED_TYPES,
        num_sub_varieties=PAPER_NUM_SUB_VARIETIES,
        pretrained=False,
    )
    output = model(torch.randn(3, 3, 64, 64))

    assert output.seed_type_logits.shape == (3, PAPER_NUM_SEED_TYPES)
    assert output.sub_logits.shape == (3, PAPER_NUM_SUB_VARIETIES)
    assert output.embedding.shape == (3, PAPER_EMBED_DIM)
    # A degenerate one-expert router keeps the MoE-shaped consumers working.
    assert output.gate_probs.shape == (3, 1)
    assert torch.equal(output.sub_logits, output.sub_margin_logits)


def test_flat_baseline_heads_are_independent():
    """Two heads, not one derived from the other, or KL alignment would be trivially 1."""
    from src.models.baselines import FlatSupervisedBaseline

    torch.manual_seed(0)
    model = FlatSupervisedBaseline(
        model_name="resnet18",
        num_seed_types=PAPER_NUM_SEED_TYPES,
        num_sub_varieties=PAPER_NUM_SUB_VARIETIES,
        pretrained=False,
    )
    assert model.seed_head.weight is not model.sub_head.weight
    assert model.seed_head.out_features == PAPER_NUM_SEED_TYPES
    assert model.sub_head.out_features == PAPER_NUM_SUB_VARIETIES
