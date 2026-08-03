"""End-to-end shapes, dataflow, gradient flow and component toggles.

These tests encode the paper's Section 5 dataflow, in particular the two places
an implementation can silently diverge: what the MoE is routed on (``z``, not the
seed projection) and what the residual adds (``P(p_s)``, not ``P(s)``).

They also pin the revision's contracts: the encoder emits ``z in R^384`` for any
SwinV2 width, the router selects exactly ``K = 2`` experts, and each of the four
architectural toggles removes its block without breaking the output contract.
"""

from __future__ import annotations

import math

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
        # Pooled by default: most invariants here are about the coarse-to-fine
        # dataflow, and a [batch, 384] output keeps them readable. The
        # grid-routing invariants live in their own section below.
        "token_mode": "pooled",
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
    """P must be applied to p_s; feeding it s would give a different vector.

    ``projected_seed`` is the residual actually added, i.e. LayerScale-gated, so
    the comparison runs through the same gain.
    """
    output = hierarchical_model(batch["features"])
    gate = hierarchical_model.residual_scale
    from_probs = gate(hierarchical_model.seed_projection(output.seed_type_probs))
    assert torch.allclose(output.projected_seed, from_probs, atol=1e-6)

    from_logits = gate(hierarchical_model.seed_projection(output.seed_type_logits))
    assert not torch.allclose(output.projected_seed, from_logits, atol=1e-8)


def test_layer_scale_starts_the_residual_near_zero(hierarchical_model, batch):
    """LayerScale replaces the loss-side attempt to control the Eq. 9 residual.

    A cosine penalty on ``(h + P(p_s), h)`` is minimised by ``P(p_s) = 0`` -- it
    rewards deleting the connection it regularises. A learned per-channel gain
    initialised at 1e-4 gives the same "start small, grow if it helps" behaviour
    with no such fixed point, and it is trainable so the model can undo it.
    """
    gain = hierarchical_model.residual_scale.gamma
    assert gain.requires_grad
    assert torch.allclose(gain, torch.full_like(gain, 1e-4))

    output = hierarchical_model(batch["features"])
    ratio = output.projected_seed.norm(dim=-1) / output.moe_features.norm(dim=-1)
    assert (ratio < 0.01).all(), "the gated residual should start negligible"


def test_moe_is_routed_on_the_dino_embedding(hierarchical_model, batch):
    """Eq. 8 routes on z. Routing on P(p_s) would starve the experts of image content."""
    hierarchical_model.eval()
    with torch.no_grad():
        output = hierarchical_model(batch["features"])
        direct = hierarchical_model.moe(
            output.embedding, gate_condition=output.seed_type_probs
        )
    assert torch.allclose(output.gate_probs, direct.gate_probs, atol=1e-6)
    assert torch.allclose(output.moe_features, direct.features, atol=1e-6)


def test_experts_see_the_image_even_when_the_gate_sees_the_coarse_prediction(batch):
    """``gate_conditioning`` must change the *router*, never the experts' input.

    Making the MoE hierarchical means the gate is informed by ``p_s``. If that
    conditioning reached the experts instead, they would stop seeing the image --
    the exact defect an earlier revision had, where the MoE was routed on
    ``P(seed_logits)``.
    """
    model = make_model(gate_conditioning=True).eval()
    with torch.no_grad():
        output = model(batch["features"])
        # Same z, deliberately wrong coarse posterior: only the routing may move.
        scrambled = model.moe(
            output.embedding, gate_condition=output.seed_type_probs.flip(0)
        )
    assert not torch.allclose(output.gate_probs, scrambled.gate_probs, atol=1e-6)
    assert model.moe.experts[0].mlp[0].in_features == PAPER_EMBED_DIM


def test_gate_conditioning_does_not_backpropagate_into_the_coarse_head(batch):
    """The router is *informed* by p_s; it must not reshape it.

    Without the detach the router would acquire an incentive to make the coarse
    head easier to route on, which is an F-14-style failure one level down.
    """
    model = make_model(gate_conditioning=True)
    output = model(batch["features"])
    # A loss that touches only the routing: any gradient reaching the seed
    # classifier could only have come through the gate condition.
    output.gate_probs.square().sum().backward()
    assert model.seed_type_classifier.fc2.weight.grad is None


def test_cross_attention_uses_refined_query_and_raw_moe_key_value(hierarchical_model, batch):
    """Eq. 11: Q = h', K = V = h."""
    hierarchical_model.eval()
    with torch.no_grad():
        output = hierarchical_model(batch["features"])
        expected = hierarchical_model.cross_attention(
            query=output.refined_features.unsqueeze(1),
            key=output.moe_features.unsqueeze(1),
            value=output.moe_features.unsqueeze(1),
        ).features.squeeze(1)  # pooled mode: one token, so the squeeze is exact
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
        token_mode="pooled",
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


def test_rejects_malformed_features(hierarchical_model):
    """4-D is always wrong; a token grid is wrong specifically for a pooled head."""
    with pytest.raises(ValueError, match=r"\[batch, feature_dim\]"):
        hierarchical_model(torch.randn(4, 7, 3, PAPER_EMBED_DIM))

    # Caught at the head's boundary rather than several blocks later, where it
    # would surface as an opaque complaint from the affine attention path.
    with pytest.raises(ValueError, match="token_mode='pooled'"):
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
        "moe_router_z_loss",
        "cosine_loss",
        "residual_magnitude_loss",
        "moe_dead_experts",
        "task_weight/seed",
        "task_weight/arcface",
        "task_weight/kl",
    }
    assert all(isinstance(value, float) for value in parts.values())


def test_criterion_holds_no_parameters_under_fixed_weighting(combined_loss):
    """ArcFace centres live in the model, so model_state_dict alone is enough."""
    assert list(combined_loss.parameters()) == []


def test_uncertainty_weighting_adds_exactly_three_learnable_scalars(subvariety_to_seed_type):
    """The only parameters the criterion may ever own, and the trainer must see them.

    ``build_optimizer`` is handed the criterion for this reason: leaving these out
    would pin the task weights at their initial values while the logs reported
    them as learned.
    """
    from src.losses.hierarchical import TASK_TERMS, CombinedHierarchicalLoss

    criterion = CombinedHierarchicalLoss(
        num_seed_types=PAPER_NUM_SEED_TYPES,
        num_sub_varieties=PAPER_NUM_SUB_VARIETIES,
        subvariety_to_seed_type=subvariety_to_seed_type,
        weighting_mode="uncertainty",
    )
    parameters = list(criterion.parameters())
    assert len(parameters) == 1
    assert parameters[0].numel() == len(TASK_TERMS) == 3
    assert set(criterion.uncertainty.weights()) == set(TASK_TERMS)


def test_loss_weights_scale_the_total(hierarchical_model, batch, subvariety_to_seed_type):
    """Zeroing every weight but one must leave exactly that component."""
    from src.losses.hierarchical import CombinedHierarchicalLoss

    output = hierarchical_model(batch["features"], batch["sub_labels"])
    seed_only = CombinedHierarchicalLoss(
        num_seed_types=PAPER_NUM_SEED_TYPES,
        num_sub_varieties=PAPER_NUM_SUB_VARIETIES,
        subvariety_to_seed_type=subvariety_to_seed_type,
        lambda_seed=1.0, lambda_arcface=0.0, lambda_kl=0.0,
        lambda_moe_load=0.0, lambda_moe_sparsity=0.0, lambda_moe_z=0.0,
        lambda_cosine=0.0, lambda_residual=0.0, lambda_sub_ce=0.0,
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


def test_wo_residual_removes_only_equation_nine(subvariety_to_seed_type):
    """``wo_residual`` must be a one-factor ablation, and it was not.

    Under the submitted ``cosine_mode="residual"`` the compactness term was
    ``1 - cos(h + P(p_s), h)``, so ``use_residual=False`` set ``P(p_s) = 0`` and
    drove it to **exactly 0 for every sample, for the whole run**. The variant
    therefore removed Eq. 9 *and* the paper's Section-1 contribution in one
    toggle, and the measured gap could not attribute to either.

    With the default ``intra_class`` compactness the term is a property of the
    ArcFace embedding, so it survives the toggle. What legitimately vanishes is
    the residual *magnitude* hinge, which has no residual left to bound.
    """
    from src.losses.hierarchical import CombinedHierarchicalLoss

    def breakdown_for(cosine_mode: str):
        model = make_model(use_residual=False)
        output = model(torch.randn(8, PAPER_EMBED_DIM))
        criterion = CombinedHierarchicalLoss(
            num_seed_types=PAPER_NUM_SEED_TYPES,
            num_sub_varieties=PAPER_NUM_SUB_VARIETIES,
            subvariety_to_seed_type=subvariety_to_seed_type,
            cosine_mode=cosine_mode,
        )
        return criterion(output, torch.randint(0, 4, (8,)), torch.randint(0, 27, (8,)))

    revised = breakdown_for("intra_class")
    assert revised.cosine.item() > 1e-4, "compactness must survive the residual ablation"
    assert revised.residual.item() == pytest.approx(0.0, abs=1e-8)

    # The submitted formulation, kept as an ablation axis, still collapses --
    # which is the point of retaining it: the confound is reproducible.
    assert breakdown_for("residual").cosine.item() == pytest.approx(0.0, abs=1e-5)


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
    flags = model.component_flags()
    assert {key: flags[key] for key in ("use_moe", "use_arcface", "use_residual", "use_cross_attention")} == {
        "use_moe": False,
        "use_arcface": False,
        "use_residual": True,
        "use_cross_attention": False,
    }
    # `use_moe=false` forces gate conditioning off: a dense block has no router
    # to condition, and reporting it as on would misdescribe the run.
    assert flags["gate_conditioning"] is False
    assert flags["num_experts"] == 1 and flags["top_k"] == 1


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


# --------------------------------------------------------- grid-mode routing


def test_grid_mode_routes_every_token_and_pools_after_the_head(grid_model, token_batch):
    """The change that makes the attention modules non-degenerate.

    Pooled routing fills ``batch x K`` slots per step -- 32 at the configured
    batch size, which is far too few to estimate a 6-bin routing distribution.
    Grid routing fills ``batch x tokens x K``, and pooling moves to *after* the
    head so the MoE and the cross-attention see spatial structure rather than one
    averaged vector.
    """
    output = grid_model(token_batch["features"], token_batch["sub_labels"])
    size, tokens = token_batch["size"], token_batch["tokens"]

    assert output.embedding.shape == (size, tokens, PAPER_EMBED_DIM)
    assert output.moe_features.shape == (size, tokens, PAPER_EMBED_DIM)
    assert output.refined_features.shape == (size, tokens, PAPER_EMBED_DIM)
    assert output.tokens_per_sample == tokens

    # Routing happens per token ...
    assert output.gate_probs.shape == (size * tokens, PAPER_NUM_EXPERTS)
    assert output.top_k_indices.shape == (size * tokens, REVISED_TOP_K)
    # ... and pooling happens once, at the end.
    assert output.attended_features.shape == (size, PAPER_EMBED_DIM)
    assert output.sub_logits.shape == (size, PAPER_NUM_SUB_VARIETIES)


def test_grid_mode_produces_a_real_attention_map(grid_model, token_batch):
    """``attn_weights`` becomes a figure instead of a constant.

    Over a length-1 sequence the map is identically 1.0, so any attention-map
    visualisation drawn from the pooled path showed nothing at all.
    """
    output = grid_model(token_batch["features"], need_attn_weights=True)
    weights = output.attn_weights
    assert weights is not None
    assert weights.shape[-1] == token_batch["tokens"]
    assert not torch.allclose(weights, torch.ones_like(weights))

    pooled = make_model(token_mode="pooled")
    assert pooled(torch.randn(4, PAPER_EMBED_DIM), need_attn_weights=True).attn_weights is None


def test_grid_mode_fills_far_more_routing_slots_than_pooled(token_batch):
    """The load statistic's sample size, stated as a ratio rather than a claim."""
    grid = make_model(token_mode="grid")(token_batch["features"])
    pooled = make_model(token_mode="pooled")(token_batch["features"].mean(dim=1))
    assert grid.top_k_indices.numel() == pooled.top_k_indices.numel() * token_batch["tokens"]


# ---------------------------------------------------------- ArcFace schedules


def test_margin_warms_up_from_zero_and_reaches_full_strength(batch):
    """Full margin from step 0 is a documented convergence hazard here.

    CurricularFace reports outright divergence for a small backbone at m = 0.5 on
    a small dataset where m = 0.45 converges. The ramp is applied by the trainer
    once per epoch; at scale 0 the head *is* a cosine-softmax.
    """
    model = make_model()
    model.set_margin_scale(0.0)
    zero_margin = model(batch["features"], batch["sub_labels"])
    assert torch.equal(zero_margin.sub_logits, zero_margin.sub_margin_logits)

    model.set_margin_scale(1.0)
    full_margin = model(batch["features"], batch["sub_labels"])
    assert not torch.equal(full_margin.sub_logits, full_margin.sub_margin_logits)
    assert model.arcface.effective_margin == pytest.approx(model.arcface.margin)


def test_margin_schedule_is_monotone_and_terminates_at_one():
    from src.trainers.moe_finetune import margin_schedule

    values = [margin_schedule(epoch, total_epochs=100, warmup_fraction=0.15) for epoch in range(1, 101)]
    assert values[0] == pytest.approx(0.0)
    assert all(b >= a for a, b in zip(values, values[1:]))
    assert values[-1] == pytest.approx(1.0)


def test_router_noise_schedule_reaches_exactly_zero():
    from src.trainers.moe_finetune import router_noise_schedule

    values = [
        router_noise_schedule(epoch, total_epochs=100, initial=0.3, fraction=0.3)
        for epoch in range(1, 101)
    ]
    assert values[0] == pytest.approx(0.3)
    assert all(b <= a for a, b in zip(values, values[1:]))
    assert values[-1] == 0.0, "the deployed routing must be the routing that was measured"


def test_arcface_scale_defaults_to_the_adacos_value():
    """``s = 30`` is ArcFace's face-recognition value; C = 27 needs 4.61."""
    model = make_model()
    assert model.arcface.scale == pytest.approx(math.sqrt(2.0) * math.log(26), abs=1e-4)


def test_normface_head_removes_the_margin_and_nothing_else():
    """The true single-factor margin control.

    Swapping to a plain Linear changes the margin AND the embedding L2-norm AND
    the centre L2-norm AND the logit scale -- four factors, so the gap it measures
    is not attributable to the margin.
    """
    normface = make_model(sub_head_variant="normface")
    linear = make_model(sub_head_variant="linear")

    features = torch.randn(6, PAPER_EMBED_DIM)
    labels = torch.randint(0, PAPER_NUM_SUB_VARIETIES, (6,))

    normface_output = normface(features, labels)
    assert torch.equal(normface_output.sub_logits, normface_output.sub_margin_logits)
    # Normalised on both sides, so logits are bounded by the scale.
    assert normface_output.sub_logits.abs().max() <= normface.arcface.scale + 1e-4
    # A plain linear head has no such bound: that is one of the three extra
    # factors the submitted `wo_arcface` variant was silently changing.
    assert not hasattr(linear.arcface, "scale")


# ------------------------------------------------------------- FiLM fusion


def test_film_recovers_the_additive_form_at_initialisation(batch):
    """FiLM is a strict superset: ``gamma = 1`` is exactly "no modulation".

    Initialising it there means the ``additive`` and ``film`` variants start from
    the same function, so the ablation between them measures the extra capacity
    rather than a different initialisation.
    """
    model = make_model(fusion_mode="film").eval()
    with torch.no_grad():
        output = model(batch["features"])
    assert torch.allclose(output.refined_features, output.moe_features, atol=1e-6)
    assert torch.allclose(output.projected_seed, torch.zeros_like(output.projected_seed), atol=1e-6)


def test_film_conditions_on_the_hidden_state_not_the_saturating_posterior(batch):
    """Eq. 9 degenerates to a 4-entry codebook once the coarse head fits.

    ``p_s`` lives on the 3-simplex and saturates to one-hot early -- there are
    only 4 classes -- so ``P(p_s)`` converges to one of four fixed vectors, and
    the softmax Jacobian ``diag(p) - pp^T`` takes the fine branch's gradient into
    the coarse head to zero with it. FiLM reads the pre-softmax hidden state,
    which does neither.
    """
    model = make_model(fusion_mode="film")
    assert model.film.condition_dim == model.seed_type_classifier.hidden_dim
    assert model.film.condition_dim > PAPER_NUM_SEED_TYPES

    output = model(batch["features"], batch["sub_labels"])
    assert output.seed_hidden is not None
    assert output.seed_hidden.shape == (batch["size"], model.seed_type_classifier.hidden_dim)

    # The modulation MLP starts at exactly zero so the fusion begins as the
    # identity; any training at all moves it off zero. Nudge it to stand in for
    # that, then check the fine branch informs the coarse one through the fusion.
    with torch.no_grad():
        model.film.project[-1].weight.normal_(0.0, 0.02)
    model.zero_grad()
    model(batch["features"], batch["sub_labels"]).sub_logits.sum().backward()
    assert model.seed_type_classifier.fc1.weight.grad.abs().sum() > 0


def test_additive_fusion_gradient_vanishes_as_the_coarse_head_saturates(batch):
    """Why FiLM conditions on the hidden state: the softmax Jacobian dies.

    ``d p_s / d s = diag(p) - p p^T``, whose norm goes to 0 as ``p`` approaches
    one-hot. With only four coarse classes the seed head fits early and hard, so
    after the transient the sub-variety branch stops informing the seed branch
    through Eq. 9 entirely -- and the KL term, which also flows through softmax
    outputs, does not rescue it.
    """
    model = make_model(fusion_mode="additive")

    def residual_gradient(logit_scale: float) -> float:
        logits = torch.zeros(batch["size"], PAPER_NUM_SEED_TYPES, requires_grad=True)
        with torch.no_grad():
            logits[:, 0] = logit_scale
        probs = F.softmax(logits, dim=-1)
        model.zero_grad()
        model.seed_projection(probs).sum().backward()
        return float(logits.grad.abs().sum())

    unsaturated = residual_gradient(0.5)
    saturated = residual_gradient(30.0)
    assert saturated < unsaturated * 1e-3, "a confident coarse head must starve the residual path"
