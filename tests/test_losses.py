"""Loss bounds, directions, and gradient behaviour (paper Eqs. 1-3, 7, 10, 13)."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from src.losses.cosine import intra_class_cosine_loss, residual_cosine_loss
from src.losses.dino import CustomDINOLoss
from src.losses.hierarchical import (
    aggregate_sub_log_probs,
    build_subvariety_seed_mapping,
    hierarchical_kl_loss,
    seed_type_loss,
)
from src.losses.moe import (
    dispatch_fraction,
    expert_utilization,
    l1_sparsity_loss,
    load_balancing_loss,
    router_z_loss,
    switch_load_balancing_loss,
)
from tests.conftest import (
    SUBMITTED_CENTER_MOMENTUM,
    PAPER_EMBED_DIM,
    PAPER_NUM_EXPERTS,
    PAPER_NUM_SEED_TYPES,
    PAPER_NUM_SUB_VARIETIES,
    SUBMITTED_TEACHER_TEMP,
    REVISED_TOP_K,
    SUBMITTED_WARMUP_EPOCHS,
    SUBMITTED_WARMUP_TEACHER_TEMP,
)


# ------------------------------------------------------- MoE regularisation


def _entropy_load(gate):
    """The submitted soft-gate form, retained as an ablation axis."""
    return load_balancing_loss(gate, mode="entropy")


def test_entropy_load_balancing_is_minimised_by_uniform_utilization():
    """Uniform routing must reach the lower bound of -1 under normalisation."""
    uniform = torch.full((16, PAPER_NUM_EXPERTS), 1.0 / PAPER_NUM_EXPERTS)
    assert _entropy_load(uniform).item() == pytest.approx(-1.0, abs=1e-5)


def test_entropy_load_balancing_is_maximised_by_total_collapse():
    """All mass on one expert gives zero entropy, the loss's upper bound."""
    collapsed = torch.zeros(16, PAPER_NUM_EXPERTS)
    collapsed[:, 0] = 1.0
    assert _entropy_load(collapsed).item() == pytest.approx(0.0, abs=1e-4)


def test_entropy_load_balancing_is_bounded_in_minus_one_to_zero():
    for _ in range(20):
        gate = F.softmax(torch.randn(32, PAPER_NUM_EXPERTS) * 3, dim=-1)
        value = _entropy_load(gate).item()
        assert -1.0 - 1e-5 <= value <= 1e-5


def test_entropy_load_balancing_prefers_balance_over_collapse():
    balanced = torch.full((8, PAPER_NUM_EXPERTS), 1.0 / PAPER_NUM_EXPERTS)
    skewed = torch.zeros(8, PAPER_NUM_EXPERTS)
    skewed[:, 0] = 0.9
    skewed[:, 1:] = 0.1 / (PAPER_NUM_EXPERTS - 1)
    assert _entropy_load(balanced) < _entropy_load(skewed)


def test_unnormalized_entropy_load_balancing_matches_negative_entropy():
    gate = F.softmax(torch.randn(16, PAPER_NUM_EXPERTS), dim=-1)
    utilization = expert_utilization(gate)
    expected = torch.sum(utilization * torch.log(utilization))
    assert load_balancing_loss(gate, mode="entropy", normalize=False).item() == pytest.approx(
        expected.item(), abs=1e-6
    )


# --- The counterexample that motivated replacing the entropy form -------------

#: The router distribution from the audit: identical for every sample, and
#: catastrophic under Top-2 despite scoring 92 % of "perfect balance".
COLLAPSING_GATE = (0.30, 0.30, 0.10, 0.10, 0.10, 0.10)


def test_entropy_load_balancing_scores_a_collapsed_router_as_nearly_perfect():
    """The failure the revision exists to fix, reproduced to four decimal places.

    ``-sum u ln u = 1.64342`` and ``ln 6 = 1.79176``, so the entropy form scores
    ``-0.9172`` -- 91.7 % of the way to its own optimum. Meanwhile ``topk(G, 2)``
    picks experts {0, 1} for *every* sample, so four of six experts receive no
    gradient at all. The regulariser cannot see the dispatch it is supposed to
    balance.
    """
    gate = torch.tensor([list(COLLAPSING_GATE)]).repeat(32, 1)
    _, indices = torch.topk(gate, REVISED_TOP_K, dim=-1)

    assert load_balancing_loss(gate, indices, mode="entropy").item() == pytest.approx(
        -0.9172, abs=1e-3
    )
    # ... and the hard dispatch is maximally imbalanced.
    fractions = dispatch_fraction(indices, PAPER_NUM_EXPERTS)
    assert torch.allclose(fractions, torch.tensor([0.5, 0.5, 0.0, 0.0, 0.0, 0.0]), atol=1e-6)


def test_switch_load_balancing_sees_the_collapse_the_entropy_form_misses():
    """On the same gate, the dispatch-aware form is far from its optimum.

    ``E * sum f_i P_i`` with ``f = (.5, .5, 0, 0, 0, 0)`` and ``P = G`` gives
    ``6 * (0.5*0.3 + 0.5*0.3) = 1.8``, i.e. 0.8 above the uniform-routing floor
    of 1. The gradient flows through ``P``, pushing down the router probability
    of exactly the over-dispatched experts.
    """
    gate = torch.tensor([list(COLLAPSING_GATE)]).repeat(32, 1)
    _, indices = torch.topk(gate, REVISED_TOP_K, dim=-1)

    collapsed = load_balancing_loss(gate, indices, mode="switch").item()
    assert collapsed == pytest.approx(0.8, abs=1e-4)

    uniform = torch.full((32, PAPER_NUM_EXPERTS), 1.0 / PAPER_NUM_EXPERTS)
    _, uniform_indices = torch.topk(uniform, REVISED_TOP_K, dim=-1)
    balanced = load_balancing_loss(uniform, uniform_indices, mode="switch").item()
    assert balanced < collapsed


def test_switch_load_balancing_gradient_pushes_down_over_dispatched_experts():
    """``f`` is a coefficient; ``P`` carries the gradient. That asymmetry is the mechanism."""
    logits = torch.zeros(16, PAPER_NUM_EXPERTS, requires_grad=True)
    with torch.no_grad():
        logits[:, 0] = 3.0
        logits[:, 1] = 2.0
    gate = F.softmax(logits, dim=-1)
    _, indices = torch.topk(gate, REVISED_TOP_K, dim=-1)
    load_balancing_loss(gate, indices, mode="switch").backward()

    grad = logits.grad.sum(dim=0)
    # Reducing the loss means reducing the router probability of the two experts
    # taking all the traffic, so their logits must have positive gradient.
    assert grad[0] > 0 and grad[1] > 0
    assert grad[2:].max() < 0


def test_router_z_loss_grows_with_router_logit_magnitude():
    """The term ST-MoE adds to stop router logits running away."""
    small = router_z_loss(torch.randn(16, PAPER_NUM_EXPERTS) * 0.1)
    large = router_z_loss(torch.randn(16, PAPER_NUM_EXPERTS) * 0.1 + 10.0)
    assert large > small
    assert router_z_loss(torch.zeros(16, PAPER_NUM_EXPERTS)).item() == pytest.approx(
        math.log(PAPER_NUM_EXPERTS) ** 2, abs=1e-5
    )


def test_l1_sparsity_is_zero_when_all_mass_is_inside_top_k():
    """Bound check: with all mass on the selected experts the penalty vanishes."""
    gate = torch.zeros(4, PAPER_NUM_EXPERTS)
    gate[:, :REVISED_TOP_K] = 1.0 / REVISED_TOP_K
    indices = torch.arange(REVISED_TOP_K).expand(4, REVISED_TOP_K)
    assert l1_sparsity_loss(gate, indices).item() == pytest.approx(0.0, abs=1e-6)


def test_l1_sparsity_equals_the_discarded_routing_mass():
    gate = F.softmax(torch.randn(8, PAPER_NUM_EXPERTS), dim=-1)
    weights, indices = torch.topk(gate, REVISED_TOP_K, dim=-1)
    expected = (1.0 - weights.sum(dim=-1)).mean()
    assert l1_sparsity_loss(gate, indices).item() == pytest.approx(expected.item(), abs=1e-6)


def test_l1_sparsity_is_bounded_in_zero_to_one():
    for _ in range(20):
        gate = F.softmax(torch.randn(16, PAPER_NUM_EXPERTS), dim=-1)
        _, indices = torch.topk(gate, REVISED_TOP_K, dim=-1)
        value = l1_sparsity_loss(gate, indices).item()
        assert -1e-6 <= value <= 1.0 + 1e-6


def test_l1_sparsity_rejects_unknown_mode():
    gate = F.softmax(torch.randn(4, 6), dim=-1)
    _, indices = torch.topk(gate, 4, dim=-1)
    with pytest.raises(ValueError, match="mode"):
        l1_sparsity_loss(gate, indices, mode="nonsense")


# -------------------------------------------------------------- hierarchy KL


def test_mapping_matrix_is_one_hot_over_seed_types(subvariety_to_seed_type):
    mapping = build_subvariety_seed_mapping(
        num_sub_varieties=PAPER_NUM_SUB_VARIETIES,
        num_seed_types=PAPER_NUM_SEED_TYPES,
        subvariety_to_seed_type=subvariety_to_seed_type,
    )
    assert mapping.shape == (PAPER_NUM_SUB_VARIETIES, PAPER_NUM_SEED_TYPES)
    assert torch.all(mapping.sum(dim=1) == 1.0)
    # Column sums recover the paper's 8 / 3 / 13 / 3 split.
    assert mapping.sum(dim=0).tolist() == [8.0, 3.0, 13.0, 3.0]


def test_mapping_matrix_from_counts_matches_explicit_mapping(subvariety_to_seed_type):
    from_counts = build_subvariety_seed_mapping(
        num_sub_varieties=PAPER_NUM_SUB_VARIETIES,
        num_seed_types=PAPER_NUM_SEED_TYPES,
        subvarieties_per_seed_type=[8, 3, 13, 3],
    )
    explicit = build_subvariety_seed_mapping(
        num_sub_varieties=PAPER_NUM_SUB_VARIETIES,
        num_seed_types=PAPER_NUM_SEED_TYPES,
        subvariety_to_seed_type=subvariety_to_seed_type,
    )
    assert torch.equal(from_counts, explicit)


def test_mapping_matrix_rejects_inconsistent_input():
    with pytest.raises(ValueError):
        build_subvariety_seed_mapping(27, 4, subvariety_to_seed_type=[0, 1, 2])
    with pytest.raises(ValueError):
        build_subvariety_seed_mapping(27, 4, subvarieties_per_seed_type=[8, 3, 13])
    with pytest.raises(ValueError, match="outside"):
        build_subvariety_seed_mapping(2, 2, subvariety_to_seed_type=[0, 9])


def test_kl_is_zero_when_the_two_levels_agree(subvariety_to_seed_type):
    """Perfect hierarchical consistency must drive Eq. 10 to zero."""
    mapping = build_subvariety_seed_mapping(
        PAPER_NUM_SUB_VARIETIES, PAPER_NUM_SEED_TYPES, subvariety_to_seed_type
    )
    # Sub-variety mass concentrated on index 0 (a Millet sub-variety, parent 0),
    # and the seed head agreeing on seed type 0.
    sub_logits = torch.full((8, PAPER_NUM_SUB_VARIETIES), -50.0)
    sub_logits[:, 0] = 50.0
    seed_logits = torch.full((8, PAPER_NUM_SEED_TYPES), -50.0)
    seed_logits[:, 0] = 50.0

    assert hierarchical_kl_loss(seed_logits, sub_logits, mapping).item() == pytest.approx(0.0, abs=1e-4)


def test_kl_is_positive_when_the_two_levels_disagree(subvariety_to_seed_type):
    mapping = build_subvariety_seed_mapping(
        PAPER_NUM_SUB_VARIETIES, PAPER_NUM_SEED_TYPES, subvariety_to_seed_type
    )
    sub_logits = torch.full((8, PAPER_NUM_SUB_VARIETIES), -50.0)
    sub_logits[:, 0] = 50.0        # parent seed type 0
    seed_logits = torch.full((8, PAPER_NUM_SEED_TYPES), -50.0)
    seed_logits[:, 2] = 50.0       # but the seed head says type 2

    assert hierarchical_kl_loss(seed_logits, sub_logits, mapping).item() > 1.0


def test_kl_direction_is_seed_given_sub(subvariety_to_seed_type):
    """Eq. 10 is D_KL(P_seed || P_sub), not the reverse."""
    mapping = build_subvariety_seed_mapping(
        PAPER_NUM_SUB_VARIETIES, PAPER_NUM_SEED_TYPES, subvariety_to_seed_type
    )
    seed_logits = torch.randn(8, PAPER_NUM_SEED_TYPES)
    sub_logits = torch.randn(8, PAPER_NUM_SUB_VARIETIES)

    seed_probs = F.softmax(seed_logits, dim=-1)
    aggregated = F.softmax(sub_logits, dim=-1) @ mapping
    expected = (seed_probs * (seed_probs.clamp_min(1e-8).log() - aggregated.clamp_min(1e-8).log())).sum(-1).mean()

    actual = hierarchical_kl_loss(seed_logits, sub_logits, mapping)
    assert actual.item() == pytest.approx(expected.item(), abs=1e-5)


def test_kl_detach_stops_the_gradient_into_the_seed_branch(subvariety_to_seed_type):
    mapping = build_subvariety_seed_mapping(
        PAPER_NUM_SUB_VARIETIES, PAPER_NUM_SEED_TYPES, subvariety_to_seed_type
    )
    seed_logits = torch.randn(8, PAPER_NUM_SEED_TYPES, requires_grad=True)
    sub_logits = torch.randn(8, PAPER_NUM_SUB_VARIETIES, requires_grad=True)

    hierarchical_kl_loss(seed_logits, sub_logits, mapping, detach_seed_target=True).backward()
    assert seed_logits.grad is None or seed_logits.grad.abs().sum() == 0
    assert sub_logits.grad.abs().sum() > 0


# ------------------------------------------------------------------- cosine


def test_residual_cosine_is_zero_for_identical_vectors():
    x = torch.randn(8, PAPER_EMBED_DIM)
    assert residual_cosine_loss(x, x).item() == pytest.approx(0.0, abs=1e-6)


def test_residual_cosine_is_two_for_opposed_vectors():
    x = torch.randn(8, PAPER_EMBED_DIM)
    assert residual_cosine_loss(x, -x).item() == pytest.approx(2.0, abs=1e-5)


def test_residual_cosine_is_bounded_in_zero_to_two():
    for _ in range(20):
        a = torch.randn(16, 64)
        b = torch.randn(16, 64)
        assert -1e-6 <= residual_cosine_loss(a, b).item() <= 2.0 + 1e-6


def test_intra_class_cosine_is_zero_for_collapsed_classes():
    """Identical embeddings within each class give perfect compactness."""
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    embeddings = torch.zeros(6, 4)
    embeddings[labels == 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    embeddings[labels == 1] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    embeddings[labels == 2] = torch.tensor([0.0, 0.0, 1.0, 0.0])
    assert intra_class_cosine_loss(embeddings, labels).item() == pytest.approx(0.0, abs=1e-5)


def test_intra_class_cosine_penalises_spread():
    labels = torch.tensor([0, 0, 0, 0])
    tight = torch.tensor([[1.0, 0.02], [1.0, -0.02], [1.0, 0.01], [1.0, 0.0]])
    spread = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    assert intra_class_cosine_loss(tight, labels) < intra_class_cosine_loss(spread, labels)


def test_intra_class_cosine_rejects_mismatched_batch():
    with pytest.raises(ValueError, match="batch size"):
        intra_class_cosine_loss(torch.randn(4, 8), torch.tensor([0, 1]))


# --------------------------------------------------------------- seed / DINO


def test_seed_type_loss_matches_cross_entropy():
    logits = torch.randn(8, PAPER_NUM_SEED_TYPES)
    labels = torch.randint(0, PAPER_NUM_SEED_TYPES, (8,))
    assert seed_type_loss(logits, labels).item() == pytest.approx(
        F.cross_entropy(logits, labels).item(), abs=1e-6
    )


def test_dino_temperature_schedule_follows_equation_2():
    """Table 1: 0.02 -> 0.04 linearly over the first 5 epochs, then constant."""
    loss = CustomDINOLoss(
        out_dim=16,
        num_crops=6,
        warmup_teacher_temp=SUBMITTED_WARMUP_TEACHER_TEMP,
        teacher_temp=SUBMITTED_TEACHER_TEMP,
        warmup_teacher_temp_epochs=SUBMITTED_WARMUP_EPOCHS,
        num_epochs=20,
    )
    assert loss.teacher_temperature(0) == pytest.approx(SUBMITTED_WARMUP_TEACHER_TEMP)
    assert loss.teacher_temperature(4) == pytest.approx(SUBMITTED_TEACHER_TEMP)
    assert loss.teacher_temperature(5) == pytest.approx(SUBMITTED_TEACHER_TEMP)
    assert loss.teacher_temperature(19) == pytest.approx(SUBMITTED_TEACHER_TEMP)
    # Strictly increasing through the warmup window.
    warmup = [loss.teacher_temperature(e) for e in range(SUBMITTED_WARMUP_EPOCHS)]
    assert all(b > a for a, b in zip(warmup, warmup[1:]))


def test_dino_loss_is_non_negative_and_finite():
    loss_fn = CustomDINOLoss(
        out_dim=32, num_crops=6, warmup_teacher_temp=0.02,
        teacher_temp=0.04, warmup_teacher_temp_epochs=5, num_epochs=10,
    )
    student = [torch.randn(4, 32) for _ in range(6)]
    teacher = [torch.randn(4, 32) for _ in range(2)]
    value = loss_fn(student, teacher, epoch=0)
    assert torch.isfinite(value)
    assert value.item() >= 0.0


def test_dino_loss_counts_only_cross_view_pairs():
    """2 teacher views x 6 student views, minus 2 same-view pairs, is 10 terms."""
    loss_fn = CustomDINOLoss(
        out_dim=8, num_crops=6, warmup_teacher_temp=0.04,
        teacher_temp=0.04, warmup_teacher_temp_epochs=0, num_epochs=1,
        # This test recomputes the softmax path by hand, so it pins the EMA
        # variant; the Sinkhorn default has its own tests below.
        centering="ema", lambda_koleo=0.0,
    )
    student = torch.cat([torch.randn(2, 8) for _ in range(6)], dim=0)
    teacher = torch.cat([torch.randn(2, 8) for _ in range(2)], dim=0)

    manual_terms = 0
    student_chunks = (student / loss_fn.student_temp).chunk(6)
    teacher_probs = F.softmax(teacher / 0.04, dim=-1).chunk(2)
    total = 0.0
    for t_index, q in enumerate(teacher_probs):
        for s_index, v in enumerate(student_chunks):
            if s_index == t_index:
                continue
            total += torch.sum(-q * F.log_softmax(v, dim=-1), dim=-1).mean().item()
            manual_terms += 1
    assert manual_terms == 10

    actual = loss_fn.compute_dino_loss(student, teacher, epoch=0)
    assert actual.item() == pytest.approx(total / manual_terms, abs=1e-5)


def test_dino_center_update_follows_equation_3():
    """C_t = m * C_{t-1} + (1 - m) * qbar with m = 0.9."""
    loss_fn = CustomDINOLoss(
        out_dim=4, num_crops=6, warmup_teacher_temp=0.04, teacher_temp=0.04,
        warmup_teacher_temp_epochs=0, num_epochs=1,
        center_momentum=SUBMITTED_CENTER_MOMENTUM, centering="ema",
    )
    assert torch.allclose(loss_fn.center, torch.zeros(1, 4))

    teacher = torch.ones(8, 4) * 2.0
    loss_fn.update_center(teacher)
    expected = SUBMITTED_CENTER_MOMENTUM * 0.0 + (1 - SUBMITTED_CENTER_MOMENTUM) * 2.0
    assert torch.allclose(loss_fn.center, torch.full((1, 4), expected), atol=1e-6)


def test_dino_loss_rejects_degenerate_crop_counts():
    with pytest.raises(ValueError, match="num_crops"):
        CustomDINOLoss(
            out_dim=4, num_crops=1, warmup_teacher_temp=0.02,
            teacher_temp=0.04, warmup_teacher_temp_epochs=5, num_epochs=10,
        )


def test_dino_loss_lower_bound_is_the_teacher_entropy():
    """A student matching the teacher exactly attains the teacher's entropy."""
    loss_fn = CustomDINOLoss(
        out_dim=8, num_crops=2, warmup_teacher_temp=1.0, teacher_temp=1.0,
        warmup_teacher_temp_epochs=0, num_epochs=1, student_temp=1.0,
        centering="ema", lambda_koleo=0.0,
    )
    logits = torch.randn(4, 8)
    both_views = torch.cat([logits, logits], dim=0)
    value = loss_fn.compute_dino_loss(both_views, both_views, epoch=0)

    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * probs.log()).sum(dim=-1).mean()
    assert value.item() == pytest.approx(entropy.item(), abs=1e-5)
    assert value.item() <= math.log(8) + 1e-5


# ------------------------------------------------------- ArcFace (Eq. 13)


def test_arcface_loss_is_a_non_negative_cross_entropy():
    from src.losses.arcface import arcface_loss

    torch.manual_seed(0)
    logits = torch.randn(16, PAPER_NUM_SUB_VARIETIES) * 10
    labels = torch.randint(0, PAPER_NUM_SUB_VARIETIES, (16,))
    value = arcface_loss(logits, labels)

    assert value.item() >= 0.0
    assert torch.isfinite(value)
    assert value.item() == pytest.approx(F.cross_entropy(logits, labels).item(), abs=1e-6)


def test_arcface_margin_never_decreases_the_loss():
    """The margin exists to make the target harder, so it cannot help the objective."""
    from src.losses.arcface import arcface_loss
    from src.models.components.arcface_head import ArcFaceHead

    torch.manual_seed(1)
    head = ArcFaceHead(feature_dim=32, num_classes=PAPER_NUM_SUB_VARIETIES)
    embeddings = torch.randn(24, 32)
    labels = torch.randint(0, PAPER_NUM_SUB_VARIETIES, (24,))

    plain, margined = head(embeddings, labels)
    assert arcface_loss(margined, labels) >= arcface_loss(plain, labels) - 1e-6


def test_arcface_logits_stay_within_the_feature_scale():
    """Logits are s * cos(theta), so |logit| <= s -- an unbounded value means a bug."""
    from src.models.components.arcface_head import ArcFaceHead

    torch.manual_seed(2)
    head = ArcFaceHead(feature_dim=32, num_classes=PAPER_NUM_SUB_VARIETIES, scale=30.0)
    logits, margined = head(torch.randn(32, 32) * 100, torch.randint(0, 27, (32,)))

    assert logits.abs().max().item() <= 30.0 + 1e-4
    assert margined.abs().max().item() <= 30.0 + 1e-4
    assert torch.isfinite(logits).all() and torch.isfinite(margined).all()


def test_arcface_gradients_are_finite_at_the_cosine_boundary():
    """cos(theta) = +-1 is where an acos-based implementation produces infinities."""
    from src.models.components.arcface_head import ArcFaceHead

    head = ArcFaceHead(feature_dim=8, num_classes=4)
    # Embeddings collinear with a class centre drive the cosine to exactly 1.
    embeddings = head.weight[0].detach().clone().unsqueeze(0).repeat(4, 1).requires_grad_(True)
    labels = torch.zeros(4, dtype=torch.long)

    _, margined = head(embeddings, labels)
    F.cross_entropy(margined, labels).backward()

    assert torch.isfinite(embeddings.grad).all()
    assert torch.isfinite(head.weight.grad).all()


# ---------------------------------------------------- Top-K routing width


def test_discarded_mass_grows_as_the_routing_width_shrinks():
    """Top-2 necessarily discards at least as much gate mass as Top-4."""
    torch.manual_seed(3)
    gate = F.softmax(torch.randn(64, PAPER_NUM_EXPERTS), dim=-1)
    _, top_2 = torch.topk(gate, 2, dim=-1)
    _, top_4 = torch.topk(gate, 4, dim=-1)
    assert l1_sparsity_loss(gate, top_2) >= l1_sparsity_loss(gate, top_4)


def test_sparsity_and_load_balancing_pull_in_opposite_directions():
    """The designed tension: decisive per-sample routing versus uniform batch usage."""
    torch.manual_seed(4)
    peaked = F.softmax(torch.randn(32, PAPER_NUM_EXPERTS) * 8, dim=-1)
    flat = F.softmax(torch.randn(32, PAPER_NUM_EXPERTS) * 0.01, dim=-1)

    _, peaked_indices = torch.topk(peaked, REVISED_TOP_K, dim=-1)
    _, flat_indices = torch.topk(flat, REVISED_TOP_K, dim=-1)

    # Peaked gates concentrate mass inside the selection: lower sparsity penalty.
    assert l1_sparsity_loss(peaked, peaked_indices) < l1_sparsity_loss(flat, flat_indices)
    # Flat gates spread utilisation evenly: lower (more negative) balancing loss.
    assert load_balancing_loss(flat, mode="entropy") < load_balancing_loss(peaked, mode="entropy")


# ------------------------------------------------- combined-objective toggles


def _hierarchical_output(**model_kwargs):
    from src.models.builder import HierarchicalSeedClassifier

    torch.manual_seed(5)
    model = HierarchicalSeedClassifier(
        feature_dim=PAPER_EMBED_DIM,
        embed_dim=PAPER_EMBED_DIM,
        num_seed_types=PAPER_NUM_SEED_TYPES,
        num_sub_varieties=PAPER_NUM_SUB_VARIETIES,
        num_experts=PAPER_NUM_EXPERTS,
        top_k=REVISED_TOP_K,
        moe_hidden_dim=32,
        num_heads=4,
        dropout_rate=0.0,
        **model_kwargs,
    )
    labels_sub = torch.randint(0, PAPER_NUM_SUB_VARIETIES, (10,))
    return model, model(torch.randn(10, PAPER_EMBED_DIM), labels_sub), labels_sub


def test_disabling_kl_skips_the_term_entirely(subvariety_to_seed_type):
    from src.losses.hierarchical import CombinedHierarchicalLoss

    _, output, sub_labels = _hierarchical_output()
    seed_labels = torch.randint(0, PAPER_NUM_SEED_TYPES, (10,))

    with_kl = CombinedHierarchicalLoss(
        PAPER_NUM_SEED_TYPES, PAPER_NUM_SUB_VARIETIES,
        subvariety_to_seed_type=subvariety_to_seed_type, use_kl_loss=True,
    )(output, seed_labels, sub_labels)
    without_kl = CombinedHierarchicalLoss(
        PAPER_NUM_SEED_TYPES, PAPER_NUM_SUB_VARIETIES,
        subvariety_to_seed_type=subvariety_to_seed_type, use_kl_loss=False,
    )(output, seed_labels, sub_labels)

    assert with_kl.kl.item() > 0.0
    assert without_kl.kl.item() == 0.0
    # Removing a positive term must lower the total by exactly that term.
    assert without_kl.total.item() == pytest.approx(
        with_kl.total.item() - with_kl.kl.item(), abs=1e-5
    )


def test_use_kl_loss_false_overrides_a_nonzero_lambda(subvariety_to_seed_type):
    """The switch must win, or a leftover lambda would silently re-enable the term."""
    from src.losses.hierarchical import CombinedHierarchicalLoss

    criterion = CombinedHierarchicalLoss(
        PAPER_NUM_SEED_TYPES, PAPER_NUM_SUB_VARIETIES,
        subvariety_to_seed_type=subvariety_to_seed_type,
        lambda_kl=5.0, use_kl_loss=False,
    )
    assert criterion.lambda_kl == 0.0


def test_linear_head_makes_the_arcface_term_a_plain_cross_entropy(subvariety_to_seed_type):
    """`use_arcface=False` needs no loss-side branch; the term degrades on its own."""
    from src.losses.hierarchical import CombinedHierarchicalLoss

    _, output, sub_labels = _hierarchical_output(use_arcface=False)
    seed_labels = torch.randint(0, PAPER_NUM_SEED_TYPES, (10,))
    breakdown = CombinedHierarchicalLoss(
        PAPER_NUM_SEED_TYPES, PAPER_NUM_SUB_VARIETIES,
        subvariety_to_seed_type=subvariety_to_seed_type,
    )(output, seed_labels, sub_labels)

    assert breakdown.arcface.item() == pytest.approx(
        F.cross_entropy(output.sub_logits, sub_labels).item(), abs=1e-6
    )


def test_every_loss_component_is_finite_for_all_ablations(subvariety_to_seed_type):
    from src.losses.hierarchical import CombinedHierarchicalLoss

    seed_labels = torch.randint(0, PAPER_NUM_SEED_TYPES, (10,))
    for flag in ("use_moe", "use_arcface", "use_residual", "use_cross_attention"):
        _, output, sub_labels = _hierarchical_output(**{flag: False})
        breakdown = CombinedHierarchicalLoss(
            PAPER_NUM_SEED_TYPES, PAPER_NUM_SUB_VARIETIES,
            subvariety_to_seed_type=subvariety_to_seed_type,
        )(output, seed_labels, sub_labels)
        for name, value in breakdown.as_dict().items():
            assert math.isfinite(value), f"{flag}=False produced non-finite {name}"


# ------------------------------------- hierarchy consistency, in log space


def test_kl_gradient_survives_confident_disagreement():
    """The bug ``clamp_min(1e-8) -> log`` hid, made into a regression test.

    ``clamp_min`` has zero gradient in the clamped region. Aggregating in
    probability space and then taking a log therefore silently dropped the KL
    term whenever an aggregated seed-type probability fell below 1e-8 -- and with
    ``s = 30`` that was the *common* case, not an edge case: a cosine gap of 1.0
    becomes a logit gap of 30, i.e. a probability ratio of e^30 ~ 1e13, so
    aggregating 27 probabilities into 4 bins leaves the non-argmax bins around
    1e-13. The term was live when the two heads agreed and dead when they
    disagreed confidently, which is precisely when it was supposed to act.

    ``logsumexp`` over each parent's children is exact and its gradient is
    well-conditioned across the whole range.
    """
    from src.losses.hierarchical import build_subvariety_seed_mapping, hierarchical_kl_loss

    mapping = build_subvariety_seed_mapping(
        num_sub_varieties=PAPER_NUM_SUB_VARIETIES,
        num_seed_types=PAPER_NUM_SEED_TYPES,
        subvarieties_per_seed_type=[8, 3, 13, 3],
    )

    # Maximal disagreement: the coarse head is certain of seed type 0, the fine
    # head is certain of a sub-variety belonging to seed type 2, and both at the
    # magnitude s = 30 produces.
    seed_logits = torch.zeros(2, PAPER_NUM_SEED_TYPES)
    seed_logits[:, 0] = 30.0
    sub_logits = torch.full((2, PAPER_NUM_SUB_VARIETIES), -30.0, requires_grad=True)
    with torch.no_grad():
        sub_logits[:, 20] = 30.0  # index 20 lives under seed type 2

    loss = hierarchical_kl_loss(seed_logits, sub_logits, mapping, detach_seed_target=True)
    loss.backward()

    assert torch.isfinite(loss), "log-space aggregation must not produce inf or nan"
    assert loss.item() > 1.0, "a confident disagreement must be expensive"
    assert sub_logits.grad is not None
    assert sub_logits.grad.abs().sum() > 0, (
        "the KL term must have gradient exactly where the heads disagree confidently"
    )


def test_probability_space_aggregation_is_what_lost_the_gradient():
    """The counterfactual, so the fix is attributable rather than asserted."""
    seed_probs = torch.zeros(1, PAPER_NUM_SEED_TYPES)
    seed_probs[:, 0] = 1.0
    sub_logits = torch.full((1, PAPER_NUM_SUB_VARIETIES), -30.0, requires_grad=True)
    with torch.no_grad():
        sub_logits[:, 20] = 30.0

    mapping = torch.zeros(PAPER_NUM_SUB_VARIETIES, PAPER_NUM_SEED_TYPES)
    for index in range(PAPER_NUM_SUB_VARIETIES):
        mapping[index, 0 if index < 8 else 2] = 1.0

    aggregated = torch.matmul(F.softmax(sub_logits, dim=-1), mapping)
    # Seed type 0's aggregated probability is far below the old 1e-8 floor.
    assert aggregated[0, 0].item() < 1e-8
    F.kl_div(
        torch.log(aggregated.clamp_min(1e-8)), seed_probs, reduction="batchmean"
    ).backward()
    clamped_grad = sub_logits.grad.abs().sum().item()

    sub_logits.grad = None
    log_aggregated = aggregate_sub_log_probs(sub_logits, mapping)
    F.kl_div(log_aggregated, seed_probs, reduction="batchmean").backward()
    exact_grad = sub_logits.grad.abs().sum().item()

    assert clamped_grad == pytest.approx(0.0, abs=1e-12), "the clamped form was gradient-dead"
    assert exact_grad > 0.0


def test_aggregated_log_probabilities_are_exact():
    """``logsumexp`` over children must reproduce the probability-space sum."""
    mapping = torch.zeros(6, 2)
    mapping[:3, 0] = 1.0
    mapping[3:, 1] = 1.0

    logits = torch.randn(4, 6)
    expected = torch.matmul(F.softmax(logits, dim=-1), mapping)
    assert torch.allclose(aggregate_sub_log_probs(logits, mapping).exp(), expected, atol=1e-6)


def test_jsd_is_symmetric_and_bounded_by_log_two():
    """The alternative the hierarchy-consistency literature converged on."""
    from src.losses.hierarchical import hierarchical_kl_loss

    mapping = torch.zeros(6, 2)
    mapping[:3, 0] = 1.0
    mapping[3:, 1] = 1.0

    for _ in range(10):
        seed_logits = torch.randn(8, 2) * 5
        sub_logits = torch.randn(8, 6) * 5
        value = hierarchical_kl_loss(
            seed_logits, sub_logits, mapping, detach_seed_target=False, mode="jsd"
        )
        assert 0.0 <= value.item() <= math.log(2.0) + 1e-5


def test_detached_kl_target_blocks_gradient_into_the_coarse_head():
    """The coarse head is supervised by hard labels; KL must not renegotiate that.

    Without the detach, ``L_KL`` can be reduced by making ``P_seed`` agree with a
    confidently *wrong* fine prediction -- and since Eq. 7 fits fast on 4 classes,
    the marginal gradient it spends goes disproportionately to the hard,
    ambiguous samples where following the fine head is most likely wrong.
    """
    from src.losses.hierarchical import hierarchical_kl_loss

    mapping = torch.zeros(6, 2)
    mapping[:3, 0] = 1.0
    mapping[3:, 1] = 1.0
    # requires_grad so the loss has a graph even when the seed branch is cut off.
    sub_logits = torch.randn(8, 6, requires_grad=True)

    detached = torch.randn(8, 2, requires_grad=True)
    hierarchical_kl_loss(detached, sub_logits, mapping, detach_seed_target=True).backward()
    assert detached.grad is None or detached.grad.abs().sum() == 0
    # The fine branch must still be reshaped -- that is the term's actual job.
    assert sub_logits.grad is not None and sub_logits.grad.abs().sum() > 0

    attached = torch.randn(8, 2, requires_grad=True)
    hierarchical_kl_loss(attached, sub_logits, mapping, detach_seed_target=False).backward()
    assert attached.grad is not None and attached.grad.abs().sum() > 0


def test_tau_kl_decouples_the_hierarchy_term_from_the_arcface_scale():
    """``lambda_kl`` and ``arcface_scale`` were one hyperparameter in disguise.

    ``P_sub = softmax(s cos theta)`` is near-one-hot by construction at ``s = 30``,
    so raising the ArcFace scale silently sharpened the distribution the KL term
    measures. Dividing by ``tau_kl`` before the softmax separates them.
    """
    mapping = torch.zeros(6, 2)
    mapping[:3, 0] = 1.0
    mapping[3:, 1] = 1.0
    logits = torch.randn(8, 6) * 10

    def entropy(tau):
        probs = aggregate_sub_log_probs(logits, mapping, tau=tau).exp()
        return -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(-1).mean()

    assert entropy(10.0) > entropy(1.0), "a higher temperature must soften P_sub"


# --------------------------------------------------- EMA-centroid compactness


def test_ema_centroids_make_every_sample_contribute():
    """Per-batch centroids leave most of a batch of 16 contributing exactly zero.

    With 27 roughly-balanced classes, the expected number with two or more
    members in a batch of 16 is 27*[1 - (1-p)^16 - 16p(1-p)^15] = 3.2, so ~10 of
    16 embeddings are their own centroid and score exactly 0, and the rest are
    pulled toward a centroid estimated from two samples.
    """
    from src.losses.cosine import CosineSimilarityLoss, intra_class_cosine_loss

    torch.manual_seed(0)
    num_classes = 14

    # A batch where every class appears exactly once -- the regime a batch of 16
    # over 27 classes is close to.
    singletons = torch.randn(num_classes, 8)
    labels = torch.arange(num_classes)
    assert (labels.bincount() == 1).all(), "fixture must be all singletons"

    # Per-batch centroids: every sample is its own centroid, so the term is
    # identically zero and carries no signal at all.
    assert intra_class_cosine_loss(singletons, labels).item() == pytest.approx(0.0, abs=1e-6)

    ema = CosineSimilarityLoss(
        mode="intra_class", num_classes=num_classes, embed_dim=8, centroid_momentum=0.9
    )
    ema.train()
    ema(torch.randn(num_classes, 8), None, labels)   # seed the centroids
    scored = ema(singletons, None, labels)           # scored against history

    assert torch.isfinite(scored)
    assert ema.centroid_initialized.all()
    assert scored.item() > 1e-3, (
        "EMA centroids must give singletons a real target instead of scoring them 0"
    )


def test_residual_magnitude_hinge_is_inactive_in_the_healthy_regime():
    """It must bound the residual without rewarding its disappearance.

    That is exactly the property ``1 - cos(h + P(p_s), h)`` lacks: cosine is
    magnitude-invariant, so the cheapest way to preserve direction is to make the
    residual vanish -- and ``P(p_s) = 0`` is a global minimum of it.
    """
    from src.losses.cosine import residual_cosine_loss, residual_magnitude_loss

    base = torch.randn(8, 16)
    small = base * 0.1
    large = base * 2.0

    assert residual_magnitude_loss(small, base, tau=0.5).item() == pytest.approx(0.0, abs=1e-8)
    assert residual_magnitude_loss(large, base, tau=0.5).item() > 0.0
    # Zeroing the residual is free under the hinge -- and it is also free under
    # the cosine form, which is the defect: there the *only* way to score 0 is to
    # be parallel, and P = 0 is the cheapest way to be parallel.
    zero = torch.zeros_like(base)
    assert residual_magnitude_loss(zero, base, tau=0.5).item() == pytest.approx(0.0, abs=1e-8)
    assert residual_cosine_loss(base + zero, base).item() == pytest.approx(0.0, abs=1e-6)
