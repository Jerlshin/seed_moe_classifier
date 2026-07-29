"""MoE routing: Top-K selection, dispatch weights, and gradient flow (Eq. 8)."""

from __future__ import annotations

import pytest
import torch

from src.models.components.moe_layer import (
    DEFAULT_NUM_EXPERTS,
    DEFAULT_TOP_K,
    DenseExpertBlock,
    MixtureOfExperts,
)
from tests.conftest import PAPER_EMBED_DIM, PAPER_NUM_EXPERTS, REVISED_TOP_K


@pytest.fixture
def moe() -> MixtureOfExperts:
    torch.manual_seed(0)
    return MixtureOfExperts(
        embed_dim=PAPER_EMBED_DIM,
        num_experts=PAPER_NUM_EXPERTS,
        mlp_dim=64,
        top_k=REVISED_TOP_K,
        num_heads=4,
    )


def test_revised_expert_configuration(moe):
    """Section 5.2 specifies six experts; the revision activates the top 2."""
    assert moe.num_experts == 6
    assert moe.top_k == REVISED_TOP_K == 2
    assert len(moe.experts) == 6


def test_module_defaults_match_the_revision():
    """The default must be Top-2 everywhere, not only where a config says so."""
    assert (DEFAULT_NUM_EXPERTS, DEFAULT_TOP_K) == (6, 2)
    default = MixtureOfExperts(embed_dim=32, mlp_dim=16, num_heads=4)
    assert default.num_experts == 6
    assert default.top_k == 2


def test_output_shapes(moe):
    output = moe(torch.randn(8, PAPER_EMBED_DIM))
    assert output.features.shape == (8, PAPER_EMBED_DIM)
    assert output.gate_probs.shape == (8, PAPER_NUM_EXPERTS)
    assert output.top_k_indices.shape == (8, REVISED_TOP_K)
    assert output.top_k_weights.shape == (8, REVISED_TOP_K)
    assert output.dispatch_weights.shape == (8, PAPER_NUM_EXPERTS)


def test_gate_is_a_probability_distribution(moe):
    output = moe(torch.randn(16, PAPER_EMBED_DIM))
    assert torch.allclose(output.gate_probs.sum(dim=-1), torch.ones(16), atol=1e-5)
    assert (output.gate_probs >= 0).all()


def test_top_k_selects_the_k_largest_gate_values(moe):
    """The selected indices must be exactly the argsort top-k of the gate."""
    output = moe(torch.randn(16, PAPER_EMBED_DIM))
    expected = output.gate_probs.argsort(dim=-1, descending=True)[:, : moe.top_k]
    assert set(map(tuple, output.top_k_indices.sort(dim=-1).values.tolist())) == set(
        map(tuple, expected.sort(dim=-1).values.tolist())
    )


def test_top_k_indices_are_unique_per_sample(moe):
    output = moe(torch.randn(32, PAPER_EMBED_DIM))
    for row in output.top_k_indices:
        assert len(set(row.tolist())) == moe.top_k


def test_renormalized_weights_form_a_convex_combination(moe):
    output = moe(torch.randn(16, PAPER_EMBED_DIM))
    assert torch.allclose(output.top_k_weights.sum(dim=-1), torch.ones(16), atol=1e-5)
    assert (output.top_k_weights >= 0).all()


def test_dispatch_weights_are_zero_outside_the_selection(moe):
    """Exactly ``num_experts - top_k`` entries per row must be zero."""
    output = moe(torch.randn(16, PAPER_EMBED_DIM))
    nonzero_per_row = (output.dispatch_weights > 0).sum(dim=-1)
    assert (nonzero_per_row == moe.top_k).all()

    for row, indices in enumerate(output.top_k_indices):
        mask = torch.ones(moe.num_experts, dtype=torch.bool)
        mask[indices] = False
        assert torch.all(output.dispatch_weights[row][mask] == 0)


def test_sparse_and_dense_dispatch_agree():
    """Sparse routing is an optimisation, so it must match dense evaluation."""
    torch.manual_seed(3)
    sparse = MixtureOfExperts(embed_dim=32, num_experts=6, mlp_dim=16, top_k=4, num_heads=4)
    sparse.eval()
    dense = MixtureOfExperts(embed_dim=32, num_experts=6, mlp_dim=16, top_k=4, num_heads=4)
    dense.load_state_dict(sparse.state_dict())
    dense.sparse_dispatch = False
    dense.eval()

    x = torch.randn(10, 32)
    with torch.no_grad():
        assert torch.allclose(sparse(x).features, dense(x).features, atol=1e-5)


def test_gradients_reach_selected_experts_and_the_gate(moe):
    """The gate always learns; an expert learns exactly when it was routed to.

    At Top-2 an expert can legitimately sit out a whole batch, so asserting that
    *every* expert has a gradient would be a flaky test of a false claim. The
    real invariant is the correspondence between selection and gradient.
    """
    output = moe(torch.randn(24, PAPER_EMBED_DIM))
    output.features.sum().backward()

    assert moe.gate.weight.grad is not None
    assert moe.gate.weight.grad.abs().sum() > 0

    routed = set(output.top_k_indices.flatten().tolist())
    for index, expert in enumerate(moe.experts):
        received = any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in expert.parameters()
        )
        assert received == (index in routed), f"expert {index}: routing and gradient disagree"


def test_top_k_equal_to_num_experts_is_a_dense_mixture():
    """With top_k == num_experts the dispatch weights must equal the gate."""
    torch.manual_seed(4)
    moe = MixtureOfExperts(embed_dim=16, num_experts=3, mlp_dim=8, top_k=3, num_heads=2)
    output = moe(torch.randn(6, 16))
    assert torch.allclose(output.dispatch_weights, output.gate_probs, atol=1e-5)


def test_expert_utilization_sums_to_top_k(moe):
    output = moe(torch.randn(20, PAPER_EMBED_DIM))
    utilization = moe.expert_utilization(output.top_k_indices)
    assert utilization.shape == (PAPER_NUM_EXPERTS,)
    assert utilization.sum().item() == pytest.approx(float(moe.top_k), abs=1e-4)


@pytest.mark.parametrize(
    ("num_experts", "top_k"),
    [(6, 0), (6, 7), (0, 1), (6, -1)],
)
def test_invalid_configurations_are_rejected(num_experts, top_k):
    with pytest.raises(ValueError):
        MixtureOfExperts(embed_dim=16, num_experts=num_experts, mlp_dim=8, top_k=top_k)


def test_rejects_non_2d_input(moe):
    with pytest.raises(ValueError, match=r"\[batch, embed_dim\]"):
        moe(torch.randn(4, 5, PAPER_EMBED_DIM))


def test_rejects_wrong_embed_dim(moe):
    with pytest.raises(ValueError, match="embed_dim"):
        moe(torch.randn(4, PAPER_EMBED_DIM + 1))


# ------------------------------------------------------- efficiency accounting


def test_parameters_per_expert_is_uniform(moe):
    """All experts share an architecture, which is what makes the count a closed form."""
    counts = {sum(p.numel() for p in expert.parameters()) for expert in moe.experts}
    assert len(counts) == 1
    assert moe.parameters_per_expert() == counts.pop()


def test_dormant_parameters_count_the_experts_that_sit_out(moe):
    """(E - K) experts are idle per sample: 4 of 6 at Top-2."""
    assert moe.dormant_parameters() == 4 * moe.parameters_per_expert()


def test_top_2_leaves_twice_as_many_parameters_dormant_as_top_4():
    """The revision's efficiency claim, stated as an exact identity."""
    shared = {"embed_dim": 64, "num_experts": 6, "mlp_dim": 32, "num_heads": 4}
    top_2 = MixtureOfExperts(top_k=2, **shared)
    top_4 = MixtureOfExperts(top_k=4, **shared)
    assert top_2.dormant_parameters() == 2 * top_4.dormant_parameters()


def test_dense_mixture_has_nothing_dormant():
    moe = MixtureOfExperts(embed_dim=32, num_experts=4, mlp_dim=16, top_k=4, num_heads=4)
    assert moe.dormant_parameters() == 0


# ------------------------------------------------------------ dense bypass


def test_dense_block_matches_the_moe_output_contract():
    """`use_moe=False` must produce tensors every downstream consumer accepts."""
    torch.manual_seed(0)
    block = DenseExpertBlock(embed_dim=32, mlp_dim=16, num_heads=4)
    output = block(torch.randn(5, 32))

    assert output.features.shape == (5, 32)
    assert output.gate_probs.shape == (5, 1)
    assert output.top_k_indices.shape == (5, 1)
    assert output.top_k_weights.shape == (5, 1)
    assert output.dispatch_weights.shape == (5, 1)


def test_dense_block_reports_a_degenerate_always_on_router():
    block = DenseExpertBlock(embed_dim=16, mlp_dim=8, num_heads=2)
    output = block(torch.randn(4, 16))

    assert block.num_experts == 1
    assert block.top_k == 1
    assert block.dormant_parameters() == 0
    assert torch.allclose(output.gate_probs, torch.ones(4, 1))
    assert torch.equal(output.top_k_indices, torch.zeros(4, 1, dtype=torch.long))


def test_dense_block_evaluates_its_expert_on_every_sample():
    """No routing means every sample must reach the block's parameters."""
    torch.manual_seed(0)
    block = DenseExpertBlock(embed_dim=16, mlp_dim=8, num_heads=2)
    block(torch.randn(6, 16)).features.sum().backward()
    assert all(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in block.expert.parameters()
    )


def test_dense_block_rejects_malformed_input():
    block = DenseExpertBlock(embed_dim=16, mlp_dim=8, num_heads=2)
    with pytest.raises(ValueError, match=r"\[batch, embed_dim\]"):
        block(torch.randn(2, 3, 16))
    with pytest.raises(ValueError, match="embed_dim=16"):
        block(torch.randn(2, 8))
