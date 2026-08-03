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


def test_expert_utilization_is_the_hard_dispatch_fraction(moe):
    """``expert_utilization`` must be ``f``, the quantity the loss now balances.

    It reports the share of *routing slots* each expert took, so it sums to 1
    however wide Top-K is -- the same normalisation ``plot_expert_utilization``
    draws against its ``1/num_experts`` reference line, and the same ``f`` the
    Switch auxiliary loss multiplies by ``P``. The submitted version returned
    "fraction of samples routed to each expert", which summed to ``top_k`` and so
    could not be compared against either.
    """
    output = moe(torch.randn(20, PAPER_EMBED_DIM))
    utilization = moe.expert_utilization(output.top_k_indices)
    assert utilization.shape == (PAPER_NUM_EXPERTS,)
    assert utilization.sum().item() == pytest.approx(1.0, abs=1e-5)
    assert (utilization >= 0).all()


def test_dead_expert_counter_fires_on_a_collapsed_router(moe):
    """The metric that would have caught a collapse the entropy loss called healthy."""
    with torch.no_grad():
        # Force every token onto experts {0, 1}: the F-01 failure, made concrete.
        moe.gate.weight.zero_()
        moe.gate.bias.copy_(torch.tensor([5.0, 5.0, 0.0, 0.0, 0.0, 0.0]))
    output = moe(torch.randn(20, PAPER_EMBED_DIM))
    assert moe.dead_expert_count(output.top_k_indices) == PAPER_NUM_EXPERTS - 2


@pytest.mark.parametrize(
    ("num_experts", "top_k"),
    [(6, 0), (6, 7), (0, 1), (6, -1)],
)
def test_invalid_configurations_are_rejected(num_experts, top_k):
    with pytest.raises(ValueError):
        MixtureOfExperts(embed_dim=16, num_experts=num_experts, mlp_dim=8, top_k=top_k)


def test_accepts_a_token_grid_and_routes_every_token(moe):
    """Grid routing is the whole point of ``token_mode="grid"``.

    Routing per spatial token raises the slots filled per optimiser step from
    ``batch x K`` to ``batch x tokens x K``, which is what makes the load
    statistic estimable at these batch sizes.
    """
    batch, tokens = 4, 9
    output = moe(torch.randn(batch, tokens, PAPER_EMBED_DIM))
    assert output.features.shape == (batch, tokens, PAPER_EMBED_DIM)
    assert output.gate_probs.shape == (batch * tokens, PAPER_NUM_EXPERTS)
    assert output.top_k_indices.shape == (batch * tokens, moe.top_k)
    assert output.tokens_per_sample == tokens


def test_rejects_malformed_input(moe):
    """2-D and 3-D are both valid now; 4-D never is."""
    with pytest.raises(ValueError, match=r"\[batch, embed_dim\]"):
        moe(torch.randn(4, 5, 3, PAPER_EMBED_DIM))


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
    with pytest.raises(ValueError, match="2-D or 3-D"):
        block(torch.randn(2, 3, 4, 16))
    with pytest.raises(ValueError, match="embed_dim=16"):
        block(torch.randn(2, 8))


# ------------------------------------------------- sparse/dense optimizer parity


def test_sparse_and_dense_dispatch_agree_after_an_optimizer_step():
    """Forward equality is not enough: AdamW treats the two paths differently.

    Under sparse dispatch an expert no token reached never enters the autograd
    graph, so ``p.grad is None`` and AdamW skips it **entirely** -- including
    decoupled weight decay and the moment-buffer decay. Under dense dispatch the
    same parameters receive a zero gradient, so decay *does* apply. The two
    therefore train measurably different models, which means the "debug-only"
    dense path could not be used to validate a sparse run, and rarely-routed
    experts carried stale Adam moments across long gaps.

    ``materialize_zero_grads()`` closes it. This asserts *parameter state* after
    a step, not just the forward pass.
    """
    def build(sparse: bool):
        torch.manual_seed(0)
        return MixtureOfExperts(
            embed_dim=16, num_experts=PAPER_NUM_EXPERTS, mlp_dim=8, top_k=1,
            num_heads=2, sparse_dispatch=sparse, gate_init_std=1e-3,
        )

    sparse, dense = build(True), build(False)
    inputs = torch.randn(4, 16)

    for module in (sparse, dense):
        # Weight decay is the mechanism that diverges, so it has to be non-zero.
        optimizer = torch.optim.AdamW(module.parameters(), lr=1e-2, weight_decay=0.1)
        optimizer.zero_grad(set_to_none=True)
        module(inputs).features.sum().backward()
        module.materialize_zero_grads()
        optimizer.step()

    routed = set(sparse(inputs).top_k_indices.flatten().tolist())
    assert len(routed) < PAPER_NUM_EXPERTS, "fixture must leave at least one expert unrouted"

    for (name, left), (_, right) in zip(
        sparse.experts.named_parameters(), dense.experts.named_parameters()
    ):
        assert torch.allclose(left, right, atol=1e-6), f"{name} diverged between dispatch modes"


def test_without_materialization_the_two_paths_diverge():
    """The counterfactual, so the fix is attributable rather than asserted."""
    def build(sparse: bool):
        torch.manual_seed(0)
        return MixtureOfExperts(
            embed_dim=16, num_experts=PAPER_NUM_EXPERTS, mlp_dim=8, top_k=1,
            num_heads=2, sparse_dispatch=sparse, gate_init_std=1e-3,
        )

    sparse, dense = build(True), build(False)
    inputs = torch.randn(4, 16)

    for module in (sparse, dense):
        optimizer = torch.optim.AdamW(module.parameters(), lr=1e-2, weight_decay=0.1)
        optimizer.zero_grad(set_to_none=True)
        module(inputs).features.sum().backward()
        optimizer.step()   # no materialization

    assert any(
        not torch.allclose(left, right, atol=1e-6)
        for (_, left), (_, right) in zip(
            sparse.experts.named_parameters(), dense.experts.named_parameters()
        )
    ), "unrouted experts must skip weight decay when their grads stay None"


def test_control_routers_own_no_trainable_routing_parameters():
    """``hash`` and ``uniform`` are controls for *learned* routing, not for sparsity."""
    for mode in ("hash", "uniform"):
        module = MixtureOfExperts(
            embed_dim=16, num_experts=PAPER_NUM_EXPERTS, mlp_dim=8,
            top_k=REVISED_TOP_K, num_heads=2, router_mode=mode,
        )
        assert module.gate is None and module.noise_gate is None
        output = module(torch.randn(8, 16))
        assert output.top_k_indices.shape == (8, REVISED_TOP_K)
        assert torch.allclose(output.gate_probs.sum(dim=-1), torch.ones(8), atol=1e-5)


def test_uniform_router_spreads_mass_evenly():
    module = MixtureOfExperts(
        embed_dim=16, num_experts=PAPER_NUM_EXPERTS, mlp_dim=8,
        top_k=PAPER_NUM_EXPERTS, num_heads=2, router_mode="uniform",
    )
    output = module(torch.randn(8, 16))
    assert torch.allclose(
        output.gate_probs, torch.full_like(output.gate_probs, 1.0 / PAPER_NUM_EXPERTS), atol=1e-6
    )


def test_gating_noise_is_training_only_and_anneals_to_nothing():
    """The only exploration deterministic Top-K has, and it must reach exactly 0.

    ``topk`` is flat almost everywhere, so nothing in the objective can say "this
    token should have gone to expert 4". The noise has to vanish before the end
    of training, or the routing that is deployed is not the routing that was
    measured.
    """
    module = MixtureOfExperts(
        embed_dim=16, num_experts=PAPER_NUM_EXPERTS, mlp_dim=8,
        top_k=REVISED_TOP_K, num_heads=2, noise_std=1.0,
    )
    assert module.noise_gate is not None
    inputs = torch.randn(8, 16)

    module.eval()
    torch.manual_seed(0)
    first = module(inputs).gate_logits
    torch.manual_seed(1)
    assert torch.allclose(first, module(inputs).gate_logits, atol=1e-6), "eval must be noiseless"

    module.train()
    module.set_noise_scale(0.0)
    torch.manual_seed(0)
    annealed = module(inputs).gate_logits
    torch.manual_seed(1)
    assert torch.allclose(annealed, module(inputs).gate_logits, atol=1e-6)


def test_noise_gate_is_not_allocated_when_noise_is_disabled():
    """A parameter no configuration can reach must not be built or counted."""
    module = MixtureOfExperts(
        embed_dim=16, num_experts=PAPER_NUM_EXPERTS, mlp_dim=8,
        top_k=REVISED_TOP_K, num_heads=2, noise_std=0.0,
    )
    assert module.noise_gate is None


def test_pooled_experts_do_not_allocate_dead_attention_parameters():
    """~2.07 M provably unreachable parameters were being reported as active.

    Six experts and one cross-attention block, each packing 295,680 parameters
    into Q and K slices that a length-1 sequence can never use. They were counted
    in ``ParameterReport.total`` and in ``active = total - dormant``, so the
    paper's "Active Params (M)" column included them.
    """
    affine = MixtureOfExperts(
        embed_dim=64, num_experts=2, mlp_dim=16, top_k=1, num_heads=4, token_mixing="affine"
    )
    attention = MixtureOfExperts(
        embed_dim=64, num_experts=2, mlp_dim=16, top_k=1, num_heads=4, token_mixing="attention"
    )
    assert affine.experts[0].attn is None
    assert attention.experts[0].attn is not None
    assert affine.parameters_per_expert() < attention.parameters_per_expert()


def test_capacity_matched_dense_block_is_wider_than_a_single_expert():
    """``wo_moe`` conflated routing with a top_k-fold cut in active capacity.

    The full model activates ``top_k`` experts per token; a dense block activates
    one. Matching the feed-forward width is what makes the gap attributable to
    the gate.
    """
    naive = DenseExpertBlock(embed_dim=32, mlp_dim=16, num_heads=4, capacity_multiplier=1)
    matched = DenseExpertBlock(embed_dim=32, mlp_dim=16, num_heads=4, capacity_multiplier=2)
    assert matched.parameters_per_expert() > naive.parameters_per_expert()
    assert matched(torch.randn(4, 32)).features.shape == (4, 32)
