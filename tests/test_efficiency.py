"""Parameter accounting, FLOPs, latency and the Top-2 versus Top-4 comparison.

The active-parameter arithmetic is the substantive claim here: it is derived,
not measured, so it can be asserted exactly rather than approximately.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.models.builder import HierarchicalSeedClassifier
from src.utils.efficiency import (
    EfficiencyReport,
    ParameterReport,
    benchmark_latency,
    count_dormant_parameters,
    count_parameters,
    estimate_flops,
    parameter_report,
    profile_model,
    top_k_saving,
)
from tests.conftest import (
    PAPER_EMBED_DIM,
    PAPER_NUM_EXPERTS,
    PAPER_NUM_SEED_TYPES,
    PAPER_NUM_SUB_VARIETIES,
    REVISED_TOP_K,
    SUBMITTED_TOP_K,
)

CPU = torch.device("cpu")


def make_model(top_k: int = REVISED_TOP_K, **overrides) -> HierarchicalSeedClassifier:
    torch.manual_seed(0)
    kwargs = {
        "feature_dim": PAPER_EMBED_DIM,
        "embed_dim": PAPER_EMBED_DIM,
        "num_seed_types": PAPER_NUM_SEED_TYPES,
        "num_sub_varieties": PAPER_NUM_SUB_VARIETIES,
        "num_experts": PAPER_NUM_EXPERTS,
        "top_k": top_k,
        "moe_hidden_dim": 32,
        "num_heads": 4,
        "dropout_rate": 0.0,
    }
    kwargs.update(overrides)
    return HierarchicalSeedClassifier(**kwargs)


# ------------------------------------------------------------ basic counting


def test_count_parameters_reports_total_and_trainable():
    model = nn.Linear(4, 3)  # 12 weights + 3 biases
    total, trainable = count_parameters(model)
    assert total == 15
    assert trainable == 15

    model.bias.requires_grad = False
    _, trainable = count_parameters(model)
    assert trainable == 12


def test_shared_parameters_are_counted_once():
    """An encoder and a head that tie weights must not be double-counted."""
    shared = nn.Linear(4, 4)
    total, _ = count_parameters(shared, shared)
    assert total == sum(p.numel() for p in shared.parameters())


def test_a_model_without_a_router_has_nothing_dormant():
    dormant, num_experts, top_k = count_dormant_parameters(nn.Linear(4, 4))
    assert (dormant, num_experts, top_k) == (0, 1, 1)


# -------------------------------------------------------- active parameters


def test_active_equals_total_minus_the_unrouted_experts():
    model = make_model()
    report = parameter_report(model)
    per_expert = model.moe.parameters_per_expert()

    assert report.num_experts == PAPER_NUM_EXPERTS
    assert report.top_k == REVISED_TOP_K
    assert report.dormant == (PAPER_NUM_EXPERTS - REVISED_TOP_K) * per_expert
    assert report.active == report.total - report.dormant
    assert report.active < report.total


def test_total_parameters_are_identical_for_top_2_and_top_4():
    """Routing width changes what runs, never what the checkpoint stores."""
    assert parameter_report(make_model(top_k=2)).total == parameter_report(make_model(top_k=4)).total


def test_top_2_activates_fewer_parameters_than_top_4():
    top_2 = parameter_report(make_model(top_k=REVISED_TOP_K))
    top_4 = parameter_report(make_model(top_k=SUBMITTED_TOP_K))
    assert top_2.active < top_4.active
    assert top_2.dormant == 2 * top_4.dormant


def test_top_k_saving_reports_the_revision_comparison():
    model = make_model(top_k=REVISED_TOP_K)
    saving = top_k_saving(model, reference_top_k=SUBMITTED_TOP_K)

    per_expert = model.moe.parameters_per_expert()
    assert saving["parameters_per_expert"] == per_expert
    # Two extra experts stay dormant, so exactly 2 experts' worth is saved.
    assert saving["parameters_saved"] == pytest.approx(2 * per_expert)
    assert 0.0 < saving["parameters_saved_fraction"] < 1.0


def test_top_k_saving_is_empty_without_a_router():
    assert top_k_saving(make_model(use_moe=False)) == {}


def test_dormant_fraction_is_a_proportion():
    report = parameter_report(make_model())
    assert 0.0 < report.dormant_fraction < 1.0
    assert report.total_millions == pytest.approx(report.total / 1e6)


# ----------------------------------------------------------------- FLOPs


def test_flop_counting_returns_a_positive_total_or_none():
    """Degrades to None on torch builds without the counter, never raises."""
    model = make_model()
    flops = estimate_flops(model, torch.randn(2, PAPER_EMBED_DIM))
    assert flops is None or flops > 0


def test_flop_counting_leaves_training_mode_untouched():
    model = make_model()
    model.train()
    estimate_flops(model, torch.randn(2, PAPER_EMBED_DIM))
    assert model.training is True


# ---------------------------------------------------------------- latency


def test_benchmark_reports_one_entry_per_batch_size():
    model = make_model()
    reports = benchmark_latency(
        model, torch.randn(4, PAPER_EMBED_DIM), CPU, batch_sizes=(1, 4), warmup=1, iterations=2
    )
    assert [entry.batch_size for entry in reports] == [1, 4]
    for entry in reports:
        assert entry.latency_ms_per_batch > 0
        assert entry.latency_ms_per_sample == pytest.approx(
            entry.latency_ms_per_batch / entry.batch_size
        )
        assert entry.throughput_fps == pytest.approx(1000.0 / entry.latency_ms_per_sample)


def test_benchmark_restores_training_mode():
    model = make_model()
    model.train()
    benchmark_latency(model, torch.randn(2, PAPER_EMBED_DIM), CPU, batch_sizes=(2,), warmup=0, iterations=1)
    assert model.training is True


def test_benchmark_resizes_the_example_batch_in_both_directions():
    model = make_model()
    reports = benchmark_latency(
        model, torch.randn(3, PAPER_EMBED_DIM), CPU, batch_sizes=(1, 8), warmup=0, iterations=1
    )
    assert [entry.batch_size for entry in reports] == [1, 8]


# ------------------------------------------------------------ full report


def test_profile_model_produces_a_complete_report():
    model = make_model()
    report = profile_model(
        model,
        torch.randn(2, PAPER_EMBED_DIM),
        CPU,
        name="unit-test",
        batch_sizes=(1, 2),
        warmup=0,
        iterations=1,
    )

    assert isinstance(report, EfficiencyReport)
    assert isinstance(report.parameters, ParameterReport)
    assert report.name == "unit-test"
    assert report.primary_latency.batch_size == 1
    assert "top-2/6 experts" in report.summary_line()

    metrics = report.as_metrics()
    assert metrics["efficiency/top_k"] == float(REVISED_TOP_K)
    assert metrics["efficiency/total_parameters_m"] > metrics["efficiency/active_parameters_m"]
    assert all(isinstance(value, float) for value in metrics.values())


def test_profile_model_includes_an_external_encoder():
    """Latency must describe the deployed path, so the encoder's weights count."""
    model = make_model()
    encoder = nn.Sequential(nn.Linear(16, PAPER_EMBED_DIM))
    report = profile_model(
        model,
        torch.randn(2, 16),
        CPU,
        extra_modules=[encoder],
        batch_sizes=(2,),
        warmup=0,
        iterations=1,
        forward_fn=lambda batch: model(encoder(batch)),
    )
    standalone = parameter_report(model).total
    assert report.parameters.total > standalone


def test_report_serialises_to_json_safe_primitives():
    import json

    report = profile_model(
        make_model(), torch.randn(1, PAPER_EMBED_DIM), CPU,
        batch_sizes=(1,), warmup=0, iterations=1,
    )
    json.dumps(report.as_dict())  # raises TypeError on a stray tensor or ndarray
