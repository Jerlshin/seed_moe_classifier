"""Evaluation metrics, especially the KL alignment rate of paper Table 3."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.metrics import (
    classification_metrics,
    confusion_matrices,
    evaluate_hierarchical,
    expert_utilization_counts,
    kl_alignment_rate,
    misclassification_rates,
    per_class_metrics,
    roc_auc_ovr,
    tsne_projection,
)
from tests.conftest import (
    PAPER_NUM_EXPERTS,
    PAPER_NUM_SEED_TYPES,
    PAPER_NUM_SUB_VARIETIES,
    REVISED_TOP_K,
)

SEED_NAMES = ["Millet", "Mustard", "Rice", "Seasame"]


def test_classification_metrics_on_perfect_predictions():
    labels = [0, 1, 2, 3, 0, 1]
    metrics = classification_metrics(labels, labels)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1_macro"] == pytest.approx(1.0)


def test_classification_metrics_are_bounded():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 4, 200)
    y_pred = rng.integers(0, 4, 200)
    for value in classification_metrics(y_true, y_pred).values():
        assert 0.0 <= value <= 1.0


def test_per_class_metrics_cover_every_class_even_when_absent():
    """A class missing from a fold must still appear, with zero support."""
    entries = per_class_metrics([0, 0, 1], [0, 0, 1], SEED_NAMES, PAPER_NUM_SEED_TYPES)
    assert len(entries) == PAPER_NUM_SEED_TYPES
    assert [entry.name for entry in entries] == SEED_NAMES
    assert entries[0].support == 2
    assert entries[3].support == 0
    assert entries[3].f1 == 0.0


def test_alignment_is_one_when_the_hierarchy_agrees(subvariety_to_seed_type):
    """Every predicted sub-variety's parent equals the predicted seed type."""
    sub_pred = np.arange(PAPER_NUM_SUB_VARIETIES)
    seed_pred = np.array(subvariety_to_seed_type)
    report = kl_alignment_rate(seed_pred, sub_pred, subvariety_to_seed_type, seed_pred, SEED_NAMES)
    assert report.overall == pytest.approx(1.0)
    assert all(rate == pytest.approx(1.0) for rate in report.per_seed_type.values())


def test_alignment_is_zero_when_the_hierarchy_never_agrees(subvariety_to_seed_type):
    sub_pred = np.arange(PAPER_NUM_SUB_VARIETIES)
    parents = np.array(subvariety_to_seed_type)
    seed_pred = (parents + 1) % PAPER_NUM_SEED_TYPES
    report = kl_alignment_rate(seed_pred, sub_pred, subvariety_to_seed_type, parents, SEED_NAMES)
    assert report.overall == pytest.approx(0.0)


def test_alignment_breakdown_groups_by_true_seed_type(subvariety_to_seed_type):
    """Reproduces the shape of Table 3: one rate per seed type, plus an overall."""
    # Four samples: rice agrees, mustard disagrees.
    parents = np.array(subvariety_to_seed_type)
    rice_sub = int(np.where(parents == 2)[0][0])
    mustard_sub = int(np.where(parents == 1)[0][0])

    seed_true = np.array([2, 2, 1, 1])
    sub_pred = np.array([rice_sub, rice_sub, mustard_sub, mustard_sub])
    seed_pred = np.array([2, 2, 0, 0])  # mustard rows predicted as Millet

    report = kl_alignment_rate(seed_pred, sub_pred, subvariety_to_seed_type, seed_true, SEED_NAMES)
    assert report.per_seed_type["Rice"] == pytest.approx(1.0)
    assert report.per_seed_type["Mustard"] == pytest.approx(0.0)
    assert report.overall == pytest.approx(0.5)
    assert report.support_per_seed_type["Rice"] == 2


def test_alignment_rejects_out_of_range_sub_variety(subvariety_to_seed_type):
    with pytest.raises(ValueError, match="no entry"):
        kl_alignment_rate([0], [999], subvariety_to_seed_type)


def test_alignment_report_flattens_to_tracker_metrics(subvariety_to_seed_type):
    report = kl_alignment_rate(
        np.array([0, 1]), np.array([0, 8]), subvariety_to_seed_type,
        np.array([0, 1]), SEED_NAMES,
    )
    metrics = report.as_metrics("kl_alignment")
    assert "kl_alignment/overall" in metrics
    assert "kl_alignment/Millet" in metrics


def test_misclassification_rate_is_one_minus_recall():
    rates = misclassification_rates([0, 0, 0, 1], [0, 1, 1, 1], ["a", "b"], 2)
    assert rates["a"] == pytest.approx(2 / 3)
    assert rates["b"] == pytest.approx(0.0)


def test_confusion_matrix_shape_and_totals():
    matrix = confusion_matrices([0, 1, 2, 3], [0, 1, 2, 3], PAPER_NUM_SEED_TYPES)
    assert matrix.shape == (PAPER_NUM_SEED_TYPES, PAPER_NUM_SEED_TYPES)
    assert np.allclose(np.diag(matrix), 1.0)
    assert matrix.sum() == 4


def test_confusion_matrix_normalizes_by_row():
    matrix = confusion_matrices([0, 0, 1], [0, 1, 1], 2, normalize=True)
    assert np.allclose(matrix.sum(axis=1), 1.0)
    assert matrix[0, 0] == pytest.approx(0.5)


def test_expert_utilization_sums_to_one():
    rng = np.random.default_rng(1)
    indices = rng.integers(0, PAPER_NUM_EXPERTS, size=(64, REVISED_TOP_K))
    utilization = expert_utilization_counts(indices, PAPER_NUM_EXPERTS)
    assert utilization.shape == (PAPER_NUM_EXPERTS,)
    assert utilization.sum() == pytest.approx(1.0)


def test_expert_utilization_detects_collapse():
    indices = np.zeros((32, REVISED_TOP_K), dtype=np.int64)
    utilization = expert_utilization_counts(indices, PAPER_NUM_EXPERTS)
    assert utilization[0] == pytest.approx(1.0)
    assert utilization[1:].sum() == pytest.approx(0.0)


def test_roc_auc_skips_classes_absent_from_the_split():
    """27 classes but only 2 present must not raise; it scores what it can."""
    rng = np.random.default_rng(2)
    y_true = np.array([0, 0, 1, 1])
    y_score = rng.random((4, PAPER_NUM_SUB_VARIETIES))
    result = roc_auc_ovr(y_true, y_score, PAPER_NUM_SUB_VARIETIES)
    assert 0.0 <= result["auc_macro_ovr"] <= 1.0
    assert result["auc_classes_scored"] == 2


def test_roc_auc_is_one_for_a_perfect_ranking():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    assert roc_auc_ovr(y_true, y_score, 2)["auc_macro_ovr"] == pytest.approx(1.0)


def test_tsne_returns_none_for_tiny_inputs():
    assert tsne_projection(np.random.rand(3, 8)) is None


def test_tsne_projects_to_two_dimensions():
    projection = tsne_projection(np.random.rand(60, 16), perplexity=5.0)
    assert projection is not None
    assert projection.shape == (60, 2)


def test_evaluate_hierarchical_assembles_the_full_paper_report(subvariety_to_seed_type):
    rng = np.random.default_rng(3)
    size = 120
    sub_true = rng.integers(0, PAPER_NUM_SUB_VARIETIES, size)
    seed_true = np.array(subvariety_to_seed_type)[sub_true]
    sub_pred = sub_true.copy()
    sub_pred[:20] = (sub_pred[:20] + 1) % PAPER_NUM_SUB_VARIETIES  # inject errors
    seed_pred = np.array(subvariety_to_seed_type)[sub_pred]

    scores = rng.random((size, PAPER_NUM_SUB_VARIETIES))
    experts = rng.integers(0, PAPER_NUM_EXPERTS, size=(size, REVISED_TOP_K))

    evaluation = evaluate_hierarchical(
        seed_true=seed_true, seed_pred=seed_pred,
        sub_true=sub_true, sub_pred=sub_pred,
        subvariety_to_seed_type=subvariety_to_seed_type,
        num_seed_types=PAPER_NUM_SEED_TYPES,
        num_sub_varieties=PAPER_NUM_SUB_VARIETIES,
        seed_type_names=SEED_NAMES,
        sub_variety_names=[f"sub{i}" for i in range(PAPER_NUM_SUB_VARIETIES)],
        sub_scores=scores, top_k_indices=experts, num_experts=PAPER_NUM_EXPERTS,
    )

    assert evaluation.seed_confusion.shape == (PAPER_NUM_SEED_TYPES, PAPER_NUM_SEED_TYPES)
    assert evaluation.sub_confusion.shape == (PAPER_NUM_SUB_VARIETIES, PAPER_NUM_SUB_VARIETIES)
    assert len(evaluation.per_class_seed) == PAPER_NUM_SEED_TYPES
    assert len(evaluation.per_class_sub) == PAPER_NUM_SUB_VARIETIES
    # seed_pred is derived from sub_pred, so the hierarchy is consistent by construction.
    assert evaluation.alignment.overall == pytest.approx(1.0)

    scalars = evaluation.scalar_metrics()
    assert "seed_type/accuracy" in scalars
    assert "sub_variety/accuracy" in scalars
    assert "kl_alignment/overall" in scalars
    assert "moe/expert_0_utilization" in scalars
    assert all(isinstance(value, float) for value in scalars.values())
