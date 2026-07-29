"""Evaluation metrics reported in the paper (Section 6).

Coverage of the paper's quantitative results:

===========================================  ===========================================
Paper artefact                               Function
===========================================  ===========================================
Table 2, per-seed-type P/R/F1                :func:`per_class_metrics`
Section 6.2.1, 97.15% seed-type accuracy     :func:`classification_metrics`
Section 6.3, 74.84% sub-variety accuracy     :func:`classification_metrics`
Section 6.2, "area under the ROC curve"      :func:`roc_auc_ovr`
Table 3, KL alignment rate (95.94% overall)  :func:`kl_alignment_rate`
Fig. 10, confusion matrix                    :func:`confusion_matrices`
Fig. 12, per-sub-variety misclassification   :func:`misclassification_rates`
Section 5.2, expert utilisation              :func:`expert_utilization_counts`
Figs. 8-9, t-SNE of the embeddings           :func:`tsne_projection`
===========================================  ===========================================

The alignment rate deserves a precise definition, since the paper only names it.
A prediction is *hierarchically aligned* when the parent seed type of the
predicted sub-variety equals the independently predicted seed type::

    aligned_i = (parent[argmax sub_logits_i] == argmax seed_logits_i)

The overall rate is the mean over all samples; the per-seed-type breakdown
groups by the **true** seed type, which is what makes a row like the paper's
"Mustard: 0.7189" a statement about mustard samples.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _as_array(values: Any) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu()
    return np.asarray(values)


@dataclass
class ClassMetrics:
    """Precision / recall / F1 / support for a single class."""

    index: int
    name: str
    precision: float
    recall: float
    f1: float
    support: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "support": self.support,
        }


@dataclass
class AlignmentReport:
    """Hierarchical consistency between the two prediction heads (paper Table 3)."""

    overall: float
    per_seed_type: dict[str, float] = field(default_factory=dict)
    support_per_seed_type: dict[str, int] = field(default_factory=dict)

    def as_metrics(self, prefix: str = "kl_alignment") -> dict[str, float]:
        """Flatten into scalar metrics for the experiment tracker."""
        metrics = {f"{prefix}/overall": self.overall}
        for name, rate in self.per_seed_type.items():
            metrics[f"{prefix}/{name}"] = rate
        return metrics


def classification_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Accuracy plus macro, micro and weighted precision / recall / F1.

    ``f1_micro`` is reported because the revision's summary table asks for it.
    For single-label multi-class predictions it is numerically identical to
    accuracy -- every error is simultaneously one false positive and one false
    negative, so the micro-averaged precision and recall both collapse to the
    accuracy. It is included as a column because reviewers expect to see it, not
    because it carries information beyond ``accuracy``.
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    y_true = _as_array(y_true)
    y_pred = _as_array(y_pred)
    if y_true.size == 0:
        return {}

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def per_class_metrics(
    y_true: Any,
    y_pred: Any,
    class_names: Sequence[str] | None = None,
    num_classes: int | None = None,
) -> list[ClassMetrics]:
    """Per-class precision / recall / F1 / support (paper Table 2 and Fig. 11)."""
    from sklearn.metrics import precision_recall_fscore_support

    y_true = _as_array(y_true)
    y_pred = _as_array(y_pred)
    if num_classes is None:
        num_classes = len(class_names) if class_names else int(max(y_true.max(), y_pred.max())) + 1
    labels = list(range(num_classes))
    names = list(class_names) if class_names else [str(index) for index in labels]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    return [
        ClassMetrics(
            index=index,
            name=names[index] if index < len(names) else str(index),
            precision=float(precision[index]),
            recall=float(recall[index]),
            f1=float(f1[index]),
            support=int(support[index]),
        )
        for index in labels
    ]


def roc_auc_ovr(y_true: Any, y_score: Any, num_classes: int | None = None) -> dict[str, float]:
    """Macro one-vs-rest ROC AUC, tolerating classes absent from the split.

    ``sklearn.metrics.roc_auc_score(..., multi_class="ovr")`` requires every
    class to appear in ``y_true``, which a stratified validation fold over 27
    fine-grained classes does not guarantee. This computes the binary AUC for
    each class that has both positive and negative samples, macro-averages what
    survives, and reports how many classes contributed.
    """
    from sklearn.metrics import roc_auc_score

    y_true = _as_array(y_true)
    y_score = _as_array(y_score)
    if y_true.size == 0 or y_score.ndim != 2:
        return {}
    if num_classes is None:
        num_classes = y_score.shape[1]

    scores: list[float] = []
    for class_index in range(min(num_classes, y_score.shape[1])):
        positives = (y_true == class_index).astype(np.int8)
        if positives.sum() == 0 or positives.sum() == positives.size:
            continue
        try:
            scores.append(float(roc_auc_score(positives, y_score[:, class_index])))
        except ValueError:
            continue

    if not scores:
        return {}
    return {
        "auc_macro_ovr": float(np.mean(scores)),
        "auc_classes_scored": float(len(scores)),
    }


def kl_alignment_rate(
    seed_type_predictions: Any,
    sub_variety_predictions: Any,
    subvariety_to_seed_type: Sequence[int],
    seed_type_labels: Any | None = None,
    seed_type_names: Sequence[str] | None = None,
) -> AlignmentReport:
    """Hierarchical consistency rate, overall and per seed type (paper Table 3).

    Args:
        seed_type_predictions: ``argmax`` of the stage-1 logits, shape ``[n]``.
        sub_variety_predictions: ``argmax`` of the un-margined ArcFace logits.
        subvariety_to_seed_type: Parent seed type of each sub-variety index.
        seed_type_labels: True seed types, used to group the breakdown. When
            omitted, the breakdown groups by predicted seed type instead.
        seed_type_names: Names for the breakdown keys.
    """
    seed_pred = _as_array(seed_type_predictions).astype(np.int64)
    sub_pred = _as_array(sub_variety_predictions).astype(np.int64)
    if seed_pred.size == 0:
        return AlignmentReport(overall=float("nan"))

    parent = np.asarray(list(subvariety_to_seed_type), dtype=np.int64)
    if int(sub_pred.max(initial=-1)) >= parent.size:
        raise ValueError(
            f"sub-variety prediction {int(sub_pred.max())} has no entry in "
            f"subvariety_to_seed_type (length {parent.size})"
        )

    aligned = parent[sub_pred] == seed_pred
    grouping = _as_array(seed_type_labels).astype(np.int64) if seed_type_labels is not None else seed_pred

    num_seed_types = len(seed_type_names) if seed_type_names else int(grouping.max()) + 1
    names = list(seed_type_names) if seed_type_names else [str(i) for i in range(num_seed_types)]

    per_seed_type: dict[str, float] = {}
    support: dict[str, int] = {}
    for seed_index in range(num_seed_types):
        mask = grouping == seed_index
        name = names[seed_index] if seed_index < len(names) else str(seed_index)
        support[name] = int(mask.sum())
        if mask.any():
            per_seed_type[name] = float(aligned[mask].mean())

    return AlignmentReport(
        overall=float(aligned.mean()),
        per_seed_type=per_seed_type,
        support_per_seed_type=support,
    )


def misclassification_rates(
    y_true: Any,
    y_pred: Any,
    class_names: Sequence[str] | None = None,
    num_classes: int | None = None,
) -> dict[str, float]:
    """Per-class misclassification rate ``1 - recall`` (paper Fig. 12)."""
    y_true = _as_array(y_true)
    y_pred = _as_array(y_pred)
    if num_classes is None:
        num_classes = len(class_names) if class_names else int(max(y_true.max(), y_pred.max())) + 1
    names = list(class_names) if class_names else [str(i) for i in range(num_classes)]

    rates: dict[str, float] = {}
    for class_index in range(num_classes):
        mask = y_true == class_index
        if not mask.any():
            continue
        name = names[class_index] if class_index < len(names) else str(class_index)
        rates[name] = float((y_pred[mask] != class_index).mean())
    return rates


def confusion_matrices(
    y_true: Any,
    y_pred: Any,
    num_classes: int,
    normalize: bool = False,
) -> np.ndarray:
    """Confusion matrix of shape ``[num_classes, num_classes]`` (paper Fig. 10).

    ``normalize=True`` divides each row by its support, giving per-true-class
    rates that stay readable when class supports differ.
    """
    from sklearn.metrics import confusion_matrix

    matrix = confusion_matrix(
        _as_array(y_true),
        _as_array(y_pred),
        labels=list(range(num_classes)),
    ).astype(np.float64)
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)
    return matrix


def expert_utilization_counts(top_k_indices: Any, num_experts: int) -> np.ndarray:
    """Fraction of routing slots taken by each expert, shape ``[num_experts]``.

    The result sums to 1 across experts, so a perfectly balanced router gives
    ``1 / num_experts`` everywhere regardless of the Top-K value.
    """
    indices = _as_array(top_k_indices).reshape(-1).astype(np.int64)
    if indices.size == 0:
        return np.zeros(num_experts, dtype=np.float64)
    counts = np.bincount(indices, minlength=num_experts).astype(np.float64)
    return counts / counts.sum()


def tsne_projection(
    embeddings: Any,
    perplexity: float = 30.0,
    seed: int = 42,
    max_samples: int | None = 2000,
) -> np.ndarray | None:
    """Project embeddings to 2-D with t-SNE (paper Figs. 8-9).

    Returns ``None`` when there are too few samples for a meaningful
    perplexity, which t-SNE requires to satisfy ``perplexity < n_samples``.
    """
    from sklearn.manifold import TSNE

    vectors = _as_array(embeddings).astype(np.float64)
    if vectors.ndim != 2 or vectors.shape[0] < 5:
        return None
    if max_samples is not None and vectors.shape[0] > max_samples:
        vectors = vectors[:max_samples]

    effective_perplexity = min(perplexity, max((vectors.shape[0] - 1) / 3.0, 1.0))
    if vectors.shape[0] <= effective_perplexity + 1:
        return None

    return TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        random_state=seed,
    ).fit_transform(vectors)


@dataclass
class HierarchicalEvaluation:
    """Everything the paper reports for one evaluation pass."""

    seed_type: dict[str, float]
    sub_variety: dict[str, float]
    alignment: AlignmentReport
    per_class_seed: list[ClassMetrics]
    per_class_sub: list[ClassMetrics]
    seed_confusion: np.ndarray
    sub_confusion: np.ndarray
    sub_misclassification: dict[str, float]
    expert_utilization: np.ndarray

    def scalar_metrics(self) -> dict[str, float]:
        """Flatten every scalar into tracker-ready ``prefix/name`` keys."""
        metrics: dict[str, float] = {}
        metrics.update({f"seed_type/{key}": value for key, value in self.seed_type.items()})
        metrics.update({f"sub_variety/{key}": value for key, value in self.sub_variety.items()})
        metrics.update(self.alignment.as_metrics("kl_alignment"))
        for entry in self.per_class_seed:
            metrics[f"seed_type_class/{entry.name}/precision"] = entry.precision
            metrics[f"seed_type_class/{entry.name}/recall"] = entry.recall
            metrics[f"seed_type_class/{entry.name}/f1"] = entry.f1
        for entry in self.per_class_sub:
            metrics[f"sub_variety_class/{entry.name}/f1"] = entry.f1
        for index, share in enumerate(self.expert_utilization):
            metrics[f"moe/expert_{index}_utilization"] = float(share)
        return metrics


def evaluate_hierarchical(
    seed_true: Any,
    seed_pred: Any,
    sub_true: Any,
    sub_pred: Any,
    subvariety_to_seed_type: Sequence[int],
    num_seed_types: int,
    num_sub_varieties: int,
    seed_type_names: Sequence[str] | None = None,
    sub_variety_names: Sequence[str] | None = None,
    sub_scores: Any | None = None,
    top_k_indices: Any | None = None,
    num_experts: int = 6,
) -> HierarchicalEvaluation:
    """Run the full paper evaluation suite over one set of predictions."""
    seed_metrics = classification_metrics(seed_true, seed_pred)
    sub_metrics = classification_metrics(sub_true, sub_pred)
    if sub_scores is not None:
        sub_metrics.update(roc_auc_ovr(sub_true, sub_scores, num_classes=num_sub_varieties))

    return HierarchicalEvaluation(
        seed_type=seed_metrics,
        sub_variety=sub_metrics,
        alignment=kl_alignment_rate(
            seed_type_predictions=seed_pred,
            sub_variety_predictions=sub_pred,
            subvariety_to_seed_type=subvariety_to_seed_type,
            seed_type_labels=seed_true,
            seed_type_names=seed_type_names,
        ),
        per_class_seed=per_class_metrics(seed_true, seed_pred, seed_type_names, num_seed_types),
        per_class_sub=per_class_metrics(sub_true, sub_pred, sub_variety_names, num_sub_varieties),
        seed_confusion=confusion_matrices(seed_true, seed_pred, num_seed_types),
        sub_confusion=confusion_matrices(sub_true, sub_pred, num_sub_varieties),
        sub_misclassification=misclassification_rates(
            sub_true, sub_pred, sub_variety_names, num_sub_varieties
        ),
        expert_utilization=(
            expert_utilization_counts(top_k_indices, num_experts)
            if top_k_indices is not None
            else np.zeros(num_experts)
        ),
    )
