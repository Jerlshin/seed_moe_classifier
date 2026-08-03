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

The revision adds four measurements the submitted suite could not make:

:func:`expert_label_nmi`
    Utilisation bars show *balance*, not *specialisation*. Normalised mutual
    information between the routed expert and the label says whether the router
    learned anything about the task, which is the question Section 5.2's claim
    actually rests on. Computable from an existing ``test_predictions.npz`` with
    no retraining.

:func:`expected_calibration_error` / :func:`fit_temperature`
    ``sub_scores`` are ``softmax(s cos theta)`` and therefore overconfident by
    construction. A hierarchical classifier's practical value depends on
    trustworthy confidence, so ECE is reported with a temperature fitted on the
    validation split.

:func:`mcnemar_test`
    All variants share a byte-identical test split, so their predictions are
    *paired* and McNemar's exact test is both valid and more powerful than
    comparing independent confidence intervals. It is what turns "wo_moe is
    0.6 pp lower" into a claim. Implemented against ``scipy.stats.binomtest``
    rather than statsmodels so the reporting path adds no new dependency.

:func:`per_seed_type_breakdown`
    The hierarchy is 13 rice + 8 millet + 3 + 3, so an overall figure is
    structurally dominated by rice. This reports each seed type's sub-variety
    accuracy and macro-F1 separately.

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


def expert_label_nmi(
    expert_indices: Any,
    labels: Any,
    tokens_per_sample: int = 1,
) -> float:
    """Normalised mutual information between the top-1 expert and the label.

    Balanced utilisation is necessary but not sufficient for the "expert
    specialisation" the paper claims: six experts each taking a sixth of the
    traffic at random are perfectly balanced and perfectly uninformative. NMI
    against the labels is the measurement that separates the two, and it is what
    makes a seed-conditioned router's effect reportable rather than asserted.

    Args:
        expert_indices: ``[n, top_k]`` routed expert indices from
            ``test_predictions.npz``.
        labels: ``[n_samples]`` seed-type or sub-variety labels.
        tokens_per_sample: Routing tokens per image. Above 1 (grid routing) the
            per-image expert is the modal top-1 choice across that image's
            tokens, since the label is a property of the image, not the token.
    """
    from sklearn.metrics import normalized_mutual_info_score

    indices = _as_array(expert_indices)
    targets = _as_array(labels).reshape(-1)
    if indices.size == 0 or targets.size == 0:
        return float("nan")
    if indices.ndim == 1:
        indices = indices.reshape(-1, 1)

    top1 = indices[:, 0].astype(np.int64)
    tokens = max(int(tokens_per_sample), 1)
    if tokens > 1:
        if top1.size % tokens != 0:
            return float("nan")
        grouped = top1.reshape(-1, tokens)
        num_experts = int(grouped.max()) + 1
        counts = np.stack([(grouped == e).sum(axis=1) for e in range(num_experts)], axis=1)
        top1 = counts.argmax(axis=1)

    if top1.size != targets.size:
        return float("nan")
    if np.unique(top1).size < 2 or np.unique(targets).size < 2:
        # One expert took everything, or one class is present: MI is 0 either
        # way, and reporting it as such is more useful than reporting nan.
        return 0.0
    return float(normalized_mutual_info_score(targets, top1))


def fit_temperature(
    logits: Any,
    labels: Any,
    max_iterations: int = 200,
    learning_rate: float = 0.01,
) -> float:
    """Fit a single softmax temperature by NLL (Guo et al., 2017).

    Fitted on **validation** predictions and applied to the test ones; fitting it
    on the test split would make the resulting ECE meaningless.

    Args:
        logits: ``[n, num_classes]`` pre-softmax scores.
        labels: ``[n]`` integer targets.
    """
    import torch

    scores = torch.as_tensor(_as_array(logits), dtype=torch.float32)
    targets = torch.as_tensor(_as_array(labels), dtype=torch.long)
    if scores.ndim != 2 or scores.shape[0] == 0:
        return 1.0

    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=learning_rate, max_iter=max_iterations)

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(scores / log_temperature.exp(), targets)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except RuntimeError:
        return 1.0
    return float(log_temperature.detach().exp().clamp(0.05, 100.0))


def expected_calibration_error(
    probabilities: Any,
    labels: Any,
    num_bins: int = 15,
) -> dict[str, float]:
    """Equal-width-binned ECE plus the mean confidence/accuracy gap.

    ``ece`` is the support-weighted mean absolute gap between confidence and
    accuracy over ``num_bins`` confidence bins. ``overconfidence`` is
    ``mean(confidence) - accuracy``, signed, which says *which way* the model is
    miscalibrated -- for a ``softmax(s cos theta)`` head the answer is
    emphatically positive before temperature scaling.
    """
    probs = _as_array(probabilities).astype(np.float64)
    targets = _as_array(labels).reshape(-1).astype(np.int64)
    if probs.ndim != 2 or probs.shape[0] == 0 or probs.shape[0] != targets.size:
        return {}

    confidence = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == targets).astype(np.float64)

    edges = np.linspace(0.0, 1.0, int(num_bins) + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if not mask.any():
            continue
        ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())

    return {
        "ece": float(ece),
        "mean_confidence": float(confidence.mean()),
        "accuracy": float(correct.mean()),
        "overconfidence": float(confidence.mean() - correct.mean()),
    }


def mcnemar_test(
    reference_correct: Any,
    variant_correct: Any,
    exact: bool = True,
) -> dict[str, float]:
    """McNemar's paired test between two classifiers on the same samples.

    Valid precisely because the ablation suite guarantees a byte-identical test
    split across variants: the two prediction vectors are paired, so the
    discordant counts are the whole of the evidence and the concordant ones carry
    none.

    Args:
        reference_correct: Boolean per-sample correctness of the reference
            (normally ``full_model``).
        variant_correct: Boolean per-sample correctness of the variant.
        exact: Use the exact binomial test. ``False`` uses the
            continuity-corrected chi-square approximation, which is only
            appropriate when ``n01 + n10`` is large.

    Returns ``n01`` (reference right, variant wrong), ``n10`` (the reverse), the
    p-value, and the accuracy difference. Empty when the inputs disagree in
    length.
    """
    reference = _as_array(reference_correct).reshape(-1).astype(bool)
    variant = _as_array(variant_correct).reshape(-1).astype(bool)
    if reference.size == 0 or reference.size != variant.size:
        return {}

    n01 = int((reference & ~variant).sum())
    n10 = int((~reference & variant).sum())
    discordant = n01 + n10

    if discordant == 0:
        p_value = 1.0
    elif exact:
        from scipy.stats import binomtest

        p_value = float(binomtest(min(n01, n10), n=discordant, p=0.5).pvalue)
    else:
        from scipy.stats import chi2

        statistic = (abs(n01 - n10) - 1.0) ** 2 / discordant
        p_value = float(chi2.sf(statistic, df=1))

    return {
        "n01": float(n01),
        "n10": float(n10),
        "discordant": float(discordant),
        "p_value": p_value,
        "accuracy_delta": float(variant.mean() - reference.mean()),
    }


def holm_bonferroni(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down adjustment over a family of comparisons.

    Six ablations each tested against the full model is six chances to find a
    difference; without correction the family-wise error rate is around 26 % at
    nominal 0.05.
    """
    if not p_values:
        return {}
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * float(value)))
        adjusted[name] = running
    return adjusted


def per_seed_type_breakdown(
    sub_true: Any,
    sub_pred: Any,
    subvariety_to_seed_type: Sequence[int],
    seed_type_names: Sequence[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Sub-variety accuracy and macro-F1 within each seed type.

    13 of 27 sub-varieties are rice, so an overall macro-F1 still lets the
    largest branch dominate the impression the table gives. This makes each
    branch's difficulty visible separately.
    """
    from sklearn.metrics import accuracy_score, f1_score

    true = _as_array(sub_true).reshape(-1).astype(np.int64)
    pred = _as_array(sub_pred).reshape(-1).astype(np.int64)
    if true.size == 0:
        return {}

    parent = np.asarray(list(subvariety_to_seed_type), dtype=np.int64)
    num_seed_types = int(parent.max()) + 1 if parent.size else 0
    names = list(seed_type_names) if seed_type_names else [str(i) for i in range(num_seed_types)]

    breakdown: dict[str, dict[str, float]] = {}
    for seed_index in range(num_seed_types):
        mask = parent[true] == seed_index
        if not mask.any():
            continue
        name = names[seed_index] if seed_index < len(names) else str(seed_index)
        breakdown[name] = {
            "accuracy": float(accuracy_score(true[mask], pred[mask])),
            "f1_macro": float(f1_score(true[mask], pred[mask], average="macro", zero_division=0)),
            "support": float(int(mask.sum())),
        }
    return breakdown


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
    per_seed_type_sub: dict[str, dict[str, float]] = field(default_factory=dict)
    calibration: dict[str, float] = field(default_factory=dict)
    expert_nmi: dict[str, float] = field(default_factory=dict)
    dead_experts: int = 0

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
        for name, values in self.per_seed_type_sub.items():
            for key, value in values.items():
                metrics[f"sub_variety_by_seed_type/{name}/{key}"] = float(value)
        for index, share in enumerate(self.expert_utilization):
            metrics[f"moe/expert_{index}_utilization"] = float(share)
        metrics["moe/dead_experts"] = float(self.dead_experts)
        for key, value in self.expert_nmi.items():
            metrics[f"moe/nmi_{key}"] = float(value)
        for key, value in self.calibration.items():
            metrics[f"calibration/{key}"] = float(value)
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
    tokens_per_sample: int = 1,
) -> HierarchicalEvaluation:
    """Run the full paper evaluation suite over one set of predictions."""
    seed_metrics = classification_metrics(seed_true, seed_pred)
    sub_metrics = classification_metrics(sub_true, sub_pred)
    if sub_scores is not None:
        sub_metrics.update(roc_auc_ovr(sub_true, sub_scores, num_classes=num_sub_varieties))

    utilization = (
        expert_utilization_counts(top_k_indices, num_experts)
        if top_k_indices is not None
        else np.zeros(num_experts)
    )
    expert_nmi: dict[str, float] = {}
    if top_k_indices is not None and num_experts > 1:
        expert_nmi = {
            "seed_type": expert_label_nmi(top_k_indices, seed_true, tokens_per_sample),
            "sub_variety": expert_label_nmi(top_k_indices, sub_true, tokens_per_sample),
        }

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
        expert_utilization=utilization,
        per_seed_type_sub=per_seed_type_breakdown(
            sub_true, sub_pred, subvariety_to_seed_type, seed_type_names
        ),
        calibration=(
            expected_calibration_error(sub_scores, sub_true) if sub_scores is not None else {}
        ),
        expert_nmi=expert_nmi,
        dead_experts=int((utilization == 0).sum()) if num_experts > 1 else 0,
    )
