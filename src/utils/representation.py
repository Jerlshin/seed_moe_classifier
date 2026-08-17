"""Representation-quality metrics for a **self-supervised** encoder (stage 1).

:mod:`src.utils.metrics` scores the predictions of a *trained classifier*. After
stage 1 there is no classifier: DINO produces an encoder and a 2,048-way
prototype head whose classes have no names. Asking "is this encoder good?"
therefore needs a different instrument set, and this module is it.

Three families, because they fail independently and a run can pass one while
failing another:

**Label-free geometry.** :func:`spectral_report` (RankMe, participation ratio,
stable rank, explained variance) detects *dimensional collapse* -- the failure
mode where the loss falls, nothing looks wrong, and the 768-D feature actually
occupies a handful of directions. :func:`alignment_uniformity` reports the two
quantities Wang & Isola (2020) showed contrastive objectives trade off, and
:func:`feature_statistics` catches dead channels. None of these needs a label,
so none of them can be inflated by the labels leaking anywhere.

**Label-based readout.** :func:`linear_probe` and :func:`knn_classifier` are the
two canonical SSL evaluations, and they answer different questions: the probe
asks whether the classes are *linearly* separable in this space, the k-NN asks
whether they are separable at all under plain cosine distance, with no fitted
parameters to launder a badly shaped space into a good score. Both take explicit
train/test index arrays so the caller supplies the **grouped** split this dataset
requires (see :func:`~src.trainers.moe_finetune.split_dataset`); a crop-level
split would let near-duplicate crops of one photograph sit on both sides and
turn either number into a memorisation score.

**Structure recovered without labels.** :func:`kmeans_report` and
:func:`prototype_report` ask whether the *unsupervised* partition already lines
up with the taxonomy -- k-means over the features, and the argmax over DINO's own
prototypes. High NMI here is the strongest available evidence that the objective
learned the task's structure rather than a readout being fitted to find it.

Everything is numpy in, JSON-safe dict out, so the caller can dump the whole
report and re-plot it later without recomputation. Heavy dependencies are
imported inside the functions that need them, matching :mod:`src.utils.metrics`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

#: Guard for divisions by a norm or a variance. Large enough that a genuinely
#: dead direction produces 0 rather than a huge number, small enough not to
#: perturb a live one.
EPS = 1e-8


def as_float_array(values: Any) -> np.ndarray:
    """Detach anything tensor-like and return a contiguous ``float64`` array."""
    if hasattr(values, "detach"):
        values = values.detach().cpu()
    return np.ascontiguousarray(np.asarray(values, dtype=np.float64))


def l2_normalize(features: Any) -> np.ndarray:
    """Row-wise L2 normalisation, safe on zero rows.

    Every cosine-space measurement here starts with this. Doing it once at the
    top of each function rather than trusting the caller is deliberate: an
    un-normalised feature matrix silently converts a cosine metric into a
    magnitude-weighted one, and SwinV2's pooled features do not have uniform
    norms.
    """
    matrix = as_float_array(features)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, EPS)


# --------------------------------------------------------------- label-free


@dataclass
class SpectrumReport:
    """Singular-value structure of one feature matrix.

    ``rankme`` is Garrido et al. (2023): the exponential of the entropy of the
    L1-normalised singular values. It is the label-free quantity that tracks
    downstream probe accuracy across SSL runs, and it is what makes "the loss
    fell but the features collapsed" a measurable statement rather than a worry.

    ``participation_ratio`` and ``stable_rank`` are computed on the **centred**
    matrix, so they describe the spread of the *variance* rather than of the
    features' absolute positions; a large common mean component inflates the
    leading singular value of the uncentred matrix and would otherwise make a
    collapsed representation look low-rank in only one of the three numbers.
    """

    num_samples: int
    dim: int
    rankme: float
    rankme_normalized: float
    participation_ratio: float
    stable_rank: float
    dims_for_90_variance: int
    dims_for_95_variance: int
    dims_for_99_variance: int
    top1_variance_share: float
    top10_variance_share: float
    singular_values: list[float] = field(default_factory=list)
    explained_variance_ratio: list[float] = field(default_factory=list)

    def as_dict(self, include_curves: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        if not include_curves:
            payload.pop("singular_values", None)
            payload.pop("explained_variance_ratio", None)
        return payload

    def as_metrics(self, prefix: str = "spectrum") -> dict[str, float]:
        return {
            f"{prefix}/rankme": self.rankme,
            f"{prefix}/rankme_normalized": self.rankme_normalized,
            f"{prefix}/participation_ratio": self.participation_ratio,
            f"{prefix}/stable_rank": self.stable_rank,
            f"{prefix}/dims_for_95_variance": float(self.dims_for_95_variance),
            f"{prefix}/top1_variance_share": self.top1_variance_share,
        }


def spectral_report(features: Any, max_curve_points: int = 768) -> SpectrumReport:
    """Dimensional-collapse diagnostics for a ``[n, d]`` feature matrix.

    RankMe is computed on the raw matrix as published; the variance-based
    quantities are computed after centring. Both are reported because they
    disagree in an informative way: a representation with a huge mean offset and
    little variance has high RankMe and a tiny participation ratio.
    """
    matrix = as_float_array(features)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError(f"Expected a [n, d] matrix with n >= 2, got {matrix.shape}")

    singular = np.linalg.svd(matrix, compute_uv=False)
    weights = singular / (singular.sum() + EPS) + 1e-7
    rankme = float(np.exp(-(weights * np.log(weights)).sum()))

    centred = matrix - matrix.mean(axis=0, keepdims=True)
    centred_singular = np.linalg.svd(centred, compute_uv=False)
    eigenvalues = centred_singular**2
    total = float(eigenvalues.sum()) + EPS
    explained = eigenvalues / total
    cumulative = np.cumsum(explained)

    def dims_for(threshold: float) -> int:
        reached = np.searchsorted(cumulative, threshold) + 1
        return int(min(reached, explained.size))

    participation = float((eigenvalues.sum() ** 2) / ((eigenvalues**2).sum() + EPS))
    stable = float(eigenvalues.sum() / (eigenvalues.max() + EPS))
    stride = max(1, explained.size // max(max_curve_points, 1))

    return SpectrumReport(
        num_samples=int(matrix.shape[0]),
        dim=int(matrix.shape[1]),
        rankme=rankme,
        rankme_normalized=float(rankme / max(min(matrix.shape), 1)),
        participation_ratio=participation,
        stable_rank=stable,
        dims_for_90_variance=dims_for(0.90),
        dims_for_95_variance=dims_for(0.95),
        dims_for_99_variance=dims_for(0.99),
        top1_variance_share=float(explained[0]),
        top10_variance_share=float(explained[: min(10, explained.size)].sum()),
        singular_values=[float(value) for value in centred_singular[::stride]],
        explained_variance_ratio=[float(value) for value in explained[::stride]],
    )


def feature_statistics(features: Any, dead_std_threshold: float = 1e-3) -> dict[str, float]:
    """Norm, spread and dead-channel counts of a feature matrix.

    ``dead_dims`` counts channels whose standard deviation across the dataset is
    below ``dead_std_threshold`` -- a channel that never moves carries no
    information no matter how large its value is, and a growing count of them is
    the earliest visible symptom of partial collapse.
    """
    matrix = as_float_array(features)
    norms = np.linalg.norm(matrix, axis=1)
    per_dim_std = matrix.std(axis=0)
    normalized = l2_normalize(matrix)
    mean_direction = normalized.mean(axis=0)
    return {
        "mean_norm": float(norms.mean()),
        "std_norm": float(norms.std()),
        "min_norm": float(norms.min()),
        "max_norm": float(norms.max()),
        "mean_per_dim_std": float(per_dim_std.mean()),
        "min_per_dim_std": float(per_dim_std.min()),
        "dead_dims": float((per_dim_std < dead_std_threshold).sum()),
        "dead_dim_fraction": float((per_dim_std < dead_std_threshold).mean()),
        # ||mean of unit vectors||: 1.0 means every sample points the same way
        # (total collapse), ~0 means the directions cancel.
        "mean_direction_norm": float(np.linalg.norm(mean_direction)),
    }


def alignment_uniformity(
    view_a: Any,
    view_b: Any,
    alignment_power: float = 2.0,
    uniformity_temperature: float = 2.0,
    max_samples: int = 4096,
    seed: int = 0,
) -> dict[str, float]:
    """Alignment and uniformity of Wang & Isola (2020).

    ``alignment`` is the mean squared distance between two augmented views of the
    same image on the unit sphere -- lower is better, and it is the only number
    here that measures whether the *invariance* the objective asks for actually
    holds. ``uniformity`` is ``log E[exp(-t ||u - v||^2)]`` over distinct pairs --
    lower (more negative) means the features spread more evenly over the sphere.

    A collapsed encoder scores a perfect alignment of 0 and a uniformity near
    ``log 1 = 0``; a random encoder scores excellent uniformity and terrible
    alignment. Neither number means anything on its own, which is why they are
    returned together.
    """
    left = l2_normalize(view_a)
    right = l2_normalize(view_b)
    if left.shape != right.shape:
        raise ValueError(f"View shapes must match, got {left.shape} and {right.shape}")

    rng = np.random.default_rng(seed)
    if left.shape[0] > max_samples:
        chosen = rng.choice(left.shape[0], size=max_samples, replace=False)
        left, right = left[chosen], right[chosen]

    alignment = float((np.linalg.norm(left - right, axis=1) ** alignment_power).mean())

    # Uniformity over the pooled set of both views, off-diagonal pairs only.
    pooled = np.concatenate([left, right], axis=0)
    squared = np.maximum(2.0 - 2.0 * (pooled @ pooled.T), 0.0)
    off_diagonal = ~np.eye(pooled.shape[0], dtype=bool)
    uniformity = float(np.log(np.exp(-uniformity_temperature * squared[off_diagonal]).mean() + EPS))

    cosine = (left * right).sum(axis=1)
    return {
        "alignment": alignment,
        "uniformity": uniformity,
        "positive_pair_cosine_mean": float(cosine.mean()),
        "positive_pair_cosine_std": float(cosine.std()),
        "pairs": float(left.shape[0]),
    }


def augmentation_consistency(
    clean: Any,
    augmented: Any,
    labels: Any,
    max_samples: int = 3000,
    seed: int = 0,
) -> dict[str, float]:
    """Does an augmented view still retrieve its own clean view?

    Three cosine distributions and one retrieval score:

    * ``same_image`` -- clean vs augmented view of the same crop. This is the
      invariance the DINO objective was trained to produce.
    * ``same_class`` / ``different_class`` -- clean vs clean, within and across
      sub-variety.
    * ``self_retrieval_top1`` -- fraction of augmented views whose nearest clean
      neighbour is their own clean view. This is the strictest available check
      that the invariance is *identity-preserving* rather than achieved by
      throwing the instance away, and it needs no trained parameters.

    ``same_image`` above ``same_class`` above ``different_class`` is the ordering
    a useful encoder produces; the *gaps* are what a table should report, because
    the absolute cosines are inflated by the shared mean direction of any
    pretrained trunk.
    """
    clean_matrix = l2_normalize(clean)
    augmented_matrix = l2_normalize(augmented)
    targets = as_float_array(labels).reshape(-1).astype(np.int64)
    if clean_matrix.shape != augmented_matrix.shape:
        raise ValueError("clean and augmented must have identical shapes")
    if targets.size != clean_matrix.shape[0]:
        raise ValueError("labels must have one entry per row")

    rng = np.random.default_rng(seed)
    if clean_matrix.shape[0] > max_samples:
        chosen = rng.choice(clean_matrix.shape[0], size=max_samples, replace=False)
        clean_matrix, augmented_matrix, targets = (
            clean_matrix[chosen],
            augmented_matrix[chosen],
            targets[chosen],
        )

    same_image = (clean_matrix * augmented_matrix).sum(axis=1)
    similarity = clean_matrix @ clean_matrix.T
    same_class_mask = targets[:, None] == targets[None, :]
    np.fill_diagonal(same_class_mask, False)
    different_class_mask = targets[:, None] != targets[None, :]

    cross = augmented_matrix @ clean_matrix.T
    self_retrieval = float((cross.argmax(axis=1) == np.arange(cross.shape[0])).mean())

    same_class_mean = float(similarity[same_class_mask].mean()) if same_class_mask.any() else float("nan")
    different_mean = (
        float(similarity[different_class_mask].mean()) if different_class_mask.any() else float("nan")
    )
    return {
        "same_image_cosine_mean": float(same_image.mean()),
        "same_image_cosine_std": float(same_image.std()),
        "same_class_cosine_mean": same_class_mean,
        "different_class_cosine_mean": different_mean,
        "same_image_minus_same_class": float(same_image.mean() - same_class_mean),
        "same_class_minus_different_class": float(same_class_mean - different_mean),
        "self_retrieval_top1": self_retrieval,
        "samples": float(clean_matrix.shape[0]),
    }


def class_separability(features: Any, labels: Any, class_names: Sequence[str] | None = None) -> dict[str, Any]:
    """Cosine-space separability of the labelled classes, overall and per class.

    ``silhouette_cosine`` is the headline: it is bounded in ``[-1, 1]``, needs no
    fitted model, and unlike a probe accuracy it cannot be rescued by a readout
    finding a clever hyperplane. ``fisher_ratio`` is ``trace(S_b) / trace(S_w)``
    on the normalised features, i.e. between-class variance per unit of
    within-class variance -- the quantity a linear readout actually exploits.

    Per-class silhouettes are returned as well, because on a 27-class taxonomy
    with 13 rice varieties the mean hides exactly the classes a reader cares
    about.
    """
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_samples

    matrix = l2_normalize(features)
    targets = as_float_array(labels).reshape(-1).astype(np.int64)
    if targets.size != matrix.shape[0]:
        raise ValueError("labels must have one entry per row")
    unique = np.unique(targets)
    if unique.size < 2:
        return {"silhouette_cosine": float("nan"), "num_classes": int(unique.size)}

    per_sample = silhouette_samples(matrix, targets, metric="cosine")

    centroids = np.stack([matrix[targets == label].mean(axis=0) for label in unique])
    centroids_unit = l2_normalize(centroids)
    centroid_similarity = centroids_unit @ centroids_unit.T
    off_diagonal = ~np.eye(unique.size, dtype=bool)

    intra: list[float] = []
    for position, label in enumerate(unique):
        members = matrix[targets == label]
        if members.shape[0] < 2:
            continue
        intra.append(float((members @ centroids_unit[position]).mean()))

    grand_mean = matrix.mean(axis=0)
    within = 0.0
    between = 0.0
    for position, label in enumerate(unique):
        members = matrix[targets == label]
        centre = members.mean(axis=0)
        within += float(((members - centre) ** 2).sum())
        between += float(members.shape[0] * ((centre - grand_mean) ** 2).sum())

    names = list(class_names) if class_names is not None else [str(label) for label in unique]
    per_class = {
        (names[int(label)] if int(label) < len(names) else str(label)): float(
            per_sample[targets == label].mean()
        )
        for label in unique
    }

    return {
        "silhouette_cosine": float(per_sample.mean()),
        "davies_bouldin": float(davies_bouldin_score(matrix, targets)),
        "calinski_harabasz": float(calinski_harabasz_score(matrix, targets)),
        "mean_intra_class_cosine": float(np.mean(intra)) if intra else float("nan"),
        "mean_inter_centroid_cosine": float(centroid_similarity[off_diagonal].mean()),
        "max_inter_centroid_cosine": float(centroid_similarity[off_diagonal].max()),
        "fisher_ratio": float(between / (within + EPS)),
        "num_classes": int(unique.size),
        "per_class_silhouette": per_class,
    }


def centroid_similarity_matrix(features: Any, labels: Any, num_classes: int) -> np.ndarray:
    """``[C, C]`` cosine similarity between class centroids, NaN for absent classes.

    Ordered by label index, so with sub-varieties sorted under their seed type the
    matrix has a block structure and the figure shows directly whether the
    *hierarchy* is present in the representation -- which is the property the
    stage-2 KL term is going to assume.
    """
    matrix = l2_normalize(features)
    targets = as_float_array(labels).reshape(-1).astype(np.int64)
    centroids = np.full((num_classes, matrix.shape[1]), np.nan)
    for label in range(num_classes):
        members = matrix[targets == label]
        if members.shape[0]:
            centroids[label] = members.mean(axis=0)
    unit = centroids / np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), EPS)
    return unit @ unit.T


def linear_cka(features_a: Any, features_b: Any, max_samples: int = 4096, seed: int = 0) -> float:
    """Linear CKA (Kornblith et al., 2019) between two representations of the same samples.

    Rows must correspond: element ``i`` of both matrices is the same image. Used
    here to answer "how far did 100 epochs of self-distillation actually move the
    representation away from its ImageNet initialisation?" -- a question no
    accuracy number answers, because a large accuracy change can come from a
    small rotation and a small one from a large upheaval.

    Computed in the feature-covariance form, which is ``O(d^2)`` rather than
    ``O(n^2)``, so the sample cap only exists to bound memory on the centring.
    """
    left = as_float_array(features_a)
    right = as_float_array(features_b)
    if left.shape[0] != right.shape[0]:
        raise ValueError("CKA needs the same samples in both matrices")

    rng = np.random.default_rng(seed)
    if left.shape[0] > max_samples:
        chosen = rng.choice(left.shape[0], size=max_samples, replace=False)
        left, right = left[chosen], right[chosen]

    left = left - left.mean(axis=0, keepdims=True)
    right = right - right.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(left.T @ right, ord="fro") ** 2
    left_norm = np.linalg.norm(left.T @ left, ord="fro")
    right_norm = np.linalg.norm(right.T @ right, ord="fro")
    return float(cross / (left_norm * right_norm + EPS))


# ------------------------------------------------------------- label readout


def knn_classifier(
    train_features: Any,
    train_labels: Any,
    test_features: Any,
    test_labels: Any,
    num_classes: int,
    k: int = 20,
    temperature: float = 0.07,
    chunk_size: int = 512,
) -> dict[str, Any]:
    """DINO's weighted cosine k-NN readout.

    Neighbour votes are weighted ``exp(cos / T)``, which is the form Caron et al.
    (2021) report and the reason the temperature matters: at ``T = 0.07`` a
    neighbour at cosine 0.9 counts ``exp(0.9/0.07) / exp(0.8/0.07) ~ 4.1`` times
    a neighbour at 0.8, so the readout is dominated by the closest few even when
    ``k`` is large.

    This is the evaluation with **no fitted parameters**: it reports the metric
    structure of the space as it stands. A probe far above the k-NN means the
    classes are linearly separable but not compactly clustered -- useful to know
    before stage 2, whose ArcFace and compactness terms act on exactly that gap.

    Returns accuracy, macro F1, the per-sample predictions and the class-score
    matrix (so AUC and calibration can be scored from the same call).
    """
    from sklearn.metrics import accuracy_score, f1_score

    bank = l2_normalize(train_features)
    queries = l2_normalize(test_features)
    bank_labels = as_float_array(train_labels).reshape(-1).astype(np.int64)
    query_labels = as_float_array(test_labels).reshape(-1).astype(np.int64)
    if bank.shape[1] != queries.shape[1]:
        raise ValueError("train and test features must share a width")

    neighbours = int(min(max(k, 1), bank.shape[0]))
    scores = np.zeros((queries.shape[0], num_classes), dtype=np.float64)

    for start in range(0, queries.shape[0], max(chunk_size, 1)):
        stop = min(start + max(chunk_size, 1), queries.shape[0])
        similarity = queries[start:stop] @ bank.T
        top = np.argpartition(-similarity, neighbours - 1, axis=1)[:, :neighbours]
        rows = np.arange(top.shape[0])[:, None]
        top_similarity = similarity[rows, top]
        weights = np.exp(top_similarity / max(temperature, EPS))
        for column in range(neighbours):
            np.add.at(
                scores,
                (np.arange(start, stop), bank_labels[top[:, column]]),
                weights[:, column],
            )

    totals = scores.sum(axis=1, keepdims=True)
    probabilities = scores / np.maximum(totals, EPS)
    predictions = scores.argmax(axis=1)
    return {
        "k": int(neighbours),
        "temperature": float(temperature),
        "accuracy": float(accuracy_score(query_labels, predictions)),
        "f1_macro": float(f1_score(query_labels, predictions, average="macro", zero_division=0)),
        "predictions": predictions.astype(np.int64),
        "scores": probabilities.astype(np.float32),
        "train_size": int(bank.shape[0]),
        "test_size": int(queries.shape[0]),
    }


def linear_probe(
    train_features: Any,
    train_labels: Any,
    test_features: Any,
    test_labels: Any,
    num_classes: int,
    regularisation: float = 1.0,
    max_iterations: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Multinomial logistic regression on frozen, L2-normalised features.

    The canonical SSL readout, and deliberately *closed-form-ish* rather than a
    second training loop: L-BFGS on cached features is convex, converges to the
    same answer every time, and removes the learning-rate/schedule/augmentation
    confounds that would otherwise sit between the encoder and its score. The
    SGD-trained counterpart already exists as
    ``experiment=baseline_linear_probe``; the two answer the same question, and
    this one answers it without a GPU.

    ``regularisation`` is sklearn's ``C``. Features are L2-normalised first, so
    one ``C`` is comparable across encoders whose feature norms differ by an order
    of magnitude -- without that, a sweep would partly be measuring feature scale.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    train = l2_normalize(train_features)
    test = l2_normalize(test_features)
    train_targets = as_float_array(train_labels).reshape(-1).astype(np.int64)
    test_targets = as_float_array(test_labels).reshape(-1).astype(np.int64)

    # No `multi_class=` argument: sklearn removed it, and L-BFGS on a
    # multi-class problem is multinomial softmax regression by default in every
    # version that still accepts the keyword. Passing it would raise on >= 1.7
    # and change nothing below that.
    model = LogisticRegression(
        C=float(regularisation),
        max_iter=int(max_iterations),
        solver="lbfgs",
        n_jobs=1,
        random_state=int(seed),
    )
    model.fit(train, train_targets)

    predictions = model.predict(test)
    # Map sklearn's column order (only the classes seen in training) onto the
    # full label space, so downstream AUC/ECE code can index by class id.
    probabilities = np.zeros((test.shape[0], num_classes), dtype=np.float64)
    probabilities[:, model.classes_.astype(np.int64)] = model.predict_proba(test)

    train_predictions = model.predict(train)
    return {
        "accuracy": float(accuracy_score(test_targets, predictions)),
        "f1_macro": float(f1_score(test_targets, predictions, average="macro", zero_division=0)),
        "train_accuracy": float(accuracy_score(train_targets, train_predictions)),
        "regularisation": float(regularisation),
        "predictions": predictions.astype(np.int64),
        "probabilities": probabilities.astype(np.float32),
        "logits": np.log(np.maximum(probabilities, 1e-12)).astype(np.float32),
        "train_size": int(train.shape[0]),
        "test_size": int(test.shape[0]),
        "classes_fitted": int(model.classes_.size),
        "iterations": int(np.max(np.atleast_1d(model.n_iter_))),
        "weight_norm": float(np.linalg.norm(model.coef_)),
    }


def select_regularisation(
    train_features: Any,
    train_labels: Any,
    val_features: Any,
    val_labels: Any,
    num_classes: int,
    grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
    max_iterations: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Pick the probe's ``C`` on a validation split, never on the test split.

    Returns the winner and the whole sweep, because "the probe was tuned" is a
    claim a reader should be able to audit, and a flat sweep is itself a result:
    it says the number is not an artefact of the regularisation choice.
    """
    sweep: list[dict[str, float]] = []
    best: tuple[float, float] = (-1.0, float(grid[0]))
    for candidate in grid:
        outcome = linear_probe(
            train_features,
            train_labels,
            val_features,
            val_labels,
            num_classes=num_classes,
            regularisation=candidate,
            max_iterations=max_iterations,
            seed=seed,
        )
        sweep.append({"C": float(candidate), "val_accuracy": outcome["accuracy"]})
        if outcome["accuracy"] > best[0]:
            best = (outcome["accuracy"], float(candidate))
    return {"best_C": best[1], "best_val_accuracy": best[0], "sweep": sweep}


def low_shot_probe(
    features: Any,
    labels: Any,
    train_indices: Any,
    test_indices: Any,
    num_classes: int,
    shots: Sequence[int] = (1, 5, 10, 25),
    repeats: int = 5,
    regularisation: float = 1.0,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Probe accuracy as a function of labels per class, with repeat variance.

    This is the measurement that says what a pretrained encoder is *for*. The
    headline probe number uses ~7.5 k labels; the practical question for a seed
    lab is what happens with five images per variety, and the answer is a
    property of the representation rather than of the classifier.

    Each shot count is sampled ``repeats`` times from ``train_indices`` with a
    different RNG stream, and the mean +- SD is reported: at one label per class
    the sampling variance is larger than any difference between encoders, so a
    single draw would be noise presented as a result.
    """
    matrix = as_float_array(features)
    targets = as_float_array(labels).reshape(-1).astype(np.int64)
    train_pool = np.asarray(train_indices, dtype=np.int64)
    test_pool = np.asarray(test_indices, dtype=np.int64)

    per_class_pool = {
        label: train_pool[targets[train_pool] == label] for label in range(num_classes)
    }

    results: list[dict[str, Any]] = []
    for shot in shots:
        accuracies: list[float] = []
        macro_f1s: list[float] = []
        for repeat in range(max(repeats, 1)):
            rng = np.random.default_rng(int(seed) + 1000 * int(shot) + repeat)
            chosen: list[int] = []
            for label, pool in per_class_pool.items():
                if pool.size == 0:
                    continue
                take = int(min(shot, pool.size))
                chosen.extend(rng.choice(pool, size=take, replace=False).tolist())
            if not chosen:
                continue
            subset = np.asarray(chosen, dtype=np.int64)
            if np.unique(targets[subset]).size < 2:
                continue
            outcome = linear_probe(
                matrix[subset],
                targets[subset],
                matrix[test_pool],
                targets[test_pool],
                num_classes=num_classes,
                regularisation=regularisation,
                seed=int(seed) + repeat,
            )
            accuracies.append(outcome["accuracy"])
            macro_f1s.append(outcome["f1_macro"])
        if not accuracies:
            continue
        results.append(
            {
                "shots": int(shot),
                "repeats": len(accuracies),
                "accuracy_mean": float(np.mean(accuracies)),
                "accuracy_std": float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else 0.0,
                "f1_macro_mean": float(np.mean(macro_f1s)),
                "f1_macro_std": float(np.std(macro_f1s, ddof=1)) if len(macro_f1s) > 1 else 0.0,
            }
        )
    return results


# ------------------------------------------------- structure without labels


def hungarian_cluster_accuracy(labels: Any, clusters: Any) -> float:
    """Best achievable accuracy over all cluster-to-class assignments.

    Purity rewards splitting a class across many clusters; this does not, because
    the assignment is one-to-one. Reporting both separates "the clusters are pure"
    from "the clustering recovered the taxonomy".
    """
    from scipy.optimize import linear_sum_assignment

    targets = as_float_array(labels).reshape(-1).astype(np.int64)
    assignments = as_float_array(clusters).reshape(-1).astype(np.int64)
    if targets.size == 0 or targets.size != assignments.size:
        return float("nan")

    size = int(max(targets.max(), assignments.max())) + 1
    contingency = np.zeros((size, size), dtype=np.int64)
    np.add.at(contingency, (assignments, targets), 1)
    rows, columns = linear_sum_assignment(-contingency)
    return float(contingency[rows, columns].sum() / targets.size)


def cluster_purity(labels: Any, clusters: Any) -> float:
    """Support-weighted fraction of each cluster taken by its majority class."""
    targets = as_float_array(labels).reshape(-1).astype(np.int64)
    assignments = as_float_array(clusters).reshape(-1).astype(np.int64)
    if targets.size == 0:
        return float("nan")
    total = 0
    for cluster in np.unique(assignments):
        members = targets[assignments == cluster]
        if members.size:
            total += int(np.bincount(members).max())
    return float(total / targets.size)


def kmeans_report(
    features: Any,
    labels: Any,
    num_clusters: int,
    seed: int = 0,
    n_init: int = 10,
) -> dict[str, float]:
    """K-means over the frozen features, scored against the labels.

    Run on L2-normalised features, so Euclidean k-means is spherical k-means and
    the partition is a cosine one -- the same geometry every other measurement
    here uses.

    ``adjusted_mutual_info`` is reported next to NMI because NMI is biased upward
    by the cluster count, and at ``k = 27`` over 27 classes that bias is not
    negligible.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import (
        adjusted_mutual_info_score,
        adjusted_rand_score,
        normalized_mutual_info_score,
    )

    matrix = l2_normalize(features)
    targets = as_float_array(labels).reshape(-1).astype(np.int64)
    clusters = int(max(num_clusters, 2))
    model = KMeans(n_clusters=clusters, n_init=int(n_init), random_state=int(seed))
    assignments = model.fit_predict(matrix)
    return {
        "num_clusters": float(clusters),
        "nmi": float(normalized_mutual_info_score(targets, assignments)),
        "adjusted_mutual_info": float(adjusted_mutual_info_score(targets, assignments)),
        "adjusted_rand": float(adjusted_rand_score(targets, assignments)),
        "purity": cluster_purity(targets, assignments),
        "cluster_accuracy": hungarian_cluster_accuracy(targets, assignments),
        "inertia": float(model.inertia_),
    }


def prototype_report(
    prototype_logits: Any,
    labels: Any,
    num_prototypes: int | None = None,
) -> dict[str, Any]:
    """What DINO's own 2,048-way prototype head learned, scored against the labels.

    This is the only measurement that looks at the objective's *output* space
    rather than at the trunk's features, and it is the closest thing available to
    "did the self-supervised task itself become the seed task?". The head is
    never used downstream -- stage 2 loads the trunk only -- so a high NMI here is
    evidence about the objective, not about the artifact being shipped.

    ``active_prototypes`` is the count that receives at least one argmax over the
    dataset. With ``K = 2048`` and 9,357 images a healthy run uses many but not
    all; a handful would mean the prototype layer collapsed even if the trunk did
    not.
    """
    from sklearn.metrics import normalized_mutual_info_score

    logits = as_float_array(prototype_logits)
    targets = as_float_array(labels).reshape(-1).astype(np.int64)
    if logits.ndim != 2:
        raise ValueError(f"prototype_logits must be [n, K], got {logits.shape}")
    total = int(num_prototypes or logits.shape[1])

    assignments = logits.argmax(axis=1)
    counts = np.bincount(assignments, minlength=total).astype(np.float64)
    shares = counts / max(counts.sum(), EPS)
    live = shares[shares > 0]
    entropy = float(-(live * np.log(live)).sum()) if live.size else 0.0

    return {
        "num_prototypes": float(total),
        "active_prototypes": float(int((counts > 0).sum())),
        "active_fraction": float((counts > 0).mean()),
        "usage_entropy_nats": entropy,
        "usage_entropy_normalized": float(entropy / np.log(total)) if total > 1 else float("nan"),
        "usage_perplexity": float(np.exp(entropy)),
        "top1_prototype_share": float(shares.max()),
        "top10_prototype_share": float(np.sort(shares)[::-1][:10].sum()),
        "nmi_vs_labels": float(normalized_mutual_info_score(targets, assignments)),
        "purity_vs_labels": cluster_purity(targets, assignments),
        "usage_shares": shares.tolist(),
    }


def retrieval_report(
    features: Any,
    labels: Any,
    k_values: Sequence[int] = (1, 5, 10),
    groups: Any | None = None,
    chunk_size: int = 512,
) -> dict[str, float]:
    """Cosine nearest-neighbour retrieval precision@k and mAP@k.

    ``groups`` is the load-bearing argument. On this dataset a crop's nearest
    neighbour is usually another crop of the **same source photograph**, so an
    ungrouped precision@1 near 1.0 measures near-duplicate matching rather than
    sub-variety similarity. Passing the source-photograph ids excludes same-group
    neighbours from every ranking, which turns the number into a statement about
    generalisation across photographs; both are returned so the gap is visible.
    """
    matrix = l2_normalize(features)
    targets = as_float_array(labels).reshape(-1).astype(np.int64)
    group_ids = as_float_array(groups).reshape(-1).astype(np.int64) if groups is not None else None
    count = matrix.shape[0]
    max_k = int(max(k_values))
    if count <= max_k + 1:
        return {}

    precision_hits = {int(k): 0.0 for k in k_values}
    average_precision = {int(k): 0.0 for k in k_values}
    scored = 0

    for start in range(0, count, max(chunk_size, 1)):
        stop = min(start + max(chunk_size, 1), count)
        similarity = matrix[start:stop] @ matrix.T
        rows = np.arange(start, stop)
        similarity[np.arange(stop - start), rows] = -np.inf
        if group_ids is not None:
            same_group = group_ids[rows][:, None] == group_ids[None, :]
            similarity[same_group] = -np.inf

        usable = np.isfinite(similarity).sum(axis=1)
        keep = usable >= max_k
        if not keep.any():
            continue
        order = np.argsort(-similarity[keep], axis=1)[:, :max_k]
        matches = targets[order] == targets[rows[keep]][:, None]
        scored += int(keep.sum())
        for k in k_values:
            k = int(k)
            window = matches[:, :k]
            precision_hits[k] += float(window.mean(axis=1).sum())
            positions = np.arange(1, k + 1)
            running = np.cumsum(window, axis=1) / positions
            denominator = np.maximum(window.sum(axis=1), 1)
            average_precision[k] += float(((running * window).sum(axis=1) / denominator).sum())

    if scored == 0:
        return {}
    report: dict[str, float] = {"queries": float(scored)}
    for k in k_values:
        k = int(k)
        report[f"precision_at_{k}"] = precision_hits[k] / scored
        report[f"map_at_{k}"] = average_precision[k] / scored
    return report
