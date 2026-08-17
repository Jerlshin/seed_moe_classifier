"""Stage-1 representation metrics, pinned against cases with known answers.

Every test here constructs a synthetic feature matrix whose correct answer is
known analytically or by construction, because that is the only way to test a
metric: an assertion that ``silhouette > 0.3`` on real features pins the encoder,
not the code. The three shapes used throughout are

* ``separated`` -- ``k`` tight, mutually orthogonal clusters. Every readout should
  be perfect and every geometry measure near its optimum.
* ``collapsed`` -- one direction plus noise. RankMe and the participation ratio
  must fall to ~1, and that is the failure mode the whole label-free family exists
  to detect.
* ``random`` -- isotropic noise with labels attached at random. Every readout must
  fall to chance, and every structure measure to ~0.

A test that fails here is reporting a broken measurement, not a worse encoder.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.representation import (
    alignment_uniformity,
    augmentation_consistency,
    centroid_similarity_matrix,
    class_separability,
    cluster_purity,
    feature_statistics,
    hungarian_cluster_accuracy,
    kmeans_report,
    knn_classifier,
    l2_normalize,
    linear_cka,
    linear_probe,
    low_shot_probe,
    prototype_report,
    retrieval_report,
    select_regularisation,
    spectral_report,
)

NUM_CLASSES = 6
PER_CLASS = 40
DIM = 24


@pytest.fixture
def separated() -> tuple[np.ndarray, np.ndarray]:
    """Tight, mutually orthogonal clusters: the "everything works" reference."""
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(NUM_CLASSES), PER_CLASS)
    centres = np.eye(NUM_CLASSES, DIM) * 6.0
    features = centres[labels] + rng.normal(scale=0.25, size=(labels.size, DIM))
    return features, labels


@pytest.fixture
def collapsed() -> tuple[np.ndarray, np.ndarray]:
    """One live direction plus isotropic noise: a dimensionally collapsed encoder."""
    rng = np.random.default_rng(1)
    labels = np.repeat(np.arange(NUM_CLASSES), PER_CLASS)
    direction = np.zeros(DIM)
    direction[0] = 1.0
    features = np.outer(rng.normal(size=labels.size), direction) * 5.0
    features += rng.normal(scale=1e-3, size=features.shape)
    return features, labels


@pytest.fixture
def unstructured() -> tuple[np.ndarray, np.ndarray]:
    """Isotropic noise with random labels: every readout must sit at chance."""
    rng = np.random.default_rng(2)
    labels = np.repeat(np.arange(NUM_CLASSES), PER_CLASS)
    return rng.normal(size=(labels.size, DIM)), labels


# ------------------------------------------------------------------ geometry


def test_l2_normalize_is_unit_norm_and_survives_zero_rows():
    matrix = np.array([[3.0, 4.0], [0.0, 0.0]])
    normalized = l2_normalize(matrix)
    assert np.isclose(np.linalg.norm(normalized[0]), 1.0)
    # A zero row has no direction; the guard must return zeros rather than NaN,
    # because a single NaN row poisons every downstream cosine.
    assert np.all(np.isfinite(normalized))
    assert np.allclose(normalized[1], 0.0)


def test_rankme_separates_full_rank_from_collapsed(separated, collapsed):
    full = spectral_report(separated[0])
    dead = spectral_report(collapsed[0])

    # Six orthogonal centres plus isotropic noise: the variance lives in ~6
    # directions, so 95 % of it is reached well before the ambient 24.
    assert full.dims_for_95_variance <= NUM_CLASSES + 1
    assert full.participation_ratio > 3.0

    # One direction: the participation ratio is the number this detects, and it
    # must be indistinguishable from 1.
    assert dead.participation_ratio < 1.2
    assert dead.top1_variance_share > 0.99
    assert dead.rankme < full.rankme


def test_spectral_report_rejects_degenerate_input():
    with pytest.raises(ValueError):
        spectral_report(np.zeros((1, 5)))


def test_feature_statistics_counts_dead_channels():
    rng = np.random.default_rng(3)
    features = rng.normal(size=(100, 10))
    features[:, 3:6] = 2.0  # constant channels: present, but carrying nothing
    report = feature_statistics(features)
    assert report["dead_dims"] == 3.0
    assert np.isclose(report["dead_dim_fraction"], 0.3)


def test_mean_direction_norm_flags_total_collapse():
    identical = np.tile(np.array([[1.0, 2.0, 3.0]]), (50, 1))
    assert feature_statistics(identical)["mean_direction_norm"] == pytest.approx(1.0)
    rng = np.random.default_rng(4)
    spread = rng.normal(size=(4000, 32))
    assert feature_statistics(spread)["mean_direction_norm"] < 0.1


def test_class_separability_orders_separated_above_random(separated, unstructured):
    good = class_separability(*separated)
    bad = class_separability(*unstructured)
    assert good["silhouette_cosine"] > 0.8
    assert abs(bad["silhouette_cosine"]) < 0.1
    assert good["fisher_ratio"] > bad["fisher_ratio"]
    # Lower Davies-Bouldin is better, which is the one metric here whose direction
    # is inverted; a test that asserted "greater" would pass on a broken sign.
    assert good["davies_bouldin"] < bad["davies_bouldin"]
    assert len(good["per_class_silhouette"]) == NUM_CLASSES


def test_centroid_similarity_matrix_is_symmetric_with_unit_diagonal(separated):
    features, labels = separated
    matrix = centroid_similarity_matrix(features, labels, NUM_CLASSES)
    assert matrix.shape == (NUM_CLASSES, NUM_CLASSES)
    assert np.allclose(np.diag(matrix), 1.0)
    assert np.allclose(matrix, matrix.T, atol=1e-9)


def test_centroid_similarity_marks_absent_classes_as_nan(separated):
    features, labels = separated
    matrix = centroid_similarity_matrix(features, labels, NUM_CLASSES + 2)
    # A class with no members has no centroid. NaN says so; a zero row would read
    # as "orthogonal to everything", which is a claim rather than an absence.
    assert np.all(np.isnan(matrix[NUM_CLASSES:, :]))


def test_linear_cka_is_one_for_identity_and_invariant_to_rotation(separated):
    features, _ = separated
    rng = np.random.default_rng(5)
    rotation = np.linalg.qr(rng.normal(size=(DIM, DIM)))[0]
    assert linear_cka(features, features) == pytest.approx(1.0, abs=1e-6)
    # Invariance to an orthogonal transform is the property that makes CKA a
    # sensible "how much did the representation change" measure at all.
    assert linear_cka(features, features @ rotation) == pytest.approx(1.0, abs=1e-6)
    assert linear_cka(features, rng.normal(size=features.shape)) < 0.3


def test_alignment_uniformity_trade_off_is_visible():
    rng = np.random.default_rng(6)
    features = rng.normal(size=(300, 16))
    tight = alignment_uniformity(features, features + rng.normal(scale=0.01, size=features.shape))
    loose = alignment_uniformity(features, rng.normal(size=features.shape))
    assert tight["alignment"] < loose["alignment"]
    assert tight["positive_pair_cosine_mean"] > 0.99

    # A collapsed encoder wins alignment outright and loses uniformity outright,
    # which is exactly why neither number is reported alone.
    collapsed = np.tile(rng.normal(size=(1, 16)), (300, 1))
    degenerate = alignment_uniformity(collapsed, collapsed)
    assert degenerate["alignment"] == pytest.approx(0.0, abs=1e-9)
    assert degenerate["uniformity"] > tight["uniformity"]


def test_augmentation_consistency_orders_the_three_populations(separated):
    features, labels = separated
    rng = np.random.default_rng(7)
    augmented = features + rng.normal(scale=0.05, size=features.shape)
    report = augmentation_consistency(features, augmented, labels)
    assert report["same_image_cosine_mean"] > report["same_class_cosine_mean"]
    assert report["same_class_cosine_mean"] > report["different_class_cosine_mean"]
    assert report["self_retrieval_top1"] == pytest.approx(1.0)


def test_augmentation_consistency_rejects_mismatched_shapes(separated):
    features, labels = separated
    with pytest.raises(ValueError):
        augmentation_consistency(features, features[:10], labels)


# ------------------------------------------------------------------- readout


def test_knn_is_perfect_on_separated_and_chance_on_noise(separated, unstructured):
    for features, labels, expected in ((separated, None, 1.0), (unstructured, None, 1.0 / NUM_CLASSES)):
        matrix, targets = features
        train = np.arange(0, matrix.shape[0], 2)
        test = np.arange(1, matrix.shape[0], 2)
        outcome = knn_classifier(
            matrix[train], targets[train], matrix[test], targets[test], NUM_CLASSES, k=5
        )
        assert outcome["accuracy"] == pytest.approx(expected, abs=0.12)
        assert outcome["scores"].shape == (test.size, NUM_CLASSES)
        # Votes are normalised, so the score rows are a distribution and can be
        # fed straight to the calibration and AUC code.
        assert np.allclose(outcome["scores"].sum(axis=1), 1.0, atol=1e-5)


def test_knn_clamps_k_to_the_bank_size(separated):
    features, labels = separated
    outcome = knn_classifier(features[:3], labels[:3], features[3:6], labels[3:6], NUM_CLASSES, k=100)
    assert outcome["k"] == 3


def test_linear_probe_is_perfect_on_separated_and_chance_on_noise(separated, unstructured):
    for (matrix, targets), expected in ((separated, 1.0), (unstructured, 1.0 / NUM_CLASSES)):
        train = np.arange(0, matrix.shape[0], 2)
        test = np.arange(1, matrix.shape[0], 2)
        outcome = linear_probe(
            matrix[train], targets[train], matrix[test], targets[test], NUM_CLASSES
        )
        assert outcome["accuracy"] == pytest.approx(expected, abs=0.15)
        assert outcome["probabilities"].shape == (test.size, NUM_CLASSES)
        assert np.allclose(outcome["probabilities"].sum(axis=1), 1.0, atol=1e-4)


def test_linear_probe_maps_unseen_classes_to_zero_probability(separated):
    """The probability matrix is indexed by *global* class id, always.

    sklearn's ``predict_proba`` has one column per class it saw in training. When
    a grouped split leaves a class entirely on one side -- which this dataset
    guarantees for five sub-varieties -- the naive assignment shifts every column
    left of the gap and silently relabels the predictions.
    """
    features, labels = separated
    keep = labels < NUM_CLASSES - 2  # train without the last two classes
    outcome = linear_probe(
        features[keep], labels[keep], features, labels, num_classes=NUM_CLASSES
    )
    assert outcome["probabilities"].shape[1] == NUM_CLASSES
    assert np.allclose(outcome["probabilities"][:, NUM_CLASSES - 2 :], 0.0)
    assert outcome["classes_fitted"] == NUM_CLASSES - 2


def test_select_regularisation_returns_a_member_of_the_grid(separated):
    features, labels = separated
    train = np.arange(0, features.shape[0], 2)
    val = np.arange(1, features.shape[0], 2)
    grid = (0.01, 1.0, 100.0)
    outcome = select_regularisation(
        features[train], labels[train], features[val], labels[val], NUM_CLASSES, grid=grid
    )
    assert outcome["best_C"] in grid
    assert len(outcome["sweep"]) == len(grid)
    assert outcome["best_val_accuracy"] == max(row["val_accuracy"] for row in outcome["sweep"])


def test_low_shot_probe_improves_with_more_labels(separated):
    features, labels = separated
    train = np.arange(0, features.shape[0], 2)
    test = np.arange(1, features.shape[0], 2)
    rows = low_shot_probe(features, labels, train, test, NUM_CLASSES, shots=(1, 10), repeats=3)
    assert [row["shots"] for row in rows] == [1, 10]
    assert rows[1]["accuracy_mean"] >= rows[0]["accuracy_mean"]
    assert all(row["repeats"] == 3 for row in rows)


def test_low_shot_probe_never_exceeds_the_available_pool(separated):
    """Asking for more shots than exist takes what there is, without raising."""
    features, labels = separated
    train = np.concatenate(
        [np.arange(index * PER_CLASS, index * PER_CLASS + 3) for index in range(NUM_CLASSES)]
    )
    test = np.setdiff1d(np.arange(features.shape[0]), train)
    rows = low_shot_probe(features, labels, train, test, NUM_CLASSES, shots=(50,), repeats=2)
    assert rows and rows[0]["shots"] == 50
    assert rows[0]["accuracy_mean"] > 0.9  # 3 per class of well-separated data


def test_low_shot_probe_skips_a_draw_that_cannot_be_fitted(separated):
    """A single-class training pool yields no row rather than a degenerate fit.

    Reachable on this dataset: a grouped split can leave a shot budget covering
    one class, and ``LogisticRegression`` on one class does not raise -- it fits a
    model that predicts that class for everything, which would enter the table as
    a real number.
    """
    features, labels = separated
    single_class = np.arange(PER_CLASS)  # all of class 0
    rows = low_shot_probe(
        features, labels, single_class, np.arange(PER_CLASS, features.shape[0]),
        NUM_CLASSES, shots=(2,), repeats=2,
    )
    assert rows == []


# ----------------------------------------------------- structure, no labels


def test_kmeans_recovers_separated_clusters(separated, unstructured):
    good = kmeans_report(*separated, num_clusters=NUM_CLASSES, seed=0)
    bad = kmeans_report(*unstructured, num_clusters=NUM_CLASSES, seed=0)
    assert good["nmi"] == pytest.approx(1.0, abs=1e-6)
    assert good["cluster_accuracy"] == pytest.approx(1.0)
    assert bad["nmi"] < 0.1
    # ARI is chance-corrected and AMI is bias-corrected; both must sit at ~0 on
    # random labels where raw NMI is merely small.
    assert abs(bad["adjusted_rand"]) < 0.05
    assert abs(bad["adjusted_mutual_info"]) < 0.05


def test_purity_and_hungarian_accuracy_disagree_on_over_clustering():
    labels = np.repeat([0, 1], 50)
    # Each true class split across two clusters: perfectly pure, but a one-to-one
    # assignment can only claim half of it. Reporting both is what distinguishes
    # "pure clusters" from "recovered the taxonomy".
    clusters = np.concatenate([np.repeat([0, 1], 25), np.repeat([2, 3], 25)])
    assert cluster_purity(labels, clusters) == pytest.approx(1.0)
    assert hungarian_cluster_accuracy(labels, clusters) == pytest.approx(0.5)


def test_hungarian_accuracy_is_permutation_invariant():
    labels = np.repeat(np.arange(4), 10)
    permuted = (labels + 2) % 4
    assert hungarian_cluster_accuracy(labels, permuted) == pytest.approx(1.0)


def test_prototype_report_detects_a_collapsed_head():
    rng = np.random.default_rng(8)
    labels = np.repeat(np.arange(4), 25)

    # Every image on one prototype: active count 1, zero entropy, NMI 0.
    collapsed = np.full((labels.size, 32), -10.0)
    collapsed[:, 0] = 10.0
    dead = prototype_report(collapsed, labels)
    assert dead["active_prototypes"] == 1.0
    assert dead["usage_entropy_nats"] == pytest.approx(0.0)
    assert dead["top1_prototype_share"] == pytest.approx(1.0)
    assert dead["nmi_vs_labels"] == pytest.approx(0.0, abs=1e-9)

    # One prototype per class: NMI 1, purity 1, and only four of 32 in use.
    aligned = rng.normal(scale=0.01, size=(labels.size, 32))
    aligned[np.arange(labels.size), labels] = 10.0
    good = prototype_report(aligned, labels)
    assert good["active_prototypes"] == 4.0
    assert good["nmi_vs_labels"] == pytest.approx(1.0, abs=1e-6)
    assert good["purity_vs_labels"] == pytest.approx(1.0)
    assert len(good["usage_shares"]) == 32


def test_retrieval_group_exclusion_changes_the_answer():
    """Same-group neighbours must be excluded, and doing so must matter.

    Two crops of one photograph sit next to each other in feature space whatever
    their labels are. This builds that situation explicitly: pairs share a group
    and a position but not a class, so ungrouped retrieval scores 0 at k=1 and
    grouped retrieval recovers the real class structure.
    """
    rng = np.random.default_rng(9)
    labels = np.repeat(np.arange(4), 20)
    groups = np.arange(labels.size) // 2  # neighbouring pairs share a photograph

    features = np.eye(4, 8)[labels] * 3.0 + rng.normal(scale=0.05, size=(labels.size, 8))
    # Force each pair to be each other's nearest neighbour, across classes.
    for pair in range(0, labels.size, 2):
        offset = rng.normal(scale=0.01, size=8) + 20.0
        features[pair] = offset
        features[pair + 1] = offset + rng.normal(scale=1e-4, size=8)
        labels[pair + 1] = (labels[pair] + 1) % 4

    ungrouped = retrieval_report(features, labels, k_values=(1,))
    grouped = retrieval_report(features, labels, k_values=(1,), groups=groups)
    assert ungrouped["precision_at_1"] < grouped["precision_at_1"]


def test_retrieval_precision_is_perfect_on_separated_clusters(separated):
    features, labels = separated
    report = retrieval_report(features, labels, k_values=(1, 5))
    assert report["precision_at_1"] == pytest.approx(1.0)
    assert report["map_at_5"] == pytest.approx(1.0)


def test_retrieval_returns_empty_when_k_exceeds_the_corpus(separated):
    features, labels = separated
    assert retrieval_report(features[:3], labels[:3], k_values=(10,)) == {}
