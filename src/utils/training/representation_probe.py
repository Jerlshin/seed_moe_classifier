"""In-training representation diagnostics, and the checkpoint choice they drive.

The problem this solves
-----------------------

Stage 1's loss is a cross entropy against a teacher that moved during training,
so it is comparable to nothing -- not across runs, not across checkpoints of one
run, not against a threshold. Measured on the shipped 100-epoch run, **94.8 % of
the final loss was irreducible target entropy** and 80 % of its total drop was
that entropy falling rather than the student learning. Choosing a checkpoint by
"lowest loss" therefore selects on the teacher's temperature schedule.

What actually happened on that run, from ``tables/milestone_progression.csv``:

======= ===================== ================
epoch   27-way probe accuracy k-NN accuracy
======= ===================== ================
25      0.6276                0.5019
50      **0.6358**            0.5017
100     0.6284                0.5003
======= ===================== ================

The best encoder was at **epoch 50**, the run went to 100, and nothing in the
training loop could have known -- the milestone probes were run weeks later by a
separate process. Two of those epochs' worth of GPU time bought a *worse*
encoder than the one already on disk.

This module closes that loop. It extracts frozen features from the *current*
student trunk on a held-out-free, augmentation-free pass over the corpus, scores
a small battery of diagnostics that are known to track downstream transfer, and
hands the trainer a single comparable number per milestone. The trainer then
keeps the best encoder under its own name, and can stop when the number stops
improving.

Why these diagnostics
---------------------

Three families that fail independently, which is why no one of them is the
selection metric on its own (``architecture/08`` has the long version):

* **Frozen readout** -- a linear probe and a parameter-free weighted cosine k-NN.
  This is the quantity stage 2 actually consumes, and the default selection
  metric. Scored under the **same crop-level stratified protocol the primary
  pipeline uses**, so the number the selector optimises and the number the
  pipeline reports are the same measurement.
* **Label-free geometry** -- RankMe, participation ratio, stable rank,
  alignment and uniformity. These detect the two failure modes a probe cannot:
  dimensional collapse (which a probe on 9k samples can hide) and a
  uniformity-dominated solution that scores well on the probe while destroying
  the invariance the objective asked for.
* **Nuisance** -- within-sub-variety source-photograph decodability. Stage 1's
  one demonstrated effect on the shipped run was cutting this from +10.0 pp
  above chance to +3.5 pp. It is also the **gate**: an arm that wins the readout
  while pushing this back up has learned the photograph confound. Read jointly
  with the readout -- an encoder that discards everything scores chance here and
  is not thereby good.

Cost
----

One forward pass per image at ``probe.max_samples`` images, no augmentation, no
gradient, under inference mode and the run's own autocast policy, plus a few
seconds of scikit-learn. On SwinV2-Tiny at 256 px that is a few seconds of GPU
per probe against minutes per epoch, so probing every few epochs is free
relative to the run. It is deliberately **not** free enough to run every epoch by
default: ``probe.every_epochs`` exists so the cost is a decision.

Determinism
-----------

Feature extraction runs under a forked RNG and a fixed subsample seed, so a probe
never perturbs the augmentation stream and two runs of the same configuration
probe the same images. The probe's own folds come from ``StratifiedKFold`` with
an explicit ``random_state``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)

#: Selection metrics the trainer understands, and the direction that is better.
#:
#: ``probe_accuracy`` is the default because it is the quantity stage 2 consumes.
#: ``knn_accuracy`` is the parameter-free alternative -- it cannot be won by a
#: linearly-separable-but-badly-clustered space, which is a real failure mode
#: here. ``teacher_student_kl`` is included so "select on the objective" is
#: available as an explicit *control*, not because it is a good idea: see the
#: module docstring for why the loss cannot rank checkpoints.
SELECTION_METRICS: dict[str, str] = {
    "probe_accuracy": "max",
    "probe_f1_macro": "max",
    "knn_accuracy": "max",
    "probe_plus_knn": "max",
    "teacher_student_kl": "min",
}


@dataclass
class ProbeResult:
    """One milestone's diagnostics, flattened for a CSV row."""

    epoch: int
    global_step: int
    metrics: dict[str, float] = field(default_factory=dict)
    seconds: float = 0.0

    def row(self, **extra: Any) -> dict[str, Any]:
        return {
            "epoch": int(self.epoch),
            "global_step": int(self.global_step),
            "probe_seconds": float(self.seconds),
            **{key: float(value) for key, value in self.metrics.items()},
            **extra,
        }

    def value(self, metric: str) -> float:
        return float(self.metrics.get(metric, float("nan")))


@torch.inference_mode()
def extract_features(
    backbone: torch.nn.Module,
    loader: Any,
    device: torch.device,
    amp_context: Any = None,
    max_batches: int | None = None,
) -> dict[str, np.ndarray]:
    """Frozen features, labels and source groups over an evaluation loader.

    The trunk is put in ``eval()`` for the pass and restored to whatever mode it
    was in afterwards. That restore is load-bearing: stage 1's teacher is a
    ``deepcopy`` of the student that ``model.train()`` puts in training mode, and
    a probe that left the student in ``eval()`` would silently disable stochastic
    depth for the rest of the run.

    Returns ``pooled`` features plus ``sub_labels``, ``seed_labels`` and
    ``source_groups`` as numpy arrays in loader order.
    """
    from contextlib import nullcontext

    was_training = backbone.training
    backbone.eval()
    features: list[np.ndarray] = []
    sub_labels: list[np.ndarray] = []
    seed_labels: list[np.ndarray] = []
    try:
        for index, batch in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            images, seed_target, sub_target = batch[0], batch[1], batch[2]
            images = images.to(device, non_blocking=True)
            with (amp_context() if amp_context is not None else nullcontext()):
                output = backbone(images)
            # A trunk may emit tokens or a pooled vector depending on
            # `feature_stage`; the probe wants one vector per image either way.
            if output.ndim == 3:
                output = output.mean(dim=1)
            features.append(output.detach().float().cpu().numpy())
            sub_labels.append(np.asarray(sub_target).reshape(-1))
            seed_labels.append(np.asarray(seed_target).reshape(-1))
    finally:
        backbone.train(was_training)

    return {
        "pooled": np.concatenate(features, axis=0) if features else np.zeros((0, 0), np.float32),
        "sub_labels": (
            np.concatenate(sub_labels, axis=0).astype(np.int64) if sub_labels else np.zeros(0, np.int64)
        ),
        "seed_labels": (
            np.concatenate(seed_labels, axis=0).astype(np.int64) if seed_labels else np.zeros(0, np.int64)
        ),
    }


def stratified_readout(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    folds: int = 3,
    regularisation: float = 10.0,
    max_iterations: int = 500,
    knn_k: int = 10,
    knn_temperature: float = 0.07,
    seed: int = 42,
) -> dict[str, float]:
    """Out-of-fold linear probe and k-NN under **crop-level stratified** folds.

    The protocol is the primary pipeline's, deliberately: the number this returns
    and the number stage 2 reports are then the same measurement under the same
    splitting rule, so "the selector picked the encoder that scores best
    downstream" is a claim about one protocol rather than a comparison across
    two.

    That also means it inherits the protocol's known property -- crops from one
    source photograph appear on both sides, so the absolute value is ~18 pp above
    what a photograph-disjoint split would give. It is used here to **rank
    checkpoints of one run**, where that offset is common to every candidate and
    cancels. Do not quote it as a generalisation estimate.

    ``max_iterations`` defaults low and ``folds`` to 3 because this runs inside a
    training loop: the ranking is stable long before L-BFGS has fully converged,
    and the alternative is a probe that costs more than the epochs it is meant to
    save.
    """
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedKFold

    from src.utils.representation import knn_classifier, linear_probe

    if features.shape[0] < folds * 2:
        return {}

    # A class with fewer members than there are folds cannot be stratified.
    counts = np.bincount(labels, minlength=num_classes)
    usable = int(min(folds, int(counts[counts > 0].min())))
    if usable < 2:
        return {}

    splitter = StratifiedKFold(n_splits=usable, shuffle=True, random_state=int(seed))
    probe_predictions = np.zeros_like(labels)
    knn_predictions = np.zeros_like(labels)
    train_accuracies: list[float] = []

    for train_index, test_index in splitter.split(features, labels):
        probe = linear_probe(
            features[train_index],
            labels[train_index],
            features[test_index],
            labels[test_index],
            num_classes=num_classes,
            regularisation=regularisation,
            max_iterations=max_iterations,
            seed=int(seed),
        )
        probe_predictions[test_index] = probe["predictions"]
        train_accuracies.append(probe["train_accuracy"])

        neighbours = knn_classifier(
            features[train_index],
            labels[train_index],
            features[test_index],
            labels[test_index],
            num_classes=num_classes,
            k=knn_k,
            temperature=knn_temperature,
        )
        knn_predictions[test_index] = neighbours["predictions"]

    probe_accuracy = float(accuracy_score(labels, probe_predictions))
    knn_accuracy = float(accuracy_score(labels, knn_predictions))
    train_accuracy = float(np.mean(train_accuracies)) if train_accuracies else float("nan")
    return {
        "probe_accuracy": probe_accuracy,
        "probe_f1_macro": float(
            f1_score(labels, probe_predictions, average="macro", zero_division=0)
        ),
        "probe_train_accuracy": train_accuracy,
        # Positive means the probe fits the training fold better than the test
        # fold, which on 9k crops and 768 dimensions it always will. Watch the
        # *trend* across milestones: a gap that widens while the probe accuracy
        # is flat is the encoder becoming more separable on its own corpus and
        # no more transferable.
        "probe_generalisation_gap": train_accuracy - probe_accuracy,
        "knn_accuracy": knn_accuracy,
        "knn_f1_macro": float(
            f1_score(labels, knn_predictions, average="macro", zero_division=0)
        ),
        # A composite for `selection_metric: probe_plus_knn`. The two disagree
        # exactly when the space is linearly separable but not compactly
        # clustered, which is the regime stage 2's ArcFace and compactness terms
        # act on -- so averaging them is a defensible way to not be fooled by
        # either alone. It is not the default, because a composite hides which
        # half moved.
        "probe_plus_knn": 0.5 * (probe_accuracy + knn_accuracy),
        "readout_folds": float(usable),
    }


def geometry_metrics(features: np.ndarray) -> dict[str, float]:
    """Label-free spectrum statistics: RankMe, participation ratio, stable rank."""
    from src.utils.representation import feature_statistics, spectral_report

    report = spectral_report(features)
    statistics = feature_statistics(features)
    return {
        "rankme": float(report.rankme),
        "rankme_normalized": float(report.rankme_normalized),
        "participation_ratio": float(report.participation_ratio),
        "stable_rank": float(report.stable_rank),
        "dims_for_95_variance": float(report.dims_for_95_variance),
        "top1_variance_share": float(report.top1_variance_share),
        "top10_variance_share": float(report.top10_variance_share),
        "feature_mean_norm": float(statistics["mean_norm"]),
        "feature_dead_dim_fraction": float(statistics["dead_dim_fraction"]),
        # ||mean of the unit vectors||. 1.0 is total collapse; a value that
        # climbs across milestones while the probe holds is the representation
        # concentrating on one direction, which RankMe alone can miss.
        "feature_mean_direction_norm": float(statistics["mean_direction_norm"]),
    }


class RepresentationProbe:
    """Runs the diagnostics and remembers what each milestone scored.

    ``probe.max_samples`` subsamples the **dataset**, not the extracted features:
    the caller builds the loader over a fixed random subset, so the forward pass
    itself gets cheaper. Truncating the loader instead would be wrong here rather
    than merely crude -- ``ImageFolder`` enumerates in class order, so the first
    N images are the first few sub-varieties and the probe would be scoring a
    different, easier task.

    Args:
        loader_factory: Zero-argument callable returning a fresh evaluation
            dataloader. A factory rather than a loader because a persistent
            worker pool held open for the whole run, for a pass that happens
            every few epochs, is worker memory the training loader wants.
        num_classes: Sub-variety count, for the probe's label space.
        source_groups: Source-photograph id per sample, **in the order the
            loader yields them**, for the nuisance measurement. ``None`` skips
            it. When the loader is over a subset, these must be the subset's
            groups -- a full-corpus array here would silently pair each feature
            with another image's provenance.
        config: The ``experiment.training.probe`` node.
    """

    def __init__(
        self,
        loader_factory: Any,
        num_classes: int,
        source_groups: np.ndarray | None = None,
        config: Mapping[str, Any] | None = None,
        logger: logging.Logger | None = None,
    ):
        self.loader_factory = loader_factory
        self.num_classes = int(num_classes)
        self.source_groups = source_groups
        self.logger = logger or LOGGER
        options = dict(config or {})
        self.folds = int(options.get("folds", 3) or 3)
        self.regularisation = float(options.get("regularisation", 10.0) or 10.0)
        self.max_iterations = int(options.get("max_iterations", 500) or 500)
        self.knn_k = int(options.get("knn_k", 10) or 10)
        self.seed = int(options.get("seed", 42) or 42)
        self.max_samples = int(options.get("max_samples", 0) or 0)
        self.geometry = bool(options.get("geometry", True))
        self.nuisance = bool(options.get("nuisance", True))
        self.nuisance_folds = int(options.get("nuisance_folds", 3) or 3)
        self.history: list[ProbeResult] = []

    def run(
        self,
        backbone: torch.nn.Module,
        device: torch.device,
        epoch: int,
        global_step: int,
        amp_context: Any = None,
    ) -> ProbeResult:
        """Extract features and score them. Never raises into the training loop."""
        started = time.perf_counter()
        metrics: dict[str, float] = {}
        try:
            # Fork the global RNG. Nothing in the probe should sample, but a
            # dataloader with `shuffle=False` still constructs a generator, and a
            # diagnostic that shifted the augmentation stream would make the run
            # depend on how often it was probed.
            with torch.random.fork_rng(devices=[]):
                torch.random.manual_seed(self.seed)
                extracted = extract_features(
                    backbone, self.loader_factory(), device, amp_context=amp_context
                )
            metrics = self._score(extracted)
        except Exception as exc:  # pragma: no cover - a diagnostic must not kill a run
            self.logger.warning(
                "Representation probe at epoch %s failed (%s); training continues and this "
                "milestone carries no probe metrics.",
                epoch, exc, exc_info=True,
            )
        result = ProbeResult(
            epoch=int(epoch),
            global_step=int(global_step),
            metrics=metrics,
            seconds=time.perf_counter() - started,
        )
        self.history.append(result)
        return result

    def _score(self, extracted: Mapping[str, np.ndarray]) -> dict[str, float]:
        features = np.asarray(extracted["pooled"], dtype=np.float64)
        labels = np.asarray(extracted["sub_labels"], dtype=np.int64)
        seeds = np.asarray(extracted["seed_labels"], dtype=np.int64)
        groups = self.source_groups
        if features.shape[0] == 0:
            return {}

        metrics: dict[str, float] = {"probe_samples": float(features.shape[0])}
        metrics.update(
            stratified_readout(
                features,
                labels,
                num_classes=self.num_classes,
                folds=self.folds,
                regularisation=self.regularisation,
                max_iterations=self.max_iterations,
                knn_k=self.knn_k,
                seed=self.seed,
            )
        )
        # The 4-way parent task, essentially free once the features exist. It
        # saturates near 0.98 for every encoder including ImageNet, so it is a
        # sanity floor rather than a discriminator -- a run whose seed-type
        # accuracy falls has broken something structural.
        seed_readout = stratified_readout(
            features,
            seeds,
            num_classes=int(seeds.max()) + 1 if seeds.size else 1,
            folds=self.folds,
            regularisation=self.regularisation,
            max_iterations=self.max_iterations,
            knn_k=self.knn_k,
            seed=self.seed,
        )
        metrics.update(
            {
                "seed_probe_accuracy": seed_readout.get("probe_accuracy", float("nan")),
                "seed_knn_accuracy": seed_readout.get("knn_accuracy", float("nan")),
            }
        )

        if self.geometry:
            metrics.update(geometry_metrics(features))
            metrics.update(self._class_structure(features, labels))

        if self.nuisance and groups is not None and groups.size == labels.size:
            from src.utils.representation import nuisance_decodability

            report = nuisance_decodability(
                features,
                labels,
                groups,
                folds=self.nuisance_folds,
                seed=self.seed,
            )
            metrics.update(
                {
                    "nuisance_photograph_accuracy": float(report["within_class_photo_accuracy"]),
                    "nuisance_chance": float(report["chance"]),
                    # The number to read. Stage 1's one demonstrated effect on
                    # the shipped run was driving this from +10.0 pp to +3.5 pp.
                    "nuisance_above_chance": float(report["above_chance"]),
                    "nuisance_classes_scored": float(report["classes_scored"]),
                }
            )
        return metrics

    @staticmethod
    def _class_structure(features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        from src.utils.representation import class_separability, kmeans_report

        out: dict[str, float] = {}
        try:
            separability = class_separability(features, labels)
            out["silhouette_sub"] = float(separability["silhouette_cosine"])
            out["fisher_ratio_sub"] = float(separability["fisher_ratio"])
            out["mean_intra_class_cosine"] = float(separability["mean_intra_class_cosine"])
            out["mean_inter_centroid_cosine"] = float(separability["mean_inter_centroid_cosine"])
        except Exception:  # pragma: no cover - degenerate feature matrices
            pass
        try:
            clusters = kmeans_report(features, labels, num_clusters=int(labels.max()) + 1)
            out["kmeans_nmi"] = float(clusters["nmi"])
            out["kmeans_ari"] = float(clusters["adjusted_rand"])
            out["kmeans_cluster_accuracy"] = float(clusters["cluster_accuracy"])
        except Exception:  # pragma: no cover
            pass
        return out


class CheckpointSelector:
    """Keeps the best encoder by a representation metric, and says when to stop.

    Two jobs the training loop cannot do from the loss:

    **Selection.** The best encoder of the shipped run was at epoch 50 of 100 and
    nothing knew. Every probed epoch is compared on ``metric``; the winner is
    copied to a stable filename, so "the best encoder" is a path that exists at
    the end of the run rather than a number in a table someone has to act on.

    **Stopping.** ``patience`` consecutive probes without a new best ends the run.
    This is *not* early stopping on a validation loss -- there is no validation
    loss here, and the objective's own loss keeps improving long after the
    representation stops. It is "the diagnostics have plateaued, and the shipped
    run's own milestone curve says they do plateau".

    ``min_delta`` guards against stopping on noise: the probe's fold-to-fold
    spread is on the order of a percentage point, so an improvement smaller than
    that is not evidence of anything.
    """

    def __init__(
        self,
        metric: str = "probe_accuracy",
        patience: int = 0,
        min_delta: float = 0.0,
        logger: logging.Logger | None = None,
    ):
        if metric not in SELECTION_METRICS:
            raise ValueError(
                f"selection metric must be one of {sorted(SELECTION_METRICS)}, got {metric!r}"
            )
        self.metric = metric
        self.direction = SELECTION_METRICS[metric]
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.logger = logger or LOGGER
        self.best_value: float = float("-inf") if self.direction == "max" else float("inf")
        self.best_epoch: int = -1
        self.since_improvement: int = 0
        self.rows: list[dict[str, Any]] = []

    def _improves(self, value: float) -> bool:
        if value != value:  # NaN never wins
            return False
        if self.direction == "max":
            return value > self.best_value + self.min_delta
        return value < self.best_value - self.min_delta

    def consider(self, result: ProbeResult) -> bool:
        """Record ``result``; return ``True`` when it is the new best."""
        value = result.value(self.metric)
        improved = self._improves(value)
        if improved:
            self.best_value, self.best_epoch = value, int(result.epoch)
            self.since_improvement = 0
        elif value == value:
            self.since_improvement += 1
        self.rows.append(
            result.row(
                selection_metric=self.metric,
                selection_value=value,
                is_best=int(improved),
                best_so_far=self.best_value,
                best_epoch=self.best_epoch,
                probes_since_improvement=self.since_improvement,
            )
        )
        return improved

    def should_stop(self) -> bool:
        """True once ``patience`` probes have passed with no improvement."""
        return self.patience > 0 and self.since_improvement >= self.patience

    def summary(self) -> dict[str, Any]:
        return {
            "selection_metric": self.metric,
            "selection_direction": self.direction,
            "best_value": self.best_value if self.best_epoch >= 0 else float("nan"),
            "best_epoch": self.best_epoch,
            "probes": len(self.rows),
            "probes_since_improvement": self.since_improvement,
            "patience": self.patience,
            "min_delta": self.min_delta,
        }


def publish_best_encoder(
    source: str | Path,
    destination: str | Path,
    logger: logging.Logger | None = None,
) -> str | None:
    """Copy the winning milestone encoder to a stable filename, atomically.

    A plain ``shutil.copy`` truncates its destination first, so a job killed
    mid-copy leaves a zero-length file where the previous best used to be --
    the same failure ``atomic_save`` exists to prevent for checkpoints.
    """
    import os
    import shutil

    log = logger or LOGGER
    source_path, destination_path = Path(source), Path(destination)
    if not source_path.exists():
        log.warning("Cannot publish best encoder: %s does not exist.", source_path)
        return None
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_suffix(destination_path.suffix + ".tmp")
    try:
        shutil.copyfile(source_path, temporary)
        os.replace(temporary, destination_path)
    except Exception as exc:  # pragma: no cover
        log.warning("Unable to publish best encoder %s: %s", source_path, exc)
        temporary.unlink(missing_ok=True)
        return None
    return str(destination_path)


def milestone_table(history: Sequence[ProbeResult]) -> list[dict[str, Any]]:
    """Every probe as a row, for ``csv/probe_history.csv``."""
    return [result.row() for result in history]
