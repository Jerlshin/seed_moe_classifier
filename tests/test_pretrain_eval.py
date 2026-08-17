"""The stage-1 evaluation stage: protocol, provenance, and cache safety.

The metrics themselves are covered by ``test_representation.py``. What is covered
here is the machinery that decides *which* numbers get computed and whether they
describe the checkpoint the report names -- the parts whose failure modes are
silent:

* a feature cache that serves stale features after a retrain,
* an out-of-fold protocol that quietly drops crops or leaks a photograph across
  the fold boundary,
* an encoder list that admits two primaries, or none,
* an event-stream parser that mixes the step and epoch axes.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest

from src.trainers.pretrain_eval import (
    EncoderSpec,
    flatten_metrics,
    grouped_cv_readout,
    json_safe,
    load_cached_features,
    resolve_encoder_specs,
    sha256_of,
    strip_private,
    subsample_dataset,
    write_csv,
)
from src.utils.evaluation import parse_pretrain_dynamics

LOGGER = logging.getLogger("test")


# ------------------------------------------------------------- feature cache


def write_cache(path, digest: str, rows: int = 40, dim: int = 8, stages: bool = False) -> None:
    payload = {
        "pooled": np.zeros((rows, dim), dtype=np.float32),
        "checkpoint_sha256": np.array(digest),
    }
    if stages:
        payload["stage_stage1"] = np.zeros((rows, 4), dtype=np.float32)
    np.savez_compressed(path, **payload)


def test_cache_is_reused_only_for_the_same_checkpoint_digest(tmp_path):
    """Keyed on the bytes, not the filename.

    ``dino_backbone_epoch_0100.pth`` is a name every pretraining run reuses. A
    cache keyed on the path serves the previous run's features after a retrain,
    and every number in the report is then about weights that no longer exist --
    with nothing anywhere to say so.
    """
    path = tmp_path / "encoder.npz"
    write_cache(path, digest="abc123")

    hit = load_cached_features(path, "abc123", expected_samples=40, need_stages=False, logger=LOGGER)
    assert hit is not None and hit[0].shape == (40, 8)

    assert load_cached_features(path, "different", 40, False, LOGGER) is None


def test_cache_is_rejected_when_the_row_count_moved(tmp_path):
    path = tmp_path / "encoder.npz"
    write_cache(path, digest="abc123", rows=40)
    assert load_cached_features(path, "abc123", expected_samples=41, need_stages=False, logger=LOGGER) is None


def test_cache_is_rejected_when_stage_features_are_needed_but_absent(tmp_path):
    path = tmp_path / "encoder.npz"
    write_cache(path, digest="abc123", stages=False)
    assert load_cached_features(path, "abc123", 40, need_stages=True, logger=LOGGER) is None

    with_stages = tmp_path / "with_stages.npz"
    write_cache(with_stages, digest="abc123", stages=True)
    hit = load_cached_features(with_stages, "abc123", 40, need_stages=True, logger=LOGGER)
    assert hit is not None and "stage1" in hit[1]


def test_unreadable_cache_degrades_to_recomputation(tmp_path):
    path = tmp_path / "broken.npz"
    path.write_bytes(b"not an npz")
    assert load_cached_features(path, "abc123", 40, False, LOGGER) is None


def test_sha256_of_missing_file_is_empty_not_an_error(tmp_path):
    assert sha256_of(tmp_path / "absent.pth") == ""
    real = tmp_path / "present.bin"
    real.write_bytes(b"hello")
    digest = sha256_of(real)
    assert len(digest) == 64
    # Streamed in chunks; a second call must agree with the first.
    assert digest == sha256_of(real)


# --------------------------------------------------------- out-of-fold protocol


@pytest.fixture
def grouped_features():
    """Six classes, three photographs each, features that separate the classes."""
    rng = np.random.default_rng(0)
    labels, groups = [], []
    for label in range(6):
        for photograph in range(3):
            labels.extend([label] * 10)
            groups.extend([label * 3 + photograph] * 10)
    labels = np.array(labels)
    groups = np.array(groups)
    features = np.eye(6, 12)[labels] * 5.0 + rng.normal(scale=0.3, size=(labels.size, 12))
    return features, labels, groups


def test_out_of_fold_covers_every_crop_and_every_class(grouped_features):
    features, labels, groups = grouped_features
    report = grouped_cv_readout(
        features, labels, groups,
        num_classes=6, num_folds=3, regularisation=1.0, max_iterations=500,
        knn_k=5, knn_temperature=0.07, seed=0,
    )
    assert report["coverage"] == pytest.approx(1.0)
    assert report["folds"] == 3
    # Every fold sees every class, which is the whole reason for stratifying.
    assert all(count == 6 for count in report["classes_present_per_fold"])
    assert report["out_of_fold_accuracy"] > 0.9
    assert np.all(report["_predictions"] >= 0)


def test_out_of_fold_never_lets_a_photograph_straddle_its_fold(grouped_features):
    """The property that makes the number honest, verified rather than assumed.

    Re-derived here from the same splitter the readout uses: if any photograph
    contributed crops to both halves of any fold, the readout would be scoring
    near-duplicates it had trained on.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    _, labels, groups = grouped_features
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=0)
    for train_index, val_index in splitter.split(np.zeros(labels.size), labels, groups):
        assert not set(groups[train_index]) & set(groups[val_index])


def test_out_of_fold_accuracy_is_pooled_not_averaged(grouped_features):
    """The pooled score is one metric over all predictions, not a mean of folds.

    They differ whenever folds have unequal sizes, and the pooled figure is the
    one the confusion matrix is consistent with -- so a report that quoted the mean
    beside that matrix would be quoting two different numbers.
    """
    features, labels, groups = grouped_features
    report = grouped_cv_readout(
        features, labels, groups,
        num_classes=6, num_folds=3, regularisation=1.0, max_iterations=500,
        knn_k=5, knn_temperature=0.07, seed=0,
    )
    covered = report["_predictions"] >= 0
    recomputed = float((report["_predictions"][covered] == labels[covered]).mean())
    assert report["out_of_fold_accuracy"] == pytest.approx(recomputed)


# ------------------------------------------------------------- encoder specs


def make_cfg(entries):
    from omegaconf import OmegaConf

    return OmegaConf.create({"experiment": {"evaluation": {"encoders": entries}}})


def test_exactly_one_primary_encoder_is_required(tmp_path):
    checkpoint = tmp_path / "a.pth"
    checkpoint.write_bytes(b"x")
    two_primaries = make_cfg(
        [
            {"label": "a", "checkpoint": str(checkpoint), "role": "primary"},
            {"label": "b", "checkpoint": str(checkpoint), "role": "primary"},
        ]
    )
    with pytest.raises(ValueError, match="primary"):
        resolve_encoder_specs(two_primaries, LOGGER)

    with pytest.raises(ValueError, match="primary"):
        resolve_encoder_specs(
            make_cfg([{"label": "a", "checkpoint": str(checkpoint), "role": "milestone"}]), LOGGER
        )


def test_a_missing_milestone_is_skipped_not_fatal(tmp_path):
    """A pruned epoch-25 encoder costs one table row, not the whole evaluation."""
    checkpoint = tmp_path / "primary.pth"
    checkpoint.write_bytes(b"x")
    cfg = make_cfg(
        [
            {"label": "primary", "checkpoint": str(checkpoint), "role": "primary"},
            {"label": "gone", "checkpoint": str(tmp_path / "absent.pth"), "role": "milestone"},
        ]
    )
    specs = resolve_encoder_specs(cfg, LOGGER)
    assert [spec.label for spec in specs] == ["primary"]


def test_no_evaluable_encoders_is_an_explicit_failure():
    with pytest.raises(FileNotFoundError, match="No evaluable encoders"):
        resolve_encoder_specs(make_cfg([]), LOGGER)


def test_specs_without_a_checkpoint_are_kept(tmp_path):
    """``imagenet_init`` and ``random_init`` have no file, and must survive."""
    checkpoint = tmp_path / "primary.pth"
    checkpoint.write_bytes(b"x")
    cfg = make_cfg(
        [
            {"label": "primary", "checkpoint": str(checkpoint), "role": "primary"},
            {"label": "imagenet_init", "checkpoint": None, "pretrained": True, "role": "control_imagenet"},
            {"label": "random_init", "checkpoint": None, "pretrained": False, "role": "control"},
        ]
    )
    specs = resolve_encoder_specs(cfg, LOGGER)
    assert [spec.label for spec in specs] == ["primary", "imagenet_init", "random_init"]
    assert specs[1].pretrained is True and specs[1].checkpoint is None


# ------------------------------------------------------------------ plumbing


def test_subsampling_is_class_balanced_and_reports_that_it_happened(synthetic_dataset_root):
    from src.datasets.dataset import get_finetune_dataset

    dataset = get_finetune_dataset(str(synthetic_dataset_root), transform=None)
    total = len(dataset.samples)
    classes = len(dataset.subvariety_to_idx)

    assert subsample_dataset(dataset, max_samples=0, seed=0, logger=LOGGER) is False
    assert len(dataset.samples) == total

    assert subsample_dataset(dataset, max_samples=classes * 2, seed=0, logger=LOGGER) is True
    labels = np.array([sub for _, _, sub in dataset.samples])
    # Every class still present, and no class over-represented: a subsample that
    # dropped classes would change what the 27-way numbers even mean.
    assert np.unique(labels).size == classes
    assert np.bincount(labels).max() == np.bincount(labels).min()


def test_strip_private_removes_working_arrays_but_keeps_scalars():
    report = {
        "enc": {
            "probe": {"test_accuracy": 0.5, "_predictions": np.zeros(3)},
            "list": [{"keep": 1, "_drop": 2}],
        }
    }
    cleaned = strip_private(report)
    assert cleaned["enc"]["probe"] == {"test_accuracy": 0.5}
    assert cleaned["enc"]["list"] == [{"keep": 1}]


def test_flatten_metrics_drops_curves_and_keeps_scalars():
    reports = {
        "enc": {
            "spectrum": {"rankme": 12.5, "singular_values": [1.0, 0.5]},
            "probe": {"sub_variety": {"test_accuracy": 0.4, "regularisation_sweep": [{"C": 1.0}]}},
            "cka_vs_imagenet_init": 0.3,
        }
    }
    flat = flatten_metrics(reports, {"loss_final": 5.6})
    assert flat["enc/spectrum/rankme"] == pytest.approx(12.5)
    assert flat["enc/probe/sub_variety/test_accuracy"] == pytest.approx(0.4)
    assert flat["enc/cka_vs_imagenet_init"] == pytest.approx(0.3)
    assert flat["stage1_dynamics/loss_final"] == pytest.approx(5.6)
    # A tracker scalar per singular value is 768 useless rows.
    assert not any("singular_values" in key for key in flat)


def test_flatten_metrics_drops_nan_rather_than_logging_it():
    flat = flatten_metrics({"enc": {"spectrum": {"rankme": float("nan"), "ok": 1.0}}}, {})
    assert "enc/spectrum/rankme" not in flat
    assert flat["enc/spectrum/ok"] == pytest.approx(1.0)


def test_json_safe_handles_numpy_and_nan():
    payload = json_safe(
        {"a": np.float32(1.5), "b": np.arange(3), "c": float("nan"), "d": {"e": np.int64(2)}}
    )
    assert payload == {"a": 1.5, "b": [0, 1, 2], "c": None, "d": {"e": 2}}
    json.dumps(payload)  # must not raise


def test_write_csv_unions_columns_and_blanks_missing_cells(tmp_path):
    path = tmp_path / "table.csv"
    write_csv(path, [{"a": 1.0, "b": "x"}, {"a": float("nan"), "c": 2}])
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "a,b,c"
    # NaN is written blank: in a results table it reads as a failed run rather
    # than as "not measured".
    assert lines[2].startswith(",,")


# --------------------------------------------------------- dynamics recovery


def test_dynamics_parser_separates_the_step_and_epoch_axes(tmp_path):
    """``epoch/*`` is re-indexed 1..N; ``train/*`` keeps the global step.

    Both are logged with the tracker's ``step`` argument, which for epoch metrics
    is the epoch number -- so plotting them on one axis would put epoch 3 next to
    optimizer step 3.
    """
    path = tmp_path / "events.jsonl"
    records = [
        {"type": "backend", "device": "cuda"},
        {"type": "metrics", "step": 0, "metrics": {"train/loss": 7.0, "train/teacher_entropy": 6.0}},
        {"type": "metrics", "step": 1, "metrics": {"epoch/loss": 7.5, "epoch/duration_seconds": 100.0}},
        {"type": "metrics", "step": 250, "metrics": {"train/loss": 6.0, "train/teacher_entropy": 5.0}},
        {"type": "metrics", "step": 2, "metrics": {"epoch/loss": 6.5, "epoch/duration_seconds": 110.0}},
        "{not json",
    ]
    path.write_text(
        "\n".join(record if isinstance(record, str) else json.dumps(record) for record in records),
        encoding="utf-8",
    )

    dynamics = parse_pretrain_dynamics(path)
    assert dynamics.epoch_series["epoch/loss"] == ([1.0, 2.0], [7.5, 6.5])
    assert dynamics.step_series["train/loss"] == ([0.0, 250.0], [7.0, 6.0])
    assert "backend" in dynamics.events

    summary = dynamics.summary()
    assert summary["epochs_completed"] == 2
    assert summary["loss_final"] == pytest.approx(6.5)
    assert summary["loss_improvement_total"] == pytest.approx(1.0)
    assert summary["training_hours"] == pytest.approx(210.0 / 3600.0)


def test_dynamics_parser_tolerates_a_missing_file(tmp_path):
    dynamics = parse_pretrain_dynamics(tmp_path / "nothing.jsonl")
    assert dynamics.series("epoch/loss") == ([], [])
    assert dynamics.summary()["epochs_completed"] == 0


def test_encoder_spec_round_trips_to_a_json_safe_dict():
    spec = EncoderSpec(label="a", checkpoint="/tmp/a.pth", role="primary", description="d")
    payload = spec.as_dict()
    json.dumps(payload)
    assert payload["role"] == "primary"
