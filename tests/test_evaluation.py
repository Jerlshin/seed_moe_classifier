"""Cross-run reporting: prediction dumps, the summary table, publication figures.

These cover the on-disk contract between a training run and
``scripts/generate_plots.py``. A break here means figures or a results table
silently stop reflecting the runs that produced them.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from src.utils.evaluation import (
    EXTRA_COLUMNS,
    PREDICTIONS_FILENAME,
    REQUESTED_COLUMNS,
    RunSummary,
    collect_run_summaries,
    load_test_predictions,
    save_publication_figures,
    save_test_predictions,
    write_summary_csv,
)
from src.utils.metrics import evaluate_hierarchical
from tests.conftest import PAPER_NUM_SEED_TYPES, PAPER_NUM_SUB_VARIETIES


@pytest.fixture
def predictions(subvariety_to_seed_type):
    """A small, hierarchy-consistent set of held-out predictions."""
    rng = np.random.default_rng(0)
    size = 60
    parent = np.asarray(subvariety_to_seed_type)
    sub_true = rng.integers(0, PAPER_NUM_SUB_VARIETIES, size=size)
    sub_pred = rng.integers(0, PAPER_NUM_SUB_VARIETIES, size=size)
    return {
        "seed_true": parent[sub_true],
        "seed_pred": parent[sub_pred],
        "sub_true": sub_true,
        "sub_pred": sub_pred,
        "sub_scores": rng.random((size, PAPER_NUM_SUB_VARIETIES)).astype(np.float32),
        "embeddings": rng.standard_normal((size, 384)).astype(np.float32),
        "expert_indices": rng.integers(0, 6, size=(size, 2)),
        "seed_type_names": [f"seed_{i}" for i in range(PAPER_NUM_SEED_TYPES)],
        "sub_variety_names": [f"sub_{i:02d}" for i in range(PAPER_NUM_SUB_VARIETIES)],
        "subvariety_to_seed_type": subvariety_to_seed_type,
    }


@pytest.fixture
def evaluation(predictions):
    return evaluate_hierarchical(
        seed_true=predictions["seed_true"],
        seed_pred=predictions["seed_pred"],
        sub_true=predictions["sub_true"],
        sub_pred=predictions["sub_pred"],
        subvariety_to_seed_type=predictions["subvariety_to_seed_type"],
        num_seed_types=PAPER_NUM_SEED_TYPES,
        num_sub_varieties=PAPER_NUM_SUB_VARIETIES,
        seed_type_names=predictions["seed_type_names"],
        sub_variety_names=predictions["sub_variety_names"],
        sub_scores=predictions["sub_scores"],
        top_k_indices=predictions["expert_indices"],
        num_experts=6,
    )


def make_summary(name: str = "full_model", **overrides) -> RunSummary:
    payload = {
        "name": name,
        "group": "ablation",
        "metrics": {
            "sub_variety/accuracy": 0.7484,
            "sub_variety/precision_macro": 0.71,
            "sub_variety/recall_macro": 0.70,
            "sub_variety/f1_macro": 0.705,
            "sub_variety/f1_micro": 0.7484,
            "seed_type/accuracy": 0.9715,
            "seed_type/f1_macro": 0.968,
            "kl_alignment/overall": 0.9594,
        },
        "efficiency": {
            "parameters": {
                "total_millions": 10.2,
                "active_millions": 7.6,
                "num_experts": 6,
                "top_k": 2,
            },
            "gflops_per_sample": 13.2,
            "peak_memory_mb": 408.0,
            "latencies": [
                {"batch_size": 1, "latency_ms_per_sample": 27.9, "throughput_fps": 35.8},
                {"batch_size": 8, "latency_ms_per_sample": 9.1, "throughput_fps": 109.9},
            ],
        },
        "history": {"train_loss": [3.0, 2.0, 1.5], "validation_loss": [3.1, 2.2, 2.3]},
    }
    payload.update(overrides)
    return RunSummary(**payload)


# ----------------------------------------------------------- prediction dumps


def test_predictions_round_trip(tmp_path, predictions):
    path = save_test_predictions(tmp_path, **predictions)
    assert path.endswith(PREDICTIONS_FILENAME)

    loaded = load_test_predictions(path)
    assert np.array_equal(loaded["sub_true"], predictions["sub_true"])
    assert np.array_equal(loaded["sub_pred"], predictions["sub_pred"])
    assert loaded["embeddings"].shape == predictions["embeddings"].shape
    assert [str(name) for name in loaded["subvariety_names"]] == predictions["sub_variety_names"]
    assert list(loaded["subvariety_to_seed_type"]) == list(predictions["subvariety_to_seed_type"])


def test_predictions_are_optional_beyond_the_labels(tmp_path, predictions):
    """Scores, embeddings and routing are extras; a dump without them must still load."""
    minimal = {
        key: value
        for key, value in predictions.items()
        if key not in {"sub_scores", "embeddings", "expert_indices"}
    }
    loaded = load_test_predictions(save_test_predictions(tmp_path, **minimal))
    assert "embeddings" not in loaded
    assert "sub_true" in loaded


# --------------------------------------------------------------- summaries


def test_summary_round_trips_through_json(tmp_path):
    original = make_summary()
    path = original.save(tmp_path)
    restored = RunSummary.load(path)

    assert restored.name == original.name
    assert restored.group == original.group
    assert restored.metrics == original.metrics
    assert restored.history == original.history


def test_summary_row_maps_metrics_to_the_requested_columns():
    row = make_summary().as_row()

    assert row["Model/Variant"] == "full_model"
    # Headline columns describe the 27-class task the architecture exists for.
    assert row["Accuracy"] == pytest.approx(0.7484)
    assert row["Macro F1"] == pytest.approx(0.705)
    assert row["Micro F1"] == pytest.approx(0.7484)
    # Stored as a fraction, reported as a percentage.
    assert row["KL Alignment Rate (%)"] == pytest.approx(95.94)
    assert row["Total Params (M)"] == pytest.approx(10.2)
    assert row["Active Params (M)"] == pytest.approx(7.6)
    # The headline latency is the smallest benchmarked batch size.
    assert row["Inference Latency (ms)"] == pytest.approx(27.9)
    assert row["Seed-Type Accuracy"] == pytest.approx(0.9715)


def test_summary_row_covers_every_declared_column():
    row = make_summary().as_row()
    assert set(row) == set(REQUESTED_COLUMNS) | set(EXTRA_COLUMNS)


def test_missing_measurements_do_not_crash_the_row():
    row = RunSummary(name="bare").as_row()
    assert row["Model/Variant"] == "bare"
    assert row["Total Params (M)"] is None
    assert row["Accuracy"] != row["Accuracy"]  # NaN when the metric was never recorded


def test_csv_has_the_requested_columns_first(tmp_path):
    path = write_summary_csv(tmp_path / "summary_metrics.csv", [make_summary()])
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    assert rows[0][: len(REQUESTED_COLUMNS)] == REQUESTED_COLUMNS
    assert rows[0] == REQUESTED_COLUMNS + EXTRA_COLUMNS
    assert len(rows) == 2
    assert rows[1][0] == "full_model"


def test_csv_writes_blanks_rather_than_nan(tmp_path):
    """A blank cell reads as 'not measured'; 'nan' reads as a failed run."""
    path = write_summary_csv(tmp_path / "summary_metrics.csv", [RunSummary(name="bare")])
    text = path and open(path, encoding="utf-8").read()
    assert "nan" not in text.lower()
    assert "None" not in text


def test_csv_holds_one_row_per_run(tmp_path):
    summaries = [make_summary("full_model"), make_summary("wo_moe"), make_summary("wo_kl")]
    path = write_summary_csv(tmp_path / "summary_metrics.csv", summaries)
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["Model/Variant"] for row in rows] == ["full_model", "wo_moe", "wo_kl"]


def test_collect_finds_summaries_one_level_down(tmp_path):
    for name in ("full_model", "wo_moe"):
        make_summary(name).save(tmp_path / "ablations" / name)
    make_summary("resnet50", group="baseline").save(tmp_path / "baselines" / "resnet50")

    found = collect_run_summaries([tmp_path / "ablations", tmp_path / "baselines"])
    assert sorted(summary.name for summary in found) == ["full_model", "resnet50", "wo_moe"]
    # Sorted by group, so baselines precede ablations alphabetically.
    assert found[0].group == "ablation"


def test_collect_ignores_missing_and_malformed_directories(tmp_path):
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "summary.json").write_text("{not json", encoding="utf-8")
    assert collect_run_summaries([tmp_path / "broken", tmp_path / "absent"]) == []


def test_summary_json_is_valid_and_readable(tmp_path):
    path = make_summary().save(tmp_path)
    payload = json.loads(open(path, encoding="utf-8").read())
    assert payload["name"] == "full_model"
    assert payload["efficiency"]["parameters"]["top_k"] == 2


# ----------------------------------------------------------------- figures


def test_publication_figures_are_written(tmp_path, evaluation, predictions):
    written = save_publication_figures(
        evaluation,
        output_dir=tmp_path,
        prefix="full_model",
        embeddings=predictions["embeddings"],
        seed_labels=predictions["seed_true"].tolist(),
        sub_labels=predictions["sub_true"].tolist(),
        history={"train_loss": [3.0, 2.0], "validation_loss": [3.1, 2.2]},
        dpi=72,          # keep the test fast; production runs at 300
        max_tsne_samples=40,
    )

    for expected in (
        "confusion_seed_type",
        "confusion_sub_variety",
        "metric_heatmap_sub_variety",
        "misclassification_sub_variety",
        "expert_utilization",
        "loss_curves",
        "tsne_seed_type",
        "tsne_sub_variety",
    ):
        assert expected in written, f"{expected} was not produced"
        path = written[expected]
        assert path.endswith(".png")
        assert path.split("/")[-1].startswith("full_model_")
        assert (tmp_path / f"full_model_{expected}.png").stat().st_size > 0


def test_figures_are_produced_without_embeddings(tmp_path, evaluation):
    """t-SNE is optional; the confusion matrices must not depend on it."""
    written = save_publication_figures(evaluation, output_dir=tmp_path, dpi=72)
    assert "confusion_sub_variety" in written
    assert "tsne_seed_type" not in written


def test_confusion_matrix_labels_every_sub_variety(evaluation):
    """All 27 classes must appear on both axes, unabbreviated."""
    from src.utils.visualization import plot_confusion_matrix

    names = [entry.name for entry in evaluation.per_class_sub]
    figure = plot_confusion_matrix(evaluation.sub_confusion, names, annotate_threshold=0)
    axes = figure.axes[0]

    assert [label.get_text() for label in axes.get_yticklabels()] == names
    assert [label.get_text() for label in axes.get_xticklabels()] == names


def test_tsne_overlays_class_names_on_clusters():
    """The overlay is what makes 27 similar colours readable."""
    from src.utils.visualization import plot_tsne

    rng = np.random.default_rng(1)
    projection = rng.standard_normal((40, 2))
    labels = rng.integers(0, 4, size=40)
    names = ["Rice", "Millet", "Mustard", "Amaranthus"]

    figure = plot_tsne(projection, labels, names, annotate_clusters=True)
    annotated = {text.get_text() for text in figure.axes[0].texts}
    assert annotated <= set(names)
    assert annotated, "no cluster labels were drawn"
    assert figure.axes[0].get_legend() is not None
