"""Cross-run reporting: prediction dumps, summary tables, publication figures.

:mod:`src.utils.metrics` scores one set of predictions and
:mod:`src.utils.visualization` draws one figure. This module is the layer above
both: it defines the on-disk contract that lets a *later* process compare runs
that finished hours apart, on different machines, without re-running any of them.

The contract has two files per run, written into that run's ``save_path``:

``test_predictions.npz``
    Raw held-out predictions, scores, 384-D embeddings, routed expert indices
    and class names. Keeping the raw arrays -- rather than only the figures --
    means a reviewer asking for a differently-normalised confusion matrix or a
    re-coloured t-SNE costs a second of replotting instead of a full retrain.

``summary.json``
    Scalar metrics, the efficiency report, the loss history and the component
    flags. :func:`collect_run_summaries` globs these into the comparison table.

Column order in ``summary_metrics.csv`` follows the revision request exactly
(:data:`REQUESTED_COLUMNS`); anything additionally useful is appended after
those, so the requested table can be sliced off the left without editing.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.metrics import HierarchicalEvaluation, tsne_projection
from src.utils.visualization import (
    plot_confusion_matrix,
    plot_expert_utilization,
    plot_loss_curves,
    plot_metric_heatmap,
    plot_misclassification_rates,
    plot_tsne,
    save_figure,
)

PREDICTIONS_FILENAME = "test_predictions.npz"
SUMMARY_FILENAME = "summary.json"

#: The exact columns the revision asks for, in the order requested.
REQUESTED_COLUMNS = [
    "Model/Variant",
    "Accuracy",
    "Precision",
    "Recall",
    "Macro F1",
    "Micro F1",
    "KL Alignment Rate (%)",
    "Total Params (M)",
    "Active Params (M)",
    "Inference Latency (ms)",
]

#: Additional context appended to the right of the requested columns.
EXTRA_COLUMNS = [
    "Seeds",
    "p (vs full)",
    "p (Holm)",
    "Seed-Type Accuracy",
    "Seed-Type Macro F1",
    "Sub-Variety AUC (macro OvR)",
    "ECE",
    "Expert NMI (sub-variety)",
    "Dead Experts",
    "Experts",
    "Top-K",
    "Throughput (FPS)",
    "GFLOPs/sample",
    "Peak Memory (MB)",
    "Split Protocol",
    "Group",
    "Run Directory",
]


# ------------------------------------------------------------------- summaries


@dataclass
class RunSummary:
    """One finished training run, as persisted to ``summary.json``.

    ``metrics`` holds the flattened scalars produced by
    :meth:`~src.utils.metrics.HierarchicalEvaluation.scalar_metrics`, keyed
    ``seed_type/...``, ``sub_variety/...`` and ``kl_alignment/...``.
    """

    name: str
    group: str = "experiment"
    run_dir: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    efficiency: dict[str, Any] = field(default_factory=dict)
    history: dict[str, list[float]] = field(default_factory=dict)
    component_flags: dict[str, Any] = field(default_factory=dict)
    loss_flags: dict[str, Any] = field(default_factory=dict)
    """Every loss-side setting a variant can move. Without this a ``wo_kl`` run's
    machine-readable trace was byte-identical to ``full_model``'s -- only the
    variant name distinguished them, which is exactly the kind of implicit
    difference this repository otherwise refuses to tolerate."""

    split: dict[str, Any] = field(default_factory=dict)
    """Split protocol, seed, and the dataset's provenance diagnostics. The
    headline accuracy is uninterpretable without knowing whether crops of the
    same source photograph could appear on both sides of the boundary."""

    fold_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    """``{metric: {mean, std, min, max, folds}}`` across cross-validation folds.
    Reporting the *best* fold's test score is a selection procedure whose
    expected value exceeds a single fold's, so the mean is what the table uses;
    the best fold is kept only for the artifact that gets shipped."""

    runtime: dict[str, Any] = field(default_factory=dict)
    """How the run was executed: world size, backend, autocast dtype, whether it
    compiled. None of it changes the objective, and all of it changes what a
    throughput or latency number in ``efficiency`` means -- so a reader comparing
    two rows can tell whether they were produced on the same footing without
    having to find the logs."""

    config: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)

    def save(self, directory: str | Path) -> str:
        """Write ``summary.json`` into ``directory`` and return its path."""
        path = Path(directory) / SUMMARY_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_json_safe(asdict(self)), indent=2, sort_keys=True), encoding="utf-8")
        return str(path)

    @classmethod
    def load(cls, path: str | Path) -> "RunSummary":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {key: payload.get(key) for key in cls.__dataclass_fields__ if key in payload}
        return cls(**known)

    # ------------------------------------------------------------ table rows

    def _metric(self, key: str, default: float = float("nan")) -> float:
        value = self.metrics.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _efficiency(self, *path: str) -> Any:
        node: Any = self.efficiency
        for key in path:
            if not isinstance(node, Mapping) or key not in node:
                return None
            node = node[key]
        return node

    def _primary_latency(self) -> Mapping[str, Any] | None:
        latencies = self._efficiency("latencies")
        if not latencies:
            return None
        return min(latencies, key=lambda entry: entry.get("batch_size", 0))

    def as_row(self) -> dict[str, Any]:
        """One row of ``summary_metrics.csv``.

        The headline accuracy/precision/recall/F1 columns describe the
        **sub-variety** task: it is the 27-class problem the architecture exists
        to solve, and the one where the variants actually separate. Seed-type
        numbers follow in the extra columns rather than being averaged in, which
        would blend two tasks of very different difficulty into one meaningless
        figure.
        """
        latency = self._primary_latency() or {}
        alignment = self._metric("kl_alignment/overall")
        return {
            "Model/Variant": self.name,
            "Accuracy": self._metric("sub_variety/accuracy"),
            "Precision": self._metric("sub_variety/precision_macro"),
            "Recall": self._metric("sub_variety/recall_macro"),
            "Macro F1": self._metric("sub_variety/f1_macro"),
            "Micro F1": self._metric("sub_variety/f1_micro"),
            "KL Alignment Rate (%)": alignment * 100.0 if alignment == alignment else alignment,
            "Total Params (M)": self._efficiency("parameters", "total_millions"),
            "Active Params (M)": self._efficiency("parameters", "active_millions"),
            "Inference Latency (ms)": latency.get("latency_ms_per_sample"),
            "Seeds": 1,
            "p (vs full)": None,
            "p (Holm)": None,
            "Seed-Type Accuracy": self._metric("seed_type/accuracy"),
            "Seed-Type Macro F1": self._metric("seed_type/f1_macro"),
            "Sub-Variety AUC (macro OvR)": self._metric("sub_variety/auc_macro_ovr"),
            "ECE": self._metric("calibration/ece"),
            "Expert NMI (sub-variety)": self._metric("moe/nmi_sub_variety"),
            "Dead Experts": self._metric("moe/dead_experts"),
            "Experts": self._efficiency("parameters", "num_experts"),
            "Top-K": self._efficiency("parameters", "top_k"),
            "Throughput (FPS)": latency.get("throughput_fps"),
            "GFLOPs/sample": self._efficiency("gflops_per_sample"),
            "Peak Memory (MB)": self._efficiency("peak_memory_mb"),
            "Split Protocol": self.split.get("protocol", ""),
            "Group": self.group,
            "Run Directory": self.run_dir,
        }


def collect_run_summaries(roots: Iterable[str | Path]) -> list[RunSummary]:
    """Load every ``summary.json`` found under ``roots``, sorted by group then name.

    Globs three directory levels, which is what the multi-seed layout
    ``outputs/{group}/{variant}/seed{n}/`` needs.
    """
    summaries: list[RunSummary] = []
    for root in roots:
        base = Path(root)
        if not base.exists():
            continue
        candidates = [base / SUMMARY_FILENAME] if (base / SUMMARY_FILENAME).exists() else []
        for depth in ("*", "*/*", "*/*/*"):
            candidates.extend(sorted(base.glob(f"{depth}/{SUMMARY_FILENAME}")))
        for path in dict.fromkeys(candidates):  # de-duplicate, keep order
            try:
                summaries.append(RunSummary.load(path))
            except (OSError, json.JSONDecodeError, TypeError):
                continue
    return sorted(summaries, key=lambda summary: (summary.group, summary.name))


# ------------------------------------------------------------- multi-seed table


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    finite = [float(value) for value in values if value == value]
    if not finite:
        return float("nan"), float("nan")
    mean = sum(finite) / len(finite)
    if len(finite) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in finite) / (len(finite) - 1)
    return mean, variance**0.5


def aggregate_by_variant(summaries: Sequence[RunSummary]) -> list[dict[str, Any]]:
    """Collapse repeated seeds of one variant into a single mean +- std row.

    Why this exists, stated concretely. On a 1,871-image test split at ~95 %
    accuracy the 95 % CI half-width on a *single* accuracy is +-0.99 pp, and on a
    *difference* of two accuracies +-1.40 pp -- before any training-seed variance
    (dropout, shuffling, router initialisation, and for a MoE specifically, which
    experts happen to win the early race). For a 27-class fine-grained task where
    component contributions of 0.5--2 pp are the normal magnitude, a single run
    per variant cannot resolve the table it is being asked to support.

    Rows carry ``{column}`` (the mean) and ``{column} SD``, plus ``Seeds``.
    """
    grouped: dict[tuple[str, str], list[RunSummary]] = {}
    for summary in summaries:
        grouped.setdefault((summary.group, summary.name), []).append(summary)

    rows: list[dict[str, Any]] = []
    for (group, name), runs in sorted(grouped.items()):
        per_run = [run.as_row() for run in runs]
        row: dict[str, Any] = dict(per_run[0])

        for column in REQUESTED_COLUMNS + EXTRA_COLUMNS:
            values = [entry.get(column) for entry in per_run]
            numeric = [value for value in values if isinstance(value, (int, float))]
            if len(numeric) != len(values) or not numeric:
                continue
            mean, std = _mean_std(numeric)
            row[column] = mean
            if len(runs) > 1:
                row[f"{column} SD"] = std

        # Set after the averaging loop: `Seeds` is a count of the runs, not a
        # per-run measurement to average (every run reports 1).
        row["Seeds"] = len(runs)
        row["Group"] = group
        row["Model/Variant"] = name
        row["Run Directory"] = "; ".join(sorted(run.run_dir for run in runs))
        rows.append(row)
    return rows


def paired_significance(
    prediction_paths: dict[str, Sequence[str | Path]],
    reference: str = "full_model",
    task: str = "sub",
) -> dict[str, dict[str, float]]:
    """McNemar's exact test of every variant against ``reference``.

    Every variant in a suite trains on the byte-identical split, so their test
    predictions are **paired** -- which makes McNemar both valid and strictly
    more powerful than comparing independent confidence intervals, and it needs
    nothing beyond the ``test_predictions.npz`` files already on disk.

    Multi-seed runs are pooled by concatenating each variant's predictions in a
    fixed seed order; that keeps the pairing intact as long as every variant ran
    the same seeds, which the suite scripts guarantee.

    Args:
        prediction_paths: ``{variant: [test_predictions.npz, ...]}``.
        reference: Variant every other is compared against.
        task: ``"sub"`` or ``"seed"``.
    """
    from src.utils.metrics import holm_bonferroni, mcnemar_test

    def correctness(paths: Sequence[str | Path]) -> np.ndarray | None:
        chunks = []
        for path in sorted(str(item) for item in paths):
            try:
                payload = load_test_predictions(path)
            except (OSError, ValueError):
                return None
            true_key, pred_key = (f"{task}_true", f"{task}_pred")
            if true_key not in payload or pred_key not in payload:
                return None
            chunks.append(payload[true_key] == payload[pred_key])
        return np.concatenate(chunks) if chunks else None

    if reference not in prediction_paths:
        return {}
    reference_correct = correctness(prediction_paths[reference])
    if reference_correct is None:
        return {}

    results: dict[str, dict[str, float]] = {}
    for variant, paths in prediction_paths.items():
        if variant == reference:
            continue
        variant_correct = correctness(paths)
        if variant_correct is None or variant_correct.size != reference_correct.size:
            continue
        outcome = mcnemar_test(reference_correct, variant_correct)
        if outcome:
            results[variant] = outcome

    adjusted = holm_bonferroni({name: value["p_value"] for name, value in results.items()})
    for name, value in adjusted.items():
        results[name]["p_value_holm"] = value
    return results


def write_summary_csv(
    path: str | Path,
    summaries: Sequence[RunSummary],
    float_format: str = "{:.4f}",
    aggregate: bool = True,
    significance: Mapping[str, Mapping[str, float]] | None = None,
) -> str:
    """Write the comparison table and return its path.

    Missing measurements are written as empty cells rather than ``nan``: a blank
    reads unambiguously as "not measured", whereas ``nan`` in a results table
    invites a reader to treat it as a failed run.

    Args:
        path: Destination CSV.
        summaries: Loaded run summaries.
        float_format: Numeric formatting.
        aggregate: Collapse repeated seeds of a variant into mean +- SD. Set
            ``False`` for the per-run table.
        significance: ``{variant: {p_value, p_value_holm, ...}}`` from
            :func:`paired_significance`, filled into the ``p (vs full)`` and
            ``p (Holm)`` columns.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = aggregate_by_variant(summaries) if aggregate else [summary.as_row() for summary in summaries]
    for row in rows:
        outcome = (significance or {}).get(str(row.get("Model/Variant")))
        if outcome:
            row["p (vs full)"] = outcome.get("p_value")
            row["p (Holm)"] = outcome.get("p_value_holm")

    columns = list(REQUESTED_COLUMNS + EXTRA_COLUMNS)
    # Standard deviations sit immediately right of the column they qualify.
    extra = [key for row in rows for key in row if key not in columns]
    columns.extend(sorted(dict.fromkeys(extra)))

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _format_cell(row.get(column), float_format) for column in columns})
    return str(output)


def _format_cell(value: Any, float_format: str) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:  # NaN
            return ""
        return float_format.format(value)
    return str(value)


# ----------------------------------------------------------------- predictions


def save_test_predictions(
    directory: str | Path,
    seed_true: Sequence[int],
    seed_pred: Sequence[int],
    sub_true: Sequence[int],
    sub_pred: Sequence[int],
    seed_type_names: Sequence[str],
    sub_variety_names: Sequence[str],
    subvariety_to_seed_type: Sequence[int],
    sub_scores: np.ndarray | None = None,
    embeddings: np.ndarray | None = None,
    expert_indices: np.ndarray | None = None,
    sub_logits: np.ndarray | None = None,
    tokens_per_sample: int = 1,
    filename: str = PREDICTIONS_FILENAME,
) -> str:
    """Persist raw held-out predictions so figures can be regenerated offline.

    ``sub_logits`` are stored alongside ``sub_scores`` because temperature
    scaling has to be fitted on logits; recovering them from float32 softmax
    output is not reliable. ``tokens_per_sample`` records how many routing
    decisions each image contributed, which is what lets
    :func:`~src.utils.metrics.expert_label_nmi` line ``expert_indices`` up with
    the labels under grid routing.
    """
    output = Path(directory) / filename
    output.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, np.ndarray] = {
        "seed_true": np.asarray(seed_true, dtype=np.int64),
        "seed_pred": np.asarray(seed_pred, dtype=np.int64),
        "sub_true": np.asarray(sub_true, dtype=np.int64),
        "sub_pred": np.asarray(sub_pred, dtype=np.int64),
        "seed_type_names": np.array(list(seed_type_names), dtype=object),
        "subvariety_names": np.array(list(sub_variety_names), dtype=object),
        "subvariety_to_seed_type": np.asarray(list(subvariety_to_seed_type), dtype=np.int64),
        "tokens_per_sample": np.asarray(int(tokens_per_sample), dtype=np.int64),
    }
    if sub_scores is not None:
        payload["sub_scores"] = np.asarray(sub_scores, dtype=np.float32)
    if sub_logits is not None:
        payload["sub_logits"] = np.asarray(sub_logits, dtype=np.float32)
    if embeddings is not None:
        payload["embeddings"] = np.asarray(embeddings, dtype=np.float32)
    if expert_indices is not None:
        payload["expert_indices"] = np.asarray(expert_indices, dtype=np.int64)

    np.savez_compressed(output, **payload)
    return str(output)


def load_test_predictions(path: str | Path) -> dict[str, np.ndarray]:
    """Load a ``test_predictions.npz`` written by :func:`save_test_predictions`."""
    with np.load(Path(path), allow_pickle=True) as archive:
        return {key: archive[key] for key in archive.files}


# --------------------------------------------------------------------- figures


def save_publication_figures(
    evaluation: HierarchicalEvaluation,
    output_dir: str | Path,
    prefix: str = "",
    embeddings: np.ndarray | None = None,
    seed_labels: Sequence[int] | None = None,
    sub_labels: Sequence[int] | None = None,
    history: Mapping[str, Sequence[float]] | None = None,
    dpi: int = 300,
    tsne_perplexity: float = 30.0,
    max_tsne_samples: int = 2000,
) -> dict[str, str]:
    """Write every publication figure for one run and return ``{name: path}``.

    Confusion matrices are row-normalised. With 27 sub-varieties at very
    different supports, raw counts make a frequent class look accurate purely by
    being frequent; normalising by true-class support is what makes the diagonal
    comparable down the rows.

    Rendered at ``dpi=300`` -- print resolution -- rather than the 150 used for
    the live training-log figures.

    Args:
        evaluation: Scored predictions for the run.
        output_dir: Directory to write into.
        prefix: Filename prefix, normally the variant name.
        embeddings: 384-D embeddings for the t-SNE panels.
        seed_labels / sub_labels: True labels colouring those panels.
        history: Named loss series for the train-vs-validation figure.
        dpi: Output resolution.
        tsne_perplexity / max_tsne_samples: t-SNE configuration.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}_" if prefix else ""
    written: dict[str, str] = {}

    def write(name: str, figure) -> None:
        written[name] = save_figure(figure, directory / f"{stem}{name}.png", dpi=dpi)

    seed_names = [entry.name for entry in evaluation.per_class_seed]
    sub_names = [entry.name for entry in evaluation.per_class_sub]

    write(
        "confusion_seed_type",
        plot_confusion_matrix(
            evaluation.seed_confusion,
            seed_names,
            title="Seed-type confusion (row-normalised)",
        ),
    )
    write(
        "confusion_sub_variety",
        plot_confusion_matrix(
            evaluation.sub_confusion,
            sub_names,
            title="Sub-variety confusion (row-normalised)",
            # 27 classes: annotating every cell is unreadable, but the full,
            # unabbreviated tick labels are kept on both axes.
            annotate_threshold=0,
        ),
    )
    write("metric_heatmap_sub_variety", plot_metric_heatmap(evaluation.per_class_sub))
    write("misclassification_sub_variety", plot_misclassification_rates(evaluation.sub_misclassification))
    write("expert_utilization", plot_expert_utilization(evaluation.expert_utilization))

    if history:
        series = {name: list(values) for name, values in history.items() if len(values) > 0}
        if series:
            write(
                "loss_curves",
                plot_loss_curves(series, title="Training vs. validation loss", xlabel="Epoch"),
            )

    if embeddings is not None and len(embeddings) > 0:
        projection = tsne_projection(
            embeddings, perplexity=tsne_perplexity, max_samples=max_tsne_samples
        )
        if projection is not None:
            count = projection.shape[0]
            if seed_labels is not None:
                write(
                    "tsne_seed_type",
                    plot_tsne(
                        projection,
                        list(seed_labels)[:count],
                        seed_names,
                        title="t-SNE of learned embeddings, by seed type",
                        annotate_clusters=True,
                    ),
                )
            if sub_labels is not None:
                write(
                    "tsne_sub_variety",
                    plot_tsne(
                        projection,
                        list(sub_labels)[:count],
                        sub_names,
                        title="t-SNE of learned embeddings, by sub-variety",
                        annotate_clusters=True,
                    ),
                )

    return written


# ------------------------------------------------------- stage-1 event stream


def load_event_stream(path: str | Path) -> list[dict[str, Any]]:
    """Parse a run's ``events.jsonl``, skipping malformed lines.

    Every trainer appends to this file unconditionally (``ExperimentTracker``),
    so it is the one machine-readable record of a finished run that exists even
    when TensorBoard and W&B were unavailable -- which is exactly the situation a
    later reviewer is in. Malformed trailing lines are skipped rather than raised
    on: a run killed mid-write leaves a truncated last line, and refusing to read
    the other 99.9 % of it would be the wrong trade.
    """
    records: list[dict[str, Any]] = []
    file = Path(path)
    if not file.exists():
        return records
    with file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


@dataclass
class PretrainDynamics:
    """Stage-1 training dynamics recovered from one run's event stream.

    ``step_series`` and ``epoch_series`` are ``{metric: (x, y)}`` with ``x`` the
    global step or the epoch index, so a figure can be redrawn from a finished
    run without W&B, TensorBoard, or the checkpoint.

    The point of separating the two is that they answer different questions. The
    epoch series is the *learning curve* -- was the objective still improving when
    the budget ran out. The step series carries the collapse diagnostics
    (``train/teacher_entropy``, ``train/prototype_utilization``), which are
    per-step device tensors and are the only evidence available that the run's
    representation did not quietly degenerate on its way to a falling loss.
    """

    run_dir: str = ""
    step_series: dict[str, tuple[list[float], list[float]]] = field(default_factory=dict)
    epoch_series: dict[str, tuple[list[float], list[float]]] = field(default_factory=dict)
    events: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def series(self, key: str) -> tuple[list[float], list[float]]:
        """``(x, y)`` for ``key``, from whichever stream carries it."""
        if key in self.epoch_series:
            return self.epoch_series[key]
        return self.step_series.get(key, ([], []))

    def final(self, key: str, window: int = 1) -> float:
        """Mean of the last ``window`` values of ``key``, or NaN."""
        _, values = self.series(key)
        if not values:
            return float("nan")
        tail = values[-max(int(window), 1) :]
        return float(sum(tail) / len(tail))

    def initial(self, key: str, window: int = 1) -> float:
        """Mean of the first ``window`` values of ``key``, or NaN."""
        _, values = self.series(key)
        if not values:
            return float("nan")
        head = values[: max(int(window), 1)]
        return float(sum(head) / len(head))

    def summary(self) -> dict[str, Any]:
        """Scalars a report can quote, plus the two derived judgements.

        ``loss_improvement_last_quarter`` is the number that decides whether the
        epoch budget was the binding constraint: it is the drop in epoch loss over
        the final 25 % of training, in the same units as the loss. Near zero means
        the run converged and more epochs would buy little; still-falling means the
        schedule ended early, whatever the loss curve looks like at full scale.

        ``teacher_entropy_vs_floor`` is the collapse verdict, and it is reported as
        a *ratio to the structural floor* rather than as a bare entropy, because
        ``H`` scales with ``log K`` and an exactly doubly-stochastic Sinkhorn
        assignment cannot go below ``log(K / B_teacher)`` in the first place.
        """
        epochs, losses = self.series("epoch/loss")
        summary: dict[str, Any] = {
            "epochs_completed": len(epochs),
            "steps_logged": len(self.step_series.get("train/loss", ([], []))[0]),
            "loss_initial": self.initial("epoch/loss"),
            "loss_final": self.final("epoch/loss"),
        }
        if losses:
            quarter = max(len(losses) // 4, 1)
            summary["loss_improvement_last_quarter"] = float(losses[-quarter] - losses[-1])
            summary["loss_improvement_total"] = float(losses[0] - losses[-1])
            summary["loss_min"] = float(min(losses))
            summary["loss_min_epoch"] = int(epochs[losses.index(min(losses))]) if epochs else -1

        entropy_final = self.final("train/teacher_entropy", window=20)
        floor = self.final("train/teacher_entropy_min", window=1)
        ceiling = self.final("train/teacher_entropy_max", window=1)
        summary.update(
            {
                "teacher_entropy_final": entropy_final,
                "teacher_entropy_floor": floor,
                "teacher_entropy_ceiling": ceiling,
                "teacher_entropy_normalized_final": self.final("train/teacher_entropy_normalized", window=20),
                "teacher_entropy_vs_floor": (
                    float(entropy_final / floor) if floor and floor == floor else float("nan")
                ),
                "prototype_utilization_final": self.final("train/prototype_utilization", window=20),
                "prototype_perplexity_final": self.final("train/prototype_perplexity", window=20),
                "koleo_initial": self.initial("train/koleo", window=20),
                "koleo_final": self.final("train/koleo", window=20),
                "gradient_norm_final": self.final("train/gradient_norm", window=20),
                "images_per_second_mean": _mean_of(self.series("epoch/images_per_second")[1]),
                "data_wait_fraction_mean": _mean_of(self.series("epoch/data_wait_fraction")[1]),
                "epoch_duration_seconds_mean": _mean_of(self.series("epoch/duration_seconds")[1]),
                "peak_memory_mb_max": (
                    max(self.series("epoch/peak_memory_mb")[1])
                    if self.series("epoch/peak_memory_mb")[1]
                    else float("nan")
                ),
            }
        )
        total = self.series("epoch/duration_seconds")[1]
        summary["training_hours"] = float(sum(total) / 3600.0) if total else float("nan")
        return summary


def _mean_of(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if value == value]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def parse_pretrain_dynamics(
    events_path: str | Path,
    step_keys: Sequence[str] = (),
    epoch_keys: Sequence[str] = (),
) -> PretrainDynamics:
    """Recover :class:`PretrainDynamics` from a stage-1 ``events.jsonl``.

    ``step_keys`` and ``epoch_keys`` are prefixes; empty tuples take
    ``train/*`` and ``epoch/*`` respectively, which is what the stage-1 trainer
    emits. Non-metric events (``stage1_budget``, ``milestone_checkpoint``,
    ``shared_backbone``, ``learning_rate``, ``model_shapes``) are kept whole under
    :attr:`PretrainDynamics.events`, because they carry the run's provenance and a
    report needs to quote it rather than re-derive it.
    """
    records = load_event_stream(events_path)
    step_prefixes = tuple(step_keys) or ("train/",)
    epoch_prefixes = tuple(epoch_keys) or ("epoch/",)

    step_series: dict[str, tuple[list[float], list[float]]] = {}
    epoch_series: dict[str, tuple[list[float], list[float]]] = {}
    events: dict[str, list[dict[str, Any]]] = {}

    for record in records:
        kind = str(record.get("type", ""))
        if kind != "metrics":
            events.setdefault(kind, []).append(record)
            continue
        step = float(record.get("step", 0) or 0)
        for key, value in (record.get("metrics") or {}).items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            if key.startswith(epoch_prefixes):
                bucket = epoch_series
            elif key.startswith(step_prefixes):
                bucket = step_series
            else:
                continue
            x_values, y_values = bucket.setdefault(key, ([], []))
            x_values.append(step)
            y_values.append(float(value))

    # Epoch metrics are logged with `step = epoch`, but the trainer also emits a
    # step-indexed record in the same second; re-index the epoch stream on a
    # 1..N counter so the two are never plotted against the same x axis by
    # accident.
    for key, (_, y_values) in epoch_series.items():
        epoch_series[key] = ([float(index) for index in range(1, len(y_values) + 1)], y_values)

    return PretrainDynamics(
        run_dir=str(Path(events_path).parent),
        step_series=step_series,
        epoch_series=epoch_series,
        events=events,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)
