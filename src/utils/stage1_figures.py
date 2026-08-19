"""Publication figures for one stage-1 run, generated from its own CSV artifacts.

Everything here reads ``csv/*.csv`` and nothing else. That is the point: the
figures and the tables are then the same numbers by construction, and the whole
suite can be regenerated from a finished run's directory without a checkpoint, a
GPU, W&B, or the trainer -- which is what ``scripts/plot_stage1_run.py`` does.

Five figures, each answering a question the others cannot:

``01_optimization``
    Is the *student* learning? Plots ``KL(q||p)`` first and the raw cross entropy
    second, with the teacher's entropy and its structural floor beside them. The
    raw loss is demoted deliberately: on the shipped run it was **94.8 %
    irreducible target entropy** and 80 % of its total drop was the teacher
    sharpening. Reading it as a learning curve is how "the loss plateaued at
    epoch 20" became a conclusion about the budget when the learnable part was
    still falling at epoch 93.

``02_representation``
    Is the *representation* improving? The probe and k-NN readouts against the
    milestone epochs, with the selected checkpoint marked, plus the geometry
    (RankMe, participation ratio) and the nuisance gate on the same x axis. This
    is the figure that decides the epoch budget.

``03_collapse``
    The diagnostics whose failure a plausible-looking loss curve hides: teacher
    entropy against its own floor and ceiling, prototype perplexity and
    utilisation, KoLeo. Entropy is plotted *with its bounds drawn in*, because
    ``H`` scales with ``log K`` and under Sinkhorn cannot fall below
    ``log(K / B_teacher)`` -- 3.47 of a 7.62 maximum at ``K = 2048``, so nearly
    half the nominal range is unreachable and a bare entropy axis invites the
    wrong reading.

``04_views``
    What the augmentation actually produced: native pixels behind each view
    family, the upsample factor, and the deterministic-fallback rate. This is the
    figure that makes the local-view problem visible -- a median 598 native
    pixels rendered into 65,536.

``05_throughput``
    Where the wall clock went. Images/s, epoch duration, loop-blocked fraction
    and peak memory. ``loop_blocked_fraction`` is annotated as an **upper bound**
    on GPU idleness rather than a measurement of it, because nothing synchronises
    inside the step and turning ``1 - blocked`` into a busy time gives the CPU
    enqueue time.

Missing inputs degrade to a missing panel, never to an exception: a run
interrupted before its first probe should still get its optimization figure.
"""

from __future__ import annotations

import csv
import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

#: Written next to every PNG. A vector copy is what a paper needs and a raster
#: copy is what a reviewer opens; both cost one extra ``savefig``.
FIGURE_FORMATS = ("png", "pdf")


def read_csv_columns(path: str | Path) -> dict[str, list[float]]:
    """Load a wide metric CSV as ``{column: [float]}``, skipping empty cells.

    Empty cells become ``nan`` rather than being dropped, so every column stays
    the same length as the index column and a series can be plotted against it
    without re-aligning. ``nan`` is what matplotlib already treats as a gap.
    """
    file_path = Path(path)
    if not file_path.exists():
        return {}
    columns: dict[str, list[float]] = {}
    with file_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for name in reader.fieldnames or []:
            columns[name] = []
        for row in reader:
            for name in columns:
                raw = (row.get(name) or "").strip()
                try:
                    columns[name].append(float(raw) if raw else float("nan"))
                except ValueError:
                    columns[name].append(float("nan"))
    return columns


def _series(columns: Mapping[str, Sequence[float]], index: str, metric: str):
    """``(x, y)`` for one metric, or ``None`` when it is absent or all-NaN."""
    if metric not in columns or index not in columns:
        return None
    x_values = list(columns[index])
    y_values = list(columns[metric])
    pairs = [
        (x, y)
        for x, y in zip(x_values, y_values)
        if isinstance(y, float) and not math.isnan(y) and not math.isnan(x)
    ]
    if not pairs:
        return None
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def _panel(
    columns: Mapping[str, Sequence[float]],
    index: str,
    title: str,
    metrics: Mapping[str, str],
    ylabel: str = "",
    hlines: Sequence[Mapping[str, Any]] = (),
    **extra: Any,
) -> dict[str, Any] | None:
    """One panel spec, or ``None`` when none of its metrics were logged."""
    series = {}
    for label, metric in metrics.items():
        found = _series(columns, index, metric)
        if found is not None:
            series[label] = found
    if not series:
        return None
    return {
        "title": title,
        "xlabel": index.replace("_", " "),
        "ylabel": ylabel,
        "series": series,
        "hlines": list(hlines),
        **extra,
    }


def _last(columns: Mapping[str, Sequence[float]], metric: str) -> float:
    found = [
        value
        for value in columns.get(metric, [])
        if isinstance(value, float) and not math.isnan(value)
    ]
    return found[-1] if found else float("nan")


def _save(figure, directory: Path, name: str, dpi: int) -> list[str]:
    from src.utils.visualization import save_figure

    paths: list[str] = []
    for index, suffix in enumerate(FIGURE_FORMATS):
        # Only the last write closes the canvas; the rest need it alive.
        paths.append(
            save_figure(
                figure,
                directory / f"{name}.{suffix}",
                dpi=dpi,
                close=index == len(FIGURE_FORMATS) - 1,
            )
        )
    return paths


def optimization_figure(epoch: Mapping[str, Sequence[float]], step: Mapping[str, Sequence[float]]):
    """Figure 1: the objective, decomposed."""
    from src.utils.visualization import plot_series_panels

    panels = [
        _panel(
            epoch,
            "epoch",
            "Learnable part: KL(teacher || student)",
            {"KL(q||p)": "teacher_student_kl"},
            ylabel="nats",
        ),
        _panel(
            epoch,
            "epoch",
            "Cross entropy = KL + H(teacher)",
            {
                "CE": "dino_cross_entropy",
                "KL(q||p)": "teacher_student_kl",
                "H(teacher)": "teacher_entropy_cross_view",
                "total objective": "loss",
            },
            ylabel="nats",
        ),
        _panel(step, "step", "Learning rate", {"lr": "lr"}, ylabel="lr"),
        _panel(
            step,
            "step",
            "Teacher temperature and momentum",
            {"tau_teacher": "teacher_temp", "EMA momentum": "teacher_momentum"},
        ),
        _panel(
            step,
            "step",
            "Gradient norm (pre-clip) and weight decay",
            {
                "clipped total norm": "clipped_gradient_norm",
                "per-tensor total": "gradient_norm",
                "weight decay": "weight_decay",
            },
        ),
        _panel(
            step,
            "step",
            "Update-to-weight ratio",
            {"median |dW| / |W|": "update_ratio_median"},
            ylabel="ratio",
            yscale="log",
        ),
    ]
    live = [panel for panel in panels if panel]
    if not live:
        return None
    return plot_series_panels(
        live,
        columns=3,
        suptitle=(
            "Stage 1 optimization. Read KL(q||p), not the raw loss: the cross entropy is "
            "mostly the teacher's entropy."
        ),
    )


def representation_figure(probe: Mapping[str, Sequence[float]], best_epoch: float | None = None):
    """Figure 2: the readout, the geometry and the nuisance gate."""
    from src.utils.visualization import plot_series_panels

    panels = [
        _panel(
            probe,
            "epoch",
            "Frozen readout (crop-level stratified CV)",
            {"linear probe": "probe_accuracy", "weighted k-NN": "knn_accuracy"},
            ylabel="27-way accuracy",
        ),
        _panel(
            probe,
            "epoch",
            "Macro F1 and probe generalisation gap",
            {
                "probe macro-F1": "probe_f1_macro",
                "train - test accuracy": "probe_generalisation_gap",
            },
        ),
        _panel(
            probe,
            "epoch",
            "Nuisance gate: within-class photograph identity",
            {"above chance": "nuisance_above_chance"},
            ylabel="accuracy above chance",
            hlines=[{"y": 0.0, "label": "chance (nothing encoded)", "style": ":"}],
        ),
        _panel(
            probe,
            "epoch",
            "Spectrum: dimensional collapse",
            {"RankMe": "rankme", "participation ratio": "participation_ratio", "stable rank": "stable_rank"},
        ),
        _panel(
            probe,
            "epoch",
            "Cluster structure",
            {
                "k-means NMI": "kmeans_nmi",
                "silhouette (sub-variety)": "silhouette_sub",
            },
        ),
        _panel(
            probe,
            "epoch",
            "Collapse guards on the shipped space",
            {
                "||mean unit vector||": "feature_mean_direction_norm",
                "dead-dim fraction": "feature_dead_dim_fraction",
                "top-1 variance share": "top1_variance_share",
            },
        ),
    ]
    live = [panel for panel in panels if panel]
    if not live:
        return None
    figure = plot_series_panels(
        live,
        columns=3,
        suptitle=(
            "Stage 1 representation. The readout decides the epoch budget; the nuisance panel "
            "is the gate an arm must not win by re-learning the photograph."
        ),
    )
    if best_epoch is not None and best_epoch >= 0:
        for axis in figure.axes:
            axis.axvline(
                float(best_epoch), color="#D55E00", linestyle="-.", linewidth=1.0, alpha=0.8
            )
    return figure


def collapse_figure(step: Mapping[str, Sequence[float]]):
    """Figure 3: the diagnostics a plausible loss curve hides."""
    from src.utils.visualization import plot_series_panels

    floor = _last(step, "teacher_entropy_min")
    ceiling = _last(step, "teacher_entropy_max")
    bounds = []
    if not math.isnan(floor):
        bounds.append({"y": floor, "label": "structural floor log(K/B)", "style": "--"})
    if not math.isnan(ceiling):
        bounds.append({"y": ceiling, "label": "ceiling log K", "style": ":"})

    panels = [
        _panel(
            step,
            "step",
            "Teacher entropy, with its bounds",
            {"H(teacher)": "teacher_entropy"},
            ylabel="nats",
            hlines=bounds,
        ),
        _panel(
            step,
            "step",
            "Prototype usage",
            {
                "perplexity": "prototype_perplexity",
                "utilisation": "prototype_utilization",
            },
        ),
        _panel(
            step,
            "step",
            "KoLeo and KL to uniform",
            {"KoLeo": "koleo", "KL(mean assignment || uniform)": "prototype_kl_to_uniform"},
        ),
        _panel(
            step,
            "step",
            "Normalised teacher entropy",
            {"H / log K": "teacher_entropy_normalized"},
            ylim=(0.0, 1.0),
        ),
    ]
    live = [panel for panel in panels if panel]
    if not live:
        return None
    return plot_series_panels(
        live,
        columns=2,
        suptitle=(
            "Collapse diagnostics. Entropy is conditional on K and the batch -- read it against "
            "the drawn floor, never against zero."
        ),
    )


def view_geometry_figure(rows: Sequence[Mapping[str, Any]]):
    """Figure 4: what the augmentation actually built each view from."""
    import matplotlib.pyplot as plt

    from src.utils.visualization import use_publication_style

    families = [row for row in rows if row.get("family") in {"global", "local"}]
    if not families:
        return None
    use_publication_style()
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 3.4))
    labels = [str(row["family"]) for row in families]
    positions = range(len(families))

    def numbers(key: str) -> list[float]:
        return [float(row.get(key, float("nan"))) for row in families]

    # Native pixels, as a p5-p50-p95 range bar. A bar chart of the median alone
    # would hide that the p5 of a local view is ~130 px under the old recipe.
    medians = numbers("native_pixels_p50")
    low = [median - value for median, value in zip(medians, numbers("native_pixels_p5"))]
    high = [value - median for median, value in zip(medians, numbers("native_pixels_p95"))]
    axes[0].bar(list(positions), medians, yerr=[low, high], capsize=4, color="#0072B2")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("native source pixels")
    axes[0].set_title("Content behind one view (p5 / p50 / p95)")

    axes[1].bar(list(positions), numbers("upsample_factor_median"), color="#E69F00")
    axes[1].set_ylabel("x")
    axes[1].set_title("Median upsample to the encoder's input")

    axes[2].bar(list(positions), numbers("deterministic_fallback_rate"), color="#CC79A7")
    axes[2].set_ylabel("fraction of draws")
    axes[2].set_title("RandomResizedCrop deterministic-centre-crop fallback")

    for axis in axes:
        axis.set_xticks(list(positions))
        axis.set_xticklabels(labels)
    figure.suptitle(
        "View geometry. A view's information content is its native pixel count, not its tensor size."
    )
    figure.tight_layout()
    return figure


def throughput_figure(epoch: Mapping[str, Sequence[float]]):
    """Figure 5: where the wall clock and the memory went."""
    from src.utils.visualization import plot_series_panels

    panels = [
        _panel(epoch, "epoch", "Throughput", {"images/s": "images_per_second"}, ylabel="images/s"),
        _panel(epoch, "epoch", "Epoch duration", {"seconds": "duration_seconds"}, ylabel="s"),
        _panel(
            epoch,
            "epoch",
            "Loop blocked in the dataloader (UPPER BOUND on GPU idle)",
            {"loop blocked": "loop_blocked_fraction", "GPU busy (measured)": "gpu_busy_fraction"},
            ylabel="fraction of epoch",
            ylim=(0.0, 1.0),
        ),
        _panel(
            epoch,
            "epoch",
            "Peak memory",
            {"allocated": "peak_memory_mb", "reserved": "peak_reserved_mb"},
            ylabel="MB",
        ),
    ]
    live = [panel for panel in panels if panel]
    if not live:
        return None
    return plot_series_panels(
        live,
        columns=2,
        suptitle=(
            "Cost. `loop_blocked_fraction` bounds GPU idleness from above -- the queued work "
            "drains during that window, so 1 - blocked is the CPU enqueue time, not a busy time."
        ),
    )


def generate_stage1_figures(
    csv_dir: str | Path,
    output_dir: str | Path,
    dpi: int = 300,
    best_epoch: float | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, list[str]]:
    """Write every figure this module can build from ``csv_dir``.

    Returns ``{figure name: [paths]}`` for the ones that had data. A figure whose
    inputs are missing is skipped and named in the log rather than raising, so an
    interrupted run still gets whatever it earned.
    """
    log = logger or LOGGER
    source = Path(csv_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    from src.utils.visualization import use_publication_style

    use_publication_style()

    epoch = read_csv_columns(source / "metrics_epoch.csv")
    step = read_csv_columns(source / "metrics_train.csv")
    probe = read_csv_columns(source / "metrics_probe.csv")
    geometry_rows: list[dict[str, Any]] = []
    geometry_path = source / "view_geometry.csv"
    if geometry_path.exists():
        with geometry_path.open(newline="", encoding="utf-8") as handle:
            geometry_rows = [dict(row) for row in csv.DictReader(handle)]

    if best_epoch is None:
        selection = read_csv_columns(source / "checkpoint_selection.csv")
        candidates = [
            value
            for value in selection.get("best_epoch", [])
            if isinstance(value, float) and not math.isnan(value)
        ]
        best_epoch = candidates[-1] if candidates else None

    builders = {
        "01_optimization": lambda: optimization_figure(epoch, step),
        "02_representation": lambda: representation_figure(probe, best_epoch),
        "03_collapse": lambda: collapse_figure(step),
        "04_view_geometry": lambda: view_geometry_figure(geometry_rows),
        "05_throughput": lambda: throughput_figure(epoch),
    }

    written: dict[str, list[str]] = {}
    for name, build in builders.items():
        try:
            figure = build()
        except Exception as exc:  # pragma: no cover - a figure must not kill a run
            log.warning("Stage-1 figure %s failed: %s", name, exc, exc_info=True)
            continue
        if figure is None:
            log.info("Stage-1 figure %s skipped: no data for any of its panels.", name)
            continue
        written[name] = _save(figure, destination, name, dpi)
    if written:
        log.info(
            "Wrote %s stage-1 figures to %s (%s each).",
            len(written), destination, "/".join(FIGURE_FORMATS),
        )
    return written
