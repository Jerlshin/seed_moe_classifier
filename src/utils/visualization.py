"""Matplotlib figures reproducing the paper's qualitative results (Section 6).

Every function returns a ``matplotlib.figure.Figure`` so the caller decides
whether to save it, push it to TensorBoard, or log it to W&B. The ``Agg``
backend is forced at import time because training runs are headless.

Figure map:

=========================================  ==================================
Paper figure                               Function
=========================================  ==================================
Fig. 8-9, t-SNE clustering                 :func:`plot_tsne`
Fig. 10, seed-type confusion matrix        :func:`plot_confusion_matrix`
Fig. 11, metric-wise sub-variety heatmap   :func:`plot_metric_heatmap`
Fig. 12, misclassification rates           :func:`plot_misclassification_rates`
Section 5.2, expert utilisation            :func:`plot_expert_utilization`
=========================================  ==================================
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend selection)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.utils.metrics import ClassMetrics  # noqa: E402


def denormalize_images(
    images: torch.Tensor,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """Return image tensors denormalized to ``[0, 1]`` for logging or inspection."""
    if images.ndim == 3:
        images = images.unsqueeze(0)
    mean_tensor = torch.tensor(mean, device=images.device, dtype=images.dtype).view(1, -1, 1, 1)
    std_tensor = torch.tensor(std, device=images.device, dtype=images.dtype).view(1, -1, 1, 1)
    return (images * std_tensor + mean_tensor).clamp(0, 1)


def make_image_grid(images: torch.Tensor, max_images: int = 8) -> torch.Tensor:
    """Make a simple horizontal grid without depending on ``torchvision.utils``."""
    images = images[:max_images]
    if images.ndim != 4:
        raise ValueError(f"Expected [batch, channels, height, width], got {tuple(images.shape)}")
    return torch.cat([image for image in images], dim=-1)


def _figure_size(num_labels: int, base: float = 6.0, per_label: float = 0.32) -> float:
    """Grow the canvas with the label count so 27 tick labels stay readable."""
    return float(np.clip(base + per_label * num_labels, base, 22.0))


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: Sequence[str] | None = None,
    title: str = "Confusion matrix",
    normalize: bool = True,
    colormap: str = "Blues",
    annotate_threshold: int = 12,
) -> "plt.Figure":
    """Confusion matrix heatmap (paper Fig. 10).

    Args:
        matrix: Square counts or rates, shape ``[n, n]``.
        class_names: Axis tick labels.
        title: Figure title.
        normalize: Row-normalise counts into per-true-class rates.
        colormap: Any matplotlib colormap name.
        annotate_threshold: Write the numeric value in each cell only when the
            matrix is at most this wide; 27x27 annotations are unreadable.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if normalize and matrix.sum() > 0 and matrix.max() > 1.0:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)

    size = _figure_size(matrix.shape[0])
    figure, axes = plt.subplots(figsize=(size, size * 0.85))
    image = axes.imshow(matrix, interpolation="nearest", cmap=colormap, vmin=0.0)
    figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04)

    labels = list(class_names) if class_names else [str(i) for i in range(matrix.shape[0])]
    axes.set_xticks(range(matrix.shape[1]))
    axes.set_yticks(range(matrix.shape[0]))
    axes.set_xticklabels(labels, rotation=90, fontsize=8)
    axes.set_yticklabels(labels, fontsize=8)
    axes.set_xlabel("Predicted")
    axes.set_ylabel("True")
    axes.set_title(title)

    if matrix.shape[0] <= annotate_threshold:
        midpoint = matrix.max() / 2.0 if matrix.max() > 0 else 0.5
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axes.text(
                    column,
                    row,
                    f"{matrix[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if matrix[row, column] > midpoint else "black",
                )

    figure.tight_layout()
    return figure


def plot_metric_heatmap(
    per_class: Sequence[ClassMetrics],
    title: str = "Sub-variety metrics",
    colormap: str = "viridis",
) -> "plt.Figure":
    """Precision / recall / F1 heatmap over classes (paper Fig. 11)."""
    names = [entry.name for entry in per_class]
    values = np.array(
        [[entry.precision, entry.recall, entry.f1] for entry in per_class],
        dtype=np.float64,
    )

    figure, axes = plt.subplots(figsize=(6.0, _figure_size(len(names), base=3.0, per_label=0.28)))
    image = axes.imshow(values, aspect="auto", cmap=colormap, vmin=0.0, vmax=1.0)
    figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04)

    axes.set_xticks(range(3))
    axes.set_xticklabels(["Precision", "Recall", "F1"])
    axes.set_yticks(range(len(names)))
    axes.set_yticklabels(names, fontsize=8)
    axes.set_title(title)

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axes.text(
                column,
                row,
                f"{values[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if values[row, column] < 0.6 else "black",
            )

    figure.tight_layout()
    return figure


def plot_misclassification_rates(
    rates: Mapping[str, float],
    title: str = "Misclassification rate per sub-variety",
    color: str = "#c0504d",
) -> "plt.Figure":
    """Horizontal bar chart of ``1 - recall`` per class (paper Fig. 12)."""
    ordered = sorted(rates.items(), key=lambda item: item[1], reverse=True)
    names = [name for name, _ in ordered]
    values = [value for _, value in ordered]

    figure, axes = plt.subplots(figsize=(7.5, _figure_size(len(names), base=3.0, per_label=0.26)))
    axes.barh(range(len(names)), values, color=color)
    axes.set_yticks(range(len(names)))
    axes.set_yticklabels(names, fontsize=8)
    axes.invert_yaxis()
    axes.set_xlim(0.0, 1.0)
    axes.set_xlabel("Misclassification rate")
    axes.set_title(title)
    axes.grid(axis="x", alpha=0.3)

    figure.tight_layout()
    return figure


def plot_expert_utilization(
    utilization: Sequence[float],
    title: str = "MoE expert utilisation",
    color: str = "#4f81bd",
) -> "plt.Figure":
    """Bar chart of routing share per expert, with the balanced reference line.

    The dashed line marks ``1 / num_experts``: the entropy load-balancing term
    from Section 5.2 is exactly the pressure pulling the bars towards it.
    """
    values = np.asarray(list(utilization), dtype=np.float64)
    figure, axes = plt.subplots(figsize=(6.0, 3.5))
    axes.bar(range(len(values)), values, color=color)
    if len(values) > 0:
        axes.axhline(
            1.0 / len(values),
            linestyle="--",
            color="#888888",
            label=f"balanced ({1.0 / len(values):.3f})",
        )
        axes.legend(fontsize=8)
    axes.set_xticks(range(len(values)))
    axes.set_xticklabels([f"E{i}" for i in range(len(values))])
    axes.set_xlabel("Expert")
    axes.set_ylabel("Share of routing slots")
    axes.set_title(title)
    axes.grid(axis="y", alpha=0.3)

    figure.tight_layout()
    return figure


def plot_tsne(
    projection: np.ndarray,
    labels: Sequence[int],
    class_names: Sequence[str] | None = None,
    title: str = "t-SNE of learned embeddings",
    colormap: str = "tab20",
    point_size: float = 8.0,
    annotate_clusters: bool = False,
    min_annotated_cluster: int = 3,
) -> "plt.Figure":
    """Scatter of a 2-D t-SNE projection coloured by class (paper Figs. 8-9).

    Args:
        projection: 2-D t-SNE output, shape ``[n, 2]``.
        labels: Class index per point.
        class_names: Names used by the legend and the on-plot overlays.
        title: Figure title.
        colormap: Any matplotlib colormap name.
        point_size: Scatter marker size.
        annotate_clusters: Also print each class name at its cluster's median
            position. With 27 classes a legend alone forces the reader to match
            27 similar colours by eye; the overlay says which blob is which
            directly. The *median* is used rather than the mean because t-SNE
            routinely strands a few points of a class far from its main mass,
            and a mean would drag the label into empty space between clusters.
        min_annotated_cluster: Skip the overlay for classes with fewer points
            than this, whose centroid is not meaningful.
    """
    projection = np.asarray(projection, dtype=np.float64)
    labels = np.asarray(labels)[: projection.shape[0]]
    unique = np.unique(labels)
    colors = plt.get_cmap(colormap)(np.linspace(0, 1, max(len(unique), 1)))

    def name_for(label: int) -> str:
        if class_names is not None and 0 <= int(label) < len(class_names):
            return str(class_names[int(label)])
        return str(label)

    figure, axes = plt.subplots(figsize=(9.0, 7.0) if annotate_clusters else (7.5, 6.5))
    for position, label in enumerate(unique):
        mask = labels == label
        axes.scatter(
            projection[mask, 0],
            projection[mask, 1],
            s=point_size,
            color=colors[position],
            label=name_for(label),
            alpha=0.75,
            linewidths=0,
        )

    if annotate_clusters:
        for position, label in enumerate(unique):
            mask = labels == label
            if int(mask.sum()) < min_annotated_cluster:
                continue
            centre = np.median(projection[mask], axis=0)
            axes.annotate(
                name_for(label),
                xy=(centre[0], centre[1]),
                fontsize=7,
                fontweight="bold",
                ha="center",
                va="center",
                color="black",
                # A translucent halo keeps the text legible over dense clusters
                # without hiding the points underneath it.
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": colors[position],
                    "edgecolor": "black",
                    "linewidth": 0.4,
                    "alpha": 0.75,
                },
            )

    axes.set_title(title)
    axes.set_xlabel("t-SNE 1")
    axes.set_ylabel("t-SNE 2")
    if len(unique) <= 30:
        axes.legend(
            fontsize=6,
            markerscale=1.6,
            ncol=1 if annotate_clusters else 2,
            loc="center left" if annotate_clusters else "best",
            bbox_to_anchor=(1.01, 0.5) if annotate_clusters else None,
            framealpha=0.85,
            title="Class",
            title_fontsize=7,
        )

    figure.tight_layout()
    return figure


def plot_loss_curves(
    history: Mapping[str, Sequence[float]],
    title: str = "Training curves",
    xlabel: str = "Epoch",
) -> "plt.Figure":
    """Line plot of one or more named series (paper Fig. 6)."""
    figure, axes = plt.subplots(figsize=(7.0, 4.0))
    for name, values in history.items():
        axes.plot(range(1, len(values) + 1), list(values), label=name, linewidth=1.6)
    axes.set_xlabel(xlabel)
    axes.set_ylabel("Value")
    axes.set_title(title)
    axes.grid(alpha=0.3)
    axes.legend(fontsize=8)

    figure.tight_layout()
    return figure


def save_figure(figure: "plt.Figure", path: Any, dpi: int = 150, close: bool = True) -> str:
    """Write ``figure`` to ``path``, creating parent directories as needed."""
    from pathlib import Path

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(figure)
    return str(output)
