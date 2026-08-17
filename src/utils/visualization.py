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

import matplotlib.colors  # noqa: E402
import matplotlib.lines  # noqa: E402
import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend selection)
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.utils.metrics import ClassMetrics  # noqa: E402

# ---------------------------------------------------------------------------
# Publication style.
#
# The training-time figures above are diagnostics: they are read once, in a
# browser, next to a loss curve. The stage-1 evaluation figures are meant for a
# paper, where they are read at 300 dpi in a single column at roughly half the
# width they were drawn at. That changes what is legible, so the style below
# raises tick and label sizes relative to the figure, thins the grid, drops the
# top and right spines, and pins a colour-blind-safe categorical cycle.
#
# Applied by `use_publication_style()` rather than at import, because importing a
# module must not silently restyle a caller's unrelated figures.
# ---------------------------------------------------------------------------

#: Okabe-Ito, which stays distinguishable under the three common colour-vision
#: deficiencies and in greyscale print. Used for series, encoders and seed types
#: -- anything with few enough categories to name.
OKABE_ITO = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
)

PUBLICATION_RC: dict[str, Any] = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#D9D9D9",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.fontsize": 8,
    "legend.frameon": False,
    "lines.linewidth": 1.6,
    "lines.markersize": 4.0,
    "figure.titlesize": 11,
    "figure.titleweight": "bold",
}


def use_publication_style() -> None:
    """Apply :data:`PUBLICATION_RC` and the Okabe-Ito colour cycle globally."""
    from cycler import cycler

    plt.rcParams.update(PUBLICATION_RC)
    plt.rcParams["axes.prop_cycle"] = cycler(color=list(OKABE_ITO))


def category_colors(count: int, colormap: str = "tab20") -> np.ndarray:
    """``count`` distinguishable RGBA colours.

    Okabe-Ito below nine categories, a sampled colormap above it. 27
    sub-varieties cannot be given eight hues, and pretending otherwise is how a
    27-class scatter ends up with four indistinguishable blues.
    """
    if count <= len(OKABE_ITO):
        return np.array([matplotlib.colors.to_rgba(color) for color in OKABE_ITO[:count]])
    return plt.get_cmap(colormap)(np.linspace(0, 1, max(count, 1)))


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


# ---------------------------------------------------------------------------
# Stage-1 representation figures.
#
# These are the primitives the stage-1 evaluation composes. They are generic on
# purpose -- a panel spec, a bar group, a matrix -- so the evaluation decides what
# a figure *says* and this module only decides how it looks.
# ---------------------------------------------------------------------------


def plot_series_panels(
    panels: Sequence[Mapping[str, Any]],
    columns: int = 3,
    panel_size: tuple[float, float] = (4.2, 2.9),
    suptitle: str | None = None,
) -> "plt.Figure":
    """Grid of line panels sharing nothing but a figure.

    Each entry of ``panels`` is a dict with ``title``, ``xlabel``, ``ylabel`` and
    ``series`` (``{label: (x, y)}``), plus optional ``hlines``
    (``[{y, label, color, style}]``), ``ylim``, ``yscale`` and ``bands``
    (``{label: (x, low, high)}``).

    Written as one function rather than fifteen because a stage-1 dynamics figure
    is fifteen near-identical line plots, and fifteen near-identical functions is
    how a figure module stops being maintainable.
    """
    count = max(len(panels), 1)
    rows = -(-count // max(columns, 1))
    figure, axes_grid = plt.subplots(
        rows,
        min(columns, count),
        figsize=(panel_size[0] * min(columns, count), panel_size[1] * rows),
        squeeze=False,
    )
    flat = [axis for row in axes_grid for axis in row]

    for axis, panel in zip(flat, panels):
        for label, (x_values, y_values) in dict(panel.get("series", {})).items():
            axis.plot(np.asarray(x_values), np.asarray(y_values), label=label)
        for label, (x_values, low, high) in dict(panel.get("bands", {})).items():
            axis.fill_between(
                np.asarray(x_values), np.asarray(low), np.asarray(high), alpha=0.18, linewidth=0, label=label
            )
        for line in panel.get("hlines", []) or []:
            axis.axhline(
                float(line["y"]),
                linestyle=line.get("style", "--"),
                color=line.get("color", "#666666"),
                linewidth=1.0,
                label=line.get("label"),
            )
        axis.set_title(str(panel.get("title", "")))
        axis.set_xlabel(str(panel.get("xlabel", "")))
        axis.set_ylabel(str(panel.get("ylabel", "")))
        if panel.get("yscale"):
            axis.set_yscale(str(panel["yscale"]))
        if panel.get("xscale"):
            axis.set_xscale(str(panel["xscale"]))
        if panel.get("ylim"):
            axis.set_ylim(*panel["ylim"])
        handles, labels = axis.get_legend_handles_labels()
        if labels:
            axis.legend(loc=panel.get("legend_loc", "best"))

    for axis in flat[len(panels) :]:
        axis.axis("off")
    if suptitle:
        figure.suptitle(suptitle)
    figure.tight_layout()
    return figure


def plot_grouped_bars(
    categories: Sequence[str],
    groups: Mapping[str, Sequence[float]],
    ylabel: str = "",
    title: str = "",
    errors: Mapping[str, Sequence[float]] | None = None,
    reference: float | None = None,
    reference_label: str = "",
    annotate: bool = True,
    figsize: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> "plt.Figure":
    """Clustered bar chart with optional error bars and a reference line.

    ``reference`` draws the chance level. A 27-class accuracy of 0.30 is either
    excellent or catastrophic depending on whether the reader remembers that
    chance is 0.037, and a dashed line removes the ambiguity from the figure
    rather than leaving it to the caption.
    """
    labels = list(categories)
    names = list(groups)
    width = 0.82 / max(len(names), 1)
    positions = np.arange(len(labels), dtype=np.float64)
    colors = category_colors(len(names))
    # Width from the *bar* count, not the category count: eight bars in a 5-inch
    # axes leaves no room for the value labels, and the value labels are half the
    # reason a bar chart is used for four numbers instead of a table.
    bars_total = max(len(labels) * max(len(names), 1), 1)
    default_size = (max(4.6, 0.52 * bars_total + 1.8), 3.6)

    figure, axis = plt.subplots(figsize=figsize or default_size)
    finite = [
        float(value)
        for series in groups.values()
        for value in series
        if isinstance(value, (int, float)) and value == value
    ]
    finite_max = max(finite, default=1.0)
    # Negative bars are kept, not clamped: a negative silhouette is a *result*
    # (the classes are not separated clusters), and a floor at zero would print it
    # as 0.000 next to genuinely-zero quantities.
    finite_min = min(finite, default=0.0)

    for index, name in enumerate(names):
        offset = (index - (len(names) - 1) / 2) * width
        values = np.asarray(list(groups[name]), dtype=np.float64)
        error = np.asarray(list(errors[name]), dtype=np.float64) if errors and name in errors else None
        bars = axis.bar(
            positions + offset,
            values,
            width=width * 0.92,
            label=name,
            color=colors[index],
            yerr=error,
            capsize=2.5 if error is not None else 0,
            error_kw={"elinewidth": 0.9},
        )
        if annotate:
            # Vertical labels once there is more than one series per category:
            # horizontal ones from adjacent bars run into each other and render as
            # "0.9170.917", which is worse than no label at all.
            rotation = 90 if len(names) > 1 else 0
            for index_in_group, (bar, value) in enumerate(zip(bars, values)):
                if not np.isfinite(value):
                    continue
                # Anchor above the error bar's cap, not above the bar: at the bar
                # the label sits inside the whisker and the two overlap.
                cap = float(error[index_in_group]) if error is not None else 0.0
                cap = cap if np.isfinite(cap) else 0.0
                axis.annotate(
                    f"{value:.3f}",
                    (bar.get_x() + bar.get_width() / 2, value + cap),
                    textcoords="offset points",
                    xytext=(0, 4),
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    rotation=rotation,
                )

    if reference is not None:
        axis.axhline(
            float(reference),
            linestyle=":",
            color="#444444",
            linewidth=1.0,
            label=reference_label or f"chance ({reference:.3f})",
        )

    axis.set_xticks(positions)
    long_labels = max((len(str(item)) for item in labels), default=0) >= 10
    axis.set_xticklabels(
        labels,
        rotation=22 if long_labels else 0,
        ha="right" if long_labels else "center",
    )
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    # Headroom for the rotated value labels, unless the caller pinned the range.
    if ylim:
        axis.set_ylim(*ylim)
    else:
        top = finite_max * (1.30 if annotate else 1.05)
        bottom = min(0.0, finite_min * 1.35)
        axis.set_ylim(bottom, top if top > bottom else bottom + 1.0)
        if bottom < 0.0:
            axis.axhline(0.0, color="#333333", linewidth=0.8)
    axis.grid(axis="x", visible=False)
    # Legend outside on the right rather than beneath: below the axes it collides
    # with rotated category labels, and every figure here has at least two.
    axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1)
    figure.tight_layout()
    return figure


def plot_spectrum(
    spectra: Mapping[str, Sequence[float]],
    cumulative: Mapping[str, Sequence[float]] | None = None,
    annotations: Mapping[str, str] | None = None,
    title: str = "Feature covariance spectrum",
) -> "plt.Figure":
    """Log-log singular-value spectrum plus the cumulative explained variance.

    The two panels answer the same question from opposite ends. The left decays
    smoothly for a healthy representation and falls off a cliff for a collapsed
    one; the right says how many of the 768 directions are actually needed, which
    is the number a reader can compare against the 384 the stage-2 projection
    keeps.
    """
    figure, (left, right) = plt.subplots(1, 2, figsize=(9.0, 3.4))
    colors = category_colors(len(spectra))

    for index, (name, values) in enumerate(spectra.items()):
        series = np.asarray(list(values), dtype=np.float64)
        series = series / max(series[0], 1e-12)
        ranks = np.arange(1, series.size + 1)
        label = f"{name} ({annotations[name]})" if annotations and name in annotations else name
        left.plot(ranks, np.maximum(series, 1e-12), label=label, color=colors[index])

    left.set_xscale("log")
    left.set_yscale("log")
    # Floor the axis: when n < d the trailing singular values are numerically zero
    # and an unclipped log axis spends ten decades on them, compressing the decay
    # that the panel exists to show.
    left.set_ylim(bottom=1e-6)
    left.set_xlabel("Singular-value index")
    left.set_ylabel("Singular value (normalised)")
    left.set_title("Spectrum")
    left.legend()

    if cumulative:
        for index, (name, values) in enumerate(cumulative.items()):
            series = np.cumsum(np.asarray(list(values), dtype=np.float64))
            right.plot(np.arange(1, series.size + 1), series, label=name, color=colors[index])
        right.axhline(0.95, linestyle=":", color="#444444", linewidth=1.0, label="95 % variance")
        right.set_xscale("log")
        right.set_xlabel("Number of principal directions")
        right.set_ylabel("Cumulative explained variance")
        right.set_ylim(0.0, 1.02)
        right.set_title("Cumulative variance")
        right.legend(loc="lower right")
    else:
        right.axis("off")

    figure.suptitle(title)
    figure.tight_layout()
    return figure


def plot_similarity_matrix(
    matrix: np.ndarray,
    labels: Sequence[str],
    block_boundaries: Sequence[int] = (),
    block_labels: Sequence[str] = (),
    title: str = "Class-centroid cosine similarity",
    colormap: str = "RdBu_r",
    symmetric_scale: bool = True,
) -> "plt.Figure":
    """Class-by-class similarity heatmap with hierarchy block dividers.

    ``block_boundaries`` are indices where a parent group ends. Drawing them is
    the whole point of the figure: if the representation carries the taxonomy,
    the blocks are visibly warmer inside than outside, and that is the property
    the stage-2 hierarchical KL term takes for granted.
    """
    values = np.asarray(matrix, dtype=np.float64).copy()
    # The diagonal is 1.0 by construction and carries no information. Left in, it
    # takes the top of the colour scale and compresses every off-diagonal entry
    # -- which is the entire content of the figure -- into the middle two shades.
    # Masked out instead, and the scale is set from the off-diagonal range.
    off_diagonal = ~np.eye(values.shape[0], dtype=bool)
    finite = values[off_diagonal & np.isfinite(values)]
    np.fill_diagonal(values, np.nan)
    if finite.size and symmetric_scale:
        span = float(np.nanmax(np.abs(finite)))
        low, high = -span, span
    elif finite.size:
        low, high = float(finite.min()), float(finite.max())
    else:
        low, high = -1.0, 1.0
    size = float(np.clip(4.0 + 0.30 * values.shape[0], 5.0, 14.0))

    figure, axis = plt.subplots(figsize=(size, size * 0.86))
    colours = plt.get_cmap(colormap).copy()
    colours.set_bad("#FFFFFF")
    image = axis.imshow(values, cmap=colours, vmin=low, vmax=high, interpolation="nearest")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="cosine similarity")

    axis.set_xticks(range(values.shape[1]))
    axis.set_yticks(range(values.shape[0]))
    axis.set_xticklabels(list(labels), rotation=90)
    axis.set_yticklabels(list(labels))
    axis.grid(visible=False)

    for boundary in block_boundaries:
        axis.axhline(boundary - 0.5, color="black", linewidth=1.1)
        axis.axvline(boundary - 0.5, color="black", linewidth=1.1)

    if block_labels and block_boundaries:
        edges = [0, *list(block_boundaries), values.shape[0]]
        for name, start, stop in zip(block_labels, edges[:-1], edges[1:]):
            axis.text(
                (start + stop - 1) / 2,
                -1.2,
                str(name),
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    # Padded well clear of the block labels, which sit above row 0.
    axis.set_title(title, pad=34 if block_labels else 12)
    figure.tight_layout()
    return figure


def plot_curves_with_bands(
    series: Mapping[str, tuple[Sequence[float], Sequence[float], Sequence[float]]],
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    logx: bool = False,
    reference: float | None = None,
    reference_label: str = "",
) -> "plt.Figure":
    """Mean line with a +-SD band per series (``{name: (x, mean, sd)}``).

    Used for the low-shot curve, where at one label per class the sampling SD is
    larger than the difference between encoders. A plain line there would present
    noise as a result; the band makes the overlap visible.
    """
    figure, axis = plt.subplots(figsize=(5.4, 3.4))
    colors = category_colors(len(series))
    for index, (name, (x_values, mean, deviation)) in enumerate(series.items()):
        x_array = np.asarray(list(x_values), dtype=np.float64)
        mean_array = np.asarray(list(mean), dtype=np.float64)
        sd_array = np.asarray(list(deviation), dtype=np.float64)
        axis.plot(x_array, mean_array, marker="o", label=name, color=colors[index])
        axis.fill_between(
            x_array, mean_array - sd_array, mean_array + sd_array, alpha=0.18, linewidth=0, color=colors[index]
        )
    if reference is not None:
        axis.axhline(
            float(reference), linestyle=":", color="#444444", linewidth=1.0, label=reference_label or "chance"
        )
    if logx:
        axis.set_xscale("log")
        axis.set_xticks(sorted({int(v) for values, _, _ in series.values() for v in values}))
        axis.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        # A log axis also labels its minor ticks ("2 x 10^0"), which on a shot-count
        # axis are not shot counts and read as an error.
        axis.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    return figure


def plot_per_class_bars(
    names: Sequence[str],
    values: Sequence[float],
    group_of: Sequence[int] | None = None,
    group_names: Sequence[str] | None = None,
    supports: Sequence[int] | None = None,
    xlabel: str = "",
    title: str = "",
    reference: float | None = None,
    reference_label: str = "",
) -> "plt.Figure":
    """Sorted horizontal bars coloured by parent group, annotated with support.

    Colouring by seed type is what makes the figure a *hierarchical* result: if
    the weak classes are all one colour, that is a statement about a branch of the
    taxonomy rather than about 27 unrelated classes.
    """
    order = np.argsort(np.asarray(list(values), dtype=np.float64))
    ordered_names = [str(names[i]) for i in order]
    ordered_values = [float(values[i]) for i in order]
    colors = category_colors(len(set(group_of)) if group_of is not None else 1)
    bar_colors = (
        [colors[int(group_of[i])] for i in order] if group_of is not None else [OKABE_ITO[0]] * len(order)
    )

    figure, axis = plt.subplots(figsize=(6.4, max(3.0, 0.26 * len(ordered_names) + 1.2)))
    bars = axis.barh(range(len(ordered_names)), ordered_values, color=bar_colors, height=0.74)
    axis.set_yticks(range(len(ordered_names)))
    axis.set_yticklabels(ordered_names)
    axis.set_xlabel(xlabel)
    axis.set_title(title)
    axis.set_xlim(0.0, 1.02)
    axis.grid(axis="y", visible=False)

    for index, bar in enumerate(bars):
        text = f"{ordered_values[index]:.2f}"
        if supports is not None:
            text += f"  (n={int(supports[order[index]])})"
        axis.annotate(
            text,
            (bar.get_width(), bar.get_y() + bar.get_height() / 2),
            textcoords="offset points",
            xytext=(3, 0),
            va="center",
            fontsize=6.5,
        )

    if reference is not None:
        axis.axvline(
            float(reference), linestyle=":", color="#444444", linewidth=1.0, label=reference_label or "chance"
        )
    if group_of is not None and group_names is not None:
        handles = [
            matplotlib.patches.Patch(facecolor=colors[index], label=str(name))
            for index, name in enumerate(group_names)
        ]
        # Above the axes, not inside: the sorted bars leave their empty space at the
        # bottom-right, which is exactly where a low-F1 class's value label goes.
        axis.legend(
            handles=handles,
            ncol=min(len(handles), 4),
            loc="lower right",
            bbox_to_anchor=(1.0, 1.005),
        )
    figure.tight_layout()
    return figure


def _draw_distributions(
    axis: Any,
    distributions: Mapping[str, Sequence[float]],
    bins: int,
    annotate_means: bool,
) -> None:
    """Overlay normalised histograms with a dashed mean line each."""
    colors = category_colors(len(distributions))
    for index, (name, values) in enumerate(distributions.items()):
        array = np.asarray(list(values), dtype=np.float64)
        array = array[np.isfinite(array)]
        if array.size == 0:
            continue
        axis.hist(
            array,
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.35,
            color=colors[index],
            label=f"{name} (mean {array.mean():.3f})" if annotate_means else name,
        )
        axis.axvline(array.mean(), color=colors[index], linewidth=1.2, linestyle="--")


def plot_distribution_overlay(
    distributions: Mapping[str, Sequence[float]],
    xlabel: str = "",
    title: str = "",
    bins: int = 60,
    annotate_means: bool = True,
) -> "plt.Figure":
    """Overlaid normalised histograms, one per named distribution.

    Densities rather than counts, because the populations being compared differ in
    size by orders of magnitude. The legend sits below the axes: these are peaked
    distributions and an in-axes legend lands on the tallest bars.
    """
    figure, axis = plt.subplots(figsize=(6.0, 3.6))
    _draw_distributions(axis, distributions, bins, annotate_means)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Density")
    axis.set_title(title)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1)
    figure.tight_layout()
    return figure


def plot_distribution_panels(
    panels: Mapping[str, Mapping[str, Sequence[float]]],
    xlabel: str = "",
    title: str = "",
    bins: int = 60,
    share_x: bool = True,
) -> "plt.Figure":
    """One histogram panel per group, with a shared x axis.

    For the invariance figure the alternative — six overlaid histograms in one
    axes — is unreadable, and worse, it invites comparing an absolute cosine
    across encoders. Cosines are inflated by whatever mean direction a pretrained
    trunk happens to have, so only the *gaps within* a panel are comparable. One
    panel per encoder makes that structural rather than a caption's problem.
    """
    names = list(panels)
    figure, axes_row = plt.subplots(
        1, max(len(names), 1), figsize=(4.6 * max(len(names), 1), 3.6), squeeze=False, sharex=share_x
    )
    for axis, name in zip(axes_row[0], names):
        _draw_distributions(axis, panels[name], bins, annotate_means=True)
        axis.set_title(name)
        axis.set_xlabel(xlabel)
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), fontsize=7)
    axes_row[0][0].set_ylabel("Density")
    if title:
        figure.suptitle(title)
    figure.tight_layout()
    return figure


def plot_embedding_comparison(
    projections: Mapping[str, tuple[np.ndarray, Sequence[int]]],
    class_names: Sequence[str] | None = None,
    title: str = "",
    point_size: float = 5.0,
    annotate: bool = False,
) -> "plt.Figure":
    """Side-by-side 2-D scatters sharing one legend and one colour assignment.

    The shared colour assignment is the point: two independently-legended
    scatters of 27 classes cannot be compared by eye, and the comparison is the
    figure's only purpose.
    """
    names = list(projections)
    figure, axes_row = plt.subplots(
        1, max(len(names), 1), figsize=(4.6 * max(len(names), 1) + 1.6, 4.4), squeeze=False
    )
    axes = axes_row[0]

    all_labels = sorted({int(label) for _, labels in projections.values() for label in labels})
    colors = category_colors(len(all_labels))
    color_for = {label: colors[index] for index, label in enumerate(all_labels)}

    def name_for(label: int) -> str:
        if class_names is not None and 0 <= label < len(class_names):
            return str(class_names[label])
        return str(label)

    for axis, name in zip(axes, names):
        projection, labels = projections[name]
        projection = np.asarray(projection, dtype=np.float64)
        labels = np.asarray(list(labels))[: projection.shape[0]]
        for label in all_labels:
            mask = labels == label
            if not mask.any():
                continue
            axis.scatter(
                projection[mask, 0],
                projection[mask, 1],
                s=point_size,
                color=color_for[label],
                linewidths=0,
                alpha=0.8,
            )
            if annotate and int(mask.sum()) >= 3:
                centre = np.median(projection[mask], axis=0)
                axis.annotate(
                    name_for(label),
                    xy=(centre[0], centre[1]),
                    fontsize=6,
                    ha="center",
                    va="center",
                    bbox={
                        "boxstyle": "round,pad=0.15",
                        "facecolor": color_for[label],
                        "edgecolor": "black",
                        "linewidth": 0.3,
                        "alpha": 0.75,
                    },
                )
        axis.set_title(name)
        axis.set_xlabel("t-SNE 1")
        axis.set_ylabel("t-SNE 2")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.grid(visible=False)

    handles = [
        matplotlib.lines.Line2D(
            [], [], marker="o", linestyle="", markersize=4, color=color_for[label], label=name_for(label)
        )
        for label in all_labels
    ]
    figure.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        ncol=1 if len(handles) <= 30 else 2,
        title="Class",
        title_fontsize=8,
    )
    if title:
        figure.suptitle(title)
    figure.tight_layout()
    return figure


def plot_reliability_diagram(
    confidences: Sequence[float],
    correct: Sequence[bool],
    num_bins: int = 15,
    title: str = "Reliability",
    ece: float | None = None,
) -> "plt.Figure":
    """Confidence-vs-accuracy diagram with the bin-population histogram beneath."""
    confidence = np.asarray(list(confidences), dtype=np.float64)
    hits = np.asarray(list(correct), dtype=np.float64)
    edges = np.linspace(0.0, 1.0, int(num_bins) + 1)
    centres, accuracies, populations = [], [], []
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        populations.append(float(mask.mean()))
        centres.append(float((lower + upper) / 2))
        accuracies.append(float(hits[mask].mean()) if mask.any() else np.nan)

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(4.4, 4.6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    top.plot([0, 1], [0, 1], linestyle=":", color="#444444", linewidth=1.0, label="perfect calibration")
    top.plot(centres, accuracies, marker="o", color=OKABE_ITO[0], label="observed")
    top.set_ylabel("Accuracy")
    top.set_ylim(0.0, 1.02)
    label = title if ece is None else f"{title} (ECE {ece:.3f})"
    top.set_title(label)
    top.legend(loc="upper left")

    bottom.bar(centres, populations, width=1.0 / max(num_bins, 1) * 0.9, color="#999999")
    bottom.set_xlabel("Confidence")
    bottom.set_ylabel("Fraction")
    bottom.set_xlim(0.0, 1.0)
    figure.tight_layout()
    return figure


def plot_retrieval_examples(
    queries: Sequence[np.ndarray],
    neighbours: Sequence[Sequence[np.ndarray]],
    query_labels: Sequence[str],
    neighbour_correct: Sequence[Sequence[bool]],
    title: str = "Cosine nearest neighbours in the frozen embedding",
) -> "plt.Figure":
    """Query image plus its nearest neighbours, framed green (same class) or red.

    The qualitative counterpart to precision@k, and the only figure here that
    shows what the encoder is actually confusing: two crops of visibly different
    grain that land next to each other say more about the failure than a number
    does.
    """
    rows = len(queries)
    columns = 1 + (len(neighbours[0]) if rows else 0)
    figure, axes_grid = plt.subplots(
        rows, columns, figsize=(1.28 * columns, 1.42 * rows), squeeze=False
    )

    for row in range(rows):
        axis = axes_grid[row][0]
        axis.imshow(np.clip(np.asarray(queries[row]), 0.0, 1.0))
        axis.set_xticks([])
        axis.set_yticks([])
        axis.grid(visible=False)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("#222222")
            spine.set_linewidth(1.4)
        axis.set_ylabel(str(query_labels[row]), fontsize=6.5, rotation=0, ha="right", va="center", labelpad=6)
        if row == 0:
            axis.set_title("query", fontsize=7)

        for column, image in enumerate(neighbours[row], start=1):
            neighbour_axis = axes_grid[row][column]
            neighbour_axis.imshow(np.clip(np.asarray(image), 0.0, 1.0))
            neighbour_axis.set_xticks([])
            neighbour_axis.set_yticks([])
            neighbour_axis.grid(visible=False)
            colour = "#009E73" if neighbour_correct[row][column - 1] else "#D55E00"
            for spine in neighbour_axis.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(colour)
                spine.set_linewidth(1.8)
            if row == 0:
                neighbour_axis.set_title(f"NN {column}", fontsize=7)

    figure.suptitle(title)
    figure.tight_layout()
    return figure
