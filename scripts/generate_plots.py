#!/usr/bin/env python
"""Build the publication figures and the cross-run comparison table.

    python scripts/generate_plots.py
    python scripts/generate_plots.py --roots outputs/ablations outputs/baselines
    python scripts/generate_plots.py --dpi 600 --no-figures     # table only

Reads what the training runs already wrote -- ``summary.json`` for the scalar
metrics and efficiency report, ``test_predictions.npz`` for the raw held-out
predictions -- and produces, into ``outputs/reports/``:

* ``summary_metrics.csv``: one row per variant or baseline, with the columns the
  revision requests (accuracy, precision, recall, macro/micro F1, KL alignment
  rate, total and active parameters, inference latency) plus seed-type metrics,
  AUC, throughput, FLOPs and peak memory.
* ``{variant}_confusion_seed_type.png``: 4-class matrix, row-normalised.
* ``{variant}_confusion_sub_variety.png``: all 27 sub-varieties with full,
  unabbreviated tick labels on both axes.
* ``{variant}_tsne_seed_type.png`` / ``_tsne_sub_variety.png``: 384-D test
  embeddings projected to 2-D, with class names overlaid on the clusters and a
  colour legend.
* ``{variant}_loss_curves.png``: training against validation loss.
* per-class metric heatmaps, misclassification rates, expert utilisation.

Nothing here retrains or reloads a model, so re-plotting at a different DPI or
normalisation costs seconds. Run it after ``run_ablations.py`` and
``run_baselines.py``, or at any later point.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.trainers.runner import output_root
from src.utils.evaluation import (
    PREDICTIONS_FILENAME,
    RunSummary,
    collect_run_summaries,
    load_test_predictions,
    save_publication_figures,
    write_summary_csv,
)
from src.utils.metrics import evaluate_hierarchical


def parse_args() -> argparse.Namespace:
    root = output_root()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=[
            str(root / "ablations"),
            str(root / "baselines"),
            str(root / "finetune_hierarchical_moe"),
        ],
        help="Directories to scan for summary.json files.",
    )
    parser.add_argument("--output-dir", default=str(root / "reports"), help="Where to write the report.")
    parser.add_argument("--dpi", type=int, default=300, help="Figure resolution (default: 300, print quality).")
    parser.add_argument("--no-figures", action="store_true", help="Write only summary_metrics.csv.")
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--max-tsne-samples", type=int, default=2000)
    return parser.parse_args()


def regenerate_figures(summary: RunSummary, output_dir: Path, args: argparse.Namespace) -> int:
    """Re-score one run's saved predictions and write its figures. Returns the count."""
    predictions_path = summary.artifacts.get("predictions") or (
        Path(summary.run_dir) / PREDICTIONS_FILENAME
    )
    if not Path(predictions_path).exists():
        print(f"  {summary.name}: no {PREDICTIONS_FILENAME}, skipping figures")
        return 0

    data = load_test_predictions(predictions_path)
    seed_names = [str(name) for name in data["seed_type_names"]]
    sub_names = [str(name) for name in data["subvariety_names"]]

    # Re-scoring from the raw predictions rather than trusting summary.json means
    # the figures and the table are computed by the same code, at plot time.
    evaluation = evaluate_hierarchical(
        seed_true=data["seed_true"],
        seed_pred=data["seed_pred"],
        sub_true=data["sub_true"],
        sub_pred=data["sub_pred"],
        subvariety_to_seed_type=data["subvariety_to_seed_type"].tolist(),
        num_seed_types=len(seed_names),
        num_sub_varieties=len(sub_names),
        seed_type_names=seed_names,
        sub_variety_names=sub_names,
        sub_scores=data.get("sub_scores"),
        top_k_indices=data.get("expert_indices"),
        num_experts=int(summary.efficiency.get("parameters", {}).get("num_experts", 6) or 6),
    )

    written = save_publication_figures(
        evaluation,
        output_dir=output_dir,
        prefix=summary.name,
        embeddings=data.get("embeddings"),
        seed_labels=data["seed_true"].tolist(),
        sub_labels=data["sub_true"].tolist(),
        history=summary.history,
        dpi=args.dpi,
        tsne_perplexity=args.tsne_perplexity,
        max_tsne_samples=args.max_tsne_samples,
    )
    print(f"  {summary.name}: {len(written)} figures")
    return len(written)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = collect_run_summaries(args.roots)
    if not summaries:
        print(
            "No summary.json found under:\n  "
            + "\n  ".join(args.roots)
            + "\n\nRun scripts/run_ablations.py or scripts/run_baselines.py first."
        )
        return 1

    print(f"Found {len(summaries)} runs:")
    for summary in summaries:
        print(f"  [{summary.group}] {summary.name} <- {summary.run_dir}")

    csv_path = write_summary_csv(output_dir / "summary_metrics.csv", summaries)
    print(f"\nSummary table: {csv_path}")

    if not args.no_figures:
        print("\nRegenerating figures:")
        total = sum(regenerate_figures(summary, output_dir, args) for summary in summaries)
        print(f"\nWrote {total} figures to {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
