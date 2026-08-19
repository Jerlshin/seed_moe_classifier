#!/usr/bin/env python
"""Regenerate a stage-1 run's publication figures from its CSV artifacts.

    python scripts/plot_stage1_run.py outputs/hydra/2026-08-19/12-08-26
    python scripts/plot_stage1_run.py <run dir> --output figures/ --dpi 600

The trainer already writes these at the end of a run. This exists so they can be
rebuilt without one -- after editing a figure, on a laptop, from a run directory
copied off a server, or for a run that was killed before its final block.

It reads ``<run dir>/csv/*.csv`` and nothing else: no checkpoint, no GPU, no
event-stream parser, no W&B. That constraint is what makes the figures and the
tables the same numbers by construction, and it is why the trainer itself
generates them from the CSVs rather than from the values it still has in memory.

Point it at either the Hydra run directory or the ``csv`` directory inside it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def resolve_csv_dir(run_dir: Path) -> Path:
    """Accept the run directory or the ``csv`` directory inside it."""
    if (run_dir / "csv").is_dir():
        return run_dir / "csv"
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="Hydra run directory, or its csv/ subdirectory.")
    parser.add_argument(
        "--output",
        default=None,
        help="Where to write the figures (default: <run dir>/figures/stage1).",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--best-epoch",
        type=float,
        default=None,
        help="Mark this epoch on the representation figure. Read from checkpoint_selection.csv "
        "when omitted.",
    )
    args = parser.parse_args()

    import logging

    from src.utils.stage1_figures import generate_stage1_figures

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        print(f"Not a directory: {run_dir}", file=sys.stderr)
        return 1

    csv_dir = resolve_csv_dir(run_dir)
    available = sorted(path.name for path in csv_dir.glob("*.csv"))
    if not available:
        print(
            f"No CSV files under {csv_dir}. A run recorded before the CSV sink existed has only "
            "events.jsonl; parse it with src.utils.evaluation.parse_pretrain_dynamics instead.",
            file=sys.stderr,
        )
        return 1
    print(f"Reading {csv_dir}: {', '.join(available)}")

    output = Path(args.output) if args.output else run_dir / "figures" / "stage1"
    written = generate_stage1_figures(
        csv_dir, output, dpi=int(args.dpi), best_epoch=args.best_epoch
    )
    if not written:
        print("No figure had data for any of its panels.", file=sys.stderr)
        return 1
    for name, paths in written.items():
        print(f"  {name}: {', '.join(paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
