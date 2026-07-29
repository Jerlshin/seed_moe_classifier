#!/usr/bin/env python
"""Run the component-wise ablation suite.

    python scripts/run_ablations.py                        # all six variants
    python scripts/run_ablations.py --variants wo_moe wo_kl
    python scripts/run_ablations.py --dry-run              # print commands only
    python scripts/run_ablations.py -- data.batch_size=8 experiment.training.epochs=20

Each variant removes exactly one architectural ingredient from the full model
and trains everything else identically: same encoder weights, same split, same
optimiser, same schedule, same seed. That is what makes the resulting deltas
attributable to the removed component.

Results land in ``outputs/ablations/{variant}/``, one self-contained directory
per variant holding its Hydra config snapshot, logs, checkpoints, figures,
``summary.json`` and ``test_predictions.npz``. Build the comparison table with::

    python scripts/generate_plots.py

**Pretraining is not repeated.** All six variants read the single published
encoder at ``outputs/checkpoints/dinov2_swinv2_pretrained.pth``. Re-running
self-supervised pretraining per variant would make each row partly a function of
its own initialisation, and would cost six times as much for a worse experiment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.trainers.runner import (
    VariantSpec,
    default_checkpoint_path,
    ensure_pretrained_checkpoint,
    output_root,
    print_summary,
    run_suite,
    write_suite_manifest,
)

#: The six variants the revision asks for, in reporting order.
ABLATION_VARIANTS: list[VariantSpec] = [
    VariantSpec(
        name="full_model",
        description="Full framework: Top-2 MoE + ArcFace + residual + KL + cross-attention",
        overrides=[],
    ),
    VariantSpec(
        name="wo_moe",
        description="No sparse routing: one dense transformer block feeds the heads",
        # A dense block of identical architecture replaces the six experts, so
        # the delta measures *routing* rather than a missing layer's capacity.
        overrides=["model.head.use_moe=false"],
    ),
    VariantSpec(
        name="wo_arcface",
        description="Sub-variety classified by a linear head under categorical cross-entropy",
        overrides=["model.head.use_arcface=false"],
    ),
    VariantSpec(
        name="wo_residual",
        description="No 4D -> 384D projected seed-type fusion (Eq. 9)",
        overrides=["model.head.use_residual=false"],
    ),
    VariantSpec(
        name="wo_kl",
        description="No stage-1/stage-2 KL divergence alignment loss (Eq. 10)",
        overrides=["model.head.use_kl_loss=false"],
    ),
    VariantSpec(
        name="wo_cross_attn",
        description="No Q/K/V refinement: h'' = h' (Eqs. 11-12 skipped)",
        overrides=["model.head.use_cross_attention=false"],
    ),
]

VARIANTS_BY_NAME = {spec.name: spec for spec in ABLATION_VARIANTS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=sorted(VARIANTS_BY_NAME),
        default=[spec.name for spec in ABLATION_VARIANTS],
        help="Subset of variants to run (default: all).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Shared DINOv2-SwinV2 encoder checkpoint. "
        "Defaults to $SEED_PRETRAIN_BACKBONE, else outputs/checkpoints/dinov2_swinv2_pretrained.pth.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Root for outputs/ (default: $SEED_OUTPUT_DIR, else ./outputs).",
    )
    parser.add_argument(
        "--allow-missing-checkpoint",
        action="store_true",
        help="Train from a randomly initialised encoder when the checkpoint is absent. "
        "Smoke runs only -- results are not comparable.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without running them.")
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Abort the suite at the first failing variant instead of continuing.",
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Extra Hydra overrides applied to every variant, after a bare '--'.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.output_root) if args.output_root else output_root()
    checkpoint = ensure_pretrained_checkpoint(
        args.checkpoint or default_checkpoint_path(),
        allow_missing=args.allow_missing_checkpoint,
    )
    # argparse.REMAINDER keeps the separating "--" when one is present.
    extra = [item for item in args.overrides if item != "--"]

    if checkpoint is None:
        print("WARNING: no pretrained checkpoint; training from a random encoder.", flush=True)
        # The config's default checkpoint path still points at the shared file,
        # and the encoder raises if it is absent. Say "no checkpoint" explicitly
        # rather than leaving the default to fail.
        extra = [*extra, "model.backbone.checkpoint_path=null"]

    specs = [VARIANTS_BY_NAME[name] for name in args.variants]

    print(f"Ablation suite: {len(specs)} variants -> {root / 'ablations'}")
    print(f"Shared encoder: {checkpoint if checkpoint else '(none)'}")
    if extra:
        print(f"Extra overrides: {' '.join(extra)}")

    results = run_suite(
        specs,
        root=root,
        checkpoint=checkpoint,
        extra_overrides=extra,
        dry_run=args.dry_run,
        stop_on_failure=args.stop_on_failure,
    )

    if not args.dry_run:
        manifest = write_suite_manifest(root / "ablations" / "suite_manifest.json", specs, results)
        print(f"Suite manifest: {manifest}")
    return print_summary(results)


if __name__ == "__main__":
    raise SystemExit(main())
