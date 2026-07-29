#!/usr/bin/env python
"""Run the supervised and simple-hierarchical baselines.

    python scripts/run_baselines.py                          # all three
    python scripts/run_baselines.py --models resnet50
    python scripts/run_baselines.py --dry-run
    python scripts/run_baselines.py -- experiment.training.epochs=30

Three reference points, each answering a different question:

``resnet50``
    ImageNet-pretrained ResNet-50, trained end to end. The conventional CNN
    comparison.

``swin_tiny``
    ImageNet-pretrained Swin-T, trained end to end. Same shifted-window family
    as the proposed encoder but supervised rather than self-supervised, so the
    gap isolates DINOv2 pretraining plus the hierarchical head.

``hierarchical_cce``
    Two-stage coarse-to-fine with plain cross-entropy at both levels and no MoE,
    cross-attention or ArcFace. Answers how much comes from being hierarchical
    at all, as opposed to from the specific machinery on top. It reuses the
    proposed model's encoder and code path with toggles flipped, so it is the
    tightest of the three controls.

Results land in ``outputs/baselines/{model}/``. The two end-to-end baselines own
their backbones and therefore ignore the DINOv2 checkpoint by design;
``hierarchical_cce`` reads the same shared encoder as the ablation suite, which
is what keeps it comparable with the full model.

Build the combined table afterwards with ``python scripts/generate_plots.py``.
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

BASELINE_VARIANTS: list[VariantSpec] = [
    VariantSpec(
        name="resnet50",
        description="ImageNet-pretrained ResNet-50, supervised end to end",
        group="baseline",
        experiment="baseline_resnet50",
    ),
    VariantSpec(
        name="swin_tiny",
        description="ImageNet-pretrained Swin-T, supervised end to end",
        group="baseline",
        experiment="baseline_swin_tiny",
    ),
    VariantSpec(
        name="hierarchical_cce",
        description="Two-stage hierarchy, plain CCE at both levels, no MoE/cross-attn/ArcFace",
        group="baseline",
        experiment="baseline_hierarchical_cce",
    ),
]

VARIANTS_BY_NAME = {spec.name: spec for spec in BASELINE_VARIANTS}

#: Baselines that own their backbone and must not be handed the DINOv2 encoder.
END_TO_END_BASELINES = {"resnet50", "swin_tiny"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(VARIANTS_BY_NAME),
        default=[spec.name for spec in BASELINE_VARIANTS],
        help="Subset of baselines to run (default: all).",
    )
    parser.add_argument("--checkpoint", default=None, help="Shared DINOv2-SwinV2 encoder checkpoint.")
    parser.add_argument("--output-root", default=None, help="Root for outputs/.")
    parser.add_argument(
        "--allow-missing-checkpoint",
        action="store_true",
        help="Proceed when the shared encoder is absent. Only hierarchical_cce needs it.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without running them.")
    parser.add_argument("--stop-on-failure", action="store_true", help="Abort at the first failure.")
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Extra Hydra overrides applied to every baseline, after a bare '--'.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.output_root) if args.output_root else output_root()
    specs = [VARIANTS_BY_NAME[name] for name in args.models]
    extra = [item for item in args.overrides if item != "--"]

    # Only resolve the shared encoder if a selected baseline actually consumes
    # it; requiring it to run ResNet-50 would be a pointless barrier.
    needs_encoder = any(spec.name not in END_TO_END_BASELINES for spec in specs)
    checkpoint = None
    if needs_encoder:
        checkpoint = ensure_pretrained_checkpoint(
            args.checkpoint or default_checkpoint_path(),
            allow_missing=args.allow_missing_checkpoint,
        )
        if checkpoint is None:
            print("WARNING: no pretrained checkpoint; hierarchical_cce uses a random encoder.", flush=True)
            # The config default still points at the shared file, and the
            # encoder raises when it is absent, so say "none" explicitly.
            extra = [*extra, "model.backbone.checkpoint_path=null"]

    print(f"Baseline suite: {len(specs)} models -> {root / 'baselines'}")
    if extra:
        print(f"Extra overrides: {' '.join(extra)}")

    results = []
    for spec in specs:
        # An end-to-end baseline builds its own ImageNet backbone. Passing it a
        # SwinV2 DINO checkpoint would be meaningless at best and a silent
        # partial load at worst.
        spec_checkpoint = None if spec.name in END_TO_END_BASELINES else checkpoint
        results.extend(
            run_suite(
                [spec],
                root=root,
                checkpoint=spec_checkpoint,
                extra_overrides=extra,
                dry_run=args.dry_run,
                stop_on_failure=args.stop_on_failure,
            )
        )
        if args.stop_on_failure and results and not results[-1].succeeded:
            break

    if not args.dry_run:
        manifest = write_suite_manifest(root / "baselines" / "suite_manifest.json", specs, results)
        print(f"Suite manifest: {manifest}")
    return print_summary(results)


if __name__ == "__main__":
    raise SystemExit(main())
