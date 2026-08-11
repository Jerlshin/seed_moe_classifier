#!/usr/bin/env python
"""Run the supervised and simple-hierarchical baselines.

    python scripts/run_baselines.py                          # all five
    python scripts/run_baselines.py --models resnet50
    python scripts/run_baselines.py --dry-run
    python scripts/run_baselines.py -- experiment.training.epochs=30

Five reference points, each answering a different question:

``linear_probe``
    Frozen self-supervised encoder plus two linear heads, plain CE. The
    cheapest possible read of what stage 1 alone learned, with no head
    machinery to credit or blame.

``swinv2_supervised``
    ImageNet SwinV2-Base with the full hierarchical head, but no self-supervised
    stage at all. Isolates what DINO pretraining contributes while holding the
    encoder family and head architecture fixed.

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
    tightest of the five controls.

Results land in ``outputs/baselines/{model}/``. ``resnet50``, ``swin_tiny`` and
``swinv2_supervised`` own their backbones and therefore ignore the DINOv2
checkpoint by design; ``linear_probe`` and ``hierarchical_cce`` read the same
shared encoder as the ablation suite, which is what keeps them comparable with
the full model.

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
    DEFAULT_SEEDS,
    VariantSpec,
    default_checkpoint_path,
    ensure_pretrained_checkpoint,
    expand_seeds,
    output_root,
    print_summary,
    parse_gpu_list,
    run_suite,
    write_suite_manifest,
)

BASELINE_VARIANTS: list[VariantSpec] = [
    VariantSpec(
        name="linear_probe",
        description="Frozen self-supervised encoder + two linear layers, plain CE",
        group="baseline",
        experiment="baseline_linear_probe",
    ),
    VariantSpec(
        name="swinv2_supervised",
        description="ImageNet SwinV2-Base + the full hierarchical head, no self-supervised stage",
        group="baseline",
        experiment="baseline_swinv2_supervised",
    ),
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

#: Baselines that own their backbone and must not be handed the self-supervised
#: encoder. A SwinV2 state dict would at best be ignored by a ResNet and at worst
#: partially loaded; `swinv2_supervised` is the shape-compatible case and is
#: excluded deliberately, since its whole purpose is to NOT read stage 1.
END_TO_END_BASELINES = {"resnet50", "swin_tiny", "swinv2_supervised"}

#: Learning rates swept per end-to-end baseline. A single shared value cannot be
#: right for both end-to-end ImageNet fine-tuning and frozen-encoder head
#: training, and "our method wins against an under-tuned baseline" is the most
#: common objection any comparison table attracts. Six extra runs remove it.
LR_SWEEP = (1e-5, 3e-5, 1e-4)
SWEEPABLE = {"resnet50", "swin_tiny"}


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
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Training seeds to repeat every baseline over (default: 42-46).",
    )
    parser.add_argument(
        "--lr-sweep",
        action="store_true",
        help="Also sweep {1e-5, 3e-5, 1e-4} for resnet50 and swin_tiny, reporting each one's best. "
        "Removes the under-tuned-baseline objection.",
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help=(
            "Run variants concurrently, one per device: '0,1' or 'auto'. This is the "
            "preferred way to use more than one GPU for stage 2 -- the runs are already "
            "independent processes, so there is no gradient traffic and each variant keeps "
            "the exact numerics of a single-GPU run. Default: one variant at a time on the "
            "default device."
        ),
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
    selected = [VARIANTS_BY_NAME[name] for name in args.models]
    if args.lr_sweep:
        expanded: list[VariantSpec] = []
        for spec in selected:
            if spec.name not in SWEEPABLE:
                expanded.append(spec)
                continue
            for lr in LR_SWEEP:
                expanded.append(
                    VariantSpec(
                        name=f"{spec.name}_lr{lr:g}",
                        description=f"{spec.description} (lr={lr:g})",
                        overrides=[*spec.overrides, f"experiment.training.learning_rate={lr}"],
                        group=spec.group,
                        experiment=spec.experiment,
                    )
                )
        selected = expanded
    specs = expand_seeds(selected, args.seeds)
    extra = [item for item in args.overrides if item != "--"]

    # Only resolve the shared encoder if a selected baseline actually consumes
    # it; requiring it to run ResNet-50 would be a pointless barrier.
    needs_encoder = any(
        not any(spec.name.startswith(name) for name in END_TO_END_BASELINES) for spec in specs
    )
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

    print(
        f"Baseline suite: {len(selected)} models x {len(args.seeds)} seeds "
        f"= {len(specs)} runs -> {root / 'baselines'}"
    )
    if extra:
        print(f"Extra overrides: {' '.join(extra)}")

    def checkpoint_for(spec) -> Path | None:
        """Which encoder checkpoint this baseline should start from, if any.

        An end-to-end baseline builds its own ImageNet backbone, so passing it
        the SwinV2 DINO checkpoint would be meaningless at best and a silent
        partial load at worst. Resolving this per spec -- rather than looping in
        the caller and calling ``run_suite`` once per variant -- is what lets the
        baseline suite be sharded across GPUs like the ablation suite.
        """
        if any(spec.name.startswith(name) for name in END_TO_END_BASELINES):
            return None
        return checkpoint

    results = run_suite(
        specs,
        root=root,
        checkpoint=checkpoint_for,
        extra_overrides=extra,
        dry_run=args.dry_run,
        stop_on_failure=args.stop_on_failure,
        gpus=parse_gpu_list(args.gpus),
    )

    if not args.dry_run:
        manifest = write_suite_manifest(root / "baselines" / "suite_manifest.json", specs, results)
        print(f"Suite manifest: {manifest}")
    return print_summary(results)


if __name__ == "__main__":
    raise SystemExit(main())
