#!/usr/bin/env python
"""Run the component-wise ablation suite.

    python scripts/run_ablations.py                        # every variant x every seed
    python scripts/run_ablations.py --variants wo_moe wo_kl
    python scripts/run_ablations.py --seeds 42              # single-seed smoke sweep
    python scripts/run_ablations.py --dry-run              # print commands only
    python scripts/run_ablations.py -- data.batch_size=8 experiment.training.epochs=20

Each variant changes the full model along one named axis and trains everything
else identically: same encoder weights, same split, same optimiser, same
schedule. That is what makes the resulting deltas attributable.

**Every variant runs at five seeds.** One run per variant cannot resolve the
table it is being asked to support: on a 1,871-image test split at ~95 %
accuracy the 95 % CI half-width on a *difference* of two accuracies is +-1.40 pp,
before any training-seed variance -- and component contributions of 0.5-2 pp are
the normal magnitude for a 27-class fine-grained task. Runs land in
``outputs/ablations/{variant}/seed{n}/``; ``scripts/generate_plots.py``
aggregates them to mean +- SD and adds McNemar's exact test against
``full_model``, which is valid precisely because every variant shares the
byte-identical test split.

Each run directory is self-contained: Hydra config snapshot, logs, checkpoints,
figures, ``summary.json`` and ``test_predictions.npz``.

**Pretraining is not repeated.** Every variant reads the single published
encoder at ``outputs/checkpoints/dinov2_swinv2_pretrained.pth``. Re-running
self-supervised pretraining per variant would make each row partly a function of
its own initialisation, and would cost many times as much for a worse experiment.
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

#: The component-wise variants, in reporting order.
#
# A named "one-toggle" ablation is only interpretable if it flips one factor. Of
# the six the revision originally asked for, only ``wo_kl`` and
# ``wo_cross_attn`` did:
#
#   wo_moe        routing + active capacity (2 experts -> 1 block) + 2 regularisers
#   wo_arcface    margin + embedding L2-norm + centre L2-norm + logit scale
#   wo_residual   Eq. 9 fusion + the residual magnitude term
#
# The extra controls below split each of those into its true single factors.
# ``full_model`` remains the reference every one is compared against.
ABLATION_VARIANTS: list[VariantSpec] = [
    VariantSpec(
        name="full_model",
        description="Full framework: Top-2 MoE + ArcFace + residual + KL + cross-attention",
        overrides=[],
    ),
    VariantSpec(
        name="wo_moe",
        description="No sparse routing: one dense block (NOT capacity-matched)",
        # Kept as the historical comparison. Note it changes three things at
        # once: routing, active capacity, and the two MoE regularisers, which
        # evaluate to exactly zero on the degenerate one-expert gate -- so this
        # variant optimises a strictly smaller objective than the full model.
        overrides=["model.head.use_moe=false"],
    ),
    VariantSpec(
        name="wo_moe_capacity_matched",
        description="No routing, feed-forward width doubled to match Top-2's active capacity",
        # The full model activates 2 experts per token and a dense block
        # activates 1, so the naive wo_moe gap conflates routing with a 2x cut in
        # active FLOPs -- the very quantity the efficiency section is built on.
        overrides=["model.head.use_moe=false", "model.head.dense_capacity_multiplier=2"],
    ),
    VariantSpec(
        name="moe_fixed_router",
        description="All six experts, routing by a fixed hash instead of a learned gate",
        # The only configuration that can say whether the router learned
        # anything: it holds sparse capacity fixed and removes *learning* from
        # the routing. If this matches full_model, Section 5.2's claim is empty.
        overrides=["model.head.router_mode=hash"],
    ),
    VariantSpec(
        name="moe_uniform_router",
        description="All six experts at weight 1/E: ensembling with no sparsity",
        overrides=["model.head.router_mode=uniform", "model.head.top_k=6"],
    ),
    VariantSpec(
        name="wo_gate_conditioning",
        description="Router sees only z, not the coarse posterior (flat router beside a coarse head)",
        overrides=["model.head.gate_conditioning=false"],
    ),
    VariantSpec(
        name="wo_margin_only",
        description="NormFace: L2-normalised embedding and centres kept, margin removed",
        # The true single-factor margin control.
        overrides=["model.head.sub_head_variant=normface"],
    ),
    VariantSpec(
        name="wo_angular_head",
        description="Plain linear head: removes margin AND normalisation AND logit scale",
        # Honestly labelled: this is the four-factor change the submitted suite
        # called `wo_arcface`.
        overrides=["model.head.sub_head_variant=linear"],
    ),
    VariantSpec(
        name="wo_residual",
        description="No seed-type fusion (Eq. 9); the residual magnitude term goes with it",
        overrides=["model.head.use_residual=false"],
    ),
    VariantSpec(
        name="wo_layer_scale",
        description="Eq. 9 residual kept but ungated: the submitted free additive residual",
        overrides=["model.head.residual_layer_scale=null"],
    ),
    VariantSpec(
        name="film_fusion",
        description="FiLM conditioning on the seed head's hidden state instead of the additive residual",
        overrides=["model.head.fusion_mode=film"],
    ),
    VariantSpec(
        name="wo_kl",
        description="No stage-1/stage-2 KL divergence alignment loss (Eq. 10)",
        overrides=["model.head.use_kl_loss=false"],
    ),
    VariantSpec(
        name="kl_jsd",
        description="Symmetric Jensen-Shannon hierarchy consistency instead of forward KL",
        overrides=["model.loss.kl_mode=jsd"],
    ),
    VariantSpec(
        name="wo_cross_attn",
        description="No Q/K/V refinement: h'' = h' (Eqs. 11-12 skipped)",
        overrides=["model.head.use_cross_attention=false"],
    ),
    VariantSpec(
        name="pooled_tokens",
        description="Submitted pooling: one token before the head, so attention is affine",
        # The direct measurement of what keeping the token grid buys.
        overrides=["model.head.token_mode=pooled"],
    ),
    VariantSpec(
        name="load_entropy",
        description="Submitted soft-gate entropy load balancing instead of the Switch f.P form",
        overrides=["model.loss.moe_load_mode=entropy"],
    ),
    VariantSpec(
        name="wo_stage2_augmentation",
        description="Submitted stage-2 pipeline: deterministic resize, no flip, no crop",
        overrides=[
            "experiment.training.horizontal_flip_prob=0.0",
            "experiment.training.random_resized_crop_scale=null",
        ],
    ),
    VariantSpec(
        name="leakage_ungrouped",
        description="Crop-level splitting: measures how much the source-photograph leak was worth",
        # Not an architecture ablation. The full model under the submitted split
        # protocol, so the delta against full_model quantifies the leakage
        # directly. That delta is a result, not an embarrassment -- reporting it
        # is what turns a fatal reviewer objection into a methods subsection.
        overrides=["experiment.training.split_protocol=stratified"],
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
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Training seeds to repeat every variant over (default: 42-46). "
        "Pass a single seed for a smoke sweep; the table then reports no dispersion.",
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

    specs = expand_seeds([VARIANTS_BY_NAME[name] for name in args.variants], args.seeds)

    print(
        f"Ablation suite: {len(args.variants)} variants x {len(args.seeds)} seeds "
        f"= {len(specs)} runs -> {root / 'ablations'}"
    )
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
        gpus=parse_gpu_list(args.gpus),
    )

    if not args.dry_run:
        manifest = write_suite_manifest(root / "ablations" / "suite_manifest.json", specs, results)
        print(f"Suite manifest: {manifest}")
    return print_summary(results)


if __name__ == "__main__":
    raise SystemExit(main())
