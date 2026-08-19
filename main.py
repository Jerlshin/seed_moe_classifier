"""Stage launcher.

    python main.py pretrain-v2                # the v2 stage-1 recipe (SwinV2-Tiny)
    python main.py eval-pretrain-v2           # evaluate it, crop-level protocol
    python main.py finetune-v2                # stage 2, crop-level train/val/test

    python main.py pretrain                   # DINO self-supervised pretraining
    python main.py pretrain --gpus 2          # the same, over 2 GPUs with DDP
    python main.py eval-pretrain              # evaluate the stage-1 representation
    python main.py screen-backbones           # Phase 0: initialisation screen, no training
    python main.py eval-frozen                # the frozen-trunk reference stage 1 must beat
    python main.py finetune                   # hierarchical MoE finetuning
    python main.py ablation                   # flat-classifier ablation
    python main.py smoke                      # 2-batch dry run of both stages

Anything after the stage name is forwarded verbatim as Hydra overrides:

    python main.py finetune data.batch_size=8 experiment.training.epochs=5
    python main.py finetune model.head.top_k=4        # submitted Top-4 routing
    python main.py finetune model.head.use_moe=false  # single-component ablation

``--gpus`` delegates to ``scripts/launch.py``, which pins one ``$SEED_RUN_ID`` so
every rank composes the same output directory and then launches the module under
``torch.distributed.run``. ``--gpus 1`` (the default) runs the module directly:
a one-member process group buys nothing and adds a failure mode.

Resuming an interrupted run needs no extra flags -- the same command line does
it, because ``resume=auto`` starts fresh when there is nothing to continue:

    python main.py pretrain --gpus 2 experiment.training.resume=auto \\
        experiment.training.max_runtime_minutes=520

``eval-pretrain`` sits between the two stages and needs no extra flags -- it reads
the milestone encoders the pretraining stage already published:

    python main.py eval-pretrain
    python main.py eval-pretrain experiment.evaluation.split.protocol=stratified

For the full suites, use the dedicated runners instead, which handle output
layout, shared-checkpoint reuse and result aggregation:

    python scripts/run_ablations.py --gpus 0,1  # six variants, sharded by GPU
    python scripts/run_baselines.py           # ResNet-50, Swin-T, hierarchical CCE
    python scripts/generate_plots.py          # figures + summary_metrics.csv
    python scripts/dry_run.py                 # synthetic end-to-end smoke test

Stage-1 arm suites (the STAGE1_CHANGES.md sequence) have their own runner, which
pins per-arm checkpoint, publication and evaluation paths -- without that, arms
silently overwrite each other's encoders and each other's eval output:

    python scripts/run_stage1_ablations.py --arms conf/stage1_arms/phase0.yaml
    python scripts/run_stage1_ablations.py --arms conf/stage1_arms/phase1.yaml
    python scripts/report_raw_photographs.py  # source photographs never cropped

Each stage runs in a subprocess so Hydra owns its own working directory and
logging configuration, exactly as if the module were launched directly.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PRETRAIN = [sys.executable, "-m", "src.trainers.contrastive_pretrain", "experiment=pretrain_swinv2_dino"]
FINETUNE = [sys.executable, "-m", "src.trainers.moe_finetune", "experiment=finetune_hierarchical_moe"]
ABLATION = [sys.executable, "-m", "src.trainers.moe_finetune", "experiment=ablation_flat_classifier"]
# Stage-1 evaluation. Not a training stage: it loads finished encoders and scores
# the representation itself (probe, k-NN, clustering, geometry, invariance,
# prototypes), so it belongs between the two stages rather than after both.
EVAL_PRETRAIN = [
    sys.executable,
    "-m",
    "src.trainers.pretrain_eval",
    "experiment=eval_pretrain_representation",
]
# Phase-0 screens: no training, no checkpoint written, no handoff touched.
#
# `screen-backbones` is the initialisation screen. It is worth running first
# because the stage-1 evaluation's own decomposition is
# `random 0.3804 -> +0.2449 ImageNet-1k -> +0.0031 DINO`: the initialisation was
# never treated as a variable and is worth 79x what the self-distillation run
# bought. It includes the `swinv2_base_window16_256.ms_in1k` control that
# separates *capacity* from the *IN-22k corpus*, which is the row that decides
# the trunk.
#
# `eval-frozen` is the reference every stage-1 arm is measured against: the
# chosen trunk, frozen, with no in-domain training at all.
SCREEN_BACKBONES = [
    sys.executable, "-m", "src.trainers.pretrain_eval", "experiment=screen_backbones",
]
EVAL_FROZEN = [
    sys.executable, "-m", "src.trainers.pretrain_eval", "experiment=eval_frozen_reference",
]

# ---------------------------------------------------------------- the v2 pipeline
#
# The stage-1 redesign: SwinV2-Tiny, view geometry sized for 52 x 51 px crops,
# colour preserved, and an epoch budget chosen by a representation probe rather
# than by a loss curve. `STAGE1_V2.md` is the specification; the config headers
# carry the measurements. The downstream split is crop-level stratified
# throughout, which is a deliberate protocol choice with a measured ~18 pp
# consequence -- `finetune_v2_crop_level`'s header states it.
PRETRAIN_V2 = [
    sys.executable, "-m", "src.trainers.contrastive_pretrain",
    "experiment=pretrain_v2_swinv2_tiny",
]
EVAL_PRETRAIN_V2 = [
    sys.executable, "-m", "src.trainers.pretrain_eval", "experiment=eval_pretrain_v2",
]
FINETUNE_V2 = [
    sys.executable, "-m", "src.trainers.moe_finetune", "experiment=finetune_v2_crop_level",
]

COMMANDS: dict[str, list[list[str]]] = {
    "pretrain": [PRETRAIN],
    "pretrain-v2": [PRETRAIN_V2],
    "eval-pretrain-v2": [EVAL_PRETRAIN_V2],
    "finetune-v2": [FINETUNE_V2],
    "eval-pretrain": [EVAL_PRETRAIN],
    "screen-backbones": [SCREEN_BACKBONES],
    "eval-frozen": [EVAL_FROZEN],
    "finetune": [FINETUNE],
    "ablation": [ABLATION],
}

# A 2-batch, 1-epoch pass through both stages. Enough to catch shape errors,
# config typos and broken logging without needing a GPU. Efficiency profiling is
# off because a latency sweep on two batches measures nothing but takes as long
# as the training it follows.
SMOKE_OVERRIDES = [
    "data.batch_size=2",
    "data.num_workers=0",
    "experiment.training.epochs=1",
    "experiment.training.max_batches=2",
    "tracking.wandb.enabled=false",
]
#: Stages ``scripts/launch.py`` knows how to run under ``torch.distributed.run``.
LAUNCHER_STAGES = ("pretrain", "pretrain-v2", "finetune", "finetune-v2", "ablation")

COMMANDS["smoke"] = [
    [
        *PRETRAIN,
        *SMOKE_OVERRIDES,
        # `effective_batch_size` is the authority, so at `data.batch_size=2` it
        # would otherwise derive 16 accumulation steps for a 2-batch run —
        # harmless, but it makes the smoke test exercise a schedule no real run
        # uses. Pin the effective batch to the micro-batch instead, which keeps
        # accumulation at 1 and the derived learning rate meaningful.
        "experiment.training.effective_batch_size=2",
    ],
    [
        *FINETUNE,
        *SMOKE_OVERRIDES,
        "experiment.training.test_size=0.3",
        "experiment.efficiency.measure_latency=false",
    ],
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run seed MoE training stages.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("stage", choices=sorted(COMMANDS), help="Training stage to run.")
    parser.add_argument(
        "--gpus",
        default="1",
        help=(
            "Processes to launch with DDP: a count, or 'auto' for every visible CUDA "
            "device. Delegates to scripts/launch.py. Not available for 'smoke', which "
            "is a shape check rather than a training run, nor for the evaluation and "
            "screening stages, which are one forward pass per encoder and have no "
            "gradient to reduce."
        ),
    )
    # `parse_known_args`, not a REMAINDER positional: a REMAINDER swallows
    # everything after the stage name, so `main.py pretrain --gpus 2` would
    # forward "--gpus 2" to Hydra as an override and fail there. Hydra overrides
    # are `key=value` tokens that never start with a dash, so the split is
    # unambiguous whichever order the two are written in.
    args, overrides = parser.parse_known_args()
    overrides = [item for item in overrides if item != "--"]

    multi_gpu = args.gpus.strip().lower() not in {"1", ""}
    if multi_gpu:
        if args.stage not in LAUNCHER_STAGES:
            parser.error(
                f"--gpus is not supported for the {args.stage!r} stage; it is a single-process "
                "pass over the dataset, not a training run."
            )
        launcher = str(Path(__file__).resolve().parent / "scripts" / "launch.py")
        command = [sys.executable, launcher, args.stage, "--gpus", args.gpus, *overrides]
        print(f"$ {' '.join(command)}", flush=True)
        return subprocess.call(command)

    for command in COMMANDS[args.stage]:
        full_command = [*command, *overrides]
        print(f"$ {' '.join(full_command)}", flush=True)
        exit_code = subprocess.call(full_command)
        if exit_code != 0:
            return exit_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
