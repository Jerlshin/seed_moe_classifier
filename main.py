"""Stage launcher.

    python main.py pretrain                   # DINO self-supervised pretraining
    python main.py pretrain --gpus 2          # the same, over 2 GPUs with DDP
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

For the full suites, use the dedicated runners instead, which handle output
layout, shared-checkpoint reuse and result aggregation:

    python scripts/run_ablations.py --gpus 0,1  # six variants, sharded by GPU
    python scripts/run_baselines.py           # ResNet-50, Swin-T, hierarchical CCE
    python scripts/generate_plots.py          # figures + summary_metrics.csv
    python scripts/dry_run.py                 # synthetic end-to-end smoke test

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

COMMANDS: dict[str, list[list[str]]] = {
    "pretrain": [PRETRAIN],
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
LAUNCHER_STAGES = ("pretrain", "finetune", "ablation")

COMMANDS["smoke"] = [
    [*PRETRAIN, *SMOKE_OVERRIDES],
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
            "is a shape check rather than a training run."
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
                f"--gpus is not supported for the {args.stage!r} stage; it runs two short "
                "single-process passes to check shapes and configuration."
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
