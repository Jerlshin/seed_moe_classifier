"""Stage launcher.

    python main.py pretrain                   # DINO self-supervised pretraining
    python main.py finetune                   # hierarchical MoE finetuning
    python main.py ablation                   # flat-classifier ablation
    python main.py smoke                      # 2-batch dry run of both stages

Anything after the stage name is forwarded verbatim as Hydra overrides:

    python main.py finetune data.batch_size=8 experiment.training.epochs=5
    python main.py finetune model.head.top_k=4        # submitted Top-4 routing
    python main.py finetune model.head.use_moe=false  # single-component ablation

For the full suites, use the dedicated runners instead, which handle output
layout, shared-checkpoint reuse and result aggregation:

    python scripts/run_ablations.py           # six component-wise variants
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
        "overrides",
        nargs=argparse.REMAINDER,
        help="Additional Hydra overrides, for example data.batch_size=4.",
    )
    args = parser.parse_args()

    for command in COMMANDS[args.stage]:
        full_command = [*command, *args.overrides]
        print(f"$ {' '.join(full_command)}", flush=True)
        exit_code = subprocess.call(full_command)
        if exit_code != 0:
            return exit_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
