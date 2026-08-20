#!/usr/bin/env python
"""Launch a training stage on one process or many, on any platform.

    python scripts/launch.py pretrain                    # 1 process
    python scripts/launch.py pretrain --gpus 2           # DDP over 2 GPUs
    python scripts/launch.py pretrain --gpus auto        # every visible GPU
    python scripts/launch.py finetune --gpus 2 data.batch_size=8
    python scripts/launch.py pretrain --gpus 2 --dry-run

Anything after the stage name is forwarded verbatim as Hydra overrides.

What this adds over calling ``torchrun`` yourself
-------------------------------------------------

**One output directory for the whole job.** Hydra resolves ``${now:...}`` inside
each process, so two ranks that start in the same second can still land in
different run directories -- and the rank that did would write its share of the
artifacts somewhere nobody looks. This pins ``$SEED_RUN_ID`` once, before the
processes exist, and every rank composes the same ``hydra.run.dir``. (The
trainers additionally broadcast rank 0's resolved directory, so a bare
``torchrun`` is still correct; this just makes it correct *and* tidy.)

**No process group when there is nothing to distribute.** ``--gpus 1`` runs the
module directly rather than under an elastic agent: a one-member process group
buys nothing and adds a failure mode (a stale ``MASTER_PORT`` from a previous
run) to the common path.

**A port that is free.** ``--standalone`` picks one, so two jobs on the same
machine do not collide.

**It works on Windows.** ``torch.distributed.run`` is invoked as a module rather
than relying on a ``torchrun`` console script being on ``PATH``, and the backend
selection in ``src/utils/training/distributed.py`` falls back to Gloo where NCCL
is not built.

On Kaggle
---------

A notebook cell can shell out to this directly::

    !cd /kaggle/working/seed-moe-classifier && \
        python scripts/launch.py pretrain --gpus 2 \
            experiment.training.resume=auto \
            experiment.training.max_runtime_minutes=520

``resume=auto`` plus ``max_runtime_minutes`` is the pair that makes a long
run survive a session limit: the run stops itself with a complete checkpoint
before the platform kills it, and the identical command continues from there.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

#: Stage name -> the module that implements it, plus its experiment config.
STAGES = {
    "pretrain": ("src.trainers.contrastive_pretrain", "experiment=pretrain_dino"),
    "finetune": ("src.trainers.moe_finetune", "experiment=finetune_hierarchical_moe"),
    "finetune-grouped": ("src.trainers.moe_finetune", "experiment=finetune_grouped_diagnostic"),
    "ablation": ("src.trainers.moe_finetune", "experiment=ablation_flat_classifier"),
}


def visible_gpu_count() -> int:
    """How many CUDA devices this process can see. ``0`` when there are none."""
    try:
        import torch

        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:  # pragma: no cover - torch is a hard dependency in practice
        return 0


def resolve_processes(requested: str, logger=print) -> int:
    """Turn ``--gpus`` into a process count, refusing to over-subscribe.

    ``auto`` uses every visible device. An explicit count larger than the number
    of devices is an error rather than a warning: the extra ranks would either
    fail to claim a device or share one, and a job that quietly runs two ranks on
    one GPU is slower than the single-process run it replaced while looking like
    it scaled.
    """
    available = visible_gpu_count()
    if requested.strip().lower() == "auto":
        return max(available, 1)

    count = int(requested)
    if count < 1:
        raise ValueError(f"--gpus must be at least 1, got {count}")
    if count > 1 and available == 0:
        logger(
            f"No CUDA devices are visible, so --gpus {count} would run {count} CPU ranks. "
            "That is supported (Gloo) and useful only for testing the distributed path."
        )
    elif count > available > 0:
        raise ValueError(
            f"--gpus {count} was requested but only {available} CUDA device(s) are visible. "
            "Lower --gpus, or set CUDA_VISIBLE_DEVICES to expose more."
        )
    return count


def build_command(stage: str, processes: int, overrides: list[str]) -> list[str]:
    """The exact argv to execute for ``stage`` at ``processes`` ranks."""
    module, experiment = STAGES[stage]
    if processes <= 1:
        return [sys.executable, "-m", module, experiment, *overrides]
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={processes}",
        "-m",
        module,
        experiment,
        *overrides,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("stage", choices=sorted(STAGES))
    parser.add_argument(
        "--gpus",
        default="1",
        help="Processes to launch: a count, or 'auto' for every visible CUDA device.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Shared output-directory id. Defaults to a timestamp; every rank uses this one.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the command and exit.")
    # See main.py: a REMAINDER positional would swallow this script's own flags
    # when they are written after the stage name, which is where they read
    # naturally. Hydra overrides never start with a dash, so the split is safe.
    args, overrides = parser.parse_known_args()
    overrides = [item for item in overrides if item != "--"]
    processes = resolve_processes(args.gpus)

    environment = dict(os.environ)
    run_id = args.run_id or environment.get("SEED_RUN_ID") or datetime.now().strftime(
        "%Y-%m-%d/%H-%M-%S"
    )
    environment["SEED_RUN_ID"] = run_id
    # Each rank runs its own dataloader workers and its own BLAS pool. Left
    # unset, OpenMP defaults to one thread per core *per process*, so N ranks
    # oversubscribe the machine by N and every rank gets slower.
    environment.setdefault("OMP_NUM_THREADS", str(max(os.cpu_count() // max(processes, 1), 1)))

    command = build_command(args.stage, processes, overrides)
    print(f"SEED_RUN_ID={run_id}")
    print(f"OMP_NUM_THREADS={environment['OMP_NUM_THREADS']}")
    print(f"$ {' '.join(command)}", flush=True)
    if args.dry_run:
        return 0
    return subprocess.call(command, cwd=str(PROJECT_ROOT), env=environment)


if __name__ == "__main__":
    raise SystemExit(main())
