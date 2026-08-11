#!/usr/bin/env bash
#
# Launch a training stage or a full experiment suite, single- or multi-GPU,
# with server path defaults.
#
#   scripts/train_distributed.sh pretrain
#   GPUS=2 scripts/train_distributed.sh pretrain           # DDP over 2 GPUs
#   GPUS=auto scripts/train_distributed.sh pretrain        # every visible GPU
#   scripts/train_distributed.sh finetune data.batch_size=32
#   GPUS=0,1 scripts/train_distributed.sh ablations        # one variant per GPU
#   scripts/train_distributed.sh baselines
#   scripts/train_distributed.sh verify                    # numerical checks
#   scripts/train_distributed.sh report
#
# Extra arguments pass through as Hydra overrides.
#
# GPUS means two different things, deliberately, because the two stages want
# different kinds of parallelism:
#
#   * for `pretrain` / `finetune` / `ablation` it is a COUNT, and the stage runs
#     as one DDP job across that many ranks;
#   * for `ablations` / `baselines` it is a DEVICE LIST, and the suite runs one
#     variant per device concurrently. That is the better use of a second GPU
#     for stage 2: 18 variants x 5 seeds are already independent processes, so
#     there is no gradient traffic and each variant keeps the exact numerics of
#     a single-GPU run.
#
# On Windows, or anywhere without bash, use scripts/launch.py directly -- it is
# the same launcher and takes the same arguments.
set -euo pipefail

STAGE="${1:-}"
if [[ -z "${STAGE}" ]]; then
  echo "Usage: [GPUS=n] scripts/train_distributed.sh pretrain|finetune|ablation|ablations|baselines|verify|report [overrides...]"
  exit 2
fi
shift || true

export SEED_DATA_ROOT="${SEED_DATA_ROOT:-/workspace/data/Hierarchical_SeedData/Cropped_Samples}"
export SEED_OUTPUT_DIR="${SEED_OUTPUT_DIR:-/workspace/outputs}"
mkdir -p "${SEED_OUTPUT_DIR}"

# The single encoder every downstream run reuses. Published by the pretrain
# stage; exported here so the finetune, ablation and baseline stages all resolve
# to the same file without any manual path plumbing.
export SEED_PRETRAIN_BACKBONE="${SEED_PRETRAIN_BACKBONE:-${SEED_OUTPUT_DIR}/checkpoints/dinov2_swinv2_pretrained.pth}"

GPUS="${GPUS:-1}"

case "${STAGE}" in
  pretrain|finetune|ablation)
    # scripts/launch.py pins one $SEED_RUN_ID so every rank composes the same
    # Hydra output directory, and runs the module directly (no process group)
    # when GPUS is 1.
    python scripts/launch.py "${STAGE}" --gpus "${GPUS}" "$@"
    ;;
  ablations)
    if [[ "${GPUS}" == "1" ]]; then
      python scripts/run_ablations.py -- "$@"
    else
      python scripts/run_ablations.py --gpus "${GPUS}" -- "$@"
    fi
    ;;
  baselines)
    if [[ "${GPUS}" == "1" ]]; then
      python scripts/run_baselines.py -- "$@"
    else
      python scripts/run_baselines.py --gpus "${GPUS}" -- "$@"
    fi
    ;;
  verify)
    python scripts/verify_runtime.py "$@"
    ;;
  report)
    python scripts/generate_plots.py "$@"
    ;;
  *)
    echo "Unknown stage: ${STAGE}"
    exit 2
    ;;
esac
