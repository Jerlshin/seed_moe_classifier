#!/usr/bin/env bash
#
# Launch a training stage or a full experiment suite with vast.ai path defaults.
#
#   scripts/train_distributed.sh pretrain
#   scripts/train_distributed.sh finetune data.batch_size=32
#   scripts/train_distributed.sh ablations
#   scripts/train_distributed.sh baselines
#   scripts/train_distributed.sh report
#
# Despite the name this is a single-process launcher; distributed training is not
# implemented. Extra arguments pass through as Hydra overrides.
set -euo pipefail

STAGE="${1:-}"
if [[ -z "${STAGE}" ]]; then
  echo "Usage: scripts/train_distributed.sh pretrain|finetune|ablation|ablations|baselines|report [overrides...]"
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

case "${STAGE}" in
  pretrain)
    python -m src.trainers.contrastive_pretrain experiment=pretrain_swinv2_dino "$@"
    ;;
  finetune)
    python -m src.trainers.moe_finetune experiment=finetune_hierarchical_moe "$@"
    ;;
  ablation)
    python -m src.trainers.moe_finetune experiment=ablation_flat_classifier "$@"
    ;;
  ablations)
    python scripts/run_ablations.py -- "$@"
    ;;
  baselines)
    python scripts/run_baselines.py -- "$@"
    ;;
  report)
    python scripts/generate_plots.py "$@"
    ;;
  *)
    echo "Unknown stage: ${STAGE}"
    exit 2
    ;;
esac
