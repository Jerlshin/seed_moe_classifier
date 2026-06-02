import os
import sys
import time

import hydra
from omegaconf import DictConfig

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.training import (
    ExperimentTracker,
    collect_device_stats,
    select_device,
    setup_experiment_logger,
    snapshot_run_configuration,
)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    logger = setup_experiment_logger(
        log_dir=cfg.tracking.output_dir,
        name="seed_moe.moe_finetune",
        level=cfg.tracking.log_level,
        console=cfg.tracking.console,
        structured_jsonl=cfg.tracking.structured_jsonl,
    )
    tracker = ExperimentTracker(cfg, logger)
    started = time.perf_counter()

    try:
        logger.info("========== Starting Hierarchical MoE Finetuning ==========")
        snapshot_paths = snapshot_run_configuration(cfg, cfg.tracking.output_dir)
        logger.info(
            "Saved run configuration snapshots.",
            extra={"snapshots": {key: str(value) for key, value in snapshot_paths.items()}},
        )

        device = select_device(cfg.device)
        logger.info("Selected training device: %s", device)
        tracker.log_metrics(collect_device_stats(device), step=0)

        logger.info(
            "Finetune instrumentation is ready. Model/data training code is not implemented yet."
        )
        raise NotImplementedError(
            "src/train/moe_finetune.py has logging/tracking scaffolding, but the "
            "Hierarchical MoE model, supervised datasets, and train/validation loops "
            "are not implemented in this repository yet."
        )

    except Exception:
        logger.exception("Hierarchical MoE finetuning failed.")
        tracker.log_event("exception", {"stage": "moe_finetuning"})
        raise
    finally:
        tracker.log_event("training_end", {"duration_seconds": time.perf_counter() - started})
        tracker.close()


if __name__ == "__main__":
    main()
