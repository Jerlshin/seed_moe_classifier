"""Stage 1: DINOv2 self-supervised pretraining (paper Section 4, Table 1).

    python main.py pretrain
    python main.py pretrain data.batch_size=2 experiment.training.epochs=1 \
        experiment.training.max_batches=2

Per training step:

1. Build ``2 + local_crops_number`` augmented views of each image.
2. The **teacher** sees only the 2 global crops; the **student** sees all views.
3. The DINO loss scores every cross-view pair (Eq. 1).
4. Backprop into the student only; clip gradients at 3.0 (Table 1); cancel the
   projection head's final-layer gradients during the first epoch.
5. Advance the teacher by EMA at momentum 0.996 (Table 1).

The run ends by writing two files: ``dino_pretrained_final.pth`` (full state) and
``dino_pretrained_backbone.pth`` (a bare ``student_backbone`` state dict). The
latter is the **only** handoff to stage 2.
"""

from __future__ import annotations

import os
import random
import shutil
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.optim as optim
from lightly.models.utils import update_momentum
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.datasets.dataset import get_pretrain_dataloader
from src.datasets.transforms import get_dino_transforms
from src.losses.dino import CustomDINOLoss
from src.models.backbones.swinv2_dino import DINO, build_dino
from src.utils.training import (
    CheckpointManager,
    ExperimentTracker,
    collect_device_stats,
    log_attention_maps,
    select_device,
    setup_experiment_logger,
    snapshot_run_configuration,
    to_cpu_state_dict,
)
from src.utils.visualization import plot_loss_curves


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch, and make cuDNN deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_scheduler(optimizer: optim.Optimizer, cfg: DictConfig):
    """Cosine decay over the full run (paper Section 6.1), or ``None``."""
    name = OmegaConf.select(cfg, "experiment.training.scheduler.name", default=None)
    if name is None:
        return None
    if name == "cosine":
        t_max = OmegaConf.select(cfg, "experiment.training.scheduler.t_max", default=None)
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(t_max or cfg.experiment.training.epochs),
            eta_min=float(OmegaConf.select(cfg, "experiment.training.scheduler.eta_min", default=0.0)),
        )
    raise ValueError(f"Unsupported scheduler: {name}")


def save_dino_checkpoint(
    model: DINO,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    checkpoint_manager: CheckpointManager,
    filename: str,
    include_optimizer: bool = True,
    include_teacher: bool = True,
    rolling_prefix: str | None = None,
) -> str:
    """Save student weights, and optionally the teacher and optimizer state."""
    payload = {
        "epoch": epoch,
        "student_backbone": to_cpu_state_dict(model.student_backbone.state_dict()),
        "student_head": to_cpu_state_dict(model.student_head.state_dict()),
    }
    if include_teacher:
        payload["teacher_backbone"] = to_cpu_state_dict(model.teacher_backbone.state_dict())
        payload["teacher_head"] = to_cpu_state_dict(model.teacher_head.state_dict())
    if include_optimizer:
        payload["optimizer"] = optimizer.state_dict()
        payload["scheduler"] = scheduler.state_dict() if scheduler is not None else None
    return checkpoint_manager.save(filename, payload, rolling_prefix=rolling_prefix)


def publish_shared_backbone(cfg: DictConfig, backbone_file: Path, logger) -> Path | None:
    """Copy the pretrained backbone to the shared path all downstream runs read.

    Configured by ``experiment.training.shared_backbone_path``. Returns the
    destination, or ``None`` when publishing is disabled or fails -- a failure
    here must not discard a completed pretraining run, since the per-stage copy
    at ``backbone_file`` is already safely on disk.
    """
    destination = OmegaConf.select(cfg, "experiment.training.shared_backbone_path", default=None)
    if not destination:
        return None

    target = Path(str(destination))
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(backbone_file, target)
    except OSError as exc:
        logger.warning("Could not publish the shared backbone to %s: %s", target, exc)
        return None

    logger.info("Published shared backbone for downstream runs: %s", target)
    return target


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = setup_experiment_logger(
        log_dir=cfg.tracking.output_dir,
        name="seed_moe.dino_pretrain",
        level=cfg.tracking.log_level,
        console=cfg.tracking.console,
        structured_jsonl=cfg.tracking.structured_jsonl,
    )
    tracker = ExperimentTracker(cfg, logger)
    training_started = time.perf_counter()

    try:
        logger.info("========== DINO pretraining: %s ==========", cfg.experiment.name)
        seed_everything(int(cfg.seed))
        snapshot_paths = snapshot_run_configuration(cfg, cfg.tracking.output_dir)
        logger.info(
            "Saved run configuration snapshots.",
            extra={"snapshots": {key: str(value) for key, value in snapshot_paths.items()}},
        )

        device = select_device(cfg.device)
        logger.info("Selected training device: %s", device)
        tracker.log_metrics(collect_device_stats(device), step=0)

        # ------------------------------------------------------------- data
        transform = get_dino_transforms(
            int(cfg.data.image_size), int(cfg.data.local_crop_size), cfg.data.augmentation
        )
        dataloader = get_pretrain_dataloader(
            data_dir=cfg.data.root_path,
            transform=transform,
            batch_size=int(cfg.data.batch_size),
            num_workers=int(cfg.data.num_workers),
            dataset_format=str(cfg.data.dataset_format),
            image_size=int(cfg.data.image_size),
            pin_memory=bool(cfg.data.pin_memory) and device.type == "cuda",
            drop_last=bool(cfg.data.drop_last),
        )
        if len(dataloader) == 0:
            raise RuntimeError("Dataloader is empty. Lower the batch size or check data.root_path.")
        num_crops = transform.num_crops
        logger.info(
            "Loaded %s batches of %s images, %s views each (2 global + %s local).",
            len(dataloader),
            cfg.data.batch_size,
            num_crops,
            cfg.data.augmentation.local_crops_number,
        )

        # ------------------------------------------------------------ model
        logger.info("Initialising DINO with backbone %s.", cfg.model.backbone.name)
        model = build_dino(
            backbone_cfg=cfg.model.backbone,
            head_cfg=cfg.model.head,
            freeze_last_layer_epochs=int(cfg.experiment.training.freeze_last_layer_epochs),
        ).to(device)
        tracker.log_model_watch(model)
        student_parameters = model.student_parameters()
        tracker.log_metrics(
            {"model/student_parameters": sum(p.numel() for p in student_parameters)}, step=0
        )

        criterion = CustomDINOLoss(
            out_dim=int(cfg.model.head.out_dim),
            num_crops=num_crops,
            warmup_teacher_temp=float(cfg.model.loss.warmup_teacher_temp),
            teacher_temp=float(cfg.model.loss.teacher_temp),
            warmup_teacher_temp_epochs=int(cfg.model.loss.warmup_teacher_temp_epochs),
            num_epochs=int(cfg.experiment.training.epochs),
            student_temp=float(cfg.model.loss.student_temp),
            center_momentum=float(cfg.model.loss.center_momentum),
            num_global_crops=2,
        ).to(device)

        optimizer = optim.AdamW(
            student_parameters,
            lr=float(cfg.experiment.training.learning_rate),
            weight_decay=float(cfg.experiment.training.weight_decay),
        )
        scheduler = build_scheduler(optimizer, cfg)

        save_path = Path(cfg.experiment.training.save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        checkpoint_manager = CheckpointManager(
            save_path, keep_last_n=int(cfg.experiment.training.keep_last_n_checkpoints)
        )

        # ------------------------------------------------------- loop config
        epochs = int(cfg.experiment.training.epochs)
        max_batches = OmegaConf.select(cfg, "experiment.training.max_batches", default=None)
        momentum = float(cfg.experiment.training.momentum_teacher)
        clip_grad = cfg.experiment.training.clip_grad
        save_interval = int(cfg.experiment.training.save_interval)
        save_full_checkpoints = bool(cfg.experiment.training.save_full_checkpoints)
        save_teacher = bool(cfg.experiment.training.save_teacher_in_checkpoints)

        intervals = cfg.tracking.intervals
        artifacts = cfg.tracking.artifacts
        log_every_steps = int(intervals.log_every_steps)
        device_every_steps = int(intervals.device_every_steps)
        global_step = 0
        loss_history: list[float] = []

        logger.info("Training for %s epochs at lr=%s, teacher momentum=%s.",
                    epochs, cfg.experiment.training.learning_rate, momentum)

        for epoch in range(epochs):
            epoch_started = time.perf_counter()
            total_loss = 0.0
            batches_seen = 0
            model.train()

            for batch_idx, (_originals, views, _targets, filenames) in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                views = [view.to(device, non_blocking=True) for view in views]

                # Teacher sees only the two global crops (paper Fig. 7).
                with torch.no_grad():
                    teacher_out = [model.forward_teacher(view) for view in views[:2]]
                student_out = [model.forward_student(view) for view in views]
                loss = criterion(student_out, teacher_out, epoch=epoch)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                gradient_norm = None
                if artifacts.log_gradient_norms and global_step % log_every_steps == 0:
                    gradient_norm = tracker.log_gradient_norms(model, global_step)

                clipped_norm = None
                if clip_grad is not None and float(clip_grad) > 0:
                    clipped_norm = torch.nn.utils.clip_grad_norm_(
                        student_parameters, max_norm=float(clip_grad)
                    )

                # Section 6.1: last layer frozen for the first epoch.
                model.student_head.cancel_last_layer_gradients(current_epoch=epoch)

                if (
                    artifacts.log_gradient_histograms
                    and int(intervals.gradient_histogram_every_epochs) > 0
                    and (epoch + 1) % int(intervals.gradient_histogram_every_epochs) == 0
                    and batch_idx == 0
                ):
                    tracker.log_gradient_histograms(model, global_step)

                optimizer.step()

                # EMA teacher update (Table 1: momentum 0.996).
                update_momentum(model.student_backbone, model.teacher_backbone, m=momentum)
                update_momentum(model.student_head, model.teacher_head, m=momentum)

                total_loss += float(loss.detach())
                batches_seen += 1

                if global_step % log_every_steps == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    step_metrics = {
                        "loss": float(loss.detach()),
                        "lr": lr,
                        "teacher_temp": criterion.teacher_temperature(epoch),
                    }
                    if gradient_norm is not None:
                        step_metrics["gradient_norm"] = gradient_norm
                    if clipped_norm is not None:
                        step_metrics["clipped_gradient_norm"] = float(clipped_norm)
                    tracker.log_metrics(step_metrics, global_step, prefix="train")
                    logger.info(
                        "Step %s | epoch=%s batch=%s loss=%.5f lr=%.6g tau_t=%.4f",
                        global_step, epoch + 1, batch_idx + 1, float(loss.detach()),
                        lr, criterion.teacher_temperature(epoch),
                    )

                if global_step % device_every_steps == 0:
                    tracker.log_metrics(collect_device_stats(device), global_step)

                if (
                    artifacts.log_embeddings
                    and int(intervals.embedding_every_epochs) > 0
                    and (epoch + 1) % int(intervals.embedding_every_epochs) == 0
                    and batch_idx == 0
                ):
                    tracker.log_embeddings(
                        "dino/student_projection",
                        student_out[0],
                        global_step,
                        metadata=[str(item) for item in filenames],
                    )

                if (
                    artifacts.log_attention_maps
                    and int(intervals.attention_every_epochs) > 0
                    and (epoch + 1) % int(intervals.attention_every_epochs) == 0
                    and batch_idx == 0
                ):
                    log_attention_maps(
                        tracker, model.student_backbone, views[0], global_step,
                        logger=logger, max_images=int(artifacts.max_attention_images),
                    )

                global_step += 1

            if batches_seen == 0:
                raise RuntimeError("No batches were processed. Check max_batches and the dataloader.")

            average_loss = total_loss / batches_seen
            loss_history.append(average_loss)
            epoch_seconds = time.perf_counter() - epoch_started

            if scheduler is not None:
                scheduler.step()
            new_lr = optimizer.param_groups[0]["lr"]

            tracker.log_metrics(
                {
                    "loss": average_loss,
                    "duration_seconds": epoch_seconds,
                    "batches": batches_seen,
                    "lr": new_lr,
                    "teacher_temp": criterion.teacher_temperature(epoch),
                },
                epoch + 1,
                prefix="epoch",
            )
            logger.info(
                "Epoch %s/%s | loss=%.5f batches=%s duration=%.2fs",
                epoch + 1, epochs, average_loss, batches_seen, epoch_seconds,
            )

            if (
                artifacts.log_parameter_histograms
                and int(intervals.histogram_every_epochs) > 0
                and (epoch + 1) % int(intervals.histogram_every_epochs) == 0
            ):
                tracker.log_parameter_histograms(model, global_step)

            figure_every = int(OmegaConf.select(cfg, "tracking.intervals.figure_every_epochs", default=5))
            if figure_every > 0 and (epoch + 1) % figure_every == 0 and len(loss_history) > 1:
                tracker.log_figure(
                    "pretrain/loss_curve",
                    plot_loss_curves({"DINO loss": loss_history}, title="DINO pretraining loss"),
                    epoch + 1,
                )

            if save_interval > 0 and (epoch + 1) % save_interval == 0:
                checkpoint_file = save_dino_checkpoint(
                    model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch + 1,
                    checkpoint_manager=checkpoint_manager,
                    filename=f"dino_checkpoint_epoch_{epoch + 1:04d}.pth",
                    include_optimizer=save_full_checkpoints,
                    include_teacher=save_teacher,
                    rolling_prefix="dino_checkpoint_epoch_",
                )
                tracker.log_event("checkpoint", {"epoch": epoch + 1, "path": checkpoint_file})
                logger.info("Saved interval checkpoint: %s", checkpoint_file)

        # ------------------------------------------------------- final saves
        final_file = save_dino_checkpoint(
            model=model, optimizer=optimizer, scheduler=scheduler, epoch=epochs,
            checkpoint_manager=checkpoint_manager, filename="dino_pretrained_final.pth",
            include_optimizer=save_full_checkpoints, include_teacher=save_teacher,
        )
        # The bare backbone state dict is the handoff to stage 2.
        backbone_file = save_path / "dino_pretrained_backbone.pth"
        torch.save(to_cpu_state_dict(model.student_backbone.state_dict()), backbone_file)

        # Publish the same weights at the shared, stage-independent path that
        # every downstream run reads. The ablation and baseline suites compare
        # architectures, so they must all start from *one* set of encoder
        # weights; a per-run pretraining stage would make each variant's result
        # partly a function of its own self-supervised seed.
        shared_file = publish_shared_backbone(cfg, backbone_file, logger)
        if shared_file is not None:
            tracker.log_event("shared_backbone", {"path": str(shared_file)})

        if loss_history:
            tracker.log_figure(
                "pretrain/loss_curve",
                plot_loss_curves({"DINO loss": loss_history}, title="DINO pretraining loss"),
                epochs,
            )
        tracker.log_artifact(backbone_file, name="dino_pretrained_backbone", artifact_type="model")

        total_seconds = time.perf_counter() - training_started
        tracker.log_event(
            "training_complete",
            {
                "duration_seconds": total_seconds,
                "checkpoint": final_file,
                "student_backbone": str(backbone_file),
            },
        )
        logger.info(
            "Pretraining complete in %.2fs. Final: %s. Backbone for stage 2: %s",
            total_seconds, final_file, backbone_file,
        )

    except Exception:
        logger.exception("DINO pretraining failed.")
        tracker.log_event("exception", {"stage": "dino_pretraining"})
        raise
    finally:
        tracker.log_event("training_end", {"duration_seconds": time.perf_counter() - training_started})
        tracker.close()


if __name__ == "__main__":
    main()
