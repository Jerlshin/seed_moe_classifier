import os
import sys
import time
import torch
import hydra
from omegaconf import DictConfig
import torch.optim as optim
from lightly.loss import DINOLoss
from lightly.models.utils import update_momentum

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)

from src.data.transforms import get_dino_transforms
from src.data.dataset import get_pretrain_dataloader
from src.models.backbones.swinv2_dino import DINO
from src.utils.training import (
    ExperimentTracker,
    collect_device_stats,
    log_attention_maps,
    select_device,
    setup_experiment_logger,
    snapshot_run_configuration,
)

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
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
        logger.info("========== Starting DINO Pretraining ==========")
        snapshot_paths = snapshot_run_configuration(cfg, cfg.tracking.output_dir)
        logger.info("Saved run configuration snapshots.", extra={"snapshots": {k: str(v) for k, v in snapshot_paths.items()}})

        device = select_device(cfg.device)
        logger.info("Selected training device: %s", device)
        tracker.log_metrics(collect_device_stats(device), step=0)
        
        logger.info("Initializing data transforms and loader.")
        transform = get_dino_transforms(cfg.data.image_size, cfg.data.local_crop_size)
        dataloader = get_pretrain_dataloader(
            data_dir=cfg.data.root_path,
            transform=transform,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers
        )
        if len(dataloader) == 0:
            raise RuntimeError("Dataloader is empty. Lower batch size or check the dataset path.")

        logger.info("Initializing DINO model with backbone: %s", cfg.experiment.model.backbone_name)
        model = DINO(
            backbone_name=cfg.experiment.model.backbone_name,
            input_dim=cfg.experiment.model.input_dim,
            hidden_dim=cfg.experiment.model.projection_hidden_dim,
            bottleneck_dim=cfg.experiment.model.projection_bottleneck_dim,
            out_dim=cfg.experiment.model.projection_out_dim,
            pretrained=cfg.experiment.model.pretrained,
            dynamic_img_size=cfg.experiment.model.dynamic_img_size,
        ).to(device)
        tracker.log_model_watch(model)

        criterion = DINOLoss(
            output_dim=cfg.experiment.model.projection_out_dim,
            warmup_teacher_temp_epochs=cfg.experiment.training.warmup_teacher_temp_epochs
        ).to(device)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.experiment.training.learning_rate,
            weight_decay=cfg.experiment.training.weight_decay
        )

        os.makedirs(cfg.experiment.training.save_path, exist_ok=True)

        logger.info("Beginning training loop.")
        epochs = cfg.experiment.training.epochs
        max_batches = cfg.experiment.training.max_batches
        momentum = cfg.experiment.training.momentum_teacher
        log_every_steps = int(cfg.tracking.intervals.log_every_steps)
        device_every_steps = int(cfg.tracking.intervals.device_every_steps)
        histogram_every_epochs = int(cfg.tracking.intervals.histogram_every_epochs)
        gradient_histogram_every_epochs = int(cfg.tracking.intervals.gradient_histogram_every_epochs)
        embedding_every_epochs = int(cfg.tracking.intervals.embedding_every_epochs)
        attention_every_epochs = int(cfg.tracking.intervals.attention_every_epochs)
        global_step = 0

        for epoch in range(epochs):
            epoch_started = time.perf_counter()
            total_loss = 0.0
            batches_seen = 0
            model.train()
            logger.info("Epoch %s/%s started.", epoch + 1, epochs)

            for batch_idx, (views, _targets, filenames) in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                update_momentum(model.student_backbone, model.teacher_backbone, m=momentum)
                update_momentum(model.student_head, model.teacher_head, m=momentum)

                views = [view.to(device) for view in views]
                teacher_out = [model.forward_teacher(view) for view in views[:2]]
                student_out = [model.forward_student(view) for view in views]
                loss = criterion(teacher_out, student_out, epoch=epoch)
                total_loss += loss.item()
                batches_seen += 1

                optimizer.zero_grad()
                loss.backward()
                model.student_head.cancel_last_layer_gradients(current_epoch=epoch)

                gradient_norm = None
                if cfg.tracking.artifacts.log_gradient_norms and global_step % log_every_steps == 0:
                    gradient_norm = tracker.log_gradient_norms(model, global_step)

                if (
                    cfg.tracking.artifacts.log_gradient_histograms
                    and gradient_histogram_every_epochs > 0
                    and (epoch + 1) % gradient_histogram_every_epochs == 0
                    and batch_idx == 0
                ):
                    tracker.log_gradient_histograms(model, global_step)

                optimizer.step()

                if global_step % log_every_steps == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    step_metrics = {
                        "loss": loss.item(),
                        "lr": lr,
                    }
                    if gradient_norm is not None:
                        step_metrics["gradient_norm"] = gradient_norm
                    tracker.log_metrics(step_metrics, global_step, prefix="train")
                    logger.info(
                        "Step %s | epoch=%s batch=%s loss=%.5f lr=%.6g",
                        global_step,
                        epoch + 1,
                        batch_idx + 1,
                        loss.item(),
                        lr,
                    )

                if global_step % device_every_steps == 0:
                    tracker.log_metrics(collect_device_stats(device), global_step)

                if (
                    cfg.tracking.artifacts.log_embeddings
                    and embedding_every_epochs > 0
                    and (epoch + 1) % embedding_every_epochs == 0
                    and batch_idx == 0
                ):
                    metadata = [str(item) for item in filenames]
                    tracker.log_embeddings("dino/student_projection", student_out[0], global_step, metadata=metadata)

                if (
                    cfg.tracking.artifacts.log_attention_maps
                    and attention_every_epochs > 0
                    and (epoch + 1) % attention_every_epochs == 0
                    and batch_idx == 0
                ):
                    log_attention_maps(
                        tracker,
                        model.student_backbone,
                        views[0],
                        global_step,
                        logger=logger,
                        max_images=cfg.tracking.artifacts.max_attention_images,
                    )

                global_step += 1

            if batches_seen == 0:
                raise RuntimeError("No batches were processed. Check max_batches and dataloader settings.")

            avg_loss = total_loss / batches_seen
            epoch_seconds = time.perf_counter() - epoch_started
            tracker.log_metrics(
                {
                    "loss": avg_loss,
                    "duration_seconds": epoch_seconds,
                    "batches": batches_seen,
                    "lr": optimizer.param_groups[0]["lr"],
                },
                epoch + 1,
                prefix="epoch",
            )
            logger.info(
                "Epoch %s/%s complete | loss=%.5f batches=%s duration=%.2fs",
                epoch + 1,
                epochs,
                avg_loss,
                batches_seen,
                epoch_seconds,
            )

            if cfg.tracking.artifacts.log_parameter_histograms and histogram_every_epochs > 0 and (epoch + 1) % histogram_every_epochs == 0:
                tracker.log_parameter_histograms(model, global_step)

        save_file = os.path.join(cfg.experiment.training.save_path, "dino_pretrained_backbone.pth")
        torch.save(model.student_backbone.state_dict(), save_file)
        total_seconds = time.perf_counter() - training_started
        tracker.log_event("training_complete", {"duration_seconds": total_seconds, "checkpoint": save_file})
        logger.info("Training complete in %.2fs. Backbone weights saved to %s", total_seconds, save_file)

    except Exception:
        logger.exception("DINO pretraining failed.")
        tracker.log_event("exception", {"stage": "dino_pretraining"})
        raise
    finally:
        tracker.close()

if __name__ == "__main__":
    main()
