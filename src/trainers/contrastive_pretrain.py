"""Stage 1: DINO-style self-supervised pretraining (paper Section 4, Table 1).

DINO (Caron et al., 2021) with a SwinV2 trunk, plus the two DINOv2 components
that do not require patch tokens -- KoLeo and Sinkhorn-Knopp centering. It is
**not** DINOv2: there is no iBOT patch objective and no untied heads. See
``src/losses/dino.py``.

    python main.py pretrain
    python main.py pretrain data.batch_size=2 experiment.training.epochs=1 \
        experiment.training.max_batches=2

Per training step:

1. Build ``2 + local_crops_number`` augmented views of each image.
2. The **teacher** sees only the 2 global crops; the **student** sees all views.
3. The DINO loss scores every cross-view pair (Eq. 1).
4. Backprop into the student only; clip gradients at 3.0 (Table 1); cancel the
   projection head's final-layer gradients during the first epoch.
5. Advance the teacher by EMA at a cosine-scheduled momentum (0.996 -> 1.0).

Gradients accumulate over ``gradient_accumulation_steps`` micro-batches before
stepping, because every collapse guard in DINO is a batch statistic and
``batch_size=16`` is far below the regime they were designed for.

The run ends by writing two files: ``dino_pretrained_final.pth`` (full state) and
``dino_pretrained_backbone.pth`` (a bare ``student_backbone`` state dict). The
latter is the **only** handoff to stage 2.

How a step is executed
======================

The arithmetic above is the paper's. What follows is how it is scheduled, and it
is the difference between a run that finishes and one that does not: at 6 views
per image and a 256 px window, a naive implementation of this loop spends most
of its wall clock not computing.

**One backbone call, not six.** All ``6B`` student views go through
``forward_student_views`` as a single stacked tensor, and the teacher's ``2B``
globals as another. A SwinV2 block at batch 16 does not fill a GPU -- its kernels
are launch-bound -- and the per-view loop paid that overhead six times for the
same total arithmetic. Peak activation memory is unchanged, because the loop
already kept all six autograd graphs alive until backward.

**Local crops cross the PCIe bus at their own size.** ``resize_local_to_global``
upsamples 101 px crops to 256 px so SwinV2's fixed windows accept them. Doing
that on the dataloader worker means four of every six views are collated,
pinned and copied at **6.4x their information content**; doing it with one
``F.interpolate`` after the copy is the same function (bicubic, applied after
normalisation, exactly as the CPU pipeline ordered it) for a fraction of the
CPU and bus cost. Views also arrive as ``uint8`` and are normalised here, which
is another 4x off all three.

**The un-augmented view is not built.** ``_originals`` was constructed for every
sample of every epoch -- a resize, a float conversion, a collate and 768 KB of
transfer each -- and then dropped on the floor by this loop.

**Nothing synchronises inside the step.** Every ``float(tensor)`` blocks the CPU
until the queued backward drains, so the next batch cannot start being enqueued.
The loss accumulator stays a device tensor, the loss diagnostics stay device
tensors (``CustomDINOLoss.metrics_enabled``), and the whole lot is converted once
per logging interval. ``data_wait_fraction`` in the step log is the direct
measurement of whether any of this is working: it is the share of wall clock the
loop spent blocked in the dataloader, and if it is high the bottleneck is the
CPU pipeline, not the GPU.

**The EMA is two kernels, not ~880.** See
:class:`~src.utils.training.ema.TeacherEmaUpdater`.

Precision and compilation are configured under ``experiment.training`` -- see
``conf/experiment/pretrain_swinv2_dino.yaml``. ``amp: auto`` selects bf16 on
Ampere and later, fp16 with a ``GradScaler`` on older CUDA cards, and off on
CPU/MPS. The Sinkhorn normaliser, the prototype log-softmax and the KoLeo
distances are pinned to fp32 inside the autocast region by ``src/losses/dino.py``;
those three are the parts of this objective that fp16 cannot hold.
"""

from __future__ import annotations

import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.datasets.dataset import MultiCropBatch, get_pretrain_dataloader
from src.datasets.transforms import get_dino_transforms
from src.losses.dino import CustomDINOLoss
from src.models.backbones.swinv2_dino import DINO, build_dino
from src.utils.training import (
    CheckpointManager,
    ExperimentTracker,
    TeacherEmaUpdater,
    autocast_context,
    build_grad_scaler,
    collect_device_stats,
    configure_backend,
    log_attention_maps,
    resolve_amp,
    select_device,
    setup_experiment_logger,
    snapshot_run_configuration,
    to_cpu_state_dict,
)
from src.utils.visualization import plot_loss_curves


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch.

    cuDNN's determinism and autotuning flags are **not** set here -- they belong
    to :func:`~src.utils.training.device.configure_backend`, which the caller
    invokes with the run's ``deterministic`` setting. Stage 1 produces a single
    checkpoint rather than a row in a comparison table, so it defaults to
    autotuning; stage 2, where variants must be comparable, does not.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_value(start: float, end: float, step: int, total: int) -> float:
    """Cosine interpolation from ``start`` to ``end`` over ``total`` steps.

    DINO schedules both the teacher momentum (0.996 -> 1.0) and the weight decay
    (0.04 -> 0.4) this way. The point of the momentum schedule is a fast-adapting
    teacher early and a stable target late; a constant value gives neither end of
    that trade-off, which is what the submitted configuration did.
    """
    if total <= 1 or start == end:
        return float(end if total <= 1 else start)
    progress = min(max(step, 0), total) / total
    return float(end + (start - end) * (1.0 + math.cos(math.pi * progress)) / 2.0)


def build_optimizer(parameters, cfg: DictConfig, device: torch.device, logger) -> optim.Optimizer:
    """AdamW over the student's parameters (Section 6.1), fused where available.

    The fused implementation runs the whole update as a handful of multi-tensor
    kernels instead of ~440 small ones. That matters more here than the FLOP
    count suggests: with ``gradient_accumulation_steps: 4`` the optimizer fires
    once per four micro-batches, and the update's cost is almost entirely kernel
    launches on tensors far too small to hide them.

    Falls back to the reference implementation if the build does not support it,
    which is a performance difference and never a numerical one.
    """
    kwargs = {
        "lr": float(cfg.experiment.training.learning_rate),
        "weight_decay": float(cfg.experiment.training.weight_decay),
    }
    if device.type == "cuda" and bool(
        OmegaConf.select(cfg, "experiment.training.fused_optimizer", default=True)
    ):
        try:
            return optim.AdamW(parameters, fused=True, **kwargs)
        except (RuntimeError, TypeError, ValueError) as exc:
            logger.warning("Fused AdamW unavailable (%s); using the default implementation.", exc)
    return optim.AdamW(parameters, **kwargs)


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


# --------------------------------------------------------------- view assembly


class ViewBatcher:
    """Turn a :class:`~src.datasets.dataset.MultiCropBatch` into model inputs.

    Owns the two pieces of per-view work that were moved off the dataloader:
    normalisation of ``uint8`` views, and the local-crop upsample. Both are
    applied here in the same order the CPU pipeline used --
    ``/255 -> (x - mean) / std -> bicubic resize`` -- so the tensors the backbone
    sees are the same function of the augmented crop either way.

    The order matters and is not interchangeable. Normalisation is affine and
    bicubic interpolation is a partition-of-unity kernel, so the two commute in
    exact arithmetic; but resizing *first* would mean resizing ``uint8``, which
    clamps the bicubic overshoot into [0, 255] and quantises it. That is a
    different transform, and it is why ``output_uint8`` forces
    ``defer_local_upsample`` in ``src/datasets/transforms.py``.

    ``antialias=True`` on the interpolate call is equally load-bearing, and it is
    the trap in this refactor. ``torchvision.transforms.Resize`` defaults to
    ``antialias=True``, and -- contrary to the usual intuition that antialiasing
    only matters when *down*sampling -- torch's antialiased bicubic kernel is not
    the same kernel as the plain one on the way up either. Dropping the flag
    reproduces the CPU pipeline to within ~0.2 in normalised units, which is far
    too small to look like a bug and easily large enough to change what the model
    learns from a 101 px crop. With the flag, the two paths agree bitwise.

    Args:
        image_size: Resolution the backbone requires (256 for
            ``swinv2_*_window16_256``).
        mean / std: Channel statistics, held as ``[1, 3, 1, 1]`` device tensors
            so normalisation is two fused broadcasts rather than a per-channel
            loop.
        device: Where the model lives.
    """

    def __init__(
        self,
        image_size: int,
        mean: tuple[float, ...],
        std: tuple[float, ...],
        device: torch.device,
    ):
        self.image_size = int(image_size)
        self.device = device
        self.mean = torch.tensor(mean, device=device, dtype=torch.float32).view(1, -1, 1, 1)
        self.std = torch.tensor(std, device=device, dtype=torch.float32).view(1, -1, 1, 1)

    def _to_device(self, views: torch.Tensor) -> torch.Tensor:
        """``[V, B, C, H, W]`` -> ``[V * B, C, H, W]`` float, on device.

        ``flatten(0, 1)`` on the collated tensor is a view, not a copy, and it
        preserves the **view-major** block order the loss chunks on.
        """
        views = views.to(self.device, non_blocking=True)
        views = views.flatten(0, 1)
        if views.dtype == torch.uint8:
            views = views.float().div_(255.0).sub_(self.mean).div_(self.std)
        return views

    def __call__(self, batch: MultiCropBatch) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return ``(student_views, teacher_views, batch_size)``.

        ``teacher_views`` is the ``[2B, ...]`` global block and is the same
        storage as the first ``2B`` rows of ``student_views``, so the globals are
        normalised and transferred once.
        """
        batch_size = batch.global_views.shape[1]
        globals_ = self._to_device(batch.global_views)

        if batch.local_views is None:
            return globals_, globals_, batch_size

        locals_ = self._to_device(batch.local_views)
        if locals_.shape[-1] != self.image_size or locals_.shape[-2] != self.image_size:
            # `antialias=True` is what makes this identical to the
            # `T.Resize(..., BICUBIC)` it replaces -- see the class docstring.
            locals_ = F.interpolate(
                locals_,
                size=(self.image_size, self.image_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        student_views = torch.cat([globals_, locals_], dim=0)
        return student_views, globals_, batch_size


# ------------------------------------------------------------------ checkpoints


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
    """Save student weights, and optionally the teacher and optimizer state.

    Reads the *uncompiled* modules deliberately: ``DINO`` keeps any compiled
    callables off the module tree precisely so these keys stay free of the
    ``_orig_mod.`` prefix ``torch.compile`` would otherwise introduce.
    """
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


# ------------------------------------------------------------------- main loop


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
        backend = configure_backend(
            device,
            allow_tf32=bool(OmegaConf.select(cfg, "experiment.training.allow_tf32", default=True)),
            cudnn_benchmark=bool(
                OmegaConf.select(cfg, "experiment.training.cudnn_benchmark", default=True)
            ),
            deterministic=bool(
                OmegaConf.select(cfg, "experiment.training.deterministic", default=False)
            ),
            matmul_precision=str(
                OmegaConf.select(cfg, "experiment.training.matmul_precision", default="high")
            ),
            logger=logger,
        )
        tracker.log_event("backend", backend)
        logger.info("Selected training device: %s", device)
        tracker.log_metrics(collect_device_stats(device), step=0)

        amp = resolve_amp(device, OmegaConf.select(cfg, "experiment.training.amp", default="auto"))
        scaler = build_grad_scaler(amp)
        logger.info(
            "Mixed precision: %s (grad scaler: %s). bf16 needs no loss scaling; fp16 does.",
            amp.label,
            "on" if scaler is not None else "off",
        )

        # ------------------------------------------------------------- data
        transform = get_dino_transforms(
            int(cfg.data.image_size),
            int(cfg.data.local_crop_size),
            cfg.data.augmentation,
            # This loop never reads the un-augmented view; building it would cost
            # a resize and a 256x256x3 float per sample per epoch for nothing.
            return_original=False,
        )
        loader_generator = torch.Generator()
        loader_generator.manual_seed(int(cfg.seed))
        dataloader = get_pretrain_dataloader(
            data_dir=cfg.data.root_path,
            transform=transform,
            batch_size=int(cfg.data.batch_size),
            num_workers=int(cfg.data.num_workers),
            dataset_format=str(cfg.data.dataset_format),
            image_size=int(cfg.data.image_size),
            pin_memory=bool(cfg.data.pin_memory) and device.type == "cuda",
            drop_last=bool(cfg.data.drop_last),
            persistent_workers=bool(
                OmegaConf.select(cfg, "data.persistent_workers", default=True)
            ),
            prefetch_factor=int(OmegaConf.select(cfg, "data.prefetch_factor", default=4)),
            cache_images=bool(OmegaConf.select(cfg, "data.cache_images", default=True)),
            cache_limit_mb=float(OmegaConf.select(cfg, "data.cache_limit_mb", default=4096)),
            generator=loader_generator,
            logger=logger,
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
        logger.info(
            "View geometry | emitted sizes=%s dtype=%s local upsample on %s",
            transform.view_sizes,
            "uint8" if transform.output_uint8 else "float32",
            "device" if transform.upsample_locals_on_device else "CPU",
        )
        batcher = ViewBatcher(
            image_size=int(cfg.data.image_size),
            mean=transform.normalize_mean,
            std=transform.normalize_std,
            device=device,
        )

        # ------------------------------------------------------------ model
        logger.info("Initialising DINO with backbone %s.", cfg.model.backbone.name)
        model = build_dino(
            backbone_cfg=cfg.model.backbone,
            head_cfg=cfg.model.head,
            freeze_last_layer_epochs=int(cfg.experiment.training.freeze_last_layer_epochs),
        ).to(device)
        runtime = model.configure_runtime(
            compile_enabled=bool(
                OmegaConf.select(cfg, "experiment.training.compile.enabled", default=False)
            ),
            compile_mode=str(
                OmegaConf.select(cfg, "experiment.training.compile.mode", default="default")
            ),
            grad_checkpointing=bool(
                OmegaConf.select(cfg, "experiment.training.grad_checkpointing", default=False)
            ),
            channels_last=bool(
                OmegaConf.select(cfg, "experiment.training.channels_last", default=False)
            ),
            logger=logger,
        )
        tracker.log_event("model_runtime", runtime)
        forward_chunk_size = OmegaConf.select(
            cfg, "experiment.training.forward_chunk_size", default=None
        )
        forward_chunk_size = int(forward_chunk_size) if forward_chunk_size else None

        tracker.log_model_watch(model)
        student_parameters = model.student_parameters()
        tracker.log_metrics(
            {"model/student_parameters": sum(p.numel() for p in student_parameters)}, step=0
        )
        ema = TeacherEmaUpdater(model.ema_pairs())
        logger.info("EMA covers %s teacher tensors, updated with fused foreach kernels.", len(ema))

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
            centering=str(OmegaConf.select(cfg, "model.loss.centering", default="sinkhorn")),
            sinkhorn_iterations=int(
                OmegaConf.select(cfg, "model.loss.sinkhorn_iterations", default=3)
            ),
            lambda_koleo=float(OmegaConf.select(cfg, "model.loss.lambda_koleo", default=0.0)),
        ).to(device)
        logger.info(
            "Stage 1 objective: DINO self-distillation | centering=%s koleo=%s prototypes=%s",
            criterion.centering,
            criterion.lambda_koleo,
            int(cfg.model.head.out_dim),
        )

        optimizer = build_optimizer(student_parameters, cfg, device, logger)
        scheduler = build_scheduler(optimizer, cfg)

        save_path = Path(cfg.experiment.training.save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        checkpoint_manager = CheckpointManager(
            save_path, keep_last_n=int(cfg.experiment.training.keep_last_n_checkpoints)
        )

        # ------------------------------------------------------- loop config
        epochs = int(cfg.experiment.training.epochs)
        max_batches = OmegaConf.select(cfg, "experiment.training.max_batches", default=None)
        momentum_start = float(cfg.experiment.training.momentum_teacher)
        momentum_final = OmegaConf.select(
            cfg, "experiment.training.momentum_teacher_final", default=None
        )
        weight_decay_start = float(cfg.experiment.training.weight_decay)
        weight_decay_final = OmegaConf.select(
            cfg, "experiment.training.weight_decay_final", default=None
        )
        accumulation_steps = max(
            int(OmegaConf.select(cfg, "experiment.training.gradient_accumulation_steps", default=1)), 1
        )
        # Every collapse guard in DINO is a batch statistic, so the number that
        # matters is the effective batch, not `data.batch_size`.
        logger.info(
            "Effective batch: %s x %s accumulation = %s images (%s teacher views/step, "
            "%s student views/step)",
            int(cfg.data.batch_size),
            accumulation_steps,
            int(cfg.data.batch_size) * accumulation_steps,
            int(cfg.data.batch_size) * accumulation_steps * 2,
            int(cfg.data.batch_size) * accumulation_steps * num_crops,
        )
        clip_grad = cfg.experiment.training.clip_grad
        save_interval = int(cfg.experiment.training.save_interval)
        save_full_checkpoints = bool(cfg.experiment.training.save_full_checkpoints)
        save_teacher = bool(cfg.experiment.training.save_teacher_in_checkpoints)

        intervals = cfg.tracking.intervals
        artifacts = cfg.tracking.artifacts
        log_every_steps = int(intervals.log_every_steps)
        device_every_steps = int(intervals.device_every_steps)
        # Per-parameter gradient norms cost one host synchronisation *per
        # tensor* -- ~440 of them for this student. The clipped total norm is
        # free (it is computed anyway), so the per-tensor breakdown runs on its
        # own, much rarer, interval.
        gradient_norm_every_steps = int(
            OmegaConf.select(cfg, "tracking.intervals.gradient_norm_every_steps", default=200)
        )
        global_step = 0
        loss_history: list[float] = []

        momentum = momentum_start
        logger.info(
            "Training for %s epochs at lr=%s, teacher momentum=%s -> %s.",
            epochs,
            cfg.experiment.training.learning_rate,
            momentum_start,
            momentum_final if momentum_final is not None else "(constant)",
        )

        for epoch in range(epochs):
            epoch_started = time.perf_counter()
            # Accumulated on the device. Summing `float(loss)` here would put a
            # full pipeline stall in every micro-batch.
            total_loss = torch.zeros((), device=device, dtype=torch.float32)
            batches_seen = 0
            data_wait = 0.0
            model.train()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()

            momentum = (
                cosine_value(momentum_start, float(momentum_final), epoch, epochs)
                if momentum_final is not None
                else momentum_start
            )

            optimizer.zero_grad(set_to_none=True)
            micro_in_window = 0
            wait_started = time.perf_counter()

            for batch_idx, batch in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                data_wait += time.perf_counter() - wait_started

                is_last_batch = (
                    batch_idx + 1 == len(dataloader)
                    or (max_batches is not None and batch_idx + 1 >= max_batches)
                )
                is_step = (micro_in_window + 1) % accumulation_steps == 0 or is_last_batch
                will_log = is_step and (global_step % log_every_steps == 0)
                # The collapse diagnostics are only ever read on logging steps,
                # so on every other step they are not computed at all.
                criterion.metrics_enabled = will_log

                student_views, teacher_views, batch_size = batcher(batch)

                with autocast_context(amp):
                    # One fused forward for the teacher's 2B globals ...
                    teacher_out = model.forward_teacher_views(
                        teacher_views, chunk_size=forward_chunk_size
                    )
                    # ... and one for the student's 6B views, view-major.
                    student_out, bottleneck = model.forward_student_views(
                        student_views, return_bottleneck=True, chunk_size=forward_chunk_size
                    )
                    # KoLeo measures uniformity of the *representation*, so it
                    # reads the bottleneck of the global views, not the prototype
                    # logits. The globals are the leading 2B rows.
                    student_embeddings = (
                        bottleneck[: 2 * batch_size] if criterion.lambda_koleo > 0 else None
                    )
                    loss = criterion(
                        student_out,
                        teacher_out,
                        epoch=epoch,
                        # Explicit view identifiers. Matching by position is only
                        # correct while the student's first two views are the two
                        # globals in the teacher's order -- an invariant nothing
                        # enforced, and one whose violation would silently skip a
                        # global-local pair and include a same-view pair.
                        student_view_ids=transform.view_ids,
                        teacher_view_ids=transform.global_view_ids,
                        student_embeddings=student_embeddings,
                    )

                scaled = loss / accumulation_steps
                if scaler is not None:
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()

                gradient_norm = None
                clipped_norm = None

                if is_step:
                    if scaler is not None:
                        # Gradients must be back on their true scale before
                        # anything inspects or clips them.
                        scaler.unscale_(optimizer)

                    if (
                        artifacts.log_gradient_norms
                        and gradient_norm_every_steps > 0
                        and global_step % gradient_norm_every_steps == 0
                    ):
                        gradient_norm = tracker.log_gradient_norms(model, global_step)

                    if clip_grad is not None and float(clip_grad) > 0:
                        clipped_norm = torch.nn.utils.clip_grad_norm_(
                            student_parameters, max_norm=float(clip_grad)
                        )

                    # Section 6.1: last layer frozen for the first epoch. Cancelled
                    # *after* clipping and *before* step(), matching the reference
                    # implementation's ordering.
                    model.student_head.cancel_last_layer_gradients(current_epoch=epoch)

                    if (
                        artifacts.log_gradient_histograms
                        and int(intervals.gradient_histogram_every_epochs) > 0
                        and (epoch + 1) % int(intervals.gradient_histogram_every_epochs) == 0
                        and batch_idx == 0
                    ):
                        tracker.log_gradient_histograms(model, global_step)

                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    # EMA teacher update, cosine-scheduled 0.996 -> 1.0 (DINO).
                    ema.update(momentum)
                    micro_in_window = 0
                else:
                    micro_in_window += 1

                total_loss += loss.detach().float()
                batches_seen += 1

                if will_log:
                    lr = optimizer.param_groups[0]["lr"]
                    elapsed = max(time.perf_counter() - epoch_started, 1e-9)
                    step_metrics = {
                        # The single synchronisation point of the step, taken
                        # only on logging steps.
                        "loss": float(loss.detach()),
                        "lr": lr,
                        "teacher_temp": criterion.teacher_temperature(epoch),
                        "teacher_momentum": momentum,
                        "weight_decay": optimizer.param_groups[0]["weight_decay"],
                        "images_per_second": batches_seen * batch_size / elapsed,
                        "views_per_second": batches_seen * batch_size * num_crops / elapsed,
                        # If this is large the GPU is starving and the fix is in
                        # data.num_workers / prefetch_factor / cache_images, not
                        # in the model.
                        "data_wait_fraction": data_wait / elapsed,
                        # The loss curve of a partially collapsed run looks
                        # perfectly plausible. These are the numbers that do not.
                        **criterion.last_metrics,
                    }
                    if gradient_norm is not None:
                        step_metrics["gradient_norm"] = gradient_norm
                    if clipped_norm is not None:
                        step_metrics["clipped_gradient_norm"] = float(clipped_norm)
                    if scaler is not None:
                        step_metrics["grad_scale"] = float(scaler.get_scale())
                    tracker.log_metrics(step_metrics, global_step, prefix="train")
                    logger.info(
                        "Step %s | epoch=%s batch=%s loss=%.5f lr=%.6g tau_t=%.4f "
                        "img/s=%.1f data_wait=%.1f%%",
                        global_step, epoch + 1, batch_idx + 1, step_metrics["loss"],
                        lr, criterion.teacher_temperature(epoch),
                        step_metrics["images_per_second"],
                        step_metrics["data_wait_fraction"] * 100.0,
                    )

                if is_step and global_step % device_every_steps == 0:
                    tracker.log_metrics(collect_device_stats(device), global_step)

                if (
                    artifacts.log_embeddings
                    and int(intervals.embedding_every_epochs) > 0
                    and (epoch + 1) % int(intervals.embedding_every_epochs) == 0
                    and batch_idx == 0
                ):
                    tracker.log_embeddings(
                        "dino/student_projection",
                        student_out[:batch_size].detach(),
                        global_step,
                        metadata=[str(item) for item in batch.paths],
                    )

                if (
                    artifacts.log_attention_maps
                    and int(intervals.attention_every_epochs) > 0
                    and (epoch + 1) % int(intervals.attention_every_epochs) == 0
                    and batch_idx == 0
                ):
                    log_attention_maps(
                        tracker, model.student_backbone, student_views[:batch_size], global_step,
                        logger=logger, max_images=int(artifacts.max_attention_images),
                    )

                if is_step:
                    global_step += 1
                wait_started = time.perf_counter()

            if batches_seen == 0:
                raise RuntimeError("No batches were processed. Check max_batches and the dataloader.")

            average_loss = float(total_loss) / batches_seen
            loss_history.append(average_loss)
            epoch_seconds = time.perf_counter() - epoch_started

            if scheduler is not None:
                scheduler.step()
            new_lr = optimizer.param_groups[0]["lr"]

            # Weight decay is cosine-scheduled 0.04 -> 0.4 in DINO. The submitted
            # constant 0.01 sat below even the schedule's starting value.
            if weight_decay_final is not None:
                decay = cosine_value(
                    weight_decay_start, float(weight_decay_final), epoch + 1, epochs
                )
                for group in optimizer.param_groups:
                    group["weight_decay"] = decay

            epoch_metrics = {
                "loss": average_loss,
                "duration_seconds": epoch_seconds,
                "batches": batches_seen,
                "lr": new_lr,
                "teacher_temp": criterion.teacher_temperature(epoch),
                "images_per_second": batches_seen * int(cfg.data.batch_size) / epoch_seconds,
                "data_wait_fraction": data_wait / epoch_seconds,
            }
            if device.type == "cuda":
                epoch_metrics["peak_memory_mb"] = torch.cuda.max_memory_allocated() / 1024**2
            tracker.log_metrics(epoch_metrics, epoch + 1, prefix="epoch")
            logger.info(
                "Epoch %s/%s | loss=%.5f batches=%s duration=%.2fs img/s=%.1f "
                "data_wait=%.1f%% peak_mem=%.0fMB",
                epoch + 1, epochs, average_loss, batches_seen, epoch_seconds,
                epoch_metrics["images_per_second"],
                epoch_metrics["data_wait_fraction"] * 100.0,
                epoch_metrics.get("peak_memory_mb", 0.0),
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
                "amp": amp.label,
                **{f"runtime_{key}": value for key, value in runtime.items()},
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
