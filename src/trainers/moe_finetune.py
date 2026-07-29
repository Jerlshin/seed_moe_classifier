"""Stage 2: hierarchical MoE finetuning, ablations and baselines (paper Section 5).

    python main.py finetune
    python main.py finetune data.batch_size=8 experiment.training.epochs=5
    python scripts/run_ablations.py --variants full_model wo_moe
    python scripts/run_baselines.py --models resnet50 swin_tiny

One trainer serves all three, which is the point: an ablation that ran through a
second training loop would differ from the full model in ways nobody
intentionally chose. Variants differ only by Hydra overrides -- component
toggles, loss weights, or a swap of the model builder for a supervised baseline.

Reports every quantity the paper's Section 6 does -- seed-type and sub-variety
accuracy/F1/AUC, per-class precision-recall-F1, the KL alignment rate broken
down by seed type, confusion matrices, per-sub-variety misclassification rates,
MoE expert utilisation, t-SNE projections -- plus the revision's additions:
computational efficiency (total vs. active parameters, FLOPs, latency,
throughput, peak memory) and side-by-side train/validation loss curves for
overfitting diagnosis.

Every run writes ``summary.json`` and ``test_predictions.npz`` into its save
path, which is the contract ``scripts/generate_plots.py`` reads to build the
cross-run comparison table without retraining anything.

By default the SwinV2 encoder is frozen and only the head trains; set
``model.backbone.freeze=false`` to fine-tune end to end as Section 4 describes.
"""

from __future__ import annotations

import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.datasets.dataset import HierarchicalSeedDataset, get_finetune_dataset
from src.datasets.transforms import get_supervised_transforms
from src.losses.hierarchical import CombinedHierarchicalLoss, build_combined_loss
from src.models.baselines import IdentityEncoder, build_baseline
from src.models.builder import build_encoder, build_hierarchical_moe
from src.utils.efficiency import EfficiencyReport, profile_model
from src.utils.evaluation import RunSummary, save_test_predictions
from src.utils.metrics import HierarchicalEvaluation, evaluate_hierarchical, tsne_projection
from src.utils.training import (
    CheckpointManager,
    ExperimentTracker,
    collect_device_stats,
    select_device,
    setup_experiment_logger,
    snapshot_run_configuration,
    to_cpu_state_dict,
)
from src.utils.visualization import (
    plot_confusion_matrix,
    plot_expert_utilization,
    plot_loss_curves,
    plot_metric_heatmap,
    plot_misclassification_rates,
    plot_tsne,
)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch, and make cuDNN deterministic.

    Reproducibility across the ablation and baseline suites depends on this: the
    variants are compared against each other, so an unseeded split or
    initialisation would show up as a spurious difference between variants.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------- splits


def stratification_labels(dataset: HierarchicalSeedDataset) -> np.ndarray:
    """Composite ``seed * 1000 + sub`` key so splits stay stratified on both levels."""
    return np.array([seed * 1000 + sub for _, seed, sub in dataset.samples])


def split_dataset(
    dataset: HierarchicalSeedDataset,
    test_size: float,
    num_folds: int,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Carve out a held-out test set, then build ``num_folds`` train/val splits.

    Driven entirely by ``seed``, so every variant in the ablation suite sees the
    byte-identical partition. Comparing variants trained on different splits
    would confound the architecture change with the split.
    """
    labels = stratification_labels(dataset)
    indices = np.arange(len(dataset))

    if test_size > 0:
        train_val_indices, test_indices = train_test_split(
            indices, test_size=test_size, stratify=labels, random_state=seed
        )
    else:
        train_val_indices, test_indices = indices, np.array([], dtype=np.int64)

    train_val_labels = labels[train_val_indices]
    if num_folds > 1:
        splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        splits = [
            (train_val_indices[train_idx], train_val_indices[val_idx])
            for train_idx, val_idx in splitter.split(np.zeros(len(train_val_labels)), train_val_labels)
        ]
    else:
        train_idx, val_idx = train_test_split(
            train_val_indices,
            test_size=max(test_size, 0.2),
            stratify=train_val_labels,
            random_state=seed,
        )
        splits = [(train_idx, val_idx)]

    return splits, test_indices


def save_split_manifest(
    output_dir: str | Path,
    splits: list[tuple[np.ndarray, np.ndarray]],
    test_indices: np.ndarray,
    dataset: HierarchicalSeedDataset,
) -> str:
    """Persist split indices and class mappings so evaluation stays reproducible."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / "split_manifest.npz"
    payload: dict[str, Any] = {
        "test_indices": test_indices,
        "seed_type_to_idx": np.array(list(dataset.seed_type_to_idx.items()), dtype=object),
        "subvariety_to_idx": np.array(list(dataset.subvariety_to_idx.items()), dtype=object),
        "subvariety_to_seed_type": np.array(dataset.get_subvariety_to_seed_type()),
    }
    for fold, (train_indices, val_indices) in enumerate(splits, start=1):
        payload[f"fold_{fold}_train_indices"] = train_indices
        payload[f"fold_{fold}_val_indices"] = val_indices
    np.savez_compressed(path, **payload)
    return str(path)


# ------------------------------------------------------------------ model wiring


def build_model_and_encoder(cfg: DictConfig, device: torch.device) -> tuple[nn.Module, nn.Module]:
    """Return ``(encoder, model)`` for either the proposed framework or a baseline.

    Dispatches on ``model.head.name``:

    ``flat_supervised``
        An end-to-end supervised baseline. It owns its backbone, so the encoder
        slot is filled by :class:`~src.models.baselines.IdentityEncoder` and the
        images reach the model untouched.

    anything else
        The proposed framework: a DINOv2-pretrained SwinV2 encoder emitting
        ``z in R^384``, plus the hierarchical head.
    """
    head_name = str(OmegaConf.select(cfg, "model.head.name", default="hierarchical_moe"))

    if head_name == "flat_supervised":
        return IdentityEncoder().to(device), build_baseline(cfg.model.head).to(device)

    encoder = build_encoder(cfg.model.backbone, embed_dim=int(cfg.model.head.embed_dim)).to(device)
    model = build_hierarchical_moe(cfg.model.head).to(device)
    return encoder, model


# ----------------------------------------------------------------- optimisation


def build_optimizer(modules: list[nn.Module], cfg: DictConfig) -> optim.Optimizer:
    """Build the optimizer over every trainable parameter in ``modules``.

    The encoder must be included, not just the head: even in the frozen recipe
    the encoder owns the Eq. 4 projection to ``z``, which is trainable. Omitting
    it would silently freeze the one layer that adapts the backbone's 1024
    channels to the head's 384 -- the head would train against a random
    projection.

    Parameters are de-duplicated by identity, so a module appearing twice does
    not get two optimizer entries and therefore two updates per step.
    """
    seen: dict[int, nn.Parameter] = {}
    for module in modules:
        if module is None:
            continue
        for parameter in module.parameters():
            if parameter.requires_grad:
                seen[id(parameter)] = parameter
    trainable = list(seen.values())
    if not trainable:
        raise RuntimeError(
            "No trainable parameters found. Every module is frozen, so there is nothing to optimise."
        )

    name = str(cfg.experiment.training.optimizer.name).lower()
    lr = float(cfg.experiment.training.learning_rate)
    weight_decay = float(cfg.experiment.training.weight_decay)

    if name == "adamw":
        return optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return optim.Adam(trainable, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return optim.SGD(trainable, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {cfg.experiment.training.optimizer.name}")


def build_scheduler(optimizer: optim.Optimizer, cfg: DictConfig):
    """Build the LR scheduler, or ``None`` when the config disables it."""
    name = OmegaConf.select(cfg, "experiment.training.scheduler.name", default=None)
    if name is None:
        return None
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(cfg.experiment.training.epochs),
            eta_min=float(OmegaConf.select(cfg, "experiment.training.scheduler.eta_min", default=0.0)),
        )
    raise ValueError(f"Unsupported scheduler: {name}")


# ------------------------------------------------------------------ epoch loop


@dataclass
class EpochAccumulator:
    """Collects predictions across an epoch so metrics are computed once, at the end.

    Computing accuracy/F1 per batch and averaging them is not the same number as
    computing them over the whole epoch; the paper reports the latter.
    """

    seed_true: list[int] = field(default_factory=list)
    seed_pred: list[int] = field(default_factory=list)
    sub_true: list[int] = field(default_factory=list)
    sub_pred: list[int] = field(default_factory=list)
    sub_scores: list[np.ndarray] = field(default_factory=list)
    embeddings: list[np.ndarray] = field(default_factory=list)
    expert_indices: list[np.ndarray] = field(default_factory=list)
    loss_sums: dict[str, float] = field(default_factory=dict)
    total_loss: float = 0.0
    batches: int = 0

    def update(self, output, loss_parts: dict[str, float], seed_labels, sub_labels, keep_embeddings: int) -> None:
        self.seed_pred.extend(output.seed_type_logits.argmax(dim=1).detach().cpu().tolist())
        self.seed_true.extend(seed_labels.detach().cpu().tolist())
        self.sub_pred.extend(output.sub_logits.argmax(dim=1).detach().cpu().tolist())
        self.sub_true.extend(sub_labels.detach().cpu().tolist())
        self.sub_scores.append(F.softmax(output.sub_logits.detach(), dim=-1).cpu().numpy())
        self.expert_indices.append(output.top_k_indices.detach().cpu().numpy())

        collected = sum(chunk.shape[0] for chunk in self.embeddings)
        if collected < keep_embeddings:
            self.embeddings.append(output.sub_embeddings.detach().float().cpu().numpy())

        self.total_loss += loss_parts["total_loss"]
        for key, value in loss_parts.items():
            self.loss_sums[key] = self.loss_sums.get(key, 0.0) + value
        self.batches += 1

    def mean_losses(self) -> dict[str, float]:
        if self.batches == 0:
            return {}
        return {key: value / self.batches for key, value in self.loss_sums.items()}

    def stacked_scores(self) -> np.ndarray | None:
        return np.concatenate(self.sub_scores, axis=0) if self.sub_scores else None

    def stacked_embeddings(self) -> np.ndarray | None:
        return np.concatenate(self.embeddings, axis=0) if self.embeddings else None

    def stacked_experts(self) -> np.ndarray | None:
        return np.concatenate(self.expert_indices, axis=0) if self.expert_indices else None


@dataclass
class LossHistory:
    """Per-epoch train and validation losses, for the overfitting diagnostic.

    The gap between the two curves is the signal: a validation loss that turns
    upward while the training loss keeps falling is the classic overfitting
    signature, and on a 27-class problem with a few hundred samples per class it
    is the failure mode most worth watching for.
    """

    train: list[float] = field(default_factory=list)
    validation: list[float] = field(default_factory=list)

    def append(self, train_loss: float, validation_loss: float) -> None:
        self.train.append(float(train_loss))
        self.validation.append(float(validation_loss))

    @property
    def latest_gap(self) -> float:
        """``validation - train`` for the most recent epoch."""
        if not self.train or not self.validation:
            return float("nan")
        return self.validation[-1] - self.train[-1]

    def as_series(self) -> dict[str, list[float]]:
        return {"train_loss": list(self.train), "validation_loss": list(self.validation)}


def forward_batch(
    encoder: nn.Module,
    model: nn.Module,
    criterion: CombinedHierarchicalLoss,
    batch,
    device: torch.device,
    use_amp: bool,
    training: bool,
):
    """Run encoder -> head -> loss for one batch."""
    images, seed_labels, sub_labels = batch[:3]
    images = images.to(device, non_blocking=True)
    seed_labels = seed_labels.to(device, non_blocking=True)
    sub_labels = sub_labels.to(device, non_blocking=True)

    amp_context = torch.autocast(device_type=device.type) if use_amp else nullcontext()
    with amp_context:
        features = encoder(images)
        # The ArcFace margin is only placed during training; at evaluation the
        # margin logits equal the plain logits, so metrics stay comparable.
        output = model(features, sub_variety_labels=sub_labels if training else None)
        breakdown = criterion(output, seed_labels, sub_labels)

    return output, breakdown, seed_labels, sub_labels


def run_epoch(
    encoder: nn.Module,
    model: nn.Module,
    criterion: CombinedHierarchicalLoss,
    loader: DataLoader,
    device: torch.device,
    tracker: ExperimentTracker,
    logger,
    epoch: int,
    global_step: int,
    phase: str,
    dataset: HierarchicalSeedDataset,
    optimizer: optim.Optimizer | None = None,
    max_batches: int | None = None,
    clip_grad: float | None = None,
    use_amp: bool = False,
    num_experts: int = 6,
    max_tsne_samples: int = 2000,
) -> tuple[dict[str, Any], HierarchicalEvaluation, EpochAccumulator, int]:
    """Run one train or evaluation epoch and return its metrics."""
    is_train = optimizer is not None
    model.train(is_train)
    encoder.train(is_train)
    criterion.train(is_train)

    accumulator = EpochAccumulator()
    log_every = int(OmegaConf.select(tracker.cfg, "tracking.intervals.log_every_steps", default=10))
    log_grad_norms = bool(
        OmegaConf.select(tracker.cfg, "tracking.artifacts.log_gradient_norms", default=True)
    )

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        grad_context = nullcontext() if is_train else torch.no_grad()
        with grad_context:
            output, breakdown, seed_labels, sub_labels = forward_batch(
                encoder=encoder,
                model=model,
                criterion=criterion,
                batch=batch,
                device=device,
                use_amp=use_amp,
                training=is_train,
            )

        if is_train:
            breakdown.total.backward()
            if clip_grad is not None and clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(
                    [group_param for group in optimizer.param_groups for group_param in group["params"]],
                    max_norm=float(clip_grad),
                )
            optimizer.step()

        accumulator.update(
            output,
            breakdown.as_dict(),
            seed_labels,
            sub_labels,
            keep_embeddings=max_tsne_samples,
        )

        if is_train:
            global_step += 1
            if global_step % log_every == 0:
                tracker.log_metrics(
                    {"loss": float(breakdown.total.detach())}, global_step, prefix="train/step"
                )
                if log_grad_norms:
                    tracker.log_gradient_norms(model, global_step)

    if accumulator.batches == 0:
        raise RuntimeError(f"No batches processed during {phase}; check max_batches and the split sizes.")

    seed_names, sub_names = dataset.get_ordered_class_names()
    evaluation = evaluate_hierarchical(
        seed_true=accumulator.seed_true,
        seed_pred=accumulator.seed_pred,
        sub_true=accumulator.sub_true,
        sub_pred=accumulator.sub_pred,
        subvariety_to_seed_type=dataset.get_subvariety_to_seed_type(),
        num_seed_types=len(seed_names),
        num_sub_varieties=len(sub_names),
        seed_type_names=seed_names,
        sub_variety_names=sub_names,
        sub_scores=accumulator.stacked_scores(),
        top_k_indices=accumulator.stacked_experts(),
        num_experts=num_experts,
    )

    metrics: dict[str, Any] = {
        "loss": accumulator.total_loss / accumulator.batches,
        "batches": accumulator.batches,
        **accumulator.mean_losses(),
        **evaluation.scalar_metrics(),
    }
    tracker.log_metrics(metrics, epoch, prefix=phase)

    logger.info(
        "%s epoch %s | loss=%.5f seed_acc=%.4f sub_acc=%.4f kl_align=%.4f batches=%s",
        phase,
        epoch,
        metrics["loss"],
        evaluation.seed_type.get("accuracy", float("nan")),
        evaluation.sub_variety.get("accuracy", float("nan")),
        evaluation.alignment.overall,
        accumulator.batches,
    )
    return metrics, evaluation, accumulator, global_step


# --------------------------------------------------------------------- figures


def log_evaluation_artifacts(
    tracker: ExperimentTracker,
    evaluation: HierarchicalEvaluation,
    accumulator: EpochAccumulator,
    dataset: HierarchicalSeedDataset,
    step: int,
    phase: str,
    logger,
) -> None:
    """Push the paper's Section 6 figures and tables to the tracker."""
    artifacts = OmegaConf.select(tracker.cfg, "tracking.artifacts", default={})
    seed_names, sub_names = dataset.get_ordered_class_names()

    def enabled(key: str, default: bool = True) -> bool:
        return bool(artifacts.get(key, default)) if artifacts else default

    try:
        if enabled("log_confusion_matrices"):
            tracker.log_figure(
                f"{phase}/confusion_seed_type",
                plot_confusion_matrix(evaluation.seed_confusion, seed_names, "Seed type confusion"),
                step,
            )
            tracker.log_figure(
                f"{phase}/confusion_sub_variety",
                plot_confusion_matrix(evaluation.sub_confusion, sub_names, "Sub-variety confusion"),
                step,
            )
            tracker.log_figure(
                f"{phase}/misclassification_sub_variety",
                plot_misclassification_rates(evaluation.sub_misclassification),
                step,
            )
            tracker.log_figure(
                f"{phase}/metric_heatmap_sub_variety",
                plot_metric_heatmap(evaluation.per_class_sub),
                step,
            )

        if enabled("log_expert_utilization"):
            tracker.log_figure(
                f"{phase}/expert_utilization",
                plot_expert_utilization(evaluation.expert_utilization),
                step,
            )
            experts = accumulator.stacked_experts()
            if experts is not None:
                tracker.log_histogram(f"{phase}/expert_routing", experts.reshape(-1), step)

        if enabled("log_per_class_tables"):
            tracker.log_table(
                f"{phase}/per_class_seed_type",
                ["class", "precision", "recall", "f1", "support"],
                [
                    [e.name, e.precision, e.recall, e.f1, e.support]
                    for e in evaluation.per_class_seed
                ],
                step,
            )
            tracker.log_table(
                f"{phase}/per_class_sub_variety",
                ["class", "precision", "recall", "f1", "support"],
                [
                    [e.name, e.precision, e.recall, e.f1, e.support]
                    for e in evaluation.per_class_sub
                ],
                step,
            )
            tracker.log_table(
                f"{phase}/kl_alignment",
                ["seed_type", "alignment_rate", "support"],
                [
                    [name, rate, evaluation.alignment.support_per_seed_type.get(name, 0)]
                    for name, rate in evaluation.alignment.per_seed_type.items()
                ],
                step,
            )

        if enabled("log_tsne"):
            embeddings = accumulator.stacked_embeddings()
            if embeddings is not None:
                max_samples = int(artifacts.get("max_tsne_samples", 2000)) if artifacts else 2000
                perplexity = float(artifacts.get("tsne_perplexity", 30.0)) if artifacts else 30.0
                projection = tsne_projection(
                    embeddings, perplexity=perplexity, max_samples=max_samples
                )
                if projection is not None:
                    count = projection.shape[0]
                    tracker.log_figure(
                        f"{phase}/tsne_seed_type",
                        plot_tsne(
                            projection,
                            accumulator.seed_true[:count],
                            seed_names,
                            "t-SNE coloured by seed type",
                            annotate_clusters=True,
                        ),
                        step,
                    )
                    tracker.log_figure(
                        f"{phase}/tsne_sub_variety",
                        plot_tsne(
                            projection,
                            accumulator.sub_true[:count],
                            sub_names,
                            "t-SNE coloured by sub-variety",
                            annotate_clusters=True,
                        ),
                        step,
                    )
    except Exception as exc:
        # Figures are diagnostics; never let one abort a training run.
        logger.warning("Failed to log evaluation artifacts for %s: %s", phase, exc)


def log_loss_curves(tracker: ExperimentTracker, history: LossHistory, step: int, logger) -> None:
    """Log the side-by-side train/validation loss figure (overfitting diagnostic)."""
    if not history.train:
        return
    try:
        tracker.log_figure(
            "diagnostics/loss_curves",
            plot_loss_curves(history.as_series(), title="Training vs. validation loss"),
            step,
        )
    except Exception as exc:
        logger.warning("Failed to log loss curves: %s", exc)


# ---------------------------------------------------------------- efficiency


def profile_run(
    encoder: nn.Module,
    model: nn.Module,
    dataset: HierarchicalSeedDataset,
    cfg: DictConfig,
    device: torch.device,
    logger,
) -> EfficiencyReport | None:
    """Measure parameters, FLOPs, latency, throughput and peak memory.

    Profiles the *deployed* path -- encoder and head together -- because that is
    what an inference latency figure has to mean. Profiling the head alone would
    report a number no user could ever observe.
    """
    if not bool(OmegaConf.select(cfg, "experiment.efficiency.enabled", default=True)):
        return None

    image_size = int(cfg.data.image_size)
    batch_sizes = list(
        OmegaConf.select(cfg, "experiment.efficiency.batch_sizes", default=[1, 8, 32]) or [1, 8, 32]
    )
    example = torch.randn(max(batch_sizes[0], 1), 3, image_size, image_size, device=device)

    def forward(batch: torch.Tensor):
        return model(encoder(batch))

    try:
        report = profile_model(
            model,
            example,
            device,
            name=str(cfg.experiment.name),
            extra_modules=[encoder],
            batch_sizes=batch_sizes,
            warmup=int(OmegaConf.select(cfg, "experiment.efficiency.warmup", default=3)),
            iterations=int(OmegaConf.select(cfg, "experiment.efficiency.iterations", default=10)),
            forward_fn=forward,
            measure_flops=bool(OmegaConf.select(cfg, "experiment.efficiency.measure_flops", default=True)),
            measure_latency=bool(
                OmegaConf.select(cfg, "experiment.efficiency.measure_latency", default=True)
            ),
        )
    except Exception as exc:
        logger.warning("Efficiency profiling failed: %s", exc)
        return None

    logger.info("Efficiency | %s", report.summary_line())
    for note in report.notes:
        logger.debug("Efficiency note: %s", note)
    return report


# ----------------------------------------------------------------- checkpoints


def save_checkpoint(
    checkpoint_manager: CheckpointManager,
    filename: str,
    encoder: nn.Module,
    model: nn.Module,
    criterion: CombinedHierarchicalLoss,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    fold: int,
    dataset: HierarchicalSeedDataset,
    include_optimizer: bool = False,
    rolling_prefix: str | None = None,
) -> str:
    """Write a checkpoint carrying everything needed to reproduce inference.

    The encoder state is stored alongside the head because the Eq. 4 projection
    is trained here; a head-only checkpoint would be unusable without it.
    """
    payload = {
        "epoch": epoch,
        "fold": fold,
        "model_state_dict": to_cpu_state_dict(model.state_dict()),
        "encoder_state_dict": to_cpu_state_dict(encoder.state_dict()),
        "criterion_state_dict": to_cpu_state_dict(criterion.state_dict()),
        "seed_type_to_idx": dataset.seed_type_to_idx,
        "subvariety_to_idx": dataset.subvariety_to_idx,
        "subvariety_to_seed_type": dataset.get_subvariety_to_seed_type(),
    }
    if include_optimizer:
        payload["optimizer_state_dict"] = optimizer.state_dict()
        payload["scheduler_state_dict"] = scheduler.state_dict() if scheduler is not None else None
    return checkpoint_manager.save(filename, payload, rolling_prefix=rolling_prefix)


# ------------------------------------------------------------------------ main


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
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
        logger.info("========== Hierarchical MoE finetuning: %s ==========", cfg.experiment.name)
        seed_everything(int(cfg.seed))
        snapshot_paths = snapshot_run_configuration(cfg, cfg.tracking.output_dir)
        logger.info(
            "Saved run configuration snapshots.",
            extra={"snapshots": {key: str(value) for key, value in snapshot_paths.items()}},
        )

        device = select_device(cfg.device)
        use_amp = bool(OmegaConf.select(cfg, "experiment.training.amp", default=False)) and device.type == "cuda"
        logger.info("Selected training device: %s (amp=%s)", device, use_amp)
        tracker.log_metrics(collect_device_stats(device), step=0)

        # ---------------------------------------------------------- dataset
        transform = get_supervised_transforms(
            image_size=int(cfg.data.image_size),
            train=True,
            normalize_mean=cfg.data.augmentation.normalize_mean,
            normalize_std=cfg.data.augmentation.normalize_std,
            horizontal_flip_prob=float(
                OmegaConf.select(cfg, "experiment.training.horizontal_flip_prob", default=0.0)
            ),
        )
        dataset = get_finetune_dataset(
            data_dir=cfg.data.root_path,
            transform=transform,
            save_csv_path=OmegaConf.select(cfg, "data.save_csv_path", default=None),
        )
        seed_names, sub_names = dataset.get_ordered_class_names()
        logger.info(
            "Loaded %s samples: %s seed types %s, %s sub-varieties.",
            len(dataset),
            len(seed_names),
            seed_names,
            len(sub_names),
        )
        if len(seed_names) != int(cfg.data.num_seed_types) or len(sub_names) != int(cfg.data.num_sub_varieties):
            raise ValueError(
                f"Dataset has {len(seed_names)} seed types and {len(sub_names)} sub-varieties, but the "
                f"config declares {cfg.data.num_seed_types}/{cfg.data.num_sub_varieties}. "
                "Update conf/data/hierarchical_seeds.yaml -- mismatched counts silently corrupt every label."
            )

        # ------------------------------------------------------------ splits
        splits, test_indices = split_dataset(
            dataset=dataset,
            test_size=float(cfg.experiment.training.test_size),
            num_folds=int(cfg.experiment.training.num_folds),
            seed=int(cfg.seed),
        )
        output_dir = Path(cfg.experiment.training.save_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        split_manifest = save_split_manifest(output_dir, splits, test_indices, dataset)
        tracker.log_event("split_manifest", {"path": split_manifest})

        pin_memory = bool(cfg.data.pin_memory) and device.type == "cuda"
        num_workers = int(cfg.data.num_workers)
        batch_size = int(cfg.data.batch_size)
        epochs = int(cfg.experiment.training.epochs)
        max_batches = OmegaConf.select(cfg, "experiment.training.max_batches", default=None)
        max_tsne = int(OmegaConf.select(cfg, "tracking.artifacts.max_tsne_samples", default=2000))
        figure_every = int(OmegaConf.select(cfg, "tracking.intervals.figure_every_epochs", default=5))

        checkpoint_manager = CheckpointManager(
            output_dir, keep_last_n=int(cfg.experiment.training.keep_last_n_checkpoints)
        )
        include_optimizer = bool(cfg.experiment.training.save_optimizer_state)

        def make_loader(indices: np.ndarray, shuffle: bool, drop_last: bool = False) -> DataLoader:
            return DataLoader(
                Subset(dataset, indices),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last,
            )

        global_step = 0
        best_val_loss = float("inf")
        best_state: dict[str, Any] | None = None
        history = LossHistory()
        num_experts = 1

        # ------------------------------------------------------------- folds
        for fold, (train_indices, val_indices) in enumerate(splits, start=1):
            logger.info("Fold %s/%s: %s train / %s val", fold, len(splits), len(train_indices), len(val_indices))
            encoder, model = build_model_and_encoder(cfg, device)
            num_experts = int(getattr(model, "num_experts", 1))

            if getattr(encoder, "load_report", None) is not None:
                report = encoder.load_report
                logger.info(
                    "Loaded backbone checkpoint: %s missing, %s unexpected keys.",
                    len(report["missing_keys"]),
                    len(report["unexpected_keys"]),
                )
                if report["missing_keys"] or report["unexpected_keys"]:
                    logger.warning(
                        "Backbone checkpoint did not match exactly. First missing=%s, first unexpected=%s",
                        report["missing_keys"][:3],
                        report["unexpected_keys"][:3],
                    )

            criterion = build_combined_loss(
                cfg.model.loss,
                num_seed_types=len(seed_names),
                num_sub_varieties=len(sub_names),
                subvariety_to_seed_type=dataset.get_subvariety_to_seed_type(),
            ).to(device)
            optimizer = build_optimizer([encoder, model], cfg)
            scheduler = build_scheduler(optimizer, cfg)

            if fold == 1:
                flags = getattr(model, "component_flags", lambda: {})()
                logger.info("Component flags: %s", flags or "n/a")
                tracker.log_metrics({f"model/{key}": value for key, value in flags.items()}, step=0)
                tracker.log_model_watch(model)

            train_loader = make_loader(train_indices, shuffle=True, drop_last=bool(cfg.data.drop_last))
            val_loader = make_loader(val_indices, shuffle=False)

            for epoch in range(1, epochs + 1):
                epoch_started = time.perf_counter()
                train_metrics, _, _, global_step = run_epoch(
                    encoder, model, criterion, train_loader, device, tracker, logger,
                    epoch, global_step, phase=f"fold_{fold}/train", dataset=dataset,
                    optimizer=optimizer, max_batches=max_batches,
                    clip_grad=cfg.experiment.training.clip_grad, use_amp=use_amp,
                    num_experts=num_experts, max_tsne_samples=max_tsne,
                )
                val_metrics, val_evaluation, val_accumulator, global_step = run_epoch(
                    encoder, model, criterion, val_loader, device, tracker, logger,
                    epoch, global_step, phase=f"fold_{fold}/validation", dataset=dataset,
                    max_batches=max_batches, use_amp=False,
                    num_experts=num_experts, max_tsne_samples=max_tsne,
                )
                if scheduler is not None:
                    scheduler.step()

                absolute_epoch = (fold - 1) * epochs + epoch
                history.append(train_metrics["loss"], val_metrics["loss"])

                if figure_every > 0 and (epoch % figure_every == 0 or epoch == epochs):
                    log_evaluation_artifacts(
                        tracker, val_evaluation, val_accumulator, dataset,
                        absolute_epoch, f"fold_{fold}/validation", logger,
                    )
                    log_loss_curves(tracker, history, absolute_epoch, logger)

                tracker.log_metrics(
                    {
                        "duration_seconds": time.perf_counter() - epoch_started,
                        "lr": optimizer.param_groups[0]["lr"],
                        "train_loss": train_metrics["loss"],
                        "validation_loss": val_metrics["loss"],
                        # Positive and growing means the model is memorising the
                        # training split rather than generalising.
                        "overfitting_gap": history.latest_gap,
                    },
                    step=absolute_epoch,
                    prefix="epoch",
                )

                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    checkpoint_path = save_checkpoint(
                        checkpoint_manager, "best_hierarchical_moe.pth", encoder, model, criterion,
                        optimizer, scheduler, epoch, fold, dataset, include_optimizer,
                    )
                    best_state = {
                        "encoder": encoder, "model": model, "criterion": criterion,
                        "optimizer": optimizer, "scheduler": scheduler,
                        "epoch": epoch, "fold": fold,
                    }
                    tracker.log_event("checkpoint", {"type": "best", "path": checkpoint_path, "loss": best_val_loss})

                save_interval = int(cfg.experiment.training.save_interval)
                if save_interval > 0 and epoch % save_interval == 0:
                    checkpoint_path = save_checkpoint(
                        checkpoint_manager, f"model_fold{fold}_epoch{epoch:04d}.pth", encoder, model,
                        criterion, optimizer, scheduler, epoch, fold, dataset, include_optimizer,
                        rolling_prefix=f"model_fold{fold}_epoch",
                    )
                    tracker.log_event("checkpoint", {"type": "interval", "path": checkpoint_path})

            logger.info("Fold %s complete.", fold)

        # -------------------------------------------------------------- test
        if best_state is None:
            raise RuntimeError("Training finished without producing a best checkpoint.")

        test_evaluation: HierarchicalEvaluation | None = None
        test_accumulator: EpochAccumulator | None = None
        if len(test_indices) > 0:
            logger.info("Evaluating best checkpoint (fold %s, epoch %s) on %s held-out samples.",
                        best_state["fold"], best_state["epoch"], len(test_indices))
            _, test_evaluation, test_accumulator, _ = run_epoch(
                best_state["encoder"], best_state["model"], best_state["criterion"],
                make_loader(test_indices, shuffle=False), device, tracker, logger,
                epoch=1, global_step=global_step, phase="test", dataset=dataset,
                max_batches=max_batches, use_amp=False,
                num_experts=num_experts, max_tsne_samples=max_tsne,
            )
            log_evaluation_artifacts(
                tracker, test_evaluation, test_accumulator, dataset, 1, "test", logger
            )
            logger.info(
                "Test | seed_acc=%.4f sub_acc=%.4f kl_alignment=%.4f",
                test_evaluation.seed_type.get("accuracy", float("nan")),
                test_evaluation.sub_variety.get("accuracy", float("nan")),
                test_evaluation.alignment.overall,
            )

        final_path = save_checkpoint(
            checkpoint_manager, "hierarchical_moe_final.pth", best_state["encoder"],
            best_state["model"], best_state["criterion"], best_state["optimizer"],
            best_state["scheduler"], best_state["epoch"], best_state["fold"], dataset, include_optimizer,
        )
        tracker.log_artifact(final_path, name="hierarchical_moe_final", artifact_type="model")

        # -------------------------------------------------------- efficiency
        efficiency = profile_run(
            best_state["encoder"], best_state["model"], dataset, cfg, device, logger
        )
        if efficiency is not None:
            tracker.log_metrics(efficiency.as_metrics(), step=0)

        # ------------------------------------------------------ run artifacts
        summary_path = write_run_summary(
            cfg=cfg,
            output_dir=output_dir,
            model=best_state["model"],
            evaluation=test_evaluation,
            accumulator=test_accumulator,
            dataset=dataset,
            efficiency=efficiency,
            history=history,
            checkpoint_path=final_path,
            logger=logger,
        )
        tracker.log_event("run_summary", {"path": summary_path})

        total_seconds = time.perf_counter() - started
        tracker.log_event("training_complete", {"duration_seconds": total_seconds, "checkpoint": final_path})
        logger.info("Finetuning complete in %.2fs. Final checkpoint: %s", total_seconds, final_path)

    except Exception:
        logger.exception("Hierarchical MoE finetuning failed.")
        tracker.log_event("exception", {"stage": "moe_finetuning"})
        raise
    finally:
        tracker.log_event("training_end", {"duration_seconds": time.perf_counter() - started})
        tracker.close()


def write_run_summary(
    cfg: DictConfig,
    output_dir: Path,
    model: nn.Module,
    evaluation: HierarchicalEvaluation | None,
    accumulator: EpochAccumulator | None,
    dataset: HierarchicalSeedDataset,
    efficiency: EfficiencyReport | None,
    history: LossHistory,
    checkpoint_path: str,
    logger,
) -> str:
    """Persist ``summary.json`` and ``test_predictions.npz`` for cross-run reporting.

    Written even when there is no held-out split, so a run always leaves a
    machine-readable trace; the metrics block is simply empty in that case.
    """
    artifacts = {"checkpoint": checkpoint_path}

    if evaluation is not None and accumulator is not None:
        seed_names, sub_names = dataset.get_ordered_class_names()
        try:
            artifacts["predictions"] = save_test_predictions(
                output_dir,
                seed_true=accumulator.seed_true,
                seed_pred=accumulator.seed_pred,
                sub_true=accumulator.sub_true,
                sub_pred=accumulator.sub_pred,
                seed_type_names=seed_names,
                sub_variety_names=sub_names,
                subvariety_to_seed_type=dataset.get_subvariety_to_seed_type(),
                sub_scores=accumulator.stacked_scores(),
                embeddings=accumulator.stacked_embeddings(),
                expert_indices=accumulator.stacked_experts(),
            )
        except Exception as exc:
            logger.warning("Failed to save test predictions: %s", exc)

    summary = RunSummary(
        name=str(OmegaConf.select(cfg, "experiment.variant", default=None) or cfg.experiment.name),
        group=str(OmegaConf.select(cfg, "experiment.group", default="experiment")),
        run_dir=str(output_dir),
        metrics=dict(evaluation.scalar_metrics()) if evaluation is not None else {},
        efficiency=efficiency.as_dict() if efficiency is not None else {},
        history=history.as_series(),
        component_flags=getattr(model, "component_flags", lambda: {})(),
        config={
            "backbone": str(OmegaConf.select(cfg, "model.backbone.name", default="")),
            "head": str(OmegaConf.select(cfg, "model.head.name", default="")),
            "embed_dim": OmegaConf.select(cfg, "model.head.embed_dim", default=None),
            "num_experts": OmegaConf.select(cfg, "model.head.num_experts", default=None),
            "top_k": OmegaConf.select(cfg, "model.head.top_k", default=None),
            "epochs": OmegaConf.select(cfg, "experiment.training.epochs", default=None),
            "learning_rate": OmegaConf.select(cfg, "experiment.training.learning_rate", default=None),
            "batch_size": OmegaConf.select(cfg, "data.batch_size", default=None),
            "seed": OmegaConf.select(cfg, "seed", default=None),
        },
        artifacts=artifacts,
    )
    path = summary.save(output_dir)
    logger.info("Wrote run summary to %s", path)
    return path


if __name__ == "__main__":
    main()
