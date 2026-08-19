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
from collections.abc import Sequence
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
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)
from torch.utils.data import DataLoader, DistributedSampler, Subset, WeightedRandomSampler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.datasets.dataset import HierarchicalSeedDataset, get_finetune_dataset
from src.datasets.transforms import get_supervised_transforms
from src.losses.hierarchical import CombinedHierarchicalLoss, build_combined_loss
from src.models.baselines import IdentityEncoder, build_baseline, build_linear_probe
from src.models.builder import build_encoder, build_hierarchical_moe
from src.utils.efficiency import EfficiencyReport, profile_model
from src.utils.evaluation import RunSummary, save_test_predictions
from src.utils.metrics import HierarchicalEvaluation, evaluate_hierarchical, tsne_projection
from src.utils.training import (
    AmpConfig,
    CheckpointManager,
    ExperimentTracker,
    InterruptGuard,
    PeriodicSaver,
    TrainingProgress,
    autocast_context,
    all_reduce_max,
    barrier,
    broadcast_object,
    build_checkpoint_payload,
    build_grad_scaler,
    collect_device_stats,
    collect_rng_states,
    configure_backend,
    load_checkpoint_payload,
    resolve_amp,
    resolve_num_workers,
    resolve_resume_path,
    restore_components,
    restore_rng_states,
    setup_distributed,
    setup_experiment_logger,
    shutdown_distributed,
    snapshot_run_configuration,
    to_cpu_state_dict,
)

#: Rolling resume checkpoints for this stage. Distinct from
#: ``best_hierarchical_moe.pth`` and ``hierarchical_moe_final.pth``, which are
#: inference artifacts: those carry weights, this carries the optimizer moments,
#: the fold and epoch position and every rank's RNG state.
RESUME_PREFIX = "finetune_resume"
RESUME_KEEP_LAST = 2

#: Evaluation is deliberately **not** autocast. Training in bf16 and scoring in
#: fp32 costs one extra forward's worth of precision and buys comparability: the
#: ablation table compares eighteen variants against each other and against
#: baselines, and a metric that moved with the autocast dtype would make those
#: differences partly an artefact of which card the variant happened to run on.
AMP_DISABLED = AmpConfig(enabled=False, dtype=None, device_type="cpu", needs_scaler=False)
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

    Note that this seeds the *process*. DataLoader worker processes need their
    own seeding, which :func:`make_worker_init_fn` supplies -- without it the
    augmentation stream is not reproducible even though the split is.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WorkerSeeder:
    """Per-worker seeding so the augmentation stream is reproducible.

    A class rather than a closure because ``worker_init_fn`` is **pickled** into
    each worker under the ``spawn`` start method (the default on macOS, and the
    only option on Windows), and a locally-defined function cannot be pickled.
    As a closure this raised ``Can't get local object
    'make_worker_init_fn.<locals>.worker_init_fn'`` the moment ``num_workers > 0``
    on anything but Linux, which is why the smoke instructions specify
    ``data.num_workers=0``. Nothing changes under ``fork``.
    """

    def __init__(self, seed: int):
        self.seed = int(seed)

    def __call__(self, worker_id: int) -> None:
        worker_seed = self.seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed % (2**32))
        torch.manual_seed(worker_seed)


def make_worker_init_fn(seed: int) -> WorkerSeeder:
    """Picklable per-worker seeder for ``seed``."""
    return WorkerSeeder(seed)


# --------------------------------------------------------------------- splits

#: How the held-out partition is drawn.
SPLIT_PROTOCOLS = ("grouped", "stratified", "grouped_cv")


def stratification_labels(dataset: HierarchicalSeedDataset) -> np.ndarray:
    """Sub-variety label, which is the whole stratification key.

    A note on a claim this repository used to make. The composite key
    ``seed * 1000 + sub`` was justified as necessary because "stratifying on
    sub-variety alone would not guarantee seed-type balance". That is false: the
    sub-variety labels are **global** (0..26) and each has exactly one parent, so
    ``seed = parent(sub)`` is a deterministic function of ``sub`` and the map
    ``sub -> seed*1000 + sub`` is a bijection. The composite key therefore induced
    *exactly* the same strata as ``sub`` alone. The code was correct and the
    reason was wrong; this returns the simpler key that produces the same
    partition.
    """
    return np.array([sub for _, _, sub in dataset.samples], dtype=np.int64)


def split_dataset(
    dataset: HierarchicalSeedDataset,
    test_size: float,
    num_folds: int,
    seed: int,
    protocol: str = "grouped",
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, dict[str, Any]]:
    """Carve out a held-out test set, then build ``num_folds`` train/val splits.

    Driven entirely by ``seed``, so every variant in the suite sees the
    byte-identical partition. Comparing variants trained on different splits
    would confound the architecture change with the split.

    ``protocol="grouped"`` (default) keeps every crop of one source photograph on
    one side of the boundary. This matters here more than in most datasets: the
    9,357 crops under ``Cropped_Samples`` come from **81 source photographs**,
    a mean of 115 crops per source. Under crop-level splitting, near-duplicate
    views of the same physical seeds -- same lighting, same background, same
    sensor noise, overlapping bounding boxes -- appear in both train and test,
    and the measured accuracy is substantially a memorisation score.

    ``protocol="stratified"`` reproduces the submitted crop-level splitting. It
    is retained deliberately: running the full model under both and reporting the
    delta turns a fatal methodological objection into a measured result. See
    ``scripts/run_ablations.py``'s ``leakage_ungrouped`` variant.

    ``protocol="grouped_cv"`` takes **no held-out test split at all**. Instead
    ``StratifiedGroupKFold`` partitions every crop into ``num_folds``
    photograph-disjoint folds, each crop is held out exactly once, and the
    trainer concatenates the per-fold validation predictions into one
    out-of-fold prediction set covering the whole dataset.

    The reason is a property of this dataset rather than a preference. The
    ``grouped`` protocol's ``GroupShuffleSplit`` takes 20 % of the 81
    photographs *unstratified*, and most sub-varieties have crops from only ~3 --
    so the test side holds **14 of the 27 classes**, and a macro-F1 on it is
    mechanically capped near 14/27 for reasons that have nothing to do with the
    model. ``grouped_cv`` keeps every fold photograph-disjoint, scores every
    class, and is the same thing ``grouped_cv_readout`` already does in
    ``pretrain_eval.py`` -- which is what makes the stage-1 and stage-2 headline
    numbers directly comparable.

    It is **not** the default, because every published stage-2 number was
    produced under ``grouped`` and switching silently would make the new table
    incomparable with the old one. ``classes_present_in_test`` is reported next
    to every split so the cap is visible either way.

    **A limit the protocol cannot fix.** Five of the 27 sub-varieties have crops
    from exactly one source photograph, so no grouped split can put any of their
    crops in both train and test. Grouped stratification therefore degrades to
    "that class is entirely in one partition" for those five. The returned
    report names them; the honest reading is that their scores measure
    within-photograph generalisation whatever the splitter does, and the paper
    must say so.

    Returns ``(splits, test_indices, report)``.
    """
    if protocol not in SPLIT_PROTOCOLS:
        raise ValueError(f"protocol must be one of {SPLIT_PROTOCOLS}, got {protocol!r}")

    labels = stratification_labels(dataset)
    indices = np.arange(len(dataset))
    groups = dataset.source_groups() if protocol in {"grouped", "grouped_cv"} else None
    report: dict[str, Any] = {"protocol": protocol, "seed": int(seed), "num_folds": int(num_folds)}

    if protocol == "grouped_cv":
        if int(num_folds) < 2:
            raise ValueError(
                "grouped_cv needs num_folds >= 2: the folds ARE the evaluation, so a single fold "
                "would leave most crops unscored."
            )
        splitter = StratifiedGroupKFold(
            n_splits=int(num_folds), shuffle=True, random_state=seed
        )
        splits = [
            (indices[train_idx], indices[val_idx])
            for train_idx, val_idx in splitter.split(np.zeros(len(labels)), labels, groups)
        ]
        test_indices = np.array([], dtype=np.int64)
        report.update(leakage_report(dataset, splits, test_indices, protocol))
        # There is no test split, so `classes_present_in_test` would read 0 and
        # look like the pathology this protocol exists to remove. The comparable
        # quantity is how many classes appear in the union of the folds'
        # held-out halves -- which is every class, and is exactly the point.
        report["classes_present_in_test"] = int(
            len({int(dataset.samples[index][2]) for _, validation in splits for index in validation})
        )
        report["out_of_fold"] = True
        report["out_of_fold_note"] = (
            "No held-out test split: every crop is held out exactly once across the folds, and "
            "the headline metrics are computed from the concatenated out-of-fold predictions."
        )
        return splits, test_indices, report

    if test_size > 0:
        if protocol == "grouped":
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            train_val_indices, test_indices = next(splitter.split(indices, labels, groups))
        else:
            train_val_indices, test_indices = train_test_split(
                indices, test_size=test_size, stratify=labels, random_state=seed
            )
    else:
        train_val_indices, test_indices = indices, np.array([], dtype=np.int64)

    train_val_labels = labels[train_val_indices]
    if num_folds > 1:
        if protocol == "grouped":
            fold_splitter = StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=seed)
            fold_iter = fold_splitter.split(
                np.zeros(len(train_val_labels)), train_val_labels, groups[train_val_indices]
            )
        else:
            fold_splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
            fold_iter = fold_splitter.split(np.zeros(len(train_val_labels)), train_val_labels)
        splits = [
            (train_val_indices[train_idx], train_val_indices[val_idx]) for train_idx, val_idx in fold_iter
        ]
    elif protocol == "grouped":
        inner = GroupShuffleSplit(n_splits=1, test_size=max(test_size, 0.2), random_state=seed)
        train_idx, val_idx = next(
            inner.split(train_val_indices, train_val_labels, groups[train_val_indices])
        )
        splits = [(train_val_indices[train_idx], train_val_indices[val_idx])]
    else:
        train_idx, val_idx = train_test_split(
            train_val_indices,
            test_size=max(test_size, 0.2),
            stratify=train_val_labels,
            random_state=seed,
        )
        splits = [(train_idx, val_idx)]

    report.update(leakage_report(dataset, splits, test_indices, protocol))
    return splits, test_indices, report


def leakage_report(
    dataset: HierarchicalSeedDataset,
    splits: list[tuple[np.ndarray, np.ndarray]],
    test_indices: np.ndarray,
    protocol: str,
) -> dict[str, Any]:
    """Count source photographs that straddle a partition boundary.

    Reported for *both* protocols, so the stratified run carries an explicit
    record of how much leakage it has rather than leaving it to be inferred.
    """
    groups = dataset.source_groups()
    test_groups = set(groups[test_indices].tolist()) if test_indices.size else set()
    train_groups: set[int] = set()
    for train_indices, _ in splits:
        train_groups.update(groups[train_indices].tolist())

    shared = sorted(train_groups & test_groups)
    _, sub_names = dataset.get_ordered_class_names()
    test_classes = {int(dataset.samples[i][2]) for i in test_indices.tolist()}
    train_classes: set[int] = set()
    for train_indices, _ in splits:
        train_classes.update(int(dataset.samples[i][2]) for i in train_indices.tolist())

    return {
        "num_source_groups": int(len(set(groups.tolist()))),
        "test_size": int(test_indices.size),
        # E5: quoted next to every macro-F1, because a 27-way macro-F1 scored on
        # a test split that holds 14 classes is mechanically capped near 14/27
        # for reasons unrelated to the model. `grouped_cv` makes this 27.
        "classes_present_in_test": int(len(test_classes)),
        "num_classes": int(len(sub_names)),
        "shared_source_groups": len(shared),
        "leaked_test_fraction": (
            float(np.isin(groups[test_indices], list(shared)).mean()) if test_indices.size else 0.0
        ),
        "sub_varieties_missing_from_train": sorted(
            sub_names[index] for index in test_classes - train_classes
        ),
        "sub_varieties_missing_from_test": sorted(
            sub_names[index] for index in train_classes - test_classes
        ),
        "protocol": protocol,
    }


def save_split_manifest(
    output_dir: str | Path,
    splits: list[tuple[np.ndarray, np.ndarray]],
    test_indices: np.ndarray,
    dataset: HierarchicalSeedDataset,
    protocol: str = "grouped",
) -> str:
    """Persist split indices, group keys and class mappings for reproducibility."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / "split_manifest.npz"
    payload: dict[str, Any] = {
        "test_indices": test_indices,
        "seed_type_to_idx": np.array(list(dataset.seed_type_to_idx.items()), dtype=object),
        "subvariety_to_idx": np.array(list(dataset.subvariety_to_idx.items()), dtype=object),
        "subvariety_to_seed_type": np.array(dataset.get_subvariety_to_seed_type()),
        # Persisted so a later reviewer can verify the grouping rather than
        # trusting the protocol name.
        "source_groups": dataset.source_groups(),
        "split_protocol": np.array(protocol),
    }
    for fold, (train_indices, val_indices) in enumerate(splits, start=1):
        payload[f"fold_{fold}_train_indices"] = train_indices
        payload[f"fold_{fold}_val_indices"] = val_indices
    np.savez_compressed(path, **payload)
    return str(path)


def build_balanced_sampler(
    dataset: HierarchicalSeedDataset,
    indices: np.ndarray,
    generator: torch.Generator | None = None,
) -> WeightedRandomSampler:
    """Inverse-frequency sampler over sub-varieties.

    The hierarchy is 13 rice + 8 millet + 3 + 3, so seed-type accuracy is
    structurally dominated by rice. Macro-F1 is reported but nothing was training
    for it; this is the option that does.
    """
    sub_labels = np.array([dataset.samples[i][2] for i in indices], dtype=np.int64)
    counts = np.bincount(sub_labels, minlength=int(sub_labels.max()) + 1).astype(np.float64)
    weights = 1.0 / np.clip(counts[sub_labels], 1.0, None)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(indices),
        replacement=True,
        generator=generator,
    )


# ------------------------------------------------------------------ model wiring


def build_model_and_encoder(cfg: DictConfig, device: torch.device) -> tuple[nn.Module, nn.Module]:
    """Return ``(encoder, model)`` for the proposed framework or a baseline.

    Dispatches on ``model.head.name``:

    ``flat_supervised``
        An end-to-end supervised baseline. It owns its backbone, so the encoder
        slot is filled by :class:`~src.models.baselines.IdentityEncoder` and the
        images reach the model untouched.

    ``linear_probe``
        The *real* self-supervised encoder plus two linear layers. Shares the
        encoder path with the proposed model on purpose: the probe must see
        byte-identical features, or the gap to the full head would measure a
        different representation rather than a different head.

    anything else
        The proposed framework: a self-supervised SwinV2 encoder emitting
        ``z in R^384``, plus the hierarchical head.
    """
    head_name = str(OmegaConf.select(cfg, "model.head.name", default="hierarchical_moe"))

    if head_name == "flat_supervised":
        return IdentityEncoder().to(device), build_baseline(cfg.model.head).to(device)

    # The encoder's routing granularity has to match the head's, or the head
    # silently receives one token where it expected 64 (or vice versa) and every
    # attention module quietly becomes affine again.
    token_mode = str(OmegaConf.select(cfg, "model.head.token_mode", default="grid"))
    encoder = build_encoder(
        cfg.model.backbone,
        embed_dim=int(cfg.model.head.embed_dim),
        token_mode="pooled" if head_name == "linear_probe" else token_mode,
    ).to(device)

    if head_name == "linear_probe":
        return encoder, build_linear_probe(cfg.model.head).to(device)
    return encoder, build_hierarchical_moe(cfg.model.head).to(device)


# ------------------------------------------------------------------- schedules


def margin_schedule(epoch: int, total_epochs: int, warmup_fraction: float) -> float:
    """Linear ArcFace margin ramp in ``[0, 1]`` over the first ``warmup_fraction``.

    Applying the full margin from step 0 is a documented convergence hazard on
    small backbones and small datasets; CurricularFace reports outright
    divergence at ``m = 0.5`` where ``m = 0.45`` converges.
    """
    if warmup_fraction <= 0.0:
        return 1.0
    warmup_epochs = max(int(round(total_epochs * float(warmup_fraction))), 1)
    return min(max(epoch - 1, 0) / warmup_epochs, 1.0)


def router_noise_schedule(epoch: int, total_epochs: int, initial: float, fraction: float) -> float:
    """Linear anneal of the gating noise to exactly zero.

    ``topk`` is flat almost everywhere, so nothing in the objective can say "this
    token should have gone to expert 4". Gaussian gating noise is the only
    exploration deterministic Top-K has; it has to reach zero before the end so
    the deployed routing is the routing that was measured.
    """
    if initial <= 0.0 or fraction <= 0.0:
        return 0.0
    anneal_epochs = max(int(round(total_epochs * float(fraction))), 1)
    return float(initial) * max(0.0, 1.0 - max(epoch - 1, 0) / anneal_epochs)


# ----------------------------------------------------------------- optimisation


def build_optimizer(modules: list[nn.Module], cfg: DictConfig) -> optim.Optimizer:
    """Build the optimizer over every trainable parameter in ``modules``.

    The encoder must be included, not just the head: even in the frozen recipe
    the encoder owns the Eq. 4 projection to ``z``, which is trainable. Omitting
    it would silently freeze the one layer that adapts the backbone's 1024
    channels to the head's 384 -- the head would train against a random
    projection.

    The **criterion** must be included too under
    ``weighting_mode="uncertainty"``, where it owns three ``log sigma^2``
    scalars. They are the only learnable parameters the loss has ever held, and
    omitting them would leave the task weights pinned at their initial values
    while the logs reported them as "learned".

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

    logits: list[np.ndarray] = field(default_factory=list)
    tokens_per_sample: int = 1

    def update(self, output, loss_parts: dict[str, float], seed_labels, sub_labels, keep_embeddings: int) -> None:
        self.seed_pred.extend(output.seed_type_logits.argmax(dim=1).detach().cpu().tolist())
        self.seed_true.extend(seed_labels.detach().cpu().tolist())
        self.sub_pred.extend(output.sub_logits.argmax(dim=1).detach().cpu().tolist())
        self.sub_true.extend(sub_labels.detach().cpu().tolist())
        self.sub_scores.append(F.softmax(output.sub_logits.detach(), dim=-1).cpu().numpy())
        # Raw logits are kept alongside the probabilities so temperature scaling
        # can be fitted afterwards; refitting from softmax output would need the
        # inverse of a transform that is not invertible in float32.
        self.logits.append(output.sub_logits.detach().float().cpu().numpy())
        self.expert_indices.append(output.top_k_indices.detach().cpu().numpy())
        self.tokens_per_sample = int(getattr(output, "tokens_per_sample", 1))

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

    def stacked_logits(self) -> np.ndarray | None:
        return np.concatenate(self.logits, axis=0) if self.logits else None

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


class TrainStep(nn.Module):
    """``encoder -> head -> loss`` as one module, so DDP has one forward to wrap.

    DDP synchronises the parameters of the module it wraps, on the backward of a
    forward called through *it*. Stage 2's trainable parameters are spread over
    three objects -- the encoder's Eq. 4 projection, the hierarchical head, and
    (under ``weighting_mode="uncertainty"``) the criterion's three learnable
    ``log sigma^2`` scalars -- and anything left outside the wrapper receives no
    all-reduce and drifts apart across ranks while every log looks identical.
    Wrapping all three is the only arrangement in which that cannot happen.

    Registering these modules here does not copy them: ``model.state_dict()`` is
    unchanged, which is what keeps ``model_state_dict`` in the checkpoint free of
    a ``module.`` prefix that ``checkpoint_strict: false`` would turn into a
    silent zero-key load.
    """

    def __init__(self, encoder: nn.Module, model: nn.Module, criterion: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.model = model
        self.criterion = criterion

    def forward(self, images: torch.Tensor, seed_labels: torch.Tensor, sub_labels: torch.Tensor, training: bool):
        features = self.encoder(images)
        # The ArcFace margin is only placed during training; at evaluation the
        # margin logits equal the plain logits, so metrics stay comparable.
        output = self.model(features, sub_variety_labels=sub_labels if training else None)
        breakdown = self.criterion(output, seed_labels, sub_labels)
        return output, breakdown


def forward_batch(
    encoder: nn.Module,
    model: nn.Module,
    criterion: CombinedHierarchicalLoss,
    batch,
    device: torch.device,
    amp: AmpConfig,
    training: bool,
    step_module: nn.Module | None = None,
):
    """Run encoder -> head -> loss for one batch.

    ``step_module`` is the DDP-wrapped :class:`TrainStep` when the run is
    distributed. Calling the raw modules instead would leave the gradients
    unsynchronised, so every rank would train its own model with identical-looking
    loss curves.
    """
    images, seed_labels, sub_labels = batch[:3]
    images = images.to(device, non_blocking=True)
    seed_labels = seed_labels.to(device, non_blocking=True)
    sub_labels = sub_labels.to(device, non_blocking=True)

    with autocast_context(amp):
        if step_module is not None:
            output, breakdown = step_module(images, seed_labels, sub_labels, training)
        else:
            features = encoder(images)
            output = model(features, sub_variety_labels=sub_labels if training else None)
            breakdown = criterion(output, seed_labels, sub_labels)

    return output, breakdown, seed_labels, sub_labels


def per_term_gradient_norms(
    model: nn.Module,
    criterion: CombinedHierarchicalLoss,
    output,
    seed_labels: torch.Tensor,
    sub_labels: torch.Tensor,
) -> dict[str, float]:
    """Gradient norm and pairwise cosine of each loss term at the shared trunk.

    For a multi-term objective this is the single highest-value diagnostic
    available, and it is the direct empirical test for the hypotheses the audit
    could only argue analytically: whether ``L_ArcFace`` dominates the budget,
    whether ``L_cos`` opposes it, and whether ``L_KL`` opposes ``L_seed``. A
    persistently negative cosine between two terms is the signal to reweight or
    apply gradient surgery.

    Costs one extra backward per term, so the trainer calls it rarely
    (``tracking.intervals.gradient_probe_every_steps``, default 50).
    """
    shared = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and ("sub_variety_embedding" in name or "moe" in name)
    ]
    if not shared:
        return {}

    parts = criterion.component_losses(output, seed_labels, sub_labels)
    flattened: dict[str, torch.Tensor] = {}
    for name, term in parts.items():
        if not term.requires_grad or float(term.detach()) == 0.0:
            continue
        grads = torch.autograd.grad(term, shared, retain_graph=True, allow_unused=True)
        vector = torch.cat(
            [g.reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1) for g, p in zip(grads, shared)]
        )
        flattened[name] = vector

    metrics = {f"grad_norm/{name}": float(vector.norm()) for name, vector in flattened.items()}
    pairs = [("arcface", "kl"), ("arcface", "seed"), ("cosine", "arcface"), ("kl", "seed")]
    for left, right in pairs:
        if left in flattened and right in flattened:
            metrics[f"grad_cosine/{left}_vs_{right}"] = float(
                F.cosine_similarity(flattened[left], flattened[right], dim=0)
            )
    return metrics


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
    amp: AmpConfig = AMP_DISABLED,
    scaler=None,
    num_experts: int = 6,
    max_tsne_samples: int = 2000,
    step_module: nn.Module | None = None,
) -> tuple[dict[str, Any], HierarchicalEvaluation, EpochAccumulator, int]:
    """Run one train or evaluation epoch and return its metrics.

    ``step_module`` is the DDP-wrapped :class:`TrainStep`, used for training
    only. Evaluation deliberately does **not** go through it: this trainer
    produces the rows of the comparison table, and a metric assembled from
    per-rank shards would depend on how many GPUs the variant happened to run
    on. The caller runs evaluation on one rank against the raw modules, which are
    parameter-identical across the job.
    """
    is_train = optimizer is not None
    model.train(is_train)
    encoder.train(is_train)
    criterion.train(is_train)

    accumulator = EpochAccumulator()
    log_every = int(OmegaConf.select(tracker.cfg, "tracking.intervals.log_every_steps", default=10))
    probe_every = int(
        OmegaConf.select(tracker.cfg, "tracking.intervals.gradient_probe_every_steps", default=50)
    )
    log_grad_norms = bool(
        OmegaConf.select(tracker.cfg, "tracking.artifacts.log_gradient_norms", default=True)
    )
    log_term_grads = bool(
        OmegaConf.select(tracker.cfg, "tracking.artifacts.log_per_term_gradients", default=True)
    )
    materialize_grads = getattr(model, "materialize_expert_grads", None)
    dead_expert_total = 0

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
                amp=amp if is_train else AMP_DISABLED,
                training=is_train,
                step_module=step_module if is_train else None,
            )

        term_gradients: dict[str, float] = {}
        if is_train:
            probe = (
                log_term_grads
                and probe_every > 0
                and (global_step + 1) % probe_every == 0
                # Off under DDP. The probe runs one extra backward per loss term
                # through the same graph the reducer is armed on, and a
                # diagnostic is not worth any risk of desynchronising the
                # collective. Run the variant on one GPU to collect it.
                and step_module is None
            )
            if probe:
                term_gradients = per_term_gradient_norms(
                    model, criterion, output, seed_labels, sub_labels
                )
            if scaler is not None:
                scaler.scale(breakdown.total).backward()
            else:
                breakdown.total.backward()

            # Optimizer parity. Under sparse dispatch an expert no token reached
            # has `grad is None`, and AdamW skips such parameters entirely --
            # including decoupled weight decay and moment decay. The dense path
            # gives them a zero gradient, so without this the two dispatch modes
            # train measurably different models and the "debug-only" dense path
            # could not be used to validate a sparse run.
            #
            # Runs before `unscale_`, which is the only correct order: an expert
            # that was routed to has a real, still-scaled gradient here, and
            # zeros are invariant to the scale either way.
            if materialize_grads is not None:
                materialize_grads()

            if scaler is not None:
                # Gradients must be back on their true scale before anything
                # clips them -- clipping scaled gradients at 3.0 would clip
                # essentially every step to essentially nothing.
                scaler.unscale_(optimizer)

            if clip_grad is not None and clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(
                    [group_param for group in optimizer.param_groups for group_param in group["params"]],
                    max_norm=float(clip_grad),
                )

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

        dead_expert_total += int(breakdown.dead_experts)
        accumulator.update(
            output,
            breakdown.as_dict(),
            seed_labels,
            sub_labels,
            keep_embeddings=max_tsne_samples,
        )

        if is_train:
            global_step += 1
            if term_gradients:
                tracker.log_metrics(term_gradients, global_step, prefix="train/step")
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
        tokens_per_sample=accumulator.tokens_per_sample,
    )

    metrics: dict[str, Any] = {
        "loss": accumulator.total_loss / accumulator.batches,
        "batches": accumulator.batches,
        # Averaged over batches, so a value above 0 means experts were sitting
        # out *within* steps even if the epoch total looks balanced.
        "mean_dead_experts_per_step": dead_expert_total / accumulator.batches,
        **accumulator.mean_losses(),
        **evaluation.scalar_metrics(),
    }
    tracker.log_metrics(metrics, epoch, prefix=phase)

    logger.info(
        "%s epoch %s | loss=%.5f seed_acc=%.4f sub_acc=%.4f kl_align=%.4f "
        "dead_experts=%s nmi_sub=%.3f batches=%s",
        phase,
        epoch,
        metrics["loss"],
        evaluation.seed_type.get("accuracy", float("nan")),
        evaluation.sub_variety.get("accuracy", float("nan")),
        evaluation.alignment.overall,
        evaluation.dead_experts,
        evaluation.expert_nmi.get("sub_variety", float("nan")),
        accumulator.batches,
    )
    return metrics, evaluation, accumulator, global_step


# --------------------------------------------------------------------- figures


def merge_out_of_fold(
    parts: Sequence[tuple[np.ndarray, EpochAccumulator]],
    dataset: HierarchicalSeedDataset,
    num_samples: int,
) -> tuple[HierarchicalEvaluation, EpochAccumulator]:
    """Concatenate per-fold held-out predictions into one out-of-fold evaluation.

    Under ``split_protocol: grouped_cv`` there is no held-out test split: each of
    the ``num_folds`` photograph-disjoint folds scores its own validation half
    with the model that fold trained, and every crop is therefore predicted
    exactly once by a model that never saw it. Concatenating gives one prediction
    per crop over the whole dataset, all 27 classes present -- which is what the
    confusion matrix, the per-class table and the headline number are computed
    from, and the same protocol ``pretrain_eval``'s ``grouped_cv_readout`` uses.

    **What this is and is not.** It is an estimate of the *recipe*: K different
    models contributed, so it is not any single shipped model's test score, and
    the two must not be quoted interchangeably. It is also not comparable with a
    ``grouped`` number, whose test split holds 14 of the 27 classes -- the whole
    reason this protocol exists.

    The parts are re-ordered into dataset order before scoring, so the arrays
    line up with ``dataset.samples`` and a downstream consumer can join them
    against the labels or the source groups by index.
    """
    seed_names, sub_names = dataset.get_ordered_class_names()
    merged = EpochAccumulator()
    order: list[int] = []
    for indices, accumulator in parts:
        order.extend(int(value) for value in np.asarray(indices).reshape(-1).tolist())
        merged.seed_true.extend(accumulator.seed_true)
        merged.seed_pred.extend(accumulator.seed_pred)
        merged.sub_true.extend(accumulator.sub_true)
        merged.sub_pred.extend(accumulator.sub_pred)
        merged.sub_scores.extend(accumulator.sub_scores)
        merged.logits.extend(accumulator.logits)
        merged.expert_indices.extend(accumulator.expert_indices)
        merged.embeddings.extend(accumulator.embeddings)
        merged.total_loss += accumulator.total_loss
        merged.batches += accumulator.batches
        merged.tokens_per_sample = accumulator.tokens_per_sample
        for key, value in accumulator.loss_sums.items():
            merged.loss_sums[key] = merged.loss_sums.get(key, 0.0) + value

    if len(order) != len(merged.sub_true):
        raise RuntimeError(
            f"Out-of-fold assembly saw {len(order)} fold indices against "
            f"{len(merged.sub_true)} predictions; a fold's loader did not cover its split."
        )
    # Dataset order, so index i is sample i.
    permutation = np.argsort(np.asarray(order, dtype=np.int64), kind="stable")
    for name in ("seed_true", "seed_pred", "sub_true", "sub_pred"):
        values = np.asarray(getattr(merged, name), dtype=np.int64)[permutation]
        setattr(merged, name, values.tolist())
    for name in ("sub_scores", "logits", "expert_indices"):
        chunks = getattr(merged, name)
        if chunks:
            setattr(merged, name, [np.concatenate(chunks, axis=0)[permutation]])

    evaluation = evaluate_hierarchical(
        seed_true=merged.seed_true,
        seed_pred=merged.seed_pred,
        sub_true=merged.sub_true,
        sub_pred=merged.sub_pred,
        subvariety_to_seed_type=dataset.get_subvariety_to_seed_type(),
        num_seed_types=len(seed_names),
        num_sub_varieties=len(sub_names),
        seed_type_names=seed_names,
        sub_variety_names=sub_names,
        sub_scores=merged.stacked_scores(),
        top_k_indices=merged.stacked_experts(),
        num_experts=int(np.max(merged.stacked_experts()) + 1) if merged.expert_indices else 1,
        tokens_per_sample=merged.tokens_per_sample,
    )
    return evaluation, merged


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
    sample_indices: np.ndarray | None = None,
) -> EfficiencyReport | None:
    """Measure parameters, FLOPs, latency, throughput and peak memory.

    Profiles the *deployed* path -- encoder and head together -- because that is
    what an inference latency figure has to mean. Profiling the head alone would
    report a number no user could ever observe.

    ``sample_indices`` supplies **real, distinct** images for the latency sweep.
    Tiling one image to reach batch 32 gives 32 identical gate logits and
    therefore exactly ``K`` expert kernels for the whole batch, which understates
    sparse-dispatch overhead precisely where the benchmark is trying to measure
    it.
    """
    if not bool(OmegaConf.select(cfg, "experiment.efficiency.enabled", default=True)):
        return None

    image_size = int(cfg.data.image_size)
    batch_sizes = list(
        OmegaConf.select(cfg, "experiment.efficiency.batch_sizes", default=[1, 8, 32]) or [1, 8, 32]
    )
    example = torch.randn(max(batch_sizes[0], 1), 3, image_size, image_size, device=device)

    sample_pool: torch.Tensor | None = None
    if sample_indices is not None and len(sample_indices) > 0:
        try:
            wanted = min(max(batch_sizes), len(sample_indices))
            sample_pool = torch.stack(
                [dataset[int(index)][0] for index in sample_indices[:wanted]]
            ).to(device)
        except Exception as exc:  # pragma: no cover - profiling must never abort a run
            logger.debug("Could not build a distinct-sample pool for benchmarking: %s", exc)

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
            warmup=int(OmegaConf.select(cfg, "experiment.efficiency.warmup", default=5)),
            iterations=int(OmegaConf.select(cfg, "experiment.efficiency.iterations", default=50)),
            forward_fn=forward,
            measure_flops=bool(OmegaConf.select(cfg, "experiment.efficiency.measure_flops", default=True)),
            measure_latency=bool(
                OmegaConf.select(cfg, "experiment.efficiency.measure_latency", default=True)
            ),
            sample_pool=sample_pool,
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
    # A rank must own its device before the first tensor exists, and NCCL binds
    # its communicator to whatever device is current when the group comes up.
    context = setup_distributed(
        str(cfg.device),
        timeout_minutes=float(
            OmegaConf.select(cfg, "experiment.training.ddp.timeout_minutes", default=30)
        ),
    )
    # Hydra evaluates `${now:...}` per process, so ranks launched in the same
    # second can still resolve different directories. Rank 0's is authoritative.
    output_dir_str = broadcast_object(str(cfg.tracking.output_dir), context)
    logger = setup_experiment_logger(
        log_dir=output_dir_str,
        name="seed_moe.moe_finetune",
        level=cfg.tracking.log_level,
        console=bool(cfg.tracking.console) and context.is_main,
        structured_jsonl=cfg.tracking.structured_jsonl,
        rank=context.rank,
        world_size=context.world_size,
    )
    tracker = ExperimentTracker(cfg, logger, enabled=context.is_main)
    started = time.perf_counter()
    # Bound before the try so the `finally` can restore the signal handlers even
    # if construction below raises.
    guard: InterruptGuard | None = None

    try:
        logger.info("========== Hierarchical MoE finetuning: %s ==========", cfg.experiment.name)
        if context.enabled:
            logger.info(
                "Distributed run | %s ranks, backend=%s, this rank owns %s. Training is "
                "sharded; evaluation runs on rank 0 against the same parameters, so every "
                "reported metric is the single-process one.",
                context.world_size, context.backend, context.device,
            )
        seed_everything(int(cfg.seed))
        if context.is_main:
            snapshot_paths = snapshot_run_configuration(cfg, output_dir_str)
            logger.info(
                "Saved run configuration snapshots.",
                extra={"snapshots": {key: str(value) for key, value in snapshot_paths.items()}},
            )

        device = context.device
        # Determinism stays on here, unlike stage 1. This trainer produces the
        # rows of the comparison table, and every variant must see a byte-
        # identical split and the same kernels; cuDNN autotuning can pick a
        # different algorithm per process. TF32 is separate and safe: it is the
        # same algorithm every run, just accumulated at lower precision.
        backend = configure_backend(
            device,
            allow_tf32=bool(OmegaConf.select(cfg, "experiment.training.allow_tf32", default=True)),
            deterministic=bool(
                OmegaConf.select(cfg, "experiment.training.deterministic", default=True)
            ),
            matmul_precision=str(
                OmegaConf.select(cfg, "experiment.training.matmul_precision", default="high")
            ),
            logger=logger,
        )
        tracker.log_event("backend", {**backend, "distributed": context.as_dict()})
        amp = resolve_amp(
            device, OmegaConf.select(cfg, "experiment.training.amp", default="auto"), logger=logger
        )
        scaler = build_grad_scaler(amp)
        logger.info(
            "Selected training device: %s (amp=%s, grad scaler=%s). Evaluation always runs in fp32.",
            device, amp.label, "on" if scaler is not None else "off",
        )
        tracker.log_metrics(collect_device_stats(device), step=0)

        # ---------------------------------------------------------- dataset
        crop_scale = OmegaConf.select(
            cfg, "experiment.training.random_resized_crop_scale", default=[0.8, 1.0]
        )
        transform = get_supervised_transforms(
            image_size=int(cfg.data.image_size),
            train=True,
            normalize_mean=cfg.data.augmentation.normalize_mean,
            normalize_std=cfg.data.augmentation.normalize_std,
            horizontal_flip_prob=float(
                OmegaConf.select(cfg, "experiment.training.horizontal_flip_prob", default=0.5)
            ),
            random_resized_crop_scale=list(crop_scale) if crop_scale else None,
            vertical_flip_prob=float(
                OmegaConf.select(cfg, "experiment.training.vertical_flip_prob", default=0.0)
            ),
            rotation_degrees=float(
                OmegaConf.select(cfg, "experiment.training.rotation_degrees", default=0.0)
            ),
        )
        # Evaluation must never see a stochastic crop, or the val/test numbers
        # would be augmentation noise rather than a measurement.
        eval_transform = get_supervised_transforms(
            image_size=int(cfg.data.image_size),
            train=False,
            normalize_mean=cfg.data.augmentation.normalize_mean,
            normalize_std=cfg.data.augmentation.normalize_std,
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
        group_report = dataset.group_report()
        logger.info(
            "Provenance: %s crops from %s source photographs (mean %.1f crops/source). "
            "%s sub-varieties come from a single photograph: %s",
            group_report["num_samples"],
            group_report["num_source_groups"],
            group_report["mean_crops_per_source"],
            group_report["num_single_group_sub_varieties"],
            group_report["single_group_sub_varieties"] or "none",
        )

        split_protocol = str(
            OmegaConf.select(cfg, "experiment.training.split_protocol", default="grouped")
        )
        splits, test_indices, split_report = split_dataset(
            dataset=dataset,
            test_size=float(cfg.experiment.training.test_size),
            num_folds=int(cfg.experiment.training.num_folds),
            seed=int(cfg.seed),
            protocol=split_protocol,
        )
        split_report.update(group_report)
        split_report["corpus"] = dataset.corpus_fingerprint()
        logger.info(
            "Split | protocol=%s, %d of %d sub-varieties present in the %s. A macro-F1 scored on "
            "fewer classes than the taxonomy has is capped below 1 by the split, not by the model.",
            split_protocol,
            int(split_report.get("classes_present_in_test", 0)),
            int(split_report.get("num_classes", len(sub_names))),
            "union of the held-out folds" if split_protocol == "grouped_cv" else "test split",
        )
        if split_protocol == "grouped_cv":
            logger.info(
                "grouped_cv | no held-out test split: %d photograph-disjoint folds, every crop "
                "held out exactly once, headline metrics computed from the concatenated "
                "out-of-fold predictions over all %d crops. This is the same protocol "
                "pretrain_eval's grouped_cv_readout uses, so the stage-1 and stage-2 numbers are "
                "directly comparable -- and it is NOT the protocol any previously published "
                "number here was produced under.",
                len(splits), len(dataset),
            )
        if split_report["shared_source_groups"]:
            logger.warning(
                "Split protocol %r leaves %s source photographs on both sides of the test "
                "boundary, covering %.1f%% of the test set. The reported accuracy is "
                "partly a memorisation score.",
                split_protocol,
                split_report["shared_source_groups"],
                split_report["leaked_test_fraction"] * 100.0,
            )
        if split_report["sub_varieties_missing_from_train"]:
            logger.warning(
                "Grouped splitting left these sub-varieties out of training entirely: %s. "
                "They have too few source photographs to appear on both sides.",
                split_report["sub_varieties_missing_from_train"],
            )

        output_dir = Path(cfg.experiment.training.save_path)
        if context.is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
            split_manifest = save_split_manifest(
                output_dir, splits, test_indices, dataset, protocol=split_protocol
            )
            tracker.log_event("split_manifest", {"path": split_manifest, **split_report})
        barrier(context)

        pin_memory = bool(cfg.data.pin_memory) and device.type == "cuda"
        num_workers = resolve_num_workers(
            OmegaConf.select(cfg, "data.num_workers", default=4),
            context,
            auto_cap=int(OmegaConf.select(cfg, "data.num_workers_auto_cap", default=16) or 16),
            logger=logger,
        )
        batch_size = int(cfg.data.batch_size)
        epochs = int(cfg.experiment.training.epochs)
        max_batches = OmegaConf.select(cfg, "experiment.training.max_batches", default=None)
        max_tsne = int(OmegaConf.select(cfg, "tracking.artifacts.max_tsne_samples", default=2000))
        figure_every = int(OmegaConf.select(cfg, "tracking.intervals.figure_every_epochs", default=5))

        checkpoint_manager = CheckpointManager(
            output_dir,
            keep_last_n=int(cfg.experiment.training.keep_last_n_checkpoints),
            enabled=context.is_main,
        )
        resume_manager = CheckpointManager(
            output_dir, keep_last_n=RESUME_KEEP_LAST, enabled=context.is_main
        )
        include_optimizer = bool(cfg.experiment.training.save_optimizer_state)

        use_balanced_sampler = bool(
            OmegaConf.select(cfg, "experiment.training.balanced_sampler", default=False)
        )
        # Seeded generator + per-worker seeding: without both, the split is
        # reproducible but the shuffling and augmentation stream are not, which
        # is enough to move an ablation gap of the size this table reports.
        loader_generator = torch.Generator()
        loader_generator.manual_seed(int(cfg.seed))
        worker_init_fn = make_worker_init_fn(int(cfg.seed))
        logger.info(
            "Determinism | cudnn.deterministic=%s cudnn.benchmark=%s seed=%s balanced_sampler=%s",
            torch.backends.cudnn.deterministic,
            torch.backends.cudnn.benchmark,
            int(cfg.seed),
            use_balanced_sampler,
        )

        # Two views of the same underlying tree: augmented for training,
        # deterministic for evaluation.
        eval_dataset = get_finetune_dataset(data_dir=cfg.data.root_path, transform=eval_transform)

        def make_loader(
            indices: np.ndarray,
            shuffle: bool,
            drop_last: bool = False,
            train: bool = False,
        ) -> DataLoader:
            """Build one loader. Only the **training** loader is ever sharded.

            Evaluation loaders stay whole on whichever rank runs them, because
            this trainer's output is a row of the comparison table: a metric
            stitched together from per-rank shards would carry a dependence on
            how many GPUs that variant happened to get, and the gaps the table
            reports are 0.5-2 pp.
            """
            source = dataset if train else eval_dataset
            sampler = (
                build_balanced_sampler(dataset, indices, loader_generator)
                if train and use_balanced_sampler
                else None
            )
            if train and context.enabled:
                if sampler is not None:
                    # WeightedRandomSampler draws with replacement from a global
                    # weight vector and has no per-rank partition; composing the
                    # two would either duplicate samples across ranks or drop the
                    # inverse-frequency weighting. Refusing is the honest option:
                    # both features work, just not together.
                    raise ValueError(
                        "experiment.training.balanced_sampler=true is not supported under DDP: "
                        "WeightedRandomSampler and DistributedSampler cannot be composed without "
                        "changing the sampling distribution. Run this variant on one GPU, or "
                        "shard the ablation suite across GPUs instead (scripts/run_ablations.py "
                        "--gpus)."
                    )
                sampler = DistributedSampler(
                    Subset(source, indices),
                    num_replicas=context.world_size,
                    rank=context.rank,
                    shuffle=shuffle,
                    seed=int(cfg.seed),
                    # Equal batch counts per rank. An uneven tail means one rank
                    # entering an all-reduce its peers have already left.
                    drop_last=True,
                )
            # `persistent_workers` matters more here than the epoch count
            # suggests: this trainer builds a *fresh* loader per fold and per
            # evaluation pass, and every short-lived pool pays full worker
            # startup for a handful of batches.
            worker_kwargs = (
                {
                    "persistent_workers": bool(
                        OmegaConf.select(cfg, "data.persistent_workers", default=True)
                    ),
                    "prefetch_factor": int(
                        OmegaConf.select(cfg, "data.prefetch_factor", default=4)
                    ),
                }
                if num_workers > 0
                else {}
            )
            return DataLoader(
                Subset(source, indices),
                batch_size=batch_size,
                shuffle=shuffle and sampler is None,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last,
                generator=loader_generator,
                worker_init_fn=worker_init_fn,
                **worker_kwargs,
            )

        margin_warmup = float(
            OmegaConf.select(cfg, "experiment.training.margin_warmup_fraction", default=0.15)
        )
        router_noise_std = float(OmegaConf.select(cfg, "model.head.router_noise_std", default=0.0))
        router_noise_fraction = float(
            OmegaConf.select(cfg, "experiment.training.router_noise_fraction", default=0.3)
        )

        global_step = 0
        best_val_loss = float("inf")
        best_state: dict[str, Any] | None = None
        history = LossHistory()
        num_experts = 1
        fold_test_metrics: list[dict[str, float]] = []
        #: (val_indices, accumulator) per fold under `grouped_cv`, concatenated
        #: after the fold loop into one out-of-fold prediction set.
        out_of_fold_parts: list[tuple[np.ndarray, EpochAccumulator]] = []

        # ------------------------------------------------------------- resume
        #
        # Stage 2 resumes at **epoch** granularity -- fold, epoch, global step,
        # optimizer moments, best-so-far and every rank's RNG -- where stage 1
        # resumes at micro-batch granularity. The asymmetry is deliberate and is
        # about what an epoch costs: stage 1's is tens of minutes of
        # self-distillation over six views per image, stage 2's is one pass of a
        # ~9 M-parameter head over ~7.5 k images against a frozen encoder. Losing
        # at most one stage-2 epoch to an interruption is not worth the cost of
        # replaying its batches.
        resume_path = resolve_resume_path(
            OmegaConf.select(cfg, "experiment.training.resume", default=False),
            output_dir,
            patterns=(f"{RESUME_PREFIX}*.pth",),
            logger=logger,
        )
        resume_path = broadcast_object(str(resume_path) if resume_path else None, context)
        resume_payload = (
            load_checkpoint_payload(resume_path, map_location=device, logger=logger)
            if resume_path
            else None
        )
        resume_progress = TrainingProgress.from_dict(
            (resume_payload or {}).get("progress")
        )
        resume_fold = int((resume_payload or {}).get("fold", 1))
        if resume_payload is not None:
            global_step = resume_progress.global_step
            best_val_loss = resume_progress.best_metric
            history = LossHistory(
                train=list((resume_payload.get("history") or {}).get("train_loss", [])),
                validation=list((resume_payload.get("history") or {}).get("validation_loss", [])),
            )
            logger.info(
                "Resume | continuing at fold %s, epoch %s, step %s (best val loss %.5f).",
                resume_fold, resume_progress.epoch + 1, global_step, best_val_loss,
            )
            tracker.log_event(
                "resume", {"path": resume_path, "fold": resume_fold, **resume_progress.as_dict()}
            )

        periodic = PeriodicSaver(
            OmegaConf.select(cfg, "experiment.training.resume_every_minutes", default=None)
        )
        # Installed rather than entered as a context manager so the fold loop
        # below keeps its indentation; restored in this function's `finally`.
        guard = InterruptGuard(
            OmegaConf.select(cfg, "experiment.training.max_runtime_minutes", default=None),
            logger=logger,
        ).install()
        interrupted = False

        # ------------------------------------------------------------- folds
        for fold, (train_indices, val_indices) in enumerate(splits, start=1):
            if fold < resume_fold:
                logger.info("Fold %s already completed before the interruption; skipping.", fold)
                continue
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
                embed_dim=int(OmegaConf.select(cfg, "model.head.embed_dim", default=384)),
                num_experts=num_experts,
            ).to(device)
            # The criterion joins the optimizer because uncertainty weighting
            # owns three learnable scalars; omitting it would pin the task
            # weights while reporting them as learned.
            optimizer = build_optimizer([encoder, model, criterion], cfg)
            scheduler = build_scheduler(optimizer, cfg)

            # Restore into *this* fold's freshly built objects. Building first
            # and restoring second is the only order that works, because each
            # fold constructs its own encoder, head, criterion and optimizer.
            fold_components = {
                "model_state_dict": model,
                "encoder_state_dict": encoder,
                "criterion_state_dict": criterion,
                "optimizer_state_dict": optimizer,
                "scheduler_state_dict": scheduler,
                "scaler_state_dict": scaler,
            }
            start_epoch = 1
            if resume_payload is not None and fold == resume_fold:
                restore_components(resume_payload, fold_components, strict=True, logger=logger)
                restore_rng_states(resume_payload, context, loader_generator, logger=logger)
                start_epoch = int(resume_progress.epoch) + 1
                if start_epoch > epochs:
                    logger.info("Fold %s was already complete at the interruption.", fold)
                # Consumed: later folds start fresh, as they would have.
                resume_payload = None

            step_module = None
            if context.enabled:
                from torch.nn.parallel import DistributedDataParallel

                from src.utils.training.distributed import buffer_sync_kwarg

                step_module = DistributedDataParallel(
                    TrainStep(encoder, model, criterion),
                    device_ids=[context.local_rank] if device.type == "cuda" else None,
                    output_device=context.local_rank if device.type == "cuda" else None,
                    **buffer_sync_kwarg(False),
                    gradient_as_bucket_view=True,
                    # Required here, unlike stage 1, and for a reason that is
                    # architectural rather than incidental: under sparse
                    # dispatch an expert that no token routed to genuinely
                    # receives no gradient that step, so DDP cannot assume every
                    # parameter will be marked ready. The alternative -- dense
                    # dispatch -- would change what the MoE ablation measures.
                    #
                    # The arithmetic still comes out right: DDP contributes a
                    # zero for the unused parameter and averages over the world,
                    # which is exactly the global per-sample mean when the tokens
                    # that would have driven it live on another rank.
                    find_unused_parameters=True,
                )
                logger.info(
                    "DDP | encoder projection + head + criterion wrapped across %s ranks "
                    "(find_unused_parameters=True for sparse expert dispatch).",
                    context.world_size,
                )

            if fold == 1:
                flags = getattr(model, "component_flags", lambda: {})()
                logger.info("Component flags: %s", flags or "n/a")
                logger.info("Loss flags: %s", criterion.loss_flags())
                tracker.log_metrics(
                    {
                        f"model/{key}": value
                        for key, value in flags.items()
                        if isinstance(value, (int, float, bool))
                    },
                    step=0,
                )
                tracker.log_model_watch(model)

            train_loader = make_loader(
                train_indices, shuffle=True, drop_last=bool(cfg.data.drop_last), train=True
            )
            val_loader = make_loader(val_indices, shuffle=False)

            for epoch in range(start_epoch, epochs + 1):
                epoch_started = time.perf_counter()
                # The sampler's permutation is a function of `seed + epoch`.
                # Without this call, every epoch replays the first one's order --
                # no error, no change in loss magnitude, and the run silently
                # becomes one epoch repeated.
                if context.enabled and isinstance(train_loader.sampler, DistributedSampler):
                    train_loader.sampler.set_epoch(epoch)

                # Schedules that must move once per epoch, before the loop that
                # reads them.
                margin_scale = margin_schedule(epoch, epochs, margin_warmup)
                noise_scale = router_noise_schedule(
                    epoch, epochs, router_noise_std, router_noise_fraction
                )
                if hasattr(model, "set_margin_scale"):
                    model.set_margin_scale(margin_scale)
                if hasattr(model, "set_router_noise"):
                    model.set_router_noise(noise_scale)

                train_metrics, _, _, global_step = run_epoch(
                    encoder, model, criterion, train_loader, device, tracker, logger,
                    epoch, global_step, phase=f"fold_{fold}/train", dataset=dataset,
                    optimizer=optimizer, max_batches=max_batches,
                    clip_grad=cfg.experiment.training.clip_grad, amp=amp, scaler=scaler,
                    num_experts=num_experts, max_tsne_samples=max_tsne,
                    step_module=step_module,
                )
                # Evaluation runs on one rank against the whole validation split.
                # The parameters are identical across the job -- DDP guarantees
                # it -- so this is the single-process metric by construction,
                # rather than a reduction over shards that would carry a
                # dependence on the GPU count into the comparison table.
                if context.is_main:
                    val_metrics, val_evaluation, val_accumulator, _ = run_epoch(
                        encoder, model, criterion, val_loader, device, tracker, logger,
                        epoch, global_step, phase=f"fold_{fold}/validation", dataset=dataset,
                        max_batches=max_batches, amp=AMP_DISABLED,
                        num_experts=num_experts, max_tsne_samples=max_tsne,
                    )
                else:
                    val_metrics, val_evaluation, val_accumulator = None, None, None
                # Every rank needs the loss to agree on the best-checkpoint
                # decision below; the heavyweight evaluation objects stay where
                # they were computed.
                val_metrics = broadcast_object(val_metrics, context)

                if scheduler is not None:
                    scheduler.step()

                absolute_epoch = (fold - 1) * epochs + epoch
                history.append(train_metrics["loss"], val_metrics["loss"])

                # `val_evaluation` exists only on the rank that ran the
                # evaluation; the others hold None and have nothing to plot.
                if (
                    val_evaluation is not None
                    and figure_every > 0
                    and (epoch % figure_every == 0 or epoch == epochs)
                ):
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
                        "margin_scale": margin_scale,
                        "router_noise": noise_scale,
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

                # Resume state, written at the epoch boundary where every rank is
                # in lockstep and the optimizer has just stepped. `stop` is
                # latched locally but decided globally: the RNG gather inside the
                # write is a collective, so one rank taking it alone would hang
                # the job at the collective timeout.
                stop = guard.should_stop()
                due = periodic.due() or stop.requested or epoch == epochs
                if context.enabled:
                    flags = torch.tensor(
                        [1.0 if due else 0.0, 1.0 if stop.requested else 0.0],
                        device=device, dtype=torch.float32,
                    )
                    all_reduce_max(flags, context)
                    due, stop_now = bool(flags[0] > 0), bool(flags[1] > 0)
                else:
                    stop_now = stop.requested

                if due:
                    resume_manager.save(
                        f"{RESUME_PREFIX}_fold{fold}_epoch{epoch:04d}.pth",
                        build_checkpoint_payload(
                            components=fold_components,
                            progress=TrainingProgress(
                                epoch=epoch,
                                global_step=global_step,
                                micro_step=0,
                                best_metric=best_val_loss,
                                completed=(fold == len(splits) and epoch == epochs),
                            ),
                            context=context,
                            config=cfg,
                            rng_states=collect_rng_states(context, loader_generator),
                            extra={"fold": fold, "history": history.as_series()},
                        ),
                        rolling_prefix=RESUME_PREFIX,
                    )
                    periodic.mark()

                if stop_now:
                    interrupted = True
                    logger.warning(
                        "Stopping after fold %s epoch %s (%s). Relaunch the identical command "
                        "with experiment.training.resume=auto to continue.",
                        fold, epoch, stop.reason or "peer request",
                    )
                    break

            if interrupted:
                break

            # Score *this* fold's final model on the held-out test split before
            # moving on. Reporting only the best fold's test metrics is a
            # selection over K folds, and the expected value of a maximum
            # exceeds the expected value of a single draw -- so the headline
            # number would be optimistically biased the moment num_folds > 1,
            # and incomparable with the num_folds=1 numbers already collected.
            # The mean across folds is what the table uses; best-fold selection
            # survives only for the artifact that gets profiled and shipped.
            if len(test_indices) > 0 and len(splits) > 1 and context.is_main:
                _, fold_test_evaluation, _, _ = run_epoch(
                    encoder, model, criterion, make_loader(test_indices, shuffle=False),
                    device, tracker, logger, epoch=fold, global_step=global_step,
                    phase=f"fold_{fold}/test", dataset=dataset, max_batches=max_batches,
                    amp=AMP_DISABLED, num_experts=num_experts, max_tsne_samples=max_tsne,
                )
                fold_test_metrics.append(dict(fold_test_evaluation.scalar_metrics()))

            # E5: under `grouped_cv` there is no held-out test split -- the folds
            # ARE the evaluation. Each fold's final model scores its own
            # photograph-disjoint held-out half, and the predictions are
            # concatenated below into one out-of-fold set covering every crop and
            # every class. Collected after the epoch loop so the model is this
            # fold's finished one, and only on the fold's OWN validation indices,
            # which the fold never trained on.
            if split_protocol == "grouped_cv" and context.is_main:
                _, _, oof_accumulator, _ = run_epoch(
                    encoder, model, criterion, val_loader,
                    device, tracker, logger, epoch=fold, global_step=global_step,
                    phase=f"fold_{fold}/out_of_fold", dataset=dataset,
                    # NOT `max_batches`. Full coverage of the fold IS the
                    # protocol -- every crop held out exactly once -- so a
                    # truncated pass would leave the assembly with fewer
                    # predictions than indices. `max_batches` is a smoke knob
                    # and honouring it here would turn the smoke run's failure
                    # mode from "fast" into "incoherent".
                    max_batches=None,
                    amp=AMP_DISABLED, num_experts=num_experts, max_tsne_samples=max_tsne,
                )
                out_of_fold_parts.append((val_indices, oof_accumulator))

            logger.info("Fold %s complete.", fold)
            barrier(context)

        if interrupted:
            tracker.log_event(
                "training_interrupted",
                {"global_step": global_step, "duration_seconds": time.perf_counter() - started},
            )
            return

        # -------------------------------------------------------------- test
        if best_state is None:
            raise RuntimeError("Training finished without producing a best checkpoint.")

        # Everything from here is evaluation and reporting against parameters
        # every rank already holds identically, so it runs once. The barrier at
        # the end keeps the other ranks from tearing down the process group --
        # and, under NCCL, their share of the CUDA context -- while rank 0 is
        # still profiling and writing.
        test_evaluation: HierarchicalEvaluation | None = None
        test_accumulator: EpochAccumulator | None = None
        efficiency = None
        final_path = str(output_dir / "hierarchical_moe_final.pth")

        if context.is_main:
            if out_of_fold_parts:
                # E5. `grouped_cv` has no held-out split, so the reported
                # evaluation is the concatenation of the folds' own
                # photograph-disjoint held-out halves: every crop scored exactly
                # once, by a model that never trained on it, and all 27 classes
                # present. Each fold contributed a DIFFERENT model, which is the
                # honest price of covering the whole dataset -- so this is a
                # protocol-level estimate of the recipe, not a single shipped
                # model's test score, and `summary.json` says so.
                test_evaluation, test_accumulator = merge_out_of_fold(
                    out_of_fold_parts, dataset, len(dataset)
                )
                log_evaluation_artifacts(
                    tracker, test_evaluation, test_accumulator, dataset, 1, "out_of_fold", logger
                )
                logger.info(
                    "Out-of-fold (%d folds, %d crops, all %d classes) | seed_acc=%.4f "
                    "sub_acc=%.4f sub_f1_macro=%.4f kl_alignment=%.4f",
                    len(out_of_fold_parts), len(test_accumulator.sub_true),
                    len(sub_names),
                    test_evaluation.seed_type.get("accuracy", float("nan")),
                    test_evaluation.sub_variety.get("accuracy", float("nan")),
                    test_evaluation.sub_variety.get("f1_macro", float("nan")),
                    test_evaluation.alignment.overall,
                )
            elif len(test_indices) > 0:
                logger.info("Evaluating best checkpoint (fold %s, epoch %s) on %s held-out samples.",
                            best_state["fold"], best_state["epoch"], len(test_indices))
                _, test_evaluation, test_accumulator, _ = run_epoch(
                    best_state["encoder"], best_state["model"], best_state["criterion"],
                    make_loader(test_indices, shuffle=False), device, tracker, logger,
                    epoch=1, global_step=global_step, phase="test", dataset=dataset,
                    max_batches=max_batches, amp=AMP_DISABLED,
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
                best_state["scheduler"], best_state["epoch"], best_state["fold"], dataset,
                include_optimizer,
            )
            tracker.log_artifact(final_path, name="hierarchical_moe_final", artifact_type="model")

            # ---------------------------------------------------- efficiency
            # Latency and throughput are reported for the deployed single-device
            # path, so this is measured on one rank whatever the training
            # topology was -- a "latency" that depended on the GPU count of the
            # training job would not be a number anyone could act on.
            efficiency = profile_run(
                best_state["encoder"], best_state["model"], eval_dataset, cfg, device, logger,
                sample_indices=test_indices if len(test_indices) else None,
            )
            if efficiency is not None:
                tracker.log_metrics(efficiency.as_metrics(), step=0)

            # -------------------------------------------------- run artifacts
            summary_path = write_run_summary(
                cfg=cfg,
                output_dir=output_dir,
                model=best_state["model"],
                criterion=best_state["criterion"],
                evaluation=test_evaluation,
                accumulator=test_accumulator,
                dataset=dataset,
                efficiency=efficiency,
                history=history,
                checkpoint_path=final_path,
                split_report=split_report,
                fold_metrics=aggregate_fold_metrics(fold_test_metrics),
                logger=logger,
                distributed=context.as_dict(),
            )
            tracker.log_event("run_summary", {"path": summary_path})
        barrier(context)

        total_seconds = time.perf_counter() - started
        tracker.log_event("training_complete", {"duration_seconds": total_seconds, "checkpoint": final_path})
        logger.info("Finetuning complete in %.2fs. Final checkpoint: %s", total_seconds, final_path)

    except Exception:
        logger.exception("Hierarchical MoE finetuning failed.")
        tracker.log_event("exception", {"stage": "moe_finetuning", "rank": context.rank})
        raise
    finally:
        if guard is not None:
            guard.restore()
        tracker.log_event("training_end", {"duration_seconds": time.perf_counter() - started})
        tracker.close()
        # No barrier: this also runs when one rank has raised, and a barrier
        # there would replace the traceback with a collective timeout.
        shutdown_distributed(context)


def aggregate_fold_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """``{metric: {mean, std, min, max, folds}}`` across cross-validation folds."""
    if len(fold_metrics) < 2:
        return {}
    keys = set(fold_metrics[0])
    for entry in fold_metrics[1:]:
        keys &= set(entry)

    aggregated: dict[str, dict[str, float]] = {}
    for key in sorted(keys):
        values = [float(entry[key]) for entry in fold_metrics if entry[key] == entry[key]]
        if not values:
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1) if len(values) > 1 else 0.0
        aggregated[key] = {
            "mean": mean,
            "std": variance**0.5,
            "min": min(values),
            "max": max(values),
            "folds": float(len(values)),
        }
    return aggregated


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
    criterion: CombinedHierarchicalLoss | None = None,
    split_report: dict[str, Any] | None = None,
    fold_metrics: dict[str, dict[str, float]] | None = None,
    distributed: dict[str, Any] | None = None,
) -> str:
    """Persist ``summary.json`` and ``test_predictions.npz`` for cross-run reporting.

    Written even when there is no held-out split, so a run always leaves a
    machine-readable trace; the metrics block is simply empty in that case.

    ``loss_flags`` and ``split`` are recorded alongside the architectural flags
    because without them two runs that differ only in the objective or only in
    the split protocol are byte-identical in ``summary.json``, and only the
    variant *name* separates them.
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
                sub_logits=accumulator.stacked_logits(),
                tokens_per_sample=accumulator.tokens_per_sample,
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
        loss_flags=criterion.loss_flags() if criterion is not None else {},
        split=dict(split_report or {}),
        fold_metrics=dict(fold_metrics or {}),
        runtime={
            "distributed": dict(distributed or {}),
            "amp": str(OmegaConf.select(cfg, "experiment.training.amp", default="auto")),
            "compile": OmegaConf.select(cfg, "experiment.training.compile.enabled", default=None),
            "deterministic": OmegaConf.select(
                cfg, "experiment.training.deterministic", default=None
            ),
        },
        config={
            "backbone": str(OmegaConf.select(cfg, "model.backbone.name", default="")),
            "head": str(OmegaConf.select(cfg, "model.head.name", default="")),
            "embed_dim": OmegaConf.select(cfg, "model.head.embed_dim", default=None),
            "num_experts": OmegaConf.select(cfg, "model.head.num_experts", default=None),
            "top_k": OmegaConf.select(cfg, "model.head.top_k", default=None),
            "token_mode": OmegaConf.select(cfg, "model.head.token_mode", default=None),
            # Which trunk stage the encoder reads. `final` is what every
            # published number was produced under; `stage3` reads `layers.2`,
            # which the stage-1 audit measured as the better frozen readout. It
            # belongs here rather than only in the log because the two are not
            # comparable and a table row must say which one it is.
            "feature_stage": OmegaConf.select(
                cfg, "model.backbone.feature_stage", default="final"
            ),
            "split_protocol": OmegaConf.select(
                cfg, "experiment.training.split_protocol", default="grouped"
            ),
            "epochs": OmegaConf.select(cfg, "experiment.training.epochs", default=None),
            "num_folds": OmegaConf.select(cfg, "experiment.training.num_folds", default=None),
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
