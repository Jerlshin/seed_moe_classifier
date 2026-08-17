"""Stage-1 evaluation: what did DINO self-distillation actually learn?

    python main.py eval-pretrain
    python -m src.trainers.pretrain_eval experiment=eval_pretrain_representation

Stage 1 produces an encoder, not a classifier. There is no accuracy to report and
no held-out loss that means anything -- DINO's loss is defined against a teacher
that moved during training, so it is not comparable across runs, let alone across
checkpoints of one run. This module is the instrument set that *is* appropriate,
and it is deliberately built around three questions that fail independently:

1. **Did the representation collapse?** Label-free geometry: RankMe, the
   participation ratio, the explained-variance curve, dead channels, and
   alignment/uniformity over the same multi-crop augmentation the objective was
   trained on. None of this can be inflated by a readout, and a run can pass the
   loss curve and fail here.

2. **Is the representation discriminative, and how usefully?** A linear probe and
   a parameter-free weighted k-NN over frozen features, on the **grouped**
   (source-photograph-disjoint) split stage 2 uses -- plus the crop-level split,
   because the delta between them is the size of the near-duplicate leakage and is
   itself a result. Then per-class and per-seed-type breakdowns, a low-shot curve,
   calibration, and layer-wise probes that say *which* stage of the trunk holds
   the discriminative signal.

3. **Did the structure emerge without labels?** K-means and DINO's own 2,048-way
   prototype argmax scored against the taxonomy (NMI, AMI, ARI, purity, Hungarian
   cluster accuracy), and the class-centroid similarity matrix, which is where the
   *hierarchy* either is or is not visible.

Controls, and why these and not others
--------------------------------------

Every number above is meaningless in isolation: a frozen ImageNet SwinV2-Small
already scores well on many fine-grained tasks. The comparisons this repository's
design already supports are therefore run alongside the primary encoder:

``imagenet_init``
    The identical architecture at its ImageNet-1k initialisation -- i.e. the
    encoder stage 1 *started from*. The difference is the entire contribution of
    in-domain self-distillation, and it is the stage-1 analogue of
    ``experiment=control_imagenet_frozen``.

``random_init``
    The same architecture untrained. Fixes the floor: a surprising amount of a
    fine-grained probe score is available from a random convolutional prior plus a
    linear readout, and without this row a mediocre encoder looks informative.

``dino_epoch25`` / ``dino_epoch50``
    The permanently-kept milestone encoders. "Did 100 epochs earn their cost over
    25?" is answerable only because those files still exist, which is the reason
    ``experiment.training.save_epochs`` writes them.

What this module does **not** claim
-----------------------------------

Stage-1 pretraining was label-free but saw the whole image set, so a probe's test
split is unseen *photographs* rather than images the encoder never encountered.
That is the standard SSL protocol (DINO linear-probes ImageNet-train after
pretraining on ImageNet-train), but it bounds the claim: these numbers estimate
in-domain readout quality, not transfer to a new acquisition session. The grouped
split is what keeps the *readout* honest; nothing available on this dataset makes
the *encoder* honest in that stronger sense, and ``summary.json`` records the
caveat next to the number rather than leaving it to a caption.

Artifacts (all under ``experiment.evaluation.save_path``)
---------------------------------------------------------

``summary.json``         :class:`~src.utils.evaluation.RunSummary`, so
                         ``scripts/generate_plots.py`` and the cross-run table
                         read this evaluation with no special case.
``metrics.json``         Every measurement, nested by encoder and analysis.
``tables/*.csv``         Flat tables: encoder comparison, per-class, low-shot,
                         layer-wise, milestones, prototype usage.
``figures/*.png``        Publication figures at 300 dpi.
``features/*.npz``       Cached features, so any figure can be redrawn without a
                         forward pass.
``test_predictions.npz`` The probe's held-out predictions in the repository's
                         standard prediction-dump format.
``provenance.json``      Checkpoint SHA-256s, split manifest hash, library
                         versions, git commit, resolved config.
``split_manifest.npz``   The exact indices used, written by the stage-2 helper.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.datasets.dataset import HierarchicalSeedDataset, get_finetune_dataset
from src.datasets.transforms import get_dino_transforms, get_supervised_transforms
from src.models.backbones.swinv2_dino import DINOHead
from src.models.builder import BackboneFeatureExtractor
from src.trainers.moe_finetune import (
    save_split_manifest,
    seed_everything,
    split_dataset,
    stratification_labels,
)
from src.utils.efficiency import profile_model
from src.utils.evaluation import (
    RunSummary,
    parse_pretrain_dynamics,
    save_test_predictions,
)
from src.utils.metrics import (
    classification_metrics,
    confusion_matrices,
    evaluate_hierarchical,
    expected_calibration_error,
    fit_temperature,
    misclassification_rates,
    per_class_metrics,
    per_seed_type_breakdown,
    roc_auc_ovr,
    tsne_projection,
)
from src.utils.representation import (
    alignment_uniformity,
    augmentation_consistency,
    centroid_similarity_matrix,
    class_separability,
    feature_statistics,
    kmeans_report,
    knn_classifier,
    l2_normalize,
    linear_cka,
    linear_probe,
    low_shot_probe,
    prototype_report,
    retrieval_report,
    select_regularisation,
    spectral_report,
)
from src.utils.training import (
    ExperimentTracker,
    configure_backend,
    describe_accelerator,
    select_device,
    setup_experiment_logger,
    snapshot_run_configuration,
)
from src.utils.visualization import (
    plot_confusion_matrix,
    plot_curves_with_bands,
    plot_distribution_panels,
    plot_embedding_comparison,
    plot_grouped_bars,
    plot_metric_heatmap,
    plot_misclassification_rates,
    plot_per_class_bars,
    plot_reliability_diagram,
    plot_retrieval_examples,
    plot_series_panels,
    plot_similarity_matrix,
    plot_spectrum,
    plot_tsne,
    save_figure,
    use_publication_style,
)

#: The four SwinV2 stages, named as timm names them. Layer-wise probing walks
#: these; ``pooled`` is the trunk's final normalised output, which is what stage 2
#: consumes.
STAGE_KEYS = ("stage1", "stage2", "stage3", "stage4")

# numpy's Accelerate (macOS) BLAS raises spurious `divide by zero encountered in
# matmul` on *any* large `@`, including `np.random.randn(300, 8) @
# np.random.randn(8, 300)`. Verified on this platform; it is a backend artifact,
# not a property of the operands. This evaluation is almost entirely large
# matmuls, so leaving it unfiltered fills the log with warnings that look like a
# numerical failure and are not. Scoped to this module and to that exact message,
# so a genuine invalid value anywhere else still surfaces.
warnings.filterwarnings(
    "ignore",
    message=r".*(divide by zero|overflow|invalid value) encountered in matmul.*",
    category=RuntimeWarning,
)


# --------------------------------------------------------------------- specs


@dataclass
class EncoderSpec:
    """One encoder to evaluate: where its weights come from and what it controls for."""

    label: str
    checkpoint: str | None = None
    pretrained: bool = False
    role: str = "control"
    description: str = ""
    capture_stages: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "checkpoint": self.checkpoint,
            "pretrained": self.pretrained,
            "role": self.role,
            "description": self.description,
        }


@dataclass
class FeatureBundle:
    """Cached features for one encoder over the whole dataset, in dataset order."""

    spec: EncoderSpec
    pooled: np.ndarray
    stages: dict[str, np.ndarray] = field(default_factory=dict)
    checkpoint_sha256: str = ""
    load_report: dict[str, Any] = field(default_factory=dict)
    parameters: int = 0
    extraction_seconds: float = 0.0

    @property
    def label(self) -> str:
        return self.spec.label


# ----------------------------------------------------------------- utilities


def sha256_of(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file, or ``""`` when it does not exist.

    Recorded for every checkpoint the evaluation reads. A results table that
    names ``dino_backbone_epoch_0100.pth`` is not reproducible -- that filename
    is reused by every run -- whereas a digest identifies the bytes.
    """
    file = Path(path)
    if not file.exists():
        return ""
    digest = hashlib.sha256()
    with file.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_checkpoints(left: str | Path, right: str | Path) -> tuple[str, str]:
    """Compare two bare state-dict checkpoints, and say *how* they compare.

    Returns ``(verdict, detail)`` with ``verdict`` in
    ``{"identical_bytes", "identical_weights", "different", "unreadable"}``.

    A digest comparison alone is not enough, and getting this wrong produces a
    false alarm that is worse than no check. Measured on this repository's own
    artifacts: ``dino_backbone_epoch_0100.pth`` and the published
    ``dinov2_swinv2_pretrained.pth`` have **different SHA-256s and all 423 tensors
    bit-identical** -- the milestone goes through ``atomic_save`` and the handoff
    through ``torch.save`` plus a copy, and torch's zip container does not
    serialise byte-identically across those paths. So the digest is the fast path
    and a tensor comparison is the authority.
    """
    left_path, right_path = Path(left), Path(right)
    if not (left_path.exists() and right_path.exists()):
        return "unreadable", "one of the files does not exist"
    if sha256_of(left_path) == sha256_of(right_path):
        return "identical_bytes", "same SHA-256"

    try:
        first = torch.load(left_path, map_location="cpu", weights_only=False)
        second = torch.load(right_path, map_location="cpu", weights_only=False)
    except Exception as error:  # noqa: BLE001 - any load failure is "cannot compare"
        return "unreadable", f"{type(error).__name__} while loading"
    if not isinstance(first, Mapping) or not isinstance(second, Mapping):
        return "unreadable", "not a state dict"
    if set(first) != set(second):
        missing = len(set(first) ^ set(second))
        return "different", f"{missing} keys differ between the two"

    worst = 0.0
    for key, value in first.items():
        other = second[key]
        if not hasattr(value, "shape") or value.shape != other.shape:
            return "different", f"{key} has shape {tuple(value.shape)} vs {tuple(other.shape)}"
        worst = max(worst, float((value.float() - other.float()).abs().max()))
    if worst == 0.0:
        return (
            "identical_weights",
            f"all {len(first)} tensors bit-identical; only the serialised container differs",
        )
    return "different", f"largest absolute weight difference {worst:.3e}"


def git_commit(root: Path) -> str:
    """Current commit, with a ``-dirty`` suffix when the tree has changes."""
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return f"{revision}-dirty" if status else revision


def json_safe(value: Any) -> Any:
    """Recursively convert numpy/torch scalars and arrays into JSON-safe values."""
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and value != value:
        return None
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], float_format: str = "{:.6f}") -> str:
    """Write ``rows`` as a CSV whose columns are the union of every row's keys."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    def cell(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return "" if value != value else float_format.format(value)
        return str(value)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: cell(row.get(column)) for column in columns})
    return str(path)


def materialise_teacher_encoder(
    resume_path: str | Path | None,
    destination: str | Path,
    search_dir: str | Path | None,
    logger,
) -> Path | None:
    """Write the EMA **teacher** trunk out of a resume checkpoint as a bare state dict.

    Why this exists at all. DINO's teacher is an exponential moving average of the
    student, and Caron et al. evaluate and *ship* the teacher: the averaging is a
    variance reduction, and on k-NN it is normally the better of the two by a small
    but consistent margin. This pipeline publishes the **student**
    (`dino_pretrained_backbone.pth`, and every `dino_backbone_epoch_*.pth`), because
    `save_teacher_in_checkpoints: false` keeps the teacher out of the interval and
    milestone artifacts to save disk.

    The teacher does survive in the *resume* checkpoint, which carries it by
    necessity. So the comparison is answerable from what is already on disk, and it
    matters: if the teacher wins, stage 2 should load it and the change is free.

    Returns the destination path, or ``None`` when there is no resume checkpoint to
    read. Writing is skipped when the destination already exists, so this is cheap
    on re-runs.
    """
    from src.utils.training import atomic_save

    output = Path(destination)
    if output.exists():
        logger.info("Teacher trunk already extracted at %s.", output)
        return output

    candidate = Path(resume_path) if resume_path else None
    if candidate is None and search_dir is not None:
        # Newest resume checkpoint wins; step numbers are zero-padded so the
        # lexical order is the numeric one.
        found = sorted(Path(search_dir).glob("dino_resume_step*.pth"))
        candidate = found[-1] if found else None
    if candidate is None or not candidate.exists():
        logger.info(
            "No resume checkpoint found, so the EMA teacher cannot be evaluated. "
            "Set experiment.evaluation.teacher_from_resume to a dino_resume_step*.pth."
        )
        return None

    payload = torch.load(candidate, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "teacher_backbone" not in payload:
        logger.warning(
            "%s carries no 'teacher_backbone' (save_teacher_in_checkpoints was false for the "
            "interval artifacts and this file is not a resume checkpoint); skipping the teacher.",
            candidate,
        )
        return None

    atomic_save(payload["teacher_backbone"], output)
    logger.info(
        "Extracted the EMA teacher trunk from %s (epoch %s) to %s. DINO ships the teacher; this "
        "pipeline publishes the student, so the two are now both evaluable.",
        candidate.name, payload.get("epoch", "?"), output,
    )
    return output


def resolve_encoder_specs(cfg: DictConfig, logger) -> list[EncoderSpec]:
    """Build the encoder list from config, dropping entries whose file is missing.

    A missing milestone is a warning rather than an error: the primary checkpoint
    is the point of the run, and an evaluation that refuses to start because the
    epoch-25 encoder was pruned would be worse than one that reports four rows and
    says why.
    """
    entries = OmegaConf.select(cfg, "experiment.evaluation.encoders", default=None) or []
    specs: list[EncoderSpec] = []
    stage_labels = set(
        OmegaConf.select(cfg, "experiment.evaluation.layerwise.encoders", default=None) or []
    )
    for entry in entries:
        item = dict(entry)
        checkpoint = item.get("checkpoint")
        checkpoint = str(checkpoint) if checkpoint else None
        if checkpoint and not Path(checkpoint).exists():
            logger.warning(
                "Skipping encoder %r: checkpoint not found at %s.", item.get("label"), checkpoint
            )
            continue
        label = str(item["label"])
        specs.append(
            EncoderSpec(
                label=label,
                checkpoint=checkpoint,
                pretrained=bool(item.get("pretrained", False)),
                role=str(item.get("role", "control")),
                description=str(item.get("description", "")),
                capture_stages=label in stage_labels,
            )
        )
    if not specs:
        raise FileNotFoundError(
            "No evaluable encoders. Set experiment.evaluation.encoders, and check that "
            "the stage-1 checkpoints exist under experiment.evaluation.pretrain_run_dir."
        )
    primaries = [spec for spec in specs if spec.role == "primary"]
    if len(primaries) != 1:
        raise ValueError(
            f"Exactly one encoder must carry role='primary', found {len(primaries)}: "
            f"{[spec.label for spec in primaries]}"
        )
    return specs


def load_cached_features(
    path: Path,
    checkpoint_sha256: str,
    expected_samples: int,
    need_stages: bool,
    logger,
) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
    """Reuse a cached feature dump, but only when it is provably the same features.

    Three conditions, all of which have to hold: the cache records the same
    checkpoint digest, it has the same number of rows as the dataset now being
    evaluated, and it carries the per-stage arrays if this run needs them. Any
    mismatch returns ``None`` and the features are recomputed.

    Checking the digest rather than the path is the point. ``dino_backbone_epoch_0100.pth``
    is a filename every pretraining run reuses, so a cache keyed on the path would
    silently serve the previous run's features after a retrain -- and every number
    in the report would be about weights that no longer exist.
    """
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=True) as archive:
            stored = str(archive["checkpoint_sha256"]) if "checkpoint_sha256" in archive else ""
            pooled = archive["pooled"]
            stages = {
                key.removeprefix("stage_"): archive[key]
                for key in archive.files
                if key.startswith("stage_")
            }
    except Exception as error:  # noqa: BLE001 - see below
        # Deliberately broad. A truncated or corrupt cache raises whatever the
        # zip/pickle layer happens to raise (`BadZipFile`, `UnpicklingError`,
        # `EOFError`, ...), and every one of them is recoverable by definition:
        # the fallback is to recompute the features, which is always correct. A
        # narrow except here turns a half-written file from a killed run into a
        # crash on the next launch.
        logger.warning("Ignoring unreadable feature cache at %s (%s).", path, type(error).__name__)
        return None

    if stored != checkpoint_sha256:
        logger.info("Feature cache at %s is for a different checkpoint; recomputing.", path.name)
        return None
    if pooled.shape[0] != expected_samples:
        logger.info(
            "Feature cache at %s has %d rows against %d samples; recomputing.",
            path.name, pooled.shape[0], expected_samples,
        )
        return None
    if need_stages and not stages:
        logger.info("Feature cache at %s has no per-stage arrays; recomputing.", path.name)
        return None

    logger.info(
        "Reusing cached features for %s (%d x %d, digest %s). Delete %s to force a "
        "recomputation.",
        path.stem, pooled.shape[0], pooled.shape[1], (stored or "none")[:16], path,
    )
    return pooled.astype(np.float32), {key: value.astype(np.float32) for key, value in stages.items()}


def subsample_dataset(
    dataset: HierarchicalSeedDataset,
    max_samples: int,
    seed: int,
    logger,
) -> bool:
    """Shrink ``dataset`` in place to at most ``max_samples`` crops, class-balanced.

    Exists for the smoke path: a 2-minute end-to-end run that exercises every
    analysis and every figure is worth having, and the alternative -- a synthetic
    dataset -- would not exercise the real checkpoint loading. Balanced per
    sub-variety and group-aware only incidentally, so a subsampled run's *numbers*
    are not comparable to a full one; the log says so.
    """
    if max_samples <= 0 or max_samples >= len(dataset.samples):
        return False
    rng = np.random.default_rng(int(seed))
    labels = np.array([sub for _, _, sub in dataset.samples], dtype=np.int64)
    per_class = max(int(max_samples // max(len(dataset.subvariety_to_idx), 1)), 2)
    keep: list[int] = []
    for label in range(len(dataset.subvariety_to_idx)):
        pool = np.flatnonzero(labels == label)
        if pool.size == 0:
            continue
        keep.extend(rng.choice(pool, size=int(min(per_class, pool.size)), replace=False).tolist())
    keep = sorted(keep)
    dataset.samples = [dataset.samples[index] for index in keep]
    logger.warning(
        "SUBSAMPLED to %d crops (%d per sub-variety) because "
        "experiment.evaluation.max_samples=%d. This is a plumbing check: the numbers it "
        "produces are not comparable to a full run, and the feature cache is disabled so a "
        "smoke run cannot overwrite a full one's features.",
        len(dataset.samples), per_class, max_samples,
    )
    return True


def discover_events_path(output_root: Path, logger) -> Path | None:
    """Find the stage-1 run's ``events.jsonl`` under ``outputs/hydra/``.

    Hydra names run directories by timestamp, so the path to a finished stage-1
    run is not knowable from a config file. Rather than make the user paste it,
    this picks the candidate with the most epoch records that also carries the
    stage-1 collapse diagnostics -- i.e. the longest DINO run in the tree, which is
    the one whose checkpoints are being evaluated. An explicit
    ``experiment.evaluation.dynamics.events_path`` always wins.
    """
    candidates = sorted((output_root / "hydra").glob("*/*/events.jsonl"))
    best: tuple[int, Path | None] = (0, None)
    for candidate in candidates:
        try:
            dynamics = parse_pretrain_dynamics(candidate)
        except OSError:
            continue
        if "train/teacher_entropy" not in dynamics.step_series:
            continue
        epochs = len(dynamics.epoch_series.get("epoch/loss", ([], []))[1])
        if epochs > best[0]:
            best = (epochs, candidate)
    if best[1] is not None:
        logger.info(
            "Stage-1 dynamics discovered at %s (%d epochs logged). Override with "
            "experiment.evaluation.dynamics.events_path.",
            best[1], best[0],
        )
    return best[1]


def build_backbone(
    spec: EncoderSpec,
    cfg: DictConfig,
    device: torch.device,
    seed: int,
) -> BackboneFeatureExtractor:
    """Instantiate one frozen SwinV2 trunk for ``spec``.

    ``random_init`` is seeded here rather than relying on the process-wide seed:
    the floor control has to be the *same* random encoder every run, or the row it
    contributes moves between reports for no reason anybody can see.
    """
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    extractor = BackboneFeatureExtractor(
        model_name=str(cfg.model.backbone.name),
        checkpoint_path=spec.checkpoint,
        pretrained=bool(spec.pretrained),
        dynamic_img_size=bool(OmegaConf.select(cfg, "model.backbone.dynamic_img_size", default=True)),
        strict=False,
        freeze=True,
        drop_path_rate=0.0,
    )
    return extractor.to(device).eval()


# ------------------------------------------------------------ feature dumps


def _pool_grid(features: torch.Tensor) -> torch.Tensor:
    """Mean-pool timm's ``[B, H, W, C]`` stage output to ``[B, C]``."""
    if features.ndim == 4:
        return features.mean(dim=(1, 2))
    if features.ndim == 3:
        return features.mean(dim=1)
    return features


@torch.no_grad()
def extract_features(
    extractor: BackboneFeatureExtractor,
    loader: DataLoader,
    device: torch.device,
    capture_stages: bool = False,
    logger=None,
    label: str = "",
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """One deterministic pass over ``loader``, returning pooled and per-stage features.

    Per-stage features are captured with forward hooks *in the same pass*, because
    a second pass over 9,357 images at 25.6 GFLOPs/view costs as much as the first
    and produces bit-identical inputs. The hooks pool immediately, so peak memory
    is ``batch x channels`` rather than ``batch x 64 x channels``.

    Runs in fp32 with no autocast. Every metric downstream is a geometry
    measurement on these vectors, and an autocast dtype that changes the fifth
    decimal of a cosine similarity would change a silhouette score for reasons
    unrelated to the encoder.
    """
    captured: dict[str, list[torch.Tensor]] = {key: [] for key in STAGE_KEYS}
    handles = []
    if capture_stages:
        layers = getattr(extractor.backbone, "layers", None)
        if layers is None:
            raise AttributeError("Backbone exposes no `layers`; cannot capture stage features.")

        def make_hook(key: str):
            def hook(_module, _inputs, output):
                captured[key].append(_pool_grid(output).detach().float().cpu())

            return hook

        for index, key in enumerate(STAGE_KEYS[: len(layers)]):
            handles.append(layers[index].register_forward_hook(make_hook(key)))

    pooled: list[torch.Tensor] = []
    started = time.perf_counter()
    try:
        for batch_index, batch in enumerate(loader):
            images = batch[0].to(device, non_blocking=True)
            pooled.append(extractor(images).detach().float().cpu())
            if logger is not None and batch_index % 50 == 0:
                logger.info(
                    "  [%s] batch %d/%d", label, batch_index + 1, len(loader)
                )
    finally:
        for handle in handles:
            handle.remove()

    features = torch.cat(pooled).numpy().astype(np.float32)
    stages = {
        key: torch.cat(values).numpy().astype(np.float32)
        for key, values in captured.items()
        if values
    }
    if logger is not None:
        logger.info(
            "  [%s] %d x %d features in %.1fs%s",
            label,
            features.shape[0],
            features.shape[1],
            time.perf_counter() - started,
            f", stages {[f'{k}:{v.shape[1]}' for k, v in stages.items()]}" if stages else "",
        )
    return features, stages


@torch.no_grad()
def extract_view_pairs(
    extractor: BackboneFeatureExtractor,
    dataset: HierarchicalSeedDataset,
    indices: np.ndarray,
    cfg: DictConfig,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Features of the two DINO **global** views for each of ``indices``.

    The augmentation is the stage-1 pipeline, not the stage-2 evaluation
    transform, and that is the whole point: alignment measures whether the
    invariance the objective actually asked for was achieved, so it has to be
    measured over the distribution the objective was trained on. Using a milder
    transform would report an easier number for every encoder and would flatter
    the untrained controls most.

    Seeded per call so the same crops are drawn for every encoder; without that,
    alignment differences between encoders would be partly differences in which
    crops each one happened to see. Both RNGs are seeded, because the blur and
    solarize steps draw from Python's ``random`` while the crops draw from torch.
    """
    import random

    from PIL import Image

    augmentation = get_dino_transforms(
        image_size=int(cfg.data.image_size),
        local_crop_size=int(cfg.data.local_crop_size),
        augmentation_cfg=OmegaConf.to_container(cfg.data.augmentation, resolve=True),
        local_crops_number=0,
        return_original=False,
        output_uint8=False,
        defer_local_upsample=False,
    )

    left: list[torch.Tensor] = []
    right: list[torch.Tensor] = []
    torch.manual_seed(int(seed))
    random.seed(int(seed))

    for start in range(0, indices.size, batch_size):
        block = indices[start : start + batch_size]
        first, second = [], []
        for index in block:
            path = dataset.samples[int(index)][0]
            with Image.open(path) as handle:
                image = handle.convert("RGB")
            views = augmentation(image)[1]
            first.append(views[0])
            second.append(views[1])
        for buffer, views in ((left, first), (right, second)):
            stacked = torch.stack(views).to(device)
            buffer.append(extractor(stacked).detach().float().cpu())

    return (
        torch.cat(left).numpy().astype(np.float32),
        torch.cat(right).numpy().astype(np.float32),
    )


# ---------------------------------------------------------------- analyses


def probe_task(
    features: np.ndarray,
    labels: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    test_indices: np.ndarray,
    num_classes: int,
    grid: Sequence[float],
    max_iterations: int,
    seed: int,
    run_cv: bool,
) -> dict[str, Any]:
    """Linear probe for one label level: tune, cross-validate, then score the test split.

    The order matters and is the reason this is one function rather than three
    calls at the call site. ``C`` is selected on fold 1's validation half only; the
    cross-validation then *reuses* that ``C`` across all folds, and the test score
    comes from a refit on every non-test sample. Selecting ``C`` per fold would
    make the CV spread an underestimate, and selecting it on the test split would
    make the headline number meaningless.
    """
    train_indices, val_indices = folds[0]
    selection = select_regularisation(
        features[train_indices],
        labels[train_indices],
        features[val_indices],
        labels[val_indices],
        num_classes=num_classes,
        grid=grid,
        max_iterations=max_iterations,
        seed=seed,
    )
    best_c = selection["best_C"]

    fit_indices = np.unique(np.concatenate([train_indices, val_indices]))
    held_out = linear_probe(
        features[fit_indices],
        labels[fit_indices],
        features[test_indices],
        labels[test_indices],
        num_classes=num_classes,
        regularisation=best_c,
        max_iterations=max_iterations,
        seed=seed,
    )

    report: dict[str, Any] = {
        "regularisation_selected": best_c,
        "regularisation_sweep": selection["sweep"],
        "test_accuracy": held_out["accuracy"],
        "test_f1_macro": held_out["f1_macro"],
        "train_accuracy": held_out["train_accuracy"],
        "train_size": held_out["train_size"],
        "test_size": held_out["test_size"],
        "weight_norm": held_out["weight_norm"],
        "generalisation_gap": held_out["train_accuracy"] - held_out["accuracy"],
    }
    report["_predictions"] = held_out["predictions"]
    report["_probabilities"] = held_out["probabilities"]
    report["_logits"] = held_out["logits"]
    report["_fit_indices"] = fit_indices

    if run_cv and len(folds) > 1:
        scores, macro = [], []
        for fold_train, fold_val in folds:
            outcome = linear_probe(
                features[fold_train],
                labels[fold_train],
                features[fold_val],
                labels[fold_val],
                num_classes=num_classes,
                regularisation=best_c,
                max_iterations=max_iterations,
                seed=seed,
            )
            scores.append(outcome["accuracy"])
            macro.append(outcome["f1_macro"])
        report.update(
            {
                "cv_folds": len(scores),
                "cv_accuracy_mean": float(np.mean(scores)),
                "cv_accuracy_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                "cv_f1_macro_mean": float(np.mean(macro)),
                "cv_f1_macro_std": float(np.std(macro, ddof=1)) if len(macro) > 1 else 0.0,
                "cv_accuracy_folds": [float(value) for value in scores],
            }
        )
    return report


def grouped_cv_readout(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    num_classes: int,
    num_folds: int,
    regularisation: float,
    max_iterations: int,
    knn_k: int,
    knn_temperature: float,
    seed: int,
    min_groups_per_class: int = 2,
) -> dict[str, Any]:
    """Out-of-fold readout over the **whole** dataset, group-disjoint and stratified.

    Why this exists, and why it is the number to headline.

    The single held-out split stage 2 uses is a ``GroupShuffleSplit``: it takes 20 %
    of the 81 source photographs with no stratification. On this dataset that is
    not a detail -- most sub-varieties have crops from only ~3 photographs, so a
    20 % draw of photographs leaves **13 of the 27 sub-varieties entirely absent
    from the test side**. Accuracy on that split is a 27-way classifier scored on
    14 classes, and its macro-F1 is mechanically capped near 14/27 because the
    absent classes contribute zero. Both facts are properties of the *split*, not
    of the encoder.

    ``StratifiedGroupKFold`` over every crop fixes both without weakening the
    protocol: each fold's held-out half is still photograph-disjoint from its
    training half, every crop is held out exactly once, and every class appears.
    Concatenating the folds gives one out-of-fold prediction per crop -- 9,357 of
    them, all 27 classes present, no leakage -- which is what the confusion matrix
    and the per-class table should be computed from.

    The single-split number is still reported, because it is the one directly
    comparable to a stage-2 run. They answer different questions and the report
    carries both.

    ``min_groups_per_class`` adds the correction that this dataset makes
    unavoidable. A class whose crops all come from **one** source photograph is
    never in the training half of the fold that holds it out, so it cannot be
    predicted and scores exactly zero — by construction, for a provenance reason,
    identically for every encoder. Five of the 27 sub-varieties are in that
    position and they are 14.8 % of the crops. The report therefore carries both
    the all-class figure and the figure restricted to classes a photograph-disjoint
    protocol can actually test, because quoting only the first understates every
    encoder by the same ~11 points and quoting only the second hides that a fifth
    of the taxonomy is unmeasurable.
    """
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedGroupKFold

    splitter = StratifiedGroupKFold(n_splits=int(num_folds), shuffle=True, random_state=int(seed))
    predictions = np.full(labels.shape, -1, dtype=np.int64)
    knn_predictions = np.full(labels.shape, -1, dtype=np.int64)
    probabilities = np.zeros((labels.size, num_classes), dtype=np.float32)
    fold_accuracy: list[float] = []
    fold_f1: list[float] = []
    fold_knn: list[float] = []
    fold_classes: list[int] = []

    for train_index, val_index in splitter.split(np.zeros(labels.size), labels, groups):
        probe = linear_probe(
            features[train_index], labels[train_index],
            features[val_index], labels[val_index],
            num_classes=num_classes,
            regularisation=regularisation,
            max_iterations=max_iterations,
            seed=seed,
        )
        predictions[val_index] = probe["predictions"]
        probabilities[val_index] = probe["probabilities"]
        fold_accuracy.append(probe["accuracy"])
        fold_f1.append(probe["f1_macro"])
        fold_classes.append(int(np.unique(labels[val_index]).size))

        neighbours = knn_classifier(
            features[train_index], labels[train_index],
            features[val_index], labels[val_index],
            num_classes=num_classes,
            k=int(knn_k),
            temperature=float(knn_temperature),
        )
        knn_predictions[val_index] = neighbours["predictions"]
        fold_knn.append(neighbours["accuracy"])

    def spread(values: Sequence[float]) -> tuple[float, float]:
        return (
            float(np.mean(values)),
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        )

    accuracy_mean, accuracy_std = spread(fold_accuracy)
    f1_mean, f1_std = spread(fold_f1)
    knn_mean, knn_std = spread(fold_knn)
    covered = predictions >= 0

    # Classes a photograph-disjoint protocol can test at all.
    groups_per_class = {
        int(label): np.unique(groups[labels == label]).size for label in np.unique(labels)
    }
    testable = {label for label, count in groups_per_class.items() if count >= min_groups_per_class}
    testable_mask = covered & np.isin(labels, list(testable))
    restricted: dict[str, Any] = {
        "min_groups_per_class": float(min_groups_per_class),
        "testable_classes": float(len(testable)),
        "untestable_classes": float(len(groups_per_class) - len(testable)),
        "testable_crops": float(int(testable_mask.sum())),
        "testable_crop_fraction": float(testable_mask.mean()),
    }
    if testable_mask.any():
        restricted["out_of_fold_accuracy"] = float(
            accuracy_score(labels[testable_mask], predictions[testable_mask])
        )
        restricted["out_of_fold_f1_macro"] = float(
            f1_score(
                labels[testable_mask], predictions[testable_mask], average="macro", zero_division=0
            )
        )
        restricted["out_of_fold_knn_accuracy"] = float(
            accuracy_score(labels[testable_mask], knn_predictions[testable_mask])
        )

    return {
        "testable_classes_only": restricted,
        "protocol": f"StratifiedGroupKFold({num_folds}) over every crop, out-of-fold predictions",
        "folds": len(fold_accuracy),
        "classes_present_per_fold": fold_classes,
        "probe_accuracy_mean": accuracy_mean,
        "probe_accuracy_std": accuracy_std,
        "probe_f1_macro_mean": f1_mean,
        "probe_f1_macro_std": f1_std,
        "probe_accuracy_folds": [float(value) for value in fold_accuracy],
        "knn_accuracy_mean": knn_mean,
        "knn_accuracy_std": knn_std,
        # Pooled out-of-fold scores. Not a mean of fold scores: this is one metric
        # over all 9,357 predictions, which is what the confusion matrix shows.
        "out_of_fold_accuracy": float(accuracy_score(labels[covered], predictions[covered])),
        "out_of_fold_f1_macro": float(
            f1_score(labels[covered], predictions[covered], average="macro", zero_division=0)
        ),
        "out_of_fold_knn_accuracy": float(
            accuracy_score(labels[covered], knn_predictions[covered])
        ),
        "coverage": float(covered.mean()),
        "_predictions": predictions,
        "_knn_predictions": knn_predictions,
        "_probabilities": probabilities,
    }


def knn_task(
    features: np.ndarray,
    labels: np.ndarray,
    fit_indices: np.ndarray,
    test_indices: np.ndarray,
    num_classes: int,
    k_values: Sequence[int],
    temperature: float,
) -> dict[str, Any]:
    """Weighted cosine k-NN at each ``k``, keeping the best ``k``'s predictions."""
    per_k: dict[str, Any] = {}
    best: tuple[float, int, dict[str, Any]] = (-1.0, 0, {})
    for k in k_values:
        outcome = knn_classifier(
            features[fit_indices],
            labels[fit_indices],
            features[test_indices],
            labels[test_indices],
            num_classes=num_classes,
            k=int(k),
            temperature=float(temperature),
        )
        per_k[f"k{int(k)}"] = {
            "accuracy": outcome["accuracy"],
            "f1_macro": outcome["f1_macro"],
        }
        if outcome["accuracy"] > best[0]:
            best = (outcome["accuracy"], int(k), outcome)
    return {
        "per_k": per_k,
        "best_k": best[1],
        "best_accuracy": best[0],
        "best_f1_macro": best[2].get("f1_macro", float("nan")),
        "_predictions": best[2].get("predictions"),
        "_scores": best[2].get("scores"),
    }


def evaluate_encoder(
    bundle: FeatureBundle,
    seed_labels: np.ndarray,
    sub_labels: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    test_indices: np.ndarray,
    settings: Mapping[str, Any],
    seed_names: Sequence[str],
    sub_names: Sequence[str],
    groups: np.ndarray,
    logger,
) -> dict[str, Any]:
    """Every per-encoder measurement, for one encoder's cached features."""
    features = bundle.pooled
    num_seed_types = len(seed_names)
    num_sub = len(sub_names)
    fit_indices = np.unique(np.concatenate([folds[0][0], folds[0][1]]))
    rng = np.random.default_rng(int(settings["seed"]))

    logger.info("Analysing %s (%s)", bundle.label, bundle.spec.role)
    report: dict[str, Any] = {
        "spec": bundle.spec.as_dict(),
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "load_report": bundle.load_report,
        "backbone_parameters": bundle.parameters,
        "extraction_seconds": bundle.extraction_seconds,
        "feature_dim": int(features.shape[1]),
        "num_samples": int(features.shape[0]),
    }

    # ------------------------------------------------- label-free geometry
    report["spectrum"] = spectral_report(features).as_dict()
    report["feature_statistics"] = feature_statistics(features)

    cap = int(settings["geometry_max_samples"])
    subset = (
        rng.choice(features.shape[0], size=cap, replace=False)
        if features.shape[0] > cap
        else np.arange(features.shape[0])
    )
    report["separability_full"] = {
        "sub_variety": class_separability(features[subset], sub_labels[subset], sub_names),
        "seed_type": class_separability(features[subset], seed_labels[subset], seed_names),
        "samples": int(subset.size),
    }
    report["separability_test_split"] = {
        "sub_variety": class_separability(features[test_indices], sub_labels[test_indices], sub_names),
        "seed_type": class_separability(features[test_indices], seed_labels[test_indices], seed_names),
        "samples": int(test_indices.size),
    }

    # --------------------------------------------------------- label readout
    report["knn"] = {
        "sub_variety": knn_task(
            features, sub_labels, fit_indices, test_indices, num_sub,
            settings["knn_k_values"], settings["knn_temperature"],
        ),
        "seed_type": knn_task(
            features, seed_labels, fit_indices, test_indices, num_seed_types,
            settings["knn_k_values"], settings["knn_temperature"],
        ),
    }
    run_cv = bundle.label in settings["cv_encoders"]
    report["probe"] = {
        "sub_variety": probe_task(
            features, sub_labels, folds, test_indices, num_sub,
            settings["probe_grid"], settings["probe_max_iterations"], settings["seed"], run_cv,
        ),
        "seed_type": probe_task(
            features, seed_labels, folds, test_indices, num_seed_types,
            settings["probe_grid"], settings["probe_max_iterations"], settings["seed"], run_cv,
        ),
    }

    # The primary protocol for anything per-class: out-of-fold over every crop, so
    # all 27 classes are scored. See `grouped_cv_readout` for why the single
    # held-out split cannot carry that weight on this dataset.
    if settings["grouped_cv_enabled"]:
        report["grouped_cv"] = {
            "sub_variety": grouped_cv_readout(
                features, sub_labels, groups, num_sub,
                num_folds=settings["grouped_cv_folds"],
                regularisation=report["probe"]["sub_variety"]["regularisation_selected"],
                max_iterations=settings["probe_max_iterations"],
                knn_k=report["knn"]["sub_variety"]["best_k"],
                knn_temperature=settings["knn_temperature"],
                seed=settings["seed"],
            ),
            "seed_type": grouped_cv_readout(
                features, seed_labels, groups, num_seed_types,
                num_folds=settings["grouped_cv_folds"],
                regularisation=report["probe"]["seed_type"]["regularisation_selected"],
                max_iterations=settings["probe_max_iterations"],
                knn_k=report["knn"]["seed_type"]["best_k"],
                knn_temperature=settings["knn_temperature"],
                seed=settings["seed"],
            ),
        }
        sub_cv = report["grouped_cv"]["sub_variety"]
        restricted = sub_cv["testable_classes_only"]
        logger.info(
            "  out-of-fold | 27-way probe %.4f (macro F1 %.4f, fold SD %.4f), k-NN %.4f, "
            "4-way probe %.4f | restricted to the %d of %d classes with >= 2 source "
            "photographs (%.0f%% of crops): probe %.4f, macro F1 %.4f, k-NN %.4f",
            sub_cv["out_of_fold_accuracy"], sub_cv["out_of_fold_f1_macro"],
            sub_cv["probe_accuracy_std"], sub_cv["out_of_fold_knn_accuracy"],
            report["grouped_cv"]["seed_type"]["out_of_fold_accuracy"],
            int(restricted["testable_classes"]),
            int(restricted["testable_classes"] + restricted["untestable_classes"]),
            100.0 * restricted["testable_crop_fraction"],
            restricted.get("out_of_fold_accuracy", float("nan")),
            restricted.get("out_of_fold_f1_macro", float("nan")),
            restricted.get("out_of_fold_knn_accuracy", float("nan")),
        )

    # --------------------------------------------- structure without labels
    if settings["clustering_enabled"]:
        report["kmeans"] = {
            "k_sub_variety": kmeans_report(
                features, sub_labels, num_sub, seed=int(settings["seed"])
            ),
            "k_seed_type": kmeans_report(
                features, seed_labels, num_seed_types, seed=int(settings["seed"])
            ),
        }

    if settings["retrieval_enabled"]:
        report["retrieval"] = {
            "ungrouped": retrieval_report(features, sub_labels, settings["retrieval_k_values"]),
            "grouped_excluded": retrieval_report(
                features, sub_labels, settings["retrieval_k_values"], groups=groups
            ),
        }

    return report


def hierarchical_report(
    seed_true: np.ndarray,
    seed_pred: np.ndarray,
    sub_true: np.ndarray,
    sub_pred: np.ndarray,
    sub_scores: np.ndarray,
    subvariety_to_seed_type: Sequence[int],
    seed_names: Sequence[str],
    sub_names: Sequence[str],
) -> dict[str, Any]:
    """Score the probe's two independent readouts as a hierarchy.

    Reuses :func:`~src.utils.metrics.evaluate_hierarchical` unchanged, with
    ``num_experts=1`` because there is no MoE at stage 1. The interesting column
    is ``kl_alignment``: two *independently fitted* linear readouts agreeing on
    the parent seed type is a property of the representation, and it is the
    quantity stage 2's KL term is built on the assumption of.
    """
    evaluation = evaluate_hierarchical(
        seed_true=seed_true,
        seed_pred=seed_pred,
        sub_true=sub_true,
        sub_pred=sub_pred,
        subvariety_to_seed_type=subvariety_to_seed_type,
        num_seed_types=len(seed_names),
        num_sub_varieties=len(sub_names),
        seed_type_names=seed_names,
        sub_variety_names=sub_names,
        sub_scores=sub_scores,
        top_k_indices=None,
        num_experts=1,
    )
    return {"evaluation": evaluation, "metrics": evaluation.scalar_metrics()}


# ------------------------------------------------------------------ figures


def dynamics_panels(dynamics, summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Panel specs for the stage-1 training-dynamics figure."""
    epochs, losses = dynamics.series("epoch/loss")
    steps, entropy = dynamics.series("train/teacher_entropy")
    floor = summary.get("teacher_entropy_floor")
    ceiling = summary.get("teacher_entropy_ceiling")

    panels: list[dict[str, Any]] = [
        {
            "title": "DINO self-distillation loss",
            "xlabel": "Epoch",
            "ylabel": "Loss (nats)",
            "series": {"epoch mean": (epochs, losses)},
        },
        {
            "title": "Learning-rate schedule (warmup then cosine)",
            "xlabel": "Epoch",
            "ylabel": "Learning rate",
            "series": {"lr": dynamics.series("epoch/lr")},
        },
        {
            "title": "Teacher temperature and weight-decay schedules",
            "xlabel": "Optimizer step",
            "ylabel": "Value",
            "series": {
                "teacher temp": dynamics.series("train/teacher_temp"),
                "weight decay": dynamics.series("train/weight_decay"),
            },
        },
        {
            "title": "Teacher entropy against its structural bounds",
            "xlabel": "Optimizer step",
            "ylabel": "H (nats)",
            "series": {"teacher entropy": (steps, entropy)},
            "hlines": [
                {"y": floor, "label": f"floor log(K/B) = {floor:.2f}", "color": "#D55E00"},
                {"y": ceiling, "label": f"ceiling log K = {ceiling:.2f}", "color": "#009E73"},
            ]
            if floor and ceiling
            else [],
        },
        {
            # Autoscaled, not pinned to [0, 1]: both series sit at ~1.0 for the
            # whole run, and the *absence* of any downward drift is the result. A
            # fixed axis would render two flat lines at the top of an empty panel
            # and hide the variation that is there.
            "title": "Soft prototype occupancy (no collapse of the head)",
            "xlabel": "Optimizer step",
            "ylabel": "Fraction of K",
            "series": {
                "utilisation": dynamics.series("train/prototype_utilization"),
                "perplexity / K": (
                    dynamics.series("train/prototype_perplexity")[0],
                    [
                        value / max(summary.get("prototypes", 2048), 1)
                        for value in dynamics.series("train/prototype_perplexity")[1]
                    ],
                ),
            },
        },
        {
            "title": "KoLeo uniformity term and gradient norm",
            "xlabel": "Optimizer step",
            "ylabel": "Value",
            "series": {
                "KoLeo": dynamics.series("train/koleo"),
                "gradient norm": dynamics.series("train/gradient_norm"),
            },
        },
        {
            "title": "Throughput",
            "xlabel": "Epoch",
            "ylabel": "Images / second",
            "series": {"images/s": dynamics.series("epoch/images_per_second")},
        },
        {
            # Separate panel, because a stall fraction in [0, 1] plotted beside a
            # throughput of ~17 img/s is a flat line on the x axis.
            "title": "Share of wall clock blocked on the dataloader",
            "xlabel": "Epoch",
            "ylabel": "Data-wait fraction",
            "series": {"data-wait": dynamics.series("epoch/data_wait_fraction")},
            "ylim": (0.0, 1.02),
        },
    ]
    return panels


def build_figures(
    output_dir: Path,
    bundles: Mapping[str, FeatureBundle],
    reports: Mapping[str, Any],
    extras: Mapping[str, Any],
    seed_labels: np.ndarray,
    sub_labels: np.ndarray,
    seed_names: Sequence[str],
    sub_names: Sequence[str],
    subvariety_to_seed_type: Sequence[int],
    settings: Mapping[str, Any],
    dynamics,
    dynamics_summary: Mapping[str, Any],
    logger,
) -> dict[str, str]:
    """Draw every publication figure and return ``{name: path}``."""
    use_publication_style()
    figures_dir = output_dir / "figures"
    dpi = int(settings["figure_dpi"])
    written: dict[str, str] = {}
    primary = settings["primary_label"]
    control = settings.get("control_label")
    order = list(bundles)

    def write(name: str, figure) -> None:
        written[name] = save_figure(figure, figures_dir / f"{name}.png", dpi=dpi)
        logger.info("  figure %s", written[name])

    # 01 -- stage-1 training dynamics from the event stream.
    if dynamics is not None and dynamics.series("epoch/loss")[1]:
        write(
            "fig01_pretrain_dynamics",
            plot_series_panels(
                dynamics_panels(dynamics, dynamics_summary),
                columns=3,
                suptitle="Stage-1 DINO training dynamics (recovered from events.jsonl)",
            ),
        )

    # 02 -- the readout comparison across encoders.
    def readout(label: str, path: Sequence[str], default: float = float("nan")) -> float:
        node: Any = reports[label]
        for key in path:
            if not isinstance(node, Mapping) or key not in node:
                return default
            node = node[key]
        return float(node)

    # 02 -- the headline readout comparison, on the out-of-fold protocol that
    # covers every crop and every class.
    has_cv = all("grouped_cv" in reports[label] for label in order)
    if has_cv:
        write(
            "fig02_readout_comparison",
            plot_grouped_bars(
                categories=order,
                groups={
                    "probe, 27-way (all classes)": [
                        readout(l, ("grouped_cv", "sub_variety", "out_of_fold_accuracy")) for l in order
                    ],
                    "probe, 27-way (22 testable classes)": [
                        readout(
                            l,
                            ("grouped_cv", "sub_variety", "testable_classes_only", "out_of_fold_accuracy"),
                        )
                        for l in order
                    ],
                    "k-NN, 27-way (all classes)": [
                        readout(l, ("grouped_cv", "sub_variety", "out_of_fold_knn_accuracy"))
                        for l in order
                    ],
                    "probe, 4-way": [
                        readout(l, ("grouped_cv", "seed_type", "out_of_fold_accuracy")) for l in order
                    ],
                    "k-NN, 4-way": [
                        readout(l, ("grouped_cv", "seed_type", "out_of_fold_knn_accuracy"))
                        for l in order
                    ],
                },
                errors={
                    "probe, 27-way (all classes)": [
                        readout(l, ("grouped_cv", "sub_variety", "probe_accuracy_std")) for l in order
                    ],
                    "probe, 4-way": [
                        readout(l, ("grouped_cv", "seed_type", "probe_accuracy_std")) for l in order
                    ],
                },
                ylabel="Out-of-fold accuracy",
                title=(
                    "Frozen-feature readouts, 5-fold photograph-disjoint cross-validation "
                    "(every crop held out once)"
                ),
                reference=1.0 / len(sub_names),
                reference_label=f"27-way chance ({1 / len(sub_names):.3f})",
                ylim=(0.0, 1.12),
            ),
        )

    # 02b -- the same encoders on the single held-out split stage 2 uses. Kept as
    # a separate figure rather than a second series: the two protocols score
    # different numbers of classes, so putting them in one axis would invite the
    # reader to subtract them.
    write(
        "fig02b_readout_single_split",
        plot_grouped_bars(
            categories=order,
            groups={
                "linear probe (27-way)": [readout(l, ("probe", "sub_variety", "test_accuracy")) for l in order],
                "k-NN (27-way)": [readout(l, ("knn", "sub_variety", "best_accuracy")) for l in order],
                "linear probe (4-way)": [readout(l, ("probe", "seed_type", "test_accuracy")) for l in order],
                "k-NN (4-way)": [readout(l, ("knn", "seed_type", "best_accuracy")) for l in order],
            },
            ylabel="Held-out accuracy",
            title=(
                "Single held-out grouped split (the stage-2 protocol; "
                f"{settings.get('single_split_classes', '?')} of {len(sub_names)} classes present)"
            ),
            reference=1.0 / len(sub_names),
            reference_label=f"27-way chance ({1 / len(sub_names):.3f})",
            ylim=(0.0, 1.12),
        ),
    )

    # 03 -- label-free geometry across encoders.
    write(
        "fig03_geometry_comparison",
        plot_grouped_bars(
            categories=order,
            groups={
                "RankMe / d": [readout(l, ("spectrum", "rankme_normalized")) for l in order],
                # Reported with its sign. A negative cosine silhouette over 27
                # fine-grained classes is the normal outcome and is itself the
                # finding: the classes are linearly separable long before they are
                # separated *clusters*.
                "silhouette (27-way, cosine)": [
                    readout(l, ("separability_full", "sub_variety", "silhouette_cosine"))
                    for l in order
                ],
                "k-means NMI (k=27)": [readout(l, ("kmeans", "k_sub_variety", "nmi")) for l in order],
                "k-means cluster accuracy": [
                    readout(l, ("kmeans", "k_sub_variety", "cluster_accuracy")) for l in order
                ],
            },
            ylabel="Value",
            title="Label-free geometry and unsupervised structure",
        ),
    )

    # 04 -- spectra.
    write(
        "fig04_spectrum",
        plot_spectrum(
            spectra={l: reports[l]["spectrum"]["singular_values"] for l in order},
            cumulative={l: reports[l]["spectrum"]["explained_variance_ratio"] for l in order},
            annotations={
                l: f"RankMe {reports[l]['spectrum']['rankme']:.0f}, "
                f"{reports[l]['spectrum']['dims_for_95_variance']} dims @95 %"
                for l in order
            },
            title="Feature covariance spectrum (768-D pooled trunk output)",
        ),
    )

    # 05/06 -- t-SNE of the primary encoder.
    projections = extras.get("tsne", {})
    if primary in projections:
        projection, indices = projections[primary]
        write(
            "fig05_tsne_primary_seed_type",
            plot_tsne(
                projection,
                seed_labels[indices],
                seed_names,
                title=f"t-SNE, {primary} frozen features, by seed type",
                annotate_clusters=True,
            ),
        )
        write(
            "fig06_tsne_primary_sub_variety",
            plot_tsne(
                projection,
                sub_labels[indices],
                sub_names,
                title=f"t-SNE, {primary} frozen features, by sub-variety",
                annotate_clusters=True,
            ),
        )
    if len(projections) > 1:
        write(
            "fig07_tsne_encoder_comparison",
            plot_embedding_comparison(
                {
                    label: (projection, sub_labels[indices])
                    for label, (projection, indices) in projections.items()
                },
                class_names=sub_names,
                title="Same images, same t-SNE settings, different encoders (coloured by sub-variety)",
            ),
        )

    # 08/09/10 -- the primary probe's error structure.
    hierarchy = extras.get("hierarchy")
    if hierarchy is not None:
        evaluation = hierarchy["evaluation"]
        write(
            "fig08_probe_confusion_sub_variety",
            plot_confusion_matrix(
                evaluation.sub_confusion,
                sub_names,
                title=f"Linear-probe confusion, 27 sub-varieties ({primary}, row-normalised)",
                annotate_threshold=0,
            ),
        )
        write(
            "fig09_probe_confusion_seed_type",
            plot_confusion_matrix(
                evaluation.seed_confusion,
                seed_names,
                title=f"Linear-probe confusion, 4 seed types ({primary}, row-normalised)",
            ),
        )
        write("fig10_probe_metric_heatmap", plot_metric_heatmap(evaluation.per_class_sub))
        write(
            "fig11_probe_misclassification",
            plot_misclassification_rates(
                evaluation.sub_misclassification,
                title=f"Per-sub-variety misclassification rate, linear probe on {primary}",
            ),
        )
        write(
            "fig12_probe_per_class_f1",
            plot_per_class_bars(
                names=[entry.name for entry in evaluation.per_class_sub],
                values=[entry.f1 for entry in evaluation.per_class_sub],
                group_of=list(subvariety_to_seed_type),
                group_names=list(seed_names),
                supports=[entry.support for entry in evaluation.per_class_sub],
                xlabel="F1",
                title=f"Per-sub-variety F1, linear probe on {primary} (colour = seed type)",
                reference=1.0 / len(sub_names),
            ),
        )

    # 13 -- class-centroid similarity, hierarchy blocks marked.
    similarity = extras.get("centroid_similarity")
    if similarity is not None:
        parents = np.asarray(list(subvariety_to_seed_type))
        boundaries = [int(np.searchsorted(parents, value + 1)) for value in range(len(seed_names) - 1)]
        write(
            "fig13_class_centroid_similarity",
            plot_similarity_matrix(
                similarity,
                sub_names,
                block_boundaries=boundaries,
                block_labels=list(seed_names),
                title=f"Sub-variety centroid cosine similarity, {primary} (diagonal masked)",
                # Sequential, scaled to the off-diagonal range. A diverging map
                # centred on zero spends half its range on similarities that do not
                # occur here -- every centroid pair is positively correlated -- and
                # the block structure is exactly what gets flattened.
                colormap="magma",
                symmetric_scale=False,
            ),
        )

    # 14 -- low-shot curve.
    low_shot = extras.get("low_shot", {})
    if low_shot:
        write(
            "fig14_low_shot_curve",
            plot_curves_with_bands(
                {
                    label: (
                        [row["shots"] for row in rows],
                        [row["accuracy_mean"] for row in rows],
                        [row["accuracy_std"] for row in rows],
                    )
                    for label, rows in low_shot.items()
                },
                xlabel="Labelled images per sub-variety",
                ylabel="Held-out 27-way accuracy",
                title="Low-shot linear probe (mean +- SD over repeated label draws)",
                logx=True,
                reference=1.0 / len(sub_names),
                reference_label=f"chance ({1 / len(sub_names):.3f})",
            ),
        )

    # 15 -- layer-wise probe.
    layerwise = extras.get("layerwise", {})
    if layerwise:
        stages = list(next(iter(layerwise.values())))
        write(
            "fig15_layerwise_probe",
            plot_grouped_bars(
                categories=stages,
                groups={label: [values[stage] for stage in stages] for label, values in layerwise.items()},
                ylabel="Held-out 27-way probe accuracy",
                title="Which stage of the trunk carries the discriminative signal?",
                reference=1.0 / len(sub_names),
            ),
        )

    # 16 -- milestone progression.
    milestones = extras.get("milestones", [])
    if len(milestones) > 1:
        write(
            "fig16_milestone_progression",
            plot_series_panels(
                [
                    {
                        "title": "Readout accuracy vs pretraining length",
                        "xlabel": "Stage-1 epochs",
                        "ylabel": "Held-out accuracy",
                        "series": {
                            "linear probe (27-way)": (
                                [row["epoch"] for row in milestones],
                                [row["probe_sub_accuracy"] for row in milestones],
                            ),
                            "k-NN (27-way)": (
                                [row["epoch"] for row in milestones],
                                [row["knn_sub_accuracy"] for row in milestones],
                            ),
                        },
                    },
                    {
                        "title": "Geometry vs pretraining length",
                        "xlabel": "Stage-1 epochs",
                        "ylabel": "Value",
                        "series": {
                            "silhouette (27-way)": (
                                [row["epoch"] for row in milestones],
                                [row["silhouette"] for row in milestones],
                            ),
                            "k-means NMI": (
                                [row["epoch"] for row in milestones],
                                [row["kmeans_nmi"] for row in milestones],
                            ),
                        },
                    },
                    {
                        "title": "RankMe vs pretraining length",
                        "xlabel": "Stage-1 epochs",
                        "ylabel": "RankMe",
                        "series": {
                            "RankMe": (
                                [row["epoch"] for row in milestones],
                                [row["rankme"] for row in milestones],
                            )
                        },
                    },
                ],
                columns=3,
                suptitle="Did the later epochs earn their cost? (milestone encoders, identical protocol)",
            ),
        )

    # 17 -- augmentation invariance, one panel per encoder.
    invariance = extras.get("invariance", {})
    if invariance:
        panels = {}
        for label, payload in invariance.items():
            populations = {
                key: payload.get(source)
                for key, source in (
                    ("same image, two global views", "_positive_cosines"),
                    ("same sub-variety, different image", "_same_class_cosines"),
                    ("different sub-variety", "_different_class_cosines"),
                )
                if payload.get(source)
            }
            if populations:
                panels[label] = populations
        if panels:
            write(
                "fig17_augmentation_invariance",
                plot_distribution_panels(
                    panels,
                    xlabel="Cosine similarity",
                    title=(
                        "Invariance to the stage-1 augmentation against class structure "
                        "(compare gaps within a panel, not cosines across panels)"
                    ),
                ),
            )

    # 18 -- prototype usage.
    prototypes = extras.get("prototypes")
    if prototypes is not None:
        shares = np.sort(np.asarray(prototypes["usage_shares"], dtype=np.float64))[::-1]
        write(
            "fig18_prototype_usage",
            plot_series_panels(
                [
                    {
                        "title": "Usage per prototype, sorted",
                        "xlabel": "Prototype rank",
                        "ylabel": "Share of dataset argmax",
                        "series": {"share": (np.arange(1, shares.size + 1), shares)},
                        # Both axes log: the distribution is a short head over
                        # thousands of exact zeros, and a linear rank axis spends
                        # 90 % of its width on the zeros.
                        "yscale": "log",
                        "xscale": "log",
                    },
                    {
                        "title": "Cumulative usage",
                        "xlabel": "Prototype rank",
                        "ylabel": "Cumulative share",
                        "series": {"cumulative": (np.arange(1, shares.size + 1), np.cumsum(shares))},
                        "xscale": "log",
                        "ylim": (0.0, 1.02),
                    },
                ],
                columns=2,
                suptitle=(
                    f"DINO's own {int(prototypes['num_prototypes'])}-way prototype head at epoch 100: "
                    f"{int(prototypes['active_prototypes'])} prototypes win an argmax, "
                    f"NMI vs sub-variety {prototypes['nmi_vs_labels']:.3f}, "
                    f"purity {prototypes['purity_vs_labels']:.3f}"
                ),
            ),
        )

    # 19 -- probe calibration.
    calibration = extras.get("calibration")
    if calibration is not None:
        write(
            "fig19_probe_reliability",
            plot_reliability_diagram(
                calibration["confidences"],
                calibration["correct"],
                num_bins=15,
                title=f"Linear-probe calibration on {primary}",
                ece=calibration["ece"],
            ),
        )

    # 20 -- qualitative retrieval.
    retrieval = extras.get("retrieval_examples")
    if retrieval is not None:
        write(
            "fig20_retrieval_examples",
            plot_retrieval_examples(
                retrieval["queries"],
                retrieval["neighbours"],
                retrieval["query_labels"],
                retrieval["correct"],
                title=(
                    f"Nearest neighbours in {primary} feature space, "
                    "excluding crops of the same source photograph"
                ),
            ),
        )

    # 21 -- leakage: grouped vs crop-level split.
    leakage = extras.get("leakage")
    if leakage:
        write(
            "fig21_split_protocol_delta",
            plot_grouped_bars(
                categories=["linear probe (27-way)", "k-NN (27-way)", "linear probe (4-way)"],
                groups={
                    "grouped (photograph-disjoint)": [
                        leakage["grouped"]["probe_sub"],
                        leakage["grouped"]["knn_sub"],
                        leakage["grouped"]["probe_seed"],
                    ],
                    "stratified (crop-level)": [
                        leakage["stratified"]["probe_sub"],
                        leakage["stratified"]["knn_sub"],
                        leakage["stratified"]["probe_seed"],
                    ],
                },
                ylabel="Held-out accuracy",
                title=f"Near-duplicate leakage: the same encoder ({primary}) under both split protocols",
                ylim=(0.0, 1.08),
            ),
        )

    # 22 -- CKA against the ImageNet initialisation.
    cka = extras.get("cka", {})
    if cka:
        write(
            "fig22_cka_vs_imagenet",
            plot_grouped_bars(
                categories=list(cka),
                groups={f"linear CKA vs {control}": [cka[label] for label in cka]},
                ylabel="Linear CKA",
                title="How far did self-distillation move the representation?",
                ylim=(0.0, 1.08),
            ),
        )

    return written


# --------------------------------------------------------------------- main


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    output_dir = str(cfg.tracking.output_dir)
    logger = setup_experiment_logger(
        log_dir=output_dir,
        name="seed_moe.pretrain_eval",
        level=cfg.tracking.log_level,
        console=bool(cfg.tracking.console),
        structured_jsonl=cfg.tracking.structured_jsonl,
    )
    tracker = ExperimentTracker(cfg, logger, enabled=True)
    started = time.perf_counter()

    try:
        logger.info("========== Stage-1 representation evaluation: %s ==========", cfg.experiment.name)
        seed_everything(int(cfg.seed))
        snapshot_paths = snapshot_run_configuration(cfg, output_dir)
        logger.info("Saved run configuration snapshots (%d files).", len(snapshot_paths))

        device = select_device(str(cfg.device))
        accelerator = describe_accelerator(device)
        logger.info("Accelerator | %s", accelerator.summary_line())
        backend = configure_backend(
            device,
            allow_tf32=False,
            deterministic=True,
            matmul_precision="highest",
            logger=logger,
        )
        logger.info(
            "Backend pinned for reproducibility: TF32 off, deterministic on, matmul=highest. "
            "Every metric here is a geometry measurement on float32 features, so a kernel that "
            "changes the fifth decimal of a cosine would change a silhouette score for reasons "
            "that have nothing to do with the encoder."
        )
        tracker.log_event("backend", backend)

        evaluation_cfg = cfg.experiment.evaluation
        save_path = Path(str(evaluation_cfg.save_path))
        save_path.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------ dataset
        transform = get_supervised_transforms(
            image_size=int(cfg.data.image_size),
            train=False,
            normalize_mean=cfg.data.augmentation.normalize_mean,
            normalize_std=cfg.data.augmentation.normalize_std,
        )
        dataset = get_finetune_dataset(
            str(cfg.data.root_path),
            transform=transform,
            save_csv_path=None,
            include_path=False,
        )
        subsampled = subsample_dataset(
            dataset,
            int(OmegaConf.select(cfg, "experiment.evaluation.max_samples", default=0) or 0),
            seed=int(cfg.seed),
            logger=logger,
        )
        seed_names, sub_names = dataset.get_ordered_class_names()
        subvariety_to_seed_type = dataset.get_subvariety_to_seed_type()
        seed_labels = np.array([seed for _, seed, _ in dataset.samples], dtype=np.int64)
        sub_labels = stratification_labels(dataset)
        groups = dataset.source_groups()
        group_report = dataset.group_report()

        if len(seed_names) != int(cfg.data.num_seed_types) or len(sub_names) != int(
            cfg.data.num_sub_varieties
        ):
            raise ValueError(
                f"Discovered {len(seed_names)} seed types and {len(sub_names)} sub-varieties, but the "
                f"config declares {cfg.data.num_seed_types} and {cfg.data.num_sub_varieties}. Label "
                "indices come from sorted directory names, so a mismatch means every index -- and "
                "therefore every checkpoint -- refers to a different class."
            )

        logger.info(
            "Dataset | %d crops, %d seed types, %d sub-varieties, %d source photographs "
            "(mean %.1f crops/photograph).",
            len(dataset), len(seed_names), len(sub_names),
            group_report["num_source_groups"], group_report["mean_crops_per_source"],
        )
        singletons = group_report["single_group_sub_varieties"]
        if singletons:
            logger.warning(
                "%d sub-varieties have crops from exactly one photograph (%s). No grouped split can "
                "put them on both sides, so their readout scores measure within-photograph "
                "generalisation whatever the protocol says.",
                len(singletons), ", ".join(singletons),
            )

        # -------------------------------------------------------------- split
        protocol = str(OmegaConf.select(cfg, "experiment.evaluation.split.protocol", default="grouped"))
        test_size = float(OmegaConf.select(cfg, "experiment.evaluation.split.test_size", default=0.2))
        num_folds = int(OmegaConf.select(cfg, "experiment.evaluation.split.num_folds", default=5))
        folds, test_indices, split_report = split_dataset(
            dataset, test_size=test_size, num_folds=num_folds, seed=int(cfg.seed), protocol=protocol
        )
        manifest = save_split_manifest(save_path, folds, test_indices, dataset, protocol=protocol)
        fit_indices = np.unique(np.concatenate([folds[0][0], folds[0][1]]))
        logger.info(
            "Split | protocol=%s, %d fit / %d test crops, %d folds, %d shared source groups "
            "(%.1f %% of test leaked).",
            protocol, fit_indices.size, test_indices.size, len(folds),
            split_report["shared_source_groups"], 100.0 * split_report["leaked_test_fraction"],
        )
        for direction in ("train", "test"):
            missing = split_report.get(f"sub_varieties_missing_from_{direction}") or []
            if missing:
                logger.warning(
                    "%d sub-varieties are absent from the %s side of the split (%s). A readout cannot "
                    "predict a class it never saw, so those samples are counted wrong and the "
                    "macro-F1 is bounded below 1 by construction. This is a property of the dataset's "
                    "provenance, not of the encoder.",
                    len(missing), direction, ", ".join(missing),
                )
        tracker.log_event("split_manifest", {"path": manifest, **json_safe(split_report)})

        # ----------------------------------------------------------- encoders
        if bool(OmegaConf.select(cfg, "experiment.evaluation.teacher.enabled", default=True)):
            materialise_teacher_encoder(
                OmegaConf.select(cfg, "experiment.evaluation.teacher.from_resume", default=None),
                OmegaConf.select(
                    cfg,
                    "experiment.evaluation.teacher.destination",
                    default=str(Path(str(evaluation_cfg.pretrain_run_dir)) / "dino_teacher_backbone.pth"),
                ),
                search_dir=str(evaluation_cfg.pretrain_run_dir),
                logger=logger,
            )
        specs = resolve_encoder_specs(cfg, logger)
        primary_label = next(spec.label for spec in specs if spec.role == "primary")
        control_label = next((spec.label for spec in specs if spec.role == "control_imagenet"), None)
        logger.info(
            "Evaluating %d encoders: %s (primary: %s).",
            len(specs), ", ".join(spec.label for spec in specs), primary_label,
        )

        batch_size = int(OmegaConf.select(cfg, "experiment.evaluation.batch_size", default=32))
        num_workers = int(OmegaConf.select(cfg, "experiment.evaluation.num_workers", default=2))
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )

        features_dir = save_path / "features"
        features_dir.mkdir(parents=True, exist_ok=True)
        reuse_cache = (
            bool(OmegaConf.select(cfg, "experiment.evaluation.reuse_cached_features", default=True))
            and not subsampled
        )
        bundles: dict[str, FeatureBundle] = {}
        primary_extractor: BackboneFeatureExtractor | None = None

        for spec in specs:
            digest = sha256_of(spec.checkpoint) if spec.checkpoint else ""
            cache_path = features_dir / f"{spec.label}.npz"
            cached = (
                load_cached_features(cache_path, digest, len(dataset), spec.capture_stages, logger)
                if reuse_cache
                else None
            )
            extractor: BackboneFeatureExtractor | None = None
            started_extraction = time.perf_counter()

            if cached is not None:
                pooled, stages = cached
                elapsed = 0.0
            else:
                logger.info(
                    "Extracting features | %s: %s", spec.label, spec.description or spec.checkpoint
                )
                extractor = build_backbone(spec, cfg, device, seed=int(cfg.seed))
                pooled, stages = extract_features(
                    extractor,
                    loader,
                    device,
                    capture_stages=spec.capture_stages,
                    logger=logger,
                    label=spec.label,
                )
                elapsed = time.perf_counter() - started_extraction
                if not subsampled:
                    np.savez_compressed(
                        cache_path,
                        pooled=pooled,
                        seed_labels=seed_labels,
                        sub_labels=sub_labels,
                        source_groups=groups,
                        checkpoint_sha256=np.array(digest),
                        backbone=np.array(str(cfg.model.backbone.name)),
                        **{f"stage_{key}": value for key, value in stages.items()},
                    )
                if extractor.load_report is not None:
                    report = extractor.load_report
                    logger.info(
                        "  checkpoint keys | %d missing, %d unexpected",
                        len(report["missing_keys"]), len(report["unexpected_keys"]),
                    )
                    if report["missing_keys"] or report["unexpected_keys"]:
                        logger.warning(
                            "  %s loaded with key mismatches -- verify the backbone name matches "
                            "the checkpoint before believing any number below.", spec.label,
                        )

            bundles[spec.label] = FeatureBundle(
                spec=spec,
                pooled=pooled,
                stages=stages,
                checkpoint_sha256=digest,
                load_report=(
                    {key: len(value) for key, value in (extractor.load_report or {}).items()}
                    if extractor is not None
                    else {"cached": 1}
                ),
                parameters=(
                    int(sum(p.numel() for p in extractor.parameters()))
                    if extractor is not None
                    else int(pooled.shape[1])
                ),
                extraction_seconds=elapsed,
            )
            if spec.label == primary_label:
                # Kept alive for the latency sweep, which has to measure the real
                # module rather than the cached array.
                primary_extractor = extractor or build_backbone(spec, cfg, device, seed=int(cfg.seed))
                bundles[spec.label].parameters = int(
                    sum(p.numel() for p in primary_extractor.parameters())
                )
            elif extractor is not None:
                del extractor

        # Is the primary encoder the same bytes stage 2 will load? Checked rather
        # than assumed: `dino_backbone_epoch_0100.pth` and the published
        # `dinov2_swinv2_pretrained.pth` are written by different code paths at
        # different points in the run, and a report about the wrong one of the two
        # would be indistinguishable from a report about the right one.
        handoff = OmegaConf.select(
            cfg, "experiment.evaluation.shared_backbone_path", default=None
        )
        handoff_match: bool | None = None
        handoff_verdict = "not_checked"
        handoff_detail = ""
        if handoff and Path(str(handoff)).exists() and bundles[primary_label].spec.checkpoint:
            handoff_verdict, handoff_detail = compare_checkpoints(
                bundles[primary_label].spec.checkpoint, str(handoff)
            )
            handoff_match = handoff_verdict in {"identical_bytes", "identical_weights"}
            (logger.info if handoff_match else logger.warning)(
                "Stage-2 handoff | %s vs the primary encoder: %s (%s). %s",
                handoff,
                handoff_verdict,
                handoff_detail,
                "Every number below is about the weights stage 2 loads."
                if handoff_match
                else "This evaluation therefore does NOT describe the published handoff.",
            )
        elif handoff:
            logger.info("Stage-2 handoff not present at %s; skipping the comparison.", handoff)

        # ---------------------------------------------------------- analyses
        settings = {
            "seed": int(cfg.seed),
            "geometry_max_samples": int(
                OmegaConf.select(cfg, "experiment.evaluation.geometry_max_samples", default=6000)
            ),
            "knn_k_values": list(
                OmegaConf.select(cfg, "experiment.evaluation.knn.k_values", default=[1, 10, 20])
            ),
            "knn_temperature": float(
                OmegaConf.select(cfg, "experiment.evaluation.knn.temperature", default=0.07)
            ),
            "probe_grid": list(
                OmegaConf.select(
                    cfg,
                    "experiment.evaluation.probe.regularisation_grid",
                    default=[0.01, 0.1, 1.0, 10.0, 100.0],
                )
            ),
            "probe_max_iterations": int(
                OmegaConf.select(cfg, "experiment.evaluation.probe.max_iterations", default=2000)
            ),
            "cv_encoders": list(
                OmegaConf.select(cfg, "experiment.evaluation.probe.cv_encoders", default=[]) or []
            ),
            "clustering_enabled": bool(
                OmegaConf.select(cfg, "experiment.evaluation.clustering.enabled", default=True)
            ),
            "retrieval_enabled": bool(
                OmegaConf.select(cfg, "experiment.evaluation.retrieval.enabled", default=True)
            ),
            "retrieval_k_values": list(
                OmegaConf.select(cfg, "experiment.evaluation.retrieval.k_values", default=[1, 5, 10])
            ),
            "figure_dpi": int(OmegaConf.select(cfg, "experiment.evaluation.figures.dpi", default=300)),
            "primary_label": primary_label,
            "control_label": control_label,
            "grouped_cv_enabled": bool(
                OmegaConf.select(cfg, "experiment.evaluation.grouped_cv.enabled", default=True)
            ),
            "grouped_cv_folds": int(
                OmegaConf.select(cfg, "experiment.evaluation.grouped_cv.num_folds", default=5)
            ),
            "single_split_classes": int(np.unique(sub_labels[test_indices]).size),
        }

        reports: dict[str, Any] = {}
        for label, bundle in bundles.items():
            reports[label] = evaluate_encoder(
                bundle, seed_labels, sub_labels, folds, test_indices, settings,
                seed_names, sub_names, groups, logger,
            )

        extras: dict[str, Any] = {
            "subvariety_to_seed_type": list(subvariety_to_seed_type),
            "primary_per_class_silhouette": reports[primary_label]["separability_full"][
                "sub_variety"
            ].get("per_class_silhouette", {}),
        }
        primary = bundles[primary_label]

        # Hierarchical scoring of the primary probe's two readouts.
        #
        # Computed twice, on purpose. The single held-out split is the one directly
        # comparable to a stage-2 run; the out-of-fold predictions cover every crop
        # and every class, which is what the confusion matrix and the per-class
        # table need (13 of 27 sub-varieties are absent from the single split's
        # test side -- a property of the photograph-level grouping, not of the
        # encoder). The figures use the out-of-fold version when it exists.
        sub_probe = reports[primary_label]["probe"]["sub_variety"]
        seed_probe = reports[primary_label]["probe"]["seed_type"]
        single_split_hierarchy = hierarchical_report(
            seed_true=seed_labels[test_indices],
            seed_pred=seed_probe["_predictions"],
            sub_true=sub_labels[test_indices],
            sub_pred=sub_probe["_predictions"],
            sub_scores=sub_probe["_probabilities"],
            subvariety_to_seed_type=subvariety_to_seed_type,
            seed_names=seed_names,
            sub_names=sub_names,
        )
        grouped_cv = reports[primary_label].get("grouped_cv")
        if grouped_cv is not None:
            covered = grouped_cv["sub_variety"]["_predictions"] >= 0
            hierarchy = hierarchical_report(
                seed_true=seed_labels[covered],
                seed_pred=grouped_cv["seed_type"]["_predictions"][covered],
                sub_true=sub_labels[covered],
                sub_pred=grouped_cv["sub_variety"]["_predictions"][covered],
                sub_scores=grouped_cv["sub_variety"]["_probabilities"][covered],
                subvariety_to_seed_type=subvariety_to_seed_type,
                seed_names=seed_names,
                sub_names=sub_names,
            )
            hierarchy["scope"] = "out_of_fold_all_crops"
        else:
            hierarchy = single_split_hierarchy
            hierarchy["scope"] = "single_held_out_split"
        extras["hierarchy"] = hierarchy
        reports[primary_label]["hierarchical_probe"] = {
            "scope": hierarchy["scope"],
            "kl_alignment_overall": hierarchy["evaluation"].alignment.overall,
            "kl_alignment_per_seed_type": hierarchy["evaluation"].alignment.per_seed_type,
            "per_seed_type_sub_variety": hierarchy["evaluation"].per_seed_type_sub,
            "auc_macro_ovr": hierarchy["evaluation"].sub_variety.get("auc_macro_ovr"),
            "single_split_kl_alignment_overall": single_split_hierarchy[
                "evaluation"
            ].alignment.overall,
        }
        logger.info(
            "Primary probe | single held-out split: 27-way %.4f (macro F1 %.4f, %d of %d classes "
            "present), 4-way %.4f | scored hierarchy on %s, independent-readout agreement %.4f",
            sub_probe["test_accuracy"], sub_probe["test_f1_macro"],
            int(np.unique(sub_labels[test_indices]).size), len(sub_names),
            seed_probe["test_accuracy"], hierarchy["scope"],
            hierarchy["evaluation"].alignment.overall,
        )

        # Calibration of the primary probe.
        #
        # Guo et al.'s protocol needs three disjoint roles: fit the classifier,
        # fit the temperature, score the result. The headline probe above is refit
        # on train+val, which leaves no held-out set for the temperature -- so
        # calibration is measured on a *second* probe fit on the train fold only,
        # with T fitted on the validation fold it never saw. Both predictions come
        # from that one model, which is what makes the before/after comparison
        # mean anything; mixing the train-fold temperature into the train+val
        # model's predictions made the "corrected" ECE worse than the raw one.
        fold_train, validation = folds[0]
        calibration_train = {
            "features": primary.pooled[fold_train],
            "labels": sub_labels[fold_train],
        }
        val_probe = linear_probe(
            calibration_train["features"], calibration_train["labels"],
            primary.pooled[validation], sub_labels[validation],
            num_classes=len(sub_names),
            regularisation=sub_probe["regularisation_selected"],
            max_iterations=settings["probe_max_iterations"],
            seed=int(cfg.seed),
        )
        test_probe = linear_probe(
            calibration_train["features"], calibration_train["labels"],
            primary.pooled[test_indices], sub_labels[test_indices],
            num_classes=len(sub_names),
            regularisation=sub_probe["regularisation_selected"],
            max_iterations=settings["probe_max_iterations"],
            seed=int(cfg.seed),
        )
        temperature = fit_temperature(val_probe["logits"], sub_labels[validation])
        raw = expected_calibration_error(test_probe["probabilities"], sub_labels[test_indices])

        def tempered_probabilities(value: float) -> np.ndarray:
            # `logits` here are log-probabilities, which differ from the fitted
            # logits by a per-row constant; softmax is shift-invariant and the
            # shift survives the division, so tempering these is exactly
            # tempering the originals.
            scaled_logits = test_probe["logits"] / max(value, 1e-3)
            scaled = np.exp(scaled_logits - scaled_logits.max(axis=1, keepdims=True))
            return scaled / scaled.sum(axis=1, keepdims=True)

        tempered = expected_calibration_error(
            tempered_probabilities(temperature), sub_labels[test_indices]
        )

        # An oracle sweep, labelled as such. It separates two questions the fitted
        # number conflates: *can* this probe be calibrated by one temperature (the
        # sweep's minimum), and *does the standard protocol find it* (the fitted
        # value). On photograph-disjoint folds those answers came apart -- the folds
        # differ in class composition, so a temperature fitted on one does not
        # transfer -- and reporting only the fitted number would present a protocol
        # failure as a property of the encoder.
        sweep = []
        for candidate in (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0):
            outcome = expected_calibration_error(
                tempered_probabilities(candidate), sub_labels[test_indices]
            )
            sweep.append({"temperature": float(candidate), "ece": outcome.get("ece", float("nan"))})
        best = min(sweep, key=lambda row: row["ece"])

        reports[primary_label]["calibration"] = {
            "protocol": (
                "probe fit on fold-1 train, temperature fitted on fold-1 validation, "
                "ECE measured on the held-out test split"
            ),
            "temperature_fitted_on_validation": float(temperature),
            "probe_accuracy_train_fold_only": test_probe["accuracy"],
            "raw": raw,
            "temperature_scaled": tempered,
            "oracle_temperature": best["temperature"],
            "oracle_ece": best["ece"],
            "oracle_note": (
                "chosen on the TEST split, so not achievable in deployment; reported only to "
                "separate 'this probe is calibratable' from 'the validation fold found the "
                "right temperature'"
            ),
            "temperature_sweep": sweep,
        }
        extras["calibration"] = {
            "confidences": test_probe["probabilities"].max(axis=1).tolist(),
            "correct": (test_probe["predictions"] == sub_labels[test_indices]).tolist(),
            "ece": raw.get("ece", float("nan")),
        }
        logger.info(
            "Probe calibration | ECE %.4f raw (mean confidence %.4f against accuracy %.4f, so "
            "%s), %.4f after the validation-fitted T=%.2f, %.4f at the oracle T=%.2f. %s",
            raw.get("ece", float("nan")),
            raw.get("mean_confidence", float("nan")),
            raw.get("accuracy", float("nan")),
            "over-confident" if raw.get("overconfidence", 0.0) > 0 else "under-confident",
            tempered.get("ece", float("nan")), temperature,
            best["ece"], best["temperature"],
            "The fitted temperature transfers."
            if tempered.get("ece", 1.0) <= raw.get("ece", 0.0)
            else "The fitted temperature does NOT transfer across photograph-disjoint folds: it "
            "overshoots, so the probe is calibratable but this protocol does not calibrate it.",
        )

        # Centroid similarity, for the hierarchy figure.
        extras["centroid_similarity"] = centroid_similarity_matrix(
            primary.pooled, sub_labels, len(sub_names)
        )

        # CKA against the ImageNet control.
        if control_label and control_label in bundles:
            extras["cka"] = {
                label: linear_cka(bundle.pooled, bundles[control_label].pooled, seed=int(cfg.seed))
                for label, bundle in bundles.items()
                if label != control_label
            }
            for label, value in extras["cka"].items():
                reports[label]["cka_vs_imagenet_init"] = value
            logger.info(
                "Linear CKA vs the ImageNet initialisation | %s",
                ", ".join(f"{label}={value:.3f}" for label, value in extras["cka"].items()),
            )

        # Low-shot curve.
        low_shot_labels = [
            label
            for label in (
                OmegaConf.select(cfg, "experiment.evaluation.probe.low_shot_encoders", default=[]) or []
            )
            if label in bundles
        ]
        if low_shot_labels:
            shots = list(
                OmegaConf.select(
                    cfg, "experiment.evaluation.probe.low_shot_shots", default=[1, 5, 10, 25]
                )
            )
            repeats = int(
                OmegaConf.select(cfg, "experiment.evaluation.probe.low_shot_repeats", default=5)
            )
            extras["low_shot"] = {}
            for label in low_shot_labels:
                logger.info("Low-shot probe | %s, shots=%s x %d repeats", label, shots, repeats)
                rows = low_shot_probe(
                    bundles[label].pooled,
                    sub_labels,
                    fit_indices,
                    test_indices,
                    num_classes=len(sub_names),
                    shots=shots,
                    repeats=repeats,
                    regularisation=reports[label]["probe"]["sub_variety"]["regularisation_selected"],
                    seed=int(cfg.seed),
                )
                extras["low_shot"][label] = rows
                reports[label]["low_shot"] = rows

        # Layer-wise probe.
        layerwise_labels = [
            label
            for label in (
                OmegaConf.select(cfg, "experiment.evaluation.layerwise.encoders", default=[]) or []
            )
            if label in bundles and bundles[label].stages
        ]
        if layerwise_labels:
            extras["layerwise"] = {}
            for label in layerwise_labels:
                bundle = bundles[label]
                accuracies: dict[str, float] = {}
                for stage_key in [*STAGE_KEYS, "pooled"]:
                    matrix = bundle.pooled if stage_key == "pooled" else bundle.stages.get(stage_key)
                    if matrix is None:
                        continue
                    # `C` is selected per stage, not shared. The stages differ in
                    # width by 8x (96 -> 768) and, because the hooks read the
                    # pre-norm output, in scale as well; one fixed `C` would report
                    # "this stage carries no signal" for a stage that merely needed
                    # different regularisation, which is a claim about the trunk
                    # made by a hyperparameter.
                    selection = select_regularisation(
                        matrix[folds[0][0]], sub_labels[folds[0][0]],
                        matrix[folds[0][1]], sub_labels[folds[0][1]],
                        num_classes=len(sub_names),
                        grid=settings["probe_grid"],
                        max_iterations=settings["probe_max_iterations"],
                        seed=int(cfg.seed),
                    )
                    outcome = linear_probe(
                        matrix[fit_indices], sub_labels[fit_indices],
                        matrix[test_indices], sub_labels[test_indices],
                        num_classes=len(sub_names),
                        regularisation=selection["best_C"],
                        max_iterations=settings["probe_max_iterations"],
                        seed=int(cfg.seed),
                    )
                    accuracies[stage_key] = outcome["accuracy"]
                extras["layerwise"][label] = accuracies
                reports[label]["layerwise_probe"] = accuracies
                logger.info(
                    "Layer-wise probe | %s: %s",
                    label, ", ".join(f"{k}={v:.4f}" for k, v in accuracies.items()),
                )

        # Augmentation invariance and alignment/uniformity.
        invariance_labels = [
            label
            for label in (
                OmegaConf.select(cfg, "experiment.evaluation.invariance.encoders", default=[]) or []
            )
            if label in bundles
        ]
        if invariance_labels and bool(
            OmegaConf.select(cfg, "experiment.evaluation.invariance.enabled", default=True)
        ):
            max_invariance = int(
                OmegaConf.select(cfg, "experiment.evaluation.invariance.max_samples", default=1500)
            )
            rng = np.random.default_rng(int(cfg.seed))
            chosen = np.sort(
                rng.choice(len(dataset), size=min(max_invariance, len(dataset)), replace=False)
            )
            extras["invariance"] = {}
            for label in invariance_labels:
                logger.info("Augmentation invariance | %s over %d images", label, chosen.size)
                extractor = build_backbone(bundles[label].spec, cfg, device, seed=int(cfg.seed))
                view_a, view_b = extract_view_pairs(
                    extractor, dataset, chosen, cfg, device, batch_size, seed=int(cfg.seed)
                )
                del extractor
                report = alignment_uniformity(view_a, view_b, seed=int(cfg.seed))
                consistency = augmentation_consistency(
                    bundles[label].pooled[chosen], view_a, sub_labels[chosen], seed=int(cfg.seed)
                )
                normalized_a = l2_normalize(view_a)
                normalized_b = l2_normalize(view_b)
                payload = {**report, **consistency}
                payload["_positive_cosines"] = (normalized_a * normalized_b).sum(axis=1).tolist()

                # The two reference populations, computed per encoder rather than
                # once for the primary. An absolute cosine is not comparable across
                # encoders -- it is inflated by whatever mean direction a trunk
                # happens to have -- so the figure can only compare the *gaps*
                # within one encoder, and that needs all three populations from
                # each one.
                clean = l2_normalize(bundles[label].pooled[chosen])
                similarity = clean @ clean.T
                same_class = sub_labels[chosen][:, None] == sub_labels[chosen][None, :]
                np.fill_diagonal(same_class, False)
                stride_same = max(1, int(same_class.sum()) // 20000)
                stride_diff = max(1, int((~same_class).sum()) // 20000)
                payload["_same_class_cosines"] = similarity[same_class][::stride_same].tolist()
                payload["_different_class_cosines"] = similarity[~same_class][::stride_diff].tolist()
                extras["invariance"][label] = payload
                reports[label]["invariance"] = {
                    key: value for key, value in payload.items() if not key.startswith("_")
                }
                logger.info(
                    "  alignment %.4f, uniformity %.4f, same-image cosine %.4f, self-retrieval@1 %.4f",
                    report["alignment"], report["uniformity"],
                    consistency["same_image_cosine_mean"], consistency["self_retrieval_top1"],
                )


        # DINO's own prototype head.
        head_checkpoint = OmegaConf.select(
            cfg, "experiment.evaluation.prototypes.checkpoint", default=None
        )
        if bool(
            OmegaConf.select(cfg, "experiment.evaluation.prototypes.enabled", default=True)
        ) and head_checkpoint:
            path = Path(str(head_checkpoint))
            if not path.exists():
                logger.warning("Prototype analysis skipped: %s not found.", path)
            else:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                state = payload.get("student_head") if isinstance(payload, Mapping) else None
                if state is None:
                    logger.warning("Prototype analysis skipped: %s has no 'student_head'.", path)
                else:
                    head = DINOHead(
                        in_dim=int(cfg.model.head.in_dim),
                        out_dim=int(cfg.model.head.out_dim),
                        use_batch_norm=cfg.model.head.use_batch_norm,
                        norm_last_layer=bool(cfg.model.head.norm_last_layer),
                        num_layers=int(cfg.model.head.num_layers),
                        hidden_dim=int(cfg.model.head.hidden_dim),
                        bottleneck_dim=int(cfg.model.head.bottleneck_dim),
                    )
                    incompatible = head.load_state_dict(state, strict=False)
                    head.eval()
                    with torch.no_grad():
                        logits, bottleneck = head(
                            torch.from_numpy(primary.pooled).float(), return_bottleneck=True
                        )
                    prototypes = prototype_report(
                        logits.numpy(), sub_labels, num_prototypes=int(cfg.model.head.out_dim)
                    )
                    prototypes["checkpoint"] = str(path)
                    prototypes["checkpoint_sha256"] = sha256_of(path)
                    prototypes["missing_keys"] = len(incompatible.missing_keys)
                    prototypes["unexpected_keys"] = len(incompatible.unexpected_keys)
                    prototypes["bottleneck_spectrum"] = spectral_report(bottleneck.numpy()).as_dict(
                        include_curves=False
                    )
                    prototypes["bottleneck_separability"] = class_separability(
                        bottleneck.numpy()[test_indices], sub_labels[test_indices], sub_names
                    )
                    extras["prototypes"] = prototypes
                    reports[primary_label]["prototypes"] = {
                        key: value
                        for key, value in prototypes.items()
                        if key != "usage_shares"
                    }
                    logger.info(
                        "Prototype head | %d of %d prototypes active, NMI vs sub-variety %.4f, "
                        "purity %.4f, top-1 share %.4f",
                        int(prototypes["active_prototypes"]), int(prototypes["num_prototypes"]),
                        prototypes["nmi_vs_labels"], prototypes["purity_vs_labels"],
                        prototypes["top1_prototype_share"],
                    )

        # Split-protocol delta (near-duplicate leakage).
        if bool(OmegaConf.select(cfg, "experiment.evaluation.split.also_stratified", default=True)):
            alt_folds, alt_test, alt_report = split_dataset(
                dataset, test_size=test_size, num_folds=num_folds, seed=int(cfg.seed),
                protocol="stratified" if protocol == "grouped" else "grouped",
            )
            alt_fit = np.unique(np.concatenate([alt_folds[0][0], alt_folds[0][1]]))
            alt_probe_sub = linear_probe(
                primary.pooled[alt_fit], sub_labels[alt_fit],
                primary.pooled[alt_test], sub_labels[alt_test],
                num_classes=len(sub_names),
                regularisation=sub_probe["regularisation_selected"],
                max_iterations=settings["probe_max_iterations"], seed=int(cfg.seed),
            )
            alt_probe_seed = linear_probe(
                primary.pooled[alt_fit], seed_labels[alt_fit],
                primary.pooled[alt_test], seed_labels[alt_test],
                num_classes=len(seed_names),
                regularisation=seed_probe["regularisation_selected"],
                max_iterations=settings["probe_max_iterations"], seed=int(cfg.seed),
            )
            alt_knn = knn_classifier(
                primary.pooled[alt_fit], sub_labels[alt_fit],
                primary.pooled[alt_test], sub_labels[alt_test],
                num_classes=len(sub_names),
                k=reports[primary_label]["knn"]["sub_variety"]["best_k"],
                temperature=settings["knn_temperature"],
            )
            extras["leakage"] = {
                "grouped": {
                    "probe_sub": sub_probe["test_accuracy"],
                    "probe_seed": seed_probe["test_accuracy"],
                    "knn_sub": reports[primary_label]["knn"]["sub_variety"]["best_accuracy"],
                },
                "stratified": {
                    "probe_sub": alt_probe_sub["accuracy"],
                    "probe_seed": alt_probe_seed["accuracy"],
                    "knn_sub": alt_knn["accuracy"],
                },
                "report": json_safe(alt_report),
            }
            extras["leakage"]["delta_probe_sub"] = (
                alt_probe_sub["accuracy"] - sub_probe["test_accuracy"]
            )
            logger.info(
                "Split protocol | crop-level probe %.4f vs grouped %.4f: %+.4f of the headline number "
                "is near-duplicate leakage, not sub-variety discrimination.",
                alt_probe_sub["accuracy"], sub_probe["test_accuracy"],
                extras["leakage"]["delta_probe_sub"],
            )

        # Milestone progression.
        milestone_rows: list[dict[str, Any]] = []
        for label, bundle in bundles.items():
            if bundle.spec.role not in {"primary", "milestone"}:
                continue
            digits = "".join(character for character in label if character.isdigit())
            if not digits:
                continue
            milestone_rows.append(
                {
                    "label": label,
                    "epoch": int(digits),
                    "probe_sub_accuracy": (
                        reports[label]
                        .get("grouped_cv", {})
                        .get("sub_variety", {})
                        .get("out_of_fold_accuracy")
                        or reports[label]["probe"]["sub_variety"]["test_accuracy"]
                    ),
                    "knn_sub_accuracy": (
                        reports[label]
                        .get("grouped_cv", {})
                        .get("sub_variety", {})
                        .get("out_of_fold_knn_accuracy")
                        or reports[label]["knn"]["sub_variety"]["best_accuracy"]
                    ),
                    "silhouette": reports[label]["separability_full"]["sub_variety"][
                        "silhouette_cosine"
                    ],
                    "kmeans_nmi": reports[label].get("kmeans", {}).get("k_sub_variety", {}).get("nmi"),
                    "rankme": reports[label]["spectrum"]["rankme"],
                }
            )
        milestone_rows.sort(key=lambda row: row["epoch"])
        extras["milestones"] = milestone_rows

        # t-SNE projections on a shared subsample.
        tsne_max = int(OmegaConf.select(cfg, "experiment.evaluation.tsne.max_samples", default=3000))
        perplexity = float(
            OmegaConf.select(cfg, "experiment.evaluation.tsne.perplexity", default=30.0)
        )
        tsne_labels = [
            label
            for label in (
                OmegaConf.select(cfg, "experiment.evaluation.tsne.encoders", default=[]) or [primary_label]
            )
            if label in bundles
        ]
        if tsne_labels:
            rng = np.random.default_rng(int(cfg.seed))
            chosen = np.sort(
                rng.choice(len(dataset), size=min(tsne_max, len(dataset)), replace=False)
            )
            extras["tsne"] = {}
            for label in tsne_labels:
                logger.info("t-SNE | %s over %d samples", label, chosen.size)
                projection = tsne_projection(
                    l2_normalize(bundles[label].pooled[chosen]),
                    perplexity=perplexity,
                    seed=int(cfg.seed),
                    max_samples=None,
                )
                if projection is not None:
                    extras["tsne"][label] = (projection, chosen)

        # Qualitative retrieval panel.
        if bool(
            OmegaConf.select(cfg, "experiment.evaluation.figures.retrieval_examples", default=True)
        ):
            extras["retrieval_examples"] = build_retrieval_panel(
                primary.pooled,
                sub_labels,
                groups,
                dataset,
                sub_names,
                rows=int(
                    OmegaConf.select(cfg, "experiment.evaluation.figures.retrieval_rows", default=6)
                ),
                neighbours=int(
                    OmegaConf.select(
                        cfg, "experiment.evaluation.figures.retrieval_neighbours", default=5
                    )
                ),
                seed=int(cfg.seed),
            )

        # ------------------------------------------------------- efficiency
        efficiency: dict[str, Any] = {}
        if bool(OmegaConf.select(cfg, "experiment.evaluation.efficiency.enabled", default=True)):
            example = next(iter(loader))[0][:1].to(device)
            pool = next(iter(loader))[0].to(device)
            batch_sizes = list(
                OmegaConf.select(
                    cfg, "experiment.evaluation.efficiency.batch_sizes", default=[1, 8, 32]
                )
            )
            report = profile_model(
                primary_extractor,
                example,
                device,
                name=f"stage1_encoder_{primary_label}",
                batch_sizes=[size for size in batch_sizes if size <= pool.shape[0]],
                warmup=int(
                    OmegaConf.select(cfg, "experiment.evaluation.efficiency.warmup", default=3)
                ),
                iterations=int(
                    OmegaConf.select(cfg, "experiment.evaluation.efficiency.iterations", default=20)
                ),
                measure_flops=bool(
                    OmegaConf.select(
                        cfg, "experiment.evaluation.efficiency.measure_flops", default=True
                    )
                ),
                sample_pool=pool,
            )
            efficiency = report.as_dict()
            logger.info("Encoder inference | %s", report.summary_line())

        # ---------------------------------------------------------- dynamics
        dynamics = None
        dynamics_summary: dict[str, Any] = {}
        events_path = OmegaConf.select(cfg, "experiment.evaluation.dynamics.events_path", default=None)
        if bool(OmegaConf.select(cfg, "experiment.evaluation.dynamics.enabled", default=True)):
            candidate = (
                Path(str(events_path))
                if events_path
                else discover_events_path(Path(str(evaluation_cfg.save_path)).parent, logger)
            )
            if candidate is not None and candidate.exists():
                dynamics = parse_pretrain_dynamics(candidate)
                dynamics_summary = dynamics.summary()
                dynamics_summary["prototypes"] = int(cfg.model.head.out_dim)
                dynamics_summary["events_path"] = str(candidate)
                logger.info(
                    "Stage-1 dynamics | %d epochs, loss %.4f -> %.4f (last-quarter improvement %.4f), "
                    "teacher entropy %.3f against a floor of %.3f, %.1f%% of wall clock spent waiting "
                    "for data, %.2f h total.",
                    dynamics_summary.get("epochs_completed", 0),
                    dynamics_summary.get("loss_initial", float("nan")),
                    dynamics_summary.get("loss_final", float("nan")),
                    dynamics_summary.get("loss_improvement_last_quarter", float("nan")),
                    dynamics_summary.get("teacher_entropy_final", float("nan")),
                    dynamics_summary.get("teacher_entropy_floor", float("nan")),
                    100.0 * dynamics_summary.get("data_wait_fraction_mean", float("nan")),
                    dynamics_summary.get("training_hours", float("nan")),
                )
            else:
                logger.warning("Stage-1 dynamics skipped: %s not found.", candidate)

        # ------------------------------------------------------------ output
        written_figures: dict[str, str] = {}
        if bool(OmegaConf.select(cfg, "experiment.evaluation.figures.enabled", default=True)):
            written_figures = build_figures(
                save_path, bundles, reports, extras, seed_labels, sub_labels, seed_names,
                sub_names, subvariety_to_seed_type, settings, dynamics, dynamics_summary, logger,
            )

        predictions_path = save_test_predictions(
            save_path,
            seed_true=seed_labels[test_indices],
            seed_pred=seed_probe["_predictions"],
            sub_true=sub_labels[test_indices],
            sub_pred=sub_probe["_predictions"],
            seed_type_names=seed_names,
            sub_variety_names=sub_names,
            subvariety_to_seed_type=subvariety_to_seed_type,
            sub_scores=sub_probe["_probabilities"],
            embeddings=primary.pooled[test_indices],
            sub_logits=sub_probe["_logits"],
            tokens_per_sample=1,
        )
        out_of_fold_path = None
        if grouped_cv is not None:
            # A second dump in the same format, over every crop. `generate_plots.py`
            # reads `test_predictions.npz` by name, so the single-split file keeps
            # that name and this one sits beside it -- the two are different
            # protocols and merging them would misreport both.
            out_of_fold_path = save_test_predictions(
                save_path,
                seed_true=seed_labels,
                seed_pred=grouped_cv["seed_type"]["_predictions"],
                sub_true=sub_labels,
                sub_pred=grouped_cv["sub_variety"]["_predictions"],
                seed_type_names=seed_names,
                sub_variety_names=sub_names,
                subvariety_to_seed_type=subvariety_to_seed_type,
                sub_scores=grouped_cv["sub_variety"]["_probabilities"],
                embeddings=primary.pooled,
                tokens_per_sample=1,
                filename="out_of_fold_predictions.npz",
            )

        tables = write_tables(save_path, reports, extras, seed_names, sub_names, hierarchy)
        clean_reports = strip_private(reports)
        metrics_payload = {
            "encoders": clean_reports,
            "stage1_dynamics": dynamics_summary,
            "split": json_safe(split_report),
            "dataset": json_safe(group_report),
            "efficiency": json_safe(efficiency),
            "leakage": json_safe({k: v for k, v in extras.get("leakage", {}).items()}),
            "milestones": json_safe(extras.get("milestones", [])),
        }
        (save_path / "metrics.json").write_text(
            json.dumps(json_safe(metrics_payload), indent=2, sort_keys=True), encoding="utf-8"
        )

        provenance = {
            "git_commit": git_commit(Path(__file__).resolve().parents[2]),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": str(device),
            "accelerator": accelerator.as_dict(),
            "seed": int(cfg.seed),
            "dataset_root": str(cfg.data.root_path),
            "backbone": str(cfg.model.backbone.name),
            "split_manifest": manifest,
            "split_manifest_sha256": sha256_of(manifest),
            "stage2_handoff_path": str(handoff) if handoff else None,
            "stage2_handoff_matches_primary": handoff_match,
            "stage2_handoff_verdict": handoff_verdict,
            "stage2_handoff_detail": handoff_detail,
            "encoders": {
                label: {
                    **bundle.spec.as_dict(),
                    "checkpoint_sha256": bundle.checkpoint_sha256,
                    "backbone_parameters": bundle.parameters,
                }
                for label, bundle in bundles.items()
            },
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        try:
            import sklearn

            provenance["scikit_learn"] = sklearn.__version__
        except ImportError:  # pragma: no cover
            provenance["scikit_learn"] = "unavailable"
        (save_path / "provenance.json").write_text(
            json.dumps(json_safe(provenance), indent=2, sort_keys=True), encoding="utf-8"
        )

        flat_metrics = flatten_metrics(clean_reports, dynamics_summary)
        tracker.log_metrics(flat_metrics, step=0)
        tracker.log_event(
            "stage1_evaluation",
            {
                "primary": primary_label,
                "figures": written_figures,
                "tables": tables,
                "predictions": predictions_path,
            },
        )

        summary = RunSummary(
            name=str(cfg.experiment.name),
            group=str(OmegaConf.select(cfg, "experiment.group", default="stage1_evaluation")),
            run_dir=str(save_path),
            metrics=flat_metrics,
            efficiency=json_safe(efficiency),
            history={},
            component_flags={
                "stage": "pretrain_evaluation",
                "backbone": str(cfg.model.backbone.name),
                "primary_encoder": primary_label,
                "encoders": [spec.label for spec in specs],
                "readout": "sklearn L-BFGS logistic regression + weighted cosine kNN",
                "encoder_frozen": True,
            },
            loss_flags={"stage1_objective": "dino_self_distillation"},
            split=json_safe({**split_report, **group_report}),
            fold_metrics=fold_metric_summary(reports, settings),
            runtime={
                "device": str(device),
                "world_size": 1,
                "amp": "disabled (fp32 throughout)",
                "deterministic": True,
                "wall_clock_seconds": time.perf_counter() - started,
                "ssl_saw_every_image": True,
                "caveat": (
                    "Stage-1 pretraining was label-free but covered the whole image set, so the "
                    "readout test split is unseen photographs rather than unseen images. Standard "
                    "SSL protocol; it bounds the claim to in-domain readout quality."
                ),
            },
            config={
                "backbone": str(cfg.model.backbone.name),
                "image_size": int(cfg.data.image_size),
                "split_protocol": protocol,
                "test_size": test_size,
                "num_folds": num_folds,
                "seed": int(cfg.seed),
            },
            artifacts={
                "metrics": str(save_path / "metrics.json"),
                "provenance": str(save_path / "provenance.json"),
                "predictions": predictions_path,
                **({"out_of_fold_predictions": out_of_fold_path} if out_of_fold_path else {}),
                "split_manifest": manifest,
                **{f"figure_{name}": path for name, path in written_figures.items()},
                **{f"table_{name}": path for name, path in tables.items()},
            },
        )
        summary_path = summary.save(save_path)
        logger.info("Wrote %s", summary_path)
        logger.info(
            "Stage-1 evaluation complete in %.1fs. %d figures, %d tables under %s",
            time.perf_counter() - started, len(written_figures), len(tables), save_path,
        )
    finally:
        tracker.close()


def build_retrieval_panel(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    dataset: HierarchicalSeedDataset,
    class_names: Sequence[str],
    rows: int = 6,
    neighbours: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    """Query images plus their nearest neighbours, for the qualitative figure.

    Same-photograph neighbours are excluded, for the same reason
    :func:`~src.utils.representation.retrieval_report` excludes them: without the
    exclusion the panel shows five near-identical crops and says nothing.

    Queries are drawn to span the difficulty range -- the classes with the best and
    worst retrieval precision -- rather than uniformly, because a uniform sample of
    27 classes at six rows is six random rice varieties.
    """
    from PIL import Image

    normalized = l2_normalize(features)
    rng = np.random.default_rng(seed)

    # Per-class precision@1 with same-group neighbours excluded, cheaply, from a
    # class-balanced sample rather than the full 9k x 9k matrix.
    sample = np.sort(rng.choice(features.shape[0], size=min(2500, features.shape[0]), replace=False))
    similarity = normalized[sample] @ normalized[sample].T
    np.fill_diagonal(similarity, -np.inf)
    similarity[groups[sample][:, None] == groups[sample][None, :]] = -np.inf
    best = similarity.argmax(axis=1)
    correct = labels[sample[best]] == labels[sample]
    per_class = {}
    for label in np.unique(labels[sample]):
        mask = labels[sample] == label
        if mask.sum() >= 5:
            per_class[int(label)] = float(correct[mask].mean())

    ordered = sorted(per_class, key=lambda label: per_class[label])
    take = min(rows, len(ordered))
    picked_classes = [ordered[index] for index in np.linspace(0, len(ordered) - 1, take).astype(int)]

    queries, neighbour_images, query_labels, correctness = [], [], [], []

    for label in picked_classes:
        candidates = np.flatnonzero(labels == label)
        query_index = int(rng.choice(candidates))
        scores = normalized @ normalized[query_index]
        scores[query_index] = -np.inf
        scores[groups == groups[query_index]] = -np.inf
        top = np.argsort(-scores)[:neighbours]

        def load(index: int) -> np.ndarray:
            with Image.open(dataset.samples[int(index)][0]) as handle:
                return np.asarray(handle.convert("RGB").resize((96, 96))) / 255.0

        queries.append(load(query_index))
        neighbour_images.append([load(index) for index in top])
        query_labels.append(
            f"{class_names[int(label)]}\nP@1={per_class.get(int(label), float('nan')):.2f}"
        )
        correctness.append([bool(labels[index] == label) for index in top])

    return {
        "queries": queries,
        "neighbours": neighbour_images,
        "query_labels": query_labels,
        "correct": correctness,
        "per_class_precision_at_1": {class_names[k]: v for k, v in per_class.items()},
    }


def strip_private(reports: Mapping[str, Any]) -> dict[str, Any]:
    """Drop ``_``-prefixed working arrays so ``metrics.json`` stays readable."""

    def clean(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: clean(item) for key, item in value.items() if not str(key).startswith("_")}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    return {label: clean(report) for label, report in reports.items()}


def fold_metric_summary(reports: Mapping[str, Any], settings: Mapping[str, Any]) -> dict[str, Any]:
    """``fold_metrics`` for :class:`~src.utils.evaluation.RunSummary`.

    The cross-validated probe spread, in the same shape stage-2 runs use, so the
    cross-run table renders this evaluation without a special case.
    """
    payload: dict[str, dict[str, float]] = {}
    for label, report in reports.items():
        probe = report.get("probe", {}).get("sub_variety", {})
        if "cv_accuracy_mean" not in probe:
            continue
        payload[f"{label}/probe_sub_variety_accuracy"] = {
            "mean": probe["cv_accuracy_mean"],
            "std": probe["cv_accuracy_std"],
            "folds": float(probe.get("cv_folds", 0)),
        }
    return payload


def flatten_metrics(reports: Mapping[str, Any], dynamics: Mapping[str, Any]) -> dict[str, float]:
    """Flatten the nested report into ``prefix/name`` scalars for the tracker."""
    flat: dict[str, float] = {}

    def walk(prefix: str, node: Any) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                walk(f"{prefix}/{key}" if prefix else str(key), value)
        elif isinstance(node, bool):
            flat[prefix] = float(node)
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            value = float(node)
            if value == value:
                flat[prefix] = value

    for label, report in reports.items():
        for section in (
            "grouped_cv",
            "spectrum",
            "feature_statistics",
            "separability_full",
            "separability_test_split",
            "knn",
            "probe",
            "kmeans",
            "retrieval",
            "calibration",
            "invariance",
            "layerwise_probe",
            "prototypes",
            "hierarchical_probe",
        ):
            if section in report:
                walk(f"{label}/{section}", report[section])
        if "cka_vs_imagenet_init" in report:
            flat[f"{label}/cka_vs_imagenet_init"] = float(report["cka_vs_imagenet_init"])
    walk("stage1_dynamics", dynamics)
    # Drop the long curves: a tracker scalar per singular value is 768 useless rows.
    return {
        key: value
        for key, value in flat.items()
        if "singular_values" not in key and "explained_variance_ratio" not in key
    }


def write_tables(
    save_path: Path,
    reports: Mapping[str, Any],
    extras: Mapping[str, Any],
    seed_names: Sequence[str],
    sub_names: Sequence[str],
    hierarchy: Mapping[str, Any] | None,
) -> dict[str, str]:
    """Write every flat CSV table and return ``{name: path}``."""
    tables_dir = save_path / "tables"
    written: dict[str, str] = {}

    def get(report: Mapping[str, Any], *path: str) -> Any:
        node: Any = report
        for key in path:
            if not isinstance(node, Mapping) or key not in node:
                return None
            node = node[key]
        return node

    encoder_rows = []
    for label, report in reports.items():
        encoder_rows.append(
            {
                "encoder": label,
                "role": get(report, "spec", "role"),
                "checkpoint": get(report, "spec", "checkpoint"),
                "checkpoint_sha256": (get(report, "checkpoint_sha256") or "")[:16],
                # Out-of-fold first: it is the number the report headlines, because
                # it covers every crop and all 27 classes.
                "oof_probe_sub_accuracy": get(
                    report, "grouped_cv", "sub_variety", "out_of_fold_accuracy"
                ),
                "oof_probe_sub_accuracy_testable_classes": get(
                    report, "grouped_cv", "sub_variety", "testable_classes_only", "out_of_fold_accuracy"
                ),
                "oof_probe_sub_f1_testable_classes": get(
                    report, "grouped_cv", "sub_variety", "testable_classes_only", "out_of_fold_f1_macro"
                ),
                "oof_probe_sub_f1_macro": get(
                    report, "grouped_cv", "sub_variety", "out_of_fold_f1_macro"
                ),
                "oof_probe_sub_fold_std": get(
                    report, "grouped_cv", "sub_variety", "probe_accuracy_std"
                ),
                "oof_knn_sub_accuracy": get(
                    report, "grouped_cv", "sub_variety", "out_of_fold_knn_accuracy"
                ),
                "oof_probe_seed_accuracy": get(
                    report, "grouped_cv", "seed_type", "out_of_fold_accuracy"
                ),
                "oof_knn_seed_accuracy": get(
                    report, "grouped_cv", "seed_type", "out_of_fold_knn_accuracy"
                ),
                "probe_sub_accuracy": get(report, "probe", "sub_variety", "test_accuracy"),
                "probe_sub_f1_macro": get(report, "probe", "sub_variety", "test_f1_macro"),
                "probe_sub_cv_mean": get(report, "probe", "sub_variety", "cv_accuracy_mean"),
                "probe_sub_cv_std": get(report, "probe", "sub_variety", "cv_accuracy_std"),
                "probe_seed_accuracy": get(report, "probe", "seed_type", "test_accuracy"),
                "probe_sub_train_accuracy": get(report, "probe", "sub_variety", "train_accuracy"),
                "probe_sub_generalisation_gap": get(
                    report, "probe", "sub_variety", "generalisation_gap"
                ),
                "knn_sub_accuracy": get(report, "knn", "sub_variety", "best_accuracy"),
                "knn_sub_best_k": get(report, "knn", "sub_variety", "best_k"),
                "knn_seed_accuracy": get(report, "knn", "seed_type", "best_accuracy"),
                "silhouette_sub": get(
                    report, "separability_full", "sub_variety", "silhouette_cosine"
                ),
                "silhouette_seed": get(report, "separability_full", "seed_type", "silhouette_cosine"),
                "fisher_ratio_sub": get(report, "separability_full", "sub_variety", "fisher_ratio"),
                "rankme": get(report, "spectrum", "rankme"),
                "dims_for_95_variance": get(report, "spectrum", "dims_for_95_variance"),
                "kmeans_nmi": get(report, "kmeans", "k_sub_variety", "nmi"),
                "kmeans_ari": get(report, "kmeans", "k_sub_variety", "adjusted_rand"),
                "kmeans_cluster_accuracy": get(report, "kmeans", "k_sub_variety", "cluster_accuracy"),
                "retrieval_p_at_1_grouped": get(
                    report, "retrieval", "grouped_excluded", "precision_at_1"
                ),
                "retrieval_p_at_1_ungrouped": get(report, "retrieval", "ungrouped", "precision_at_1"),
                "cka_vs_imagenet_init": get(report, "cka_vs_imagenet_init"),
                "alignment": get(report, "invariance", "alignment"),
                "uniformity": get(report, "invariance", "uniformity"),
                "self_retrieval_top1": get(report, "invariance", "self_retrieval_top1"),
                "backbone_parameters_m": (get(report, "backbone_parameters") or 0) / 1e6,
            }
        )
    written["encoder_comparison"] = write_csv(tables_dir / "encoder_comparison.csv", encoder_rows)

    if hierarchy is not None:
        evaluation = hierarchy["evaluation"]
        parents = list(extras.get("subvariety_to_seed_type", []))
        silhouettes = (
            extras.get("primary_per_class_silhouette", {}) if extras else {}
        )
        confusion = np.asarray(evaluation.sub_confusion, dtype=np.float64)
        row_totals = confusion.sum(axis=1, keepdims=True)
        rates = np.divide(confusion, row_totals, out=np.zeros_like(confusion), where=row_totals > 0)
        np.fill_diagonal(rates, 0.0)
        per_class_rows = []
        for index, entry in enumerate(evaluation.per_class_sub):
            worst = int(rates[index].argmax()) if rates[index].max() > 0 else -1
            per_class_rows.append(
                {
                    "sub_variety": entry.name,
                    "seed_type": (
                        seed_names[parents[index]] if index < len(parents) else ""
                    ),
                    "precision": entry.precision,
                    "recall": entry.recall,
                    "f1": entry.f1,
                    "support": entry.support,
                    "misclassification_rate": evaluation.sub_misclassification.get(entry.name),
                    # The single class this one is most often mistaken for, which is
                    # what a 27x27 matrix makes a reader hunt for by eye.
                    "top_confusion_with": (
                        evaluation.per_class_sub[worst].name if worst >= 0 else ""
                    ),
                    "top_confusion_rate": float(rates[index].max()) if worst >= 0 else 0.0,
                    "silhouette_cosine": silhouettes.get(entry.name),
                }
            )
        written["per_class_sub_variety"] = write_csv(
            tables_dir / "per_class_sub_variety.csv", per_class_rows
        )
        written["per_class_seed_type"] = write_csv(
            tables_dir / "per_class_seed_type.csv",
            [
                {
                    "seed_type": entry.name,
                    "precision": entry.precision,
                    "recall": entry.recall,
                    "f1": entry.f1,
                    "support": entry.support,
                    "kl_alignment": evaluation.alignment.per_seed_type.get(entry.name),
                    "sub_variety_accuracy_within": (
                        evaluation.per_seed_type_sub.get(entry.name, {}).get("accuracy")
                    ),
                    "sub_variety_f1_within": (
                        evaluation.per_seed_type_sub.get(entry.name, {}).get("f1_macro")
                    ),
                }
                for entry in evaluation.per_class_seed
            ],
        )

    if extras.get("low_shot"):
        written["low_shot"] = write_csv(
            tables_dir / "low_shot.csv",
            [
                {"encoder": label, **row}
                for label, rows in extras["low_shot"].items()
                for row in rows
            ],
        )

    if extras.get("layerwise"):
        written["layerwise_probe"] = write_csv(
            tables_dir / "layerwise_probe.csv",
            [
                {"encoder": label, "stage": stage, "probe_sub_accuracy": value}
                for label, values in extras["layerwise"].items()
                for stage, value in values.items()
            ],
        )

    if extras.get("milestones"):
        written["milestone_progression"] = write_csv(
            tables_dir / "milestone_progression.csv", extras["milestones"]
        )

    if extras.get("leakage"):
        leakage = extras["leakage"]
        written["split_protocol_delta"] = write_csv(
            tables_dir / "split_protocol_delta.csv",
            [
                {
                    "metric": metric,
                    "grouped": leakage["grouped"][metric],
                    "stratified": leakage["stratified"][metric],
                    "delta_stratified_minus_grouped": leakage["stratified"][metric]
                    - leakage["grouped"][metric],
                }
                for metric in ("probe_sub", "probe_seed", "knn_sub")
            ],
        )

    if extras.get("prototypes"):
        prototypes = extras["prototypes"]
        shares = np.sort(np.asarray(prototypes["usage_shares"], dtype=np.float64))[::-1]
        written["prototype_usage"] = write_csv(
            tables_dir / "prototype_usage.csv",
            [
                {"rank": index + 1, "share": float(value), "cumulative": float(cumulative)}
                for index, (value, cumulative) in enumerate(zip(shares, np.cumsum(shares)))
                if value > 0
            ],
        )

    if extras.get("retrieval_examples", {}).get("per_class_precision_at_1"):
        written["retrieval_per_class"] = write_csv(
            tables_dir / "retrieval_per_class.csv",
            [
                {"sub_variety": name, "precision_at_1_group_excluded": value}
                for name, value in extras["retrieval_examples"]["per_class_precision_at_1"].items()
            ],
        )

    return written


if __name__ == "__main__":
    main()
