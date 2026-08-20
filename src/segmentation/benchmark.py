"""Does the refined corpus actually read out better than the legacy one?

    python -m src.segmentation.benchmark
    python -m src.segmentation.benchmark --encoders handcrafted        # seconds, no GPU
    python -m src.segmentation.benchmark --protocols stratified grouped_cv

A corpus change is a claim, and this is the measurement that tests it. Nothing
here trains: two corpora are pushed through the *same* frozen encoders under the
*same* protocols, and the difference is the corpus.

Three encoders, chosen so the result cannot be an artefact of any one of them:

``handcrafted``
    Ten numpy scalars -- ``log(w*h)``, ``log(w/h)``, mean and std of R, G, B and
    grey. The repository already treats this as a reporting obligation rather
    than a control, because it is the floor a reviewer asks about first. It is
    also the sharpest instrument for *this* comparison: it is pure colour and
    size, so it says directly what the crop redefinition did to the colour cue.
``imagenet_init``
    The frozen SwinV2-Tiny trunk this pipeline starts stage 1 from, at its
    ImageNet-1k initialisation. The bar every stage-1 run is measured against,
    so a corpus change that moves it moves everything downstream.
``random_init``
    The same architecture, untrained. Separates "the corpus is more decodable"
    from "the encoder likes it".

Two protocols, because they answer different questions and the gap between them
is the point:

``stratified``
    Crop-level, the pipeline's primary protocol. Optimistic by construction --
    near-duplicate crops of one photograph land on both sides.
``grouped_cv``
    ``StratifiedGroupKFold`` over source photographs, scored out of fold, so
    every crop is predicted by a model that never saw its photograph. This is
    the number that can move for the right reason: the refined corpus draws on
    **96** photographs against the legacy corpus's 81.

The size control
----------------

The refined corpus holds ~44 % more crops, and a probe with more training data
scores higher whatever the crops look like. ``--match-size`` therefore also runs
the refined corpus subsampled to the legacy corpus's exact per-class counts, and
``--match-sources`` restricts it to the 81 photographs the legacy corpus used.
With both, a remaining difference is attributable to the crops themselves. Both
are on by default: a headline that skipped them would not be a measurement.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.dataset import get_finetune_dataset, source_image_id  # noqa: E402
from src.datasets.transforms import get_supervised_transforms  # noqa: E402
from src.utils.representation import (  # noqa: E402
    handcrafted_image_features,
    knn_classifier,
    linear_probe,
    select_regularisation,
)

LOGGER = logging.getLogger(__name__)

#: The trunk both stages use. Named here rather than read from the Hydra config
#: because this script deliberately does not compose one -- it must be runnable
#: against any two directories with no experiment attached.
BACKBONE = "swinv2_tiny_window16_256"
IMAGE_SIZE = 256

#: The probe's regularisation is SELECTED on a validation split, never fixed and
#: never selected on the test split -- the same protocol
#: :func:`src.trainers.pretrain_eval.probe_task` uses, and for the same reason.
#: A fixed ``C`` is not a neutral choice across these encoders: ``linear_probe``
#: L2-normalises its inputs, and the handcrafted vector's ``log(area)`` component
#: is ~8 against ~0.5 for the colour moments, so after normalisation the colour
#: cue survives only if the regulariser is loose enough to use it. Fixing
#: ``C = 1`` scores the ten-scalar floor at 0.10 instead of 0.54 -- a measurement
#: of the regulariser, not of the corpus.
REGULARISATION_GRID = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def load_corpus(root: Path, batch_size: int = 32, num_workers: int = 4):
    """The dataset, its labels, its source-photograph groups and its paths."""
    transform = get_supervised_transforms(IMAGE_SIZE, train=False)
    dataset = get_finetune_dataset(str(root), transform, include_path=True)
    labels = np.array([sub for _, _, sub in dataset.samples], dtype=np.int64)
    groups = dataset.source_groups()
    paths = [path for path, _, _ in dataset.samples]
    return dataset, labels, groups, paths


def frozen_features(
    root: Path,
    pretrained: bool,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 0,
) -> np.ndarray:
    """Pooled trunk features for every image in a corpus.

    ``pretrained=False`` builds the same architecture at its random
    initialisation, seeded so the "random" control is reproducible rather than a
    different network on every run.
    """
    from torch.utils.data import DataLoader

    from src.models.builder import BackboneFeatureExtractor

    torch.manual_seed(int(seed))
    backbone = BackboneFeatureExtractor(
        model_name=BACKBONE, pretrained=bool(pretrained), freeze=True
    ).to(device).eval()

    dataset, _, _, _ = load_corpus(root, batch_size, num_workers)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for images, _, _, _ in loader:
            output = backbone(images.to(device))
            # The trunk emits [B, tokens, C] or [B, C] depending on the stage
            # selected; mean-pool the grid so both give one vector per image.
            if output.ndim == 3:
                output = output.mean(dim=1)
            chunks.append(output.float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def stratified_readout(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict[str, float]:
    """Crop-level stratified split, the pipeline's primary protocol."""
    from sklearn.model_selection import train_test_split

    train_index, test_index = train_test_split(
        np.arange(labels.size), test_size=test_size, stratify=labels, random_state=seed
    )
    fit_index, val_index = train_test_split(
        train_index, test_size=0.25, stratify=labels[train_index], random_state=seed
    )
    best_c = select_regularisation(
        features[fit_index], labels[fit_index],
        features[val_index], labels[val_index],
        num_classes=num_classes, grid=REGULARISATION_GRID,
    )["best_C"]
    probe = linear_probe(
        features[train_index], labels[train_index],
        features[test_index], labels[test_index],
        num_classes=num_classes, regularisation=best_c,
    )
    neighbours = knn_classifier(
        features[train_index], labels[train_index],
        features[test_index], labels[test_index],
        num_classes=num_classes, k=20,
    )
    return {
        "probe_accuracy": probe["accuracy"],
        "probe_f1_macro": probe["f1_macro"],
        "knn_accuracy": neighbours["accuracy"],
        "knn_f1_macro": neighbours["f1_macro"],
        "regularisation": float(best_c),
        "train_size": probe["train_size"],
        "test_size": probe["test_size"],
    }


def grouped_cv_readout(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    num_classes: int,
    folds: int = 5,
    seed: int = 42,
) -> dict[str, float]:
    """Photograph-disjoint out-of-fold readout.

    Every crop is predicted by a model that never saw its source photograph, and
    the per-fold predictions are concatenated into one out-of-fold set covering
    every crop and every class -- which is what makes it comparable across
    corpora whose photograph counts differ. A single grouped split is not: it
    holds out 20 % of the photographs unstratified, and on this data that leaves
    roughly half the classes absent from the test side.
    """
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedGroupKFold

    splitter = StratifiedGroupKFold(
        n_splits=int(folds), shuffle=True, random_state=int(seed)
    )
    folds_list = list(splitter.split(features, labels, groups))
    # `C` is selected once, on the first fold's held-out half, and then reused
    # for every fold. Selecting it per fold would make the out-of-fold score an
    # optimistic estimate of a procedure nobody ran.
    first_train, first_test = folds_list[0]
    best_c = select_regularisation(
        features[first_train], labels[first_train],
        features[first_test], labels[first_test],
        num_classes=num_classes, grid=REGULARISATION_GRID,
    )["best_C"]
    predictions = np.full(labels.shape, -1, dtype=np.int64)
    for train_index, test_index in folds_list:
        outcome = linear_probe(
            features[train_index], labels[train_index],
            features[test_index], labels[test_index],
            num_classes=num_classes, regularisation=best_c,
        )
        predictions[test_index] = outcome["predictions"]
    covered = predictions >= 0
    return {
        "regularisation": float(best_c),
        "probe_accuracy": float(accuracy_score(labels[covered], predictions[covered])),
        "probe_f1_macro": float(
            f1_score(labels[covered], predictions[covered], average="macro", zero_division=0)
        ),
        "folds": int(folds),
        "num_groups": int(np.unique(groups).size),
        "coverage": float(covered.mean()),
    }


def match_to_reference(
    labels: np.ndarray,
    paths: list[str],
    reference_counts: dict[int, int],
    reference_sources: set[str] | None,
    seed: int = 0,
) -> np.ndarray:
    """Indices of a subset matching a reference corpus in size and provenance.

    Restricts to the reference's source photographs when asked, then draws each
    class down to the reference's own count for that class. Drawn without
    replacement from a seeded generator, so the control is reproducible.

    A class with fewer crops than the reference keeps all of them; the shortfall
    is reported by the caller through the resulting size, never silently padded.
    """
    keep = np.ones(labels.shape, dtype=bool)
    if reference_sources is not None:
        keep = np.array(
            [source_image_id(path).split("/")[-1] in reference_sources for path in paths]
        )
    rng = np.random.default_rng(int(seed))
    chosen: list[int] = []
    for label in np.unique(labels):
        pool = np.flatnonzero(keep & (labels == label))
        wanted = int(reference_counts.get(int(label), pool.size))
        if pool.size <= wanted:
            chosen.extend(pool.tolist())
            continue
        chosen.extend(rng.choice(pool, size=wanted, replace=False).tolist())
    return np.sort(np.asarray(chosen, dtype=np.int64))


def evaluate(
    name: str,
    root: Path,
    encoders: list[str],
    protocols: list[str],
    device: torch.device,
    args: argparse.Namespace,
    subset: np.ndarray | None = None,
    subset_label: str = "",
    cache: dict[str, np.ndarray] | None = None,
) -> list[dict[str, object]]:
    """Every (encoder, protocol) score for one corpus, or one subset of it."""
    dataset, labels, groups, paths = load_corpus(root, args.batch_size, args.num_workers)
    num_classes = len(dataset.subvariety_to_idx)
    rows: list[dict[str, object]] = []

    for encoder in encoders:
        key = f"{root}:{encoder}"
        if cache is not None and key in cache:
            features = cache[key]
        else:
            started = time.time()
            if encoder == "handcrafted":
                features = handcrafted_image_features(paths)
            else:
                features = frozen_features(
                    root, encoder == "imagenet_init", device,
                    args.batch_size, args.num_workers, args.seed,
                )
            LOGGER.info(
                "  %-14s %-14s features %s in %.0f s", name, encoder,
                features.shape, time.time() - started,
            )
            if cache is not None:
                cache[key] = features

        use = np.arange(labels.size) if subset is None else subset
        for protocol in protocols:
            if protocol == "stratified":
                scores = stratified_readout(
                    features[use], labels[use], num_classes, args.test_size, args.seed
                )
            else:
                scores = grouped_cv_readout(
                    features[use], labels[use], groups[use], num_classes, args.folds, args.seed
                )
            rows.append(
                {
                    "corpus": name + (f" [{subset_label}]" if subset_label else ""),
                    "root": str(root),
                    "subset": subset_label or "full",
                    "encoder": encoder,
                    "protocol": protocol,
                    "num_images": int(use.size),
                    "num_photographs": int(np.unique(groups[use]).size),
                    "feature_dim": int(features.shape[1]),
                    **scores,
                }
            )
            LOGGER.info(
                "    %-12s %-12s probe %.4f  f1 %.4f%s", encoder, protocol,
                scores["probe_accuracy"], scores["probe_f1_macro"],
                f"  knn {scores['knn_accuracy']:.4f}" if "knn_accuracy" in scores else "",
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    default_root = os.environ.get(
        "SEED_DATA_ROOT", "data/Hierarchical_SeedData/Cropped_Samples"
    )
    parser.add_argument("--legacy", default=default_root, help="The corpus being replaced.")
    parser.add_argument(
        "--refined",
        default=os.environ.get(
            "SEED_REFINED_DATA_ROOT", "data/Hierarchical_SeedData/Refined_Samples"
        ),
        help="The corpus produced by `python -m src.segmentation.extract`.",
    )
    parser.add_argument(
        "--encoders", nargs="+", default=["handcrafted", "imagenet_init", "random_init"],
        choices=["handcrafted", "imagenet_init", "random_init"],
    )
    parser.add_argument(
        "--protocols", nargs="+", default=["stratified", "grouped_cv"],
        choices=["stratified", "grouped_cv"],
    )
    parser.add_argument("--match-size", action="store_true", default=True)
    parser.add_argument("--no-match-size", dest="match_size", action="store_false")
    parser.add_argument("--match-sources", action="store_true", default=True)
    parser.add_argument("--no-match-sources", dest="match_sources", action="store_false")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--json",
        default=f"{os.environ.get('SEED_OUTPUT_DIR', 'outputs')}/segmentation/corpus_benchmark.json",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from src.utils.training import select_device

    device = select_device("auto")
    LOGGER.info("Device: %s", device)

    legacy_root = Path(args.legacy).expanduser()
    refined_root = Path(args.refined).expanduser()
    for root in (legacy_root, refined_root):
        if not root.exists():
            LOGGER.error("Corpus not found: %s", root)
            return 1

    cache: dict[str, np.ndarray] = {}
    rows = evaluate("legacy", legacy_root, args.encoders, args.protocols, device, args, cache=cache)
    rows += evaluate("refined", refined_root, args.encoders, args.protocols, device, args, cache=cache)

    if args.match_size or args.match_sources:
        _, legacy_labels, _, legacy_paths = load_corpus(legacy_root)
        _, refined_labels, refined_groups, refined_paths = load_corpus(refined_root)
        counts: dict[int, int] = defaultdict(int)
        for label in legacy_labels:
            counts[int(label)] += 1
        sources = (
            {source_image_id(path).split("/")[-1] for path in legacy_paths}
            if args.match_sources
            else None
        )
        subset = match_to_reference(
            refined_labels, refined_paths,
            dict(counts) if args.match_size else {},
            sources, args.seed,
        )
        label = "size+source matched" if (args.match_size and args.match_sources) else (
            "size matched" if args.match_size else "source matched"
        )
        LOGGER.info(
            "Matched control: %s of %s refined crops from %s photographs.",
            subset.size, refined_labels.size, np.unique(refined_groups[subset]).size,
        )
        rows += evaluate(
            "refined", refined_root, args.encoders, args.protocols, device, args,
            subset=subset, subset_label=label, cache=cache,
        )

    destination = Path(args.json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    LOGGER.info("\n%s", "=" * 104)
    LOGGER.info(
        "%-30s %-14s %-12s %7s %6s %9s %9s %9s",
        "corpus", "encoder", "protocol", "images", "photos", "probe", "macro F1", "kNN",
    )
    LOGGER.info("%s", "-" * 104)
    for row in rows:
        LOGGER.info(
            "%-30s %-14s %-12s %7d %6d %9.4f %9.4f %9s",
            row["corpus"], row["encoder"], row["protocol"], row["num_images"],
            row["num_photographs"], row["probe_accuracy"], row["probe_f1_macro"],
            f"{row['knn_accuracy']:.4f}" if "knn_accuracy" in row else "-",
        )
    LOGGER.info("%s\nWrote %s", "=" * 104, destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
