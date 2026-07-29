"""Dump frozen-backbone embeddings for the whole dataset to a compressed ``.npz``.

Bypasses the Hydra trainers entirely, so it is the fastest way to sanity-check a
DINO-pretrained checkpoint: cluster the embeddings, run a linear probe, or
reproduce the paper's t-SNE figures (Figs. 8-9) without launching a training run.

    python scripts/extract_features.py \
        --data-root  $SEED_DATA_ROOT \
        --checkpoint $SEED_PRETRAIN_BACKBONE \
        --max-samples 2000

These are the **backbone's** features at its native width (1024 for SwinV2-Base),
*not* the paper's ``z in R^384``. The projection to ``z`` is a trained layer
belonging to :class:`~src.models.builder.DinoV2SwinV2Encoder`, and a
freshly-initialised copy of it would project the features through random
weights -- worse than useless for inspecting what pretraining learned. To get
real 384-D embeddings, load a finished stage-2 checkpoint, or read the
``embeddings`` array from a run's ``test_predictions.npz``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.dataset import get_finetune_dataset
from src.datasets.transforms import get_supervised_transforms
from src.models.builder import BackboneFeatureExtractor
from src.utils.training import select_device


def parse_args() -> argparse.Namespace:
    output_root = os.environ.get("SEED_OUTPUT_DIR", "outputs")
    parser = argparse.ArgumentParser(
        description="Extract frozen backbone features for the seed dataset.",
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("SEED_DATA_ROOT", "data/Hierarchical_SeedData/Cropped_Samples"),
        help="Dataset root containing <seed_type>/<sub_variety>/*.png.",
    )
    parser.add_argument("--output", default=f"{output_root}/features/seed_features.npz")
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get(
            "SEED_PRETRAIN_BACKBONE",
            f"{output_root}/checkpoints/dinov2_swinv2_pretrained.pth",
        ),
        help="DINO-pretrained backbone weights. Pass 'none' to skip loading.",
    )
    parser.add_argument(
        "--backbone-name",
        default="swinv2_base_window16_256",
        help="SwinV2 variant. Must match the checkpoint's architecture.",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = select_device(args.device)

    dataset = get_finetune_dataset(
        args.data_root,
        transform=get_supervised_transforms(image_size=args.image_size, train=False),
        include_path=True,
    )
    seed_names, sub_names = dataset.get_ordered_class_names()
    print(
        f"Loaded {len(dataset)} samples: {len(seed_names)} seed types, "
        f"{len(sub_names)} sub-varieties."
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    checkpoint = None if str(args.checkpoint).lower() == "none" else args.checkpoint
    extractor = BackboneFeatureExtractor(
        model_name=args.backbone_name,
        checkpoint_path=checkpoint,
        pretrained=False,
        dynamic_img_size=True,
        strict=False,
        freeze=True,
    ).to(device)
    if extractor.load_report is not None:
        report = extractor.load_report
        print(
            f"Checkpoint loaded: {len(report['missing_keys'])} missing, "
            f"{len(report['unexpected_keys'])} unexpected keys."
        )

    features: list[np.ndarray] = []
    seed_labels: list[np.ndarray] = []
    sub_labels: list[np.ndarray] = []
    paths: list[str] = []
    seen = 0

    with torch.no_grad():
        for images, seed, sub, batch_paths in loader:
            images = images.to(device, non_blocking=True)
            features.append(extractor(images).detach().cpu().numpy())
            seed_labels.append(seed.numpy())
            sub_labels.append(sub.numpy())
            paths.extend(batch_paths)
            seen += len(batch_paths)
            if args.max_samples is not None and seen >= args.max_samples:
                break

    limit = args.max_samples
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=np.concatenate(features, axis=0)[:limit],
        seed_labels=np.concatenate(seed_labels, axis=0)[:limit],
        subvariety_labels=np.concatenate(sub_labels, axis=0)[:limit],
        paths=np.array(paths[:limit], dtype=object),
        seed_type_names=np.array(seed_names, dtype=object),
        subvariety_names=np.array(sub_names, dtype=object),
        subvariety_to_seed_type=np.array(dataset.get_subvariety_to_seed_type()),
    )
    print(f"Saved {min(seen, limit or seen)} feature vectors to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
