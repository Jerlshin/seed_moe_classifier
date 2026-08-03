"""Dataset label construction and the DINO multi-crop augmentation pipeline."""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from src.datasets.dataset import HierarchicalSeedDataset
from src.datasets.transforms import (
    DataAugmentationDINO,
    get_dino_transforms,
    get_supervised_transforms,
)
from tests.conftest import (
    PAPER_GLOBAL_CROPS,
    PAPER_LOCAL_CROPS,
    PAPER_NUM_SEED_TYPES,
    PAPER_NUM_SUB_VARIETIES,
    IMAGES_PER_SUBVARIETY,
    SUBVARIETY_COUNTS,
)


@pytest.fixture
def dataset(synthetic_dataset_root) -> HierarchicalSeedDataset:
    return HierarchicalSeedDataset(root_dir=synthetic_dataset_root)


def test_dataset_discovers_the_paper_hierarchy(dataset):
    """4 seed types and 27 sub-varieties, matching Section 3."""
    assert len(dataset.seed_type_to_idx) == PAPER_NUM_SEED_TYPES
    assert len(dataset.subvariety_to_idx) == PAPER_NUM_SUB_VARIETIES
    assert len(dataset) == PAPER_NUM_SUB_VARIETIES * IMAGES_PER_SUBVARIETY


def test_labels_follow_sorted_directory_order(dataset):
    assert list(dataset.seed_type_to_idx) == sorted(SUBVARIETY_COUNTS)
    assert list(dataset.seed_type_to_idx.values()) == list(range(PAPER_NUM_SEED_TYPES))


def test_sub_variety_labels_are_global_not_per_seed_type(dataset):
    """Indices run 0..26 across all seed types, not 0..12 within rice."""
    assert sorted(dataset.subvariety_to_idx.values()) == list(range(PAPER_NUM_SUB_VARIETIES))


def test_subvariety_to_seed_type_matches_the_paper_split(dataset):
    mapping = dataset.get_subvariety_to_seed_type()
    assert len(mapping) == PAPER_NUM_SUB_VARIETIES
    counts = [mapping.count(index) for index in range(PAPER_NUM_SEED_TYPES)]
    assert counts == [SUBVARIETY_COUNTS[name] for name in sorted(SUBVARIETY_COUNTS)]


def test_every_sample_label_pair_is_hierarchically_consistent(dataset):
    mapping = dataset.get_subvariety_to_seed_type()
    for _, seed_label, sub_label in dataset.samples:
        assert mapping[sub_label] == seed_label


def test_getitem_returns_image_and_both_labels(dataset):
    dataset.transform = get_supervised_transforms(image_size=16, train=False)
    image, seed_label, sub_label = dataset[0]
    assert image.shape == (3, 16, 16)
    assert seed_label.dtype == torch.long
    assert sub_label.dtype == torch.long


def test_ordered_class_names_align_with_label_indices(dataset):
    seed_names, sub_names = dataset.get_ordered_class_names()
    assert len(seed_names) == PAPER_NUM_SEED_TYPES
    assert len(sub_names) == PAPER_NUM_SUB_VARIETIES
    for name, index in dataset.seed_type_to_idx.items():
        assert seed_names[index] == name


def test_class_distribution_totals_match_the_sample_count(dataset):
    seed_counts, sub_counts = dataset.class_distribution()
    assert sum(seed_counts.values()) == len(dataset)
    assert sum(sub_counts.values()) == len(dataset)
    assert all(count == IMAGES_PER_SUBVARIETY for count in sub_counts.values())


def test_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        HierarchicalSeedDataset(root_dir=tmp_path / "does_not_exist")


def test_empty_root_raises(tmp_path):
    (tmp_path / "Rice" / "Chinnar").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="No supported image files"):
        HierarchicalSeedDataset(root_dir=tmp_path)


def test_csv_export_lists_every_sample(dataset, tmp_path, synthetic_dataset_root):
    csv_path = tmp_path / "meta" / "seed_dataset.csv"
    HierarchicalSeedDataset(root_dir=synthetic_dataset_root, save_csv_path=csv_path)
    assert csv_path.exists()
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(dataset) + 1  # header + rows


# ---------------------------------------------------------------- transforms


def test_dino_augmentation_produces_the_paper_crop_count():
    """Table 1: 2 global crops + 4 local crops."""
    augmentation = DataAugmentationDINO(image_size=32, local_crop_size=16)
    assert augmentation.local_crops_number == PAPER_LOCAL_CROPS
    assert augmentation.num_crops == PAPER_GLOBAL_CROPS + PAPER_LOCAL_CROPS

    image = Image.fromarray((torch.rand(48, 48, 3) * 255).to(torch.uint8).numpy(), mode="RGB")
    original, crops = augmentation(image)
    assert original.shape == (3, 32, 32)
    assert len(crops) == PAPER_GLOBAL_CROPS + PAPER_LOCAL_CROPS


def test_local_crops_are_resized_to_the_global_size_for_swin():
    """SwinV2 needs a fixed input size, so every view lands at image_size."""
    augmentation = DataAugmentationDINO(
        image_size=32, local_crop_size=16, resize_local_to_global=True
    )
    image = Image.fromarray((torch.rand(48, 48, 3) * 255).to(torch.uint8).numpy(), mode="RGB")
    _, crops = augmentation(image)
    assert all(crop.shape == (3, 32, 32) for crop in crops)


def test_local_crops_keep_their_own_size_when_not_resized():
    augmentation = DataAugmentationDINO(
        image_size=32, local_crop_size=16, resize_local_to_global=False
    )
    image = Image.fromarray((torch.rand(48, 48, 3) * 255).to(torch.uint8).numpy(), mode="RGB")
    _, crops = augmentation(image)
    assert crops[0].shape == (3, 32, 32)
    assert crops[2].shape == (3, 16, 16)


def test_get_dino_transforms_accepts_the_config_node():
    augmentation_cfg = {
        "global_crops_scale": [0.4, 1.0],
        "local_crops_scale": [0.05, 0.4],
        "local_crops_number": 4,
        "color_jitter_prob": 0.8,
        "color_jitter_brightness": 0.4,
        "color_jitter_contrast": 0.4,
        "color_jitter_saturation": 0.2,
        "color_jitter_hue": 0.1,
        "grayscale_prob": 0.2,
    }
    transform = get_dino_transforms(32, 16, augmentation_cfg)
    assert transform.num_crops == 6


def test_supervised_transform_is_deterministic_without_flip():
    transform = get_supervised_transforms(image_size=16, train=False)
    image = Image.fromarray((torch.rand(24, 24, 3) * 255).to(torch.uint8).numpy(), mode="RGB")
    assert torch.allclose(transform(image), transform(image))


def test_negative_local_crop_count_is_rejected():
    with pytest.raises(ValueError, match="local_crops_number"):
        DataAugmentationDINO(image_size=32, local_crop_size=16, local_crops_number=-1)


# ---------------------------------------------------------------- provenance


def test_source_image_id_groups_crops_by_their_source_photograph():
    """``IMG_0502_bbox137.png`` and ``IMG_0502_bbox4.png`` are the same scene."""
    from src.datasets.dataset import source_image_id

    assert source_image_id("Millet/Baryard/IMG_0502_bbox137.png") == source_image_id(
        "Millet/Baryard/IMG_0502_bbox4.png"
    )
    assert source_image_id("Millet/Baryard/IMG_0502_bbox0.png") != source_image_id(
        "Millet/Baryard/IMG_0503_bbox0.png"
    )
    # Two sub-varieties may legitimately reuse a filename, so the key is scoped.
    assert source_image_id("Millet/Baryard/IMG_0502_bbox0.png") != source_image_id(
        "Millet/Browntop/IMG_0502_bbox0.png"
    )
    # No ``_bbox`` suffix means no known provenance: every file its own group,
    # which is exactly ungrouped splitting -- the right default when unknown.
    assert source_image_id("a/b/plain.png") != source_image_id("a/b/other.png")


def test_source_groups_are_dense_contiguous_indices(synthetic_dataset_root):
    from src.datasets.dataset import HierarchicalSeedDataset

    dataset = HierarchicalSeedDataset(root_dir=synthetic_dataset_root)
    groups = dataset.source_groups()

    assert groups.shape == (len(dataset),)
    assert groups.min() == 0
    assert set(groups.tolist()) == set(range(int(groups.max()) + 1))


def test_group_report_names_the_classes_no_split_can_separate(synthetic_dataset_root):
    """The number that bounds the whole protocol, surfaced rather than inferred.

    A sub-variety whose crops all come from one photograph cannot be
    group-separated at all: for those classes no honest train/test split exists,
    and their scores measure within-photograph generalisation whatever the
    splitter does. The paper has to say so, so the trainer reports it.
    """
    from src.datasets.dataset import HierarchicalSeedDataset

    dataset = HierarchicalSeedDataset(root_dir=synthetic_dataset_root)
    report = dataset.group_report()

    assert report["num_samples"] == len(dataset)
    assert report["num_source_groups"] >= 1
    assert report["mean_crops_per_source"] >= 1.0
    assert set(report["sources_per_sub_variety"]) == set(dataset.subvariety_to_idx)
    assert report["num_single_group_sub_varieties"] == len(report["single_group_sub_varieties"])


def test_supervised_transforms_resize_to_an_explicit_square():
    """``T.Resize(int)`` preserves aspect ratio; only 3.4 % of the crops are square.

    Passing an integer would resize the shorter side only, emit variable-width
    tensors, and make ``default_collate`` raise on the first mixed batch. The
    tuple form squashes to a square instead -- a real distortion, chosen
    deliberately over a latent crash.
    """
    from PIL import Image

    from src.datasets.transforms import get_supervised_transforms

    transform = get_supervised_transforms(image_size=64, train=False)
    for size in ((21, 60), (60, 21), (48, 48)):
        tensor = transform(Image.new("RGB", size))
        assert tensor.shape == (3, 64, 64)


def test_stage_two_training_transform_is_stochastic_and_evaluation_is_not():
    """The submitted default saw each image once per epoch, deterministically."""
    from PIL import Image

    from src.datasets.transforms import get_supervised_transforms

    # A textured image: a uniform one is invariant to crop and flip, so it would
    # make this test pass for the wrong reason.
    import numpy as np

    pixels = np.tile(np.arange(40, dtype=np.uint8)[None, :, None], (55, 1, 3))
    image = Image.fromarray(pixels, mode="RGB")
    train = get_supervised_transforms(image_size=32, train=True)
    evaluate = get_supervised_transforms(image_size=32, train=False)

    torch.manual_seed(0)
    samples = [train(image) for _ in range(8)]
    assert any(not torch.equal(samples[0], other) for other in samples[1:])

    # Evaluation must never be stochastic, or val/test numbers become
    # augmentation noise rather than a measurement.
    assert torch.equal(evaluate(image), evaluate(image))
