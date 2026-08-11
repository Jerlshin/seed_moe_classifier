from __future__ import annotations

import bisect
import csv
import glob
import logging
import multiprocessing
import os
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

LOGGER = logging.getLogger(__name__)

#: Filenames under ``Cropped_Samples`` look like ``IMG_0502_bbox137.png``: many
#: crops cut from one source photograph of a tray of seeds. The stem before
#: ``_bbox<n>`` is therefore the provenance key, and crops sharing it are
#: near-duplicates of the same physical scene -- same lighting, same background,
#: same sensor noise, often the same individual seed photographed at overlapping
#: bounding boxes.
SOURCE_IMAGE_PATTERN = re.compile(r"^(?P<source>.+?)_bbox\d+$", re.IGNORECASE)


def source_image_id(image_path: str | os.PathLike[str]) -> str:
    """Provenance key for one crop: ``<sub_variety>/<source photograph>``.

    Splitting at the image level when many crops share a source photograph puts
    near-duplicates on both sides of the train/test boundary, so the reported
    accuracy measures memorisation of source-specific cues rather than
    sub-variety discrimination. The key is scoped by sub-variety directory
    because two different sub-varieties can legitimately reuse a filename.

    Falls back to the file stem when the name carries no ``_bbox`` suffix, which
    makes every such file its own group -- i.e. the same behaviour as
    ungrouped splitting, which is the correct default when provenance is
    unknown.
    """
    path = Path(image_path)
    match = SOURCE_IMAGE_PATTERN.match(path.stem)
    source = match.group("source") if match else path.stem
    return f"{path.parent.name}/{source}"


class PretrainImageFolderDataset(ImageFolder):
    """``ImageFolder`` that returns ``(original, crops, target, path)``.

    Optionally holds every decoded image in one shared RAM buffer. That is worth
    doing here for a reason specific to this dataset: the crops under
    ``Cropped_Samples`` have a median size of 52 x 51 px, so all 9,357 of them
    decode to roughly **75 MB** of raw RGB. Without the cache, a 300-epoch run
    pays 2.8 million PNG decodes for a working set that fits in a rounding error
    of system memory -- and those decodes are on the dataloader workers, which
    are the processes currently failing to keep the GPU fed.

    The cache is a single contiguous ``uint8`` array plus an offset table rather
    than a list of arrays. Under ``fork`` the pages are shared copy-on-write
    across every worker, and one array means the reference counts the workers
    touch live in the parent's few metadata pages instead of being scattered
    through 9,357 object headers.
    """

    def __init__(
        self,
        root,
        transform=None,
        cache_images: bool = False,
        cache_limit_mb: float = 4096.0,
        logger: logging.Logger | None = None,
        **kwargs,
    ):
        super().__init__(root=root, transform=transform, **kwargs)
        self._cache_buffer: np.ndarray | None = None
        self._cache_meta: list[tuple[int, int, int]] | None = None
        self.cache_bytes = 0
        if cache_images:
            self._build_cache(float(cache_limit_mb), logger or LOGGER)

    def _build_cache(self, limit_mb: float, logger: logging.Logger) -> None:
        limit_bytes = int(limit_mb * 1024 * 1024)
        arrays: list[np.ndarray] = []
        meta: list[tuple[int, int, int]] = []
        offset = 0

        for path, _ in self.samples:
            with Image.open(path) as handle:
                array = np.asarray(handle.convert("RGB"), dtype=np.uint8)
            if array.ndim != 3 or array.shape[2] != 3:
                logger.warning("Skipping the image cache: %s is not HxWx3 RGB.", path)
                return
            if offset + array.nbytes > limit_bytes:
                logger.warning(
                    "Image cache disabled: the dataset exceeds data.cache_limit_mb=%s MB "
                    "after %s of %s images. Raise the limit or set data.cache_images=false.",
                    limit_mb, len(meta), len(self.samples),
                )
                return
            arrays.append(array)
            meta.append((offset, array.shape[0], array.shape[1]))
            offset += array.nbytes

        buffer = np.empty(offset, dtype=np.uint8)
        for array, (start, _, _) in zip(arrays, meta):
            buffer[start : start + array.nbytes] = array.reshape(-1)
        del arrays

        self._cache_buffer = buffer
        self._cache_meta = meta
        self.cache_bytes = offset
        logger.info(
            "Cached %s decoded images in %.1f MB of shared memory; workers will not touch disk.",
            len(meta), offset / 1024**2,
        )

    def _decode(self, index: int, path: str) -> Image.Image:
        if self._cache_buffer is None or self._cache_meta is None:
            return self.loader(path).convert("RGB")
        offset, height, width = self._cache_meta[index]
        view = self._cache_buffer[offset : offset + height * width * 3].reshape(height, width, 3)
        return Image.fromarray(view)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self._decode(index, path)
        if self.transform is None:
            original, crops = image, []
        else:
            original, crops = self.transform(image)
        return original, crops, target, path


class MultiCropBatch(NamedTuple):
    """One collated multi-crop batch, kept **view-major**.

    ``global_views`` is ``[G, B, C, H, W]`` and ``local_views`` ``[L, B, C, h, w]``
    -- views on the outside, batch on the inside. That is not cosmetic:
    ``CustomDINOLoss`` chunks the student output into per-view blocks, so the
    model input must be ``cat([g0, g1, l0, l1, l2, l3])`` with each block holding
    the whole batch for one view. Flattening a batch-major layout instead would
    interleave views and silently score every cross-view pair against the wrong
    partner -- with a loss curve that looks entirely normal.

    ``local_views`` is ``None`` when ``local_crops_number == 0``; ``originals``
    is ``None`` whenever the augmentation was built with
    ``return_original=False``.
    """

    global_views: torch.Tensor
    local_views: torch.Tensor | None
    targets: torch.Tensor
    paths: tuple[str, ...]
    originals: torch.Tensor | None


class MultiCropCollate:
    """Collate ``(original, crops, target, path)`` samples into a :class:`MultiCropBatch`.

    Args:
        num_global_crops: How many leading views the teacher sees (paper: 2).
            The split matters because the two groups can have different spatial
            sizes when the local upsample is deferred to the GPU, and a single
            stacked tensor could not then hold both.
    """

    def __init__(self, num_global_crops: int = 2):
        self.num_global_crops = int(num_global_crops)

    def __call__(self, batch) -> MultiCropBatch:
        originals, crops, targets, paths = zip(*batch)
        num_views = len(crops[0])
        if num_views < self.num_global_crops:
            raise ValueError(
                f"Each sample must carry at least {self.num_global_crops} global views, got {num_views}."
            )
        if any(len(sample) != num_views for sample in crops):
            raise ValueError("Every sample must produce the same number of views.")

        def stack_views(view_indices: range) -> torch.Tensor:
            return torch.stack(
                [torch.stack([sample[view] for sample in crops]) for view in view_indices]
            )

        global_views = stack_views(range(self.num_global_crops))
        local_views = (
            stack_views(range(self.num_global_crops, num_views))
            if num_views > self.num_global_crops
            else None
        )
        stacked_originals = (
            torch.stack(originals) if all(item is not None for item in originals) else None
        )
        return MultiCropBatch(
            global_views=global_views,
            local_views=local_views,
            targets=torch.as_tensor(targets, dtype=torch.long),
            paths=tuple(str(item) for item in paths),
            originals=stacked_originals,
        )


class PickleBatchSeedDataset:
    """
    Notebook-compatible dataset for files containing flattened RGB image batches.

    Each batch file is expected to contain a pickled dict with at least a "data"
    key and optionally a "labels" key.
    """

    def __init__(self, root_dir: str, image_size: int, transform=None):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.transform = transform
        self.batch_paths = sorted(glob.glob(str(self.root_dir / "*_data_batch_*")))
        if not self.batch_paths:
            self.batch_paths = sorted(glob.glob(str(self.root_dir / "*.pkl")))
        if not self.batch_paths:
            raise FileNotFoundError(f"No pickle batch files found in {root_dir}")

        self.cumulative_sizes: list[int] = []
        self.batch_cache: dict[str, dict] = {}
        total_images = 0
        for batch_path in self.batch_paths:
            with open(batch_path, "rb") as file:
                batch_data = pickle.load(file)
            total_images += len(batch_data["data"])
            self.cumulative_sizes.append(total_images)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, index: int):
        batch_index = bisect.bisect_right(self.cumulative_sizes, index)
        batch_path = self.batch_paths[batch_index]
        if batch_path not in self.batch_cache:
            with open(batch_path, "rb") as file:
                self.batch_cache[batch_path] = pickle.load(file)
        batch_data = self.batch_cache[batch_path]
        local_index = index if batch_index == 0 else index - self.cumulative_sizes[batch_index - 1]

        image = Image.fromarray(self.format_image(batch_data["data"][local_index], self.image_size))
        label = batch_data.get("labels", [-1] * len(batch_data["data"]))[local_index]
        sample_id = f"{batch_path}:{local_index}"

        if self.transform is None:
            original, crops = image, []
        else:
            original, crops = self.transform(image)
        return original, crops, label, sample_id

    @staticmethod
    def format_image(image_flat: np.ndarray, size: int) -> np.ndarray:
        pixels_per_channel = size * size
        image = np.zeros((size, size, 3), dtype=image_flat.dtype)
        for channel in range(3):
            start = channel * pixels_per_channel
            end = (channel + 1) * pixels_per_channel
            image[:, :, channel] = image_flat[start:end].reshape(size, size)
        return image


class HierarchicalSeedDataset(Dataset):
    """
    Supervised seed dataset with notebook-compatible hierarchical labels.

    The expected layout is:

        root/
          seed_type/
            sub_variety/
              image files...

    Labels are assigned deterministically from sorted directory names. Sub-variety
    labels are global, matching the finetuning notebook.
    """

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(
        self,
        root_dir: str | os.PathLike[str],
        transform=None,
        save_csv_path: str | os.PathLike[str] | None = None,
        include_path: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.save_csv_path = Path(save_csv_path) if save_csv_path else None
        self.include_path = include_path

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.root_dir}")

        self.seed_type_to_idx: dict[str, int] = {}
        self.subvariety_to_idx: dict[str, int] = {}
        self.samples: list[tuple[str, int, int]] = []
        self.subvariety_to_seed_type: dict[int, int] = {}

        self._load_dataset()
        if not self.samples:
            raise FileNotFoundError(f"No supported image files found under {self.root_dir}")
        if self.save_csv_path is not None:
            self._save_to_csv()

    def _load_dataset(self) -> None:
        seed_type_dirs = sorted(path for path in self.root_dir.iterdir() if path.is_dir())
        self.seed_type_to_idx = {path.name: index for index, path in enumerate(seed_type_dirs)}

        for seed_type_dir in seed_type_dirs:
            seed_label = self.seed_type_to_idx[seed_type_dir.name]
            for subvariety_dir in sorted(path for path in seed_type_dir.iterdir() if path.is_dir()):
                subvariety_name = subvariety_dir.name
                if subvariety_name not in self.subvariety_to_idx:
                    self.subvariety_to_idx[subvariety_name] = len(self.subvariety_to_idx)
                sub_label = self.subvariety_to_idx[subvariety_name]
                self.subvariety_to_seed_type[sub_label] = seed_label

                for image_path in sorted(subvariety_dir.iterdir()):
                    if image_path.is_file() and image_path.suffix.lower() in self.image_extensions:
                        self.samples.append((str(image_path), seed_label, sub_label))

    def _save_to_csv(self) -> None:
        self.save_csv_path.parent.mkdir(parents=True, exist_ok=True)
        seed_idx_to_label, sub_idx_to_label = self.get_idx_to_label()
        with self.save_csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "image_path",
                    "seed_type",
                    "seed_type_label",
                    "subvariety",
                    "subvariety_label",
                ],
            )
            writer.writeheader()
            for image_path, seed_label, sub_label in self.samples:
                writer.writerow(
                    {
                        "image_path": image_path,
                        "seed_type": seed_idx_to_label[seed_label],
                        "seed_type_label": seed_label,
                        "subvariety": sub_idx_to_label[sub_label],
                        "subvariety_label": sub_label,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, seed_label, sub_label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        item = (
            image,
            torch.tensor(seed_label, dtype=torch.long),
            torch.tensor(sub_label, dtype=torch.long),
        )
        if self.include_path:
            return (*item, image_path)
        return item

    def get_class_mappings(self) -> tuple[dict[str, int], dict[str, int]]:
        """``(seed_type_to_idx, subvariety_to_idx)`` name -> index mappings."""
        return self.seed_type_to_idx, self.subvariety_to_idx

    def get_idx_to_label(self) -> tuple[dict[int, str], dict[int, str]]:
        """``(seed_idx_to_name, sub_idx_to_name)`` inverse mappings."""
        seed_type_idx_to_label = {value: key for key, value in self.seed_type_to_idx.items()}
        subvariety_idx_to_label = {value: key for key, value in self.subvariety_to_idx.items()}
        return seed_type_idx_to_label, subvariety_idx_to_label

    def get_ordered_class_names(self) -> tuple[list[str], list[str]]:
        """Class names ordered by label index, for metric tables and figure axes."""
        seed_names = [name for name, _ in sorted(self.seed_type_to_idx.items(), key=lambda kv: kv[1])]
        sub_names = [name for name, _ in sorted(self.subvariety_to_idx.items(), key=lambda kv: kv[1])]
        return seed_names, sub_names

    def get_subvariety_to_seed_type(self) -> list[int]:
        """Parent seed-type index for each sub-variety index, ordered by index.

        This is what builds the KL aggregation matrix ``M`` (Eq. 10) and what
        the alignment-rate metric uses, so it is derived from the directory tree
        rather than hardcoded.
        """
        return [self.subvariety_to_seed_type[index] for index in range(len(self.subvariety_to_idx))]

    def class_distribution(self) -> tuple[dict[str, int], dict[str, int]]:
        """Sample counts per seed type and per sub-variety (paper Fig. 1)."""
        seed_names, sub_names = self.get_ordered_class_names()
        seed_counts = dict.fromkeys(seed_names, 0)
        sub_counts = dict.fromkeys(sub_names, 0)
        for _, seed_label, sub_label in self.samples:
            seed_counts[seed_names[seed_label]] += 1
            sub_counts[sub_names[sub_label]] += 1
        return seed_counts, sub_counts

    def source_groups(self) -> np.ndarray:
        """Integer group id per sample, keyed on the source photograph.

        This is the group key ``StratifiedGroupKFold`` needs. Without it the
        split is at the crop level, and crops of the same physical seed under the
        same lighting land on both sides of the boundary.
        """
        keys = [source_image_id(path) for path, _, _ in self.samples]
        ordering = {key: index for index, key in enumerate(sorted(set(keys)))}
        return np.array([ordering[key] for key in keys], dtype=np.int64)

    def group_report(self) -> dict[str, object]:
        """Diagnostics describing how much provenance the tree actually has.

        Reported at the top of every run because it decides what the headline
        accuracy can mean. ``single_group_sub_varieties`` is the number that
        bounds the whole protocol: a class whose crops all come from **one**
        photograph cannot be group-separated at all, so for those classes no
        honest train/test split exists and their scores measure within-photo
        generalisation whatever the splitter does.
        """
        _, sub_names = self.get_ordered_class_names()
        groups = self.source_groups()
        per_sub: dict[str, set[int]] = {name: set() for name in sub_names}
        for (_, _, sub_label), group in zip(self.samples, groups):
            per_sub[sub_names[sub_label]].add(int(group))

        sizes = Counter(groups.tolist())
        singletons = sorted(name for name, ids in per_sub.items() if len(ids) < 2)
        return {
            "num_samples": len(self.samples),
            "num_source_groups": len(sizes),
            "mean_crops_per_source": len(self.samples) / max(len(sizes), 1),
            "min_crops_per_source": min(sizes.values()) if sizes else 0,
            "max_crops_per_source": max(sizes.values()) if sizes else 0,
            "sources_per_sub_variety": {name: len(ids) for name, ids in per_sub.items()},
            "single_group_sub_varieties": singletons,
            "num_single_group_sub_varieties": len(singletons),
        }


def get_pretrain_dataloader(
    data_dir: str,
    transform,
    batch_size: int,
    num_workers: int = 4,
    dataset_format: str = "image_folder",
    image_size: int = 256,
    pin_memory: bool = True,
    drop_last: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    num_global_crops: int = 2,
    multicrop_collate: bool = True,
    cache_images: bool = False,
    cache_limit_mb: float = 4096.0,
    generator: torch.Generator | None = None,
    logger: logging.Logger | None = None,
):
    """Build the stage-1 multi-crop loader.

    Args:
        persistent_workers: Keep worker processes alive between epochs. With 584
            batches per epoch and 300+ epochs, respawning workers every epoch
            re-imports torch and re-forks the dataset several hundred times, and
            each respawn also throws away the prefetch queue -- so every epoch
            starts with the GPU idle.
        prefetch_factor: Batches each worker runs ahead. The default of 2 is not
            enough cover here: one sample costs six independent PIL pipelines
            (crop, jitter, grayscale, blur, solarize), so per-batch CPU time is
            high and bursty, and a shallow queue drains during any hiccup.
        multicrop_collate: Emit :class:`MultiCropBatch` instead of the generic
            nested list ``default_collate`` produces. Required by the fused
            single-pass forward in the trainer.
        cache_images: Hold every decoded image in RAM; see
            :class:`PretrainImageFolderDataset`. Silently ignored when workers
            would not share it (see below).
        generator: Seeds the shuffling order.

    The cache is disabled automatically when ``num_workers > 0`` and the start
    method is not ``fork``: under ``spawn`` (macOS default, Windows always) the
    dataset is pickled into each worker, so a "shared" cache becomes one full
    copy per worker plus the pickling time, which is worse than decoding.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset path not found: {data_dir}")

    log = logger or LOGGER
    num_workers = max(int(num_workers), 0)

    if cache_images and num_workers > 0:
        start_method = multiprocessing.get_start_method(allow_none=True) or "fork"
        if start_method != "fork":
            log.warning(
                "Disabling the image cache: the %r start method copies the dataset into every "
                "worker instead of sharing it. Set data.num_workers=0 to cache anyway.",
                start_method,
            )
            cache_images = False

    if dataset_format == "image_folder":
        dataset = PretrainImageFolderDataset(
            root=data_dir,
            transform=transform,
            cache_images=cache_images,
            cache_limit_mb=cache_limit_mb,
            logger=log,
        )
    elif dataset_format == "pickle_batches":
        dataset = PickleBatchSeedDataset(root_dir=data_dir, image_size=image_size, transform=transform)
    else:
        raise ValueError(f"Unsupported pretraining dataset format: {dataset_format}")

    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        loader_kwargs["prefetch_factor"] = max(int(prefetch_factor), 1)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=MultiCropCollate(num_global_crops) if multicrop_collate else None,
        generator=generator,
        **loader_kwargs,
    )


def get_finetune_dataset(
    data_dir: str,
    transform,
    save_csv_path: str | None = None,
    include_path: bool = False,
) -> HierarchicalSeedDataset:
    return HierarchicalSeedDataset(
        root_dir=data_dir,
        transform=transform,
        save_csv_path=save_csv_path,
        include_path=include_path,
    )
