from __future__ import annotations

import bisect
import csv
import glob
import hashlib
import logging
import multiprocessing
import os
import pickle
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision.datasets import ImageFolder

LOGGER = logging.getLogger(__name__)

#: Filenames under ``Cropped_Samples`` look like ``IMG_0502_bbox137.png``: many
#: crops cut from one source photograph of a tray of seeds. The stem before
#: ``_bbox<n>`` is therefore the provenance key, and crops sharing it are
#: near-duplicates of the same physical scene -- same lighting, same background,
#: same sensor noise, often the same individual seed photographed at overlapping
#: bounding boxes.
SOURCE_IMAGE_PATTERN = re.compile(r"^(?P<source>.+?)_bbox\d+$", re.IGNORECASE)

#: File extensions the corpus fingerprint and the raw-photograph audit consider.
IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
)


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


def image_sizes(paths: Sequence[str | os.PathLike[str]]) -> list[tuple[int, int]]:
    """``(width, height)`` for each path, from the file header only.

    ``PIL.Image.open`` parses the header and defers the pixel decode, so this is
    ~10 us per file rather than ~1 ms. That matters because the caller is the
    trainer's startup path and the corpus is 9,357 files: a header read is a
    second, a decode is a minute.

    Unreadable files are skipped rather than raising. A geometry *report* that
    fails because one file is corrupt would abort a run that the dataloader is
    perfectly able to complete, and the count of what was measured is returned
    alongside the percentiles anyway.
    """
    sizes: list[tuple[int, int]] = []
    for path in paths:
        try:
            with Image.open(path) as image:
                sizes.append((int(image.size[0]), int(image.size[1])))
        except Exception:  # pragma: no cover - depends on a corrupt file
            continue
    return sizes


def corpus_fingerprint(
    root: str | os.PathLike[str],
    relative_paths: Sequence[str] | None = None,
) -> dict[str, object]:
    """A stable, machine-readable identity for the corpus a stage actually read.

    Why this exists. The stage-1 -> stage-2 handoff is a bare ``state_dict`` with
    no record of what it was trained on, and that gap was not hypothetical: the
    shipped 100-epoch encoder was self-distilled on **8,173** crops while the
    evaluation, stage 2 and every published number use **9,357**. Nothing on disk
    said so, and it was recoverable only by cross-reading two log lines against
    ``metrics.json``. ``provenance.json`` already records the checkpoint's
    SHA-256, the git commit and the library versions; the one thing it could not
    recover is the corpus.

    The digest is over the sorted list of dataset-relative POSIX paths, so it is
    stable across machines, mount points and filesystem enumeration order, and it
    changes the moment a file is added, removed or renamed. Counts and the
    per-class histogram travel with it because a digest alone cannot say *how*
    two corpora differ.

    Args:
        root: Dataset root. Used to make the paths relative and to enumerate them
            when ``relative_paths`` is not supplied.
        relative_paths: Pre-enumerated paths (absolute or relative). Pass the
            dataset's own sample list so the fingerprint describes what the
            dataset will actually read rather than what the directory happens to
            contain.

    Returns:
        ``num_samples``, ``num_classes``, ``num_source_groups``,
        ``samples_per_class`` (sorted by name), ``sha256`` and ``root``.
    """
    base = Path(root)
    if relative_paths is None:
        candidates = [
            path
            for path in sorted(base.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        names = sorted(path.relative_to(base).as_posix() for path in candidates)
    else:
        names = sorted(
            (
                Path(item).relative_to(base).as_posix()
                if Path(item).is_absolute()
                else Path(item).as_posix()
            )
            for item in relative_paths
        )

    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("utf-8"))
        digest.update(b"\n")

    per_class = Counter(
        Path(name).parent.as_posix() or "." for name in names
    )
    groups = {source_image_id(name) for name in names}
    return {
        "root": str(base),
        "num_samples": len(names),
        "num_classes": len(per_class),
        "num_source_groups": len(groups),
        "samples_per_class": dict(sorted(per_class.items())),
        "sha256": digest.hexdigest(),
    }


def describe_fingerprint_mismatch(
    left: Mapping[str, object],
    right: Mapping[str, object],
    left_name: str = "stage 1",
    right_name: str = "this run",
) -> str:
    """One human-readable line saying how two corpora differ, or ``""`` if they do not.

    Returns the empty string when the digests match, so a caller can branch on
    truthiness and log nothing in the (normal) matching case.
    """
    if not left or not right:
        return ""
    if str(left.get("sha256")) == str(right.get("sha256")):
        return ""

    left_counts = dict(left.get("samples_per_class") or {})
    right_counts = dict(right.get("samples_per_class") or {})
    changed = sorted(
        key
        for key in set(left_counts) | set(right_counts)
        if left_counts.get(key, 0) != right_counts.get(key, 0)
    )
    detail = ", ".join(
        f"{key}: {left_counts.get(key, 0)} -> {right_counts.get(key, 0)}" for key in changed[:6]
    )
    if len(changed) > 6:
        detail += f", and {len(changed) - 6} more"
    return (
        f"CORPUS MISMATCH: {left_name} used {left.get('num_samples')} images from "
        f"{left.get('num_source_groups')} source photographs (digest "
        f"{str(left.get('sha256'))[:16]}), {right_name} has {right.get('num_samples')} from "
        f"{right.get('num_source_groups')} (digest {str(right.get('sha256'))[:16]})."
        + (f" Classes that differ -- {detail}." if detail else "")
    )


def raw_photograph_coverage(
    raw_root: str | os.PathLike[str],
    cropped_root: str | os.PathLike[str],
) -> dict[str, object]:
    """Which source photographs exist but were never cropped.

    The binding constraint on this dataset is the number of *scenes*, not the
    number of crops: 9,357 crops come from 81 photographs, and within one
    photograph 89-98 % of crops have a neighbour above cosine 0.95 at 32x32 grey.
    ``RAW_Samples`` holds 99 photographs, so 18 exist and were never cropped --
    which is +22 % scenes at zero acquisition cost, concentrated on classes that
    currently have two or three.

    This function only *reports*. Cropping happens outside this repository, and
    republishing the corpus moves every published accuracy, so it must be a
    deliberate re-baseline rather than a side effect of running a script. The
    corpus fingerprint above is what keeps the before and after distinguishable.

    Matching is on the file **stem**, scoped by sub-variety directory, which is
    the same key :func:`source_image_id` builds -- so a photograph counts as used
    exactly when some crop names it.

    Returns ``{}`` when ``raw_root`` does not exist, so a caller with no raw tree
    degrades to reporting nothing rather than raising.
    """
    raw = Path(raw_root)
    cropped = Path(cropped_root)
    if not raw.exists() or not cropped.exists():
        return {}

    used: dict[str, set[str]] = {}
    for path in sorted(cropped.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        match = SOURCE_IMAGE_PATTERN.match(path.stem)
        used.setdefault(path.parent.name, set()).add(
            (match.group("source") if match else path.stem)
        )

    per_class: dict[str, dict[str, object]] = {}
    total_raw = 0
    total_unused = 0
    for path in sorted(raw.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        sub_variety = path.parent.name
        entry = per_class.setdefault(
            sub_variety, {"raw_photographs": 0, "used": 0, "unused": []}
        )
        entry["raw_photographs"] = int(entry["raw_photographs"]) + 1
        total_raw += 1
        if path.stem in used.get(sub_variety, set()):
            entry["used"] = int(entry["used"]) + 1
        else:
            entry["unused"].append(path.stem)  # type: ignore[union-attr]
            total_unused += 1

    return {
        "raw_root": str(raw),
        "cropped_root": str(cropped),
        "num_raw_photographs": total_raw,
        "num_used_photographs": total_raw - total_unused,
        "num_unused_photographs": total_unused,
        "per_sub_variety": {name: per_class[name] for name in sorted(per_class)},
    }


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
        same_photo_local_views: int = 0,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(root=root, transform=transform, **kwargs)
        self._cache_buffer: np.ndarray | None = None
        self._cache_meta: list[tuple[int, int, int]] | None = None
        self.cache_bytes = 0
        self.same_photo_local_views = max(int(same_photo_local_views), 0)
        self._seed = int(seed)
        self._source_members: dict[str, list[int]] | None = None
        self._source_of: list[str] | None = None
        if self.same_photo_local_views:
            self._build_source_index()
        if cache_images:
            self._build_cache(float(cache_limit_mb), logger or LOGGER)

    # ------------------------------------------------- provenance positives

    def _build_source_index(self) -> None:
        """Map each source photograph to the indices of the crops cut from it."""
        self._source_of = [source_image_id(path) for path, _ in self.samples]
        members: dict[str, list[int]] = {}
        for index, key in enumerate(self._source_of):
            members.setdefault(key, []).append(index)
        self._source_members = members

    def corpus_fingerprint(self) -> dict[str, object]:
        """Identity of the corpus this dataset will actually read. See
        :func:`corpus_fingerprint`."""
        return corpus_fingerprint(self.root, [path for path, _ in self.samples])

    def source_groups(self) -> np.ndarray:
        """Integer group id per sample, keyed on the source photograph.

        The same key :meth:`HierarchicalSeedDataset.source_groups` builds, so a
        stage-1 group count and a stage-2 one are the same quantity.
        """
        keys = [source_image_id(path) for path, _ in self.samples]
        ordering = {key: index for index, key in enumerate(sorted(set(keys)))}
        return np.array([ordering[key] for key in keys], dtype=np.int64)

    def source_sizes(self) -> list[tuple[int, int]]:
        """``(width, height)`` of every crop, in dataset order.

        Read from the file headers, not by decoding: ``Image.open`` is lazy, so
        ``.size`` costs a few bytes per file and the whole 9,357-image corpus is
        a second. The view-geometry report needs the true source dimensions
        because ``RandomResizedCrop``'s ``scale`` is a *fraction* of them, and a
        fraction of an unknown quantity is not a measurement.
        """
        return image_sizes([path for path, _ in self.samples])

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

    def _partner_indices(self, index: int, count: int) -> list[int]:
        """``count`` other crops of the same source photograph, or ``[]``.

        Drawn from a generator seeded on ``(seed, index, epoch-free)``, so the
        partner set is a deterministic function of the sample rather than of
        worker scheduling -- two runs at one seed see the same pairs, and a
        dataloader worker cannot change them. A photograph with only one crop
        yields nothing and the caller falls back to augmenting the anchor, which
        keeps the view count constant.
        """
        if not count or self._source_members is None or self._source_of is None:
            return []
        pool = self._source_members.get(self._source_of[index], ())
        others = [item for item in pool if item != index]
        if not others:
            return []
        rng = np.random.default_rng((self._seed, int(index)))
        return [int(others[i]) for i in rng.integers(0, len(others), size=int(count))]

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self._decode(index, path)
        if self.transform is None:
            return image, [], target, path

        partners = self._partner_indices(index, self.same_photo_local_views)
        if not partners:
            original, crops = self.transform(image)
            return original, crops, target, path

        # F1: replace the trailing local views with *other crops of the same
        # photograph*, augmented by the same local pipeline. DINO's positives are
        # augmented views of one crop, so the invariance it teaches is to
        # crop/blur/colour; the invariance the downstream task needs is to *which
        # individual seed*, and two crops of one photograph are by construction
        # two individuals of the same variety. No labels are involved -- the
        # provenance key is parsed from the filename.
        #
        # The self-detecting risk is that same-photograph crops share lighting,
        # background and sensor noise, so the objective can be satisfied by
        # learning photograph identity -- exactly what the photograph-disjoint
        # protocol punishes. `nuisance_decodability` in the stage-1 evaluation is
        # the gate: if this arm raises within-class photograph decodability, it
        # learned the confound.
        partner_images = [self._decode(other, self.samples[other][0]) for other in partners]
        original, crops = self.transform(image, partner_images=partner_images)
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

    def source_sizes(self) -> list[tuple[int, int]]:
        """``(width, height)`` of every crop, in dataset order. See
        :func:`image_sizes`."""
        return image_sizes([path for path, _, _ in self.samples])

    def corpus_fingerprint(self) -> dict[str, object]:
        """Identity of the corpus this dataset reads. See :func:`corpus_fingerprint`.

        The stage-1 trainer records the same quantity for the corpus it
        self-distilled on, and ``pretrain_eval`` compares the two. That
        comparison is the only thing that would have caught the shipped
        encoder being trained on 8,173 crops while everything downstream used
        9,357.
        """
        return corpus_fingerprint(self.root_dir, [path for path, _, _ in self.samples])

    def group_report(self, raw_root: str | os.PathLike[str] | None = None) -> dict[str, object]:
        """Diagnostics describing how much provenance the tree actually has.

        Args:
            raw_root: Optional ``RAW_Samples`` tree. When supplied, the report
                additionally names the source photographs that exist and were
                never cropped -- 18 of 99 on the shipped corpus, i.e. +22 %
                scenes available at zero acquisition cost. Reporting only; see
                :func:`raw_photograph_coverage`.

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
        coverage = (
            raw_photograph_coverage(raw_root, self.root_dir) if raw_root else {}
        )
        return {
            **({"raw_photograph_coverage": coverage} if coverage else {}),
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
    world_size: int = 1,
    rank: int = 0,
    same_photo_local_views: int = 0,
    seed: int = 0,
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
        same_photo_local_views: Replace this many local views with crops of
            *other crops from the same source photograph* -- provenance-derived
            positives. ``0`` (the default) is plain DINO multi-crop
            and is what every published number was produced under. Only the
            ``image_folder`` format carries the provenance needed for it.
        seed: Seeds the per-sample partner draw, so the pairing is a
            deterministic function of the sample rather than of which worker
            happened to build it.
        generator: Seeds the shuffling order.
        world_size / rank: Shard the dataset with a ``DistributedSampler`` when
            ``world_size > 1``. **Images** are sharded, never views: every view
            of one image is built by the rank that owns it, because the loss
            pairs a student view against the teacher's output for the same
            image.

    The cache is disabled automatically when ``num_workers > 0`` and the start
    method is not ``fork``: under ``spawn`` (macOS default, Windows always) the
    dataset is pickled into each worker, so a "shared" cache becomes one full
    copy per worker plus the pickling time, which is worse than decoding.

    Under DDP the sampler is returned as ``loader.sampler`` and the trainer
    **must** call ``set_epoch`` on it once per epoch. A ``DistributedSampler``
    derives its permutation from ``seed + epoch``, so skipping that call gives
    every epoch the identical sample order -- which does not error, does not
    change the loss magnitude, and quietly turns a 300-epoch run into one epoch
    repeated 300 times.
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
            same_photo_local_views=int(same_photo_local_views),
            seed=int(seed),
        )
    elif dataset_format == "pickle_batches":
        if int(same_photo_local_views) > 0:
            raise ValueError(
                "same_photo_local_views needs the source-photograph provenance that only the "
                "image_folder layout carries (filenames like IMG_0502_bbox137.png); the "
                "pickle_batches format has none."
            )
        dataset = PickleBatchSeedDataset(root_dir=data_dir, image_size=image_size, transform=transform)
    else:
        raise ValueError(f"Unsupported pretraining dataset format: {dataset_format}")

    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        loader_kwargs["prefetch_factor"] = max(int(prefetch_factor), 1)

    sampler = None
    if int(world_size) > 1:
        seed = int(generator.initial_seed() % (2**31)) if generator is not None else 0
        sampler = DistributedSampler(
            dataset,
            num_replicas=int(world_size),
            rank=int(rank),
            shuffle=True,
            seed=seed,
            # Equal batch counts on every rank. An uneven tail is not a rounding
            # error under DDP: the rank with the extra batch enters an all-reduce
            # its peers have already left, and the job hangs at the collective
            # timeout rather than failing.
            drop_last=True,
        )
        log.info(
            "Distributed sampler | %s images sharded over %s ranks -> %s per rank, "
            "%s batches of %s.",
            len(dataset), world_size, len(sampler), len(sampler) // max(batch_size, 1), batch_size,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
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
