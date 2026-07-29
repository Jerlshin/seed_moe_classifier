# 01 — Data Pipeline

Covers `src/datasets/dataset.py`, `src/datasets/transforms.py`,
`conf/data/hierarchical_seeds.yaml`, and the split logic inside
`src/trainers/moe_finetune.py`.

## 1. On-disk layout and label hierarchy

```text
$SEED_DATA_ROOT/
  <seed_type>/
    <sub_variety>/
      *.png | *.jpg | *.jpeg | *.bmp | *.tif | *.tiff | *.webp
```

Paper Section 3: **4 seed types, 27 sub-varieties** (13 rice + 8 millet + 3 +
3), ~9,357 images. `data.num_seed_types` / `data.num_sub_varieties` in
`conf/data/hierarchical_seeds.yaml:25-26` declare these counts, and the
trainer cross-checks them against what it discovers on disk
(`moe_finetune.py:753-758`) — a mismatch raises rather than silently
corrupting every label index.

### `HierarchicalSeedDataset` (`src/datasets/dataset.py:86-223`)

```python
HierarchicalSeedDataset(root_dir, transform=None, save_csv_path=None, include_path=False)
```

`_load_dataset` (`dataset.py:129-144`) walks `root_dir` with **sorted**
`iterdir()` at both levels and assigns indices in that order:

```python
seed_type_dirs = sorted(path for path in self.root_dir.iterdir() if path.is_dir())
self.seed_type_to_idx = {path.name: index for index, path in enumerate(seed_type_dirs)}
...
for subvariety_dir in sorted(...):
    if subvariety_name not in self.subvariety_to_idx:
        self.subvariety_to_idx[subvariety_name] = len(self.subvariety_to_idx)  # GLOBAL counter
```

Two consequences that are load-bearing:

* **Sub-variety labels are global (0..26 across all seed types)**, not
  per-seed-type (0..12 within rice, 0..7 within millet, etc.). The ArcFace
  head is a single 27-way classifier, not four independent heads.
* **Sorted order is load-bearing.** Adding, removing, or renaming a directory
  shifts every downstream label index and invalidates existing checkpoints
  (the class-index-to-name mapping stored in the checkpoint would no longer
  match).

`__getitem__` (`dataset.py:175-188`) returns
`(image, seed_label_tensor, sub_label_tensor)`, or a 4-tuple with the source
path appended when `include_path=True`.

**Helper methods used throughout the rest of the codebase:**

| Method | Returns | Consumer |
| --- | --- | --- |
| `get_class_mappings()` | `(seed_type_to_idx, subvariety_to_idx)` name→index dicts | checkpoint payload |
| `get_idx_to_label()` | inverse index→name dicts | reporting |
| `get_ordered_class_names()` | `(seed_names, sub_names)`, ordered by label index | figure axes, metric tables |
| `get_subvariety_to_seed_type()` | `list[int]`, parent seed-type index per sub-variety index | KL mapping matrix `M` (Eq. 10), alignment-rate metric |
| `class_distribution()` | sample counts per class (paper Fig. 1) | reporting |

`get_subvariety_to_seed_type()` (`dataset.py:206-213`) is derived from the
directory tree at runtime — **never hardcoded** — which is what lets
`src/losses/hierarchical.py`'s KL aggregation matrix and
`src/utils/metrics.py`'s alignment rate stay correct if the dataset changes
without a code change.

Passing `save_csv_path` writes a manifest (`image_path, seed_type,
seed_type_label, subvariety, subvariety_label`) via `_save_to_csv`
(`dataset.py:146-170`); the finetune trainer wires this to
`data.save_csv_path` = `${SEED_OUTPUT_DIR}/metadata/seed_dataset.csv`.

### Other dataset classes

* **`PretrainImageFolderDataset`** (`dataset.py:15-23`) — a `torchvision`
  `ImageFolder` subclass whose `__getitem__` also returns the multi-crop list
  and file path, for DINO pretraining. Labels here are irrelevant (DINO is
  unsupervised); only the crops matter.
* **`PickleBatchSeedDataset`** (`dataset.py:26-83`) — notebook-compatibility
  path for flattened-RGB pickle batch files (`{"data": ..., "labels": ...}`).
  Selected with `data.dataset_format=pickle_batches`. `format_image` reshapes
  a flat `[3 * size * size]` array back into `[size, size, 3]` by channel
  slice.

### Loader factories

```python
get_pretrain_dataloader(data_dir, transform, batch_size, num_workers=4,
                         dataset_format="image_folder", image_size=256,
                         pin_memory=True, drop_last=True)   # dataset.py:226-253
get_finetune_dataset(data_dir, transform, save_csv_path=None,
                     include_path=False) -> HierarchicalSeedDataset  # dataset.py:256-267
```

`get_pretrain_dataloader` always shuffles and (by default) drops the last
partial batch — important for DINO, whose cross-view loss assumes a
consistent batch size across the views it chunks (`CustomDINOLoss.forward`,
see [`02_BACKBONE_AND_SSL.md`](02_BACKBONE_AND_SSL.md)).

## 2. Train / validation / test splitting (stage 2)

Implemented in `src/trainers/moe_finetune.py:98-167`.

### Stratification key

```python
def stratification_labels(dataset) -> np.ndarray:
    return np.array([seed * 1000 + sub for _, seed, sub in dataset.samples])
```

A composite `seed_label * 1000 + sub_label` key, so `StratifiedKFold` /
`train_test_split` keep **both** hierarchy levels balanced simultaneously —
stratifying on sub-variety alone would not guarantee seed-type balance and
vice versa.

### `split_dataset(dataset, test_size, num_folds, seed)`

1. Carve out a held-out **test set** first via `train_test_split(...,
   stratify=labels, random_state=seed)`, using `experiment.training.test_size`
   (default `0.2`).
2. Over the remainder, build `num_folds` train/validation splits:
   * `num_folds > 1` → `StratifiedKFold(n_splits=num_folds, shuffle=True,
     random_state=seed)`.
   * `num_folds == 1` (the default) → a single stratified
     `train_test_split(test_size=max(test_size, 0.2), ...)`.

**Everything is driven entirely by `cfg.seed`** (default `42`, set in
`conf/config.yaml:25`), so every variant in an ablation or baseline suite sees
the **byte-identical partition**. This is what makes cross-variant comparison
valid — a different split per variant would confound the architecture change
under test with a difference in what data it happened to see.

### Persisted manifest

`save_split_manifest()` (`moe_finetune.py:147-167`) writes
`split_manifest.npz` into the run's `save_path` containing:

```text
test_indices
seed_type_to_idx, subvariety_to_idx      # name -> index mappings
subvariety_to_seed_type                   # the KL mapping's source list
fold_{n}_train_indices, fold_{n}_val_indices   # per fold
```

so an evaluation can be reproduced later without re-deriving the split.

### Loaders

`make_loader(indices, shuffle, drop_last=False)` (`moe_finetune.py:785-793`)
wraps `Subset(dataset, indices)` in a `DataLoader` with
`batch_size=data.batch_size` (16, Table 1), `num_workers=data.num_workers`,
and `pin_memory` gated on CUDA. Training loaders shuffle and honor
`data.drop_last`; validation/test loaders never shuffle or drop.

## 3. Augmentation

### Stage 1 — multi-crop DINO pipeline (`src/datasets/transforms.py:72-192`)

`DataAugmentationDINO.__call__(image)` returns
`(original_tensor, [global_1, global_2, *locals])` — `2 +
local_crops_number` views total, exposed as `.num_crops`.

| View | Count | Crop scale | Pipeline (in order) |
| --- | --- | --- | --- |
| Global 1 | 1 | `(0.4, 1.0)` | `RandomResizedCrop(image_size)` → flip → color jitter → grayscale → **Gaussian blur p=1.0** → normalize |
| Global 2 | 1 | `(0.4, 1.0)` | same, but **blur p=0.1** → **solarize p=0.2** → normalize |
| Local | 4 | `(0.05, 0.4)` | `RandomResizedCrop(local_crop_size)` → flip → color jitter → grayscale → **blur p=0.5** → normalize → (optional) resize back to `image_size` |

Shared `flip_and_color_jitter` block: `RandomHorizontalFlip(p=0.5)` →
`RandomApply([ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2,
hue=0.1)], p=0.8)` → `RandomGrayscale(p=0.2)`. All values are configured in
`conf/data/hierarchical_seeds.yaml:29-62` and match the paper's stated
protocol (`PAPER_AUDIT.md` §4 records that an earlier notebook-era config used
`0.01` jitter/grayscale probabilities, which effectively disabled color
augmentation — the current defaults follow the paper instead).

**The asymmetry between the two global crops is deliberate** — it is what
gives the teacher's two views (the only views it sees) genuinely different
appearances, which is the signal the cross-view DINO loss consumes. Without
it, teacher-view-1-vs-teacher-view-2 pairs would carry no learning signal.

`GaussianBlur` (`transforms.py:38-53`) samples radius uniformly in `[0.1,
2.0]` when applied. `Solarization` (`transforms.py:56-69`) inverts pixels
above a threshold (default 128) with probability 0.2, applied only to global
crop 2.

**`resize_local_to_global`** (`transforms.py:174-177`): local crops are
cropped at `local_crop_size` (101 px, `conf/data/hierarchical_seeds.yaml:16`)
and then resized back up to `image_size` (256 px) because SwinV2's shifted
windows require a fixed input resolution. The local/global distinction is
carried by the *crop scale*, not the final tensor size — both views end up
`256×256` before reaching the backbone.

`get_dino_transforms(image_size, local_crop_size, augmentation_cfg)`
(`transforms.py:195-200`) builds the class from the `data.augmentation` config
node, popping `local_crop_size` since that key lives one level up on the
`data` node, not inside `augmentation`.

### Stage 2 — supervised transforms (`transforms.py:203-227`)

```python
get_supervised_transforms(image_size, train=True, normalize_mean=..., normalize_std=...,
                           horizontal_flip_prob=0.0)
```

A deterministic `Resize(image_size, BICUBIC)` → optional
`RandomHorizontalFlip` (default probability **0.0** —
`experiment.training.horizontal_flip_prob`) → `ToTensor` → `Normalize`.

This is intentionally minimal: by stage 2 the representation is already
invariant from DINO pretraining, and the fine-grained visual cues that
separate 27 sub-varieties are exactly the ones heavy augmentation would
destroy.

Both pipelines normalize with ImageNet statistics —
`mean=(0.485, 0.456, 0.406)`, `std=(0.229, 0.224, 0.225)` — for compatibility
with `timm`-pretrained SwinV2 weights.

## 4. Configuration reference (`conf/data/hierarchical_seeds.yaml`)

| Key | Value | Note |
| --- | --- | --- |
| `root_path` | `${oc.env:SEED_DATA_ROOT,data/Hierarchical_SeedData/Cropped_Samples}` | |
| `dataset_format` | `"image_folder"` | or `"pickle_batches"` |
| `image_size` | 256 | must equal the SwinV2 window resolution in the backbone name |
| `local_crop_size` | 101 | |
| `batch_size` | 16 | Table 1 |
| `num_workers` | 2 | |
| `num_seed_types` / `num_sub_varieties` | 4 / 27 | Section 3 |
| `augmentation.global_crops_scale` | `[0.4, 1.0]` | |
| `augmentation.local_crops_scale` | `[0.05, 0.4]` | |
| `augmentation.local_crops_number` | 4 | |
| `augmentation.color_jitter_{brightness,contrast,saturation,hue}` | 0.4 / 0.4 / 0.2 / 0.1 | |
| `augmentation.color_jitter_prob` / `grayscale_prob` | 0.8 / 0.2 | standard DINOv2 protocol |
| `augmentation.global_blur_prob_1` / `_2` | 1.0 / 0.1 | |
| `augmentation.solarization_prob` | 0.2 | global crop 2 only |
| `augmentation.local_blur_prob` | 0.5 | |
