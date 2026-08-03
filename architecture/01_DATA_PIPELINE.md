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

Implemented in `src/trainers/moe_finetune.py`.

### What the dataset actually is, before any splitter runs

Measured from the real tree under `Cropped_Samples`:

```text
total crops                    9,357
distinct source photographs       81
mean crops per source          115.5      (range 37 - 297)
sources per sub-variety          1 : 5 classes
                                 2 : 4 classes
                                 3 : 8 classes
                                 4 : 6 classes
                                 5 : 4 classes
```

Filenames carry the provenance directly — `IMG_0502_bbox137.png` is bounding box
137 cut from photograph `IMG_0502`. **115 crops per source photograph** is what
decides how the split has to work: crops sharing a source are near-duplicates of
the same physical scene, with the same lighting, background, sensor noise, and
frequently the same individual seed at overlapping bounding boxes.

The crops are also **small**: median 52 x 51 px, and 100 % of them have both
sides under 256, so every image is upsampled roughly 5x before the backbone sees
it. Only 3.4 % are square (aspect ratios span 0.17 - 3.48).

### Stratification key

```python
def stratification_labels(dataset) -> np.ndarray:
    return np.array([sub for _, _, sub in dataset.samples])
```

**Correction to a claim this document used to make.** It previously stated that
a composite `seed * 1000 + sub` key was needed because "stratifying on
sub-variety alone would not guarantee seed-type balance". That is false.
Sub-variety labels are **global** (0..26) and each has exactly one parent, so
`seed = parent(sub)` is a deterministic function of `sub`, and the map
`sub -> seed*1000 + sub` is a **bijection** — it induces exactly the same strata.
The code was correct; the reason was not. It now uses the simpler key that
produces the identical partition.

### Group-aware splitting (`split_protocol`, default `grouped`)

`HierarchicalSeedDataset.source_groups()` returns an integer group id per sample,
keyed on `<sub_variety>/<source photograph>` (scoped by directory because two
sub-varieties may reuse a filename). `split_dataset` then:

* `grouped` — `GroupShuffleSplit` for the test carve-out and
  `StratifiedGroupKFold` for the folds, so no source photograph appears on both
  sides of any boundary.
* `stratified` — the submitted crop-level splitting, retained deliberately.

Under crop-level splitting the reported accuracy is substantially a memorisation
score: near-duplicate views of the same seeds sit in both train and test. The
literature has measured this exact error repeatedly — 0.07-0.43 MCC in OCT
classification, 1.6-2.0 dB PSNR in intrinsic decomposition, and a 93.7 %
lithology result its own authors excluded as an artifact.

### The limit no protocol can remove

**Five of the 27 sub-varieties** — `Baryard`, `Browntop`, `FingerMillet`,
`PearlMillet`, `ProsaMillet` — have crops from exactly one photograph. No grouped
split can place any of their crops on both sides, so grouped stratification
degrades to "this class is entirely in one partition" for those five. On the real
tree a 30 % grouped test split leaves three sub-varieties out of training
altogether.

The trainer logs this at startup and records it in `summary.json`. For those
classes, **no protocol available on this dataset measures across-photograph
generalisation**, and the paper has to say so — it is a dataset limitation, not
a modelling one.

### The leak as a measurement

`scripts/run_ablations.py` runs `leakage_ungrouped`: the full model under
`split_protocol=stratified` and nothing else changed. The delta against
`full_model` quantifies what the leak was worth. That is a methods result worth
reporting, and it is the form careful applied-DL papers report it in.

Both protocols emit a leakage report — shared source groups, the fraction of the
test set they cover, and any sub-variety missing from either side — so a
stratified run carries its own indictment rather than leaving it to be inferred.

### Determinism

Everything is driven by `cfg.seed`, so every variant in a suite sees the
**byte-identical partition** — which is also what makes McNemar's paired test
available in `generate_plots.py`. Beyond the split, the trainer seeds the
`DataLoader` generator and every worker (`make_worker_init_fn`) and logs the cuDNN
flags, because a reproducible split with an unreproducible augmentation stream is
enough to move an ablation gap of the size this table reports.

### Persisted manifest

`save_split_manifest()` writes `split_manifest.npz` into the run's `save_path`:

```text
test_indices
seed_type_to_idx, subvariety_to_idx            # name -> index mappings
subvariety_to_seed_type                        # the KL mapping's source list
source_groups                                  # the provenance key, per sample
split_protocol                                 # "grouped" | "stratified"
fold_{n}_train_indices, fold_{n}_val_indices   # per fold
```

`source_groups` is persisted so a reviewer can verify the grouping rather than
trust the protocol name.

### Loaders

`make_loader(indices, shuffle, drop_last=False, train=False)` wraps
`Subset(dataset, indices)` in a `DataLoader` with `batch_size=data.batch_size`,
`num_workers=data.num_workers`, `pin_memory` gated on CUDA, and the seeded
generator plus `worker_init_fn`. Training loaders shuffle and honor
`data.drop_last`; validation and test loaders never shuffle or drop, and read a
**separate dataset instance carrying the deterministic transform** — otherwise
validation numbers would be augmentation noise rather than measurements.

`experiment.training.balanced_sampler` swaps shuffling for an inverse-frequency
`WeightedRandomSampler` over sub-varieties. The hierarchy is 13 rice + 8 millet +
3 + 3, so seed-type accuracy is structurally rice-dominated; macro-F1 is reported
either way, but this is the option that trains for it.

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

### Stage 2 — supervised transforms (`transforms.py`)

```python
get_supervised_transforms(image_size, train=True, normalize_mean=..., normalize_std=...,
                          horizontal_flip_prob=0.5,
                          random_resized_crop_scale=(0.8, 1.0),
                          vertical_flip_prob=0.0, rotation_degrees=0.0)
```

Training: `RandomResizedCrop((H, W), scale=(0.8, 1.0))` -> `RandomHorizontalFlip(0.5)`
-> `ToTensor` -> `Normalize`. Evaluation: `Resize((H, W))` -> `ToTensor` ->
`Normalize`, always deterministic.

**The resize takes an explicit `(H, W)` tuple, and that is load-bearing.**
`T.Resize(int)` resizes the shorter side and preserves aspect ratio; only 3.4 %
of these crops are square, so the integer form would emit variable-width tensors
and `default_collate` would raise on the first mixed batch. The tuple squashes
every crop to a square instead — a real aspect-ratio distortion, chosen
deliberately over a latent crash, and stated at the call site.

**Why the defaults changed.** The submitted default was `horizontal_flip_prob:
0.0` and nothing else, so stage-2 training saw each image exactly once per epoch,
deterministically. The stated rationale — that the representation is already
invariant from stage 1 — is a good argument against *heavy* augmentation but not
against any: SSL invariance of the frozen **encoder** says nothing about the
sample efficiency of the **head**, which is ~9 M freshly-initialised parameters
fit from ~7.5 k training images.

`wo_stage2_augmentation` reproduces the submitted pipeline as an ablation,
because "does stage-2 augmentation help when the encoder is frozen?" is a
legitimate question whose answer belongs in the paper rather than in a config
default.

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
