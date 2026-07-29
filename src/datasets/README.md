# `src/datasets/` — data and augmentation

| File | Contents |
| --- | --- |
| `dataset.py` | `HierarchicalSeedDataset`, `PretrainImageFolderDataset`, `PickleBatchSeedDataset`, loader factories |
| `transforms.py` | `DataAugmentationDINO` multi-crop pipeline, supervised transforms |

## Label hierarchy (paper Section 3)

```text
$SEED_DATA_ROOT/
  <seed_type>/
    <sub_variety>/
      *.png | *.jpg | *.jpeg | *.bmp | *.tif | *.tiff | *.webp
```

`HierarchicalSeedDataset` walks the tree and assigns indices from **sorted**
directory names. `__getitem__` returns `(image, seed_label, sub_label)`, or
additionally the path when `include_path=True`.

Three properties that matter downstream:

* **Sub-variety labels are global**, 0..26 across all seed types — not 0..12
  within rice. The ArcFace head has 27 classes, not 4 heads of 13/8/3/3.
* **`get_subvariety_to_seed_type()`** returns the parent seed type per
  sub-variety index. This builds the KL aggregation matrix `M` (Eq. 10) and
  drives the alignment-rate metric, and is derived from the tree rather than
  hardcoded.
* **Sorted order is load-bearing.** Adding or renaming a folder shifts every
  label index and silently invalidates existing checkpoints. The trainer
  cross-checks the discovered counts against `data.num_seed_types` /
  `data.num_sub_varieties` and refuses to start on a mismatch.

Other helpers: `get_ordered_class_names()` (names ordered by label index, for
figure axes and metric tables), `get_idx_to_label()`, and
`class_distribution()` (sample counts per class, paper Fig. 1).

Passing `save_csv_path` writes a manifest of every sample with both labels.

## Multi-crop DINO augmentation (paper Sections 4, 6.1)

`DataAugmentationDINO` returns `(original_tensor, [global_1, global_2, *locals])`.

| View | Count | Crop scale | Pipeline |
| --- | --- | --- | --- |
| Global 1 | 1 | (0.4, 1.0) | resized crop → flip → jitter → **blur p=1.0** → normalize |
| Global 2 | 1 | (0.4, 1.0) | resized crop → flip → jitter → **blur p=0.1** → **solarize p=0.2** → normalize |
| Local | 4 | (0.05, 0.4) | resized crop → flip → jitter → **blur p=0.5** → normalize → resize |

Colour jitter magnitudes are the paper's: brightness ±0.4, contrast ±0.4,
saturation ±0.2, hue ±0.1.

The **asymmetry between the two global crops is the point** — it is what makes
the teacher's two views genuinely different, and the cross-view DINO loss has no
signal without it.

`resize_local_to_global` scales local crops back up to `image_size` after
cropping, because SwinV2's shifted windows require a fixed input resolution. The
local/global distinction is carried by the crop *scale*, not the tensor size.

The teacher sees only the 2 global crops; the student sees all 6. That
asymmetry is enforced in the trainer, not here.

## Other datasets

`PretrainImageFolderDataset` — a `torchvision` `ImageFolder` that also returns
the multi-crop list and the file path, for DINO pretraining.

`PickleBatchSeedDataset` — notebook-compatibility path for flattened-RGB pickle
batch files. Select with `data.dataset_format=pickle_batches`.

## Supervised transforms

`get_supervised_transforms` is a deterministic resize + normalize, with optional
horizontal flip. Stage 2 keeps augmentation minimal on purpose: the
representation is already invariant from DINO pretraining, and the fine-grained
cues that separate sub-varieties are exactly the ones heavy augmentation
destroys.
