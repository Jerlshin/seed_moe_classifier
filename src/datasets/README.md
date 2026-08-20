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

## The view policy, and how to measure any policy

`conf/data/hierarchical_seeds.yaml` — the single data group. Its header carries
the measurement behind every value; read it before changing one.

The problem it solves is that `scale` is a fraction of the **source** area, and
the source here is one seed at a median 52 × 51 px — not a scene. Measured over
all 9,357 crops through torchvision's own `get_params`:

| view recipe | native px p5/p50/p95 | upsample to 256 | real content |
| --- | --- | --- | --- |
| local `(0.05, 0.40)` at 101 px | 132 / **598** / 2,585 | 10.5× | 0.91 % |
| local `(0.30, 0.70)` at 160 px, ratio `(0.5, 2.0)` | 483 / **1,419** / 4,730 | 6.8× | 2.17 % |
| global `(0.40, 1.00)`, ratio `(0.75, 1.33)` | 648 / 1,845 / 5,680 | 6.0× | 2.82 % |
| global `(0.70, 1.00)`, ratio `(0.5, 2.0)` | 928 / 2,484 / 7,564 | 5.1× | 3.79 % |

**80 % of Eq. 1's cross-view terms are anchored on a local view**, so the first
row is what the objective was mostly being asked to learn from.

Three keys carry the policy, and each is separately switchable so an arm stays
single-factor:

`crop_ratio`
: The aspect range `RandomResizedCrop` samples. After ten failed (area, aspect)
  draws `get_params` returns a **deterministic centre crop**, and only 3.4 % of
  these crops are square — so raising the scale floor trades randomness for
  content unless the ratio widens with it. Measured global fallback rates: 3.5 %
  at `(0.40, 1.00)`/`(0.75, 1.33)`, **22.0 %** at `(0.70, 1.00)`/`(0.75, 1.33)`,
  10.2 % at `(0.70, 1.00)`/`(0.5, 2.0)`.

`min_native_pixels`
: A floor on the source pixels behind a view, raising the lower scale bound *per
  image* to `max(scale_lo, floor / area)`. It never lowers it, so it can only
  make views less destructive, and `0` is a plain `RandomResizedCrop` exactly.
  At 900 it lifts the local p5 by **+72 %** and the median by 1.1 % — a tail
  intervention, which is the whole claim.

`rotation90_prob` / `vertical_flip_prob`
: The dihedral group of order 8, via `PIL.Image.transpose` — a pixel
  permutation, **no interpolation and no black corners**, unlike
  `T.RandomRotation`. Safe *here* because seeds on a tray have no canonical
  orientation. This is what buys back the diversity the narrower crop ranges
  give up.

Colour is treated as signal rather than only as nuisance: brightness and contrast
(illumination) stay at ±0.4, saturation and hue (pigmentation) drop to ±0.1 and
±0.02, and grayscale and solarization go to zero. Mean RGB alone scores 0.3169 on
the 27-way task. Note the naive version of this argument is already refuted — the
cue *survived* the reference jitter — so it is a hypothesis with a control
(`wo_colour_policy`), not a repair.

To measure any policy without training anything:

```bash
python scripts/report_view_geometry.py --policy canonical reference
python scripts/report_view_geometry.py data.augmentation.min_native_pixels=900
```

`view_geometry_report(transform, sizes)` is the same measurement as a function;
the trainer runs it at startup and writes `csv/view_geometry.csv`.

## Sharding across GPUs

`get_pretrain_dataloader(..., world_size=N, rank=r)` attaches a
`DistributedSampler`. It splits **images**, never the view axis: the DINO loss
pairs a student view against the teacher's output for *that same image*, so all
`2 + local_crops_number` views of a sample have to be resident on one device.
Sharding views instead would turn the cross-view pairing into a collective, and
the loss curve of a run that got that wrong looks entirely normal.

`drop_last=True` on the sampler is not a rounding convenience: an uneven tail
means one rank enters an all-reduce its peers have already left, and the job
hangs at the collective timeout rather than failing.

The trainer must call `sampler.set_epoch(epoch)` every epoch. The permutation is
a function of `seed + epoch`, so skipping it gives every epoch the identical
order — no error, no change in the loss magnitude, and 100 epochs of one epoch.

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

## Corpus provenance

`corpus_fingerprint(root, paths)` returns a SHA-256 over the **sorted list of
dataset-relative POSIX paths**, plus the sample, class and source-group counts and
a per-class histogram. Both dataset classes expose it as
`.corpus_fingerprint()`, over their own sample list rather than over the
directory — the two differ the moment a sample is filtered out, and the honest
answer is what will actually be read.

It exists because the stage-1 → stage-2 handoff is a bare `state_dict` with no
record of what it was trained on, and that was not hypothetical: **the shipped
encoder was self-distilled on 8,173 crops while everything downstream uses
9,357**, recoverable only by cross-reading two log lines against `metrics.json`.
Stage 1 writes the fingerprint into `summary.json`; `pretrain_eval` reads it back
and prints `describe_fingerprint_mismatch(...)` when the corpora differ.

The digest is over *relative* paths, so it is stable across machines and mount
points and changes the instant a file is added, removed or renamed.

`raw_photograph_coverage(raw_root, cropped_root)` reports which source
photographs exist under `RAW_Samples` and were never cropped — 18 of 99 on the
shipped corpus, i.e. **+22 % scenes** at zero acquisition cost. Scene count, not
crop count, is what a photograph-disjoint protocol can resolve: within one
photograph 89–98 % of crops have a neighbour above cosine 0.95 at 32×32 grey. It
only reports; cropping happens outside this repository, and republishing the
corpus moves every published accuracy, so it must be a deliberate re-baseline.

## Provenance-derived positives

`same_photo_local_views: n` replaces the trailing `n` local views with local crops
of *other crops from the same source photograph*. The provenance key comes from
the filename (`IMG_0502_bbox137.png`), so no labels are involved.

The view **count** is unchanged, so the loss's cross-view pairing, `view_ids` and
every shape downstream are untouched; only what views 2..V depict changes. What
changes is the invariance being taught: DINO's positives are augmented views of
one crop, so it learns invariance to crop/blur/colour, while the downstream task
needs invariance to *which individual seed* — and two crops of one photograph are
by construction two individuals of the same variety.

**The risk is self-detecting, and must be checked before believing any gain.**
Same-photograph crops share lighting, background and sensor noise, so the
objective can be satisfied by learning **photograph identity** — exactly what the
photograph-disjoint protocol punishes. The evaluation's
`nuisance_photo_above_chance` is the gate (the shipped DINO encoder sits at
+3.5 pp against ImageNet's +10.0 pp); an arm that pushes it back up learned the
confound rather than the variety.

The partner draw is seeded on `(seed, sample index)`, so the pairing is a
deterministic function of the sample rather than of which worker built it. A
photograph with only one crop yields no partner and the anchor is augmented
instead, which keeps the view count constant.
