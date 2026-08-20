# seed-moe-classifier

Reference implementation of **"Hierarchical Deep Learning for Fine-Grained Seed
Classification: A Self-Supervised and Mixture-of-Experts Approach"**.

Two stages: DINO-style self-supervised pretraining of a Swin Transformer V2
encoder on the seed corpus, then a hierarchical head that classifies 4 seed types
and 27 sub-varieties with a Mixture-of-Experts, cross-attention refinement, and
ArcFace metric learning.

```text
Stage 1    ImageNet-1k → SwinV2-Tiny → DINO self-distillation (trunk unfrozen)
Stage 1.5  score the representation itself — the DINO loss cannot rank encoders
Stage 2    the resulting encoder (frozen) → hierarchical MoE head
```

```
image ──SwinV2──▶ pooled ──proj──▶ z ∈ ℝ³⁸⁴                            (Eq. 4)
                                     │
                        ┌────────────┴────────────┐
                        ▼                         ▼
              SeedTypeClassifier            MixtureOfExperts
                   s ∈ ℝ⁴  (Eq. 5)      h = Σ_{i∈Top-2} Gᵢ Eᵢ(z)  (Eq. 8)
                        │                         │
                 p_s = softmax(s)  (Eq. 6)        │
                        │                         │
                     P(p_s) ────────(+)───────────┘
                                     │
                              h' = h + P(p_s)                          (Eq. 9)
                                     │
                  CrossAttention(Q=h', K=V=h)                         (Eq. 11)
                     h'' = LayerNorm(a + Q)                           (Eq. 12)
                                     │
                        SubVarietyEmbedding ──▶ ArcFace(27)           (Eq. 13)
```

There is **one** pipeline. One data group, one trunk, one stage-1 experiment,
one stage-2 experiment, one split protocol. Everything else in `conf/` is either
a *control* (a named comparison the results table needs) or an *ablation* (one
override off the primary), and each says which it is in its own header.

> **Relationship to the submitted manuscript.** This tree implements the
> peer-review revision. The headline departures: the router activates **2 of 6**
> experts rather than 4; **SwinV2 is the only encoder** (the comparative ViT-S/14
> path was removed); stage 1 runs **SwinV2-Tiny from ImageNet-1k** rather than
> SwinV2-Base from random initialisation for 300 epochs; the multi-crop scales
> are sized for a 61 × 61 px source rather than for ImageNet; and the ArcFace
> scale is the analytic AdaCos value 4.61 rather than 30. Almost all of it is
> reversible by override — see [Reproducing the submitted
> configuration](#reproducing-the-submitted-configuration).

---

## Table of contents

1. [Install](#1-install)
2. [Environment variables](#2-environment-variables)
3. [Data layout](#3-data-layout)
3a. [Stage 0 — building the corpus from the photographs](#3a-stage-0--building-the-corpus-from-the-photographs)
4. [Validate the dataset before spending a GPU-hour](#4-validate-the-dataset-before-spending-a-gpu-hour)
5. [The canonical workflow, end to end](#5-the-canonical-workflow-end-to-end)
6. [Stage 0 — the bar stage 1 must clear](#stage-0--the-bar-stage-1-must-clear)
7. [Stage 1 — DINO self-distillation](#stage-1--dino-self-distillation)
8. [Stage 1.5 — evaluate the representation](#stage-15--evaluate-the-representation)
9. [Stage 2 — hierarchical MoE finetuning](#stage-2--hierarchical-moe-finetuning)
10. [Analysis: ablations, baselines, figures](#10-analysis-ablations-baselines-figures)
11. [Hydra overrides that matter](#11-hydra-overrides-that-matter)
12. [Multi-GPU, resuming, and rented boxes](#12-multi-gpu-resuming-and-rented-boxes)
13. [Outputs](#13-outputs)
14. [Testing and verification](#14-testing-and-verification)
15. [Repository layout](#15-repository-layout)

---

## 1. Install

```bash
python -m pip install -e ".[tracking,dev]"
```

`tracking` pulls in wandb / tensorboard / pynvml; `dev` pulls in pytest. Neither
is required to train — a missing tracking backend degrades to a warning.

Runs on Linux + CUDA, on a Kaggle T4×2, on Apple Silicon (MPS) and on CPU
without a per-platform config: precision, `torch.compile` and dataloader worker
counts are all resolved from the hardware at startup and the resolution is
written into `summary.json`.

## 2. Environment variables

Every path resolves through `oc.env`, so these are the real knobs on a rented
GPU box.

| Variable | Meaning | Default |
| --- | --- | --- |
| `SEED_DATA_ROOT` | Dataset root — the corpus every stage reads | `data/Hierarchical_SeedData/Refined_Samples` |
| `SEED_OUTPUT_DIR` | Root for run dirs, checkpoints, metadata | `outputs` |
| `SEED_PRETRAIN_BACKBONE` | The encoder every downstream run loads — the **only** handoff between the stages | `$SEED_OUTPUT_DIR/checkpoints/dino_pretrained_encoder.pth` |
| `SEED_RAW_DATA_ROOT` | `RAW_Samples` tree — stage 0's input, and the coverage audit's | `data/Hierarchical_SeedData/RAW_Samples` |
| `SEED_REFINED_DATA_ROOT` | Where stage 0 *writes* the corpus. Normally the same path as `SEED_DATA_ROOT` | `data/Hierarchical_SeedData/Refined_Samples` |
| `SEED_RUN_ID` | Shared Hydra run-directory suffix; the launcher pins it so every rank agrees | timestamp |

```bash
export SEED_RAW_DATA_ROOT=/workspace/data/Hierarchical_SeedData/RAW_Samples
export SEED_REFINED_DATA_ROOT=/workspace/data/Hierarchical_SeedData/Refined_Samples
export SEED_DATA_ROOT=$SEED_REFINED_DATA_ROOT
export SEED_OUTPUT_DIR=/workspace/outputs
```

`SEED_RUN_ID` matters under `torchrun`: Hydra evaluates `${now:...}` *per
process*, so two ranks launched in the same second can resolve different
directories. `scripts/launch.py` and `main.py --gpus` pin it before the ranks
exist, and the trainers additionally broadcast rank 0's resolved directory, so a
bare `torchrun` is correct too — this just makes it tidy as well.

## 3. Data layout

```text
$SEED_RAW_DATA_ROOT/            # 99 photographs -- the only irreplaceable artifact
  Rice/Chinnar/         IMG_0689.JPG ...
        │
        │   python main.py extract-seeds       (stage 0, see section 3a)
        ▼
$SEED_DATA_ROOT/                # 13,492 square one-seed crops from 96 of them
  Rice/
    Chinnar/            IMG_0689_bbox0007.png ...
    Chithrakar/         ...
  Millet/
    Baryard/            ...
  Amaranthus/ ...
  Mustard/ ...
```

The `_bbox<n>` suffix is not decoration: `source_image_id` parses it, and the
photograph-disjoint split protocol groups crops by the stem in front of it.

Labels come from **sorted** directory names, and sub-variety labels are
**global** (0..26 across all seed types, not per seed type). The
sub-variety → seed-type map that drives the KL hierarchy term is derived from
this tree at runtime, never hardcoded. Adding or renaming a folder shifts every
index and invalidates existing checkpoints; the trainer refuses to start when the
discovered counts disagree with `data.num_seed_types` / `num_sub_varieties`.

**Three properties of this corpus decide almost every design choice below, so
they are stated once here.**

* **It is 96 photographs, not 13,492 images.** `Refined_Samples` holds 13,492
  crops cut from 96 source photographs — a mean of 140.5 crops per source,
  encoded in the filenames. Crops from one photograph share the tray, the
  lighting and the sensor noise. Scene count, not crop count, is what a
  photograph-disjoint protocol can resolve.
* **The crops are small, and square.** Median **61 × 61 px**, 99.8 % under 256,
  so every image is upsampled ~4×; **100 %** are square, so `Resize((256, 256))`
  is a uniform rescale and the seed's true proportions reach the encoder. The
  seed itself occupies a median 50.5 % of its crop — the rest is the paper ring
  that makes the boundary visible.
* **Five sub-varieties have crops from exactly one photograph** (Baryard,
  Browntop, FingerMillet, PearlMillet, ProsaMillet). For those classes *no*
  protocol on this dataset measures across-photograph generalisation. Stage 0 did
  not change this and could not: each genuinely has one raw photograph, and the
  fix is a camera rather than a splitter.

> **The corpus was re-baselined.** Every number published before this was
> produced on `Cropped_Samples` — 9,357 hand-curated, 96.6 %-non-square crops
> from 81 photographs — and those numbers are **not** comparable with numbers
> from `Refined_Samples`. The corpus SHA-256 in every `summary.json` is what
> keeps the two distinguishable. Section 3a is the whole story; the legacy corpus
> is still on disk and still runnable as a control.

## 3a. Stage 0 — building the corpus from the photographs

```bash
python main.py extract-seeds      # RAW_Samples -> Refined_Samples + manifest + overlays
python main.py validate-seeds     # recall, duplicates, coverage, rejection census
python main.py benchmark-corpus   # is it actually better? (no training, ~25 min)
```

Run once, before a phase. It never writes under `SEED_RAW_DATA_ROOT`, never
resamples a crop, and refuses to write into a populated output root unless told
to (`segmentation.output.overwrite=true`). Full detail:
[`src/segmentation/README.md`](src/segmentation/README.md).

### What it produced

| | |
| --- | --- |
| photographs read | 99 |
| photographs used | **96** (3 excluded — see below) |
| detections adjudicated | 24,218 |
| crops written | **13,492** |
| detections rejected, with a stated reason | 10,726 |
| from touching seeds separated by watershed | 76 |
| with a neighbouring seed inpainted out | 870 |
| duplicate detections | **0** |
| wall clock, whole corpus, one CPU core | 136 s |

The three excluded photographs are not trays of seeds: `Poosa33/IMG_0667` is a
near-empty sheet, `Poosa33/IMG_0668` is a paper packet standing on a wooden
table, and `Kullakar/IMG_0713` is a labelled ziplock bag held up to the camera.
They are identified from their support fraction (0.58 / 0.63 / 0.65, against
0.76–1.00 for every genuine tray), not from a list, and they are named in
`extraction_summary.json` with the measurement that excluded them.

**Nothing is discarded silently.** Every connected component gets a row in
`manifest.csv`, an outline in its photograph's overlay, and — if it was not
written — one of the reasons in `REJECTION_REASONS`. The census:

| reason | count | what it is |
| --- | --- | --- |
| `too_small` | 9,660 | dust, chaff and paper fibre below 0.35 median seeds |
| `scene_rejected` | 3,319 | every detection in the three non-tray photographs |
| `out_of_frame` | 1,035 | seeds clipped by the photograph's own border |
| `irregular` | 214 | solidity far below this photograph's own median |
| `unresolved_cluster` | 117 | merged seeds the watershed declined to split |
| `implausible_shape` | 107 | slivers: too little area for their perimeter |
| `too_large` | 68 | paper edges, pen marks, piles |
| `blurred` / `touches_support_edge` / `low_contrast` | 72 | see the module docs |

### Evidence that seeds were not missed or duplicated

The legacy crops are **byte-identical sub-images** of the raw photographs, so
their exact bounding boxes are recoverable by template matching. That turns the
usual unanswerable question into a measurement, and `python main.py validate-seeds`
runs it:

| | |
| --- | --- |
| legacy crops with a recoverable box | 9,050 of 9,357 |
| recall of them by the refined detector | **99.71 %** (9,024) |
| legacy-only, i.e. possibly missed | **26**, individually listed and pictured |
| refined-only, i.e. newly found in the same photographs | **1,439** |
| duplicate pairs (one seed, two files) | **0** refined, **0** legacy |

The other 307 legacy crops belong to four photographs — `Chithrakar/IMG_0161`,
`IMG_0162`, `Kullakar/IMG_0711`, `IMG_0712` — whose raw file was replaced by a
differently oriented version after the crops were cut (mean absolute pixel
difference 15–22 at the recovered location, against exactly 0 everywhere else).
Their boxes describe nothing, so they are named and excluded from the
denominator rather than scored. Of the 26 legacy-only cases, roughly a fifth are
legacy crops of the **wooden table or a paper edge** rather than of a seed;
`outputs/segmentation/figures/legacy_only_seeds.png` shows all of them at native
resolution.

### Is the refined corpus better?

`python main.py benchmark-corpus` runs both corpora through the same frozen
encoders under the same protocols, with no training, and includes a size- and
source-matched control by default — because "more crops" is not "better crops".

Frozen ImageNet-1k SwinV2-Tiny, 27-way linear probe:

| protocol | legacy 9,357 / 81 photos | refined 13,492 / 96 | refined, matched to 9,347 / 81 |
| --- | --- | --- | --- |
| crop-level probe | 0.8007 | **0.8599** | 0.8449 |
| crop-level macro-F1 | 0.8004 | 0.8416 | 0.8424 |
| crop-level k-NN (no fitted parameters) | 0.7099 | 0.7629 | 0.7235 |
| photograph-disjoint, out-of-fold | 0.6194 | **0.6840** | 0.6371 |

The matched column isolates the crops themselves: **+4.42 pp** crop-level and
**+1.77 pp** photograph-disjoint from the redefinition alone, with the remainder
coming from the 15 extra photographs — the axis this dataset is genuinely short
of.

The two shortcut floors move the *other* way, which is the result worth having:

| encoder (crop-level) | legacy | refined | gap to `imagenet_init` |
| --- | --- | --- | --- |
| `handcrafted` (10 scalars) | 0.5075 | 0.4983 | +29.3 pp → **+36.2 pp** |
| `random_init` | 0.5903 | 0.5169 | +21.0 pp → **+34.3 pp** |

Ten trivial statistics score *lower* on the refined corpus, largely because
`log(aspect ratio)` — a real shape cue that used to sit in the file's dimensions —
is now constant. The cue did not disappear: it moved into the pixels,
undistorted, where an encoder has to look at the seed to use it.

### Running against the legacy corpus

It is still on disk and still a first-class control. Two overrides, and neither
is cosmetic:

```bash
SEED_DATA_ROOT=/path/to/Hierarchical_SeedData/Cropped_Samples \
python main.py pretrain data.expected_num_samples=9357 \
    data.augmentation.crop_ratio=[0.5,2.0]
```

`conf/stage1_arms/view_design.yaml` runs it as the `legacy_corpus` arm.

## 4. Validate the dataset before spending a GPU-hour

```bash
python main.py validate-data
```

Runs two reports, neither of which touches a checkpoint or a GPU:

```bash
python scripts/report_view_geometry.py     # what each DINO view is actually built from
python scripts/report_raw_photographs.py   # source photographs that exist and were never cropped
```

**`report_view_geometry.py` is the one to read first.** `RandomResizedCrop`'s
`scale` is a fraction of the **source area**, so a config does not tell you how
much of a seed a view contains — only the product of the scale range and the
source-size distribution does. This drives torchvision's own `get_params` over
the real file headers, so the numbers are what the dataloader will produce:

```bash
python scripts/report_view_geometry.py --policy canonical reference
python scripts/report_view_geometry.py --csv outputs/reports/view_geometry.csv
python scripts/report_view_geometry.py data.augmentation.min_native_pixels=900
```

Each `--policy` name matches an arm in `conf/stage1_arms/view_design.yaml`:
`reference` is DINO's ImageNet geometry (what `wo_view_redesign` trains),
`legacy_coverage` is the range that reproduces the legacy corpus's view
statistics on the refined one, and `wide_ratio` is the aspect range the legacy
corpus needed. Measured over all 13,492 refined crops, 40k draws at seed 7:

| view recipe | native px p5/50/95 | upsample | real content | centre-crop fallback |
| --- | --- | --- | --- | --- |
| global `(0.40,1.00)` — reference | 837 / 2,491 / 14,070 | 5.1× | 3.80 % | 0.1 % |
| local `(0.05,0.40)` — reference | 180 / **864** / 5,550 | 8.7× | 1.32 % | 0.0 % |
| global `(0.70,1.00)` — **canonical** | 1,178 / 3,009 / 16,641 | 4.7× | 4.59 % | 0.3 % |
| local `(0.30,0.70)` — **canonical** | 638 / **1,935** / 10,812 | 5.8× | 2.95 % | 0.0 % |

Under the reference ranges a local view is a median **29 × 29 px** fragment of
one seed inflated to 65,536 output pixels — and **8 of the 10 cross-view terms**
in Eq. 1 are anchored on such a view. The canonical policy takes that to 1,935
native pixels (2.24×) and cuts the local upsample from 8.7× to 5.8×, which also
narrows the local-vs-global resolution gap the student can use as a shortcut.

**Native pixels are not the whole story.** A view can carry plenty of sensor data
and little seed: the refined crops include a 12 % paper ring, so the seed occupies
a median 50.5 % of a crop against 79.9 % in the legacy tight boxes. The quantity
that transfers between the two corpora is the share of the seed's own bounding
box inside the view — measured on 9,024 matched seeds:

| corpus | family | scale | seed coverage p5 / median | whole seed (≥95 %) |
| --- | --- | --- | --- | --- |
| legacy | global | `(0.70,1.00)` | 0.66 / 0.875 | 24 % |
| refined | global | `(0.70,1.00)` | 0.88 / **1.000** | 78 % |
| legacy | local | `(0.30,0.70)` | 0.37 / 0.560 | 0.1 % |
| refined | local | `(0.30,0.70)` | 0.48 / **0.768** | 9 % |

At unchanged settings the re-baseline therefore makes views *less* destructive,
in the same direction the view redesign already moved: a teacher target now shows
the whole seed on median. The scale ranges were kept, because moving them would
confound the corpus change with a view-policy change — and
`legacy_view_coverage` is the arm that measures whether they should have moved.

The **fallback** column is why `crop_ratio` is a config key, and it is the one
view constant the re-baseline moved. `get_params` retries the (area, aspect) draw
ten times and then returns a *deterministic centre crop*, and whether that fires
depends on the **source's** aspect ratio:

| corpus | square | fallback @ `(0.70,1.00)` × `(0.75,1.33)` | @ `(0.50,2.00)` |
| --- | --- | --- | --- |
| `Cropped_Samples` (legacy) | 3.4 % | **22.0 %** | 10.2 % |
| `Refined_Samples` (canonical) | 100 % | **0.3 %** | 4.8 % |

On the legacy corpus the wide range fixed the fallback; on the refined one it
*causes* it, and it also applies up to a 2× **anisotropic** rescale to a seed
whose proportions the square crops exist to preserve. The canonical policy is
therefore torchvision's `(0.75, 1.33)`, and the legacy corpus needs
`data.augmentation.crop_ratio=[0.5,2.0]` back. The trainer measures the realised
rate at startup and warns above 15 % either way.

`report_raw_photographs.py` needs `SEED_RAW_DATA_ROOT` and reports which source
photographs exist and were never cropped — **3 of 99** now, and all three on
purpose (§3a). It reported 18 on the legacy corpus, of which 15 were ordinary
trays that had simply never been cropped. It never touches the dataset;
`main.py extract-seeds` is what acts, and the corpus fingerprint (§13) is what
keeps before and after distinguishable.

## 5. The canonical workflow, end to end

```bash
# ---- setup ---------------------------------------------------------------
python -m pip install -e ".[tracking,dev]"
export SEED_RAW_DATA_ROOT=/path/to/Hierarchical_SeedData/RAW_Samples
export SEED_REFINED_DATA_ROOT=/path/to/Hierarchical_SeedData/Refined_Samples
export SEED_DATA_ROOT=$SEED_REFINED_DATA_ROOT
export SEED_OUTPUT_DIR=/path/to/outputs

# ---- 0. check the machine ------------------------------------------------
python -m pytest tests/ -q            # 666 tests, no network, ~60 s
python scripts/dry_run.py             # synthetic end-to-end pipeline check, no dataset
python scripts/verify_runtime.py      # are the fast paths exact on THIS machine?

# ---- 0a. build the corpus, once, before a phase (~2.5 min) --------------
python main.py extract-seeds          # RAW_Samples -> Refined_Samples
python main.py validate-seeds         # recall vs the legacy boxes, duplicates, census
python main.py benchmark-corpus       # optional: price the re-baseline, no training
python main.py validate-data          # corpus, view geometry, uncropped photographs

# ---- 1. the bar stage 1 must clear (no training at all) -----------------
python main.py eval-frozen
python main.py screen-backbones       # optional: which initialisation transfers best?

# ---- 2. stage 1: DINO self-distillation ---------------------------------
python scripts/bench_pretrain_step.py --find-batch-size 16,24,32,48,64   # measure first
python main.py pretrain                                # or: --gpus 2

# ---- 3. stage 1.5: score the representation -----------------------------
python main.py eval-pretrain

# ---- 4. stage 2: the hierarchical MoE head ------------------------------
python main.py finetune
python main.py finetune-grouped       # the photograph-disjoint diagnostic

# ---- 5. analysis --------------------------------------------------------
python scripts/run_ablations.py --gpus 0,1
python scripts/run_baselines.py --gpus 0,1
python scripts/generate_plots.py
```

Every stage forwards anything after its name to Hydra verbatim, so the same
command line takes overrides:

```bash
python main.py finetune data.batch_size=8 experiment.training.epochs=50
python main.py pretrain model.loss.koleo_scope=all_views
```

The full stage list:

| Command | What it does | Trains? |
| --- | --- | --- |
| `python main.py validate-data` | corpus, view geometry, uncropped photographs | no |
| `python main.py eval-frozen` | the frozen-trunk reference stage 1 must beat | no |
| `python main.py screen-backbones` | frozen-feature screen across candidate trunks | no |
| `python main.py pretrain` | stage 1: DINO self-distillation | **yes** |
| `python main.py eval-pretrain` | stage 1.5: score the representation | no |
| `python main.py finetune` | stage 2: hierarchical MoE, crop-level split | **yes** |
| `python main.py finetune-grouped` | stage 2 under photograph-disjoint folds | **yes** |
| `python main.py ablation` | flat-classifier ablation | **yes** |
| `python main.py smoke` | 2-batch dry run of both stages | trivially |

---

## Stage 0 — the bar stage 1 must clear

```bash
python main.py eval-frozen        # experiment=eval_frozen_reference
```

**Run this before stage 1, and treat its number as the bar.** The stage-1
evaluation's own decomposition of an earlier 13.34-hour run was
`random 0.3804 → +0.2449 ImageNet-1k → +0.0031 DINO`: the *initialisation* was
worth 79× what the self-distillation bought. If a stage-1 run does not clear this
row by more than the fold SD, the honest claim is that the objective is not the
binding constraint on 81 scenes — and that is a publishable result, not a failure
to hide.

Measured on this trunk and this protocol (all 9,357 crops, 7,485 fit / 1,872
test, frozen, MPS, 12.75 min):

| encoder | 27-way probe | macro F1 | k-NN | 4-way | RankMe | within-class photo id |
| --- | --- | --- | --- | --- | --- | --- |
| `imagenet_init` (SwinV2-Tiny, IN-1k) | **0.8243** | 0.8234 | 0.7137 | 0.9979 | 324.4 | +11.4 pp |
| `random_init` | 0.5748 | 0.5683 | 0.4952 | 0.8873 | 27.4 | +11.6 pp |
| `handcrafted_floor` (10 scalars) | 0.4054 | 0.3367 | 0.4920 | 0.8787 | 1.4 | +8.7 pp |

Three readings that shape everything downstream:

1. **`layers.2` beats the output stage, with no training at all**: 0.8024 /
   0.8280 / **0.8638** / 0.7607 across `layers.0..3`, against 0.8243 pooled. That
   is +3.95 pp available from the *readout stage* alone, which is what
   `model.backbone.feature_stage=stage3_pooled_2x2` and the `stage3_readout` arm
   exist to chase.
2. **The crop-level headline sits +18.95 pp above the photograph-disjoint
   estimate** on the identical encoder (0.8243 vs 0.6347). See §"Stage 2" for
   what that means for the reported accuracy.
3. **Nuisance is not yet informative.** ImageNet (+11.4 pp) and an *untrained*
   trunk (+11.6 pp) are indistinguishable, so at this point the number reflects
   the images rather than the encoder. It becomes a discriminator only once
   in-domain training has run — which is exactly why it is the *gate* on the arms
   rather than a headline.

`screen-backbones` is the wider version of the same idea: every candidate trunk,
frozen and ImageNet-supervised, scored by the identical transform, folds, seed
and probe. It is the screen that chose SwinV2-Tiny:

| trunk | params | GFLOPs/view | pooled 27-way | `layers.2` 27-way |
| --- | --- | --- | --- | --- |
| `swinv2_tiny_window16_256` | 27.58 M | 13.32 | 0.6021 | **0.6243** |
| `swinv2_small_window16_256` | 48.96 M | 25.56 | 0.6053 | 0.6174 |

Tiny is −0.32 pp pooled and **+0.69 pp at the better readout stage**, for half
the FLOPs. Small is dominated on both axes. The screen also carries
`swinv2_base_window16_256.ms_in1k`, the control that separates *capacity* from
*the IN-22k corpus* — **do not adopt Base before that row has run**; capacity is
not the binding constraint here, scene diversity is.

## Stage 1 — DINO self-distillation

```bash
python main.py pretrain               # experiment=pretrain_dino
python main.py pretrain --gpus 2      # the same, as a 2-rank DDP job
```

`conf/experiment/pretrain_dino.yaml` carries the measurement behind every value.
The configured recipe:

| | Value | Why |
| --- | --- | --- |
| Backbone | `swinv2_tiny_window16_256` | 27.58 M params, 13.32 GFLOPs/view @256 — both measured. Chosen by the frozen screen above; set in `conf/model/backbone/swinv2.yaml`, which **both stages share** |
| Initialisation | ImageNet-1k (`ms_in1k`) | the trunk then **trains**; `build_dino` refuses `freeze=true` outright |
| Stochastic depth | 0.1, student only | the teacher copy is silenced — its outputs are the *targets*, so drop-path there is noise in the label |
| Views | 2 global @256 + 4 local @160 | scales `(0.70,1.00)` / `(0.30,0.70)`, aspect `(0.5,2.0)` — see §4 |
| Photometry | brightness/contrast 0.4, saturation 0.1, hue 0.02, no grayscale/solarize | illumination is nuisance and stays; pigmentation is class signal and is preserved |
| Flips / rotations | h-flip 0.5, v-flip 0.5, 90° 0.75 | the full dihedral group, losslessly (`PIL.transpose`, no interpolation) — seeds on a tray have no canonical orientation |
| Physical batch | 64, accumulation 1 | Sinkhorn/KoLeo are per-*micro*-batch, so accumulation is not a substitute |
| DINO head | 768 → 1024 → 1024 → 256 → 2048 | discarded after stage 1; nothing downstream depends on it |
| Prototypes | 2,048 | `K/B_teacher` is the quantity that matters; at batch 64 that is 16 prototypes per teacher view |
| KoLeo | per view, on the **backbone** feature | across views it is an anti-alignment term (below); the bottleneck is discarded at the end of stage 1 |
| Epochs | 50 configured, probed every 5, stopped on plateau | the **probe** picks the published encoder, not the loss |
| Learning rate | derived: `0.0005 × B_eff/256` = **1.25e-04** | Section 6.1's 0.0005 is DINO's rate at *its* reference batch of 256 |
| Warmup | 5 epochs linear, then cosine | one `SequentialLR`, resumable mid-warmup |
| Weight decay | 0.04 → 0.4 cosine, **matrices only** | biases, norms, `logit_scale` and `cpb_mlp` excluded |
| Teacher | momentum 0.996 → 1.0; τ 0.04 → 0.07 over 30 epochs | unchanged from the paper |
| Clip / freeze last layer | 3.0 / 1 epoch | unchanged (Table 1, Section 6.1) |
| Corpus check | `expected_num_samples: 9357`, `corpus_check: error` | a run against the wrong root dies at startup, not 13 GPU-hours later |

### Four things about this stage that are easy to get wrong

**The probe, not the loss, chooses the checkpoint.** The DINO loss is a cross
entropy against a teacher that moved, so `CE = H(teacher) + KL(teacher‖student)`.
Measured on a 100-epoch run: **80 % of the total loss drop was `H` falling**, the
final loss was **94.8 % irreducible target entropy**, its minimum was at epoch 90
— and the *representation* peaked at epoch 50 (0.6358) and fell to 0.6284 by
epoch 100. The pipeline published epoch 100, i.e. a measurably worse encoder, and
nothing in the loop could have known. `RepresentationProbe` now scores frozen
features at each milestone, `CheckpointSelector` keeps the winner as
`dino_best_encoder.pth`, and `experiment.training.publish: best` hands *that* to
stage 2. `patience` ends the run on a plateau.

*The probe is fitted on the whole corpus*, so publishing on it is a mild form of
selection on the evaluation. It is disclosed in `summary.json` under
`config.selection`, and `experiment.training.publish=final` is the alternative.
Its readout is crop-level stratified, matching the primary protocol, which means
its absolute value sits ~18 pp above a photograph-disjoint estimate. It ranks
checkpoints *of one run*, where that offset cancels — never quote it as a
generalisation number.

**`train/loss` is not a learning curve.** Same decomposition, same consequence:
the raw curve was flat from epoch 20 while the learnable part was still improving
at epoch 93. Every step and epoch record carries `teacher_student_kl` and
`teacher_entropy_cross_view` alongside `loss`. **Read the KL.** An arm that
changes the centering or the prototype count moves `H` directly, so its raw loss
is not comparable with any other arm's.

**Teacher entropy ships with its bounds.** `H` scales with `log K`, so halving
`out_dim` moves it by `log 2` for free — read `teacher_entropy_normalized`. And an
exactly doubly-stochastic Sinkhorn assignment cannot put a row on fewer than
`K / B_teacher` prototypes, so `H_min = log(K / B_teacher)`: at `K = 2048`,
`B_teacher = 128` that is 2.77 of a 7.62 maximum, i.e. more than a third of the
nominal range is unreachable. Do **not** label high entropy healthy or low
entropy collapse without those conditioners.

**`loop_blocked_fraction` is not a GPU-idle fraction.** It is the share of wall
clock the loop spent inside the dataloader's `__next__`; nothing synchronises
inside the step, so the queued GPU work drains *during* that window and the
metric **upper-bounds** idleness. Turning `1 - loop_blocked` into GPU-busy time
gives the CPU enqueue time instead. Set
`experiment.training.measure_gpu_busy=true` for a real CUDA-event measurement, at
the cost of one stall per logging interval.

### Stage-1 controls and arms

```bash
# capacity control: the identical recipe on SwinV2-Base at the SAME IN-1k corpus
python -m src.trainers.contrastive_pretrain experiment=pretrain_dino_base

# corpus control: SwinV2-Base pretrained on ImageNet-22k. NOT the same question
# as the line above -- run `screen-backbones` first, which separates the two.
python -m src.trainers.contrastive_pretrain experiment=pretrain_dino_base_in22k

# single-factor overrides off the primary recipe
python main.py pretrain model.loss.koleo_scope=all_views     # the anti-alignment control
python main.py pretrain model.loss.lambda_koleo=0            # is KoLeo worth anything here?
python main.py pretrain model.loss.koleo_space=bottleneck    # the discarded 256-D space
python main.py pretrain model.loss.centering=ema             # instead of Sinkhorn
python main.py pretrain data.augmentation.match_view_lowpass=true
python main.py pretrain data.augmentation.same_photo_local_views=2
python main.py pretrain data.augmentation.min_native_pixels=900
python main.py pretrain model.head.aux_stage=2               # a second head on layers.2
python main.py pretrain experiment.training.publish=final    # publish the last epoch
```

Both Base controls **must** publish to a different `shared_backbone_path`, which
they do — writing the shared one would swap the trunk under every stage-2 variant
and `checkpoint_strict: false` would report the 768-vs-1024 mismatch as one line
about missing keys.

`koleo_scope=all_views` deserves its own sentence, because it is the failure the
default fixes. The student's globals arrive view-major, so rows `[0:B]` are view
0 of every image and `[B:2B]` are view 1 of the *same* images in the same order.
Applied to the concatenated block, the nearest neighbour of row `i` is row `B+i`
— the other view of one crop — so `-log(min distance)` pushes apart exactly the
pair Eq. 1 pulls together. Measured on the run that shipped it, `alignment` got
*worse* over training (0.638 → 1.111). Keep it as a control; do not make it the
default again.

### The arm suite

Stage-1 arms have their own runner, because an arm *produces* an encoder and then
needs a second process to evaluate it — and without per-arm `save_path`,
`shared_backbone_path` and `experiment.evaluation.save_path`, the arms silently
overwrite each other's encoders and each other's `outputs/eval_pretrain/`.

```bash
python scripts/run_stage1_ablations.py --arms conf/stage1_arms/screens.yaml       # no training
python scripts/run_stage1_ablations.py --arms conf/stage1_arms/view_design.yaml   # the decomposition
python scripts/run_stage1_ablations.py --arms conf/stage1_arms/view_design.yaml --dry-run
python scripts/run_stage1_ablations.py --arms conf/stage1_arms/view_design.yaml --seeds 42 43 44
```

The arms are **data, not code**: each manifest names a base experiment, shared
overrides, and one entry per arm. `view_design.yaml` decomposes the primary
recipe into single factors — `full`, `wo_view_redesign`, `wo_colour_policy`,
`wo_dihedral`, `koleo_bottleneck`, `native_pixel_floor`, `ema_centering`,
`stage3_readout`, plus the `frozen` reference — and each entry states its
hypothesis and how to judge it.

**What would falsify the design, stated in advance.** If `full`,
`wo_view_redesign` and `wo_colour_policy` all land within ±1 SD of each other and
of `frozen`, the objective is not the binding constraint and **81 scenes is the
ceiling**. The remaining effort then belongs in data acquisition (the 18
uncropped photographs; second sessions for the five single-photograph varieties),
in the initialisation, and in the stage-2 readout — none of which is a stage-1
recipe change. The fold SD is ±0.10 on the 27-way probe, so one arm cannot
resolve anything below ~2 pp; `--seeds` is what turns a ranking into a claim.

## Stage 1.5 — evaluate the representation

```bash
python main.py eval-pretrain                                       # the full report
python main.py eval-pretrain experiment.evaluation.max_samples=270 # 1-min plumbing check
```

Stage 1 produces an encoder, not a classifier, so its loss curve cannot say
whether the run was worth its GPU-hours. This stage answers that from the
representation itself, with instrument families that fail *independently*:

* **label-free geometry** — RankMe, participation ratio, stable rank, dead
  channels, alignment/uniformity. These catch what a probe on 9 k samples hides.
* **frozen-feature readout** — a linear probe plus a parameter-free weighted
  cosine k-NN, with a low-shot curve, layer-wise probes, calibration and
  retrieval.
* **structure recovered without labels** — k-means and DINO's own 2,048-way
  prototype argmax, scored against the taxonomy.
* **nuisance** — see below.

It scores the probe-selected encoder *and* its numbered milestones, the
ImageNet-1k initialisation it started from, an untrained trunk, and a handcrafted
floor, so every number sits between a floor and a baseline. It also recovers the
stage-1 collapse diagnostics from the finished run's `events.jsonl`.

Four facts about this stage that are easy to undo:

- **`nuisance_decodability` is the gate, not a headline.** With class identity
  held constant, how well can the *source photograph* be recovered? That is the
  exact confound a photograph-disjoint protocol punishes, and it is the one axis
  on which in-domain self-distillation demonstrably worked: +10.0 pp above chance
  from the ImageNet weights, +3.5 pp after DINO — a 65 % reduction. Read it
  *jointly* with the readout: an encoder that discards everything scores exactly
  chance. It is also the mandatory gate on
  `data.augmentation.same_photo_local_views`, which can win the readout by
  re-learning the confound.
- **`handcrafted_floor` is a reporting obligation.** Ten numpy scalars (log area,
  log aspect ratio, mean/std RGB, mean/std grey) under the identical protocol.
  They score **0.5360** 27-way photograph-disjoint — 15.6 pp *above* an untrained
  48.96 M-parameter trunk. `random_init` answers "what does the architecture give
  for free"; this answers the question a reviewer asks first.
- **The feature cache is keyed on the checkpoint SHA-256, not the path.**
  `dino_backbone_epoch_0050.pth` is a name every run reuses, so a path-keyed cache
  would serve the previous run's features after a retrain. Re-running after
  changing only an analysis or a figure is therefore cheap.
- **Calibration needs three disjoint splits.** The headline probe is refit on
  train+val, so the temperature is fitted on a *second* probe trained on the
  train fold only. Mixing the two makes the temperature-scaled ECE worse than the
  raw one — a broken protocol, not a failure of temperature scaling.

Everything lands in `outputs/eval_pretrain/`: `summary.json`, `metrics.json`,
`tables/*.csv`, 22 figures at 300 dpi, cached features, and `provenance.json`
with checkpoint SHA-256s **and the corpus fingerprint on both sides of the
handoff**. The design rationale is
[`architecture/08_STAGE1_REPRESENTATION_EVALUATION.md`](architecture/08_STAGE1_REPRESENTATION_EVALUATION.md).

## Stage 2 — hierarchical MoE finetuning

```bash
SEED_PRETRAIN_BACKBONE=$SEED_OUTPUT_DIR/checkpoints/dino_pretrained_encoder.pth \
    python main.py finetune         # experiment=finetune_hierarchical_moe
```

(The variable is only needed if you want a *different* encoder — the default
already points at the path stage 1 publishes to.)

Stage 2 freezes the trunk by default and trains the head plus the Eq. 4
projection: Top-2-of-6 routing over the `8×8` token grid, the Eq. 9 residual
fusion, the Eq. 11–12 cross-attention, and an ArcFace head over 27 sub-varieties
with the analytic AdaCos scale `√2·log(C−1) = 4.61`.

### What the headline number measures, stated once

The primary protocol is **crop-level stratified**: 20 % of the *crops* held out,
stratified on sub-variety, then a stratified train/validation pair; select on
validation, report on test.

Its cost is measured rather than assumed. Under an identical frozen encoder and
probe:

| | photograph-disjoint | crop-level | delta |
| --- | --- | --- | --- |
| 27-way probe accuracy | 0.6500 | 0.8365 | **+18.65 pp** |
| 27-way k-NN accuracy | 0.5099 | 0.6891 | +17.92 pp |
| retrieval P@1 | 0.4745 (same-photo neighbours excluded) | 0.6636 | +18.91 pp |

and 89–98 % of the crops in a photograph have a neighbour above cosine 0.95 at
32 × 32 grey. **So a crop-level number answers "how well does this classify a new
crop from a *known* acquisition session" and not "…from a new session."**

Nothing hides that. `leakage_report` runs for both protocols and
`shared_source_groups`, `leaked_test_fraction` and `classes_present_in_test` land
in `summary.json` for every run, and:

```bash
python main.py finetune-grouped     # experiment=finetune_grouped_diagnostic
```

is the identical configuration under photograph-disjoint `grouped_cv`, so the gap
is a property of *this* encoder rather than a figure quoted from another one.

Two things the diagnostic number is not: it estimates the **recipe** (K different
models contributed out-of-fold predictions), not any single shipped model's test
score; and it is not comparable with the crop-level number as "better" or
"worse", only as a measurement of the gap between two questions.

`grouped_cv` rather than `grouped` for the diagnostic, because `grouped`'s
`GroupShuffleSplit` takes 20 % of the 81 photographs *unstratified* — the test
side then holds 14 of the 27 classes and a 27-way macro-F1 on it is mechanically
capped near 14/27 for reasons unrelated to the model. `grouped_cv` partitions
every crop into photograph-disjoint folds and concatenates the out-of-fold
predictions, so every class is scored. Both remain one override away:

```bash
python main.py finetune experiment.training.split_protocol=grouped
python main.py finetune experiment.training.split_protocol=grouped_cv experiment.training.num_folds=5
```

## 10. Analysis: ablations, baselines, figures

```bash
python scripts/run_ablations.py               # component-wise variants, 5 seeds each
python scripts/run_baselines.py               # linear probe, ImageNet frozen/unfrozen, ResNet-50, Swin-T, hierarchical CCE
python scripts/generate_plots.py              # figures + outputs/reports/summary_metrics.csv
```

| Variant | What it removes | Selected by |
| --- | --- | --- |
| `full_model` | nothing (Top-2 MoE + ArcFace + residual + KL + cross-attention) | — |
| `wo_moe` | sparse routing; one dense transformer block instead | `model.head.use_moe=false` |
| `wo_margin_only` | the angular margin alone (NormFace keeps the geometry) | `model.head.sub_head_variant=normface` |
| `wo_angular_head` | margin **and** normalisation **and** logit scale | `model.head.sub_head_variant=linear` |
| `wo_residual` | Eq. 9 seed-type fusion | `model.head.use_residual=false` |
| `wo_kl` | Eq. 10 hierarchy-consistency loss | `model.head.use_kl_loss=false` |
| `wo_cross_attn` | Eqs. 11–12 Q/K/V refinement | `model.head.use_cross_attention=false` |
| `linear_probe` | everything but a frozen encoder + two linear heads | `experiment=baseline_linear_probe` |
| `swinv2_supervised` | the self-supervised stage (ImageNet SwinV2, trunk unfrozen) | `experiment=baseline_swinv2_supervised` |
| `imagenet_frozen` | the self-supervised stage (ImageNet SwinV2, trunk frozen) | `experiment=control_imagenet_frozen` |
| `resnet50` | ImageNet ResNet-50, supervised end to end | `experiment=baseline_resnet50` |
| `swin_tiny` | ImageNet Swin-T, supervised end to end | `experiment=baseline_swin_tiny` |
| `hierarchical_cce` | two-stage hierarchy, plain CCE, no MoE/attn/ArcFace | `experiment=baseline_hierarchical_cce` |
| `leakage_grouped` | nothing architectural — the full model under photograph-disjoint folds | `experiment.training.split_protocol=grouped_cv` |

`scripts/run_ablations.py` carries more variants than the table above
(`wo_moe_capacity_matched`, `moe_fixed_router`, `moe_uniform_router`,
`wo_gate_conditioning`, `wo_layer_scale`, `film_fusion`, `kl_jsd`,
`pooled_tokens`, `load_entropy`, `wo_stage2_augmentation`); each `VariantSpec`
states **every** factor it changes, which is the rule for adding one. `list` them
with `python scripts/run_ablations.py --dry-run`.

`leakage_grouped` is the one row that is not comparable by McNemar against
`full_model`: it does not share a test split, and it estimates the recipe rather
than one trained model.

A disabled component is **not allocated**, so an ablation's parameter count
describes the model actually trained. `wo_moe` keeps one dense block of identical
architecture rather than deleting the layer, so its gap against the full model
measures *routing* and not a missing block's capacity.

Every variant runs at five seeds into `{group}/{variant}/seed{n}/`, because one
run per variant cannot resolve the gaps the table reports: the 95 % CI half-width
on a *difference* of two accuracies on this test split is ±1.40 pp, against
component contributions of 0.5–2 pp. `generate_plots.py` aggregates to mean ± SD
and runs McNemar's exact test, which is valid because the split is byte-identical
across variants.

All variants read the **same** published encoder at
`$SEED_OUTPUT_DIR/checkpoints/dino_pretrained_encoder.pth`, and
`ensure_pretrained_checkpoint()` refuses to start otherwise. Pretraining is never
repeated per variant — if each variant had its own self-supervised
initialisation, the table would partly measure that rather than the architectural
change under test.

Useful flags: `--dry-run` prints the commands without running them,
`--variants`/`--models` selects a subset, `--gpus 0,1` shards one variant per
device, and everything after a bare `--` is forwarded to every run as a Hydra
override.

With more than one GPU, prefer sharding the *suite* over the devices rather than
running one variant across them: the variants are already independent processes,
so there is no gradient traffic and each keeps the exact numerics of a
single-GPU run.

## 11. Hydra overrides that matter

Everything after the stage name is a Hydra override.

**Protocol and evaluation**

```bash
python main.py finetune experiment.training.split_protocol=grouped_cv experiment.training.num_folds=5
python main.py finetune experiment.training.test_size=0.3
python main.py eval-pretrain experiment.evaluation.split.protocol=grouped
python main.py eval-pretrain experiment.evaluation.max_samples=270      # plumbing check
```

**Architecture**

```bash
python main.py finetune model.head.top_k=4                  # submitted Top-4 routing
python main.py finetune model.head.token_mode=pooled        # submitted pooled head
python main.py finetune model.head.use_moe=false            # component ablation
python main.py finetune model.backbone.freeze=false         # fine-tune end to end (the paper's setting)
python main.py finetune model.backbone.feature_stage=stage3_pooled_2x2   # read layers.2
python main.py finetune model.backbone.name=swinv2_small_window16_256 model.backbone.feature_dim=768
```

Two of those deserve a sentence, because neither is comparable with a previously
published number:

- **`feature_stage`** selects which trunk stage the encoder reads: `final` (the
  default, `layers.3` post-norm, 8×8×768), `stage3` (`layers.2`, 16×16×**384** —
  natively the paper's `z`), or `stage3_pooled_2x2` (the same, 2×2-average-pooled
  back to a 64-token grid). Linear CKA between a self-distilled encoder and its
  ImageNet initialisation is 0.976 / 0.960 / 0.390 / **0.103** across
  `layers.0..3`: self-distillation rewrote the last stage to serve its
  2,048-prototype head, and that is the stage stage 2 reads. On the
  photograph-disjoint out-of-fold probe `layers.2` scored **+3.25 pp** linear and
  **+3.90 pp** with a 512-unit MLP, and the ordering held for the plain ImageNet
  weights too. Concatenating all four stages was no better than `layers.2` alone
  — this is a replacement, not a fusion. Prefer `stage3_pooled_2x2`: plain
  `stage3` quadruples the routing slots and makes the Eq. 11 attention 16× as
  expensive, and the +3.25 pp was measured on the *mean-pooled* stage-3 feature.
  `feature_stage` changes what is **read**, never what is stored, so every
  existing checkpoint loads at every stage.
- **`token_mode`** decides whether three modules are functions or affine maps.
  `grid` (default) keeps SwinV2's `8×8` token grid through the MoE and the
  cross-attention and pools afterwards; `pooled` reproduces the submitted
  architecture. Over a length-1 sequence `softmax(QKᵀ/√d)` is identically 1, so
  `Q` and `K` receive exactly zero gradient forever — under `pooled` the head
  therefore does **not allocate** those projections and substitutes the single
  `nn.Linear` that spans the identical function class. Grid routing also raises
  routing slots per step from `batch × K` to `batch × 64 × K`, which is what makes
  the load-balancing statistic estimable at batch 16.

**Loss**

```bash
python main.py finetune model.loss.lambda_kl=0.5
python main.py finetune model.head.use_kl_loss=false
python main.py pretrain model.loss.centering=ema model.loss.center_momentum=0.99
python main.py pretrain model.head.out_dim=1024
```

**Budget, throughput and precision**

```bash
python main.py pretrain data.batch_size=32 experiment.training.effective_batch_size=32
python main.py pretrain data.num_workers=16
python main.py pretrain experiment.training.amp=bf16          # auto is the default
python main.py pretrain experiment.training.sdpa_attention=false
python main.py pretrain experiment.training.compile.enabled=false
python main.py pretrain experiment.training.measure_gpu_busy=true
python main.py pretrain experiment.training.epochs=25 'experiment.training.save_epochs=[5,10,15,20,25]'
python main.py pretrain experiment.training.probe.every_epochs=10
```

**Measure before changing the batch.** Physical batch is what Sinkhorn and KoLeo
estimate from, and accumulation cannot substitute for it — if the card cannot
hold it, lower `data.batch_size` and `effective_batch_size` *together* rather
than raising accumulation:

```bash
python scripts/bench_pretrain_step.py --find-batch-size 16,24,32,48,64
python scripts/bench_pretrain_step.py --scaling 1,2          # single-GPU vs DDP
```

### Reproducing the submitted configuration

| Departure | Override that reverses it |
| --- | --- |
| Top-2 routing (paper: Top-4) | `model.head.top_k=4` |
| Grid routing (paper: pooled) | `model.head.token_mode=pooled` |
| AdaCos ArcFace scale 4.61 (paper: 30) | `model.head.arcface_scale=30.0` |
| SwinV2-Tiny (paper: Base, random init, 300 epochs) | `experiment=pretrain_dino_base model.backbone.pretrained=false experiment.training.epochs=300` |
| Corpus-sized view geometry (paper: DINO/ImageNet ranges) | `data.local_crop_size=101 'data.augmentation.global_crops_scale=[0.40,1.00]' 'data.augmentation.local_crops_scale=[0.05,0.40]' 'data.augmentation.crop_ratio=[0.75,1.3333333333333333]'` |
| Colour-preserving photometry | `data.augmentation.color_jitter_saturation=0.2 data.augmentation.color_jitter_hue=0.1 data.augmentation.grayscale_prob=0.2 data.augmentation.solarization_prob=0.2 data.augmentation.global_blur_prob_1=1.0 data.augmentation.global_blur_prob_2=0.1` |
| Per-view KoLeo on the trunk feature | `model.loss.koleo_scope=all_views model.loss.koleo_space=bottleneck` |
| Probe-selected checkpoint | `experiment.training.publish=final` |
| Frozen stage-2 trunk | `model.backbone.freeze=false` (the paper fine-tunes it) |

`conf/stage1_arms/view_design.yaml` runs the first five of those as controlled
single-factor arms, which is the measured version of the same question.

## 12. Multi-GPU, resuming, and rented boxes

```bash
python main.py pretrain --gpus 2      # pins $SEED_RUN_ID, then torch.distributed.run
python main.py pretrain --gpus auto   # every visible CUDA device
python scripts/launch.py pretrain --gpus 2 --dry-run
```

Three facts about the distributed path:

* **Images shard; views do not.** Eq. 1 pairs a student view against the
  teacher's output for *that same image*, so all `2 + local_crops_number` views of
  a sample must be resident on one device.
* **`experiment.training.effective_batch_size` is the authority**, not
  `gradient_accumulation_steps`. `data.batch_size` is the *per-rank* micro-batch,
  so launching on two GPUs would otherwise double the global batch and with it
  the LR/momentum regime every schedule is tuned to. `resolve_accumulation()`
  derives the accumulation count from the target and the world size and
  **refuses** a combination that does not divide exactly.
* **The corollary cuts against the usual reading.** Splitting 64 across two ranks
  as `32 × 2` keeps the gradient identical and *halves* what Sinkhorn and KoLeo
  estimate from, because both are per-micro-batch statistics.
  `model.loss.distributed_sinkhorn=true` normalises over the concatenated global
  batch instead — exact, and pinned against a single-process reference — but it is
  a *different objective*, so launching on two GPUs does not turn it on.

**Resume is a continuation, not a warm restart.** The same command line serves
the first launch and every relaunch, because `resume=auto` starts fresh when
there is nothing to continue:

```bash
python main.py pretrain --gpus 2 \
    experiment.training.resume=auto \
    experiment.training.max_runtime_minutes=520
```

The resume checkpoint carries the teacher, optimizer moments, scheduler,
`GradScaler`, epoch, global step, **micro-batch within the epoch**, and one RNG
snapshot per rank. Every write is atomic (`torch.save` truncates its destination
first, so a session killed mid-save would otherwise leave a zero-length file
*where the good checkpoint used to be*); keep `keep_last_n_checkpoints ≥ 2` on a
preemptible platform, because the newest file is the one a kill is most likely to
have interrupted.

On a rented server:

```bash
export SEED_DATA_ROOT=/workspace/data/Hierarchical_SeedData/Refined_Samples
export SEED_OUTPUT_DIR=/workspace/outputs
GPUS=2   scripts/train_distributed.sh pretrain      # DDP over 2 GPUs
GPUS=1   scripts/train_distributed.sh eval-pretrain
GPUS=0,1 scripts/train_distributed.sh ablations     # one variant per GPU
GPUS=0,1 scripts/train_distributed.sh baselines
scripts/train_distributed.sh report
```

[`SERVER_RUN_GUIDE.md`](SERVER_RUN_GUIDE.md) is the long form: sizing, disk
budget, what to watch in the first epoch, and what each failure mode looks like.

## 13. Outputs

```text
$SEED_OUTPUT_DIR/
  checkpoints/
    dino_pretrained_encoder.pth        # THE handoff: every downstream run reads this
  pretrain_dino/
    dino_best_encoder.pth              # the probe-selected encoder (publish: best)
    dino_pretrained_backbone.pth       # the stage-2 handoff as written by this run
    dino_pretrained_final.pth
    dino_backbone_epoch_{0005..0050}.pth   # milestone encoders, never pruned
    dino_milestone_epoch_*.pth
    summary.json                       # the resolved recipe, corpus fingerprint, budget
    events.jsonl                       # the complete record
    csv/{metrics_train,metrics_epoch,metrics_probe,probe_history,
         checkpoint_selection,view_geometry}.csv
    figures/stage1/{01_optimization,02_representation,03_collapse,
                    04_view_geometry,05_throughput}.{png,pdf}
  eval_frozen_reference/               # the no-training bar
  eval_pretrain/                       # stage-1 representation evaluation
    summary.json  metrics.json  provenance.json
    split_manifest.npz
    test_predictions.npz               # single-split probe predictions
    out_of_fold_predictions.npz        # the same format over every crop
    tables/*.csv                       # encoder comparison, per-class, low-shot, layer-wise, ...
    figures/fig01..fig22*.png          # 300 dpi
    features/{encoder}.npz             # cached frozen features, keyed on checkpoint SHA-256
  finetune_hierarchical_moe/
    best_hierarchical_moe.pth  hierarchical_moe_final.pth
    split_manifest.npz  summary.json  test_predictions.npz
  finetune_grouped_diagnostic/         # the photograph-disjoint counterpart
  ablations/{full_model,wo_moe,wo_margin_only,wo_angular_head,wo_residual,wo_kl,wo_cross_attn}/
  baselines/{linear_probe,swinv2_supervised,resnet50,swin_tiny,hierarchical_cce}/
  controls/imagenet_frozen/
  reports/summary_metrics.csv          # one row per variant, all metrics + cost
  metadata/seed_dataset.csv
```

**Run artifacts are a contract.** Every run writes `summary.json` and
`test_predictions.npz`; `scripts/generate_plots.py` reads only those and
re-scores from the raw predictions rather than trusting the stored metrics, so
the table and the figures are always computed by the same code. Keeping the *raw*
predictions and embeddings — not only the figures — means a reviewer asking for a
differently-normalised confusion matrix or a re-coloured t-SNE costs a second of
replotting rather than a full retrain.

**The corpus is recorded and cross-checked.** The stage-1 → stage-2 handoff is a
bare `state_dict`, and nothing used to record what it was trained on. That was
not hypothetical: one shipped encoder was self-distilled on 8,173 crops while
everything downstream used 9,357, recoverable only by cross-reading two log lines
against `metrics.json`. `corpus_fingerprint()` now returns a SHA-256 over the
sorted dataset-relative paths plus the sample, class and source-group counts and a
per-class histogram; stage 1 logs it, writes it to `events.jsonl` and puts it in
`summary.json`, and `eval-pretrain` reads that file back and prints a prominent
mismatch line. `data.expected_num_samples` makes the whole class of error fatal at
startup instead.

**Disk.** Defaults are tuned for a 16 GB rented root disk: parameter/gradient
histograms off, `keep_last_n_checkpoints: 2`, no optimizer state, no teacher
weights. The milestone set is the one thing that grows — every probed epoch is
forced into `save_epochs`, so the effective set is ~11 epochs at ~231 MB each
(**≈2.5 GB**) plus ~1 GB of rolling resume state. On a tight disk raise
`experiment.training.probe.every_epochs` rather than trimming `save_epochs`, so
the probe and the milestones stay in step.

### Tracking

Three sinks, configured under `conf/tracking/default.yaml`:

* **`events.jsonl`** — always on, dependency-free, survives a crashed run.
* **TensorBoard** — on by default. `tensorboard --logdir $SEED_OUTPUT_DIR`
* **Weights & Biases** — on by default in `offline` mode, so a run never blocks on
  credentials. Sync afterwards with
  `wandb sync $SEED_OUTPUT_DIR/**/wandb/offline-run-*`, or set
  `tracking.wandb.mode=online`.

Stage 2 logs, every epoch: loss curves broken down by component, seed-type and
sub-variety accuracy / F1 / precision / recall / AUC, per-class tables, the KL
alignment rate overall and per seed type, MoE expert utilisation, and the
train-vs-validation loss gap (`epoch/overfitting_gap`). Every
`tracking.intervals.figure_every_epochs`: confusion matrices for both levels, the
sub-variety metric heatmap, per-sub-variety misclassification rates,
expert-utilisation bars, train/validation loss curves, and t-SNE projections
coloured by both label levels.

### Computational efficiency

`src/utils/efficiency.py` profiles the deployed path — encoder *and* head, since
that is the only combination whose latency a user could observe — and reports
total vs. active parameters (active is total minus the `(E − K)` experts that sit
out each forward pass: a closed form, not an estimate), FLOPs via
`torch.utils.flop_counter` (real ATen dispatch, so a sparsely routed MoE is
counted at its true cost automatically), latency and throughput at several batch
sizes with warm-up and explicit device synchronisation, and peak memory where the
backend exposes it. Each measurement degrades to `None` rather than raising.

Stage 1's budget report labels **measured** and **estimated** quantities apart:
parameter counts, GFLOPs/view, peak VRAM, wall clock and throughput are measured;
per-iteration and whole-run FLOPs are estimated (they assume backward = 2×
forward and count the teacher's globals forward-only), carry an `estimated_`
prefix and print as `[ESTIMATED]`, so a chart axis cannot present a derived number
as a measured one.

## 14. Testing and verification

```bash
python -m pytest tests/ -q                 # 666 tests, no network access, ~60 s
python -m pytest tests/test_segmentation.py -q        # stage 0: detection, splitting, crop policy, audit
python -m pytest tests/test_stage1_pipeline.py -q     # view geometry, protocol, artifacts
python -m pytest tests/test_stage1_correctness.py -q  # KoLeo, loss decomposition, provenance
python scripts/dry_run.py                  # real encoder, synthetic data, full pipeline
python scripts/verify_runtime.py --gpus 2  # numerical checks on THIS machine
python scripts/diagnose_sdpa_parity.py     # per-module SDPA parity report
python main.py smoke                       # 2 batches through both stages, real dataset
```

The suite asserts the paper's numbers directly — 384-D embedding, 6 experts,
Top-2 routing, τ_t 0.04 → 0.07, momentum 0.996, gradient clip 3.0 — so a failure
means the code has drifted from the paper. Paper constants live in
`tests/conftest.py` and mostly come in **pairs** (`SUBMITTED_TOP_K` /
`REVISED_TOP_K`, `SUBMITTED_GLOBAL_CROPS_SCALE` / `CANONICAL_GLOBAL_CROPS_SCALE`,
`LEGACY_NUM_CROPS` / `DATASET_NUM_CROPS`, `LEGACY_CROP_RATIO` /
`CANONICAL_CROP_RATIO`, …), so a test asserting a bare number cannot silently
become a claim about whichever value — or whichever corpus — the reader assumed.

`tests/test_segmentation.py` runs stage 0 against a **synthetic photograph** built
to the measured properties of the real one — illumination gradient, sensor noise,
dust, a paper-edge band and a touching pair at a known position — so the whole of
stage 0 is exercised with ground truth and without the dataset on disk.

An end-to-end pre-flight on the real dataset, without a GPU:

```bash
python main.py pretrain data.batch_size=4 data.num_workers=0 \
    experiment.training.effective_batch_size=4 \
    experiment.training.epochs=2 experiment.training.max_batches=3 \
    'experiment.training.save_epochs=[1,2]' \
    experiment.training.probe.every_epochs=1 \
    experiment.training.probe.max_samples=270 \
    model.backbone.pretrained=false device=cpu
```

## 15. Repository layout

| Path | Contents |
| --- | --- |
| [`src/segmentation/`](src/segmentation/README.md) | **Stage 0**: photographs → the one-seed-per-file corpus, its audit and its benchmark |
| [`src/models/`](src/models/README.md) | SwinV2 encoder, hierarchical head, MoE / cross-attention / ArcFace, supervised baselines |
| [`src/losses/`](src/losses/README.md) | DINO, ArcFace, KL hierarchy, MoE regularisation, cosine compactness |
| [`src/datasets/`](src/datasets/README.md) | Hierarchical image-folder dataset and the DINO multi-crop pipeline |
| [`src/trainers/`](src/trainers/README.md) | Hydra training entry points and the suite runner |
| [`src/utils/`](src/utils/README.md) | Metrics, efficiency profiling, reporting, figures, tracking |
| [`conf/`](conf/README.md) | Hydra config groups — one data group, one trunk, one experiment per stage |
| [`scripts/`](scripts/README.md) | Suite runners, plotting, benchmarking, reporting, dry run |
| [`tests/`](tests/README.md) | pytest suite |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | One-page map of both stages and the interface between them |
| [`architecture/`](architecture/00_OVERVIEW.md) | Per-topic design documents |
| [`SERVER_RUN_GUIDE.md`](SERVER_RUN_GUIDE.md) | Running the pipeline on a rented / preemptible GPU box |
| [`CLAUDE.md`](CLAUDE.md) | Working notes for automated contributors: the invariants that are silent when broken |
