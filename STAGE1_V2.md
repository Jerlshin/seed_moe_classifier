# STAGE1_V2.md — the redesigned stage-1 pipeline

**Scope.** What changed in stage 1, why each change was made, what evidence
supports it, and what would falsify it. Written after an independent audit of
the existing implementation, so several items here *correct* or *narrow*
recommendations in `STAGE1_CHANGES.md` rather than implementing them as written.

**The reference point.** The shipped 100-epoch run cost 13.34 h on an H100 and
bought **+0.31 pp** of 27-way accuracy over the ImageNet-1k weights it started
from — 0.6284 against 0.6253
(`outputs/eval_pretrain/tables/encoder_comparison.csv`). Every change below
targets one of the reasons that number is so small.

---

## 0. What this audit measured independently

Three measurements were reproduced from scratch before anything was changed, and
one of them contradicts the plan it was checking.

### 0.1 The view-content figures reproduce

Monte-Carlo over all 9,357 crops through torchvision's own
`RandomResizedCrop.get_params`, against `STAGE1_CHANGES.md` §0.5's independent
measurement:

| view | this audit (p5 / p50 / p95) | §0.5 | agreement |
| --- | --- | --- | --- |
| global `scale=(0.40, 1.00)` | 648 / **1,833** / 5,796 | 675 / 2,035 / 7,101 | same order, same conclusion |
| local `scale=(0.05, 0.40)` | 132 / **598** / 2,632 | 130 / 598 / 2,608 | exact |

The corpus itself: 9,357 files, median **52 × 51 px**, area p5/p50/p95 =
1,188 / 3,024 / 9,026 px², **3.4 %** square, aspect p5/p95 = 0.52 / 1.98,
**51.0 %** with both sides under 64 px, and 0.01 % with any side ≥ 256.

So the headline claim holds: **a local view is a median 24 × 24 px fragment of
one seed inflated to 65,536 output pixels — 0.91 % real content — and 8 of the
10 cross-view terms in Eq. 1 are anchored on one.**

### 0.2 The audit's C1 proposal has a measured cost it does not mention

`STAGE1_CHANGES.md` C1 proposes `global_crops_scale=(0.70, 1.0)` and leaves the
aspect-ratio range at torchvision's default. Measured:

| policy | deterministic-centre-crop fallback |
| --- | --- |
| global `(0.40, 1.00)`, ratio `(0.75, 1.33)` — current | **3.3 %** |
| global `(0.70, 1.00)`, ratio `(0.75, 1.33)` — C1 as written | **21.5 %** |
| global `(0.70, 1.00)`, ratio `(0.50, 2.00)` — adopted | **9.3 %** |

`get_params` retries the (area, aspect) draw ten times and then returns a
**fixed centre box**. A high-area crop with aspect in (0.75, 1.33) does not fit
inside a non-square source, and 96.6 % of these crops are non-square — so
raising the scale floor buys content by spending randomness, and roughly one
global view in five would have carried no crop randomness at all.

Widening `crop_ratio` recovers it *and* raises the median content further
(2,440 vs 2,262 native px). It is therefore strictly better on both axes, and it
is why `crop_ratio` exists as a config key.

*(A first pass of this measurement counted single-attempt failures rather than
fallbacks and put the C1 rate near 67 %. Torchvision retries ten times; the
corrected figure is 21.5 %.)*

### 0.3 The KoLeo fix and the corpus fingerprint are already in place

`grouped_koleo(..., scope="per_view")` is the default and `koleo_scope=all_views`
warns. `corpus_fingerprint()` is logged, written to `events.jsonl` and carried in
`summary.json`. Both were verified in the source rather than taken from the
change log. What was **missing** is a *fatal* check — the fingerprint made a
wrong corpus discoverable afterwards, not impossible beforehand. That is now
`data.expected_num_samples`.

---

## 1. What changed

### 1.1 View generation, redesigned for 52 × 51 px sources

`src/datasets/transforms.py`, `conf/data/seed_crops_v2.yaml`.

| | submitted | v2 | measured effect |
| --- | --- | --- | --- |
| `global_crops_scale` | (0.40, 1.00) | **(0.70, 1.00)** | global native px 1,833 → 2,440 |
| `local_crops_scale` | (0.05, 0.40) | **(0.30, 0.70)** | local native px **598 → 1,440** |
| `local_crop_size` | 101 | **160** | removes an intermediate low-pass |
| `crop_ratio` | (0.75, 1.33) | **(0.50, 2.00)** | global fallback 21.5 % → 9.3 % |
| local upsample to 256 | 10.5× median, 22.3× p95 | **6.7× / 11.5×** | narrows the local/global resolution cue |
| real content per local view | 0.91 % | **2.20 %** | 2.4× |

Three new mechanisms, each separately switchable so the arms stay single-factor:

* **`crop_ratio`** — §0.2.
* **`min_native_pixels`** — a floor on the source pixels behind a view, raising
  the lower scale bound *per image* to `max(scale_lo, floor / area)`. The scale
  range is a *fraction*, so a fixed range shreds a 21 × 22 crop and an
  881 × 413 crop by the same factor. It never lowers the bound, so it can only
  make views less destructive, and at `0` it is a plain `RandomResizedCrop`
  exactly. At 900 px it lifts the local p5 from 483 → 832 (**+72 %**) and moves
  the median by 1.7 % — a tail intervention. **Off in the primary run**, run as
  the `V2-FLOOR` arm.
* **`rotation90_prob` / `vertical_flip_prob`** — the dihedral group of order 8,
  through `PIL.Image.transpose`. A pixel permutation: no interpolation, no
  resampling, no black corners, unlike `T.RandomRotation`. Label-preserving
  *here specifically* because seeds on a tray have no canonical orientation.
  This is what buys back the augmentation diversity the narrower crop ranges
  give up, which is what keeps the narrowing from being a step toward
  pretext-task memorisation on 81 scenes.

### 1.2 Colour is treated as signal, not only as nuisance

Mean RGB alone scores **0.3169** on the 27-way task and explains 78.5 % of the
between-sub-variety variance; only ~26.6 % of the within-class colour variance is
photograph-specific. The jitter is therefore split by *what it attacks*:

| | submitted | v2 | why |
| --- | --- | --- | --- |
| brightness / contrast | 0.4 / 0.4 | **0.4 / 0.4** | illumination — genuine photograph nuisance |
| saturation | 0.2 | **0.1** | pigmentation — class signal |
| hue | 0.1 | **0.02** | 0.1 is ±36° of the hue circle |
| grayscale | 0.2 | **0.0** | deletes the cue outright |
| solarization | 0.2 | **0.0** | inverts it |
| blur (g1 / g2 / local) | 1.0 / 0.1 / 0.5 | **0.5 / 0.5 / 0.5** | symmetric: blur no longer *identifies* the view family |
| blur radius max | 2.0 | **1.0** | the data has almost no content above ~52 px |

**What this does not claim.** The audit refuted the obvious version: mean RGB is
*more* linearly decodable after the shipped DINO run than before (pooled R²
0.864 → 0.908), so the jitter did not destroy the cue. This is a hypothesis about
where capacity goes, and `V2-noCOLOUR` is its control.

### 1.3 The trunk is SwinV2-Tiny

Frozen-feature screen (`STAGE1_CHANGES.md` Appendix 2), all ImageNet-supervised,
no DINO:

| trunk | params | GFLOPs/view | pooled 27-way | `layers.2` 27-way |
| --- | --- | --- | --- | --- |
| `swinv2_tiny_window16_256` | 27.6 M | 13.32 | 0.6021 | **0.6243** |
| `swinv2_small_window16_256` *(previous default)* | 49.0 M | 25.56 | 0.6053 | 0.6174 |

Tiny is −0.32 pp pooled and **+0.69 pp at the better readout stage**, for half
the FLOPs. Small is dominated. `conf/model/backbone/swinv2.yaml` is unchanged, so
this is an experiment-level choice and Small/Base remain one override away.

### 1.4 The epoch budget is decided by a probe, not by the loss

`src/utils/training/representation_probe.py`.

The loss cannot rank checkpoints. It is a cross entropy against a teacher that
moved: **94.8 %** of the shipped run's final loss was irreducible target entropy,
**80 %** of its total drop was that entropy falling, and its minimum was at epoch
90. What the representation actually did:

| epoch | 27-way probe | k-NN |
| --- | --- | --- |
| 25 | 0.6276 | 0.5019 |
| 50 | **0.6358** | 0.5017 |
| 100 | 0.6284 | 0.5003 |

**The best encoder was epoch 50 of 100, and the pipeline published epoch 100.**
The second half of that budget produced a measurably worse encoder, and nothing
in the loop could have known — the milestone probes were run later by a separate
process.

The probe closes the loop. At each probed epoch it extracts frozen features on an
augmentation-free pass and scores three families that fail independently:

* **readout** — linear probe + parameter-free weighted cosine k-NN, under the
  *same crop-level stratified protocol the primary pipeline uses*, so the number
  the selector optimises and the number the pipeline reports are one measurement;
* **geometry** — RankMe, participation ratio, stable rank, top-1 variance share,
  ‖mean unit vector‖, dead-dimension fraction. These catch what a probe on 9 k
  samples can hide;
* **nuisance** — within-sub-variety source-photograph decodability. Stage 1's one
  demonstrated effect on the shipped run was driving this from +10.0 pp above
  chance to +3.5 pp. It is also the **gate**: an arm that wins the readout while
  raising this has learned the photograph confound.

`CheckpointSelector` keeps the winner as `dino_best_encoder.pth`, and
`experiment.training.publish: best` hands *that* to stage 2 rather than the last
epoch. `patience` ends the run when the probe plateaus.

One thing to state plainly: the probe is fitted on the whole corpus, so
publishing on it is a mild form of selection on the evaluation. It is disclosed
here, recorded in `summary.json` under `config.selection`, and
`publish: final` is the alternative.

### 1.5 The corpus check is fatal, not advisory

`data.expected_num_samples: 9357` on the v2 experiment. The shipped encoder was
self-distilled on **8,173** crops while everything downstream used 9,357, nothing
on disk recorded it, and it took cross-reading two log lines against
`metrics.json` to find. The fingerprint makes that discoverable; this makes it
impossible.

### 1.6 Logging: four sinks, and the CSVs are the analysable one

`events.jsonl` stays the complete record. Added beside it:

* **`csv/metrics_train.csv`, `metrics_epoch.csv`, `metrics_probe.csv`** — wide
  format, one row per step/epoch, one column per metric, schema discovered at
  runtime and the file rewritten when it grows so it is always rectangular
  (`src/utils/training/csv_metrics.py`).
* **`csv/view_geometry.csv`** — the measured native-pixel content of each view
  family, written at startup.
* **`csv/probe_history.csv`, `csv/checkpoint_selection.csv`** — every probe, plus
  the running best and the plateau counter, so "why did it stop at epoch 30" is
  answerable from the artifacts alone.
* TensorBoard and W&B (offline) as before.

New metrics: the update-to-weight ratio `|dW| / |W|` (median/min/max across trunk
tensors) — the one optimisation diagnostic a gradient norm cannot substitute for,
since Adam normalises by the second moment; the full probe battery; and the view
geometry.

`summary.json` gains `config.augmentation_resolved` (the policy as the transform
itself sees it), `config.view_geometry` (what the augmentation *actually* built),
and `config.selection` (which epoch was chosen, on what metric).

### 1.7 Five publication figures, generated from the CSVs

`src/utils/stage1_figures.py`, PNG + PDF at 300 dpi, written automatically at the
end of a run — including an interrupted one — and regenerable from a finished run
directory with `scripts/plot_stage1_run.py`. They read `csv/*.csv` and nothing
else, so a figure and its table cannot disagree.

`01_optimization` (KL first, raw loss demoted), `02_representation` (readout,
geometry and the nuisance gate, with the selected epoch marked),
`03_collapse` (teacher entropy **with its structural floor and ceiling drawn
in**), `04_view_geometry`, `05_throughput`.

### 1.8 The downstream protocol is crop-level, and says so

`finetune_v2_crop_level` splits 20 % of the **crops** for test, stratified on
sub-variety, then a stratified train/validation pair; selection on validation,
report on test. `eval_pretrain_v2` uses the same protocol for the headline.

**What that number measures, stated once.** The 9,357 crops come from 81 source
photographs — a mean of 115 per photograph, same tray, same lighting, often
overlapping boxes. Measured on the identical encoder and probe
(`split_protocol_delta.csv`): the crop-level probe scores **+18.65 pp** above the
photograph-disjoint one (0.8365 vs 0.6500), k-NN **+17.92 pp**, and retrieval P@1
falls 0.664 → 0.475 once same-photograph neighbours are excluded. 89–98 % of the
crops in a photograph have a neighbour above cosine 0.95 at 32 × 32 grey.

So a crop-level number answers *"how well does this classify a new crop from a
known acquisition session"* and not *"…from a new session"*. Every run reports
`shared_source_groups`, `leaked_test_fraction` and `classes_present_in_test`
alongside, and `finetune_v2_grouped_diagnostic` is the same configuration under
photograph-disjoint `grouped_cv` so the gap is measured **on this encoder**
rather than quoted from the previous one.

---

## 2. What was deliberately not changed

| | why |
| --- | --- |
| `centering: sinkhorn`, `out_dim: 2048` | Real hypotheses (C3), each with its own arm. Folding them into the primary recipe would make the run uninterpretable — and both move the teacher's entropy floor `log(K/B_teacher)`, so an arm that changes either is not loss-comparable with any other. |
| `feature_stage: final` | +3.25 pp is on the table at `layers.2`, but it changes what is *read* rather than what is *learned* and would confound the view redesign. `V2-STAGE3` runs last. |
| No auxiliary head (C4) | Confounds a simultaneous readout-stage decision, by its own specification. |
| `conf/model/backbone/swinv2.yaml` still selects Small | The trunk choice belongs to the experiment, and every published number was produced with Small. Changing the group default would silently re-baseline the v1 configs. |
| Layer-wise LR decay | `STAGE1_CHANGES.md` C7's own reasoning argues it is directionally wrong here: CKA at `layers.0`/`layers.1` is 0.976/0.960, so the early layers did not drift and lowering *their* rate attacks nothing. |
| iBOT / masked patches | A global view carries a median ~2,000 real pixels rendered into 65,536; at the `layers.0` grid each token covers ~0.5 native pixels. A masked-patch objective here reconstructs the bicubic kernel. |
| Cropping the 18 unused photographs | +22 % scenes at zero acquisition cost and the single highest-value change available — but the cropping script is not in this repository, and it re-baselines every published number. `scripts/report_raw_photographs.py` and the corpus fingerprint are what make it safe when it happens. |

---

## 3. The arms, and what would falsify the design

`conf/stage1_arms/view_design.yaml`, 25 epochs each,
`python scripts/run_stage1_ablations.py --arms conf/stage1_arms/view_design.yaml`.

| arm | isolates | reading |
| --- | --- | --- |
| **V2-FROZEN** | nothing — no training | the bar. If no arm clears it by more than a fold SD, the objective is not the binding constraint. |
| **V2-FULL** | — | the recipe's own number |
| **V2-noVIEW** | the view redesign | **the** arm: V2-FULL − V2-noVIEW is the entire value of 598 → 1,440 native pixels |
| **V2-noCOLOUR** | the colour policy | judge on the per-class F1 of Jagnath/Poosa33 and AMT-2/AMT-4, not the 27-class mean |
| **V2-noDIHEDRAL** | the lossless rotations | whether narrowing the crops needed the diversity back |
| **V2-KOLEO-BOTTLENECK** | KoLeo's space | whether regularising the shipped space beats regularising a discarded one |
| **V2-FLOOR** | the native-pixel floor | whether the p5 tail was hurting |
| **V2-EMA** | Sinkhorn → EMA centering | watch `prototype_perplexity`: ~2,030 → tens is the known failure |
| **V2-STAGE3** | the readout stage | runs last; changes what is read, not what is learned |

**What would falsify the framing.** If V2-FULL, V2-noVIEW and V2-noCOLOUR all
land within ±1 SD of each other and of V2-FROZEN, then the objective is not the
binding constraint and **81 scenes is the ceiling**. The remaining effort then
belongs in data acquisition (the 18 uncropped photographs; second sessions for
the five single-photograph varieties), in the initialisation, and in the
stage-2 readout — none of which is a stage-1 recipe change.

That outcome is a result, not a failure, and this pipeline is instrumented to
report it: the fold SD is ±0.10 on the 27-way probe, so a single arm cannot
resolve anything below ~2 pp, and `--seeds` is what turns a ranking into a claim.

---

## 4. Running it

```bash
export SEED_DATA_ROOT=/path/to/Hierarchical_SeedData/Cropped_Samples
export SEED_OUTPUT_DIR=/path/to/outputs

# 0 — what the augmentation will actually build, before spending a GPU-hour
python scripts/report_view_geometry.py --compare hierarchical_seeds seed_crops_v2

# 1 — the reference the run must beat (no training)
python -m src.trainers.pretrain_eval experiment=eval_frozen_v2

# 2 — stage 1
python main.py pretrain-v2                    # or --gpus 2
python main.py pretrain-v2 --gpus 2 experiment.training.resume=auto \
    experiment.training.max_runtime_minutes=520      # preemptible platforms

# 3 — evaluate the milestones and confirm the probe picked the right one
python main.py eval-pretrain-v2

# 4 — stage 2, crop-level train/val/test
SEED_PRETRAIN_BACKBONE=$SEED_OUTPUT_DIR/checkpoints/dino_v2_swinv2_tiny.pth \
    python main.py finetune-v2

# the photograph-disjoint counterpart, for the leakage delta on THIS encoder
SEED_PRETRAIN_BACKBONE=$SEED_OUTPUT_DIR/checkpoints/dino_v2_swinv2_tiny.pth \
    python -m src.trainers.moe_finetune experiment=finetune_v2_grouped_diagnostic

# arms
python scripts/run_stage1_ablations.py --arms conf/stage1_arms/view_design.yaml
```

Pre-flight, in order, all of which run without a GPU:

```bash
python -m pytest tests/ -q                                  # full suite
python -m pytest tests/test_stage1_v2.py -q                  # the v2 contracts
python scripts/report_view_geometry.py                       # corpus + view content
python main.py pretrain-v2 data.batch_size=4 data.num_workers=0 \
    experiment.training.effective_batch_size=4 \
    experiment.training.epochs=2 experiment.training.max_batches=3 \
    'experiment.training.save_epochs=[1,2]' \
    experiment.training.probe.every_epochs=1 \
    experiment.training.probe.max_samples=270 \
    model.backbone.pretrained=false device=cpu   # end-to-end, real dataset
```

The exact configuration the primary run uses is
`conf/experiment/pretrain_v2_swinv2_tiny.yaml`, and every value in it carries the
measurement that chose it.
