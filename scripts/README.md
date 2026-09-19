# `scripts/` — utilities outside the Hydra trainers

| File | Purpose |
| --- | --- |
| `dry_run.py` | End-to-end pipeline smoke test on synthetic tensors |
| `launch.py` | Cross-platform launcher: 1 process, or N ranks under `torch.distributed.run` |
| `verify_runtime.py` | Does *this* machine still make the fast paths exact? Capabilities, parity, AMP, DDP |
| `bench_pretrain_step.py` | A/B micro-benchmark of the stage-1 training step (sdpa / compile / batch geometry / rank count), no dataset |
| `diagnose_sdpa_parity.py` | Per-module SwinV2→SDPA parity report (errors at fp32/TF32/fp64, gradients, shapes, guard verdicts) |
| `run_stage1_ablations.py` | A **stage-1** arm suite from a YAML manifest: train each arm, evaluate it, collect one table |
| `report_raw_photographs.py` | Which source photographs exist and were never cropped (reports only; never touches the data) |
| `report_view_geometry.py` | What each DINO view is actually built from, per named view policy |
| `run_ablations.py` | The component-wise ablation variants, five seeds each, optionally one per GPU |
| `run_baselines.py` | Linear-probe, the ViT/SwinV2/ConvNeXt/EfficientNet/ResNet backbone comparison, and the hierarchical-CCE and ImageNet controls |
| `run_experiments_suite.py` | **Both of the above as one resumable job**: skip what finished, retry what did not, stop before a session limit |
| `estimate_suite_cost.py` | Price that suite on *this* machine by measuring a real step, before spending GPU hours |
| `generate_plots.py` | Publication figures + `summary_metrics.csv` |
| `extract_features.py` | Dump frozen-backbone embeddings to an `.npz` |
| `train_distributed.sh` | Launch any stage or suite with server environment defaults, single- or multi-GPU |

The intended order for a full experimental campaign:

```bash
python main.py extract-seeds          # stage 0: build the corpus from RAW_Samples
python main.py validate-seeds         # audit it against the recovered legacy boxes
python scripts/dry_run.py             # verify the pipeline runs at all
python scripts/verify_runtime.py      # verify the fast paths are exact HERE
python scripts/bench_pretrain_step.py --scaling 1,2   # pick the launch geometry
python main.py pretrain --gpus 2      # produce the shared encoder, once
python main.py eval-pretrain          # is that encoder worth finetuning on?
python scripts/estimate_suite_cost.py  # what will stage 2 cost on this machine?
python scripts/run_experiments_suite.py --all --gpus 0,1   # ablations + baselines, resumable
python scripts/generate_plots.py      # collect everything into outputs/reports/
```

The last two lines can also be run as the two separate suites, which is what
`run_experiments_suite.py` delegates to:

```bash
python scripts/run_ablations.py --gpus 0,1   # 17 variants x 5 seeds
python scripts/run_baselines.py --gpus 0,1   # eleven baselines and controls
```

## `launch.py`

```bash
python scripts/launch.py pretrain --gpus 2 experiment.training.resume=auto
```

What it adds over calling `torchrun` yourself:

* **one output directory for the whole job** — Hydra resolves `${now:...}` inside
  each process, so two ranks starting in the same second can land in different
  run directories; this pins `$SEED_RUN_ID` before the processes exist;
* **no process group when there is nothing to distribute** — `--gpus 1` runs the
  module directly, because a one-member group buys nothing and adds a failure
  mode (a stale `MASTER_PORT`);
* **an `OMP_NUM_THREADS` budget split across ranks**, so N ranks do not each
  claim one thread per core;
* **it works on Windows** — `torch.distributed.run` is invoked as a module rather
  than relying on a `torchrun` console script being on `PATH`.

## `verify_runtime.py`

Answers the question the test suite cannot answer from a laptop: *does this
hardware, this driver and this torch build still make the optimisations exact?*

```bash
python scripts/verify_runtime.py --gpus 2
```

1. **Capabilities** — compute capability, bf16, TF32, SDPA backends, Triton.
2. **The portable contracts**, via pytest: SDPA parity at fp64/bf16/fp16, AMP
   dtype selection and fp32 pinning, DDP gradient equality over Gloo, exact
   resume.
3. **SDPA parity on the real trunk**, per module, in fp64 on a copy — the same
   probe the trainer runs at conversion time, printed rather than silently acted
   on. A drifted timm shows up here.
4. **DDP gradient equality over the real backend** (NCCL), which the Gloo tests
   cannot cover — a bad interconnect or mismatched NCCL build produces a *wrong
   reduction* rather than an error.

A failure in 3 or 4 does not mean the run will be wrong — both paths refuse or
fall back rather than proceeding — but it does mean the run will be slow, and
knowing that before committing 100 epochs is the point.

## `dry_run.py`

Exercises every stage on random data, with no dataset, no checkpoint and no
network access:

```
synthetic images -> SwinV2 encoder (z ∈ ℝ³⁸⁴) -> seed-type classifier
  -> Top-2 MoE -> residual fusion -> cross-attention -> ArcFace
  -> combined objective + backward -> metrics -> efficiency -> summary CSV
```

The point is to fail fast. A shape mismatch, a broken toggle, a missing tensor in
`HierarchicalOutput` or a metric that cannot consume the model's output all
surface here in seconds rather than after a data-loading stage and an epoch of
real training.

```bash
python scripts/dry_run.py
python scripts/dry_run.py --backbone swinv2_tiny_window16_256 --epochs 3
python scripts/dry_run.py --device cpu --top-k 4 --no-efficiency
```

The backbone is built with `pretrained=False`, so nothing is downloaded and the
weights are random — the losses are meaningless by design. What is verified is
that the pipeline *runs*, not that it learns.

## `run_ablations.py`

The arms the results table needs. The six below are the ones the submitted
manuscript named — `wo_arcface` renamed to `wo_margin_only`/`wo_angular_head`,
because swapping ArcFace for a plain `Linear` removes the margin *and* the
embedding normalisation *and* the centre normalisation *and* the logit scale, so
that gap was never a margin measurement. `python scripts/run_ablations.py
--help` lists the full set, including the capacity-matched and fixed-router
controls that split the confounded toggles into single factors.

| Variant | Override |
| --- | --- |
| `full_model` | *(none)* |
| `wo_moe` | `model.head.use_moe=false` |
| `wo_margin_only` | `model.head.sub_head_variant=normface` |
| `wo_angular_head` | `model.head.sub_head_variant=linear` |
| `wo_residual` | `model.head.use_residual=false` |
| `wo_kl` | `model.head.use_kl_loss=false` |
| `wo_cross_attn` | `model.head.use_cross_attention=false` |

```bash
python scripts/run_ablations.py
python scripts/run_ablations.py --variants wo_moe wo_kl
python scripts/run_ablations.py --dry-run
python scripts/run_ablations.py -- data.batch_size=8 experiment.training.epochs=20
```

Results land in `outputs/ablations/{variant}/` — one self-contained directory per
variant with its Hydra config snapshot, logs, checkpoints, figures,
`summary.json` and `test_predictions.npz`.

**Pretraining is never repeated.** Every variant reads the single published
encoder at `outputs/checkpoints/dino_pretrained_encoder.pth`. If each variant
had its own self-supervised initialisation, the table would partly measure that
instead of the architectural change under test — and the resulting numbers would
look entirely normal. `--allow-missing-checkpoint` waives the requirement for
smoke runs and prints a warning; results from such a run are not comparable.

## `run_experiments_suite.py`

One command for the whole stage-2 campaign, built to survive a Kaggle or
vast.ai session ending mid-run.

```bash
python scripts/run_experiments_suite.py --all                 # everything, 5 seeds
python scripts/run_experiments_suite.py --all --kaggle        # T4 x2 preset
python scripts/run_experiments_suite.py --all --dry-run       # print the commands, spend nothing
python scripts/run_experiments_suite.py --preset reviewer     # the minimum that answers the review
python scripts/run_experiments_suite.py --variants wo_moe wo_kl --seeds 42
python scripts/run_experiments_suite.py --status              # progress, no side effects
python scripts/run_experiments_suite.py --list                # every arm and preset
```

It **defines nothing**. `ABLATION_VARIANTS` and `BASELINE_VARIANTS` are imported
from the two suites above, so an arm added there appears here automatically and
the three cannot drift apart in what they think the experiment is.

**The artifacts on disk are the truth; `suite_status.json` is a cache.** Before
launching a cell the runner opens that run's `summary.json` and
`test_predictions.npz` and checks they are present, parseable and mutually
consistent. Three things follow, and each is a failure this pipeline actually
has on a preemptible platform:

* a session killed **between** the two writes leaves a complete-looking
  `summary.json` and no predictions. `generate_plots.py` re-scores every row
  from the raw predictions, so that run would silently vanish from the results
  table while the suite reported it done. It is re-run.
* a session killed **during** the `.npz` write leaves a file that exists and
  raises on read. Also re-run.
* a run finished by hand with a bare `python -m src.trainers.moe_finetune` is
  recognised and skipped. Deleting `suite_status.json` loses the attempt counts
  and the timings and nothing that decides what gets launched.

Within a cell, `experiment.training.resume=auto` is passed by default, so a
variant killed at epoch 70 of 100 continues from its own epoch-boundary
checkpoint rather than restarting.

**Two budgets, and they are not the same one.** `--max-runtime-minutes` is
per run and is honoured *inside* the trainer, which checkpoints and exits
cleanly at it. `--max-session-minutes` is the suite's, and stops **between**
runs once the remaining time is less than the mean run so far — deliberately
never starting a variant that cannot finish, because being killed inside one is
what produces the half-written state above. `--kaggle` sets both, plus
`--amp fp16` (a T4 is `sm_75` and has no hardware bf16) and `--gpus 0,1`.

**Provenance is checked, not assumed.** Completed runs are grouped by the corpus
SHA-256 and by the test-split identity recorded in their own `summary.json`, and
a suite spanning two of either prints a warning on every launch. Two corpora in
one table is the `Refined_Samples`/`Cropped_Samples` re-baseline; two splits
makes McNemar's paired test undefined rather than merely weaker. The shared
encoder's SHA-256 is recorded once and compared on every relaunch.

## `estimate_suite_cost.py`

```bash
python scripts/estimate_suite_cost.py                       # default suite, this device
python scripts/estimate_suite_cost.py --preset reviewer --seeds 42
python scripts/estimate_suite_cost.py --arms full_model vit_small --iterations 30
python scripts/estimate_suite_cost.py --gpus 2              # sharded wall clock
```

Measures a real forward + backward + optimiser step and a real evaluation
forward, once per distinct *model shape* in the suite, then derives the per-run
and whole-suite wall clock. Measured and derived figures are labelled apart, as
`src/utils/training/budget.py` does for stage 1.

Measured on an **Apple M5 (10-core, 16 GB), PyTorch MPS, fp32**, batch 16, at the
configured 100 epochs over 13,492 crops — the whole default suite is
**28 arms × 5 seeds ≈ 1,042 h**, so a laptop is not the machine for it:

| arm | ms/step | per run [EST] |
| --- | --- | --- |
| `linear_probe` | 264 | 4 h 58 m |
| `hierarchical_cce` | 280 | 5 h 13 m |
| `wo_moe` | 283 | 5 h 15 m |
| `vit_small` | 360 | 5 h 44 m |
| `swin_tiny` | 359 | 5 h 49 m |
| `full_model` | 439 | 7 h 36 m |
| `imagenet_frozen` | 464 | 7 h 59 m |
| `convnext_tiny` | 506 | 8 h 04 m |
| `swinv2_tiny` | 872 | 14 h 05 m |
| `swinv2_supervised` | 993 | 15 h 55 m |

The shape of that table is the useful part and it holds across devices: the two
arms that **unfreeze the SwinV2 trunk** cost roughly 2× everything else, and
`wo_moe` and `linear_probe` are the cheapest because they carry the least head.
The absolute numbers are this machine's; re-measure on the machine that will run
the suite — that is the entire point of the script, and it costs minutes.

**MLX and CoreML cannot run this suite, and the script says so rather than
pretending to model them.** MLX is a separate array framework with its own
modules and optimisers — this repository is PyTorch end to end (timm,
`torch.compile`, DDP, `GradScaler`, SDPA), so "run it on MLX" means
reimplementing stage 2, at which point the numbers stop being about this code.
CoreML is an inference runtime with no backward pass at all; it could serve the
latency row of the efficiency table for a finished checkpoint and cannot produce
a single ablation row. The Apple-silicon path that *does* work is PyTorch's
**MPS** backend, which `select_device` already resolves and this script
measures. AMP is CUDA-gated throughout, so on MPS and CPU the suite runs fp32
and `--amp fp16` is inert.

## `run_stage1_ablations.py`

```bash
python scripts/run_stage1_ablations.py --arms conf/stage1_arms/screens.yaml
python scripts/run_stage1_ablations.py --arms conf/stage1_arms/view_design.yaml
python scripts/run_stage1_ablations.py --arms conf/stage1_arms/view_design.yaml --dry-run
python scripts/run_stage1_ablations.py --arms conf/stage1_arms/view_design.yaml     --experiment pretrain_swinv2_tiny_dino
python scripts/run_stage1_ablations.py --arms conf/stage1_arms/view_design.yaml --seeds 42 43 44
python scripts/run_stage1_ablations.py --arms conf/stage1_arms/view_design.yaml --collect-only
```

**Why `run_ablations.py` could not be reused.** That script runs stage-2 head
variants: one trainer, one output tree, one shared encoder every variant *reads*.
A stage-1 suite is the opposite shape — each arm **produces** an encoder and then
needs a second process to evaluate it, and both write to paths the other arms
would otherwise overwrite. Without per-arm `experiment.training.save_path`,
`shared_backbone_path` and `experiment.evaluation.save_path`, four arms silently
overwrite each other's `outputs/eval_pretrain/` and each other's
`outputs/checkpoints/dino_pretrained_encoder.pth`, and the resulting table
compares one encoder against itself.

**The arms are data, not code.** A manifest under `conf/stage1_arms/` names a base
Hydra experiment, overrides applied to every arm, and one entry per arm. Adding an
arm is adding an entry.

| Manifest | What it runs |
| --- | --- |
| `screens.yaml` | **No training at all.** The readout screen, the initialisation screen, and the frozen reference. Run it first: it decides the trunk and the readout stage, and neither is recoverable by more self-distillation |
| `view_design.yaml` | The primary recipe decomposed into single factors — `full`, `wo_view_redesign`, `wo_colour_policy`, `wo_dihedral`, `koleo_bottleneck` — plus the knobs it deliberately left fixed (`native_pixel_floor`, `ema_centering`, `stage3_readout`) and the `frozen` reference |

Three arm shapes: `train: true` (train then evaluate), `train: false` with
`evaluate_frozen: true` (the reference — the chosen trunk, frozen, no in-domain
training), and `train: false` with an explicit `eval_experiment` (a screen).

**A suite on a different trunk must name its own evaluation.** The manifest's
`evaluation:` and `frozen_evaluation:` keys select which evaluation experiment
scores every arm, because the evaluation config carries the *trunk* and the
*protocol*: scoring a SwinV2-Tiny arm against an evaluation whose `imagenet_init`
control is SwinV2-Small turns "what did self-distillation add" into an
architecture delta. `view_design.yaml` sets them to `eval_pretrain` /
`eval_frozen_reference`; a per-arm `eval_experiment` still wins over both.

**Only `data.*` overrides carry into the evaluation.** They describe the corpus
and the augmentation, both of which the alignment measurement must reproduce;
`experiment.training.*` has no meaning in an evaluation config and Hydra's struct
mode rejects it.

**Multiple GPUs are not sharded across arms.** A stage-1 arm is a full
self-distillation run and saturates one device, so two concurrent arms halve each
other's throughput. Use both devices *within* an arm instead
(`python main.py pretrain --gpus 2` with `effective_batch_size` pinned).

The collected table headlines `oof_probe_sub_accuracy_testable_classes`, and
carries `final_teacher_student_kl` (the **learnable** half of the objective — the
raw DINO loss is ~95 % target entropy and is not comparable across arms that move
the centering) and `nuisance_photo_above_chance` (an arm that *raises* it may have
won the probe by re-learning the photograph confound the protocol punishes).

## `report_view_geometry.py`

```bash
python scripts/report_view_geometry.py
python scripts/report_view_geometry.py --policy canonical reference
python scripts/report_view_geometry.py --csv outputs/reports/geometry.csv \
    data.augmentation.min_native_pixels=900
```

`--policy` names an override set on the one `data` group rather than a second
config file: `canonical` is whatever `conf/data/hierarchical_seeds.yaml` currently
says, `reference` is DINO's ImageNet geometry as a *measurement* baseline.

**What each DINO view is actually built from, before spending a GPU-hour on it.**
`RandomResizedCrop`'s `scale` is a fraction of the *source* area, so a config
does not say how much of a seed a view contains — only the product of the scale
range and the source-size distribution does, and nothing was measuring it. On
this corpus the submitted recipe builds each local view from a median **598
native pixels** rendered into 65,536, with 80 % of Eq. 1's cross-view terms
anchored on one.

It reads the real file headers (~2 s for 9,357 files, no decode) and drives
torchvision's own `RandomResizedCrop.get_params`, so the numbers are what the
dataloader will produce rather than a model of it. That matters for one result in
particular: `get_params` retries ten times and then returns a **deterministic
centre crop**, and on a corpus that is 96.6 % non-square, raising the scale floor
pushes it into that fallback — 3.5 % at `scale=(0.40, 1.00)` against **22.0 %** at
`(0.70, 1.00)`. The rate is reported per view family and is why `crop_ratio` is a
config key.

Reads nothing but the dataset and the config; never touches a checkpoint. The
trainer runs the same measurement at startup into `csv/view_geometry.csv`.

## `plot_stage1_run.py`

```bash
python scripts/plot_stage1_run.py outputs/hydra/2026-08-19/12-08-26
python scripts/plot_stage1_run.py <run dir> --output figures/ --dpi 600
```

Regenerates a stage-1 run's five publication figures from `<run dir>/csv/*.csv`
and nothing else — no checkpoint, no GPU, no event parser, no W&B. The trainer
already writes them at the end of a run (including an interrupted one); this
exists for rebuilding them after editing a figure, on a laptop, or from a run
directory copied off a server.

The CSV-only constraint is what makes a figure and its table the same numbers by
construction, which is why the trainer generates them the same way rather than
from the values it still holds in memory.

## `report_raw_photographs.py`

```bash
python scripts/report_raw_photographs.py
python scripts/report_raw_photographs.py --raw ../Dataset/Hierarchical_SeedData/RAW_Samples
```

Set-difference between the `RAW_Samples` stems and the `_bbox` prefixes under
`Cropped_Samples`. **Reports only; it never touches the dataset.**

The binding constraint here is the number of *scenes*, not crops: 9,357 crops come
from 81 photographs, and within one photograph 89–98 % of crops have a neighbour
above cosine 0.95 at 32×32 grey. `RAW_Samples` holds **99**, so 18 exist uncropped
— +22 % scenes at zero acquisition cost, concentrated on classes that currently
have two or three. It does *not* fix the five single-photograph sub-varieties;
that needs a camera.

Two things before acting on it: look at the photographs (they may have been
excluded for blur or exposure, and an out-of-focus frame in the SSL corpus is
worse than nothing), and treat re-cropping as a **re-baseline** — every published
accuracy moves, so do it before a phase, not between arms. The corpus SHA-256 in
each run's `summary.json` keeps the before and after distinguishable.

## `run_baselines.py`

```bash
python scripts/run_baselines.py
python scripts/run_baselines.py --models resnet50 swin_tiny
```

`resnet50`, `swin_tiny` and `swinv2_supervised` own ImageNet backbones and are
trained end to end, so they are deliberately **not** given the self-supervised
checkpoint — a SwinV2 state dict would at best be ignored and at worst partially
loaded. `linear_probe` and `hierarchical_cce` do read it: `linear_probe` is
stage 1's frozen encoder plus two linear heads, and `hierarchical_cce` is the
proposed model with toggles flipped, so both must start from the same encoder
to stay comparable.

## Why each variant runs as a subprocess

Both runners shell out rather than looping in-process, for three reasons in order
of importance:

1. Hydra can only be initialised once per process. Six variants in one process
   would need `GlobalHydra.instance().clear()` between them, which leaves stale
   config state behind — exactly the sort of thing that makes an ablation table
   quietly wrong.
2. GPU memory is released completely when a process exits.
3. A crash in one variant cannot take the suite down; the runner records the
   failure and continues. Pass `--stop-on-failure` for the opposite behaviour.

## `generate_plots.py`

Reads what the training runs already wrote — `summary.json` and
`test_predictions.npz` — and produces, into `outputs/reports/`:

* `summary_metrics.csv`, one row per variant
* row-normalised confusion matrices for both hierarchy levels, with all 27
  sub-varieties labelled unabbreviated on both axes
* t-SNE scatter plots with class names overlaid on the clusters and a legend
* side-by-side training/validation loss curves
* per-class metric heatmaps, misclassification rates, expert utilisation

```bash
python scripts/generate_plots.py
python scripts/generate_plots.py --dpi 600 --no-figures
python scripts/generate_plots.py --roots outputs/ablations outputs/baselines
```

Nothing here retrains or reloads a model, so re-plotting at a different DPI or
normalisation costs seconds. It **re-scores** from the raw predictions rather
than trusting `summary.json`, so the table and the figures are computed by the
same code at plot time.

## `extract_features.py`

Runs the DINO-pretrained backbone over the whole dataset and writes features,
both label levels, and file paths to a compressed `.npz`. It bypasses the
trainers entirely, so it is the fastest way to sanity-check a pretrained
checkpoint — cluster the embeddings or run a linear probe without launching a
training run.

```bash
python scripts/extract_features.py \
  --data-root  $SEED_DATA_ROOT \
  --checkpoint $SEED_PRETRAIN_BACKBONE \
  --output     $SEED_OUTPUT_DIR/features/seed_features.npz \
  --max-samples 2000
```

Loading the result:

```python
import numpy as np
data = np.load("seed_features.npz", allow_pickle=True)
data["features"]           # [n, backbone_dim]
data["seed_labels"]        # [n]  0..3
data["subvariety_labels"]  # [n]  0..26
data["paths"]              # [n]  source image paths
```

These are the **backbone's** features at its native width (768 for SwinV2
Tiny/Small), *not* the paper's `z ∈ ℝ³⁸⁴`. The projection to `z` is a trained
layer belonging to `DinoV2SwinV2Encoder`, and a freshly-initialised copy of it
would project through random weights — worse than useless for inspecting what
pretraining learned. For real 384-D embeddings, read the `embeddings` array from a
finished run's `test_predictions.npz`.

**For anything reportable, use `python main.py eval-pretrain` instead.** It caches
the same features and adds what a claim about the encoder needs: the
photograph-disjoint protocol, the ImageNet/random/milestone controls, provenance
digests, and the metrics that detect collapse. This script remains the fastest way
to get a bare `.npz` for ad-hoc inspection without Hydra. See
[`../architecture/08_STAGE1_REPRESENTATION_EVALUATION.md`](../architecture/08_STAGE1_REPRESENTATION_EVALUATION.md).

Defaults for `--data-root`, `--checkpoint` and `--output` come from
`SEED_DATA_ROOT`, `SEED_PRETRAIN_BACKBONE` and `SEED_OUTPUT_DIR`.

## `bench_pretrain_step.py`

Reproduces one micro-batch of the stage-1 loop on synthetic data — the same
pinned `uint8` view-major layout, `ViewBatcher`, fused teacher and student
forwards, DINO loss with Sinkhorn and KoLeo, backward, and on accumulation
boundaries the clip, fused AdamW and foreach EMA. Under `torchrun` it also
reproduces the DDP wrapper and the `no_sync` pattern, so the measured gradient
traffic is the traffic a real run pays.

Reports milliseconds per micro-batch, images/s and views/s (**job-wide**, so the
DDP number is directly comparable to the single-GPU one), peak VRAM per rank, and
mean/peak SM utilisation where NVML is available.

```bash
python scripts/bench_pretrain_step.py                    # the configured geometry
python scripts/bench_pretrain_step.py --no-sdpa          # what the rewrite buys
python scripts/bench_pretrain_step.py --batch-size 64 --accum 1
python scripts/bench_pretrain_step.py --scaling 1,2      # single-GPU vs DDP table
```

`--scaling` re-invokes the script at each rank count in a **fresh process
group** — a cached allocator, a warmed autotuner and an already-negotiated NCCL
communicator would all leak from one measurement into the next — and prints the
comparison with a speedup column.

Note what `--scaling` holds fixed: the *per-rank* micro-batch, so the effective
batch grows with the rank count. That is what a scaling measurement means, and it
is **not** what a real run should do — there, set
`experiment.training.effective_batch_size` and let the accumulation be derived.

## `train_distributed.sh`

Sets the server path defaults, creates the output directory, exports
`SEED_PRETRAIN_BACKBONE`, and dispatches to the right trainer or suite —
single-process by default, multi-GPU when `GPUS` is set.

```bash
scripts/train_distributed.sh pretrain
GPUS=2 scripts/train_distributed.sh pretrain          # DDP over 2 GPUs
GPUS=auto scripts/train_distributed.sh pretrain       # every visible GPU
scripts/train_distributed.sh finetune data.batch_size=32
GPUS=0,1 scripts/train_distributed.sh ablations       # one variant per GPU
scripts/train_distributed.sh verify
scripts/train_distributed.sh report
```

`GPUS` means two different things, deliberately, because the two stages want
different kinds of parallelism. For `pretrain` / `finetune` / `ablation` it is a
**count** and the stage runs as one DDP job. For `ablations` / `baselines` it is
a **device list** and the suite runs one variant per device concurrently — which
is the better use of a second GPU for stage 2, since 18 variants × 5 seeds are
already independent processes with no gradient traffic and each keeps the exact
numerics of a single-GPU run.

Exporting `SEED_PRETRAIN_BACKBONE` up front is what makes every stage resolve to
the same encoder file without any manual path plumbing. Extra arguments pass
through as Hydra overrides. On Windows, use `scripts/launch.py` directly — it is
the same launcher and takes the same arguments.
