# seed-moe-classifier

Reference implementation of **"Hierarchical Deep Learning for Fine-Grained Seed
Classification: A Self-Supervised and Mixture-of-Experts Approach"**.

Two stages: DINO-style self-supervised pretraining of a Swin Transformer V2
encoder, then a hierarchical head that classifies 4 seed types and 27
sub-varieties with a Mixture-of-Experts, cross-attention refinement, and ArcFace
metric learning.

```text
Stage 1   ImageNet-1k → SwinV2-Small → DINO self-distillation (trunk unfrozen)
Stage 2   the resulting encoder (frozen) → hierarchical MoE head
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

> **Revision status.** This tree implements the peer-review revision. The
> headline departures from the submitted manuscript: the router activates
> **2 of 6** experts rather than 4; **SwinV2 is the only encoder** (the
> comparative ViT-S/14 path has been removed); and stage 1 now runs
> **SwinV2-Small from ImageNet-1k for 100 epochs** rather than SwinV2-Base from
> random initialisation for 300. Almost all of it is reversible by override —
> `model.head.top_k=4` restores the submitted routing,
> `experiment=pretrain_swinv2_base_dino` the submitted trunk.
> [`REVISION_NOTES.md`](REVISION_NOTES.md) is the full change record and
> [`architecture/02_BACKBONE_AND_SSL.md`](architecture/02_BACKBONE_AND_SSL.md)
> the stage-1 detail.

## Install

```bash
python -m pip install -e ".[tracking,dev]"
```

`tracking` pulls in wandb / tensorboard / pynvml; `dev` pulls in pytest.

## Run

### Single stages

```bash
python main.py pretrain       # stage 1: DINO self-supervised pretraining
python main.py eval-pretrain  # score the stage-1 representation before stage 2
python main.py finetune       # stage 2: hierarchical MoE finetuning
python main.py ablation       # flat-classifier ablation
python main.py smoke          # 2-batch dry run of both stages

python main.py pretrain --gpus 2    # the same, as a 2-rank DDP job
```

Stage-1 variants, all config-driven:

```bash
# the primary run: SwinV2-Small, ImageNet init, unfrozen, 2 global + 4 local
# crops, physical batch 32 at accumulation 1, 100 epochs, encoders kept at
# 25 / 50 / 100
python main.py pretrain

# capacity control: the identical recipe on SwinV2-Base
python -m src.trainers.contrastive_pretrain experiment=pretrain_swinv2_base_dino

# the stage-1 control (no stage 1 at all): ImageNet, frozen, straight to stage 2
python -m src.trainers.moe_finetune experiment=control_imagenet_frozen
```

### Evaluating stage 1 before committing to stage 2

Stage 1 produces an encoder, not a classifier, so its loss curve cannot say
whether the run was worth its 13.5 h. `eval-pretrain` answers that from the
representation itself, against the controls the repository's design already
supports:

```bash
python main.py eval-pretrain                    # the full report, ~35 min on one GPU/MPS
python main.py eval-pretrain experiment.evaluation.max_samples=270   # 1-min plumbing check
```

It runs a linear probe and a parameter-free weighted k-NN over frozen features on
the same **photograph-disjoint** split stage 2 uses, plus label-free geometry
(RankMe, participation ratio, alignment/uniformity, dead channels), unsupervised
structure (k-means and DINO's own 2,048-way prototype argmax scored against the
taxonomy), a low-shot curve, layer-wise probes, calibration, retrieval, and the
encoder's inference cost — for the epoch-100 encoder *and* for its epoch-25/50
milestones, the ImageNet-1k initialisation it started from, and an untrained
trunk. It also recovers the stage-1 collapse diagnostics from the finished run's
`events.jsonl`.

Everything lands in `outputs/eval_pretrain/` (`summary.json`, `metrics.json`,
`tables/*.csv`, 22 figures at 300 dpi, cached features, `provenance.json` with
checkpoint SHA-256s). Results and the parameter/epoch recommendations that follow
from them: [`STAGE1_EVALUATION.md`](STAGE1_EVALUATION.md). How it works and why
these measurements:
[`architecture/08_STAGE1_REPRESENTATION_EVALUATION.md`](architecture/08_STAGE1_REPRESENTATION_EVALUATION.md).

Re-running after changing only an analysis or a figure is cheap: features are
cached per encoder and reused when their checkpoint digest still matches.

Measure before changing the batch — physical batch is what Sinkhorn and KoLeo
estimate from, and accumulation cannot substitute for it:

```bash
python scripts/bench_pretrain_step.py --find-batch-size 16,24,32,48,64
```

Anything after the stage name is forwarded as a Hydra override:

```bash
python main.py finetune data.batch_size=8 experiment.training.epochs=50
python main.py finetune model.head.top_k=4                 # submitted routing
python main.py finetune model.loss.lambda_kl=0.5 experiment.training.num_folds=5
python main.py finetune model.backbone.freeze=false        # fine-tune end to end
```

### Experiment suites

```bash
python scripts/dry_run.py         # synthetic end-to-end smoke test, no dataset needed
python scripts/verify_runtime.py  # are the fast paths exact on THIS machine?
python main.py pretrain           # produces the shared encoder, run once
python main.py eval-pretrain      # is that encoder worth finetuning on?
python scripts/run_ablations.py   # six component-wise variants
python scripts/run_baselines.py   # linear probe, ImageNet frozen/unfrozen, ResNet-50, Swin-T, hierarchical CCE
python scripts/generate_plots.py  # figures + outputs/reports/summary_metrics.csv
```

With more than one GPU, prefer sharding the *suite* over the devices rather than
running one variant across them: 18 variants x 5 seeds are already independent
processes, so there is no gradient traffic and each variant keeps the exact
numerics of a single-GPU run.

```bash
python scripts/run_ablations.py --gpus 0,1
python scripts/run_baselines.py --gpus 0,1
```

`run_ablations.py` and `run_baselines.py` launch one subprocess per variant, all
reading the **same** published encoder at
`outputs/checkpoints/dinov2_swinv2_pretrained.pth`. Pretraining is never
repeated per variant — if each variant had its own self-supervised
initialisation, the resulting table would partly measure that rather than the
architectural change under test.

Useful flags: `--dry-run` prints the commands without running them,
`--variants`/`--models` selects a subset, and everything after a bare `--` is
forwarded to every run as a Hydra override.

## Stage-1 recipe

The table below is the configured recipe. The **executed** 100-epoch run departed
from it in three ways, all recorded in
[`STAGE1_EVALUATION.md`](STAGE1_EVALUATION.md) §1: physical batch 64 rather than
32 (hence lr 1.25e-4 rather than 6.25e-5), and `data.num_workers=0`, which left
the GPU idle for 91.6 % of the run.

| | Value | Note |
| --- | --- | --- |
| Backbone | `swinv2_small_window16_256` | 48.96 M params, 25.6 GFLOPs/view @256 — both measured. Tiny (27.58 M, 13.32 GFLOPs) remains a one-token override |
| Initialisation | ImageNet-1k (`ms_in1k`) | the trunk then **trains**; `build_dino` refuses `freeze=true` |
| Stochastic depth | 0.1, student only | the teacher copy is silenced — its outputs are the targets |
| Views | 2 global @256 + 4 local @101 | local crops kept deliberately; whether they earn their cost is an ablation to run |
| Physical batch | 32, accumulation 1 | Sinkhorn/KoLeo are per-micro-batch, so accumulation is not a substitute |
| DINO head | 768 → 1024 → 1024 → 256 → 2048 | discarded after stage 1; nothing downstream depends on it |
| Prototypes | 2,048 | `K/B_teacher = 32` prototypes per teacher view, against 128 at 8,192 |
| Epochs | 100, encoders kept at 25 / 50 / 100 | so "was 100 necessary?" is a suite away, not an assumption |
| Learning rate | derived: `0.0005 × B_eff/256` = **6.25e-05** | Section 6.1's 0.0005 is the rate at batch 256 |
| Warmup | 10 epochs, linear, then cosine | one `SequentialLR`, resumable mid-warmup |
| Weight decay | 0.04 → 0.4 cosine, **matrices only** | biases, norms, `logit_scale` and `cpb_mlp` excluded |
| Teacher | momentum 0.996 → 1.0; τ 0.04 → 0.07 over 30 epochs | unchanged |
| Clip / freeze last layer | 3.0 / 1 epoch | unchanged (Table 1, Section 6.1) |

Every run prints a copy-pasteable compute/parameter budget and writes it to
`events.jsonl` and W&B as `budget/*`, with measured and estimated quantities
labelled apart. The teacher entropy diagnostics ship with their own bounds
(`teacher_entropy_min/_max`, the normalised form, `K`, `B_teacher`,
`prototype_utilization`), because entropy read against zero says nothing —
Sinkhorn's structural floor here is 3.47 of a 7.62 maximum.

## Ablation and baseline matrix

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

A disabled component is **not allocated**, so an ablation's parameter count
describes the model actually trained. `wo_moe` keeps one dense block of
identical architecture rather than deleting the layer, so its gap against the
full model measures *routing* and not a missing block's capacity.

### Environment variables

All paths resolve through `oc.env`, so these are the real knobs on a rented GPU box:

| Variable | Meaning | Default |
| --- | --- | --- |
| `SEED_DATA_ROOT` | Dataset root | `data/Hierarchical_SeedData/Cropped_Samples` |
| `SEED_OUTPUT_DIR` | Root for run dirs, checkpoints, metadata | `outputs` |
| `SEED_PRETRAIN_BACKBONE` | Encoder every downstream run loads | `$SEED_OUTPUT_DIR/checkpoints/dinov2_swinv2_pretrained.pth` |
| `SEED_RUN_ID` | Shared Hydra run-directory suffix; the launcher pins it so every rank agrees | timestamp |

On a rented server:

```bash
export SEED_DATA_ROOT=/workspace/data/Hierarchical_SeedData/Cropped_Samples
export SEED_OUTPUT_DIR=/workspace/outputs
GPUS=2   scripts/train_distributed.sh pretrain      # DDP over 2 GPUs
GPUS=0,1 scripts/train_distributed.sh ablations     # one variant per GPU
GPUS=0,1 scripts/train_distributed.sh baselines
scripts/train_distributed.sh report
```

On a platform that ends the session before the run does (Kaggle, Colab, a
preemptible instance), add the two flags that tell the run about the limit — the
same command line then works for the first launch and every relaunch:

```bash
python main.py pretrain --gpus 2 \
    experiment.training.resume=auto \
    experiment.training.max_runtime_minutes=520
```

## Data layout

```text
$SEED_DATA_ROOT/
  Rice/
    Chinnar/            image.png ...
    Chithrakar/         ...
  Millet/
    Baryard/            ...
```

Labels come from **sorted** directory names, and sub-variety labels are
**global** (0..26 across all seed types, not per seed type). The
sub-variety → seed-type map that drives the KL hierarchy term is derived from
this tree at runtime, never hardcoded. Adding or renaming a folder shifts every
index and invalidates existing checkpoints — update
`conf/data/hierarchical_seeds.yaml` and retrain.

## Repository layout

| Path | Contents |
| --- | --- |
| [`src/models/`](src/models/README.md) | SwinV2 encoder, hierarchical head, MoE / cross-attention / ArcFace, supervised baselines |
| [`src/losses/`](src/losses/README.md) | DINO, ArcFace, KL hierarchy, MoE regularisation, cosine compactness |
| [`src/datasets/`](src/datasets/README.md) | Hierarchical image-folder dataset and the DINO multi-crop pipeline |
| [`src/trainers/`](src/trainers/README.md) | Hydra training entry points and the suite runner |
| [`src/utils/`](src/utils/README.md) | Metrics, efficiency profiling, reporting, figures, tracking |
| [`conf/`](conf/README.md) | Hydra config groups |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | One-page map of both stages and the interface between them |
| [`architecture/`](architecture/00_OVERVIEW.md) | Per-topic design documents |
| [`STAGE1_EVALUATION.md`](STAGE1_EVALUATION.md) | What the stage-1 encoder learned, measured — and what to change next |
| [`tests/`](tests/README.md) | pytest suite |
| [`scripts/`](scripts/README.md) | Suite runners, plotting, feature extraction, dry run |

## Computational efficiency

`src/utils/efficiency.py` profiles the deployed path — encoder *and* head, since
that is the only combination whose latency a user could observe — and reports:

* **Total vs. active parameters.** Active is total minus the `(E − K)` experts
  that sit out each forward pass. The difference is a closed form, not an
  estimate, because all experts share one architecture. Moving from `K=4` to
  `K=2` doubles the dormant count.
* **FLOPs** via `torch.utils.flop_counter`, which observes real ATen dispatch —
  so a sparsely routed MoE is counted at its true cost automatically.
* **Latency and throughput** at several batch sizes, with warm-up and explicit
  device synchronisation (without which an async CUDA/MPS queue times as ~0).
* **Peak memory** from the allocator, where the backend exposes it.

Each measurement degrades to `None` rather than raising, so profiling never
breaks a training run.

## Experiment tracking

Three sinks, configured under `conf/tracking/default.yaml`:

* **`events.jsonl`** — always on, dependency-free, survives a crashed run.
* **TensorBoard** — on by default. `tensorboard --logdir $SEED_OUTPUT_DIR`
* **Weights & Biases** — on by default in `offline` mode, so a run never blocks
  on credentials. Sync afterwards with
  `wandb sync $SEED_OUTPUT_DIR/**/wandb/offline-run-*`, or set
  `tracking.wandb.mode=online`.

Logged every epoch: loss curves broken down by component, seed-type and
sub-variety accuracy / F1 / precision / recall / AUC, per-class tables, the KL
alignment rate overall and per seed type, MoE expert utilisation, and the
**train-vs-validation loss gap** (`epoch/overfitting_gap`) — positive and growing
means the model is memorising the training split.

Logged every `tracking.intervals.figure_every_epochs`: confusion matrices for
both levels, the sub-variety metric heatmap, per-sub-variety misclassification
rates, expert-utilisation bars, side-by-side train/validation loss curves, and
t-SNE projections coloured by both label levels. Figures are written to
`<run_dir>/figures/` as well as to the trackers.

## Outputs

```text
$SEED_OUTPUT_DIR/
  checkpoints/
    dinov2_swinv2_pretrained.pth   # the shared encoder every downstream run reads
  pretrain_swinv2_dino/
    dino_pretrained_backbone.pth       # the stage-2 handoff
    dino_pretrained_final.pth
    dino_backbone_epoch_{0025,0050,0100}.pth   # milestone encoders, never pruned
    dino_milestone_epoch_{0025,0050,0100}.pth
  eval_pretrain/                   # stage-1 representation evaluation
    summary.json                   # RunSummary, so the cross-run table reads it too
    metrics.json                   # every measurement, nested by encoder
    provenance.json                # checkpoint SHA-256s, git commit, library versions
    split_manifest.npz
    test_predictions.npz           # single-split probe predictions
    out_of_fold_predictions.npz    # the same format over every crop
    tables/*.csv                   # encoder comparison, per-class, low-shot, layer-wise, ...
    figures/fig01..fig22*.png      # publication figures at 300 dpi
    features/{encoder}.npz         # cached frozen features, reused across re-runs
  finetune_hierarchical_moe/
    best_hierarchical_moe.pth
    hierarchical_moe_final.pth
    split_manifest.npz             # split indices + class mappings
    summary.json                   # scalar metrics, efficiency, loss history
    test_predictions.npz           # raw held-out predictions + 384-D embeddings
    hydra/                         # logs, config snapshot, tensorboard, wandb, figures
  ablations/{full_model,wo_moe,wo_arcface,wo_residual,wo_kl,wo_cross_attn}/
  baselines/{linear_probe,swinv2_supervised,resnet50,swin_tiny,hierarchical_cce}/
  controls/imagenet_frozen/         # ImageNet + frozen trunk, the stage-1 control
  reports/
    summary_metrics.csv            # one row per variant, all metrics + cost
    {variant}_confusion_seed_type.png
    {variant}_confusion_sub_variety.png     # all 27 classes, full labels
    {variant}_tsne_{seed_type,sub_variety}.png
    {variant}_loss_curves.png
    ...
  metadata/seed_dataset.csv
```

Every run writes `summary.json` and `test_predictions.npz`. Keeping the *raw*
predictions and embeddings — not only the figures — means a reviewer asking for
a differently-normalised confusion matrix or a re-coloured t-SNE costs a second
of replotting rather than a full retrain. `scripts/generate_plots.py` reads
exactly those two files.

Defaults are tuned for a small (16 GB) rented disk: parameter/gradient
histograms off, `keep_last_n_checkpoints: 1`, no optimizer state,
no teacher weights. Turning these on for a 100-epoch run is how the disk fills.

## Testing

```bash
python -m pytest tests/ -q          # 518 tests, no network access, ~95s
python scripts/dry_run.py           # real encoder, synthetic data, full pipeline
```

The suite asserts the paper's numbers directly — 384-D embedding, 6 experts,
Top-2 routing, τ_t 0.02→0.04 over 5 epochs, momentum 0.996, centering m=0.9,
gradient clip 3.0 — so a failure means the code has drifted from the paper.

## Paper fidelity

* [`PAPER_AUDIT.md`](PAPER_AUDIT.md) — every discrepancy found between the paper
  and the original implementation, what was changed, and the places where the
  paper is internally ambiguous and a documented choice was made.
* [`REVISION_NOTES.md`](REVISION_NOTES.md) — what the peer-review revision
  changed relative to the submitted manuscript, and how to reproduce either.
