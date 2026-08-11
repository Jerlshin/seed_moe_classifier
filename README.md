# seed-moe-classifier

Reference implementation of **"Hierarchical Deep Learning for Fine-Grained Seed
Classification: A Self-Supervised and Mixture-of-Experts Approach"**.

Two stages: DINO-style self-supervised pretraining of a Swin Transformer V2 encoder,
then a hierarchical head that classifies 4 seed types and 27 sub-varieties with
a Mixture-of-Experts, cross-attention refinement, and ArcFace metric learning.

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

> **Revision status.** This tree implements the peer-review revision, which
> differs from the submitted manuscript in two deliberate ways: the router
> activates **2 of 6** experts rather than 4, and **SwinV2 is the only encoder**
> — the comparative ViT-S/14 path has been removed. Both are reversible
> (`model.head.top_k=4` reproduces the submitted routing).
> [`REVISION_NOTES.md`](REVISION_NOTES.md) is the full change record.

## Install

```bash
python -m pip install -e ".[tracking,dev]"
```

`tracking` pulls in wandb / tensorboard / pynvml; `dev` pulls in pytest.

## Run

### Single stages

```bash
python main.py pretrain      # stage 1: DINO self-supervised pretraining
python main.py finetune      # stage 2: hierarchical MoE finetuning
python main.py ablation      # flat-classifier ablation
python main.py smoke         # 2-batch dry run of both stages

python main.py pretrain --gpus 2    # the same, as a 2-rank DDP job
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
python scripts/run_ablations.py   # six component-wise variants
python scripts/run_baselines.py   # linear probe, SwinV2-supervised, ResNet-50, Swin-T, hierarchical CCE
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
| `swinv2_supervised` | the self-supervised stage (ImageNet SwinV2-Base instead) | `experiment=baseline_swinv2_supervised` |
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
    dino_pretrained_backbone.pth
    dino_pretrained_final.pth
  finetune_hierarchical_moe/
    best_hierarchical_moe.pth
    hierarchical_moe_final.pth
    split_manifest.npz             # split indices + class mappings
    summary.json                   # scalar metrics, efficiency, loss history
    test_predictions.npz           # raw held-out predictions + 384-D embeddings
    hydra/                         # logs, config snapshot, tensorboard, wandb, figures
  ablations/{full_model,wo_moe,wo_arcface,wo_residual,wo_kl,wo_cross_attn}/
  baselines/{linear_probe,swinv2_supervised,resnet50,swin_tiny,hierarchical_cce}/
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
no teacher weights. Turning these on for a 300-epoch run is how the disk fills.

## Testing

```bash
python -m pytest tests/ -q          # 419 tests, no network access, ~30s
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
