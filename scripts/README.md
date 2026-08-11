# `scripts/` — utilities outside the Hydra trainers

| File | Purpose |
| --- | --- |
| `dry_run.py` | End-to-end pipeline smoke test on synthetic tensors |
| `bench_pretrain_step.py` | A/B micro-benchmark of the stage-1 training step (sdpa / compile / batch geometry), no dataset |
| `run_ablations.py` | The six component-wise ablation variants |
| `run_baselines.py` | Linear-probe, SwinV2-supervised, ResNet-50, Swin-T and hierarchical-CCE baselines |
| `generate_plots.py` | Publication figures + `summary_metrics.csv` |
| `extract_features.py` | Dump frozen-backbone embeddings to an `.npz` |
| `train_distributed.sh` | Launch any stage or suite with vast.ai environment defaults |

The intended order for a full experimental campaign:

```bash
python scripts/dry_run.py          # verify the pipeline runs at all
python main.py pretrain            # produce the shared encoder, once
python scripts/run_ablations.py    # 18 variants x 5 seeds
python scripts/run_baselines.py    # five baselines
python scripts/generate_plots.py   # collect everything into outputs/reports/
```

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
python scripts/dry_run.py --backbone swinv2_base_window16_256 --epochs 3
python scripts/dry_run.py --device cpu --top-k 4 --no-efficiency
```

The backbone is built with `pretrained=False`, so nothing is downloaded and the
weights are random — the losses are meaningless by design. What is verified is
that the pipeline *runs*, not that it learns.

## `run_ablations.py`

Six variants, each removing exactly one architectural ingredient:

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
encoder at `outputs/checkpoints/dinov2_swinv2_pretrained.pth`. If each variant
had its own self-supervised initialisation, the table would partly measure that
instead of the architectural change under test — and the resulting numbers would
look entirely normal. `--allow-missing-checkpoint` waives the requirement for
smoke runs and prints a warning; results from such a run are not comparable.

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

These are the **backbone's** features at its native width (1024 for SwinV2-Base),
*not* the paper's `z ∈ ℝ³⁸⁴`. The projection to `z` is a trained layer belonging
to `DinoV2SwinV2Encoder`, and a freshly-initialised copy of it would project
through random weights — worse than useless for inspecting what pretraining
learned. For real 384-D embeddings, read the `embeddings` array from a finished
run's `test_predictions.npz`.

Defaults for `--data-root`, `--checkpoint` and `--output` come from
`SEED_DATA_ROOT`, `SEED_PRETRAIN_BACKBONE` and `SEED_OUTPUT_DIR`.

## `train_distributed.sh`

Sets the vast.ai path defaults, creates the output directory, exports
`SEED_PRETRAIN_BACKBONE`, and dispatches to the right trainer or suite. Despite
the name it is a single-process launcher; distributed training is not
implemented.

```bash
scripts/train_distributed.sh pretrain
scripts/train_distributed.sh finetune data.batch_size=32
scripts/train_distributed.sh ablations
scripts/train_distributed.sh baselines
scripts/train_distributed.sh report
```

Exporting `SEED_PRETRAIN_BACKBONE` up front is what makes every stage resolve to
the same encoder file without any manual path plumbing. Extra arguments pass
through as Hydra overrides.
