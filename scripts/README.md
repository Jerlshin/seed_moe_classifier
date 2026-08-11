# `scripts/` — utilities outside the Hydra trainers

| File | Purpose |
| --- | --- |
| `dry_run.py` | End-to-end pipeline smoke test on synthetic tensors |
| `launch.py` | Cross-platform launcher: 1 process, or N ranks under `torch.distributed.run` |
| `verify_runtime.py` | Does *this* machine still make the fast paths exact? Capabilities, parity, AMP, DDP |
| `bench_pretrain_step.py` | A/B micro-benchmark of the stage-1 training step (sdpa / compile / batch geometry / rank count), no dataset |
| `diagnose_sdpa_parity.py` | Per-module SwinV2→SDPA parity report (errors at fp32/TF32/fp64, gradients, shapes, guard verdicts) |
| `run_ablations.py` | The six component-wise ablation variants, optionally one per GPU |
| `run_baselines.py` | Linear-probe, SwinV2-supervised, ResNet-50, Swin-T and hierarchical-CCE baselines |
| `generate_plots.py` | Publication figures + `summary_metrics.csv` |
| `extract_features.py` | Dump frozen-backbone embeddings to an `.npz` |
| `train_distributed.sh` | Launch any stage or suite with server environment defaults, single- or multi-GPU |

The intended order for a full experimental campaign:

```bash
python scripts/dry_run.py             # verify the pipeline runs at all
python scripts/verify_runtime.py      # verify the fast paths are exact HERE
python scripts/bench_pretrain_step.py --scaling 1,2   # pick the launch geometry
python main.py pretrain --gpus 2      # produce the shared encoder, once
python scripts/run_ablations.py --gpus 0,1   # 18 variants x 5 seeds
python scripts/run_baselines.py --gpus 0,1   # five baselines
python scripts/generate_plots.py      # collect everything into outputs/reports/
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
knowing that before committing 300 epochs is the point.

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
