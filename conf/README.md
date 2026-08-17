# `conf/` — Hydra configuration

```
conf/
  config.yaml                        # root: composes every group below
  data/hierarchical_seeds.yaml       # dataset paths, sizes, augmentation
  model/
    backbone/  swinv2.yaml           # the ONLY self-supervised backbone (see below)
    head/      hierarchical_moe.yaml # MoE + cross-attention + ArcFace (stage 2)
               dino.yaml             # DINO projection head (stage 1)
               flat_supervised.yaml  # supervised baseline: backbone + 2 linear heads
    loss/      arcface_kl.yaml       # combined hierarchical objective
               dino.yaml             # DINO self-distillation
               flat_cce.yaml         # plain CCE at both hierarchy levels
  experiment/  pretrain_swinv2_dino.yaml       # stage 1, primary (Small, ImageNet, unfrozen)
               pretrain_swinv2_base_dino.yaml  # stage 1, capacity control (Base)
               eval_pretrain_representation.yaml  # score the stage-1 encoder, no training
               finetune_hierarchical_moe.yaml
               control_imagenet_frozen.yaml    # ImageNet + frozen trunk, no stage 1
               ablation_flat_classifier.yaml
               baseline_resnet50.yaml
               baseline_swin_tiny.yaml
               baseline_swinv2_supervised.yaml
               baseline_linear_probe.yaml
               baseline_hierarchical_cce.yaml
  tracking/    default.yaml          # W&B + TensorBoard + jsonl
```

## Composition

`config.yaml` composes `data`, `model/backbone`, `model/head`, `model/loss`,
`experiment` and `tracking`. Every group is genuinely wired — overriding one on
the command line changes the run:

```bash
python main.py finetune model.loss.lambda_kl=0.5
python main.py finetune model.head.num_experts=8 model.head.top_k=3
python main.py finetune model.head.use_moe=false        # component ablation
```

> Earlier revisions of this repo duplicated the model settings inline inside each
> experiment file, so `conf/model/*` was dead and editing it had no effect. That
> is fixed; the values now live in one place.

Experiment files are `# @package _global_` and use `defaults` to select the head
and loss their stage needs:

```yaml
# conf/experiment/baseline_resnet50.yaml
# @package _global_
defaults:
  - finetune_hierarchical_moe          # inherit the whole training setup
  - override /model/head: flat_supervised
  - override /model/loss: flat_cce
```

Every ablation and baseline inherits from `finetune_hierarchical_moe`, so
anything changed there changes them all — which is the point: the variants must
differ only in the thing being ablated.

`eval_pretrain_representation.yaml` is the one experiment that inherits from
neither training experiment: it defines `experiment.evaluation` instead of
`experiment.training`, selects the DINO head group (the prototype analysis
reconstructs the 2,048-way head to load `student_head` out of the milestone
checkpoint), and pins `model.backbone.checkpoint_path: null` because it names its
encoders individually — a shared `checkpoint_path` would load the published
stage-2 handoff over whichever milestone is being evaluated.

One thing that group cannot do: set `tracking.*`. `config.yaml` composes
`tracking` **after** `experiment`, so a `tracking:` block inside an experiment
file is silently overwritten by `conf/tracking/default.yaml`. Override it on the
command line instead (`tracking.wandb.enabled=false`).

## `model/backbone` has exactly one option

SwinV2 is the only encoder on the self-supervised path. The comparative ViT-S/14 entry
from the submitted manuscript has been removed, and `validate_swinv2_name()` in
`src/models/builder.py` rejects any name that is not a SwinV2 variant — so a run
cannot silently fall back to a different encoder.
`tests/test_configs.py::test_only_swinv2_is_available_as_a_dinov2_backbone`
asserts the directory contains one file.

Supervised comparison backbones (ResNet-50, Swin-T) are selected through
`model/head: flat_supervised` and the `baseline_*` experiments. They are not
self-supervised-pretrained, so putting them in this group would invite exactly the mix-up
the validator prevents.

## Component toggles

Five booleans under `model.head` drive the ablation suite:

```yaml
use_moe: true              # false -> one dense transformer block, no routing
use_arcface: true          # false -> linear head; prefer sub_head_variant for margin ablations
use_residual: true         # false -> h' = h; Eq. 9 fusion removed
use_cross_attention: true  # false -> h'' = h'; Eqs. 11-12 skipped
use_kl_loss: true          # false -> Eq. 10 not computed
```

`use_kl_loss` lives on the head for consistency — all five switches in one place
— and `conf/model/loss/arcface_kl.yaml` interpolates it with
`use_kl_loss: ${model.head.use_kl_loss}`. One switch, two consumers, no way for
them to disagree.

## What each node feeds

| Node | Consumed by |
| --- | --- |
| `model.backbone` | `build_encoder`, `build_feature_extractor`, `build_dino` |
| `model.head` | `build_hierarchical_moe` / `build_baseline` / `build_dino` |
| `model.loss` | `build_combined_loss` / `CustomDINOLoss` |
| `experiment.training` | The trainer's loop, optimizer and scheduler |
| `experiment.efficiency` | `profile_run` — FLOPs, latency sweep, batch sizes |
| `data` | Dataset construction and the augmentation pipeline |
| `tracking` | `ExperimentTracker` |

Each node is passed whole to its builder, so adding a knob means adding one YAML
key and one constructor argument.

## Interpolations that matter

* `model.head.num_seed_types: ${data.num_seed_types}` and `num_sub_varieties` —
  the hierarchy sizes are declared once.
* `model.loss.use_kl_loss: ${model.head.use_kl_loss}` — the KL toggle reaches
  both the model and the objective from one source.
* Paths resolve through `${oc.env:SEED_DATA_ROOT,...}` and
  `${oc.env:SEED_OUTPUT_DIR,outputs}`, with a nested fallback so setting
  `SEED_OUTPUT_DIR` alone is enough to compose the two stages:

  ```yaml
  checkpoint_path: "${oc.env:SEED_PRETRAIN_BACKBONE,${oc.env:SEED_OUTPUT_DIR,outputs}/checkpoints/dinov2_swinv2_pretrained.pth}"
  ```

`model.head.feature_dim` is now a literal 384 rather than an interpolation of the
backbone width, because `DinoV2SwinV2Encoder` owns the projection to `z` — the
head always receives 384 regardless of which SwinV2 variant is configured.

The trainer cross-checks the discovered class counts against `data.*` and refuses
to start on a mismatch, which would otherwise surface only as unexplained
metrics.

## Paper-critical values

Every value below is asserted by `tests/test_configs.py`, each assertion citing
its source. A failure there means the configs have drifted from the paper.

| Setting | Value | Source |
| --- | --- | --- |
| `data.batch_size` | 16 | Table 1 |
| `experiment.training.epochs` (pretrain) | 300 | Table 1 |
| `experiment.training.clip_grad` | 3.0 | Table 1 |
| `experiment.training.momentum_teacher` | 0.996 | Table 1 |
| `experiment.training.lr_base` (pretrain) | 0.0005, at a reference batch of 256 | Section 6.1 |
| `model.loss.warmup_teacher_temp` → `teacher_temp` | 0.02 → 0.04 | Table 1 |
| `model.loss.warmup_teacher_temp_epochs` | 5 | Table 1 |
| `model.loss.center_momentum` | 0.9 | Eq. 3 |
| `model.head.out_dim` (DINO) | 65,536 (revised to 2,048) | Table 1 |
| `data.augmentation` crops | 2 global + 4 local | Table 1 |
| crop scales | (0.4, 1.0) / (0.05, 0.4) | Table 1 |
| `model.head.embed_dim` | 384 | Eq. 4 |
| `model.head.num_experts` | 6 | Section 5.2 |
| `model.head.top_k` | **2** | revision (paper: 4) |
| `data.num_seed_types` / `num_sub_varieties` | 4 / 27 | Section 3 |

`top_k` is one of several deliberate departures; `REVISION_NOTES.md` 0 tabulates
them all with their reversing override. `model.head.top_k=4` reproduces the
submitted configuration; see [`../REVISION_NOTES.md`](../REVISION_NOTES.md).

## Execution settings vs. objective settings

Everything under `experiment.training` that follows is about *how* a run
executes, never *what* it computes. Three resolve themselves from the hardware,
so one file runs on an A100, on Kaggle's T4x2, on Windows and on CPU:

| Key | `auto` resolves to | Notes |
| --- | --- | --- |
| `amp` | bf16 on Ampere+, **fp16 + `GradScaler` on a T4**, off on CPU/MPS | An explicit `bf16` on `sm_75` is downgraded, not refused. Safe only because Sinkhorn, the prototype log-softmax and KoLeo are pinned to fp32. |
| `compile.enabled` | on where inductor can emit a kernel (CUDA, `sm >= 7.0`, Triton) | The reason for staying eager is logged and written to the event stream. |
| `data.num_workers` | affinity-aware cores per rank, capped at 8 | Per-process, so it must be divided by the local rank count. |

The multi-GPU and resume keys:

| Key | Default | Meaning |
| --- | --- | --- |
| `effective_batch_size` | 64 | **The authority.** `gradient_accumulation_steps` is derived from it and the world size, and a combination that does not divide exactly is refused. 1 GPU at `16 x 4` and 2 at `16 x 2` are the same run. |
| `ddp.gradient_as_bucket_view` | true | Gradients alias DDP's buckets: one gradient's worth of memory back. |
| `ddp.static_graph` | false | True in practice, but a wrong promise surfaces as a hang, not an error. |
| `ddp.find_unused_parameters` | false (stage 1), true (stage 2) | Stage 2 needs it: under sparse dispatch an unrouted expert genuinely has no gradient. |
| `resume` | false | `auto` continues from the newest **valid** checkpoint and starts fresh when there is none, so one command line serves every relaunch. |
| `resume_every_minutes` | 30 | Wall-clock save trigger. `save_interval: 50` **epochs** can exceed a whole Kaggle session. |
| `resume_check_every_steps` | 20 | How often ranks compare notes about saving and stopping. Both triggers are wall-clock and differ per rank. |
| `max_runtime_minutes` | null | Stop cleanly with a checkpoint before a hard session limit. More reliable than being signalled. |
| `model.loss.distributed_sinkhorn` | false | false reproduces the single-GPU numbers exactly; true normalises over the global batch, which is a *different objective*. |

## Run-directory layout

```yaml
hydra:
  run:
    dir: "${oc.env:SEED_OUTPUT_DIR,outputs}/hydra/${now:%Y-%m-%d}/${now:%H-%M-%S}"
```

Each run gets `training.log`, `training.log.jsonl`, `events.jsonl`,
`snapshots/` (the fully resolved config, CLI args, optionally the environment),
`figures/`, `tensorboard/` and `wandb/`.

Suite runs override `hydra.run.dir` to sit inside the variant's own directory, so
one variant is one self-contained folder rather than a scatter across the global
timestamped tree.

`hydra.job.chdir` is left at its default of `false`, so relative paths in configs
resolve against the directory you launched from, not the run directory.
