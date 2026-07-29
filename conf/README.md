# `conf/` — Hydra configuration

```
conf/
  config.yaml                        # root: composes every group below
  data/hierarchical_seeds.yaml       # dataset paths, sizes, augmentation
  model/
    backbone/  swinv2.yaml           # the ONLY DINOv2 backbone (see below)
    head/      hierarchical_moe.yaml # MoE + cross-attention + ArcFace (stage 2)
               dino.yaml             # DINO projection head (stage 1)
               flat_supervised.yaml  # supervised baseline: backbone + 2 linear heads
    loss/      arcface_kl.yaml       # combined hierarchical objective
               dino.yaml             # DINO self-distillation
               flat_cce.yaml         # plain CCE at both hierarchy levels
  experiment/  pretrain_swinv2_dino.yaml
               finetune_hierarchical_moe.yaml
               ablation_flat_classifier.yaml
               baseline_resnet50.yaml
               baseline_swin_tiny.yaml
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

## `model/backbone` has exactly one option

SwinV2 is the only encoder on the DINOv2 path. The comparative ViT-S/14 entry
from the submitted manuscript has been removed, and `validate_swinv2_name()` in
`src/models/builder.py` rejects any name that is not a SwinV2 variant — so a run
cannot silently fall back to a different encoder.
`tests/test_configs.py::test_only_swinv2_is_available_as_a_dinov2_backbone`
asserts the directory contains one file.

Supervised comparison backbones (ResNet-50, Swin-T) are selected through
`model/head: flat_supervised` and the `baseline_*` experiments. They are not
DINOv2-pretrained, so putting them in this group would invite exactly the mix-up
the validator prevents.

## Component toggles

Five booleans under `model.head` drive the ablation suite:

```yaml
use_moe: true              # false -> one dense transformer block, no routing
use_arcface: true          # false -> linear head; objective becomes plain CE
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
| `experiment.training.learning_rate` (pretrain) | 0.0005 | Section 6.1 |
| `model.loss.warmup_teacher_temp` → `teacher_temp` | 0.02 → 0.04 | Table 1 |
| `model.loss.warmup_teacher_temp_epochs` | 5 | Table 1 |
| `model.loss.center_momentum` | 0.9 | Eq. 3 |
| `model.head.out_dim` (DINO) | 65,536 | Table 1 |
| `data.augmentation` crops | 2 global + 4 local | Table 1 |
| crop scales | (0.4, 1.0) / (0.05, 0.4) | Table 1 |
| `model.head.embed_dim` | 384 | Eq. 4 |
| `model.head.num_experts` | 6 | Section 5.2 |
| `model.head.top_k` | **2** | revision (paper: 4) |
| `data.num_seed_types` / `num_sub_varieties` | 4 / 27 | Section 3 |

`top_k` is the one deliberate departure. `model.head.top_k=4` reproduces the
submitted configuration; see [`../REVISION_NOTES.md`](../REVISION_NOTES.md).

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
