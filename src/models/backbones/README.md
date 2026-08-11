# `src/models/backbones/` — self-supervised encoders

Paper Section 4: DINO-style self-distillation with a Swin Transformer V2
backbone. Not DINOv2 -- see `src/losses/dino.py` for exactly which of DINOv2's
four components are and are not implemented.

| File | Contents |
| --- | --- |
| `swinv2_dino.py` | `DINO` (student/teacher pair), `DINOHead` (projection head), `build_dino` |

## Student / teacher

The teacher is a deep copy of the student at initialisation with
`requires_grad=False` throughout. It is never optimised — the trainer advances it
after every step with
`update_momentum(student, teacher, m=0.996)` (Table 1). Only the student's
parameters go into the optimizer; `DINO.student_parameters()` returns exactly
that set.

`forward_student` sees all `2 + local_crops_number` views; `forward_teacher` sees
only the 2 global crops (paper Fig. 7). `forward` returns pooled student features,
so a trained `DINO` can be used directly as a plain encoder.

## `DINOHead`

Section 4: *"an MLP with batch normalization and GELU activation functions. The
final embeddings are normalized to facilitate stable self-distillation."*

```
Linear(768, 1024)  → LayerNorm → GELU
Linear(1024, 1024) → LayerNorm → GELU
Linear(1024, 256)                     # bottleneck
L2 normalize
weight_norm(Linear(256, 65536, bias=False))
```

Two collapse guards, both on that final prototype layer:

* `norm_last_layer=True` freezes its weight-norm gain permanently.
* `freeze_last_layer_epochs=1` cancels its gradients for the first epoch
  (Section 6.1: *"the last layer was frozen for the first epoch to stabilize the
  initial training dynamics"*), via `cancel_last_layer_gradients`, which the
  trainer calls between `backward()` and `step()`.

`_set_weight_norm_gain` handles both PyTorch weight-norm APIs — the legacy
`weight_g` attribute moved under `parametrizations.weight.original0`.

## Two construction-time checks

`DINO.__init__` runs both before building anything:

* `validate_swinv2_name()` rejects any backbone name that is not a SwinV2
  variant. Self-supervised pretraining is standardised on SwinV2, so a stale or mistyped
  name must fail in the first second rather than after an epoch of
  self-distillation against the wrong encoder.
* The backbone's `num_features` must equal the head's `in_dim`. Without this, a
  mismatched `model.backbone.feature_dim` would surface only as a shape error
  deep inside the first forward pass.

## Choosing a SwinV2 variant

`conf/model/backbone/swinv2.yaml` holds `name` and `feature_dim`; everything else
— the DINO head width, the encoder's projection to `z` — interpolates from
`feature_dim`, so switching variants is a two-line config change:

| `name` | `feature_dim` | Params | GFLOPs/view @256 |
| --- | --- | --- | --- |
| `swinv2_tiny_window16_256` (**default**) | 768 | 27.58 M | 13.32 |
| `swinv2_small_window16_256` | 768 | 49.7 M | 25.9 |
| `swinv2_base_window16_256` | 1024 | 86.89 M | 43.94 |

Tiny and Base figures are measured, and `tests/test_stage1_recipe.py`
re-measures them. All three emit the same `8x8` final-stage token grid at
256 px — only the channel width differs — which is what makes the swap invisible
to stage 2's grid routing. `experiment=pretrain_swinv2_base_dino` runs the
identical stage-1 recipe on Base as a capacity control, and sets both `name` and
`feature_dim` together (`DINO.__init__` cross-checks them against the trunk's
actual `num_features` and refuses a mismatch).

`data.image_size` must equal the window resolution in the model name.

Non-SwinV2 architectures are **not** added here. Supervised comparison
backbones (ResNet-50, Swin-T) go through `src/models/baselines.py` and
`conf/experiment/baseline_*.yaml`; they are not self-supervised-pretrained, so putting
them in this group would invite exactly the mix-up the validator prevents.
