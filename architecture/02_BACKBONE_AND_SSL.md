# 02 — Backbone and Self-Supervised Pretraining

Covers `src/models/backbones/swinv2_dino.py`, `src/models/builder.py`
(`BackboneFeatureExtractor`, `DinoV2SwinV2Encoder`), `src/losses/dino.py`,
`src/trainers/contrastive_pretrain.py`, and the corresponding `conf/`
entries. Implements paper Section 4.

## 1. SwinV2 is the only DINOv2 backbone

`validate_swinv2_name(model_name)` (`src/models/builder.py:89-104`):

```python
SWINV2_PREFIX = "swinv2"

def validate_swinv2_name(model_name: str) -> str:
    if not str(model_name).startswith(SWINV2_PREFIX):
        raise ValueError(...)
    return model_name
```

Called from **both** `BackboneFeatureExtractor.__init__` and `DINO.__init__`
(`swinv2_dino.py:162`), so a stale or mistyped backbone name fails at
construction — in the first second — rather than after an epoch of
self-distillation against the wrong encoder. This is a deliberate revision:
the submitted manuscript's comparative ViT-S/14 path has been removed
entirely (`REVISION_NOTES.md` §2). Supervised comparison backbones
(ResNet-50, Swin-T) live in `src/models/baselines.py` and are reached only
through `conf/experiment/baseline_*.yaml` — never through `model/backbone`.

`conf/model/backbone/swinv2.yaml` holds the only three variants worth
choosing between:

| `name` | `feature_dim` (native width) |
| --- | --- |
| `swinv2_tiny_window16_256` | 768 |
| `swinv2_small_window16_256` | 768 |
| `swinv2_base_window16_256` (default) | 1024 |

`data.image_size` must equal the window resolution encoded in the model name
(256 for all three above).

## 2. `BackboneFeatureExtractor` — the bare trunk (`builder.py:410-527`)

```python
BackboneFeatureExtractor(model_name, checkpoint_path=None, pretrained=False,
                          dynamic_img_size=True, strict=False, freeze=True)
```

* Wraps `timm.create_model(model_name, num_classes=0, dynamic_img_size=True)`
  and pools its output to `[batch, feature_dim]` via `_pool`
  (`builder.py:502-509`): `mean(dim=(1,2))` for a `[B,H,W,C]` Swin output,
  `mean(dim=1)` for a `[B,L,C]` sequence output.
- `feature_dim` property reports the backbone's **native** width (1024 for
  Base) — this is *not* the paper's `z`; see §3.
- `freeze=True` (default) puts the module in eval mode and disables all
  gradients. `train()` is **overridden** (`builder.py:471-475`) so that a
  frozen backbone can never be pulled back into train mode by an enclosing
  `model.train()` call — this protects BatchNorm running statistics and
  dropout from mutating a trunk that is supposed to be fixed.
- `load_checkpoint(path, strict=False)` returns a
  `{"missing_keys": [...], "unexpected_keys": [...]}` report rather than
  loading silently. `_extract_backbone_state_dict` (`builder.py:511-526`)
  sniffs four checkpoint layouts this project has produced:
  `student_backbone`, `model_state_dict`, `state_dict`, or the
  `nn.Sequential(backbone, head)` `student_model` layout (keeping only keys
  prefixed `"0."`).

## 3. `DinoV2SwinV2Encoder` — the `z ∈ ℝ³⁸⁴` invariant (`builder.py:529-621`)

```python
DinoV2SwinV2Encoder(model_name, embed_dim=384, checkpoint_path=None,
                     pretrained=False, freeze_backbone=True,
                     projection_hidden_dim=None, projection_dropout=0.0)
```

No SwinV2 variant natively emits 384 channels (Tiny/Small: 768, Base: 1024),
so a learned projection is *required* for Eq. 4 (`z ∈ ℝ³⁸⁴`) to hold at all.
This class wraps a `BackboneFeatureExtractor` and adds an
`EmbeddingProjection(in_dim=backbone_dim, out_dim=embed_dim, use_norm=True)`
(`src/models/components/projections.py:34-82`, a single `Linear` +
`LayerNorm` unless `projection_hidden_dim` requests a hidden layer).

```python
def forward(self, images):
    return self.projection(self.encoder(images))   # z, shape [batch, 384]
```

Putting this projection **inside the encoder** (rather than inside the head,
which is where an earlier revision put it — `PAPER_AUDIT.md` §2.4) is what
makes `encoder(images).shape[-1] == 384` an unconditional invariant instead
of a configuration coincidence: the hierarchical head, the baselines, the
t-SNE feature dump, and the efficiency profiler all observe exactly the same
384-D space.

Two consequences that are easy to get wrong:

1. **The projection is trainable even when the SwinV2 trunk is frozen.**
   `trainable_parameters()` (`builder.py:609-611`) is therefore non-empty in
   the default (frozen-backbone) recipe. `build_optimizer([encoder, model],
   cfg)` in `moe_finetune.py:200-235` **must** include the encoder — omitting
   it silently freezes the one layer adapting 1024 backbone channels to the
   head's 384, and the head trains against a random projection with no
   error raised anywhere.
2. **Checkpoints carry `encoder_state_dict` alongside `model_state_dict`**
   (`save_checkpoint`, `moe_finetune.py:666-698`) — a head-only checkpoint
   would be unusable without the trained projection.

`build_encoder(backbone_cfg, embed_dim)` (`builder.py:658-676`) is the config
→ instance constructor used by the finetune trainer.

## 4. `DINO` student/teacher pair (`src/models/backbones/swinv2_dino.py:125-234`)

```python
DINO(backbone_name, input_dim, hidden_dim, bottleneck_dim, out_dim,
     pretrained=False, projection_layers=3, projection_use_batch_norm=True,
     projection_norm_last_layer=True, freeze_last_layer_epochs=1)
```

Construction-time checks (`DINO.__init__`, `swinv2_dino.py:158-175`):

1. `validate_swinv2_name(backbone_name)`.
2. The backbone's `num_features` must equal `input_dim` — otherwise a
   mismatched `model.backbone.feature_dim` would only surface as a shape
   error deep inside the first forward pass.

The **teacher** is a `copy.deepcopy` of the student's backbone and head at
initialization, with `deactivate_requires_grad` applied to both
(`swinv2_dino.py:189-194`) — it never receives a gradient and is advanced only
by EMA. `student_parameters()` returns exactly the student's parameters, i.e.
the only ones the pretrain trainer's optimizer owns.

```python
forward_features(x, teacher=False)   # pooled backbone features, [B, dim] or [B, L, C][:, 0] or mean-pooled
forward_student(x)  = student_head(forward_features(x, teacher=False))
forward_teacher(x)  = teacher_head(forward_features(x, teacher=True))
forward(x)          = forward_features(x)   # so a trained DINO can act as a plain encoder
```

`build_dino(backbone_cfg, head_cfg, freeze_last_layer_epochs)`
(`swinv2_dino.py:220-234`) is the config constructor.

### `DINOHead` (`swinv2_dino.py:44-123`)

Paper Section 4: *"an MLP with batch normalization and GELU activation
functions. The final embeddings are normalized to facilitate stable
self-distillation."*

```text
Linear(in_dim, 2048) → BatchNorm1d → GELU
Linear(2048, 2048)   → BatchNorm1d → GELU        (repeated for num_layers - 2 middle layers)
Linear(2048, 256)                                 # bottleneck
L2-normalize (dim=-1)
weight_norm(Linear(256, 65536, bias=False))       # last_layer
```

`num_layers=3`, `hidden_dim=2048`, `bottleneck_dim=256`, `out_dim=65536` — all
from `conf/model/head/dino.yaml`, matching Table 1's "DINO Output Dimension".

**Two collapse guards, both on the final 65,536-wide layer** — chosen because
that is where a collapsing DINO run collapses first:

* `norm_last_layer=True` calls `_set_weight_norm_gain(1.0, requires_grad=False)`
  (`swinv2_dino.py:99-110`) to **permanently freeze the weight-norm gain** at
  1.0. Handles both PyTorch weight-norm APIs (`weight_g` attribute for the
  legacy `nn.utils.weight_norm`, or `parametrizations.weight.original0` for
  the modern one).
* `cancel_last_layer_gradients(current_epoch)` (`swinv2_dino.py:117-122`)
  zeroes the final layer's gradients for the first
  `freeze_last_layer_epochs` (default 1) epochs — Section 6.1: *"the last
  layer was frozen for the first epoch to stabilize the initial training
  dynamics."* The trainer calls this **between `backward()` and
  `optimizer.step()`**, after gradient clipping — order matters, or the
  "frozen" layer still moves.

## 5. `CustomDINOLoss` — Eqs. 1-3 (`src/losses/dino.py`)

```python
CustomDINOLoss(out_dim, num_crops, warmup_teacher_temp, teacher_temp,
               warmup_teacher_temp_epochs, num_epochs, student_temp=0.1,
               center_momentum=0.9, num_global_crops=2)
```

**Cross-view objective (Eq. 1):**

$$
\mathcal{L}_{\text{DINO}} = -\frac{1}{N}\sum_i \sum_{v \neq q} q_v \cdot \log p_v
$$

Implemented in `compute_dino_loss` (`dino.py:115-144`) as a double loop over
`(teacher_view, student_view)` pairs, **skipping same-view pairs**:

```python
for teacher_index, teacher_probs in enumerate(teacher_chunks):      # 2 global
    for student_index, student_logits in enumerate(student_chunks):  # 6 total (2 global + 4 local)
        if student_index == teacher_index:
            continue    # no cross-view signal from scoring a view against itself
        loss = torch.sum(-teacher_probs * F.log_softmax(student_logits, dim=-1), dim=-1)
```

With 2 global crops (teacher) and 6 total student views, this produces
`2 × 6 − 2 = 10` cross-view terms, averaged. Skipping same-view pairs is what
forces the representation to become invariant to the augmentation rather than
to memorize a single view.

**Temperature schedule (Eq. 2), `teacher_temperature(epoch)`:** ramps
linearly from `warmup_teacher_temp` (0.02) to `teacher_temp` (0.04) over the
first `warmup_teacher_temp_epochs` (5) epochs via
`np.linspace(...)`, then holds constant (`dino.py:78-93`). A cold (low-
temperature) teacher early on would produce overly sharp targets the student
could collapse onto before it has learned anything.

**Centering (Eq. 3), `update_center`:**

$$
C_t = m \cdot C_{t-1} + (1-m) \cdot \bar{q}, \qquad m = 0.9
$$

subtracted from the teacher logits **before** the softmax in
`compute_dino_loss` (`dino.py:123,126`). Sharpening (low temperature) pushes
toward one-hot outputs; centering alone would push toward a uniform output.
The two forces in opposition are what keeps the representation from
collapsing in either direction.

Both `student_output` and `teacher_output` may be passed as either a list of
per-view tensors or one pre-stacked tensor; `_concat_outputs`
(`dino.py:155-162`) normalizes either form into one `[views * batch, dim]`
tensor before chunking.

## 6. `contrastive_pretrain.py` training loop (per step)

`src/trainers/contrastive_pretrain.py:241-337`:

1. Build `2 + local_crops_number` views via `DataAugmentationDINO`.
2. **Teacher forward on the 2 global crops only**, under `torch.no_grad()`
   (line 254-255) — paper Fig. 7.
3. **Student forward on all views** (line 256).
4. `criterion(student_out, teacher_out, epoch=epoch)` — Eq. 1, plus the
   centering-buffer update as a side effect.
5. `loss.backward()`, then optionally log gradient norms, then
   `torch.nn.utils.clip_grad_norm_(student_parameters, max_norm=3.0)`
   (Table 1), **then** `model.student_head.cancel_last_layer_gradients(epoch)`
   — cancelling must happen after clipping and before `step()`.
6. `optimizer.step()`, then EMA the teacher:
   `update_momentum(student_backbone, teacher_backbone, m=0.996)` and
   the same for the heads (`lightly.models.utils.update_momentum`, Table 1).

### Checkpoint handoff — the *only* connection between stages

At the end of training (`contrastive_pretrain.py:393-410`):

```python
final_file    = save_dino_checkpoint(..., filename="dino_pretrained_final.pth")   # full state
backbone_file = save_path / "dino_pretrained_backbone.pth"                         # bare student_backbone state dict
torch.save(to_cpu_state_dict(model.student_backbone.state_dict()), backbone_file)

shared_file = publish_shared_backbone(cfg, backbone_file, logger)
```

`publish_shared_backbone` (`contrastive_pretrain.py:110-131`) copies
`backbone_file` to `experiment.training.shared_backbone_path`, by default
`${SEED_OUTPUT_DIR}/checkpoints/dinov2_swinv2_pretrained.pth`. Failure here is
logged as a **warning, not an error** — the per-stage copy is already safely
on disk, and discarding a completed 300-epoch run over a file-copy failure
would be indefensible. Every downstream ablation, baseline, and finetune run
reads this one published file, which is what keeps the comparison suite valid
(see [`06_ABLATION_ENGINE.md`](06_ABLATION_ENGINE.md)).

## 7. Hyperparameters (`conf/experiment/pretrain_swinv2_dino.yaml`, `conf/model/loss/dino.yaml`)

| Setting | Value | Source |
| --- | --- | --- |
| `epochs` | 300 | Table 1 |
| `learning_rate` | 0.0005 | Section 6.1 |
| `weight_decay` | 0.01 | |
| `momentum_teacher` | 0.996 | Table 1 |
| `clip_grad` | 3.0 | Table 1 |
| `freeze_last_layer_epochs` | 1 | Section 6.1 |
| `optimizer.name` | AdamW | Section 6.1 |
| `scheduler.name` | cosine | Section 6.1: "cosine decay scheduler" |
| `warmup_teacher_temp` → `teacher_temp` | 0.02 → 0.04 | Table 1 |
| `warmup_teacher_temp_epochs` | 5 | Table 1 |
| `center_momentum` | 0.9 | Eq. 3 |
| `student_temp` | 0.1 | |
| `out_dim` (DINOHead) | 65,536 | Table 1 |

Logged every step (subject to `tracking.intervals.log_every_steps`): loss,
learning rate, current teacher temperature (making the Eq. 2 schedule
directly visible), gradient norms; every epoch: mean loss, duration, a loss
curve figure (paper Fig. 6) every `figure_every_epochs`.
