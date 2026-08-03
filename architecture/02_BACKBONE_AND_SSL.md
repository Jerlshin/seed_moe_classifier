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

### What stage 1 actually is

This module implements **DINO (Caron et al., 2021) with a SwinV2 trunk**, plus
two of DINOv2's four additions. It is not DINOv2, and nothing in the tree claims
it is any more.

| DINOv2 component | Present? |
| --- | --- |
| iBOT patch-level masked-prediction objective | **No** — needs token-level student/teacher outputs and a masking pipeline |
| KoLeo regulariser | **Yes** (`koleo_regularizer`) |
| Sinkhorn-Knopp centering replacing softmax-with-EMA-centering | **Yes** (`sinkhorn_knopp`), and it is the default |
| Untied image-level / patch-level head weights | N/A — one head |

The submitted tree named this DINOv2 throughout, down to the published checkpoint
filename. That is a missing-method claim with a known magnitude, not a cosmetic
naming choice: DINOv2's own ablations credit the iBOT term with ~3 % on dense
tasks and KoLeo with >8 % on retrieval. The honest description — used in the
class docstrings, the config names and the paper — is **"DINO-style
self-distillation with the KoLeo and Sinkhorn components of DINOv2"**.

### Cross-view objective (Eq. 1)

$$
\mathcal{L}_{\text{DINO}} = -\frac{1}{N}\sum_i \sum_{v \neq q} q_v \cdot \log p_v
$$

Only *cross-view* pairs contribute: a student view is never scored against the
teacher's output for that same view, which is what forces invariance to the
augmentation rather than memorisation of it. With 2 global and 4 local crops that
is `2 x 6 - 2 = 10` terms.

**View identity is now explicit.** The submitted code skipped same-view pairs
with `if student_index == teacher_index`, which is correct only while the
student's first two views happen to be the two globals in the teacher's order —
an invariant nothing enforced. If `_concat_outputs` ordering ever changed, a
global-local pair would be silently skipped and a same-view pair silently
included, with no error. `forward` now takes `student_view_ids` and
`teacher_view_ids`, supplied by `DataAugmentationDINO.view_ids` /
`.global_view_ids`, and a test asserts exactly 10 terms.

### Collapse control — both guards were set toward collapse

DINO prevents collapse by balancing two opposing forces: **sharpening** (a low
teacher temperature pulls targets toward one-hot) and **centering** (subtracting a
running mean pushes them toward uniform). The submitted configuration set both in
the collapsing direction simultaneously.

**(a) Sharpening was ~2x stronger than reference.** Config: `warmup_teacher_temp
0.02 -> teacher_temp 0.04`. DINO: `0.04 -> 0.07`. A teacher at `tau = 0.04` is
roughly twice as sharp as DINO's converged 0.07, for the whole run.

**(b) Centering was noise-dominated.** `C` lives in `R^65536` and was an EMA at
`m = 0.9` — an effective window of ~10 steps — over `2 global crops x batch 16 =
32` teacher vectors per step:

| | Submitted | DINO reference |
| --- | --- | --- |
| Teacher vectors per step | 32 | 2,048 (batch 1024 x 2) |
| Effective samples in `C` | ~320 | ~20,480 |
| Samples per estimated dimension | **0.005** | 0.31 |

That is **1/64th** the sample density DINO has, for the same 65,536 dimensions.
The counterweight to sharpening was essentially noise.

### The corrections

| Parameter | Submitted | Revision | Why |
| --- | --- | --- | --- |
| `warmup_teacher_temp -> teacher_temp` | 0.02 -> 0.04 | **0.04 -> 0.07** | match DINO; sharpening is the collapse-inducing force |
| `warmup_teacher_temp_epochs` | 5 | **30** | DINO's ramp length |
| `centering` | softmax + EMA | **`sinkhorn`** | normalises *within the batch*; nothing estimated across steps |
| `center_momentum` | 0.9 | **0.99** | only used by the `ema` path; window 10 -> 100 steps |
| `out_dim` | 65,536 | **8,192** | see §4 |
| `lambda_koleo` | — | **0.1** | uniform feature span within a batch |

### Sinkhorn-Knopp centering

```python
sinkhorn_knopp(logits, temperature, iterations=3) -> assignments  # [batch, out_dim]
```

Alternately normalises prototype and sample marginals so the assignment is doubly
stochastic **within the batch**. Because nothing is estimated across steps, (b)
above stops being a problem at all rather than being mitigated.

Computed entirely in **log space** with `logsumexp`. The direct form
`exp(logits / tau)` overflows immediately at these temperatures — unit-scale
logits at `tau = 0.04` reach `exp(25)` — and a single `inf` turns the whole
assignment into `nan`, silently, because the result is detached and only surfaces
later as a `nan` loss. This was caught by a test on the first run.

### KoLeo regulariser

$$
\mathcal{L}_{\text{koleo}} = -\frac{1}{n}\sum_i \log\Big(\min_{j \neq i} \lVert z_i - z_j \rVert\Big)
$$

Encourages a uniform span of the feature sphere within a batch, so
distinct-but-similar samples do not collapse onto each other. That is precisely
the failure mode a *fine-grained* task cannot tolerate: 27 sub-varieties of four
crops are near-duplicates by construction.

Applied to the **bottleneck**, not the 65k-wide prototype logits — uniformity is a
property of the representation, and measuring it in prototype space would measure
the head's output distribution instead. `DINOHead.forward(x,
return_bottleneck=True)` supplies it.

### Collapse diagnostics

A partially collapsed run has a **perfectly plausible-looking loss curve**, so
Fig. 6 will not reveal any of the above. `collapse_metrics()` reports, every
logged step:

* `teacher_entropy` — falling toward 0 means the targets have sharpened to one-hot
* `teacher_entropy_max` — `log(out_dim)`, for scale
* `prototype_kl_to_uniform` — rising means the batch is using a shrinking subset
  of the prototypes

Either alone is the signature the loss curve hides.

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

| Key | Value | Source |
| --- | --- | --- |
| `backbone.name` | `swinv2_base_window16_256` | Table 1 |
| `data.batch_size` | 16 | Table 1 |
| `gradient_accumulation_steps` | **4** | revision — effective batch 64 |
| `epochs` | 300 | Table 1 |
| `learning_rate` | 0.0005, cosine decay | Section 6.1 |
| `clip_grad` | 3.0 | Table 1 |
| `freeze_last_layer_epochs` | 1 | Section 6.1 |
| `momentum_teacher` | 0.996 -> **1.0, cosine** | Table 1 start, DINO schedule |
| `weight_decay` | **0.04 -> 0.4, cosine** | DINO schedule (was constant 0.01) |
| `head.out_dim` | **8192** | revision (Table 1: 65,536) |
| `head.use_batch_norm` | **`"layer"`** | revision (Section 4 says batch) |
| `loss.warmup_teacher_temp -> teacher_temp` | **0.04 -> 0.07 over 30 epochs** | revision (Table 1: 0.02 -> 0.04 over 5) |
| `loss.centering` | **`sinkhorn`** | revision |
| `loss.lambda_koleo` | **0.1** | revision |

`tests/conftest.py` carries both the submitted and the revised value for every
one of these as named constants (`SUBMITTED_TEACHER_TEMP` /
`REVISED_TEACHER_TEMP`, and so on), so a test asserting a bare number cannot
silently become a claim about whichever the reader assumed.

### Why the effective batch matters

Every collapse guard in DINO is a **batch statistic**. Batch 16 is far below the
regime they were designed for, so gradient accumulation raises the effective batch
to 64 before stepping — which also raises the teacher-vector count per optimiser
step from 32 to 128.

### The local-crop resolution confound — measured, and not removable

`local_crop_size: 101` crops are resized back up to 256 because SwinV2's shifted
windows need a fixed resolution. Every local view therefore carries a systematic
low-pass signature that global views do not, and the student can distinguish local
from global by blur alone — a shortcut that partially substitutes for the
local-to-global correspondence the multi-crop objective exists to teach. The
per-view blur probabilities (global-1: 1.0, global-2: 0.1, local: 0.5) aggravate
it.

`dynamic_img_size=True` does **not** remove this. timm accepts the flag for
SwinV2, but the attention still asserts the native resolution:

```text
>>> model = timm.create_model("swinv2_tiny_window16_256", dynamic_img_size=True)
>>> model(torch.randn(2, 3, 101, 101))
AssertionError: Input height (101) doesn't match model (256).
```

That is measured, and `tests/test_models.py` pins it, so nobody re-derives the
hypothesis from the flag's name.

The confound is therefore documented rather than removed.
`augmentation.match_view_lowpass: true` mitigates it by putting the *global* crops
through the same downsample-then-upsample cycle, so the artefact carries no
discriminative signal. It is off by default because it changes the submitted
recipe, and it is the honest fix if the shortcut turns out to matter.
