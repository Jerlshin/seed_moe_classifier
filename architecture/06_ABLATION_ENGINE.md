# 06 — Ablation Engine

Covers the five component toggles, `src/models/baselines.py`,
`src/trainers/runner.py`, `scripts/run_ablations.py`,
`scripts/run_baselines.py`, and `conf/experiment/{ablation_flat_classifier,
baseline_*}.yaml`.

## 1. Design principle: one trainer, all variants

The full model, all six component-wise ablations, and all three baselines run
through the **same** `src/trainers/moe_finetune.py`, differing only by Hydra
overrides. `build_model_and_encoder(cfg, device)`
(`moe_finetune.py:173-194`) is the single dispatch point:

```python
head_name = OmegaConf.select(cfg, "model.head.name", default="hierarchical_moe")
if head_name == "flat_supervised":
    return IdentityEncoder().to(device), build_baseline(cfg.model.head).to(device)
encoder = build_encoder(cfg.model.backbone, embed_dim=cfg.model.head.embed_dim).to(device)
model = build_hierarchical_moe(cfg.model.head).to(device)
return encoder, model
```

This is deliberate: an ablation routed through a second training loop would
differ from the full model in ways nobody intentionally chose — a different
shuffle order, a different metric reduction, a different early-stopping rule
— and the measured gap between variants would silently include all of that
noise. Because every variant's `model` emits the same `HierarchicalOutput`
dataclass (see [`00_OVERVIEW.md`](00_OVERVIEW.md) §3), the losses, the metrics
stack, and the figure generators need **zero** branching to support any
variant.

## 2. The five component toggles

All five live under `model.head` in
`conf/model/head/hierarchical_moe.yaml:55-59`:

```yaml
use_moe: true
use_arcface: true
use_residual: true
use_cross_attention: true
use_kl_loss: true
```

| Flag | Effect when `False` | Where enforced |
| --- | --- | --- |
| `use_moe` | `DenseExpertBlock` (one dense transformer block, no routing) replaces `MixtureOfExperts` | `HierarchicalSeedClassifier.__init__`, `builder.py:249-265` |
| `use_arcface` | `LinearSubVarietyHead` (plain linear + implicit CE) replaces `ArcFaceHead` | `builder.py:297-309` |
| `use_residual` | `h' = h`; `self.seed_projection = None`; `projected_seed` fabricated as zeros | `builder.py:267-277`, `forward` at `builder.py:355-362` |
| `use_cross_attention` | `h'' = h'`; `self.cross_attention = None` | `builder.py:280-289`, `forward` at `builder.py:364-376` |
| `use_kl_loss` | Eq. 10 not computed at all (not weighted to zero) | `CombinedHierarchicalLoss.__init__`/`forward`, `hierarchical.py:243-246, 281-290` |

`use_kl_loss` lives on `model.head` (not `model.loss`) purely so all five
toggles have one physical home; `conf/model/loss/arcface_kl.yaml:16` reaches
it via interpolation: `use_kl_loss: ${model.head.use_kl_loss}`. One flag, two
consumers (the loss builder reads it directly; nothing in the model itself
branches on it), no way for them to disagree.

**Two design rules every toggle obeys** (stated in `REVISION_NOTES.md` §4 and
`src/README.md`, and checked by dedicated tests in
`tests/test_models.py`):

1. **A disabled component is never allocated.** The attribute is set to
   `None`, not `nn.Identity` — an ablation's reported parameter count
   describes the model that was actually trained, not the full model with
   dead weights sitting alongside it.
2. **Replace, don't delete, when capacity is at stake.** `use_moe=False`
   substitutes `DenseExpertBlock` — architecturally identical to a single
   expert — rather than removing the layer outright, so the `wo_moe` gap
   against the full model is attributable to *routing*, not to a missing
   block's capacity. `use_arcface=False` similarly substitutes
   `LinearSubVarietyHead`, which mirrors `ArcFaceHead`'s `(logits,
   margin_logits)` return contract so the model body and the loss need no
   branch (see [`04_HIERARCHICAL_FUSION.md`](04_HIERARCHICAL_FUSION.md) §7
   and [`05_LOSS_FUNCTIONS.md`](05_LOSS_FUNCTIONS.md)).

`component_flags()` (`builder.py:321-328`) reports the four architectural
booleans for logging (`use_kl_loss` excluded — it belongs to the loss, not
the model) and is written into `summary.json`'s `component_flags` field by
`write_run_summary` (`moe_finetune.py:1018`).

## 3. The six component-wise ablation variants (`scripts/run_ablations.py`)

```python
ABLATION_VARIANTS = [
    VariantSpec("full_model",     overrides=[]),
    VariantSpec("wo_moe",         overrides=["model.head.use_moe=false"]),
    VariantSpec("wo_arcface",     overrides=["model.head.use_arcface=false"]),
    VariantSpec("wo_residual",    overrides=["model.head.use_residual=false"]),
    VariantSpec("wo_kl",          overrides=["model.head.use_kl_loss=false"]),
    VariantSpec("wo_cross_attn",  overrides=["model.head.use_cross_attention=false"]),
]
```

Each variant flips **exactly one** toggle, and every variant reads the same
published DINOv2 encoder. Results land in
`outputs/ablations/{variant}/`, one self-contained directory holding its
Hydra config snapshot, logs, checkpoints, figures, `summary.json`, and
`test_predictions.npz`.

```bash
python scripts/run_ablations.py                        # all six
python scripts/run_ablations.py --variants wo_moe wo_kl
python scripts/run_ablations.py --dry-run               # print commands only
python scripts/run_ablations.py -- data.batch_size=8 experiment.training.epochs=20
```

### `ablation_flat_classifier` — a distinct, seventh ablation

`conf/experiment/ablation_flat_classifier.yaml` (reached via `python main.py
ablation`, **not** part of the `run_ablations.py` suite) is architecturally
different from the six above: it removes the coarse stage's *influence on the
gradient* rather than one structural block. The seed-type head still runs and
is still scored, but:

```yaml
model:
  head:
    use_moe: false        # dense block, isolating expert-specialisation contribution
    use_kl_loss: false
  loss:
    lambda_seed: 0.0       # no coarse supervision
    lambda_kl: 0.0
    lambda_cosine: 0.0
    lambda_moe_load: 0.0
    lambda_moe_sparsity: 0.0
    lambda_arcface: 1.0    # ArcFace alone drives the classifier
```

The gap against the full model here measures the **combined** contribution
of coarse supervision plus expert specialization, distinct from the six
variants' one-ingredient-at-a-time isolation.

## 4. Baselines (`scripts/run_baselines.py`, `src/models/baselines.py`)

Three reference points, each answering a different question:

| Baseline | Backbone | Trained | What the gap isolates |
| --- | --- | --- | --- |
| `resnet50` | ImageNet ResNet-50 | End-to-end, supervised | Conventional CNN reference point |
| `swin_tiny` | ImageNet Swin-T (`swin_tiny_patch4_window7_224`) | End-to-end, supervised | DINOv2 pretraining + hierarchical head, with the shifted-window family held constant |
| `hierarchical_cce` | Same DINOv2-SwinV2 encoder as the proposed model | Frozen encoder, head-only (like the full model) | Whether being hierarchical at all helps, independent of MoE/cross-attention/ArcFace |

### `FlatSupervisedBaseline` (`src/models/baselines.py:74-193`)

```text
images --backbone--> pooled --Linear+LayerNorm--> z in R^384
                                                    |-- Linear --> 4 seed types  (seed_head)
                                                    |-- Linear --> 27 sub-varieties (sub_head)
```

**Two independent linear heads, not one flat 27-way classifier.** A single
27-way head would make the hierarchical alignment rate identically `1.0` by
construction — the seed type would be *derived* from the sub-variety
prediction, so the two could never disagree, making that column meaningless.
Independent heads give `resnet50`/`swin_tiny` a real, comparable alignment
measurement. The `EmbedDim=384` projection is likewise not required by the
baseline itself; it exists purely so the t-SNE panels and the embedding-space
column of the results table stay comparable with the proposed model's.

`forward` returns a `HierarchicalOutput` with a degenerate one-expert gate
(`gate_probs = ones([batch, 1])`, `top_k_indices = zeros`), `projected_seed =
zeros_like(embedding)`, `sub_margin_logits = sub_logits` (no margin — the
ArcFace term of the combined objective therefore reduces to plain CE) — so
the entire evaluation stack (metrics, KL alignment, confusion matrices,
t-SNE, trackers) runs against it unmodified. `component_flags()` reports all
four architectural booleans as `False`.

`IdentityEncoder` (`baselines.py:53-71`) is a no-op `forward(images) ->
images`, filling the encoder slot for end-to-end baselines that own their own
backbone — this keeps `moe_finetune.py`'s training loop structured as
`encoder(images) -> model(features)` for every variant without a branch.

### `hierarchical_cce` — a baseline needing zero new code

`conf/experiment/baseline_hierarchical_cce.yaml`:

```yaml
defaults:
  - finetune_hierarchical_moe
  - override /model/loss: flat_cce

model:
  head:
    use_moe: false
    use_arcface: false
    use_cross_attention: false
    use_residual: true      # keeps the coarse-to-fine link — otherwise not a hierarchical baseline
    use_kl_loss: false
```

It is the proposed model's own `HierarchicalSeedClassifier` with three
toggles flipped and `use_residual` deliberately kept on — reusing the same
encoder, DINOv2 weights, data pipeline, and evaluation code as the full model
exactly, which is what makes it the tightest of the three controls.

```bash
python scripts/run_baselines.py                          # all three
python scripts/run_baselines.py --models resnet50
python scripts/run_baselines.py -- experiment.training.epochs=30
```

`END_TO_END_BASELINES = {"resnet50", "swin_tiny"}`
(`run_baselines.py:79`) — these two are **deliberately not given** the
DINOv2 checkpoint (their `spec_checkpoint = None`,
`run_baselines.py:141`), since they own ImageNet backbones of a different
architecture family where a SwinV2 DINO state dict would at best be ignored
and at worst partially loaded. `hierarchical_cce` does read the shared
checkpoint, resolved only when at least one selected baseline actually needs
it (`needs_encoder`, `run_baselines.py:119`).

Baseline-specific learning rates are lower (`3e-5` vs. the proposed model's
default) because the **whole network** trains from ImageNet weights here,
unlike the proposed model's frozen-encoder recipe where only the head and the
Eq. 4 projection train.

## 5. Suite mechanics (`src/trainers/runner.py`)

### Shared-checkpoint discipline

```python
DEFAULT_PRETRAINED_CHECKPOINT = "checkpoints/dinov2_swinv2_pretrained.pth"

def ensure_pretrained_checkpoint(path, allow_missing=False) -> Path | None:
    if Path(path).exists():
        return Path(path)
    if allow_missing:
        return None
    raise FileNotFoundError(...)   # explains how to produce it: `python main.py pretrain`
```

(`runner.py:58-82`) Every DINOv2-path variant — the full model, all six
ablations, and `hierarchical_cce` — must start from **byte-identical**
encoder weights, or the comparison table would partly measure self-supervised
initialization noise rather than the architectural change under test.
`--allow-missing-checkpoint` waives this only for smoke runs, explicitly
marking the resulting numbers as not comparable.

### `VariantSpec` / `build_command` / subprocess execution

```python
@dataclass
class VariantSpec:
    name: str
    description: str
    overrides: list[str] = field(default_factory=list)
    group: str = "ablation"                       # or "baseline"
    experiment: str = "finetune_hierarchical_moe"
```

`build_command` (`runner.py:118-143`) assembles
`python -m src.trainers.moe_finetune experiment=... experiment.variant=...
experiment.training.save_path=... hydra.run.dir=... <spec.overrides>
model.backbone.checkpoint_path=<checkpoint> <extra_overrides>` —
**caller-supplied `extra_overrides` are appended last**, because Hydra takes
the rightmost value for a repeated key: a command-line override must always
win over the suite's own defaults.

Each variant runs as a **subprocess** (`run_variant`, `runner.py:146-169`),
for three reasons in order of importance:

1. **Hydra can only be initialized once per process.** Running six variants
   in one process would require `GlobalHydra.instance().clear()` between
   them, leaving stale config state behind — precisely the kind of bug that
   makes an ablation table quietly wrong without raising any error.
2. **GPU memory is released completely when a process exits**; six models
   built in-process would accumulate.
3. **A crash in one variant cannot take the whole suite down** — `run_suite`
   (`runner.py:172-193`) records the failure via `VariantResult.succeeded`
   and continues past it by default (`--stop-on-failure` for the opposite).

`write_suite_manifest` (`runner.py:196-217`) records every variant's
overrides, return code, duration, and command into
`outputs/{ablations,baselines}/suite_manifest.json` — a durable record of
exactly what ran with what configuration.

## 6. Config composition that keeps variants honest

Every ablation and baseline experiment file inherits from
`finetune_hierarchical_moe.yaml` via Hydra's `defaults: - finetune_hierarchical_moe`.
Because every variant shares that base, a change to shared training-loop
settings (epochs, learning rate, optimizer, scheduler, split parameters)
propagates identically to all of them — the only way for two variants to
differ is the specific override each `VariantSpec` or experiment file adds on
top. This is the config-level enforcement of the same principle behind "one
trainer, all variants" in §1.
