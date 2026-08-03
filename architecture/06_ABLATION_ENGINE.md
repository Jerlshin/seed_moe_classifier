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

## 2. What each toggle actually changes

A named "one-toggle" ablation is only interpretable if it flips one factor. Of
the six variants the revision originally shipped, **only two did**.

| Variant | Intended factor | Factors actually changed | Clean? |
| --- | --- | --- | --- |
| `wo_moe` | routing | routing + active capacity + 2 regularisers | **No** |
| `wo_arcface` | angular margin | margin + embedding L2-norm + centre L2-norm + logit scale | **No** |
| `wo_residual` | Eq. 9 fusion | Eq. 9 + the entire compactness term | **No** |
| `wo_kl` | Eq. 10 | Eq. 10 | Yes |
| `wo_cross_attn` | Eqs. 11-12 | Eqs. 11-12 | Yes |

The design principle in §1 — *"an ablation routed through a second training loop
would differ from the full model in ways nobody intentionally chose"* — is
exactly right, and the three failing rows are that same failure one level down:
the *config* differences nobody intentionally chose. The principle had to be
applied to the toggle semantics as well as to the trainer.

### `wo_moe` changed three things

1. Removed learned routing — the intended factor.
2. Halved active capacity: the full model activates `top_k = 2` experts per token
   and a `DenseExpertBlock` activates one. The gap conflated routing with a 2x
   cut in active FLOPs — the very quantity the efficiency section is built on.
3. Zeroed **both** MoE regularisers. This document used to present that as a
   feature (*"no downstream consumer needs a special case"*); the plumbing is
   elegant, but it means `wo_moe` optimised a **strictly smaller objective** than
   the full model.

Resolved by keeping `wo_moe` as the historical comparison and adding three
controls: `wo_moe_capacity_matched` (`dense_capacity_multiplier=2`, capacity held
fixed), `moe_fixed_router` (`router_mode=hash` — all six experts, sparse capacity
held fixed, *learning* removed from the routing) and `moe_uniform_router`
(`router_mode=uniform` — ensembling, sparsity removed). `moe_fixed_router` is the
important one: it is the only configuration that can say whether the router
learned anything, and it is the direct answer to the question a reviewer asks
about Section 5.2.

### `wo_arcface` changed four things

Margin, embedding L2-norm, centre L2-norm and logit scale — the last of which
also changes the sharpness of `P_sub` feeding the KL term and the geometry the
t-SNE panels visualise. See
[`04_HIERARCHICAL_FUSION.md`](04_HIERARCHICAL_FUSION.md) §7.

Resolved by adding `NormFaceHead` and splitting the variant into
`wo_margin_only` (the true single-factor margin control: normalised embedding and
centres, `m = 0`, same scale) and `wo_angular_head` (the four-factor change,
honestly named).

### `wo_residual` changed two things

Under `cosine_mode="residual"`, `use_residual=false` set `projected_seed = 0`, so
`L_cos = 1 - cos(h, h) = 0` identically for every sample for the whole run. The
toggle removed Eq. 9 **and** the paper's Section-1 contribution. Given that the
cosine term's minimisers destroy the residual anyway, this confound could
plausibly have made `wo_residual` look *better* than the full model, with no way
to explain why.

Resolved by dissolving it rather than patching it: `cosine_mode` defaults to
`intra_class`, which is a property of the ArcFace embedding and does not ride on
the residual. `wo_residual` now removes Eq. 9 and only Eq. 9; `wo_layer_scale`
isolates the gain separately. A test asserts both halves — that compactness
survives the toggle, and that the submitted formulation still collapses.

### Allocation discipline (unchanged, and worth keeping)

A disabled block is **not allocated**: the attribute is set to `None`, never
`nn.Identity`, so an ablation's reported parameter count reflects the model
actually trained. `use_moe=False` substitutes one dense block rather than nothing,
so the gap measures routing rather than a missing layer's capacity — the
rationale was right, it just needed the capacity-matched counterpart above.

### Machine-readable traces

`component_flags()` reports every axis a variant can move — `token_mode`,
`fusion_mode`, `sub_head_variant`, `router_mode`, `gate_conditioning`,
`num_experts`, `top_k`, `dense_capacity_multiplier` — and the criterion
contributes `loss_flags()` (all lambdas, `kl_mode`, `tau_kl`,
`detach_kl_seed_target`, `weighting_mode`, `cosine_mode`, `moe_load_mode`).

This matters concretely: reporting only the four architectural booleans made a
`wo_kl` run **byte-identical to `full_model`** in `summary.json`, with only the
variant *name* separating them. `summary.json` also carries a `split` block
recording the protocol and the dataset's provenance diagnostics.

## 3. The variant grid (`scripts/run_ablations.py`)

Eighteen variants, each documented with the factors it moves. `full_model` is the
reference every other is compared against.

| Variant | Override | Isolates |
| --- | --- | --- |
| `full_model` | *(none)* | reference |
| `wo_moe` | `use_moe=false` | routing + capacity + 2 regularisers (historical) |
| `wo_moe_capacity_matched` | `+ dense_capacity_multiplier=2` | routing, capacity fixed |
| `moe_fixed_router` | `router_mode=hash` | **learned** routing, sparse capacity fixed |
| `moe_uniform_router` | `router_mode=uniform`, `top_k=6` | sparsity, ensembling fixed |
| `wo_gate_conditioning` | `gate_conditioning=false` | the hierarchical link into the router |
| `wo_margin_only` | `sub_head_variant=normface` | the angular margin, alone |
| `wo_angular_head` | `sub_head_variant=linear` | margin + normalisation + scale |
| `wo_residual` | `use_residual=false` | Eq. 9 fusion |
| `wo_layer_scale` | `residual_layer_scale=null` | the residual gain |
| `film_fusion` | `fusion_mode=film` | additive vs. multiplicative conditioning |
| `wo_kl` | `use_kl_loss=false` | Eq. 10 |
| `kl_jsd` | `kl_mode=jsd` | forward KL vs. symmetric JSD |
| `wo_cross_attn` | `use_cross_attention=false` | Eqs. 11-12 |
| `pooled_tokens` | `token_mode=pooled` | **routing granularity** — the submitted architecture |
| `load_entropy` | `moe_load_mode=entropy` | dispatch-aware vs. soft-only balancing |
| `wo_stage2_augmentation` | flip + crop off | stage-2 augmentation |
| `leakage_ungrouped` | `split_protocol=stratified` | **the crop-level leak, as a number** |

Three of these are not architecture ablations and are worth calling out:

* **`pooled_tokens`** measures what keeping SwinV2's token grid buys. Under
  `pooled` the expert and cross-attention Q/K projections are not built at all,
  because a length-1 attention can never use them.
* **`load_entropy`** reproduces the submitted load-balancing loss. `L_load(entropy)`
  against `L_load(switch)` on one split converts a correctness finding into a
  small empirical result.
* **`leakage_ungrouped`** is the full model under crop-level splitting. The delta
  against `full_model` quantifies what the source-photograph leak was worth. That
  is a methods result worth reporting, not a cleanup step to hide — the applied-DL
  literature reports it exactly this way.

### Every variant runs at five seeds

`DEFAULT_SEEDS = (42, 43, 44, 45, 46)`, expanded by `expand_seeds()` into
`outputs/ablations/{variant}/seed{n}/`.

One run per variant cannot resolve the table it is being asked to support. On a
1,871-image test split at ~95 % accuracy:

```text
SE(p)         = sqrt(0.95 . 0.05 / 1871) = 0.00504  ->  +-0.99 pp   (95 % CI half-width)
SE(p1 - p2)   = sqrt(2) . 0.00504        = 0.00713  ->  +-1.40 pp   (unpaired difference)
```

So **any ablation gap below ~1.4 pp sits inside the noise floor of the test split
alone** — before any training-seed variance from dropout, shuffling, router
initialisation, or (for a MoE specifically) which experts happen to win the early
race. For a 27-class fine-grained task where component contributions of 0.5-2 pp
are the normal magnitude, that is not enough resolution.

`scripts/generate_plots.py` aggregates to **mean ± SD** and adds **McNemar's exact
test** against `full_model`, Holm-adjusted across the family. The paired test is
available precisely because the suite guarantees a byte-identical test split, and
it is strictly more powerful than comparing independent intervals. It needs
nothing beyond the `test_predictions.npz` files already on disk.

### `ablation_flat_classifier` — a distinct, seventh ablation

`conf/experiment/ablation_flat_classifier.yaml` (reached via `python main.py
ablation`, **not** part of the `run_ablations.py` suite) is architecturally
different from the variants above: it removes the coarse stage's *influence on the
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

The gap against the full model here measures the **combined** contribution of
coarse supervision plus expert specialization, distinct from the suite's
one-factor-at-a-time isolation. It is deliberately excluded from
`run_ablations.py`, and the paper should keep that framing.

## 4. Baselines (`scripts/run_baselines.py`, `src/models/baselines.py`)

Five reference points. The submitted suite had three, and was missing the one a
reviewer asks for first.

| Baseline | Backbone | Trained | What the gap isolates |
| --- | --- | --- | --- |
| **`linear_probe`** | Same self-supervised encoder, frozen | `Linear(384,4)` + `Linear(384,27)`, plain CE | **Does any of the head machinery beat a linear layer on the same features?** |
| `swinv2_supervised` | ImageNet SwinV2-Base | End-to-end, full hierarchical head | In-domain self-supervision, separated from the architecture |
| `resnet50` | ImageNet ResNet-50 | End-to-end, supervised | Conventional CNN reference point |
| `swin_tiny` | ImageNet Swin-T | End-to-end, supervised | Shifted-window family held constant |
| `hierarchical_cce` | Same encoder, frozen | Head-only, plain CE at both levels | Whether being hierarchical at all helps |

### `linear_probe` — run this one first

If the full architecture does not clear a linear layer on identical frozen
features by a comfortable, seed-stable margin, that is the single most important
number in the paper. It shares the encoder path with the proposed model exactly,
so the gap is attributable to the head rather than to a different representation.

**`hierarchical_cce` is not this control.** It keeps `use_residual: true`, so it
retains the coarse-to-fine link and the `SubVarietyEmbedding` MLP. It is a
composed point in the ablation lattice (`wo_moe` + `wo_angular_head` +
`wo_cross_attn` + `wo_kl`), not an independent baseline, and reading it as one
would overstate what the head contributes.

### `swinv2_supervised` — separating SSL from architecture

ImageNet-initialised SwinV2-Base with the *full* hierarchical head and no stage-1
checkpoint (`checkpoint_path: null`, `freeze: false`). The only variant that
separates "in-domain self-supervised pretraining" from "the architecture".
`validate_swinv2_name` reserves `model/backbone` for the self-supervised path, so
this is configured through its own experiment file.

### The under-tuned-baseline objection

`resnet50` and `swin_tiny` train end to end while the proposed model trains a head
against a frozen encoder. Those have genuinely different optimal learning rates,
and a single shared value cannot be right for both — "our method wins against an
under-tuned baseline" is the most common objection any comparison table attracts.

`--lr-sweep` runs `{1e-5, 3e-5, 1e-4}` per end-to-end baseline and reports each
one's best. Six extra runs, and the objection goes away.

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
Independent heads give the supervised baselines a real, comparable alignment
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
