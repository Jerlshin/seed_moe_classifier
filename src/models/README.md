# `src/models/` — network definitions

Implements paper Sections 4 (self-supervised backbone) and 5 (hierarchical head),
plus the supervised baselines the revision compares against.

| File | Contents |
| --- | --- |
| `builder.py` | `HierarchicalSeedClassifier`, `HierarchicalOutput`, `DinoV2SwinV2Encoder`, `BackboneFeatureExtractor`, `validate_swinv2_name`, config-driven builders |
| `baselines.py` | `FlatSupervisedBaseline` (ResNet-50 / Swin-T), `IdentityEncoder` |
| `backbones/swinv2_dino.py` | `DINO` student/teacher pair and `DINOHead` (Section 4) |
| `components/` | The reusable blocks — see [components/README.md](components/README.md) |

## `DinoV2SwinV2Encoder` (`builder.py`)

The single place the paper's embedding width is realised:

```
images ──SwinV2 trunk──▶ pooled [batch, 768 (Tiny) or 1024 (Base)] ──projection──▶ z [batch, 384]
```

No SwinV2 variant emits 384 channels (Tiny/Small: 768, Base: 1024), so a learned
projection is required for Eq. 4 to hold at all. Putting it here rather than
inside the head makes `encoder(images).shape[-1] == 384` an invariant, not a
configuration coincidence — the head, the t-SNE feature dump, the efficiency
profiler and the baselines all observe the same space.

Two consequences that matter in practice:

* **The projection is trainable even when the backbone is frozen.** It must be in
  the optimizer; `build_optimizer()` in the finetune trainer takes a list of
  modules for exactly this reason. Omitting the encoder would silently freeze the
  one layer adapting 1024 backbone channels to 384, and the head would train
  against a random projection.
* **Checkpoints store `encoder_state_dict` alongside `model_state_dict`.** A
  head-only checkpoint would be unusable.

### SwinV2 only

`validate_swinv2_name()` rejects any backbone name not starting with `swinv2`,
and is called from both `BackboneFeatureExtractor` and `DINO.__init__`. The
comparative ViT-S/14 path from the submitted manuscript has been removed, so a
run cannot silently fall back to a different encoder. Supervised comparison
backbones belong in `baselines.py`, never on the self-supervised path.

## `HierarchicalSeedClassifier` (`builder.py`)

The coarse-to-fine cascade, with the paper's equation numbers:

```
z [batch, 384]  (from the encoder; input_projection is an identity)
  │
  ├─ seed_type_classifier(z) ──────▶ s                     Eq. 5   [batch, 4]
  │                                  p_s = softmax(s)      Eq. 6
  │
  ├─ moe(z) ───────────────────────▶ h, G, top-2           Eq. 8   [batch, 384]
  │
  ├─ seed_projection(p_s) ─────────▶ P(p_s)
  │                                  h' = h + P(p_s)       Eq. 9
  │
  ├─ cross_attention(Q=h', K=V=h) ─▶ h'' = LayerNorm(a+Q)  Eqs. 11-12
  │
  ├─ sub_variety_embedding(h'') ───▶ e
  └─ arcface(e, labels) ───────────▶ logits, margin_logits Eq. 13  [batch, 27]
```

Returns a single `HierarchicalOutput` dataclass holding all of the above. Losses
and metrics read named fields, so a new term never forces a signature change.

Two details that are easy to implement wrongly, and that have dedicated tests:

* **The MoE is routed on `z`**, not on the projected seed-type vector. Routing on
  the projection would mean the experts never see the image.
* **The residual adds `P(p_s)`**, the softmax probabilities from Eq. 6 — not
  `P(s)`. Logits are unbounded, so projecting them makes the residual's magnitude
  scale with stage-1 confidence.

### Component toggles

| Flag | Effect when `False` |
| --- | --- |
| `use_moe` | `DenseExpertBlock` replaces the router — one always-on block |
| `use_arcface` | resolves `sub_head_variant` to `linear` unless set explicitly; the objective becomes CE. Prefer `sub_head_variant=normface` to ablate the **margin alone** |
| `use_residual` | `seed_projection` is `None`; `h' = h`, `projected_seed` is zeros |
| `use_cross_attention` | `cross_attention` is `None`; `h'' = h'` |

A disabled block is **not allocated** — the attribute is `None`, not
`nn.Identity` — so an ablation's parameter count describes the model actually
trained. `HierarchicalOutput` stays fully populated either way: `projected_seed`
becomes zeros rather than `None`, which keeps the residual cosine term at exactly
0 instead of NaN.

`use_kl_loss` is the fifth toggle. It lives on the head config for consistency
but is consumed by the loss builder; see [`../losses/README.md`](../losses/README.md).

### ArcFace logits: two outputs, two purposes

`sub_logits` carry **no** margin and are the correct quantity for prediction,
ranking, and the KL hierarchy term. `sub_margin_logits` carry the angular margin
on the target class and are what the ArcFace cross-entropy consumes. Passing no
labels (i.e. at evaluation) makes them identical, so metrics are never inflated
by a training-time margin. With a margin-free head (`normface`, `linear`) they are always identical,
which is exactly why the loss needs no branch for that ablation.

## `BackboneFeatureExtractor` (`builder.py`)

The bare SwinV2 trunk at its **native** width. `DinoV2SwinV2Encoder` wraps it;
use this directly only when the unprojected feature is what you want (e.g.
`scripts/extract_features.py`).

* `freeze=True` (default) keeps it in eval mode with gradients off — the
  two-stage recipe. `train()` is overridden so a frozen backbone can never be
  put back into train mode by an enclosing `model.train()` call.
* `freeze=False` fine-tunes the encoder jointly with the head, which is what
  Section 4 describes. See `PAPER_AUDIT.md` §7.1.
* `load_checkpoint` returns the missing/unexpected key report. `strict=False` is
  the default, so without this a mismatched checkpoint would load quietly and
  surface only as unexplained metrics; the trainer logs the report.

It sniffs several checkpoint layouts (`student_backbone`, `model_state_dict`,
`state_dict`, `student_model`, or a raw state dict) because this project has
produced all of them.

## `FlatSupervisedBaseline` (`baselines.py`)

ImageNet-pretrained backbone → 384-D projection → **two independent** linear
heads (4-way and 27-way), trained end to end.

Two heads rather than one for a specific reason: a flat 27-way classifier would
make the hierarchical alignment rate identically 1.0, because the seed type would
be *derived* from the sub-variety prediction and the two could not disagree by
construction. Independent heads make that column a real measurement, comparable
with the proposed model's.

It emits a `HierarchicalOutput` — with a degenerate one-expert gate — so the
entire evaluation stack runs against a baseline unmodified. `IdentityEncoder`
fills the encoder slot in the trainer, since an end-to-end baseline owns its own
backbone.

The third baseline, two-stage hierarchical CCE, needs **no code**: it is this
package's own head with four toggles flipped
(`conf/experiment/baseline_hierarchical_cce.yaml`), which is what makes it the
tightest of the three controls — identical encoder, data pipeline and evaluation.

## `DINO` (`backbones/swinv2_dino.py`)

Student and teacher share an architecture; the teacher starts as a deep copy with
gradients disabled and only ever moves by EMA (momentum 0.996, Table 1) — the
trainer drives that with `lightly.models.utils.update_momentum`.

`DINOHead` is `Linear → BatchNorm → GELU → … → Linear(bottleneck) → L2-normalize
→ weight-normed Linear(out_dim)`, matching Section 4's description. The final
layer's weight-norm gain is frozen, and its gradients are cancelled for the first
epoch (`freeze_last_layer_epochs`) — the prototype layer is where a collapsing
run collapses first. `_set_weight_norm_gain` handles both PyTorch weight-norm
APIs, since the legacy `weight_g` attribute moved under `parametrizations`.
