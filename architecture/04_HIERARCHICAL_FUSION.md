# 04 — Hierarchical Fusion: Seed-Type Classifier, Residual, Cross-Attention, ArcFace

Covers `src/models/builder.py` (`HierarchicalSeedClassifier`), and
`src/models/components/{classifiers,projections,cross_attention,arcface_head}.py`.
Implements paper Section 5, Eqs. 5-13.

## 1. The full cascade

`HierarchicalSeedClassifier.forward` (`src/models/builder.py:330-397`):

```python
embedding = self.input_projection(features)                    # z,   Eq. 4  (Identity if already 384-D)
seed_type_logits = self.seed_type_classifier(embedding)         # s,   Eq. 5  [batch, 4]
seed_type_probs  = softmax(seed_type_logits, dim=-1)            # p_s, Eq. 6  [batch, 4]

moe = self.moe(embedding)                                       # h,   Eq. 8  -- routed on z, NOT on p_s

if self.seed_projection is not None:
    projected_seed   = self.seed_projection(seed_type_probs)    # P(p_s)
    refined_features = moe.features + projected_seed            # h',  Eq. 9
else:
    projected_seed   = torch.zeros_like(moe.features)            # keeps cosine term at exactly 0, not NaN
    refined_features = moe.features

if self.cross_attention is not None:
    attention = self.cross_attention(query=refined_features.unsqueeze(1),
                                      key=moe.features.unsqueeze(1),
                                      value=moe.features.unsqueeze(1))
    attended_features = attention.features.squeeze(1)            # h'', Eq. 12
else:
    attended_features = refined_features

sub_embeddings = self.sub_variety_embedding(attended_features)
sub_logits, sub_margin_logits = self.arcface(sub_embeddings, sub_variety_labels)   # Eq. 13
```

Three details flagged in `PAPER_AUDIT.md` as easy to get wrong, each pinned
by a dedicated test in `tests/test_models.py`:

1. **The MoE consumes `z` (`embedding`), not a projection of `s` or `p_s`.**
   `moe = self.moe(embedding)`, not `self.moe(projected_seed)`. Eq. 8 is
   defined over the DINO embedding.
2. **The residual adds `P(p_s)` — softmax probabilities — not `P(s)` —
   logits.** `self.seed_projection(seed_type_probs)`, never
   `self.seed_projection(seed_type_logits)`. Logits are unbounded, so
   projecting them would make the residual's magnitude scale with stage-1
   confidence rather than staying bounded.
3. **Cross-attention's query is `h'` (the seed-type-refined feature) while
   key/value are `h` (the raw MoE output)** — `query=refined_features`,
   `key=value=moe.features`. This is Eq. 11's `Q = h'`, `K = V = h`.

`predict(features)` (`builder.py:399-403`) is a `@torch.no_grad()`
convenience returning `(seed_type_logits.argmax(-1), sub_logits.argmax(-1))`
— note it reads `sub_logits`, **never** `sub_margin_logits`.

## 2. Stage 1 — `SeedTypeClassifier` (Eqs. 5-6, `src/models/components/classifiers.py:32-90`)

$$
s = g(z) \in \mathbb{R}^4, \qquad p_s = \text{softmax}(s)
$$

```python
SeedTypeClassifier(feature_dim, num_seed_types, dropout_rate=0.3, hidden_ratio=0.5, variant="mlp")
```

* **`variant="mlp"`** (paper default): `Linear(feature_dim, hidden) → LayerNorm
  → GELU → Dropout → Linear(hidden, num_seed_types)`, `hidden = feature_dim *
  hidden_ratio`. This is the plain MLP `g` that Eq. 5 specifies.
* **`variant="se_gated"`**: the repository's earlier, heavier block — adds a
  squeeze-excitation branch (`Linear → ReLU → Linear → Sigmoid` gating the
  hidden activation) and a scalar tanh/sigmoid feature gate before the final
  linear layer, plus a small stochastic-depth dropout. Preserved verbatim
  for ablation comparison (`PAPER_AUDIT.md` §2.6) but is **not** what the
  paper specifies — `p_s = softmax(s)` is computed identically by the caller
  regardless of variant.

## 3. `SeedTypeProjection` — `P` in Eq. 9 (`src/models/components/projections.py:85-141`)

$$
h' = h + P(p_s)
$$

```python
SeedTypeProjection(num_seed_types, embed_dim, depth=2, dropout=0.1, bottleneck_ratio=0.5)
```

Maps the 4-D probability vector into the 384-D feature space via `depth`
stages of `Linear → GELU → LayerNorm → Dropout`, followed by a final
`Linear(..., embed_dim)`. `depth=1` is the minimal MLP the paper's *"MLP
projection layer"* wording literally requires; the config default is
`depth=2` (`conf/model/head/hierarchical_moe.yaml:32`); the repository's
earlier head used `depth=3`. Middle-stage width is
`embed_dim * bottleneck_ratio` (only relevant when `depth > 1`).

Allocated only when `use_residual=True`
(`HierarchicalSeedClassifier.__init__`, `builder.py:267-277`); otherwise
`self.seed_projection = None` and the head fabricates `projected_seed =
torch.zeros_like(moe.features)` in `forward` so `HierarchicalOutput` stays
fully populated and the residual cosine loss term (see
[`05_LOSS_FUNCTIONS.md`](05_LOSS_FUNCTIONS.md)) evaluates to exactly 0
(`cos(h, h) = 1`) instead of `NaN`.

## 4. `CrossAttention` — Eqs. 11-12 (`src/models/components/cross_attention.py`)

$$
a = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V, \qquad Q = h',\ K = V = h
$$
$$
h'' = \text{LayerNorm}(a + Q)
$$

```python
CrossAttention(dim, num_heads=8, dropout=0.1, variant="paper", mlp_ratio=4)
```

**`variant="paper"`** (`cross_attention.py:112-113`) implements Eq. 12
exactly:

```python
attn_output, attn_weights = MultiheadAttention(dim, num_heads, batch_first=True)(query, key, value)
return CrossAttentionOutput(LayerNorm(attn_output + query), attn_weights)
```

Because the paper operates on a pooled feature **vector** rather than a patch
sequence, `query`/`key`/`value` are each shaped `[batch, 1, dim]` — a
key/value sequence of length 1. The softmax in `MultiheadAttention` therefore
has exactly one key to attend to and evaluates to weight `1.0`; cross-
attention consequently reduces to a learned linear projection of `h` plus a
`LayerNorm`-wrapped residual from `h'`. This is a faithful reading of the
paper's equations as written over pooled vectors — the same module would
attend non-trivially if a future variant kept per-patch tokens instead of
pooling.

**`variant="gated"`**: an alternative pre-norm block —
`query_norm(query)` before attention, then an adaptive sigmoid gate
(`AdaptiveGating`, `cross_attention.py:32-45`) blending the attention output
into the query, followed by a GELU feed-forward branch with its own residual.
Kept for ablation (`cross_attention_variant: "gated"`); not the paper's
design.

Allocated only when `use_cross_attention=True`; otherwise
`self.cross_attention = None` and `forward` sets `attended_features =
refined_features` directly (i.e., `h'' = h'`), skipping Eqs. 11-12 entirely.

## 5. `SubVarietyEmbedding` (`src/models/components/classifiers.py:93-163`)

Produces the embedding vector ArcFace measures angles in — the paper does
not specify its depth.

```python
SubVarietyEmbedding(feature_dim, dropout_rate=0.3, variant="mlp", use_highway=True, expansion=4)
```

* **`variant="mlp"`** (default): `LayerNorm` → a 4-layer bottleneck MLP
  (`feature_dim → wide → widest → wide → feature_dim`, `wide = feature_dim *
  expansion // 2`, `widest = feature_dim * expansion`, GELU + Dropout between
  each) → optionally blended with the pre-MLP input through a learned
  **highway gate** (`Linear → Sigmoid`): `embeddings = gate * mlp_out + (1 -
  gate) * x`.
* **`variant="identity"`**: feeds `h''` straight to ArcFace unchanged — the
  minimal reading when the paper leaves this network unspecified.

## 6. `ArcFaceHead` — Eq. 13 (`src/models/components/arcface_head.py`)

$$
\mathcal{L} = -\sum_i \log\frac{e^{s\cos(\theta_i + m)}}{e^{s\cos(\theta_i + m)} + \sum_{j \neq i} e^{s\cos\theta_j}}
$$

```python
ArcFaceHead(feature_dim, num_classes, scale=30.0, margin=0.5, easy_margin=False)
```

Owns the learnable class centres, `self.weight: [num_classes, feature_dim]`
(Xavier-uniform initialized) — this is what makes the head **stateless from
the loss's perspective**: `model.parameters()` alone is the complete
optimization set, and `model_state_dict` alone reproduces inference.

```python
def cosine_similarity(embeddings):
    return F.linear(F.normalize(embeddings, p=2, dim=1),
                     F.normalize(self.weight, p=2, dim=1)).clamp(-1+1e-7, 1-1e-7)

def forward(embeddings, labels=None):
    cosine = self.cosine_similarity(embeddings)
    logits = scale * cosine
    if labels is None:
        return logits, logits          # inference: no margin applied
    sine = sqrt((1 - cosine**2).clamp_min(1e-9))
    target_cosine = cosine * cos_m - sine * sin_m          # cos(theta + m), expanded — NOT via acos
    target_cosine = where(cosine > threshold, target_cosine, cosine - margin_penalty)  # or easy_margin fallback
    margin_cosine = one_hot(labels) * target_cosine + (1 - one_hot(labels)) * cosine
    return logits, scale * margin_cosine
```

Key implementation choice: `cos(\theta + m)` is computed via the
**angle-addition expansion** `\cos\theta\cos m - \sin\theta\sin m`, never via
`torch.acos`. `acos`'s gradient diverges as `\cos\theta \to \pm 1` — exactly
where a well-fit embedding sits — so the expansion is what keeps gradients
finite at convergence (`PAPER_AUDIT.md` §3.3). Beyond `\theta + m > \pi`,
`\cos(\theta+m)` stops being monotonic in `\theta`; the code falls back to the
standard linear penalty `\cos\theta - m\sin m` (`threshold = \cos(\pi - m)`,
`margin_penalty = \sin(\pi-m) \cdot m`), or to the unmodified cosine when
`easy_margin=True`.

**Two outputs, two purposes** (this is the field-level contract
`HierarchicalOutput` exposes):

* **`logits`** (no margin) — the correct quantity for prediction, ranking,
  and the KL hierarchy term. Never touched by the target-class margin.
* **`margin_logits`** — margin applied only to the target class's cosine,
  and only when `labels` is not `None`. **Equal to `logits` at inference**
  (no labels passed), so evaluation metrics are never inflated by the
  training-time margin. Consumed only by the ArcFace cross-entropy.

Default hyperparameters (`conf/model/head/hierarchical_moe.yaml:35-37`):
`arcface_scale: 30.0`, `arcface_margin: 0.5` (radians), `arcface_easy_margin:
false`.

## 7. `LinearSubVarietyHead` — the `use_arcface=False` ablation (`classifiers.py:165-195`)

```python
def forward(self, embeddings, labels=None):
    logits = self.classifier(embeddings)     # plain nn.Linear
    return logits, logits                     # both outputs identical -- no margin to apply
```

Mirrors `ArcFaceHead`'s `(logits, margin_logits)` return signature exactly,
so `HierarchicalSeedClassifier.forward` needs **no branch** for this
ablation. Because both returned tensors are the same object, the combined
loss's `arcface_loss(sub_margin_logits, labels)` term degrades to exactly
plain categorical cross-entropy — with zero changes to the loss code (see
[`05_LOSS_FUNCTIONS.md`](05_LOSS_FUNCTIONS.md) and
[`06_ABLATION_ENGINE.md`](06_ABLATION_ENGINE.md)).

## 8. `input_projection` — the Eq. 4 collapse to identity inside the head

```python
if self.feature_dim == self.embed_dim and input_projection_hidden_dim is None:
    self.input_projection = nn.Identity()
else:
    self.input_projection = EmbeddingProjection(...)
```

(`builder.py:229-238`) With `DinoV2SwinV2Encoder` upstream (which already
projects to 384-D `z`), `feature_dim == embed_dim == 384` and this collapses
to a true identity costing nothing. The projection is retained only for
callers that feed the head raw, unprojected backbone features directly (e.g.
notebooks, feature dumps) — see
[`02_BACKBONE_AND_SSL.md`](02_BACKBONE_AND_SSL.md) §3 for why the projection
lives in the encoder rather than here.

## 9. Component toggle summary

Full detail and the ablation/baseline suite mechanics are in
[`06_ABLATION_ENGINE.md`](06_ABLATION_ENGINE.md); the structural effect on
this cascade is:

| Flag | `False` effect | Attribute set to |
| --- | --- | --- |
| `use_moe` | `DenseExpertBlock` (one dense block) replaces the Top-K router | `self.moe` (always allocated, different class) |
| `use_arcface` | `LinearSubVarietyHead` replaces `ArcFaceHead`; objective becomes plain CE | `self.arcface` (always allocated, different class) |
| `use_residual` | `h' = h`; Eq. 9 fusion not built | `self.seed_projection = None` |
| `use_cross_attention` | `h'' = h'`; Eqs. 11-12 skipped | `self.cross_attention = None` |

`component_flags()` (`builder.py:321-328`) reports the four architectural
booleans (`use_kl_loss` is not one of them — it lives on the head config only
for interpolation convenience but is consumed entirely by the loss builder).
