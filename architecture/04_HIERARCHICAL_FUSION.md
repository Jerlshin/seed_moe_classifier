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

## 3. Eq. 9 fusion — additive residual, or FiLM

$$
h' = h + \gamma \odot P(p_s)
$$

```python
SeedTypeProjection(num_seed_types, embed_dim, depth=2, dropout=0.1, bottleneck_ratio=0.5)
LayerScale(embed_dim, init_value=1e-4)
```

`P` maps the 4-D probability vector into the 384-D feature space via `depth`
stages of `Linear -> GELU -> LayerNorm -> Dropout`, followed by a final
`Linear(..., embed_dim)`. `depth=1` is the minimal MLP the paper's *"MLP
projection layer"* wording literally requires; the config default is `depth=2`.

### `LayerScale`, and why the control moved out of the loss

`gamma` is a learned per-channel gain initialised at `1e-4` (Touvron et al.,
2021), so the branch starts inert and has to earn its magnitude.

This replaces the loss-side attempt to control the residual. `L_cos =
1 - cos(h + P(p_s), h)` is minimised by `P(p_s) = alpha.h` — **including
`alpha = 0`, which is literally the `use_residual=False` ablation**. Cosine is
magnitude-invariant, so it constrained the residual's direction and never how
much it shifted, and the cheapest way to preserve direction is to make the
residual vanish. A learned gain has no such fixed point, and the optimiser can
raise it. See [`05_LOSS_FUNCTIONS.md`](05_LOSS_FUNCTIONS.md) §6.

`wo_layer_scale` ablates the gain, reproducing the submitted free residual.

### Why the additive form degenerates, and what FiLM does about it

Two coupled defects, both consequences of conditioning on `p_s`:

**At convergence Eq. 9 is a 4-entry codebook.** `p_s` lives on the 3-simplex.
Eq. 7 supervises the coarse head with hard labels over only 4 classes, so it fits
early and hard and `p_s -> e_c` one-hot. Then `P(p_s) -> P(e_c) in {v1, v2, v3, v4}`
— four fixed vectors. Whatever `depth` is set to, the *function class actually
realised at convergence* is a 4-row table implemented by a ~75 k-parameter MLP.

**And the gradient dies with it.** The softmax Jacobian is
`d p_s / d s = diag(p) - p p^T`, whose norm goes to 0 as `p` approaches one-hot.
So the sub-variety branch's gradient into the seed-type classifier **vanishes
exactly when the seed head becomes confident**. The claim of joint optimisation
between stages holds only during the transient; after that, `L_seed` and `L_KL`
are the only live paths into stage 1, and the KL term flows through softmax
outputs too, so it does not rescue this.
`tests/test_models.py::test_additive_fusion_gradient_vanishes_as_the_coarse_head_saturates`
measures the collapse at >1000x.

**`fusion_mode: film`** conditions on the seed classifier's **pre-softmax hidden
state** (192-D at `hidden_ratio=0.5`) and modulates multiplicatively:

$$
(\gamma, \beta) = \text{MLP}(g_{\text{hidden}}(z)), \qquad h' = \gamma \odot h + \beta
$$

* **No saturation.** The Jacobian does not collapse as the coarse head becomes
  confident, so the fine loss keeps informing the coarse branch for the whole run.
* **Capacity.** `2 x 384` dimensions modulated by a 192-D state, not 4 discrete
  choices.
* **Strictly more expressive.** `gamma = 1` recovers the additive form exactly,
  and the modulation MLP's last layer is zero-initialised so it *starts* there —
  the `additive` and `film` variants therefore begin as the same function and the
  ablation between them measures capacity rather than initialisation.
* **Bounded by construction.** `gamma = 1 + tanh(.)` lies in `(0, 2)`, so
  magnitude cannot track confidence. This preserves `PAPER_AUDIT.md` §2.2's
  invariant in spirit — its rule was "project probabilities, not logits, because
  logits are unbounded and would make the residual scale with confidence"; FiLM
  addresses the same hazard structurally.

`p_s` is still computed and exposed in `HierarchicalOutput` regardless: the KL
term and the alignment metric both need it.

### When the residual is disabled

`self.seed_projection = None` (and `self.film = None`), and `forward` fabricates
`projected_seed = torch.zeros_like(moe.features)` so `HierarchicalOutput` stays
fully populated.

Note what this used to do to the objective. Under the submitted
`cosine_mode="residual"`, `L_cos = 1 - cos(h, h) = 0` **identically, for every
sample, for the whole run** — so `wo_residual` silently removed Eq. 9 *and* the
paper's entire Section-1 contribution in one toggle, and the measured gap could
not attribute to either. With `intra_class` compactness the term is a property of
the ArcFace embedding and survives the toggle; what legitimately vanishes is the
residual magnitude hinge, which has no residual left to bound.

## 4. `CrossAttention` — Eqs. 11-12 (`src/models/components/cross_attention.py`)

$$
a = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V, \qquad Q = h',\ K = V = h
$$
$$
h'' = \text{LayerNorm}(a + Q)
$$

```python
CrossAttention(dim, num_heads=8, dropout=0.1, variant="paper", mlp_ratio=4,
               mode=resolve_attention_mode(token_mode))
```

### Sequence length is load-bearing

Over a key/value sequence of length 1 — which is what a pooled feature *vector*
gives you:

```text
softmax(QK^T / sqrt(d)) = softmax([s]) = [1]      for any scalar s
=> a = 1 . V = W_O(W_V x)                          -- Q and K appear nowhere
=> da/dW_Q = da/dW_K = 0                           -- exactly zero gradient, forever
```

An earlier version of this document stated the reduction honestly, and then did
not carry the consequences through. They are:

* **295,680 parameters per instance are unreachable.** Seven instances (six
  experts plus this block) is **~2.07 M** — counted in `ParameterReport.total`
  *and* in `active = total - dormant`, so the paper's "Total Params (M)" and
  "Active Params (M)" columns both included them.
* **`num_heads=8` is inert.** Splitting a single token across 8 heads is a no-op.
* **`attn_weights` is identically 1.0.** Any attention-map figure derived from it
  is a constant image showing nothing.

### The two modes

`mode` is chosen by `resolve_attention_mode(token_mode)`, so the head and the
attention block cannot be configured inconsistently.

| `token_mode` | `mode` | Behaviour |
| --- | --- | --- |
| `grid` | `attention` | real `MultiheadAttention` over 64 keys; heads matter; `attn_weights` is a plottable `8x8` spatial map |
| `pooled` | `affine` | a single `nn.Linear` — the exact function class a length-1 attention realises — and `attn_weights` returns `None` rather than a misleading constant |

`mode="affine"` raises if handed more than one key/value token, so a
misconfiguration fails loudly instead of silently discarding structure.

`tests/test_components.py` verifies the degeneracy directly rather than arguing
it: one backward leaves the Q/K gradient at float dust against a non-zero V
gradient, and `attn_weights` comes back as the constant 1.0.

**`variant="gated"`**: an alternative pre-norm block — `query_norm(query)` before
attention, then an adaptive sigmoid gate (`AdaptiveGating`) blending the attention
output into the query, followed by a GELU feed-forward branch with its own
residual. Kept for ablation; not the paper's design. It wraps the same attention
module, so it inherits whichever `mode` the head resolved.

Allocated only when `use_cross_attention=True`; otherwise
`self.cross_attention = None` and `forward` sets `attended_features` from
`refined_features` directly (`h'' = h'`), skipping Eqs. 11-12 entirely.

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

### The feature scale, revised

The submitted configuration used `arcface_scale: 30.0` — the value ArcFace tuned
for face recognition over 10^5 to 10^6 identities. AdaCos (Zhang et al., CVPR
2019) derives the fixed optimal scale for a `C`-class cosine-softmax as

```text
s = sqrt(2) . log(C - 1) = 1.4142 . ln 26 = 4.61      for C = 27
```

so `s = 30` was **6.5x too large for this problem**, with three measurable
consequences:

1. **Initial loss imbalance.** For random 384-D embeddings `cos(theta) ~ 0`
   (std ~ 1/sqrt(384) = 0.051). With `m = 0.5` the target logit is
   `30 . (-sin 0.5) = -14.38`, so `L_ArcFace = log(1 + 26 e^14.38) = 17.64`
   against `L_seed = ln 4 = 1.386` — a **12.7 : 1** ratio at equal lambda. The
   angular-margin term consumed essentially the whole gradient budget early on.
2. **A saturated KL branch.** `softmax(30 cos theta)` is near-one-hot by
   construction, which is what made the hierarchy term's `clamp -> log`
   composition gradient-dead (see [`05_LOSS_FUNCTIONS.md`](05_LOSS_FUNCTIONS.md)
   §4) and what secretly coupled `lambda_kl` to `arcface_scale`.
3. **Meaningless calibration.** `sub_scores` in `test_predictions.npz` are
   `softmax(30 cos theta)` — extremely overconfident by construction. ECE is now
   reported with a temperature fitted on validation.

`arcface_scale: "auto"` resolves to the AdaCos value. `arcface_dynamic: true`
instead re-derives `s` each step from the running median target angle (AdaCos
proper), removing the hyperparameter rather than retuning it.

### Margin warm-up

Applying the full margin from step 0 is a documented convergence hazard on small
backbones and small datasets — CurricularFace reports outright divergence (NaN at
~2,400 steps) for MobileFaceNet at `m = 0.5` on CASIA-WebFace, converging at
`m = 0.45`. The trainer calls `set_margin_scale()` once per epoch to ramp
`m: 0 -> margin` over the first `margin_warmup_fraction` (default 15 %) of
training. At scale 0 the head returns its plain logits, so it *is* a
cosine-softmax rather than an approximation of one.

### Sub-centres

`arcface_sub_centers: K` gives each class `K` prototypes and takes the maximum
cosine over them (Deng et al.). Built for exactly the case where one prototype
cannot explain intra-class variability — plausible if a "sub-variety" spans
multiple growing conditions or imaging sessions, which the provenance data
(81 photographs, 1-5 per class) makes possible. Available and tested; not a suite
variant, because there is no evidence of label noise here yet.

Defaults (`conf/model/head/hierarchical_moe.yaml`): `arcface_scale: "auto"`,
`arcface_margin: 0.5` (radians), `arcface_easy_margin: false`,
`arcface_sub_centers: 1`, `arcface_dynamic: false`.

## 7. Three sub-variety heads, and what each ablation actually measures

`sub_head_variant` selects among them.

| | `arcface` | `normface` | `linear` |
| --- | --- | --- | --- |
| Embedding | L2-normalised | L2-normalised | unnormalised |
| Class centres | L2-normalised | L2-normalised | unnormalised `nn.Linear` |
| Logit range | `[-s, s]` | `[-s, s]` | unbounded |
| Target margin | `cos(theta + m)` | none | none |

The submitted suite called the `linear` substitution `wo_arcface` and read the
gap as a margin measurement. It is not. That swap simultaneously removes the
margin, removes hypersphere normalisation on **both** sides, and changes the
softmax temperature by an unbounded factor — which in turn changes the sharpness
of `P_sub` feeding the KL term and the geometry the t-SNE panels visualise.

`NormFaceHead` is the single-factor control: normalised embedding and centres,
`m = 0`, same scale. So:

| Variant | Head | Measures |
| --- | --- | --- |
| `wo_margin_only` | `normface` | the angular margin, and only that |
| `wo_angular_head` | `linear` | margin + normalisation + logit scale (honestly named) |

All three return `(logits, margin_logits)`, so `HierarchicalSeedClassifier.forward`
needs **no branch** and the combined loss's `arcface_loss(sub_margin_logits, ...)`
degrades to plain categorical cross-entropy on the two margin-free heads with zero
changes to the loss code.

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

Full detail and the suite mechanics are in
[`06_ABLATION_ENGINE.md`](06_ABLATION_ENGINE.md), including the table of what
each toggle *actually* changes. The structural effect on this cascade:

| Flag | `False` effect | Attribute set to |
| --- | --- | --- |
| `use_moe` | `DenseExpertBlock` replaces the Top-K router | `self.moe` (always allocated, different class) |
| `use_arcface` | resolves `sub_head_variant` to `linear` unless set explicitly | `self.arcface` (always allocated, different class) |
| `use_residual` | `h' = h`; Eq. 9 fusion not built | `self.seed_projection`, `self.residual_scale`, `self.film` all `None` |
| `use_cross_attention` | `h'' = h'`; Eqs. 11-12 skipped | `self.cross_attention = None` |

`component_flags()` no longer reports only those four booleans. It reports every
axis a variant can move — `token_mode`, `fusion_mode`, `sub_head_variant`,
`router_mode`, `gate_conditioning`, `num_experts`, `top_k`,
`dense_capacity_multiplier` — and the criterion contributes a parallel
`loss_flags()` block. Reporting only the four architectural booleans made a
`wo_kl` run **byte-identical to `full_model`** in `summary.json`, with only the
variant *name* separating them, which is precisely the kind of implicit
difference this repository otherwise refuses to tolerate.
