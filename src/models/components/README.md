# `src/models/components/` — building blocks

Each file implements one numbered piece of paper Section 5. All are independently
constructible and independently tested (`tests/test_components.py`,
`tests/test_moe_layer.py`).

| File | Paper | Class |
| --- | --- | --- |
| `moe_layer.py` | Eq. 8 | `MixtureOfExperts`, `DenseExpertBlock`, `TransformerExpert`, `MoEOutput` |
| `projections.py` | Eqs. 4, 9 | `EmbeddingProjection`, `SeedTypeProjection` |
| `classifiers.py` | Eqs. 5-6, 13 | `SeedTypeClassifier`, `SubVarietyEmbedding`, `LinearSubVarietyHead` |
| `cross_attention.py` | Eqs. 11-12 | `CrossAttention`, `CrossAttentionOutput` |
| `arcface_head.py` | Eq. 13 | `ArcFaceHead` |

## `MixtureOfExperts` — Eq. 8

Six transformer experts, **Top-2** routing (the submitted manuscript used Top-4;
`DEFAULT_TOP_K` is the single place that value is defined):
`h = Σ_{i∈Top-K} Gᵢ Eᵢ(z)`.

The gate is a linear projection plus softmax over experts. Top-K weights are
renormalised so the selected experts form a convex combination. Returns
`MoEOutput` with five fields: `features`, `gate_probs` (the full distribution,
needed by the load-balancing term), `top_k_indices`, `top_k_weights`, and
`dispatch_weights` (the Top-K weights scattered back over all experts).

`sparse_dispatch=True` runs each expert only on the rows routed to it. A test
asserts it produces identical output to dense evaluation — it is purely an
optimisation, not a semantic change.

Input is `[batch, embed_dim]`: the paper works on a pooled feature vector, so
every sample is a single token.

### Efficiency accounting

`parameters_per_expert()` and `dormant_parameters()` give
`src/utils/efficiency.py` a closed form for the active parameter count: all
experts share one architecture, so counting the first and multiplying is exact.
`dormant = (E − K) × parameters_per_expert`, which doubles when `K` drops from
4 to 2 — the revision's efficiency claim, stated as an identity rather than a
measurement.

### Gradients and sparse routing

An expert that no sample in a batch routed to receives **no gradient that step**.
That is the defining property of a sparse MoE, but it is far more visible at
`K = 2` than at `K = 4`: a batch of 12 fills only 24 routing slots across six
experts. The entropy load-balancing term pulls utilisation back toward uniform
over an epoch, so no expert stays unrouted for long. Tests assert the precise
invariant — routed ⇔ has gradient — rather than the false claim that every expert
always learns.

## `DenseExpertBlock` — the `use_moe=False` ablation

One always-on transformer block, architecturally identical to a single expert.
Deleting the experts outright would also delete a block's worth of capacity, so
the `wo_moe` gap would confound *routing* with *depth*; keeping one dense block
means the gate is the only removed ingredient.

It returns a `MoEOutput` describing a one-expert router that always selects its
single expert with weight 1. Both MoE regularisers evaluate to exactly zero on
that gate — the entropy term because a one-dimensional distribution has no
entropy to spread, the sparsity term because no mass falls outside the selection
— so no downstream caller needs a special case.

## `CrossAttention` — Eqs. 11-12

`a = softmax(QKᵀ/√d)V`, then `h'' = LayerNorm(a + Q)`, with `Q = h'` and
`K = V = h`.

Because the paper operates on a pooled *vector*, the key/value sequence has
length 1 and the softmax is over a single key — so it evaluates to weight 1.0 and
the block reduces to a learned projection of `h` plus a residual from `h'`. That
is a faithful reading of the paper, and the machinery generalises unchanged if a
future variant keeps the patch tokens.

`variant="paper"` implements Eq. 12 exactly. `variant="gated"` is the earlier
pre-norm block with an adaptive gate and feed-forward branch, kept for ablation.

## `ArcFaceHead` — Eq. 13

Additive angular margin over L2-normalised embeddings, owning the learnable class
centres. Returns `(logits, margin_logits)`; see the `models/` README for which to
use where.

`cos(θ + m)` is expanded as `cos θ cos m − sin θ sin m` rather than routed
through `acos`, whose gradient diverges as `cos θ → ±1` — precisely where a
well-fit embedding sits. For `θ + m > π` the true cosine stops being monotonic in
`θ`; ArcFace's standard linear fallback (`cos θ − m sin m`) covers that region,
or `easy_margin` falls back to the unmodified cosine.

## `projections.py` — Eqs. 4, 9

`EmbeddingProjection` maps the backbone's output width to the paper's
`z ∈ ℝ³⁸⁴`. It lives inside `DinoV2SwinV2Encoder`, which is what makes
`encoder(images).shape[-1] == 384` an invariant for every SwinV2 variant. Inside
the head it is bypassed entirely (`nn.Identity`) when the input is already `z`.

`SeedTypeProjection` is `P` from Eq. 9, lifting the 4-D probability vector `p_s`
into the 384-D feature space so it can be added onto the MoE output. The paper
calls it "an MLP projection layer" without specifying depth, so `depth` is
configurable.

## `classifiers.py` — Eqs. 5-6

`SeedTypeClassifier` is `g` from Eq. 5. `variant="mlp"` is the paper's plain MLP
(default); `variant="se_gated"` preserves the repository's earlier
squeeze-excitation + gated block for ablation.

`SubVarietyEmbedding` produces the embedding ArcFace measures angles in. The
paper does not specify this network, so `variant="identity"` (feed `h''` straight
to ArcFace) sits alongside the default residual/highway MLP.

`LinearSubVarietyHead` is the `use_arcface=False` ablation. It mirrors
`ArcFaceHead`'s `(logits, margin_logits)` return so the model body needs no
branch, and because there is no margin both outputs are the same tensor — which
makes the combined objective's ArcFace term degrade to exactly the categorical
cross-entropy the ablation calls for, with no change to the loss code.
