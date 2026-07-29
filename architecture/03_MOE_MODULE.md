# 03 — Mixture-of-Experts Module

Covers `src/models/components/moe_layer.py` and `src/losses/moe.py`.
Implements paper Section 5.2, Eq. 8.

## 1. Architecture

```python
DEFAULT_NUM_EXPERTS = 6   # Section 5.2
DEFAULT_TOP_K = 2         # revision; submitted manuscript used 4 (SUBMITTED_TOP_K in tests/conftest.py)
```

`src/models/components/moe_layer.py:32-33` — this is the **single place**
routing width is defined; `src/utils/efficiency.py` quantifies the resulting
active-parameter saving from these two constants.

### `TransformerExpert` (`moe_layer.py:55-79`)

Each of the six experts is an identical post-norm transformer block operating
on a **length-1 token sequence** (the paper works on a pooled feature vector,
so every sample is a single token, not a patch sequence):

```python
attn_output, _ = MultiheadAttention(embed_dim, num_heads=8, batch_first=True)(x, x, x)
x = LayerNorm(x + attn_output)
mlp_out = Linear(embed_dim, mlp_dim) → GELU → Dropout → Linear(mlp_dim, embed_dim) → Dropout
return LayerNorm(x + mlp_out)
```

`mlp_dim=512`, `num_heads=8` by default (`model.head.moe_hidden_dim`,
`model.head.num_heads`).

### `MixtureOfExperts` (`moe_layer.py:82-210`)

$$
h = \sum_{i \in \text{Top-}K} G_i \, E_i(z), \qquad h \in \mathbb{R}^{d}
$$

```python
MixtureOfExperts(embed_dim, num_experts=6, mlp_dim=512, top_k=2, num_heads=8,
                  dropout=0.0, renormalize_top_k=True, sparse_dispatch=True)
```

`forward(x)` (`moe_layer.py:126-149`), `x: [batch, embed_dim]`:

```python
gate_probs = softmax(Linear(embed_dim, num_experts)(x), dim=-1)     # G, [batch, num_experts]
top_k_weights, top_k_indices = torch.topk(gate_probs, top_k, dim=-1)
if renormalize_top_k:
    top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
dispatch_weights = zeros_like(gate_probs).scatter(1, top_k_indices, top_k_weights)
```

Renormalizing the Top-K weights means the selected experts form a **convex
combination** restricted to the selection — their weights sum to exactly 1
even though the full `gate_probs` simplex sums to 1 across all six.

`MoEOutput` (`moe_layer.py:36-53`, a `NamedTuple`) carries `features` (`h`),
`gate_probs` (`G`, full distribution — needed by the load-balancing term),
`top_k_indices`, `top_k_weights`, and `dispatch_weights` (`top_k_weights`
scattered back over all six experts, zero elsewhere).

### Sparse vs. dense dispatch

`sparse_dispatch=True` (default) runs `_sparse_forward` (`moe_layer.py:151-168`):
for each expert index, gather only the rows of the batch actually routed to
it (`rows = selection[:, expert_index].nonzero(...)`), run the expert on just
those rows, and scatter-add the weighted output back. Experts that no sample
routed to are **skipped entirely** — this is the computational point of a
sparse MoE.

`sparse_dispatch=False` runs `_dense_forward` (`moe_layer.py:170-173`):
every expert runs on every row, and `dispatch_weights` (zero for unselected
experts) masks the contribution. Semantically identical, just wasteful — kept
for debugging, since it keeps a gradient flowing to unselected experts.
`tests/test_moe_layer.py` asserts the two modes produce identical output.

### Gradients under sparse dispatch

**An expert that no sample in a batch routed to receives no gradient that
step.** This is the defining property of a sparse MoE, not a defect — but it
is far more visible at `K=2` than the submitted `K=4`: a batch of 12 samples
fills only `12 × 2 = 24` routing slots across six experts, versus `12 × 4 =
48` at Top-4. `MoERegularization`'s entropy term (§3 below) is what pulls
utilization back toward uniform over an epoch, so no expert stays unrouted
for long. `tests/test_models.py::test_only_the_routed_experts_receive_gradient`
pins the exact invariant: **routed ⇔ has gradient** — not the false claim
that every expert learns every step.

### Efficiency accounting hooks

```python
parameters_per_expert()  # sum of one expert's parameter count — all experts share
                          # one architecture, so counting the first is exact
dormant_parameters()     # (num_experts - top_k) * parameters_per_expert()
```

(`moe_layer.py:188-204`) These give `src/utils/efficiency.py` a **closed
form**, not an estimate, for the active-parameter count — see
[`07_EFFICIENCY_AND_EVALUATION.md`](07_EFFICIENCY_AND_EVALUATION.md). Halving
`top_k` from 4 to 2 exactly doubles `dormant_parameters()`.

`expert_utilization(top_k_indices)` (`moe_layer.py:175-184`) returns the
fraction of the batch routed to each expert — a `[num_experts]` tensor
summing to 1, used for the utilization bar chart
(`plot_expert_utilization`).

## 2. `DenseExpertBlock` — the `use_moe=False` ablation (`moe_layer.py:213-268`)

A single always-on `TransformerExpert`, architecturally identical to one MoE
expert. It returns a `MoEOutput` describing a degenerate one-expert router
(`gate_probs = ones`, `top_k_indices = zeros`, weight 1 always):

```python
num_experts = 1
top_k = 1
dormant_parameters() -> 0   # always: a dense block activates everything it owns
```

The design rationale (`REVISION_NOTES.md` §4): deleting the experts outright
would *also* delete a transformer block's worth of capacity, confounding
*routing* with *depth*. Keeping one dense block of identical architecture
means the `wo_moe` ablation's measured gap against the full model is
attributable to the gate alone. Both MoE regularizers (§3) evaluate to
exactly zero on this degenerate gate — the entropy term because a
one-dimensional distribution has no entropy to spread, the sparsity term
because no mass ever falls outside a Top-1-of-1 selection — so no downstream
consumer (losses, metrics, trackers) needs a special case for this ablation.

## 3. MoE regularization losses (`src/losses/moe.py`)

Paper Section 5.2 introduces two terms *"to prevent expert collapse and
encourage balanced expert utilization."*

### Load-balancing loss

$$
u_i = \text{mean}_{\text{batch}}(G_i), \qquad
\mathcal{L}_{\text{load}} = -\frac{H(u)}{\log(E)} = \frac{\sum_i u_i \log u_i}{\log(E)}
$$

```python
def load_balancing_loss(gate_probs, normalize=True):
    utilization = gate_probs.mean(dim=0)                       # u, [num_experts]
    negative_entropy = sum(utilization * log(utilization.clamp_min(1e-8)))
    return negative_entropy / log(num_experts)  if normalize else negative_entropy
```

(`moe.py:57-74`) Minimizing negative entropy *maximizes* entropy, which
spreads utilization evenly across experts. The `log(E)` normalization
(`normalize_moe_entropy: true`, the default) bounds the term in **`[-1, 0]`**
— `-1` = perfectly uniform utilization, `0` = total collapse onto one expert
— so a chosen `lambda_moe_load` keeps the same meaning if `num_experts`
changes (`PAPER_AUDIT.md` §3.2 records that an earlier version left this
unnormalized, so the weight silently changed meaning with expert count).

### Sparsity (Top-K concentration) loss

$$
\mathcal{L}_{\text{sparsity}} = \text{mean}_{\text{batch}}\Big(\sum_{i \notin \text{Top-}K} G_i\Big) = \text{mean}\big(1 - \textstyle\sum_{i \in \text{Top-}K} G_i\big) \in [0, 1]
$$

```python
def l1_sparsity_loss(gate_probs, top_k_indices, mode="off_topk"):
    selected_mass = torch.gather(gate_probs, 1, top_k_indices).sum(dim=-1)
    if mode == "topk":
        return torch.gather(gate_probs, 1, top_k_indices).abs().mean()   # legacy, discouraged
    return (1.0 - selected_mass).clamp_min(0.0).mean()                    # default: off_topk
```

(`moe.py:77-101`) **`mode="off_topk"` (default)** penalizes routing mass that
lands *outside* the Top-K selection — driving it to zero concentrates all
routing mass on the K selected experts, which is a literal reading of the
paper's *"restricts the selection to only the top-K most relevant experts."*

`mode="topk"` reproduces an earlier, incorrect implementation
(`PAPER_AUDIT.md` §3.1) that penalized the *selected* experts' gate weights
directly — since the gate is a softmax whose total L1 norm is identically 1,
this instead pushes the chosen experts' confidence **down**, directly
fighting the load-balancing term rather than complementing it. Kept only for
ablation comparison via `moe_sparsity_mode: "topk"`.

**The two terms deliberately pull in opposite directions** — load-balancing
wants the *batch* spread evenly across experts; sparsity wants each *sample*
routed decisively (little mass outside its Top-K). That tension is what
yields experts that are both specialized and fully utilized. At `K=2`, four
of six experts are discarded per sample (versus two at the submitted `K=4`),
so the sparsity term operates on a larger quantity, and each batch fills half
as many routing slots — making load-balancing's pull toward uniformity even
more important for keeping an under-used expert from being abandoned.

### `MoERegularization` (`moe.py:104-149`)

```python
MoERegularization(lambda_load=0.01, lambda_sparsity=0.01,
                   normalize_entropy=True, sparsity_mode="off_topk")
```

`forward(gate_probs, top_k_indices) -> MoERegularizationOutput(total, load_balancing,
sparsity, utilization)`, where `total = lambda_load * load + lambda_sparsity * sparsity`.
`utilization` (detached) is exposed for the utilization bar chart. This
module is what `CombinedHierarchicalLoss` calls as `self.moe_regularization`
(see [`05_LOSS_FUNCTIONS.md`](05_LOSS_FUNCTIONS.md)).

Config defaults (`conf/model/loss/arcface_kl.yaml:21-22`):
`lambda_moe_load: 0.01`, `lambda_moe_sparsity: 0.01`.

## 4. Where the MoE sits in the head

`HierarchicalSeedClassifier.__init__` (`src/models/builder.py:248-265`)
builds either `MixtureOfExperts` or `DenseExpertBlock` under the attribute
name `self.moe`, selected by `use_moe`:

```python
self.moe = MixtureOfExperts(embed_dim, num_experts, moe_hidden_dim, top_k, num_heads,
                             dropout_rate, sparse_dispatch=moe_sparse_dispatch) if use_moe \
      else DenseExpertBlock(embed_dim, moe_hidden_dim, num_heads, dropout_rate)
```

In `forward` (`builder.py:353`): `moe = self.moe(embedding)` — **the MoE is
routed on `z`, the DINO embedding**, per Eq. 8. This is one of three dataflow
facts the paper audit flags as easy to get wrong: an earlier version of this
codebase routed on a projection of the 4-D seed-type logits instead, meaning
the experts never saw the image at all (`PAPER_AUDIT.md` §2.1). See
[`04_HIERARCHICAL_FUSION.md`](04_HIERARCHICAL_FUSION.md) for the full
cascade this feeds into.
