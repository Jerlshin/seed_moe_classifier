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

### Routing granularity is the head's most consequential setting

`token_mode` decides what a "routing token" is, and four things follow from it.

| | `pooled` (submitted) | `grid` (revision default) |
| --- | --- | --- |
| Input to the MoE | one pooled vector per image | SwinV2's final-stage `8x8` grid |
| Routing slots per step (batch 16, K=2) | 32 | 2,048 |
| Expert self-attention | **provably affine** — see below | genuine, over 64 keys |
| `num_heads` | inert | meaningful |
| Fine-grained evidence | mean-pooled away before the head | preserved through the head |

Mean-pooling a 64-token grid into one vector *before* a 27-way fine-grained task
discards precisely the localised texture that separates rice sub-varieties.
Pooling now happens **after** the head, via `TokenPooling` (`attention`, `gem` or
`mean`).

### `TransformerExpert` — and why length-1 attention is not attention

With a key/value sequence of length 1:

```text
softmax(QK^T / sqrt(d)) = softmax([s]) = [1]      for any scalar s
=> a = 1 . V = W_O(W_V x)                          -- Q and K appear nowhere
=> da/dW_Q = da/dW_K = 0                           -- exactly zero gradient, forever
```

So under a pooled input, each "transformer expert" is `LayerNorm(x + Linear(x))
-> MLP -> LayerNorm`: a 2-block MLP. That is a perfectly reasonable architecture.
It is not a reasonable *description*, and the consequences were not being carried
through to the numbers.

For `nn.MultiheadAttention(embed_dim=384, num_heads=8)` the packed
`in_proj_weight` is `[3.384, 384]` with `in_proj_bias [3.384]`, so the Q and K
slices are `2.384^2 + 2.384 = 295,680` parameters:

| Module | Instances | Unreachable parameters |
| --- | --- | --- |
| `TransformerExpert` | 6 | 1,774,080 |
| `CrossAttention` | 1 | 295,680 |
| **Total** | | **~2.07 M** |

Those were counted in `ParameterReport.total` **and** in
`active = total - dormant`, so the paper's "Total Params (M)" and "Active Params
(M)" columns both included them. Of the 3.95 M "saved" by moving Top-4 to Top-2,
roughly 1.18 M was never doing anything.

`TransformerExpert` therefore takes a `token_mixing` mode:

```python
# token_mixing="attention"  (token_mode="grid")
mixed, _ = MultiheadAttention(embed_dim, num_heads, batch_first=True)(x, x, x)

# token_mixing="affine"     (token_mode="pooled")
mixed = Linear(embed_dim, embed_dim)(x)     # spans the identical function class

x = LayerNorm(x + mixed)
return LayerNorm(x + MLP(x))
```

`W_O(W_V x + b_V) + b_O` is affine, so one `Linear` reproduces the pooled block's
function exactly, at 147,840 parameters instead of 591,360. Nothing unreachable
is built, and nothing unreachable is counted.

`tests/test_components.py::test_length_one_attention_is_provably_affine` verifies
the degeneracy directly: one backward leaves the Q/K gradient at float dust
against a non-zero V gradient, and `attn_weights` comes back as the constant 1.0.

### `MixtureOfExperts`

$$
h = \sum_{i \in \text{Top-}K} G_i \, E_i(z), \qquad h \in \mathbb{R}^{d}
$$

```python
MixtureOfExperts(embed_dim, num_experts=6, mlp_dim=512, top_k=2, num_heads=8,
                 dropout=0.0, renormalize_top_k=True, sparse_dispatch=True,
                 token_mixing="affine", router_mode="learned",
                 gate_condition_dim=0, noise_std=0.0, gate_init_std=1e-3)
```

`forward(x, gate_condition=None)` accepts `[batch, embed_dim]` or
`[batch, tokens, embed_dim]`; tokens are flattened to `[batch*tokens, embed_dim]`
and routed independently, then reshaped back.

```python
gate_logits = self._router_logits(flat, condition)          # noisy in training only
gate_probs  = softmax(gate_logits, dim=-1)                   # G, [tokens, num_experts]
top_k_weights, top_k_indices = torch.topk(gate_probs, top_k, dim=-1)
if renormalize_top_k:
    top_k_weights = top_k_weights / top_k_weights.sum(-1, keepdim=True).clamp_min(1e-8)
dispatch_weights = zeros_like(gate_probs).scatter(1, top_k_indices, top_k_weights)
```

Renormalizing means the selected experts form a **convex combination** restricted
to the selection. Note the consequence for the sparsity term in §3: it also means
an L1 penalty on the off-Top-K mass cannot change the module's output.

`MoEOutput` carries `features` (`h`, reshaped to the input's shape), `gate_logits`
(**pre-softmax**, required by the router z-loss), `gate_probs`, `top_k_indices`,
`top_k_weights`, `dispatch_weights`, and `tokens_per_sample` — so consumers
wanting per-image routing statistics reshape rather than assuming one row per
image.

### Sparse vs. dense dispatch — equal forward, unequal optimizer

`sparse_dispatch=True` (default) gathers only the rows routed to each expert,
runs the expert on those, and scatter-adds the weighted output back. Experts no
token reached are skipped entirely — the computational point of a sparse MoE.

`sparse_dispatch=False` runs every expert on every row and masks with
`dispatch_weights`. The two produce **identical output**, and
`tests/test_moe_layer.py` asserts that.

They were **not** identical under AdamW, and forward-equality hid it:

* Sparse: an unrouted expert never enters the autograd graph, so `p.grad is None`.
  PyTorch's AdamW skips such parameters **entirely** — including decoupled weight
  decay and the moment-buffer updates. (`zero_grad(set_to_none=True)` is in use,
  which is what makes this the live direction.)
* Dense: every expert receives a zero gradient, so decay **is** applied and the
  moments decay toward zero.

Two consequences: the "debug-only" dense path trained a measurably different
model, so it could not be used to validate a sparse run; and rarely-routed
experts carried **stale Adam moments** across long gaps, so the first update they
received when finally selected pointed in a direction computed against a network
state many steps old.

```python
model.materialize_expert_grads()   # between backward() and step()
```

`MixtureOfExperts.materialize_zero_grads()` gives every unrouted expert parameter
an explicit zero gradient, and returns how many it had to create — a direct
measure of how much of the model sat out the step. The trainer calls it, the dry
run calls it, and `tests/test_moe_layer.py` asserts **parameter state after one
optimizer step** is identical between the two dispatch modes, with a
counterfactual test showing they diverge without it.

### Gradients under sparse dispatch

**An expert that no token routed to receives no gradient that step.** This is the
defining property of a sparse MoE, not a defect.
`tests/test_models.py::test_only_the_routed_experts_receive_gradient` pins the
exact invariant: **routed <=> has gradient** — not the false claim that every
expert learns every step.

How visible this is depends entirely on `token_mode`. At `pooled` with a batch of
12 and `K=2`, 24 routing slots across six experts leaves an expert sitting out an
entire batch routinely. At `grid`, the same batch fills 1,536 slots and the dry
run measures zero dead experts.

### Efficiency accounting hooks

```python
parameters_per_expert()  # one expert's parameter count -- all experts share one
                         # architecture, so counting the first is exact
dormant_parameters()     # (num_experts - top_k) * parameters_per_expert()
```

These give `src/utils/efficiency.py` a **closed form**, not an estimate. Halving
`top_k` from 4 to 2 exactly doubles `dormant_parameters()`.

`dormant_parameters()` counts only *routing* dormancy. Parameters no
configuration can ever reach are no longer built at all (see the affine token
mixer above), so this number no longer silently includes provably dead weights.

`expert_utilization(top_k_indices)` returns the **hard dispatch fraction** `f`,
summing to 1 across experts however wide Top-K is. That is what the utilisation
figure draws and what the Switch loss multiplies by `P`. The submitted version
returned "fraction of samples routed to each expert", which summed to `top_k` and
so could not be compared against either. `dead_expert_count()` is `#{i : f_i = 0}`.

## 2. `DenseExpertBlock` — the `use_moe=False` ablation

A single always-on `TransformerExpert` returning a `MoEOutput` that describes a
degenerate one-expert router (`gate_probs = ones`, `top_k_indices = zeros`,
weight 1 always):

```python
num_experts = 1
top_k = 1
dormant_parameters() -> 0   # always: a dense block activates everything it owns
```

The rationale (`REVISION_NOTES.md` §4): deleting the experts outright would
*also* delete a block's worth of capacity, confounding *routing* with *depth*.

**That rationale identified the right hazard and only half-corrected for it.**
The full model activates `top_k` experts per token; this activates one. So the
naive `wo_moe` gap conflates routing with a `top_k`-fold reduction in active
capacity and active FLOPs — the very quantity the efficiency section is built on.

`capacity_multiplier` scales the feed-forward width so the two can be matched.
`scripts/run_ablations.py` runs both, because they answer different questions:

| Variant | Override | Measures |
| --- | --- | --- |
| `wo_moe` | `use_moe=false` | routing + capacity + 2 regularisers (historical) |
| `wo_moe_capacity_matched` | `+ dense_capacity_multiplier=2` | routing, capacity held fixed |
| `moe_fixed_router` | `router_mode=hash` | *learned* routing, sparse capacity held fixed |

Both MoE regularizers evaluate to exactly zero on the degenerate gate — the
entropy term because a one-dimensional distribution has no entropy to spread, the
Switch term because `f = P = (1)` gives `1 x 1 x 1 - 1 = 0`, the sparsity term
because no mass falls outside a Top-1-of-1 selection. That is convenient
plumbing, but note what it means: **`wo_moe` optimises a strictly smaller
objective than the full model.** That is now recorded in the run's `loss_flags`
rather than left implicit.

## 3. MoE regularization losses (`src/losses/moe.py`)

Paper Section 5.2 introduces two terms *"to prevent expert collapse and
encourage balanced expert utilization."* The revision keeps that goal and
changes the mathematics, because the submitted formulation could not reach it.

### Load balancing — why the entropy form was replaced

The submitted form was the negative entropy of the batch-mean **soft** gate:

$$
u_i = \text{mean}_{\text{batch}}(G_i), \qquad
\mathcal{L}_{\text{load}}^{\text{entropy}} = \frac{\sum_i u_i \log u_i}{\log E} \in [-1, 0]
$$

The model's behaviour, however, is decided by `topk(G, K)`, which `u` never
observes. Take a router that emits the same distribution for every sample:

```text
G = (0.30, 0.30, 0.10, 0.10, 0.10, 0.10)

-sum u_i ln u_i = 2(0.3)ln(1/0.3) + 4(0.1)ln(1/0.1) = 1.64342
log E           = ln 6                               = 1.79176
L_load          = -1.64342 / 1.79176                 = -0.9172
```

The regulariser reports **91.7 % of maximal balance**. Meanwhile `topk(G, 2)`
selects `{0, 1}` for every sample in every batch, so the true dispatch fraction
is `f = (0.5, 0.5, 0, 0, 0, 0)` and **four of six experts receive no gradient,
ever**.

The degenerate case is worse. At the entropy term's *global optimum* `G` is
exactly uniform, and `torch.topk` breaks ties toward the lowest indices — so
**the global minimum of the load-balancing loss produces maximally imbalanced
hard routing**. This is not a corner case; it is the point the regulariser pulls
toward.

`tests/test_losses.py` contains this counterexample to four decimal places.

### Load balancing — the dispatch-aware form (`moe_load_mode: switch`, default)

The Shazeer / GShard / Switch auxiliary loss couples the **hard** dispatch
fraction to the **differentiable** router probability:

$$
\mathcal{L}_{\text{load}} = E \sum_{i=1}^{E} f_i P_i, \qquad
f_i = \frac{1}{T}\sum_x \mathbb{1}[i \in \text{Top-}K(x)], \qquad
P_i = \frac{1}{T}\sum_x G_i(x)
$$

```python
def switch_load_balancing_loss(gate_probs, top_k_indices, zero_floor=True):
    num_experts = gate_probs.shape[-1]
    soft = gate_probs.mean(dim=0)                                    # P, differentiable
    hard = dispatch_fraction(top_k_indices, num_experts).detach()    # f, no gradient
    loss = num_experts * torch.sum(hard * soft)
    return loss - 1.0 if zero_floor else loss
```

`f` is non-differentiable by construction and acts as a per-expert coefficient;
`P` carries the gradient. The product means an **over-dispatched** expert gets
its router probability pushed down specifically — a test asserts exactly that
sign. The minimum is `1` at uniform routing, and the reported value subtracts it
so a balanced router still scores `0` and `lambda_moe_load` keeps its meaning
across expert counts, exactly as the `log E` normalisation did before.

On the counterexample above this scores `6 x (0.5x0.3 + 0.5x0.3) - 1 = 0.8`,
which is 0.8 above its floor rather than 92 % of the way to it.

`moe_load_mode: entropy` retains the submitted form so the two can be compared on
one split. That comparison is a small publishable result rather than an
embarrassment.

### Router z-loss

$$
\mathcal{L}_z = \frac{1}{B}\sum_i \Big(\log \sum_j e^{x_j^{(i)}}\Big)^2
$$

over the **pre-softmax** router logits (Zoph et al., ST-MoE) at the standard
`lambda_moe_z = 1e-3`. It prevents router logit growth, which is what makes Top-K
selection brittle. Applied to probabilities it would be a constant, so
`MoEOutput` carries `gate_logits` separately.

### Sparsity — kept, but off

$$
\mathcal{L}_{\text{sparsity}} = \text{mean}\Big(\sum_{i \notin \text{Top-}K} G_i\Big) \in [0, 1]
$$

**Under `renormalize_top_k=True` this term cannot change the model's function.**
The renormalised weights are invariant to any rescaling of `G` restricted to the
Top-K set, so there is a descent direction — move mass from the off-Top-K experts
onto the selection while holding their ratio fixed — along which `h` does not
change at all. Its only reliable effect is to reduce router entropy, which
directly *fights* the load term.

This document used to frame the two as productive tension: *"load-balancing wants
the batch spread; sparsity wants each sample decisive."* In the standard
formulation that framing is right. Here it is not, because renormalisation
already delivers the convex combination that per-sample decisiveness was supposed
to buy. `renormalize_top_k=True` and `lambda_moe_sparsity > 0` are mutually
redundant; the default picks the first, and `lambda_moe_sparsity: 0.0`.

`mode="topk"` still reproduces the earlier, incorrect implementation
(`PAPER_AUDIT.md` §3.1) for ablation.

### Estimator noise

`f` and `P` are estimated from one batch. At `batch_size=16` and `K=2` that is
**32 routing slots over 6 experts** — far too noisy to steer a self-reinforcing
process. Two things address it:

* `moe_utilization_momentum: 0.9` EMA-smooths both statistics across steps. The
  gradient path stays on the current batch's `P`; the history contributes only
  its detached value.
* `token_mode="grid"` fixes it at the source: each image contributes 64 routing
  tokens instead of 1, so the same step fills **2,048** slots.

The dry run shows the difference immediately — pooled routing leaves 1 dead
expert per step with utilisation spanning 0.10-0.30; grid routing leaves 0 with
utilisation spanning 0.155-0.184.

### `MoERegularization`

```python
MoERegularization(lambda_load=0.01, lambda_sparsity=0.0, lambda_z=1e-3,
                  normalize_entropy=True, sparsity_mode="off_topk",
                  load_mode="switch", utilization_momentum=0.9, num_experts=6)
```

`forward(gate_probs, top_k_indices, gate_logits) -> MoERegularizationOutput(total,
load_balancing, sparsity, z_loss, utilization, hard_utilization, dead_experts)`.

`hard_utilization` is `f` — the quantity `plot_expert_utilization` draws and the
quantity the loss now balances. Previously the figure measured `f` and the loss
measured `P`, so the diagnostic and the objective disagreed about what "balanced"
meant, and the diagnostic was the correct one.

`dead_experts` is `#{i : f_i = 0}` and is logged every epoch. It is the metric
that would have caught the collapse the entropy loss called healthy.

### Exploration

`topk` has zero gradient almost everywhere, so nothing in the objective can say
"this token should have gone to expert 4". Deterministic Top-K on a **frozen**
encoder is a rich-get-richer process with no counterweight: `z` is fixed by stage
1, so there is not even representation drift to shake a collapsed router loose.

Two mechanisms, both in `MixtureOfExperts`:

* **Noisy Top-K gating** (Shazeer et al., 2017): Gaussian noise on the router
  logits scaled by a learned softplus term, at `router_noise_std: 0.3`, annealed
  linearly to **exactly zero** over the first `router_noise_fraction` of training
  so the deployed routing is the routing that was measured. Training-mode only.
  The noise gate is **not allocated** when `noise_std = 0`, so it never becomes a
  block of parameters no configuration can reach.
* **Near-zero router init** (`gate_init_std=1e-3`), so early routing is close to
  uniform and the race is not decided by initialisation.

> This document previously asserted that *"`MoERegularization`'s entropy term is
> what pulls utilization back toward uniform over an epoch, so no expert stays
> unrouted for long."* That claim is **false in general** — the entropy term can
> be near-optimal while four experts stay permanently unrouted, which is the
> counterexample above. It is replaced by the measured dead-expert count.

### Routing controls

`router_mode` selects between the learned gate and two controls the ablation
suite needs in order to attribute anything to the MoE:

| Mode | Routing | Isolates |
| --- | --- | --- |
| `learned` | trained linear gate | the method |
| `hash` | fixed assignment by token index | sparse capacity **without** learned routing |
| `uniform` | every expert at weight `1/E` | ensembling **without** sparsity |

Neither control owns trainable routing parameters — that is what makes them
controls for *learned* routing rather than for sparse capacity. `moe_fixed_router`
is the important one: it is the only configuration that can say whether the
router learned anything, and it is the direct answer to the question a reviewer
asks about Section 5.2.

Balance is not specialisation, so `expert_label_nmi` reports normalised mutual
information between the top-1 expert and the label. Six experts each taking a
sixth of the traffic at random are perfectly balanced and perfectly
uninformative; NMI separates the two, and it is computable from an existing
`test_predictions.npz` with no retraining.

## 4. Where the MoE sits in the head

`HierarchicalSeedClassifier.__init__` (`src/models/builder.py:248-265`)
builds either `MixtureOfExperts` or `DenseExpertBlock` under the attribute
name `self.moe`, selected by `use_moe`:

```python
self.moe = MixtureOfExperts(embed_dim, num_experts, moe_hidden_dim, top_k, num_heads,
                            dropout_rate, sparse_dispatch=moe_sparse_dispatch,
                            token_mixing=resolve_attention_mode(token_mode),
                            router_mode=router_mode,
                            gate_condition_dim=num_seed_types if gate_conditioning else 0,
                            noise_std=router_noise_std) if use_moe \
      else DenseExpertBlock(embed_dim, moe_hidden_dim, num_heads, dropout_rate,
                            token_mixing=..., capacity_multiplier=dense_capacity_multiplier)
```

In `forward`:

```python
gate_condition = seed_type_probs.detach() if self.gate_conditioning else None
moe = self.moe(embedding, gate_condition=gate_condition)
```

**The experts consume `z`, the encoder embedding**, per Eq. 8. This is one of
three dataflow facts the paper audit flags as easy to get wrong: an earlier
version routed on a projection of the 4-D seed-type logits instead, meaning the
experts never saw the image at all (`PAPER_AUDIT.md` §2.1). A test asserts the
experts' input width is still `embed_dim`, so that regression cannot recur.

**The gate may additionally see the coarse posterior.** Without
`gate_conditioning`, the router and the seed head both read `z` and never
interact, so the "hierarchical MoE" is structurally a flat router sitting beside
a coarse classifier and expert specialisation has no reason to align with seed
type. Feeding the gate `p_s` makes the name true. The `detach()` is deliberate:
routing should be *informed* by the coarse prediction without the router's
gradient reshaping the coarse head, which would reintroduce the same incentive
problem the KL term's detach exists to prevent. A test asserts no gradient
reaches the seed classifier through the gate.

`wo_gate_conditioning` ablates it, and `expert_label_nmi` measures whether it
bought anything.

See [`04_HIERARCHICAL_FUSION.md`](04_HIERARCHICAL_FUSION.md) for the full cascade
this feeds into.
