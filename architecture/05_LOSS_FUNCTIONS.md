# 05 — Loss Functions

Covers `src/losses/hierarchical.py`, `src/losses/arcface.py`,
`src/losses/cosine.py`, `src/losses/moe.py` (regularizers detailed in
[`03_MOE_MODULE.md`](03_MOE_MODULE.md) §3), and
`conf/model/loss/{arcface_kl,flat_cce}.yaml`. Implements paper Section 5's
combined objective plus the revision's cosine-compactness addition (Section
1).

## 1. The combined objective (stage 2)

```text
L = w_seed     . L_seed        Eq. 7   categorical cross-entropy, 4 seed types
  + w_arcface  . L_ArcFace     Eq. 13  angular margin, 27 sub-varieties
  + w_kl       . L_KL          Eq. 10  hierarchy consistency, in log space
  + lam_load     . L_load      5.2     dispatch-aware load balancing
  + lam_sparsity . L_sparsity  5.2     L1 Top-K sparsity      (0.0 by default)
  + lam_z        . L_z         --      router z-loss (ST-MoE)
  + lam_cosine   . L_cos       1       class compactness, EMA centroids
  + lam_residual . L_res       Eq. 9   residual magnitude hinge
  + lam_sub_ce   . L_sub_CE    --      auxiliary plain CE      (0.0 by default)
```

Implemented in `CombinedHierarchicalLoss.forward`, which takes the model's
`HierarchicalOutput` directly (not positional tensors) and returns a
`LossBreakdown` NamedTuple carrying the weighted total plus every unweighted
component:

```python
LossBreakdown(total, seed, arcface, sub_ce, kl, moe_load, moe_sparsity, moe_z,
              cosine, residual, dead_experts, task_weights)
```

`.as_dict()` flattens these into tracker-ready floats, logged every epoch
automatically. `component_losses()` is exposed separately so the trainer's
gradient telemetry can backprop one term at a time **through the same code the
objective uses** — a second implementation would measure a different thing.

### Weighting

The submitted objective used seven fixed lambdas over terms whose magnitudes at
initialisation differed by ~13x. With `s = 30` and `m = 0.5`, random 384-D
embeddings give `cos(theta) ~ 0`, hence

```text
target logit = 30 . cos(theta + 0.5) ~ 30 . (-sin 0.5) = -14.38
L_ArcFace    = log(1 + 26 . e^14.38) = 17.64
L_seed       = ln 4                  =  1.386        ->  ratio 12.7 : 1
```

One term carried ~92 % of the initial gradient budget, and `L_KL` and `L_cos`
were rounding error against it. **Most of that is a consequence of the ArcFace
scale, not of the lambdas**: at the AdaCos scale (§3) the dry run measures the
ratio at ~4:1, so `weighting_mode: fixed` remains defensible.

`weighting_mode: uncertainty` learns the three **task** weights via Kendall et
al.'s homoscedastic formulation:

$$
\mathcal{L} = \sum_{t \in \{\text{seed}, \text{arc}, \text{kl}\}}
  \Big(\frac{\mathcal{L}_t}{2\sigma_t^2} + \tfrac{1}{2}\log \sigma_t^2\Big)
  + \sum_r \lambda_r \mathcal{L}_r
$$

optimising `log sigma^2` directly (clamped for stability). The three regularisers
keep fixed lambdas deliberately: they are genuinely auxiliary, small, and their
scale is meaningful — `L_load` has a zero floor by construction. The learned
`sigma_t` are also **diagnostic**: if `sigma_arcface` collapses while `sigma_kl`
explodes, the dominance hypothesis is confirmed empirically rather than argued.

**Whichever mode is chosen must be identical across every variant in a suite**,
or the ablation gaps become gaps in loss-weighting policy. `loss_flags` in
`summary.json` records it so that cannot happen silently.

### Statefulness

The criterion holds learnable parameters **only** under
`weighting_mode="uncertainty"` (three scalars), and buffers only for the EMA
class centroids and the routing statistics. ArcFace's class centres stay in the
model, so a saved `model_state_dict` alone still reproduces inference.

The trainer therefore passes the criterion to the optimizer:
`build_optimizer([encoder, model, criterion], cfg)`. Omitting it would pin the
task weights at their initial values while the logs reported them as learned.

Weights come from `conf/model/loss/arcface_kl.yaml`:

| Weight | Default | Term |
| --- | --- | --- |
| `lambda_seed` | 1.0 | Eq. 7 |
| `lambda_arcface` | 1.0 | Eq. 13 |
| `lambda_kl` | 1.0 | Eq. 10 |
| `lambda_moe_load` | 0.01 | 5.2, Switch form |
| `lambda_moe_sparsity` | **0.0** | 5.2, redundant under renormalisation |
| `lambda_moe_z` | 0.001 | router z-loss, ST-MoE standard |
| `lambda_cosine` | 0.1 | 1, class compactness |
| `lambda_residual` | 0.01 | Eq. 9 magnitude hinge |
| `lambda_sub_ce` | 0.0 | auxiliary, off |

## 2. Eq. 7 — seed-type cross-entropy

```python
def seed_type_loss(seed_type_logits, labels, label_smoothing=0.0):
    return F.cross_entropy(seed_type_logits, labels, label_smoothing=label_smoothing)
```

(`hierarchical.py:137-143`) Plain categorical cross-entropy over the 4-way
`seed_type_logits`, `s` (Eq. 5). `seed_label_smoothing` defaults to `0.0`.

## 3. Eq. 13 — ArcFace loss

```python
def arcface_loss(margin_logits, labels, label_smoothing=0.0):
    return F.cross_entropy(margin_logits, labels, label_smoothing=label_smoothing)
```

(`src/losses/arcface.py:34-47`) A thin functional wrapper: since
`ArcFaceHead` already produces `s \cdot \cos(\theta + m)` on the target class
(margin logits), a softmax cross-entropy over those logits *is* exactly Eq.
13's expression:

$$
\mathcal{L}_{\text{ArcFace}} = -\sum_i \log\frac{e^{s\cos(\theta_i+m)}}{e^{s\cos(\theta_i+m)}+\sum_{j\neq i}e^{s\cos\theta_j}}
$$

`CombinedHierarchicalLoss` calls `arcface_loss(output.sub_margin_logits,
sub_variety_labels)` — **never** `output.sub_logits` here (that quantity
drives prediction and KL, not this term). `ArcFaceLoss`
(`src/losses/arcface.py:50-88`) is a separate, self-contained module that
owns its own `ArcFaceHead` for standalone testing/ablation — it is stateful
(holds `nn.Parameter`s) and is not what the combined loss uses.

For the margin/threshold mathematics themselves, see
[`04_HIERARCHICAL_FUSION.md`](04_HIERARCHICAL_FUSION.md) §6 (`ArcFaceHead`
lives in `src/models/components/`, not `src/losses/`, since it owns
parameters the model must checkpoint).

## 4. Eq. 10 — hierarchy consistency, in log space

$$
\mathcal{L}_{\text{KL}} = D_{\text{KL}}(P_{\text{seed}} \,\|\, P_{\text{sub-agg}})
$$

The two distributions live over different label sets (4 vs. 27), so the
sub-variety distribution is first **marginalised** to seed-type granularity.

### The bug the revision fixes

The submitted implementation aggregated in probability space and then took a log:

```python
aggregated = sub_probs @ mapping_matrix
F.kl_div(torch.log(aggregated.clamp_min(1e-8)), seed_probs, reduction="batchmean")
```

`clamp_min` has **zero gradient in the clamped region**. So whenever an
aggregated seed-type probability fell below `1e-8`, the KL term contributed
exactly nothing to that entry — silently, with no NaN and no warning.

**That was the common case, not an edge case.** With `sub_logits = 30 cos(theta)`
in `[-30, 30]`, a cosine gap of 1.0 between the argmax sub-variety and a
competitor becomes a *logit* gap of 30, i.e. a probability ratio of
`e^30 ~ 1.07e13`. Aggregating 27 such probabilities into 4 bins leaves the
non-argmax bins at `1e-13` to `1e-10`, comfortably below the clamp. Concretely:
**the term was live when the two heads agreed (where it has nothing to do) and
dead when they disagreed confidently (which is the entire point of it).**

### The fix

```python
def aggregate_sub_log_probs(sub_variety_logits, mapping_matrix, tau=1.0):
    log_p_sub = F.log_softmax(sub_variety_logits / tau, dim=-1)          # [B, 27]
    children  = mapping_matrix.t().bool()                                # [4, 27]
    masked    = log_p_sub.unsqueeze(1).masked_fill(~children.unsqueeze(0), -inf)
    return torch.logsumexp(masked, dim=-1)                               # [B, 4], exact
```

`logsumexp` over each parent's children is **exact**, not a tolerance tweak: it is
numerically stable by construction (max-subtraction), needs no epsilon, and has a
well-conditioned gradient across the entire probability range — including the
confident-disagreement region where the old form was silently dead.

Two tests pin this: one asserts the term now has non-zero gradient on a maximal
disagreement, and one shows the old `clamp -> log` composition is gradient-dead on
the identical input.

### `tau_kl` — untangling two hyperparameters

`P_sub = softmax(s cos theta)` is near-one-hot by construction at `s = 30`, so
`lambda_kl` and `arcface_scale` were secretly **one** hyperparameter: changing the
scale silently changed the effective strength of the hierarchy term. `tau_kl`
divides before the softmax and separates them. At the AdaCos scale (§3),
`tau_kl = 1.0` is the right value; raise it toward `s` if `arcface_scale` is
raised.

### Direction, and the detach default

`F.kl_div(input=log q, target=p)` computes `KL(p || q)`. Passing the aggregated
sub-variety log-probabilities as `input` and the seed-type log-probabilities as
`target` (with `log_target=True`) gives the `D_KL(P_seed || P_sub)` direction
Eq. 10 asks for — swapping the arguments would silently optimize the reverse-KL
objective.

**`detach_kl_seed_target` now defaults to `true`.** With the gradient flowing into
both branches, the term is symmetric in *who moves*, and `P_seed` is already
supervised by hard labels through Eq. 7. Letting `L_KL` also push the coarse head
means it can be reduced by the coarse head becoming *less* accurate — agreeing
with a confidently wrong fine prediction. Since Eq. 7 fits fast on 4 classes, the
marginal gradient available on the seed side is spent disproportionately on the
hard, ambiguous samples, which are exactly the ones where following the fine head
is most likely to be wrong.

`KL(p||q)` is also **mode-covering in `q`**: wherever `P_seed` has mass,
`P_sub-agg` is forced to have mass, so an uncertain coarse head makes the fine
head hedge *across seed types* — smearing the very decisions the 27-way task
depends on.

### `kl_mode: jsd`

The symmetric alternative the hierarchy-consistency literature converged on
(HAF, Garg et al. 2022):

$$
\mathcal{L}_{\text{JS}} = \tfrac{1}{2}D_{\text{KL}}(P_{\text{seed}} \| M)
                          + \tfrac{1}{2}D_{\text{KL}}(P_{\text{sub-agg}} \| M),
\qquad M = \tfrac{1}{2}(P_{\text{seed}} + P_{\text{sub-agg}})
$$

Bounded by `log 2`, symmetric, and without the zero-avoidance that forces the
fine head to hedge. Computed in log space via `logsumexp`, and tested to stay
within its bound. `kl_jsd` runs it as an ablation, so `{forward, detached, jsd}`
is a three-way comparison rather than an assertion.

### `sub_logits`, never `sub_margin_logits`

The margin is a training device for the classification term, not part of the
predicted distribution the hierarchy term measures consistency against.

### Building the mapping matrix `M`

```python
build_subvariety_seed_mapping(num_sub_varieties, num_seed_types,
                              subvariety_to_seed_type=None,
                              subvarieties_per_seed_type=None) -> torch.Tensor  # [27, 4]
```

Preferred path: an explicit per-sub-variety parent list — exactly what
`HierarchicalSeedDataset.get_subvariety_to_seed_type()` produces, derived from the
directory tree at runtime (see [`01_DATA_PIPELINE.md`](01_DATA_PIPELINE.md)),
never hardcoded. `M` is registered as a non-trainable buffer so it moves with
`.to(device)` but never receives gradient. The log-space path reads it as a
boolean children mask; the stored one-hot float form is unchanged, so the buffer,
the checkpoint layout and every existing consumer are unaffected.

## 5. Auxiliary sub-variety cross-entropy (`lambda_sub_ce`, off by default)

```python
sub_ce = F.cross_entropy(output.sub_logits, sub_variety_labels) if self.lambda_sub_ce != 0.0 \
         else output.sub_logits.new_zeros(())
```

(`hierarchical.py:276-280`) The paper classifies sub-varieties with ArcFace
alone; `PAPER_AUDIT.md` §3.4 records that an earlier version had this term
weighted at `1.0` unconditionally, which was never specified by the paper.
Default `lambda_sub_ce: 0.0`; a small nonzero value is available as an
opt-in aid if the ArcFace margin destabilizes very early training.

## 6. Compactness and residual control (`src/losses/cosine.py`, paper Section 1)

Paper: *"we introduce cosine similarity loss within SwinV2's residual
connections, promoting feature compactness."*

### Why `mode="residual"` is no longer the default

The submitted term was `L_cos = 1 - cos(h + P(p_s), h)`. Characterise its
minimiser set. `L_cos = 0` iff `h'` is a positive multiple of `h`, i.e. iff

```text
P(p_s) = alpha(x) . h(x)     for some alpha >= -1,  including alpha = 0
```

So **every global minimiser either zeroes the residual outright — which *is* the
`use_residual=False` ablation — or collapses it to a scalar rescaling of `h`.**
In the second case Eq. 9 carries **one** degree of freedom of seed-type
information instead of 384. In both cases the coarse-to-fine link the
architecture exists to provide is gone.

The stated intent was the opposite: *"this lets the seed-type prior shift the
representation without rotating it away from what the MoE extracted."* But cosine
is **invariant to magnitude**, so it constrains only the residual's *direction*,
never how much it shifts. The cheapest way to preserve direction is to make the
residual small. The loss achieved "not rotating" by achieving "not shifting".

The weighting made it worse over time rather than better. `lambda_cosine = 0.1`
looks small against `lambda_arcface = 1.0`, but `L_ArcFace` decays toward 0 as the
model fits while `L_cos` has no reason to — so the cosine term's *share* of the
gradient grew monotonically through training. It was weakest during the epochs
when the residual was forming and strongest during the epochs when it could be
dismantled.

`mode="residual"` is retained as an ablation axis, and a test asserts it still
collapses `wo_residual`'s compactness term to exactly zero. The confound is
reproducible on purpose.

### `mode="intra_class"` with EMA centroids (default)

Compactness is an **intra-class** property, so this is the reading that matches
the words:

$$
\mathcal{L}_{\cos} = 1 - \text{mean}_i \cos(e_i, c_{y_i})
$$

Centroids are maintained as an **EMA over training** rather than recomputed per
batch. That is not a refinement — it is what makes the term exist. With
`batch_size = 16` over `C = 27` roughly-balanced classes, the expected number of
classes with two or more members is

```text
C . [1 - (1-p)^n - n.p.(1-p)^(n-1)]  =  27 . [1 - 0.5468 - 0.3365]  =  3.2,   p = 1/27
```

so roughly **6 of 16 embeddings would contribute a non-zero term** and the other
10 exactly zero — and the "centroid" each of those 6 was pulled toward was
estimated from **two samples**. That is not a compactness loss; it is noise with
a small mean.

With EMA centroids every sample contributes and every centroid is estimated from
the whole training history. This is centre loss (Wen et al., 2016) adapted to the
hypersphere, and it supplies exactly the intra-class complement that ArcFace's
inter-class margin does not. Centroids are **persistent buffers** — training
state a resumed run must not silently reset — and a class's first appearance
seeds its centroid outright rather than blending against a zero vector.

A test constructs an all-singleton batch: the per-batch form scores exactly 0,
the EMA form scores a real value.

### Controlling the residual structurally

Magnitude control moved out of the loss and into the architecture:

* **`LayerScale`** on the Eq. 9 branch — a learned per-channel gain initialised
  at `1e-4` (Touvron et al., 2021). It gives the "start negligible, grow only if
  it helps" behaviour the cosine penalty was reaching for, with no fixed point at
  `P = 0`. `wo_layer_scale` ablates it.
* **`residual_magnitude_loss`**, an optional hinge on the magnitude *ratio*:

$$
\mathcal{L}_{\text{res}} = \text{mean}_i \max\!\Big(0, \frac{\lVert P(p_s)_i \rVert}{\lVert h_i \rVert} - \tau\Big)^2,
\qquad \tau = 0.5
$$

  Exactly zero for every residual smaller than `tau`, so it is inactive in the
  healthy regime and has **no gradient pushing `P` toward zero**. That is the
  property the cosine form lacks, and a test asserts both halves of it.

`lambda_residual: 0.01`.

## 7. Config reference

### `conf/model/loss/arcface_kl.yaml` — full model

```yaml
use_kl_loss: ${model.head.use_kl_loss}   # single source of truth, see below
lambda_seed: 1.0
lambda_arcface: 1.0
lambda_kl: 1.0
lambda_moe_load: 0.01
lambda_moe_sparsity: 0.01
lambda_cosine: 0.1
lambda_sub_ce: 0.0
seed_label_smoothing: 0.0
arcface_label_smoothing: 0.0
detach_kl_seed_target: false
moe_sparsity_mode: "off_topk"
normalize_moe_entropy: true
cosine_mode: "residual"
```

### `conf/model/loss/flat_cce.yaml` — supervised baselines

```yaml
use_kl_loss: false
lambda_seed: 1.0
lambda_arcface: 1.0     # -> plain CE, since baselines have no margin
lambda_kl: 0.0
lambda_moe_load: 0.0    # no router to balance
lambda_moe_sparsity: 0.0
lambda_cosine: 0.0      # no Eq. 9 residual to regularise
lambda_sub_ce: 0.0      # would double-count the sub-variety term
```

This is **the same `CombinedHierarchicalLoss` class** with the
hierarchy-specific terms zeroed, rather than a second criterion
implementation — sharing one code path guarantees the baseline's
cross-entropy and the proposed model's ArcFace-as-CE degeneration (when
`use_arcface=False`) are computed identically, so any measured gap between
variants is architectural rather than an artifact of divergent loss code.

### The `wo_kl` ablation switch

```python
self.use_kl_loss = bool(use_kl_loss)
self.lambda_kl = float(lambda_kl) if self.use_kl_loss else 0.0
...
kl = hierarchical_kl_loss(...) if self.use_kl_loss else output.sub_logits.new_zeros(())
```

(`hierarchical.py:243-246, 281-290`) `use_kl_loss=False` **skips Eq. 10
entirely** — no gradient is computed for it at all — rather than computing it
and multiplying by zero. It also **overrides** a nonzero `lambda_kl` passed
alongside it, rather than being silently re-enabled by one. This flag is
interpolated from the head config (`model.loss.use_kl_loss:
${model.head.use_kl_loss}`) so all five ablation toggles have one home; see
[`06_ABLATION_ENGINE.md`](06_ABLATION_ENGINE.md).

### `build_combined_loss(loss_cfg, num_seed_types, num_sub_varieties, subvariety_to_seed_type)`

(`hierarchical.py:326-356`) The config → instance constructor the finetune
trainer calls once per fold, passing
`dataset.get_subvariety_to_seed_type()` so `M` always reflects the current
directory tree.
