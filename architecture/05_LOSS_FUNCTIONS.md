# 05 — Loss Functions

Covers `src/losses/hierarchical.py`, `src/losses/arcface.py`,
`src/losses/cosine.py`, `src/losses/moe.py` (regularizers detailed in
[`03_MOE_MODULE.md`](03_MOE_MODULE.md) §3), and
`conf/model/loss/{arcface_kl,flat_cce}.yaml`. Implements paper Section 5's
combined objective plus the revision's cosine-compactness addition (Section
1).

## 1. The combined objective (stage 2)

```text
L = λ_seed     · L_seed        Eq. 7   categorical cross-entropy, 4 seed types
  + λ_arcface  · L_ArcFace     Eq. 13  angular margin, 27 sub-varieties
  + λ_kl       · L_KL          Eq. 10  hierarchy consistency
  + λ_load     · L_load        §5.2    entropy load balancing
  + λ_sparsity · L_sparsity    §5.2    L1 Top-K sparsity
  + λ_cosine   · L_cos         §1      residual compactness
  + λ_sub_ce   · L_sub_CE      —       auxiliary plain CE, 0.0 by default
```

Implemented in `CombinedHierarchicalLoss.forward`
(`src/losses/hierarchical.py:253-316`), which takes the model's
`HierarchicalOutput` directly (not positional tensors) and returns a
`LossBreakdown` NamedTuple (`hierarchical.py:66-89`) carrying the weighted
total plus every unweighted component:

```python
LossBreakdown(total, seed, arcface, sub_ce, kl, moe_load, moe_sparsity, cosine)
```

`.as_dict()` flattens these into tracker-ready floats
(`total_loss, seed_type_loss, arcface_loss, sub_variety_ce_loss, kl_loss,
moe_load_balancing_loss, moe_sparsity_loss, cosine_loss`), logged every epoch
automatically.

**The criterion holds no learnable parameters** — ArcFace's class centres
live in the model (`ArcFaceHead.weight`), not the loss. Consequences: the
optimizer needs only the model plus the encoder
(`build_optimizer([encoder, model], cfg)`), gradient clipping covers
everything trainable, and a saved `model_state_dict` alone reproduces
inference.

Weights come from `conf/model/loss/arcface_kl.yaml`:

| Weight | Default | Paper term |
| --- | --- | --- |
| `lambda_seed` | 1.0 | Eq. 7 |
| `lambda_arcface` | 1.0 | Eq. 13 |
| `lambda_kl` | 1.0 | Eq. 10 |
| `lambda_moe_load` | 0.01 | §5.2 |
| `lambda_moe_sparsity` | 0.01 | §5.2 |
| `lambda_cosine` | 0.1 | §1 |
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

## 4. Eq. 10 — hierarchy-consistency KL divergence

$$
\mathcal{L}_{\text{KL}} = D_{\text{KL}}(P_{\text{seed}} \,\|\, P_{\text{sub}})
$$

The two distributions live over different label sets (4 vs. 27), so the
sub-variety distribution is first **aggregated** to seed-type granularity
through a fixed `[27, 4]` one-hot mapping matrix `M`:

$$
P_{\text{sub-agg}} = \text{softmax}(\text{sub\_logits}) \cdot M, \qquad
\mathcal{L}_{\text{KL}} = D_{\text{KL}}(P_{\text{seed}} \,\|\, P_{\text{sub-agg}})
$$

```python
def hierarchical_kl_loss(seed_type_logits, sub_variety_logits, mapping_matrix,
                          detach_seed_target=False):
    seed_probs = softmax(seed_type_logits, dim=-1)
    if detach_seed_target:
        seed_probs = seed_probs.detach()
    sub_probs  = softmax(sub_variety_logits, dim=-1)
    aggregated = sub_probs @ mapping_matrix
    return F.kl_div(torch.log(aggregated.clamp_min(1e-8)), seed_probs, reduction="batchmean")
```

(`hierarchical.py:146-175`) **Direction matters**: `F.kl_div(input=log q,
target=p)` computes `KL(p \| q)`. Passing the aggregated sub-variety
log-probabilities as `input` and the seed-type probabilities as `target` is
what gives the `D_KL(P_seed \| P_sub)` direction Eq. 10 asks for — swapping
the arguments would silently optimize the reverse-KL objective instead.
Verified correct by `PAPER_AUDIT.md` §3.5 and pinned by a test that
recomputes the KL by hand.

**`sub_variety_logits` here must be `sub_logits` (no margin), never
`sub_margin_logits`** — the ArcFace margin is a training device for the
classification term, not part of the predicted distribution the hierarchy
term should measure consistency against.

`detach_kl_seed_target=false` (default): the gradient flows into both
branches. Set `true` to freeze the seed-type distribution as a fixed target,
so KL only reshapes the sub-variety head.

### Building the mapping matrix `M`

```python
build_subvariety_seed_mapping(num_sub_varieties, num_seed_types,
                               subvariety_to_seed_type=None,
                               subvarieties_per_seed_type=None) -> torch.Tensor  # [27, 4]
```

(`hierarchical.py:92-134`) Preferred path: an explicit per-sub-variety parent
list — exactly what `HierarchicalSeedDataset.get_subvariety_to_seed_type()`
produces, derived from the directory tree at runtime (see
[`01_DATA_PIPELINE.md`](01_DATA_PIPELINE.md)), never hardcoded. A fallback
path accepts `subvarieties_per_seed_type` (a count list) when sub-variety
indices are known to be contiguous within each seed type. `M` is registered
as a non-trainable buffer (`self.register_buffer("mapping_matrix", mapping)`)
so it moves with `.to(device)` but never receives gradient.

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

## 6. Cosine compactness loss (`src/losses/cosine.py`, paper Section 1)

Paper: *"we introduce cosine similarity loss within SwinV2's residual
connections, promoting feature compactness."* The residual the head owns is
Eq. 9's `h' = h + P(p_s)`.

**`mode="residual"` (default)** — keep `h'` angularly aligned with the raw
MoE feature `h`:

$$
\mathcal{L}_{\cos} = 1 - \text{mean}_i \cos(h'_i, h_i) \in [0, 2]
$$

```python
def residual_cosine_loss(refined, original):
    return 1.0 - F.cosine_similarity(refined, original, dim=-1, eps=1e-8).mean()
```

(`cosine.py:40-42`) This lets the seed-type prior *shift* the representation
without *rotating* it away from what the MoE extracted — the mechanism that
stops the residual from overwriting the MoE output when stage 1 is
confidently wrong. In `CombinedHierarchicalLoss.forward`
(`hierarchical.py:292-294`): `cosine = self.cosine_loss(output.refined_features,
output.moe_features)`.

**`mode="intra_class"`** — pull every embedding toward its own class
centroid, complementing ArcFace's inter-class separation with an explicit
intra-class term:

$$
\mathcal{L}_{\cos} = 1 - \text{mean}_i \cos(e_i, \text{centroid}(\text{class of } i))
$$

```python
def intra_class_cosine_loss(embeddings, labels):
    normalized = F.normalize(embeddings, p=2, dim=-1)
    # per-class mean of normalized embeddings, itself renormalized -> centroid
    ...
    similarity = (normalized * centroids[labels]).sum(dim=-1)
    return 1.0 - similarity.mean()
```

(`cosine.py:45-70`) Classes with a single sample in the batch contribute
exactly `0` (the embedding is its own centroid) rather than distorting the
term. When this mode is active, `CombinedHierarchicalLoss` instead computes
`cosine = self.cosine_loss(output.sub_embeddings, output.sub_embeddings,
sub_variety_labels)`.

Both modes are bounded in `[0, 2]`, reaching `0` at perfect alignment.
`cosine_mode` config default: `"residual"`.

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
