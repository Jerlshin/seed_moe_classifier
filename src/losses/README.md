# `src/losses/` — objectives

| File | Paper | Contents |
| --- | --- | --- |
| `dino.py` | Eqs. 1-3, Alg. 1 | `CustomDINOLoss` — stage 1 |
| `hierarchical.py` | Eqs. 7, 10 | `CombinedHierarchicalLoss`, KL hierarchy term, mapping matrix |
| `arcface.py` | Eq. 13 | `arcface_loss` (functional), `ArcFaceLoss` (standalone) |
| `moe.py` | Section 5.2 | Entropy load balancing, L1 Top-K sparsity |
| `cosine.py` | Section 1 | Residual / intra-class compactness |

## The combined objective (stage 2)

```
L = λ_seed     · L_seed        Eq. 7   categorical cross-entropy over 4 types
  + λ_arcface  · L_ArcFace     Eq. 13  angular margin over 27 sub-varieties
  + λ_kl       · L_KL          Eq. 10  hierarchy consistency
  + λ_load     · L_load        §5.2    entropy load balancing
  + λ_sparsity · L_sparsity    §5.2    L1 Top-K sparsity
  + λ_cosine   · L_cos         §1      residual compactness
  + λ_sub_ce   · L_sub_CE      —       auxiliary, 0.0 by default
```

`CombinedHierarchicalLoss.forward` takes a `HierarchicalOutput` and returns a
`LossBreakdown` NamedTuple with the weighted total plus every unweighted
component; `as_dict()` gives tracker-ready floats. Weights come from
`conf/model/loss/arcface_kl.yaml`.

**The criterion holds no learnable parameters.** ArcFace's class centres live in
the model, so the optimiser only needs the model plus the encoder, gradient
clipping covers everything, and `model_state_dict` alone reproduces the head.

## Ablation switches

Only one of the five component toggles reaches this package:
`use_kl_loss=false` skips Eq. 10 outright rather than weighting it to zero, so no
gradient is computed for it, and it **overrides** a non-zero `lambda_kl` rather
than being silently re-enabled by one.

`use_arcface=false` deliberately needs **no switch here**. It swaps the model's
head for `LinearSubVarietyHead`, which returns its logits unchanged as
`sub_margin_logits`. `L_ArcFace` is a cross-entropy over `sub_margin_logits`, so
with no margin present it *is* the categorical cross-entropy the ablation calls
for. Keeping a single code path means the ablation cannot drift from the full
model by way of a second loss implementation; the summary table records the
result under the same column.

`conf/model/loss/flat_cce.yaml` is the same class with the hierarchy-specific
weights zeroed — used by the supervised baselines, again so that the baseline's
cross-entropy and the proposed model's are computed by identical code.

## Hierarchy consistency — Eq. 10

`L_KL = D_KL(P_seed ‖ P_sub-variety)`. The two distributions live over different
label sets (4 vs 27), so the sub-variety distribution is first aggregated to
seed-type granularity through a fixed one-hot matrix `M`:

```
P_sub_agg = softmax(sub_logits) @ M       M: [27, 4]
L_KL      = D_KL(P_seed ‖ P_sub_agg)
```

`M` comes from `HierarchicalSeedDataset.get_subvariety_to_seed_type()` — derived
from the directory tree at runtime, not hardcoded.

Two things to be careful about, both pinned by tests:

* **Direction.** `F.kl_div(input=log q, target=p)` computes `KL(p‖q)`, so the
  aggregated sub-variety log-probabilities are the `input` and the seed-type
  probabilities the `target`. Swapping them silently optimises the wrong thing.
* **Use `sub_logits`, never `sub_margin_logits`.** The ArcFace margin is a
  training device, not part of the predicted distribution.

`detach_kl_seed_target=true` freezes the seed branch as a target, so KL only
reshapes the sub-variety head.

## MoE regularisation — Section 5.2

**Load balancing** is the negative entropy of the batch-averaged gate
distribution, normalised by `log(E)`:

```
L_load = −H(u) / log(E),   uᵢ = mean_batch(Gᵢ)     ∈ [−1, 0]
```

`−1` is perfectly uniform utilisation, `0` is total collapse onto one expert.
The normalisation keeps `λ_load` meaningful if the expert count changes; disable
it with `normalize_moe_entropy: false`.

**Sparsity** penalises routing mass landing outside the Top-K selection:

```
L_sparsity = mean_batch( Σ_{i∉Top-K} Gᵢ ) = mean(1 − Σ_{i∈Top-K} Gᵢ)   ∈ [0, 1]
```

The gate is a probability simplex, so its total L1 norm is identically 1 and
penalising *that* would do nothing; what "restricts the selection to only the
top-K most relevant experts" describes is driving the discarded mass to zero.
`moe_sparsity_mode: "topk"` restores the earlier behaviour of penalising the
selected weights themselves — see `PAPER_AUDIT.md` §3.1.

The two terms deliberately oppose each other: one wants each *sample* routed
decisively, the other wants the *batch* spread evenly. That tension is what
yields specialised but fully-utilised experts.

**At Top-2 both terms matter more.** With `K = 2` of `E = 6`, four experts are
discarded per sample rather than two, so the sparsity term operates on a larger
quantity; and each batch fills half as many routing slots, so utilisation is
noisier and the load-balancing pressure is what keeps an under-used expert from
being abandoned. Both evaluate to exactly zero under `use_moe=false`, where the
degenerate one-expert gate has no entropy to spread and discards no mass.

## Cosine compactness — Section 1

*"cosine similarity loss within SwinV2's residual connections, promoting feature
compactness."* The residual the head owns is Eq. 9, so `mode="residual"`
(default) keeps `h'` angularly aligned with `h`:

```
L_cos = 1 − mean cos(h', h)      ∈ [0, 2]
```

This lets the seed-type prior *shift* the representation without rotating it away
from what the experts extracted — which is what stops the residual from
overwriting the MoE output when stage 1 is confidently wrong.

`mode="intra_class"` instead pulls each embedding toward its class centroid,
complementing ArcFace's inter-class separation with an explicit intra-class term.

## DINO loss — Eqs. 1-3

Cross-view objective: every (teacher view, student view) pair *except* same-view
pairs. With 2 global and 4 local crops that is `2 × 6 − 2 = 10` terms.

Two collapse guards, pulling in opposite directions:

* **Temperature schedule (Eq. 2).** Teacher temperature ramps linearly
  0.02 → 0.04 over 5 epochs, then holds (Table 1). A cold teacher early on
  produces sharp targets the student could collapse onto.
* **Centering (Eq. 3).** `C_t = m·C_{t−1} + (1−m)·q̄` with `m = 0.9`, subtracted
  from the teacher logits.

Sharpening alone pushes toward one-hot outputs; centering alone pushes toward
uniform. Together they hold the representation in between.
