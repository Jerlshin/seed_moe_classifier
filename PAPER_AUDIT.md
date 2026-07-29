# Paper audit

Findings from comparing `../paper/sn-article.pdf` against the implementation,
and what was done about each. Section and equation numbers refer to the paper.

> **Scope.** This document records where the *code* diverged from the *paper as
> written*. It is not a record of the peer-review revision, which deliberately
> departs from the submitted paper in two ways (Top-2 routing, SwinV2-only
> encoder). That is [`REVISION_NOTES.md`](REVISION_NOTES.md). Where the revision
> supersedes a finding below, a note points to it.

---

## 1. The repository did not run

`HierarchicalSeedClassifier.forward` unpacked three values from `MixtureOfExperts`,
which returns a five-field `MoEOutput` NamedTuple, and then called `.squeeze(1)`
on `CrossAttention`'s `CrossAttentionOutput` NamedTuple. Any forward pass raised
`ValueError: too many values to unpack (expected 3)`. The components had been
refactored to structured returns; the builder had not been updated.

**Fixed.** The head now returns a single `HierarchicalOutput` dataclass carrying
every intermediate, so adding a loss term never changes a tuple's arity again.

---

## 2. Architectural divergences from Section 5

### 2.1 The MoE was routed on the wrong tensor — *most significant finding*

Eq. 8 defines `h = Σ_{i∈Top-K} Gᵢ Eᵢ(z)`, where `z` is the DINO embedding.

The code computed `moe(projection_mlp(seed_type_logits))`: the experts consumed a
projection of the 4-D seed-type logits and **never saw the image embedding at
all**. Everything downstream of the MoE was therefore a function of 4 numbers,
not of the 384-D representation the whole pretraining stage exists to produce.

**Fixed.** The MoE is routed on `z`, per Eq. 8.

### 2.2 The residual projected logits instead of probabilities

Eq. 9 is `h' = h + P(p_s)`, and Eq. 6 defines `p_s = softmax(s)`. The code applied
`P` to the raw logits `s`. Unlike probabilities, logits are unbounded, so the
residual's magnitude scaled with stage-1 confidence.

**Fixed.** `P` consumes `p_s`. A test asserts the projection differs from the
logit-fed version, so this cannot silently regress.

### 2.3 ArcFace was implemented but never used as the classifier

Section 5.5 states sub-variety classification uses ArcFace. The code had a
correct, numerically stable `ArcFaceHead` — unused and not even exported. The
actual predictions came from a plain `nn.Linear` inside the sub-variety
classifier, with a *separate* `ArcFaceLoss` holding its own class centres applied
to the embeddings. There were effectively two independent classifiers, and the
reported accuracy came from the non-ArcFace one.

**Fixed.** `ArcFaceHead` is the sub-variety classifier. It returns
`(logits, margin_logits)`: the margin-free logits drive prediction and the KL
term, the margin logits drive the loss. Two consequences worth noting:

* The criterion now holds **no parameters**, so `model.parameters()` is the
  complete optimisation set and `model_state_dict` alone reproduces inference.
  The previous "optimizer must also own `criterion.parameters()`, but gradient
  clipping only covers `model.parameters()`" trap is gone.
* Evaluation passes no labels, so no margin is applied — metrics are not
  inflated by the training-time margin.

### 2.4 `z ∈ ℝ³⁸⁴` was not satisfied

Section 5.1 and Eq. 4 state the encoder produces a 384-dimensional vector. The
config used `feature_dim: 1024`, SwinV2-Base's native width. The two statements
in the paper (SwinV2 backbone; 384-D embedding) cannot both hold without a
projection — no SwinV2 variant emits 384 (Tiny/Small emit 768, Base 1024).

**Resolved** with an explicit `EmbeddingProjection` from the backbone width to
`embed_dim: 384`, so Eq. 4 holds for any SwinV2 variant.

> **Revision update.** That projection now lives inside `DinoV2SwinV2Encoder`
> rather than inside the head, making `encoder(images).shape[-1] == 384` an
> unconditional invariant that the head, the feature dumps, the profiler and the
> baselines all share. See `REVISION_NOTES.md` §3 — including the consequence
> that the projection must be in the optimizer even when the backbone is frozen.

### 2.5 The cosine similarity loss was missing entirely

Section 1: *"we introduce cosine similarity loss within SwinV2's residual
connections, promoting feature compactness."* Not implemented anywhere.

**Added** as `src/losses/cosine.py`, defaulting to the literal reading — keep
`h'` angularly aligned with `h` across the Eq. 9 residual, so the seed-type prior
shifts the representation without rotating it away from what the experts
extracted. An `intra_class` mode (pull embeddings toward their class centroid)
is available for the class-compactness reading discussed in Section 7.

### 2.6 Stage-1 classifier was heavier than Eq. 5 specifies

Eq. 5 says `s = g(z)` with `g` an MLP. The code added a squeeze-excitation branch
and a scalar feature gate.

**Resolved** by making it a variant: `seed_classifier_variant: "mlp"` (the
paper's, now default) or `"se_gated"` (the previous block, kept for ablation).

---

## 3. Loss function issues

### 3.1 L1 sparsity penalised the wrong quantity

Section 5.2: *"Sparsity Loss, enforced via L1 regularization, restricts the
selection to only the top-K most relevant experts."*

The code computed `mean(|gate_probs[top_k]|)` — penalising the *selected*
experts' gate weights. Since the gate is a softmax, this pushes the chosen
experts' confidence **down**, which works against selection rather than
sharpening it, and directly fights the load-balancing term.

**Fixed.** The default `off_topk` mode penalises routing mass landing *outside*
the Top-K selection (`1 - Σ_{i∈Top-K} Gᵢ`, bounded in `[0, 1]`), which is what
"restricts the selection to the top-K" describes. The old behaviour remains as
`moe_sparsity_mode: "topk"` for comparison.

### 3.2 Load-balancing entropy was unbounded

The sign was correct (minimising `Σ u log u` maximises entropy, i.e. balances
utilisation), but the magnitude depended on `log(num_experts)`, so `lambda_load`
silently changed meaning if the expert count changed.

**Changed** to normalise by `log(E)`, bounding the term in `[-1, 0]`:
`-1` = perfectly uniform, `0` = total collapse. Revert with
`normalize_moe_entropy: false`.

### 3.3 ArcFace used `acos`, which is unstable at the boundary

`src/losses/arcface.py` computed `cos(θ + m)` by routing through `torch.acos`,
whose derivative diverges as `cos θ → ±1` — exactly where a well-fit embedding
sits. It also lacked the out-of-range handling for `θ + m > π`.

**Fixed** by consolidating onto `ArcFaceHead`, which expands
`cos(θ+m) = cos θ cos m − sin θ sin m` and applies the standard linear fallback
beyond the monotonic range. A test asserts gradients stay finite at `cos θ = 1`.

### 3.4 An unspecified auxiliary CE term was weighted at 1.0

The combined loss included a plain cross-entropy over sub-variety logits at
`lambda_sub_ce: 1.0`. The paper uses ArcFace alone for sub-varieties.

**Changed** to `lambda_sub_ce: 0.0` (retained as an opt-in aid for early
convergence).

### 3.5 Verified correct, left alone

* **KL direction.** Eq. 10 is `D_KL(P_seed ‖ P_sub)`. `F.kl_div(input=log q,
  target=p)` computes `KL(p‖q)`, and the code passed the aggregated sub-variety
  log-probabilities as `input`. Correct. A test now pins the direction.
* **DINO cross-view pairing, temperature schedule, and centering** all match
  Eqs. 1–3 and Algorithm 1.

---

## 4. Hyperparameters that disagreed with Table 1 / Section 6.1

| Item | Paper | Was | Now |
| --- | --- | --- | --- |
| DINO learning rate | 0.0005 | 0.0001 | 0.0005 |
| Batch size | 16 | 8 | 16 |
| DINO head batch norm | "MLP with batch normalization" | `false` | `true` |
| Colour jitter probability | "standard DINOv2 protocol" | 0.01 | 0.8 |
| Grayscale probability | standard protocol | 0.01 | 0.2 |
| Jitter magnitudes | ±0.4 / ±0.4 / ±0.2 / ±0.1 | hardcoded | configurable, at those values |

The 0.01 jitter probabilities were a deliberate notebook-era choice to preserve
seed colour, and they effectively disabled colour augmentation. That contradicts
the paper's stated protocol, so the defaults now follow the paper. If colour
turns out to be the dominant cue for your sub-varieties, drop them back in
`conf/data/hierarchical_seeds.yaml` — the knobs are still there.

Verified already correct: 300 epochs, gradient clip 3, teacher momentum 0.996,
τ_t 0.02→0.04 over 5 warmup epochs, centering m=0.9, output dim 65,536,
2 global + 4 local crops, crop scales (0.4, 1.0) and (0.05, 0.4), AdamW with
cosine decay, last layer frozen for the first epoch.

---

## 5. Metrics that the paper reports and the code did not compute

Only accuracy and weighted P/R/F1 existed. Added in `src/utils/metrics.py`:

| Paper artefact | Added |
| --- | --- |
| Table 3 — KL alignment rate, 95.94% overall + per seed type | `kl_alignment_rate` |
| Table 2 — per-class precision / recall / F1 | `per_class_metrics` |
| Section 6.2 — "area under the ROC curve" | `roc_auc_ovr` |
| Fig. 10 — confusion matrices | `confusion_matrices` |
| Fig. 11 — sub-variety metric heatmap | `plot_metric_heatmap` |
| Fig. 12 — per-sub-variety misclassification rates | `misclassification_rates` |
| Figs. 8–9 — t-SNE of the embeddings | `tsne_projection` |
| Section 5.2 — expert utilisation | `expert_utilization_counts` |

The paper names the alignment rate but never defines it. The implemented
definition is: a sample is aligned when the parent seed type of the *predicted*
sub-variety equals the *predicted* seed type; the per-type breakdown groups by
*true* seed type, which is what makes a row like "Mustard: 0.7189" a statement
about mustard samples. This is stated in the module docstring.

`roc_auc_ovr` computes AUC per class and macro-averages what is scoreable,
because `sklearn`'s `multi_class="ovr"` requires every one of the 27 classes to
appear in the split — which a stratified validation fold does not guarantee.

---

## 6. Open items requiring your decision

### 6.1 The backbone is frozen in stage 2; the paper fine-tunes it

Section 4 ends: *"the DINOv2-Swin Transformer V2 encoder was integrated into the
proposed hierarchical classification framework and **fine-tuned** on the seed
classification dataset."* This repository trains only the head, with the encoder
in eval mode under `no_grad`.

Left as-is (it is the established, much cheaper recipe) but now a one-line
switch: `model.backbone.freeze=false`. Flipping it makes the run match the paper
at substantially higher memory and time cost.

### 6.2 Dataset folder names do not match the paper

The paper describes **amaranthus** (AMT-1, AMT-2, AMT-4) and mustard
(Jagnath, PM30, Poosa33). The dataset on disk has **Seasame** (VRI1, VRI2, VRI4)
and mustard (Jagnath, PM30, **Unknown1**).

The *structure* matches exactly — 4 seed types with 13 / 8 / 3 / 3 sub-varieties,
9,357 images against the paper's "approximately 9,500" — so all counts, the
hierarchy mapping, and every metric are unaffected. But the paper's Section 3 and
Table 3 name a seed type that is not in this dataset. Either the folder is
mislabelled or the paper is; worth resolving before submission, since Table 3
reports a per-seed-type alignment rate for "Amaranthus".

The code reads class names from the directory tree, so figures and tables will
say whatever the folders say — no code change is needed once the naming is
settled.
