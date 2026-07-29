# CHANGES.md — Independent Architecture and Mathematical Audit

**Subject:** *Hierarchical Deep Learning for Fine-Grained Seed Classification: A Self-Supervised and Mixture-of-Experts Approach* (peer-review revision tree)
**Audit basis:** `00_OVERVIEW.md` – `07_EFFICIENCY_AND_EVALUATION.md`
**Role:** independent AI research scientist / systems architect — unconstrained review
**Date:** 2026-07-29

> **Scope caveat, stated up front.** This audit was conducted against the documentation suite, not against `src/`. The suite is unusually precise — file:line anchors, explicit tensor shapes, config values — which makes most findings verifiable from it alone. Where a finding depends on code the docs describe but do not quote, it is marked **[verify]** and the exact check is given. Every quantitative estimate states its assumptions so you can recompute it.

---

## 1. Executive Summary

### 1.1 Verdict

This is a **well-engineered repository wrapped around an architecture with several load-bearing components that are mathematically inert or actively counterproductive.** The software engineering is genuinely above the norm for an academic tree: one trainer for all variants, a single frozen `HierarchicalOutput` contract, byte-identical splits and encoder checkpoints across the suite, closed-form dormant-parameter accounting, `synchronize()`-correct latency timing, raw-prediction persistence. Those are the things most papers get wrong, and this tree gets them right.

The problems are one level up, in the mathematics.

Three findings are, in my assessment, **paper-blocking** — a competent reviewer will find at least one of them, and each individually undermines a headline claim:

1. **The cross-attention module and the self-attention inside every MoE expert are provably affine maps.** Operating on a pooled length-1 token sequence, `softmax(QKᵀ/√d)` evaluates to the scalar `1` regardless of `Q` and `K`. Eq. 11–12 therefore computes `LayerNorm(W_O W_V h + h′)` — a linear layer plus a residual. The Q and K projections receive **exactly zero gradient** and never move from initialization. Estimated **≈2.07 M parameters** in the model are provably incapable of affecting any output, and they are counted in both the "Total Params" and "Active Params" columns of the results table. Any attention-map figure derived from `attn_weights` is identically 1.0. (**F-03**)

2. **The load-balancing regularizer cannot see the routing it is supposed to balance.** `L_load` is the entropy of the *batch-mean soft gate* `u_i = mean(G_i)`; the actual dispatch is `top-k(G)`. These come apart badly. A router emitting `G = (0.30, 0.30, 0.10, 0.10, 0.10, 0.10)` for every sample scores `L_load = −0.917` on a `[−1, 0]` scale — i.e. **92 % of the way to "perfect balance"** — while top-2 routes *every* sample to experts {0, 1} and the other four experts receive **zero gradient for the entire run**. The standard Shazeer/GShard/Switch auxiliary loss exists precisely because the soft-only form does not control the hard dispatch [1][2]. Worse: `plot_expert_utilization` charts the *hard* fraction, so your own diagnostic figure would show the collapse that your objective reports as healthy. (**F-01**)

3. **The cosine "compactness" loss has global minima that delete the residual branch it regularizes.** `L_cos = 1 − cos(h + P(p_s), h)` reaches exactly 0 when `P(p_s) = 0` — which *is* the `use_residual=False` ablation. Every other minimizer has `P(p_s) = α·h`, a scalar rescaling, which carries at most one degree of freedom of seed-type information instead of 384. Since `L_ArcFace` decays toward 0 during training while `L_cos` does not, the cosine term's *relative* share of the gradient budget grows monotonically — it is weakest exactly when the residual is forming and strongest when it could be dismantled. (**F-07**)

Two further findings are **methodology-blocking** rather than architecture-blocking, and in my view they are the higher practical risk to the paper:

4. **Probable crop-level train/test leakage.** The data root is `Cropped_Samples`. If ~9,357 crops derive from a smaller number of source photographs or physical seed lots, stratified *image-level* splitting places near-duplicate crops on both sides of the boundary. The measured effect of exactly this error in the literature ranges from ~0.08 to 0.43 MCC in OCT imaging [3] and 1.6–2.0 dB PSNR in intrinsic decomposition [4]. `split_dataset` has no group key. (**F-22**)

5. **Single-seed, single-fold reporting cannot support the ablation table.** With `num_folds: 1` and `cfg.seed: 42`, each of the ten variants is one run. On a 1,871-image test set at ~95 % accuracy, the 95 % CI half-width on a *single* accuracy is ±0.99 %, and on a *difference* of two accuracies ±1.40 % — before any training-seed variance is counted. Any ablation gap below roughly 1.4 pp is currently indistinguishable from split noise. (**F-23**)

### 1.2 Systemic risks

Four failure patterns recur across subsystems, and they are more useful than the individual bugs:

| Pattern | Where it shows up | Why it matters |
| --- | --- | --- |
| **Pooled-vector architecture with sequence-model machinery bolted on** | MoE expert self-attention, cross-attention, `num_heads=8` | Three "components" are affine maps. The parameter budget, the FLOP count, and the paper's Eq. 11–12 narrative all overstate what is happening. |
| **Regularizers acting on soft surrogates while the mechanism they target is hard** | `L_load` (soft gate vs. top-k dispatch), `L_sparsity` (nullified by `renormalize_top_k`), `L_cos` (direction only, magnitude free) | Each term can be driven to its optimum without changing the behaviour it exists to control. |
| **Named "one-toggle" ablations that flip more than one factor** | `wo_moe` (3 factors), `wo_arcface` (4 factors), `wo_residual` (2 factors) | Only `wo_kl` and `wo_cross_attn` are clean single-factor ablations. The measured gaps do not attribute to what their names claim. |
| **Method naming ahead of method content** | "DINOv2" implements DINOv1; "cross-attention" is a linear layer; "sparse MoE efficiency" is a ~4 % parameter delta invisible in latency | Reviewer-facing risk. The docs are honest internally (`PAPER_AUDIT.md` culture is excellent); the paper's framing is what's exposed. |

### 1.3 Strategic recommendations, ranked

| # | Action | Effort | Expected effect |
| --- | --- | --- | --- |
| **R1** | **Stop pooling before the head.** Feed SwinV2's final-stage token grid `[B, H·W, C]` to the MoE and cross-attention. This single change makes attention non-degenerate, makes `num_heads` meaningful, recovers ~2.07 M dead parameters as working ones, and is the natural fix for a *fine-grained* task where mean-pooling destroys the texture cues that separate 27 sub-varieties. | High | Converts three inert modules into real ones; likely the largest accuracy gain available. |
| **R2** | **Replace `L_load` with the Shazeer/Switch form `E·Σ fᵢPᵢ`, add router z-loss, add noisy top-k during warm-up.** | Low | Closes the dispatch-blindness hole; z-loss at β=1e-3 is standard and costs nothing [5][6]. |
| **R3** | **Fix the split protocol before rerunning anything:** group-aware splitting by source image/lot, ≥5 seeds, mean ± std, McNemar's exact test for paired variant comparisons on the shared test set. | Low–Medium | Turns the ablation table from anecdote into evidence. This is the highest-value item per hour spent. |
| **R4** | **Rework the loss stack:** log-space KL aggregation (removes a real gradient bug), JSD or detached seed target, AdaCos-scaled ArcFace (`s ≈ 4.6`, not 30) with margin warm-up, and either drop `L_cos(residual)` or move it to intra-class mode with EMA centroids. | Medium | Removes a vanishing-gradient bug and a ~13:1 initial loss-scale imbalance. |
| **R5** | **Rename honestly and add the missing controls:** call stage 1 "DINO-style self-distillation" unless you add iBOT + KoLeo + Sinkhorn; add a frozen-encoder linear-probe baseline; split each multi-factor ablation into its true single factors. | Low | Cheapest defence against the reviews you will otherwise get. |

### 1.4 What I could not assess

The efficiency argument, the metric definitions, the reporting contract, and the figure logic are all sound and I found nothing substantive to change there beyond benchmark statistics (F-27). I could not evaluate: actual measured accuracies (no results in the suite), the true image-count-per-source-photo ratio (decides F-22's severity), whether images are square (decides F-26), or the stage-2 learning rate (the docs give the baselines' `3e-5` only relatively).

---

## 2. Architectural & Mathematical Audit Findings

Severity: **C** = paper-blocking · **H** = high · **M** = medium · **L** = low / hygiene.

### 2.1 Routing Mechanics and Sparse Computation

*Sources: [`03_MOE_MODULE.md`], [`06_ABLATION_ENGINE.md`], [`00_OVERVIEW.md`] §7.2*

---

#### **F-01 [C] — `L_load` regularizes the soft gate; the model routes on the hard top-k. The two can be simultaneously optimal and catastrophic.**

[`03_MOE_MODULE.md`] §3 defines

```
u_i = mean_batch(G_i),   L_load = Σ_i u_i log u_i / log E  ∈ [−1, 0]
```

with `−1` documented as "perfectly uniform utilization". But `u` is the mean of the **full softmax** `G`, whereas the model's behaviour is determined by `topk(G, 2)`. `L_load` never observes `top_k_indices`.

**Concrete failure, computable by hand.** Let the router emit the same distribution for every sample:

```
G = (0.30, 0.30, 0.10, 0.10, 0.10, 0.10)

−Σ uᵢ ln uᵢ = −[2(0.3)ln 0.3 + 4(0.1)ln 0.1] = 0.72238 + 0.92103 = 1.64342
log E = ln 6 = 1.79176
L_load = −1.64342 / 1.79176 = −0.9172
```

The regularizer reports **91.7 % of maximal balance**. Meanwhile `topk(G, 2)` selects `{0, 1}` for every sample in every batch, so the true dispatch fraction is `f = (0.5, 0.5, 0, 0, 0, 0)` and **experts 2–5 receive no gradient, ever**. Four of six experts — 3.95 M parameters — are dead, and the objective is 92 % satisfied.

**The degenerate case is worse.** At `L_load`'s *global optimum*, `G = (1/6, …, 1/6)` exactly. `torch.topk` on tied values returns the lowest indices, so top-2 deterministically selects `{0, 1}`. **The global minimum of the load-balancing loss produces maximally imbalanced hard routing.** This is not a hypothetical corner: it is the exact point the regularizer is pulling toward.

**This is a known-solved problem and the solution was not used.** Shazeer et al. (2017), GShard, and Switch Transformer all couple the *hard* dispatch fraction to the *soft* probability [1][2][5]:

```
L_aux = α · E · Σ_i f_i · P_i,   f_i = (1/T) Σ_x 𝟙[i ∈ topk(x)],   P_i = (1/T) Σ_x G_i(x)
```

`f_i` is non-differentiable; `P_i` carries the gradient; the product means an *over-dispatched* expert gets its router probability pushed down [1]. The current formulation has only the `P` half.

**Aggravating factor.** [`07_EFFICIENCY_AND_EVALUATION.md`] §5 notes `plot_expert_utilization` draws its reference line at `1/num_experts` and computes `expert_utilization_counts(top_k_indices)` — the *hard* fraction `f`. So the figure measures `f` and the loss measures `P`. Your diagnostic and your objective disagree about what "balanced" means, and the diagnostic is the correct one.

---

#### **F-02 [H] — Under `renormalize_top_k=True`, `L_sparsity` cannot change the model's function. It is a pure router-entropy penalty.**

`_sparse_forward` computes `h = Σ_{i∈TopK} w̃_i E_i(z)` with `w̃ = w / Σw`. Let `S = Σ_{i∈TopK} G_i`. Then:

- `L_sparsity = mean(1 − S)`.
- `w̃` is **invariant** to any rescaling of `G` restricted to the top-k set.
- The selection is invariant to any transformation preserving the ordering.

Therefore there exists a descent direction for `L_sparsity` — move mass from the off-top-k experts onto the top-k set while holding `G_0/G_1` fixed — along which **`h` does not change at all**. The term has a large null space with respect to the model output. Its only reliable effect is to reduce the router's entropy.

That is precisely the direction that (a) removes what little exploration deterministic top-k routing has, (b) makes selection more brittle, and (c) drives `u_i → 0` for the unselected experts, which **directly maximizes** `L_load`. [`03_MOE_MODULE.md`] frames this as productive tension ("load-balancing wants the *batch* spread; sparsity wants each *sample* decisive"). In the standard formulation that framing is right. Here it is not, because renormalization already delivers the convex combination that per-sample decisiveness was supposed to buy.

**`renormalize_top_k=True` and `λ_sparsity > 0` are mutually redundant. Pick one.**

---

#### **F-03 [C] — Attention over a length-1 sequence is an affine map. `Q` and `K` are dead weights in six experts and the cross-attention module.**

Both `TransformerExpert` ([`03_MOE_MODULE.md`] §1) and `CrossAttention` ([`04_HIERARCHICAL_FUSION.md`] §4) call `nn.MultiheadAttention` with `[batch, 1, dim]` tensors. With a key/value sequence of length 1:

```
softmax(QKᵀ/√d) = softmax([s]) = [1]   for any scalar s
⇒ a = 1 · V = W_O (W_V x)              — Q and K appear nowhere in the output
⇒ ∂a/∂W_Q = ∂a/∂W_K ≡ 0                — exactly zero gradient, forever
```

[`04_HIERARCHICAL_FUSION.md`] §4 already states this ("cross-attention consequently reduces to a learned linear projection of `h` plus a `LayerNorm`-wrapped residual"), which is intellectually honest — but the consequences are not carried through:

**(a) Dead parameter count.** For `nn.MultiheadAttention(embed_dim=384, num_heads=8)`, the packed `in_proj_weight` is `[3·384, 384]` with `in_proj_bias [3·384]`. The `Q` and `K` slices are `2·384² + 2·384 = 295,680` parameters.

| Module | Instances | Dead params |
| --- | --- | --- |
| `TransformerExpert` (MoE) | 6 | 1,774,080 |
| `CrossAttention` (`variant="paper"`) | 1 | 295,680 |
| **Total** | | **≈ 2.07 M** |

These are counted in `ParameterReport.total`, in `active = total − dormant`, and therefore in the paper's **"Total Params (M)" and "Active Params (M)"** columns. Of the 3.95 M "saved" by moving Top-4 → Top-2, roughly **1.18 M was never doing anything** (4 dormant experts × 295,680).

**(b) `num_heads=8` is inert.** Splitting a single token across 8 heads is a no-op on the attention weights.

**(c) `attn_weights` is identically 1.0.** `CrossAttentionOutput.attn_weights` is exposed through `HierarchicalOutput`. Any attention-map visualization derived from it is a constant. If the paper contains such a figure, it should be removed.

**(d) Each "transformer expert" is really a 2-block MLP.** `LayerNorm(x + Linear(x)) → MLP → LayerNorm`. This is fine as an architecture — it is not fine as a description.

**(e) The `variant="gated"` cross-attention has the same problem**, since it wraps the same degenerate `MultiheadAttention`.

---

#### **F-04 [H] — No exploration mechanism. Deterministic top-k routing on a frozen encoder is a rich-get-richer process with no counterweight.**

`topk` has zero gradient almost everywhere; nothing in the loss says "this sample should have gone to expert 4". Shazeer et al. (2017) introduced **noisy top-k gating** — Gaussian noise on the router logits, scaled by a learned softplus term — specifically to manufacture exploration in a process that otherwise has none [1][7]. Modern LLM MoEs mostly drop the noise, but they pay for it with a *hard-dispatch-aware* auxiliary loss [7] — which, per F-01, this tree does not have. **It has neither mechanism.**

This is compounded by the frozen-encoder recipe: `z` is fixed by stage 1, so the router is fitting a static feature distribution. There is no representation drift to shake a collapsed router loose.

[`03_MOE_MODULE.md`] asserts: *"`MoERegularization`'s entropy term is what pulls utilization back toward uniform over an epoch, so no expert stays unrouted for long."* Per F-01, this claim is false in general — the entropy term can be near-optimal while four experts stay permanently unrouted. **Recommend removing this claim from the documentation** and replacing it with a measured dead-expert count.

---

#### **F-05 [H] — Sparse and dense dispatch are *not* equivalent under AdamW, despite the forward-equality test.**

`tests/test_moe_layer.py` asserts `_sparse_forward` and `_dense_forward` produce identical output. True for the forward pass — but not for the optimizer step:

- Under `_sparse_forward`, an unrouted expert never enters the autograd graph, so `p.grad is None`. PyTorch's AdamW skips such parameters **entirely** — including decoupled weight decay, and including the moment-buffer updates.
- Under `_dense_forward`, every expert receives a (zero) gradient, so decay **is** applied and moments decay toward zero.

Two consequences: (i) the "debug-only" dense path trains a measurably different model, so it cannot be used to validate a sparse run; (ii) rarely-routed experts carry **stale Adam moments** across long gaps — when they are finally selected, the first-moment direction they receive was computed against a network state many steps old. **[verify]** confirm `zero_grad(set_to_none=True)` (the ≥2.0 default) is in use; if `set_to_none=False`, the behaviour inverts and decay *does* apply.

**Batch-size interaction.** At `batch_size=16` (Table 1) and `K=2`, one step fills **32 routing slots across 6 experts** — 5.33 expected samples per expert. `L_load` is therefore an entropy estimated from 16 samples over 6 bins, recomputed independently every step. That is a very noisy control signal for a self-reinforcing process. [`03_MOE_MODULE.md`]'s own worked example uses a batch of 12; the config uses 16. Neither is enough.

---

#### **F-06 [H] — `DenseExpertBlock` is not capacity-matched to Top-2, so `wo_moe` does not isolate routing.**

[`03_MOE_MODULE.md`] §2 and [`06_ABLATION_ENGINE.md`] §2 both argue that substituting one `DenseExpertBlock` (rather than deleting the layer) makes the `wo_moe` gap attributable "to the gate alone", because the block is architecturally identical to one expert.

The full model activates **two** experts per sample. The ablation activates **one** block. The `wo_moe` gap therefore conflates *routing* with a **2× reduction in active capacity and active FLOPs** — the very quantity the efficiency section is built on. The stated rationale ("deleting the experts outright would also delete a transformer block's worth of capacity") identifies the right hazard and then only half-corrects for it.

The correct controls are one or both of:

- **Capacity-matched dense:** a `DenseExpertBlock` with `2×` the MLP width, so active parameters match Top-2.
- **Fixed-router control:** all six experts present, routing replaced by a **fixed random hash** of the sample index (cf. Hash Layers; GShard uses random secondary routing for related reasons [31]) or by uniform Top-2. This isolates *learned* routing from *sparse capacity* — a distinction the current suite cannot make at all.

---

### 2.2 Hierarchical Fusion and Feature Alignment

*Sources: [`04_HIERARCHICAL_FUSION.md`], [`00_OVERVIEW.md`] §2–3*

---

#### **F-07 [C] — `L_cos` in `mode="residual"` is minimized by destroying the residual it regularizes.**

From [`05_LOSS_FUNCTIONS.md`] §6, with `h′ = h + P(p_s)` (Eq. 9):

```
L_cos = 1 − cos(h′, h) = 1 − cos(h + P(p_s), h)
```

Characterize the minimizer set. `L_cos = 0` ⟺ `h′ ∥ h` with positive coefficient ⟺

```
P(p_s) = α(x)·h(x)   for some α(x) ≥ −1,   including the α = 0 case P(p_s) = 0
```

So **every global minimizer either zeroes the residual outright — which is literally the `use_residual=False` ablation — or collapses it to a scalar rescaling of `h`.** In the second case Eq. 9 carries **one** scalar degree of freedom of seed-type information instead of 384. In both cases the coarse-to-fine link the architecture exists to provide is gone.

The docs' stated intent is the opposite: *"this lets the seed-type prior shift the representation without rotating it away from what the MoE extracted — the mechanism that stops the residual from overwriting the MoE output when stage 1 is confidently wrong."* But cosine is **invariant to magnitude**, so it does not constrain *how much* the residual shifts — only its direction. The cheapest way to preserve direction is to make the residual small. The loss achieves "not rotating" by achieving "not shifting."

**The weighting makes this worse over time, not better.** `λ_cosine = 0.1` looks small against `λ_arcface = 1.0`. But `L_ArcFace` decays toward 0 as the model fits, while `L_cos` has no reason to. The cosine term's *share* of the total gradient grows monotonically through training — it is weakest during the epochs when the residual is forming and strongest during the epochs when it could be dismantled.

**Additional interaction with F-30:** when `use_residual=False`, `projected_seed = zeros_like(moe.features)` makes `L_cos = 1 − cos(h, h) = 0` identically. The docs present this as elegant (avoids `NaN`). It is — but it also means **`wo_residual` silently removes two terms**: Eq. 9 *and* the entire Section-1 cosine contribution. See F-30.

---

#### **F-08 [H] — At convergence the Eq. 9 residual degenerates to a 4-entry codebook, and its gradient path into stage 1 vanishes.**

`P` maps `p_s = softmax(s) ∈ Δ³` into `ℝ³⁸⁴`. As the seed-type classifier fits its own cross-entropy (Eq. 7 is supervised with hard labels and only 4 classes — it will fit early and hard), `p_s → e_c` one-hot. Then:

```
P(p_s) → P(e_c) ∈ {v₁, v₂, v₃, v₄}
```

Four fixed vectors. The Eq. 9 fusion becomes **an embedding lookup with 4 entries**, implemented by a 2-layer MLP (`depth=2`, ~75 k parameters). Whatever `depth` is set to — the docs note the paper's wording implies 1, the config uses 2, an earlier tree used 3 — the *function class actually realized at convergence* is a 4-row table.

**And the gradient dies with it.** The Jacobian of softmax is `∂p_s/∂s = diag(p) − ppᵀ`, whose norm → 0 as `p` → one-hot. So the sub-variety branch's gradient into the seed-type classifier **vanishes exactly when the seed head becomes confident**. The claim of "joint optimization stability between stages" holds only during the transient. After that, `L_seed` and `L_KL` are the only live paths into stage 1, and the fine task effectively stops informing the coarse one.

Note this is *not* fixed by the KL term, which also flows through softmax outputs.

---

#### **F-09 [M] — The "hierarchical cascade" is structurally two parallel heads plus a rank-≤4 coupling.**

[`00_OVERVIEW.md`] §2's diagram shows a cascade. The dataflow is:

```
z ──┬── SeedTypeClassifier ── s ── p_s ── P(p_s) ─┐
    └── MixtureOfExperts ─────────── h ───────────┴─(+)─→ h′
```

Both branches read the **same** `z`. `PAPER_AUDIT.md` §2.1 correctly insists the MoE route on `z` rather than on a projection of `s` (an earlier version routed on the seed logits, meaning the experts never saw the image). That fix was right. But the consequence is that **the router has no access to the coarse prediction at all**, so expert specialization has no structural reason to align with seed type. Combined with F-08, the total coarse→fine coupling in the forward pass is one of four fixed vectors.

This is worth stating plainly because the paper's contribution is "hierarchical MoE." Right now the MoE is not hierarchical; it is a flat router sitting beside a coarse classifier. **A one-line change makes the name true** — see S-09.

---

#### **F-10 [H] — ArcFace `s = 30` is ≈6.5× the analytically motivated scale for 27 classes, and dominates the objective ~13:1 at initialization.**

AdaCos (Zhang et al., CVPR 2019) derives the fixed optimal scale for a `C`-class cosine-softmax as `s̃_f ≈ √2·log(C−1)` [8][9]:

```
C = 27  ⇒  s̃_f = √2 · ln 26 = 1.4142 × 3.2581 = 4.61
```

The config uses `arcface_scale: 30.0` — the value ArcFace tuned for face recognition with 10⁵–10⁶ identities. **It is 6.5× too large for this problem.**

**Initialization-time consequence, computed.** For random 384-D embeddings, `cos θ ≈ 0` (std ≈ 1/√384 ≈ 0.051). With `m = 0.5`:

```
target logit  = 30·cos(θ+m) ≈ 30·(0·cos 0.5 − 1·sin 0.5) = 30·(−0.4794) = −14.38
other logits  ≈ 0
L_ArcFace     = log(1 + 26·e^{14.38}) ≈ 14.38 + ln 26 = 17.64
L_seed        = ln 4 = 1.386
```

**Ratio ≈ 12.7 : 1**, at `λ_arcface = λ_seed = 1.0`. The angular-margin term consumes essentially the entire gradient budget for the first phase of training. `L_KL` and `L_cos` are, at that point, rounding error. The suite's fixed-λ scheme has no mechanism to notice this.

**Downstream contamination.** `sub_logits = 30·cos θ ∈ [−30, 30]` also feeds:
- the KL aggregation (`softmax` of a ±30 logit vector is near one-hot — see F-13),
- `sub_scores` in `test_predictions.npz` (so any calibration analysis is meaningless without temperature scaling),
- `roc_auc_ovr` (rank-invariant, so unaffected — noted for completeness).

ArcFace is also known to have convergence trouble at `m = 0.5` on small backbones and small datasets; CurricularFace reports outright divergence (NaN at ~2,400 steps) for MobileFaceNet at `m = 0.5` on CASIA-WebFace, and convergence at `m = 0.45` [10]. Applying a full margin from step 0 with no warm-up is against current practice [10][11].

---

#### **F-11 [M] — `LinearSubVarietyHead` changes four things, not one, so `wo_arcface` is not a margin ablation.**

[`04_HIERARCHICAL_FUSION.md`] §7 presents `LinearSubVarietyHead` as a clean drop-in because it mirrors the `(logits, margin_logits)` contract. Structurally, yes. Semantically it changes:

| | `ArcFaceHead` | `LinearSubVarietyHead` |
| --- | --- | --- |
| Embedding | L2-normalized | unnormalized |
| Class centres | L2-normalized | unnormalized `nn.Linear` weights |
| Logit range | `[−30, 30]` (`s·cos θ`) | unbounded |
| Target margin | `cos(θ+m)` | none |

So `wo_arcface` simultaneously removes the margin, removes hypersphere normalization, and changes the softmax temperature by an unbounded factor — which in turn changes the sharpness of `P_sub` feeding the KL term and the geometry the t-SNE panels visualize. **The measured gap is not attributable to the angular margin.** The correct single-factor control is a normalized cosine head with `m = 0` (NormFace / cosine-softmax) — see S-11.

---

### 2.3 Multi-Objective Loss Dynamics

*Sources: [`05_LOSS_FUNCTIONS.md`], [`03_MOE_MODULE.md`] §3*

---

#### **F-12 [C] — The KL term's `clamp_min(1e-8)` zeroes the gradient exactly in the cases the term exists to correct.**

From [`05_LOSS_FUNCTIONS.md`] §4:

```python
aggregated = sub_probs @ mapping_matrix
return F.kl_div(torch.log(aggregated.clamp_min(1e-8)), seed_probs, reduction="batchmean")
```

`clamp_min` has **zero gradient in the clamped region**. So whenever an aggregated seed-type probability falls below `1e-8`, the KL term contributes exactly nothing to that entry — no gradient, silently.

**This is not a rare edge case; it is the common case.** `sub_logits = 30·cos θ ∈ [−30, 30]` (F-10). A cosine gap of 1.0 between the argmax sub-variety and a competitor becomes a **logit gap of 30**, i.e. a probability ratio of `e³⁰ ≈ 1.07 × 10¹³`. Aggregating 27 such probabilities into 4 bins leaves the non-argmax bins at `~10⁻¹³`–`10⁻¹⁰`, comfortably below the clamp. Concretely: the term is live when the two heads agree (where it has nothing to do) and dead when they disagree confidently (where it is the entire point).

**The direction check is correct and the aggregation is correct.** `F.kl_div(input=log q, target=p)` computing `KL(p‖q)`, the `[27,4]` one-hot `M`, the runtime derivation from the directory tree, the buffer registration, the insistence on `sub_logits` over `sub_margin_logits` — all right, all worth keeping. The bug is purely the `clamp → log` composition, and it is invisible: no NaN, no warning, just a silently absent gradient.

The fix is exact, not a workaround: aggregate in log-space with `logsumexp` (S-12).

---

#### **F-13 [H] — The distribution the KL term measures is saturated by the ArcFace scale. The two terms are coupled through `s` with no analysis.**

Even with F-12 fixed, `P_sub = softmax(30·cos θ)` is a near-one-hot distribution by construction. `KL(P_seed ‖ P_sub-agg)` between a moderately soft 4-way distribution and a near-degenerate one is dominated by whichever mass `P_seed` places on bins where `P_sub-agg` ≈ 0 — it becomes an enormous, high-variance quantity that behaves less like a consistency regularizer and more like a hard constraint with an exploding coefficient.

`λ_kl = 1.0` was chosen without reference to `s`. Changing `arcface_scale` silently changes the effective strength of the hierarchy term. These are two hyperparameters that look independent in `conf/` and are not.

---

#### **F-14 [H] — With `detach_kl_seed_target=false`, the coarse head can reduce `L_KL` by becoming *less* accurate.**

`L_KL = KL(P_seed ‖ P_sub-agg)` with gradient flowing into **both** branches (the documented default). Two problems:

1. **Bidirectional gradient permits the wrong solution.** The term is symmetric in *who moves*. `P_seed` is already supervised by hard labels through `L_seed` (Eq. 7). Letting `L_KL` also push `P_seed` means the coarse head trades label-fit for agreement with the fine head — including agreement with a *wrong* fine prediction. Since Eq. 7 fits fast (4 classes), the marginal gradient available to `L_KL` on the seed side is disproportionately spent on the hard, ambiguous samples, i.e. exactly the ones where following the sub-variety head is most likely to be wrong.

2. **`KL(p‖q)` is mode-covering in `q`.** It is zero-avoiding: wherever `P_seed` has mass, `P_sub-agg` is forced to have mass. An uncertain coarse head therefore forces the fine head to hedge *across seed types* — smearing exactly the fine-grained decisions the 27-way task depends on.

**The literature has converged the other way.** Hierarchy-consistency work now favours **symmetric Jensen–Shannon divergence** between coarse predictions and marginalized fine predictions — HAF (Garg et al., 2022) [12] and subsequent hierarchical-classification work [13] both use JSD explicitly, on the grounds that one-directional KL "compromises the fine-grained accuracy." A recent hierarchy-aware VLM study reports that tree-path KL **alone** produces severe accuracy drops (e.g. 85.1 → 4.6 on CUB-200) because it "over-penalizes fine-grained confusion" [14]. Your term is stabilized by `L_seed` and `L_ArcFace`, so you will not see that magnitude — but the direction of the effect is documented and points the same way.

---

#### **F-15 [H] — Seven fixed λ over terms whose initial magnitudes differ by ~13×, with no adaptive weighting and no per-term gradient telemetry.**

The stage-2 objective has seven terms with hand-set weights. From F-10, at initialization:

| Term | Value at init | λ | Weighted |
| --- | --- | --- | --- |
| `L_ArcFace` | ≈ 17.64 | 1.0 | **17.64** |
| `L_seed` | 1.386 | 1.0 | 1.386 |
| `L_KL` | small (both near-uniform) | 1.0 | ~0.1 |
| `L_load` | ≈ −1 (uniform init) | 0.01 | −0.01 |
| `L_sparsity` | ≈ 0.67 at uniform gate | 0.01 | 0.007 |
| `L_cos` | ≈ 0 (small random `P`) | 0.1 | ~0 |

One term carries ~92 % of the initial loss. Fixed coefficients are known to let one task dominate and prevent the others converging, and the optimal weights change over training so fixed coefficients are "rarely optimal across the full training run" [15]. The standard remedies — homoscedastic uncertainty weighting (Kendall et al., 2018), GradNorm (Chen et al., 2018), PCGrad, MGDA — are all well-established [15][16][17]; none is used here, and there is no evidence in the docs that the λ values were tuned rather than defaulted.

This matters more than usual because the ablation suite **changes which terms are active** (`wo_kl` removes one, `wo_moe` zeroes two — see F-29) without renormalizing the others. Variants are therefore compared at different effective learning rates for the terms they share.

---

#### **F-16 [M] — `cosine_mode="intra_class"` is statistically near-empty at `batch_size=16` over 27 classes.**

The documented behaviour: singleton classes contribute exactly 0 (an embedding is its own centroid). With `n = 16` samples drawn over `C = 27` roughly-balanced classes, the expected number of classes with ≥2 members is

```
C·[1 − (1−p)ⁿ − n·p·(1−p)ⁿ⁻¹],  p = 1/27
 = 27·[1 − 0.5468 − 0.3365] = 27 × 0.1168 ≈ 3.2 classes
```

so **roughly 6 of 16 embeddings contribute a nonzero term** and the other 10 contribute exactly zero — and the "centroid" each of those 6 is pulled toward is estimated from **2 samples**. This is not a compactness loss; it is noise with a small mean.

This is the more interesting mode of the two (it is what the paper's phrase "feature compactness" actually means, and it complements ArcFace's inter-class separation with the intra-class term ArcFace lacks). It just needs **EMA class centroids** rather than per-batch ones — the standard construction from center loss (Wen et al., 2016), and the same reasoning that motivates sub-center ArcFace's multiple prototypes [18].

---

#### **F-17 [M] — Stage 2 clips gradients but does not log them; stage 1 does.**

[`02_BACKBONE_AND_SSL.md`] §6 logs gradient norms every step during pretraining. [`07_EFFICIENCY_AND_EVALUATION.md`] §3's `run_epoch` performs `backward()` + clipping + `step()` with no equivalent. For a **seven-term** objective, per-term gradient-norm telemetry (`‖∂L_i/∂θ_shared‖` at the last shared layer) is the single highest-value diagnostic available, and it is the direct evidence needed to confirm or refute F-07, F-10, F-14, and F-15 empirically rather than analytically.

---

### 2.4 Self-Supervised Pretraining and Feature Extraction

*Sources: [`02_BACKBONE_AND_SSL.md`], [`01_DATA_PIPELINE.md`] §3*

---

#### **F-18 [H] — The stage-1 method is DINOv1. It is named DINOv2 throughout.**

`CustomDINOLoss` implements cross-view CE with softmax + EMA centering and a temperature schedule — that is Caron et al. (2021). DINOv2 (Oquab et al., 2023) is defined by four additions, **none of which appear** [19][20][21]:

| DINOv2 component | Present? |
| --- | --- |
| iBOT patch-level masked-prediction objective | No |
| KoLeo regularizer (uniform feature span within a batch) | No |
| Sinkhorn–Knopp centering replacing softmax-centering | No — uses DINOv1 EMA centering |
| Untied image-level / patch-level head weights | N/A (only one head) |

The class is even called `DINO`, and `publish_shared_backbone` writes `dinov2_swinv2_pretrained.pth`. Internally the docs are consistent; externally the paper claims a method it does not implement. DINOv2's own ablations attribute **~3 %** to the iBOT term for dense tasks and **>8 %** on retrieval to KoLeo [19] — so this is not a cosmetic naming issue, it is a missing-method issue with a known magnitude.

Two clean resolutions: rename to "DINO-style self-distillation," or implement KoLeo + Sinkhorn (both are ~30 lines and neither requires patch tokens). See S-18.

---

#### **F-19 [H] — Both collapse guards are mis-set in the collapsing direction simultaneously.**

DINO prevents collapse by balancing two opposing forces: **sharpening** (low teacher temperature → toward one-hot) and **centering** (subtract EMA mean → toward uniform). [`02_BACKBONE_AND_SSL.md`] §5 states this correctly. The configured values push *both* toward collapse:

**(a) Sharpening is ~2× stronger than reference.** Config: `warmup_teacher_temp 0.02 → teacher_temp 0.04`. DINO: `0.04 → 0.07` [22][23]. A teacher at τ = 0.04 is roughly twice as sharp as DINO's converged 0.07, for the whole run.

**(b) Centering is noise-dominated.** `C ∈ ℝ⁶⁵⁵³⁶` is an EMA (`m = 0.9`) of the batch mean of teacher outputs. The teacher sees **2 global crops × batch 16 = 32 vectors per step**; `m = 0.9` gives an effective window of ~10 steps:

| | This tree | DINO reference |
| --- | --- | --- |
| Teacher vectors per step | 32 | 2,048 (batch 1024 × 2) |
| Effective samples in `C` | ~320 | ~20,480 |
| Samples per estimated dimension | **0.005** | 0.31 |

The centering vector is estimated from **1/64th** the effective sample size DINO uses, for the same 65,536 dimensions. The counterweight to sharpening is essentially noise.

The two effects compound. If stage 1 is producing weak features, this is the first place to look — and the loss curve (Fig. 6) will *not* reveal it, since a partially collapsed DINO run has a perfectly plausible-looking loss curve.

---

#### **F-20 [H] — 65,536 prototypes for 9,357 images: 7 prototypes per training image, and 16.8 M parameters in one layer.**

`out_dim = 65536` is taken from Table 1, which is taken from DINO, which set it for **ImageNet-1k (1.28 M images)**.

```
This tree:  65,536 / 9,357     =  7.00  prototypes per image
DINO:       65,536 / 1,281,167 =  0.051 prototypes per image
Ratio:      137×
```

Parameter accounting for `DINOHead` (input 1024 for SwinV2-Base):

| Layer | Params |
| --- | --- |
| `Linear(1024, 2048)` | 2,099,200 |
| `Linear(2048, 2048)` | 4,196,352 |
| `Linear(2048, 256)` | 524,544 |
| `weight_norm(Linear(256, 65536))` | **16,777,216** |
| **Total** | **≈ 23.6 M** |

The prototype layer alone is **71 % of the head** and ~19 % of the student network (backbone ≈ 86.9 M). Total training exposure is `300 × 9,357 ≈ 2.81 M` image presentations — about **2 %** of DINO's 100-epoch ImageNet budget — against **137×** the prototype density. `out_dim` in the 4,096–8,192 range is the standard choice for datasets of this size and is what `lightly`-based small-data DINO setups typically use.

---

#### **F-21 [M] — Reference-implementation drift: four smaller deviations, each individually minor.**

1. **Constant teacher momentum.** `momentum_teacher: 0.996` fixed. DINO cosine-anneals `0.996 → 1.0` [22][24][25]; the point is a fast-adapting teacher early and a stable target late. A constant 0.996 gives you neither end of that trade-off.
2. **Constant weight decay.** `weight_decay: 0.01` fixed. DINO cosine-schedules `0.04 → 0.4` [22][23].
3. **Local crops are upsampled 2.53×, creating a resolution shortcut.** [`01_DATA_PIPELINE.md`] §3: local crops are taken at `local_crop_size = 101` px then **resized back up to 256** because SwinV2's shifted windows need a fixed resolution. So every local view carries a systematic low-pass signature that global views do not. The student can distinguish local from global views from blur alone — a shortcut that partially substitutes for the local-to-global correspondence the multi-crop objective is supposed to teach. This is aggravated by the blur probabilities themselves differing by view (global-1: 1.0, global-2: 0.1, local: 0.5). **[verify]** — `BackboneFeatureExtractor` passes `dynamic_img_size=True` to `timm`, which appears to contradict the documented fixed-resolution requirement; if `dynamic_img_size` genuinely works for this SwinV2 variant, local crops can stay at native resolution and the shortcut disappears.
4. **`DINOHead` uses `BatchNorm1d`; the teacher is a `deepcopy` advanced by `update_momentum`, which EMAs *parameters*, not *buffers*.** So teacher and student BN running statistics diverge — and they see different numbers of views (2 vs 6), so their batch statistics differ even in principle. DINO's official head defaults to `use_bn=False` for transformer backbones for related reasons.
5. **The same-view skip is positionally fragile.** `if student_index == teacher_index: continue` yields the correct `2×6 − 2 = 10` cross-view terms **only if** the student's first two views are the two globals in the teacher's order. Nothing enforces that invariant; if `_concat_outputs` ordering ever changes, you would silently skip a global-local pair and include a same-view pair, with no error.

**One thing the docs get exactly right and should keep:** cancelling last-layer gradients *after* clipping and *before* `step()`. That matches the official DINO ordering. (Minor note: `clip_grad_norm_` computes the total norm including gradients that are about to be zeroed, so the clip coefficient is inflated during epoch 1 — the reference implementation has the same property, so this is a footnote, not a bug.)

---

### 2.5 Data Pipeline and Evaluation Methodology

*Sources: [`01_DATA_PIPELINE.md`], [`07_EFFICIENCY_AND_EVALUATION.md`]*

---

#### **F-22 [C] — `split_dataset` has no group key. The data root is `Cropped_Samples`.**

`stratification_labels` returns `seed·1000 + sub` and `train_test_split`/`StratifiedKFold` partition at the **image level**. If the ~9,357 crops derive from a smaller set of source photographs, seed lots, imaging sessions, or plates, then near-duplicate crops of the same physical object land on both sides of the train/test boundary, and the reported test accuracy measures memorization of source-specific cues (lighting, background, sensor noise, the same individual seed) rather than sub-variety discrimination.

The empirical magnitude of this exact error, measured under controlled ablations:

- OCT classification: per-image vs. per-volume/subject splitting inflates mean MCC by **0.07–0.43** across four datasets [3].
- Intrinsic image decomposition: frame-level vs. scene-level splitting inflates test PSNR by **1.6–2.0 dB**, `p < 0.01` across three architectures [4].
- Core-sample lithology with overlapping sliding-window patches: random splitting produced an "extraordinary" 93.7 % that the authors then excluded from their results table as an artifact [26].
- Leukemia cytology: the authors attribute the field's near-perfect published results on C-NMC 2019 to patient-level leakage and demonstrate it with a controlled ablation [27].

The standard remedy is a group key — `Agriculture-Vision` assigns every crop to the split of the farmland image it came from, "guarantee[ing] that no cropped images from the same farmland will appear in multiple splits" [28].

**Severity depends entirely on a number the docs don't contain: crops per source image.** If it is 1, this finding is void. If it is 10–50, the headline accuracy is not measuring what the paper says it measures. **Determine this first, before any other change in this document.**

---

#### **F-23 [C] — One seed, one fold. The ablation table's resolution is worse than most of the gaps it needs to report.**

`cfg.seed: 42` (a single value), `num_folds: 1` (the documented default), one run per variant, one `summary.json` per variant. The suite is scrupulous about making the split *identical* across variants — which is exactly right, and makes paired testing available — but there is no repetition.

**Statistical resolution, computed.** Test set = `0.2 × 9,357 = 1,871` images. At `p ≈ 0.95`:

```
SE(p̂)        = √(0.95 × 0.05 / 1871)  = 0.00504  → ±0.99 pp  (95 % CI half-width)
SE(p̂₁ − p̂₂)  = √2 × 0.00504           = 0.00713  → ±1.40 pp  (unpaired difference)
```

So **any ablation gap below ~1.4 pp is inside the noise floor of the test split alone** — before adding training-seed variance (dropout, shuffling, router initialization, and for MoE specifically, which experts happen to win the early race — see F-04). For a 27-class fine-grained task where component contributions of 0.5–2 pp are the normal magnitude, this is not enough resolution to support the table.

Two things fix it, and both are cheap:
- **Repetition:** ≥5 seeds per variant, report mean ± std. This is 50 runs for the ablation suite; `run_suite` already handles failure isolation and `suite_manifest.json` already records per-run configuration, so the harness needs almost no change.
- **Paired testing:** because all variants share the byte-identical test split, **McNemar's exact test** on the paired prediction vectors is available, is more powerful than comparing independent CIs, and is the standard test for comparing two classifiers on the same sample. `test_predictions.npz` already stores everything needed.

---

#### **F-24 [H] — Selecting the best fold and reporting its test metrics is optimistically biased when `num_folds > 1`.**

[`07_EFFICIENCY_AND_EVALUATION.md`] §3: `profile_run` measures "the **best** checkpoint (lowest validation loss across all folds)", and `write_run_summary` writes that checkpoint's test evaluation. Taking a maximum over `K` folds and reporting the corresponding test score is a selection procedure whose expected value exceeds the expected single-fold test score. With `num_folds: 1` (default) this is inert — but the moment anyone runs `num_folds=5` for the variance F-23 asks for, the numbers become optimistic **and** incomparable to the `num_folds=1` numbers already collected.

Report **mean ± std of test metrics across folds**, and keep best-fold selection only for the deployed artifact.

---

#### **F-25 [M] — Stage-2 augmentation is effectively nil, and the stated justification does not support it.**

`get_supervised_transforms` is `Resize → RandomHorizontalFlip(p=0.0) → ToTensor → Normalize`. At the default flip probability of **0.0**, stage-2 training sees each image exactly once per epoch, deterministically.

The documented rationale: *"by stage 2 the representation is already invariant from DINO pretraining, and the fine-grained visual cues that separate 27 sub-varieties are exactly the ones heavy augmentation would destroy."* The second half is a good argument against *heavy* augmentation. The first half does not follow: SSL invariance of the **encoder** says nothing about the sample efficiency of the **head**, which is where ~9 M freshly-initialized parameters (MoE + SubVarietyEmbedding + ArcFace centres) are being fit from ~7,486 training images. A mild `RandomResizedCrop(scale=(0.8,1.0))` plus flip is standard for exactly this setting and does not destroy fine texture.

**Related gap: no class balancing.** The hierarchy is 13 rice + 8 millet + 3 + 3 sub-varieties, so seed-type accuracy is structurally dominated by rice. `class_distribution()` exists and Fig. 1 plots it, but nothing consumes it — no balanced sampler, no class-weighted loss, no effective-number reweighting. Macro-F1 *is* reported (good), but the model is not trained for it.

---

#### **F-26 [L] [verify] — `Resize(image_size, BICUBIC)` with an integer preserves aspect ratio.**

`torchvision.transforms.Resize(256)` resizes the **shorter side** to 256 and preserves aspect ratio; there is no `CenterCrop` in the documented stage-2 pipeline. Non-square inputs would therefore produce variable-width tensors and `default_collate` would raise on the first mixed batch. Either every image under `Cropped_Samples` is already square (likely, given the name) — in which case this is fine and worth a one-line comment — or this is a latent crash. Confirm and use `Resize((H, W))` explicitly either way.

---

#### **F-27 [H] — The efficiency claim is ~4 % of parameters and ~2 % for the headline Top-4→Top-2 change, it does not appear in latency, and the batch-32 benchmark measures a routing pattern that cannot occur.**

The accounting machinery is genuinely good — closed-form `dormant_parameters()`, real ATen FLOP counting rather than a hand formula, `synchronize()` before and after the timed loop, untimed warm-up, MPS's current-vs-peak caveat surfaced rather than hidden. My concerns are about what the numbers *mean*, not how they are obtained.

**(a) Magnitude.** Per-expert parameters for `TransformerExpert(384, mlp=512, heads=8)`:

```
MultiheadAttention   591,360   (in_proj 442,368 + 1,152; out_proj 147,456 + 384)
LayerNorm × 2            1,536
MLP 384→512→384        394,112
                     ─────────
per expert             987,008
6 experts            5,922,048
dormant @ K=2        3,948,032      dormant @ K=4  1,974,016
```

Against an estimated full model of ~96.5 M (SwinV2-Base ≈ 86.9 M + Eq. 4 projection ≈ 0.39 M + head ≈ 9.2 M):

```
dormant_fraction @ K=2  ≈  4.1 %
Top-4 → Top-2 saving    ≈  1.97 M  =  2.0 % of total parameters
```

A **2 %** parameter delta is a thin foundation for "the revision's central efficiency claim." It is also ~60 % illusory: **1.18 M of the 1.97 M "saved" is the dead Q/K projections from F-03**, which were never computed in the first place.

**(b) It will not show up in latency, and the profiler is honest enough to reveal that.** `profile_model` correctly measures encoder + head together, and the frozen 86.9 M SwinV2-Base dominates. Meanwhile `_sparse_forward` replaces one batched matmul with up to six small gather → matmul → scatter-add sequences; at these batch sizes the kernel-launch overhead plausibly makes sparse dispatch *slower* than dense. The paper should state that the saving is in **parameters and FLOPs, not wall-clock**, rather than letting the "Inference Latency (ms)" column sit next to "Active Params (M)" and imply a relationship.

**(c) The batch-32 measurement is a degenerate case.** `_resize_batch(example_input, batch_size)` **tiles** the example to reach the target batch. Identical rows produce identical gate logits, hence identical top-2 selections, hence **exactly 2 expert kernels launched for all 32 rows** — instead of the up-to-6 that real, diverse data produces. The benchmark therefore systematically *understates* the dispatch overhead of sparse routing at larger batches. Tile with real, distinct samples from the dataset.

**(d) Benchmark statistics.** `warmup=3, iterations=10` yields a mean with no dispersion estimate. Use ≥50 timed iterations, report **median and IQR**, and record device/driver/clock state. Ten iterations on a contended GPU is not a measurement anyone should build a table row on.

---

#### **F-28 [L] — Two reporting-contract nits worth closing.**

1. **`summary.json` cannot distinguish `wo_kl` from `full_model`.** `component_flags()` deliberately reports only the four *architectural* booleans, excluding `use_kl_loss`. So the `wo_kl` run's `component_flags` field is byte-identical to `full_model`'s; only the variant *name* separates them. For a tree whose whole design philosophy is "a run must always leave a machine-readable trace," add a `loss_flags` block carrying `use_kl_loss` and the seven λ values.

2. **The stratification rationale in [`01_DATA_PIPELINE.md`] §2 is provably false.** The doc states the composite key is needed because "stratifying on sub-variety alone would not guarantee seed-type balance." But sub-variety labels are **global** (0..26) and each sub-variety has exactly one parent, so `seed = parent(sub)` is a deterministic function of `sub`. The map `sub ↦ seed·1000 + sub` is a **bijection**, so the composite key induces *exactly* the same partition into strata as `sub` alone. The code is correct; the justification is not. (Also worth a guard: `StratifiedKFold` fails if any sub-variety has fewer than `n_splits` members.)

---

### 2.6 Ablation and Baseline Validity

*Sources: [`06_ABLATION_ENGINE.md`]*

The harness design — one trainer, subprocess isolation, shared checkpoint enforcement, rightmost-override semantics, `suite_manifest.json` — is the strongest part of the repository and I would change none of it. The problem is what the toggles *mean*.

---

#### **F-29 [H] — `wo_moe` flips three factors, not one.**

`model.head.use_moe=false` simultaneously:

1. removes **learned routing** (the intended factor),
2. halves **active capacity**: 2 experts → 1 dense block (F-06),
3. zeroes **both MoE regularizers** — the docs state this explicitly and treat it as a feature ("Both MoE regularizers evaluate to exactly zero on this degenerate gate… so no downstream consumer needs a special case"). Elegant plumbing, but it means the `wo_moe` variant optimizes a **five-term** objective while the full model optimizes seven.

The reported `wo_moe` gap is the sum of all three effects with no way to separate them.

---

#### **F-30 [H] — `wo_residual` flips two factors: it removes Eq. 9 *and* the entire cosine loss.**

`use_residual=false` sets `projected_seed = zeros_like(moe.features)`, so `refined_features = moe.features` and `L_cos = 1 − cos(h, h) = 0` **identically, for every sample, for the whole run**. The variant therefore removes both Eq. 9 and the paper's Section-1 contribution in one toggle. Given F-07 (the cosine term's minimizers destroy the residual anyway), this specific confound could plausibly make `wo_residual` look *better* than the full model — and if it does, the current suite offers no way to explain why.

---

#### **F-31 [H] — Only two of six "one-toggle" ablations are actually single-factor.**

| Variant | Intended factor | Factors actually changed | Clean? |
| --- | --- | --- | --- |
| `wo_moe` | routing | routing + active capacity + 2 regularizers | **No** (F-29) |
| `wo_arcface` | angular margin | margin + embedding L2-norm + centre L2-norm + logit scale (→ KL sharpness, t-SNE geometry) | **No** (F-11) |
| `wo_residual` | Eq. 9 fusion | Eq. 9 + `L_cos` | **No** (F-30) |
| `wo_kl` | Eq. 10 | Eq. 10 | **Yes** |
| `wo_cross_attn` | Eqs. 11–12 | Eqs. 11–12 (which per F-03 is a linear layer, so the ablation removes a linear layer) | **Yes** |
| `full_model` | — | — | reference |

[`06_ABLATION_ENGINE.md`] §1 states the design goal precisely — *"an ablation routed through a second training loop would differ from the full model in ways nobody intentionally chose… and the measured gap between variants would silently include all of that noise."* That reasoning is exactly right and is exactly what F-29/F-30/F-11 describe, one level down: the *config* differences nobody intentionally chose. The principle is sound; it needs to be applied to the toggle semantics as well as the trainer.

`ablation_flat_classifier` is handled correctly — the docs are explicit that it measures a **combined** effect and it is deliberately excluded from `run_ablations.py`. Keep that framing in the paper.

---

#### **F-32 [M] — The baseline suite is missing its most important member, and the two end-to-end baselines are probably under-tuned.**

**(a) No linear probe.** The three baselines answer "conventional CNN?", "shifted-window family held constant?", and "hierarchical at all?" — but not the question a reviewer asks first: *does any of this head machinery beat a linear layer on the same frozen features?* `hierarchical_cce` is not that control: it keeps `use_residual: true`, so it retains the coarse-to-fine link and the `SubVarietyEmbedding` MLP. It is a point in the ablation lattice (`wo_moe` + `wo_arcface` + `wo_cross_attn` + `wo_kl` composed), not an independent baseline. **Add `linear_probe`:** frozen DINO encoder → `Linear(384, 4)` and `Linear(384, 27)`, plain CE. If the full architecture does not clear it by a comfortable, seed-stable margin, that is the single most important number in the paper.

**(b) Weak-baseline risk.** `resnet50` and `swin_tiny` train end-to-end at `3e-5` — one value, no per-model sweep — while the proposed model uses a different default. End-to-end ImageNet fine-tuning and frozen-encoder head training have genuinely different optimal learning rates, and a single shared value cannot be right for both. A three-point lr sweep per baseline (`1e-5, 3e-5, 1e-4`), reporting each baseline's best, costs six extra runs and removes the most common reviewer objection to any "our method wins" table.

**(c) Missing: supervised SwinV2-Base.** The suite has no baseline that isolates *"in-domain self-supervised pretraining"* from *"the architecture."* An ImageNet-initialized SwinV2-Base with the same hierarchical head, no DINO stage, would do exactly that. The `swinv2`-only validation in `validate_swinv2_name` currently blocks this path through `model/backbone`; it would need a baseline experiment file.

**(d) Good decision worth keeping:** withholding the DINOv2 checkpoint from `resnet50`/`swin_tiny` (`spec_checkpoint = None`) is correct — a SwinV2 DINO state dict would at best be ignored and at worst partially loaded. Keep it, and keep `ensure_pretrained_checkpoint`'s refusal to start without byte-identical encoder weights.

---

## 3. Proposed Solutions and Research Enhancements

Each solution is keyed to the finding(s) it closes. Code is illustrative sketch, not drop-in patch. Where I recommend one option over another, the rejected alternative is named.

### 3.1 Routing and Sparse Computation

---

#### **S-01 → F-01, F-04 — Replace the entropy load loss with the dispatch-aware form; add router z-loss and warm-up noise.**

**Load balancing.** Adopt the Shazeer/GShard/Switch auxiliary loss, which couples the hard dispatch fraction to the differentiable router probability [1][2][5]:

$$
\mathcal{L}_{\text{load}} = E \sum_{i=1}^{E} f_i \, P_i,
\qquad
f_i = \frac{1}{T}\sum_{x} \mathbb{1}\!\left[i \in \text{Top-}K(x)\right],
\qquad
P_i = \frac{1}{T}\sum_{x} G_i(x)
$$

Minimum value **1** at uniform routing (so report `L_load − 1` if you want a zero-floor). `f` is non-differentiable and acts as a per-expert coefficient; the gradient flows through `P`, pushing down the router probability of *over-dispatched* experts specifically [1].

```python
def switch_load_balancing_loss(gate_probs, top_k_indices, num_experts):
    # gate_probs [B, E], top_k_indices [B, K]
    mask = torch.zeros_like(gate_probs).scatter_(1, top_k_indices, 1.0)
    f = mask.mean(dim=0)              # hard dispatch fraction, no grad path
    P = gate_probs.mean(dim=0)        # differentiable
    return num_experts * (f.detach() * P).sum()
```

*Rejected alternative:* keeping the entropy form and adding a separate hard-dispatch penalty. That doubles the hyperparameters for no benefit — `f·P` already encodes both quantities in one term, and it is the formulation every reference implementation uses, which matters for reviewer familiarity.

*Migration note:* keep the current entropy term available as `load_mode="entropy"` for a direct ablation. `L_load(entropy)` vs `L_load(switch)` on the same split is a publishable mini-result, and it makes F-01 a contribution rather than a correction.

**Router z-loss** (Zoph et al., ST-MoE 2022) at the standard weight `β = 1e-3` [5][6]:

$$
\mathcal{L}_{z} = \frac{1}{B}\sum_{i=1}^{B}\Big(\log \sum_{j=1}^{E} e^{x_j^{(i)}}\Big)^{2}
$$

on the pre-softmax router logits. Prevents logit growth, is reported to improve stability with no quality cost, and is exposed as a first-class hyperparameter in production MoE stacks [5][6][21].

**Exploration.** Add noisy top-k gating for the first `N` epochs, annealing the noise to zero (Shazeer et al. 2017) [1][7]:

$$
H(x)_i = (x W_g)_i + \varepsilon \cdot \mathcal{N}(0,1)\cdot \text{softplus}\big((x W_n)_i\big),
\qquad \varepsilon: 1 \to 0
$$

Also initialize `W_g` near zero so early routing is close to uniform, and log a **dead-expert counter** (`#{i : f_i = 0}` per epoch) as a first-class metric.

---

#### **S-02 → F-02 — Resolve the sparsity/renormalization contradiction. Recommendation: drop `L_sparsity`.**

Two coherent configurations exist; the current one is neither:

| Option | Config | Rationale |
| --- | --- | --- |
| **A (recommended)** | `renormalize_top_k=True`, `λ_sparsity = 0` | Renormalization already gives a convex combination over the selection. The sparsity term then has a null space w.r.t. the output (F-02) and only reduces router entropy, fighting `L_load`. |
| **B** | `renormalize_top_k=False`, `λ_sparsity > 0` | Without renormalization the *total selected mass* scales `h`, so concentrating mass on the top-K genuinely changes the function and the penalty is meaningful. |

Option A, because B makes `‖h‖` depend on router confidence, which then propagates into the ArcFace embedding norm and interacts badly with F-10. If you keep `L_sparsity` at all, keep it only as an ablation axis with the paper reporting Option A as the method.

---

#### **S-03 → F-03, and the largest expected accuracy gain — keep the token grid.**

Stop mean-pooling before the head. Take SwinV2's final-stage output `[B, H·W, C]` (for `swinv2_base_window16_256` at 256 px this is an `8×8 = 64`-token grid at `C = 1024`), project to `[B, 64, 384]`, and route/attend over it.

This single change:

| Effect | Detail |
| --- | --- |
| Attention becomes real | `softmax(QKᵀ/√d)` over 64 keys is a genuine distribution; `num_heads=8` becomes meaningful; the ~2.07 M dead Q/K parameters start working (F-03). |
| `attn_weights` becomes a figure | An 8×8 spatial attention map over the seed image is a *publishable* visualization instead of a constant. |
| Eq. 11–12 becomes literally true | The paper's cross-attention formulation is honest as written. |
| Fine-grained cues survive | Mean-pooling a 64-token grid to one 1024-D vector before a **fine-grained 27-way** task discards precisely the localized texture/shape evidence that separates rice sub-varieties. |
| Routing granularity | Per-token routing raises routing slots per step from `16 × 2 = 32` to `16 × 64 × 2 = 2,048`, which directly repairs the load-estimate noise in F-05 and makes the MoE behave like the sparse MoEs the theory is written for. |

Pooling then happens **after** the head (attention pooling or GeM), or the `[CLS]`-equivalent token is used. `dynamic_img_size=True` is already passed to `timm`, and `BackboneFeatureExtractor._pool` already distinguishes `[B,H,W,C]` from `[B,L,C]` — the plumbing is half-built.

*Rejected alternative:* keeping pooled vectors and simply deleting the attention modules. That is cheaper and strictly better than the status quo (it removes 2.07 M dead parameters and stops overclaiming), but it also removes the paper's Eq. 11–12 contribution entirely. Deletion is the honest fallback if S-03 is out of budget; **do one or the other, not neither.**

---

#### **S-04 → F-05 — Optimizer parity and estimator stability.**

1. **Make sparse and dense dispatch equivalent under the optimizer**, not just in the forward pass: after `backward()`, materialize zero grads for unrouted expert parameters (`for p in expert.parameters(): if p.grad is None: p.grad = torch.zeros_like(p)`), so decoupled weight decay and moment decay apply identically. Then extend `tests/test_moe_layer.py` to assert *parameter-state* equivalence after one optimizer step, not only forward equivalence.
2. **EMA-smooth the utilization statistic** used by the auxiliary loss: `u_ema ← 0.9·u_ema + 0.1·u_batch`, applying the loss to `u_ema`. A 16-sample estimate of a 6-bin distribution recomputed every step is too noisy to steer a self-reinforcing process.
3. **Gradient accumulation** (4–8 steps) to raise the effective routing batch, if S-03's per-token routing is not adopted.

---

#### **S-05 → F-06, F-29 — Two new controls that make `wo_moe` interpretable.**

| New variant | Override | Isolates |
| --- | --- | --- |
| `wo_moe_capacity_matched` | `use_moe=false`, `moe_hidden_dim=1024` | routing, **holding active capacity fixed** |
| `moe_fixed_router` | `router_mode=hash` (fixed hash of sample index → 2 experts) | **learned** routing vs. sparse capacity |
| `moe_uniform_router` | `router_mode=uniform` (all 6 experts, weight 1/6) | routing vs. ensembling |

`moe_fixed_router` is the important one: it is the only configuration that can tell you whether the router learned anything, and it is a direct answer to the question a reviewer will ask about Section 5.2.

**Add an expert-specialization metric.** Utilization bars show *balance*, not *specialization*. Report normalized mutual information between routing and labels:

$$
\text{NMI}\big(\text{Top-}1\text{ expert} \;;\; \text{seed type}\big),
\qquad
\text{NMI}\big(\text{Top-}1\text{ expert} \;;\; \text{sub-variety}\big)
$$

`test_predictions.npz` already stores `expert_indices` alongside the labels, so this is a scoring-time addition with no retraining — it can be computed on existing runs today.

---

### 3.2 Hierarchical Fusion

---

#### **S-06 → F-07 — Retire `cosine_mode="residual"` as the default; use a gated residual plus a real intra-class term.**

The paper's stated goal is *"feature compactness."* Compactness is an **intra-class** property. The residual-alignment formulation does not measure it and, per F-07, its minimizers delete the residual. Two replacements, used together:

**(a) Control residual magnitude structurally, not through the loss.** Replace the free additive residual with **LayerScale** (Touvron et al., 2021) — a learned per-channel gain initialized small:

$$
h' = h + \gamma \odot P(p_s), \qquad \gamma \in \mathbb{R}^{384},\; \gamma_{\text{init}} = 10^{-4}
$$

This gives the "start small, grow only if it helps" behaviour the cosine penalty was reaching for, without a loss term whose optimum is `P = 0`. If you want an explicit penalty, penalize the **magnitude ratio** with a hinge that is inactive in the healthy regime:

$$
\mathcal{L}_{\text{res}} = \max\!\left(0,\; \frac{\lVert P(p_s)\rVert_2}{\lVert h\rVert_2} - \tau\right)^{2}, \qquad \tau \approx 0.5
$$

which bounds the residual's influence *without* rewarding its disappearance.

**(b) Make the compactness term the intra-class one, with EMA centroids** (S-08).

*Rejected alternative:* keeping `mode="residual"` at a smaller `λ_cosine`. This only delays the problem — the term's minimizer is unchanged, and its relative weight still grows as `L_ArcFace` decays.

---

#### **S-07 → F-08, F-09 — Replace the additive `P(p_s)` residual with FiLM conditioning on the seed head's hidden state.**

The current fusion has two coupled defects: at convergence it is a 4-entry codebook, and its gradient to stage 1 vanishes through the saturating softmax. Both are fixed by conditioning on a **pre-softmax, higher-dimensional** representation and by using **multiplicative** modulation (FiLM, Perez et al. 2018):

$$
(\gamma, \beta) = \text{MLP}\big(g_{\text{hidden}}(z)\big), \qquad h' = \gamma \odot h + \beta
$$

where `g_hidden(z)` is the `SeedTypeClassifier`'s hidden activation (192-D at `hidden_ratio=0.5`) rather than the 4-D `p_s`. Properties:

- **No saturation.** The Jacobian does not collapse as the seed head becomes confident, so the fine loss keeps informing the coarse branch for the whole run (fixes F-08's vanishing path).
- **Capacity.** Conditioning information is `2 × 384` dimensions modulated by a 192-D hidden state, not 4 discrete choices.
- **Strictly more expressive** than additive: setting `γ = 1` recovers the current form exactly, so this is a superset and can be ablated against it (`fusion_mode: {additive, film}`).
- **Preserves `PAPER_AUDIT.md` §2.2's invariant in spirit.** The audit's rule was "project probabilities, not logits, because logits are unbounded and would make the residual scale with confidence." FiLM on a hidden state addresses the same hazard structurally: `γ` is bounded by construction (e.g. `1 + tanh(·)`), so magnitude does not track confidence.

Keep `p_s` computed and exposed in `HierarchicalOutput` regardless — the KL term and the alignment metric both need it.

---

#### **S-08 → F-09 — Make the MoE hierarchical: condition the router on the coarse prediction.**

One line changes the architecture from "flat router beside a coarse head" to the hierarchical MoE the title claims:

```python
gate_logits = self.gate(torch.cat([embedding, seed_type_probs.detach()], dim=-1))
```

The `detach()` is deliberate: routing should be *informed* by the coarse prediction without the router's gradient reshaping the coarse head (which would reintroduce an F-14-style incentive). The experts still consume `z` — `PAPER_AUDIT.md` §2.1's invariant that the experts must see the image is preserved exactly. Only the **gate** sees `p_s`.

Then verify with S-05's NMI metric: if `NMI(expert ; seed type)` rises materially against the unconditioned router, "expert specialization" becomes a measured claim rather than an asserted one.

---

#### **S-09 → F-10, F-11 — Scale, margin schedule, and a true margin ablation.**

1. **Set the scale analytically.** `arcface_scale: 4.6` for `C = 27` (`√2·ln 26`), per AdaCos [8][9]. Better: implement **dynamic AdaCos**, which is hyperparameter-free and re-derives `s` from the running median target angle each step [8][30] — this removes the parameter rather than retuning it, which is a cleaner story for a paper than "we changed 30 to 4.6."
2. **Warm up the margin:** `m: 0 → 0.5` linearly over the first ~15 % of epochs. ArcFace at full margin from step 0 is a documented convergence hazard on small backbones and small datasets [10][11]; CurricularFace reports outright divergence at `m = 0.5` where `m = 0.45` converges [10].
3. **Add the true single-factor margin control** (F-11): a `use_arcface=false` variant that keeps L2 normalization and the scale but sets `m = 0` (NormFace / cosine-softmax). Then `wo_arcface` measures the margin, and the current linear-head variant becomes a separate, honestly-labelled `wo_angular_head` measuring normalization + margin together.
4. **Optional, if label noise is suspected:** sub-center ArcFace with `K = 3` sub-centres per class [18][29]. Sub-centre ArcFace was built for exactly the case where one class prototype cannot explain intra-class variability — plausible if a "sub-variety" spans multiple growing conditions or imaging sessions. Cheap to try (`27 × 3` centres instead of 27) and it doubles as a data-cleaning tool.

---

### 3.3 Loss Stack

---

#### **S-10 → F-12, F-13 — Log-space hierarchical aggregation. This is an exact fix, not a tolerance tweak.**

Replace the `clamp → log` composition with `logsumexp` over each parent's children. The mapping matrix `M` becomes a boolean children mask; the arithmetic is exact and the gradient never vanishes:

```python
def hierarchical_kl_loss(seed_logits, sub_logits, children_mask,   # [4, 27] bool
                         tau_kl=1.0, detach_seed_target=True):
    log_p_sub = F.log_softmax(sub_logits / tau_kl, dim=-1)         # [B, 27]
    # log P_agg[b, c] = logsumexp_{j in children(c)} log_p_sub[b, j]
    masked = log_p_sub.unsqueeze(1).masked_fill(~children_mask.unsqueeze(0), float("-inf"))
    log_p_agg = torch.logsumexp(masked, dim=-1)                    # [B, 4], exact
    p_seed = F.softmax(seed_logits, dim=-1)
    if detach_seed_target:
        p_seed = p_seed.detach()
    return F.kl_div(log_p_agg, p_seed, reduction="batchmean")
```

Why this is strictly better than raising the clamp floor: `logsumexp` is numerically stable by construction (max-subtraction), needs no epsilon, and has a well-conditioned gradient across the entire probability range — including the confident-disagreement region where the current implementation is silently dead (F-12).

**Add `tau_kl` and decouple it from `s` (F-13).** With `arcface_scale = 30`, set `tau_kl ≈ s` so the KL branch sees `cos θ` rather than `30·cos θ`. If S-09 lowers `s` to ~4.6, `tau_kl = 1.0` becomes reasonable — but keep the knob so the two hyperparameters stop being secretly coupled.

`build_subvariety_seed_mapping` keeps its current signature; only the returned tensor's interpretation changes from one-hot float to boolean mask. Keep the runtime derivation from `get_subvariety_to_seed_type()` — never hardcode it.

---

#### **S-11 → F-14 — Flip the KL default to detached, and offer JSD.**

**Default change:** `detach_kl_seed_target: false → true`. The seed head is already supervised by hard labels; the KL term's job is to reshape the *fine* distribution toward hierarchical consistency, not to let the coarse head negotiate. This is a one-character config change with a real effect on what the term measures.

**Offer the symmetric alternative** as `kl_mode: jsd`, which is what the hierarchy-consistency literature has converged on [12][13]:

$$
\mathcal{L}_{\text{JS}} = \tfrac{1}{2}D_{\text{KL}}(P_{\text{seed}} \,\Vert\, M) + \tfrac{1}{2}D_{\text{KL}}(P_{\text{sub-agg}} \,\Vert\, M), \qquad M = \tfrac{1}{2}(P_{\text{seed}} + P_{\text{sub-agg}})
$$

JSD is bounded by `log 2`, symmetric, and does not have the zero-avoiding behaviour that forces the fine head to hedge across seed types (F-14). Report `{kl_forward, kl_detached, jsd}` as a three-way comparison — this converts a correctness concern into a small empirical contribution.

*Rejected alternative:* replacing the soft consistency term with a structural conditional factorization `P(sub) = P(seed)·P(sub | seed)`. This enforces consistency by construction (alignment rate becomes 1.0 identically) and is theoretically cleaner — but it would make the paper's Table 3 alignment-rate metric vacuous, and it discards the ArcFace head's flat 27-way geometry. Not worth the rewrite here.

---

#### **S-12 → F-15 — Replace fixed λ with homoscedastic uncertainty weighting for the three primary terms.**

Keep fixed λ for the three regularizers (`load`, `sparsity`, `cos`) — they are genuinely auxiliary and small. Learn the weights for the three **task** terms (`seed`, `arcface`, `kl`) via Kendall et al.'s formulation [15][17]:

$$
\mathcal{L} = \sum_{t \in \{\text{seed},\,\text{arc},\,\text{kl}\}} \left(\frac{1}{2\sigma_t^{2}}\mathcal{L}_t + \tfrac{1}{2}\log \sigma_t^{2}\right) + \sum_{r} \lambda_r \mathcal{L}_r
$$

optimizing `log σ_t²` directly (clamped for stability). This eliminates hand-tuned task weights, adapts as `L_ArcFace` decays, and — importantly for this repository — the learned `σ_t` are **diagnostic**: if `σ_arcface` collapses while `σ_kl` explodes, F-15's dominance hypothesis is confirmed empirically.

*Rejected alternative:* GradNorm. It is arguably better at identifying under-trained tasks [16], but it needs per-task gradient access at a shared layer every step — a real intrusion into a training loop whose single-code-path design is this repository's best feature. Uncertainty weighting adds three scalars and touches nothing else.

Whichever is chosen, **it must be identical across all variants**, or the ablation gaps become gaps in loss-weighting policy.

---

#### **S-13 → F-16 — EMA class centroids for the intra-class cosine term.**

```python
# buffer: centroids [27, 384], L2-normalized, momentum 0.9
with torch.no_grad():
    for c in labels.unique():
        batch_mean = F.normalize(normalized[labels == c].mean(0), dim=-1)
        centroids[c] = F.normalize(0.9 * centroids[c] + 0.1 * batch_mean, dim=-1)
loss = 1.0 - (normalized * centroids[labels]).sum(-1).mean()
```

Every sample now contributes (fixing the ~60 %-empty batch in F-16) and centroids are estimated from the whole training history rather than 2 samples. This is center loss (Wen et al., 2016) adapted to the hypersphere, and it supplies the intra-class compactness that ArcFace's inter-class margin does not — the actual complementarity the paper's Section 1 claims.

---

#### **S-14 → F-17 — Per-term gradient telemetry.**

Log, every `log_every_steps`, at the last shared parameter (the Eq. 4 projection or the MoE output):

```
grad_norm/{seed, arcface, kl, cosine, moe_load, moe_sparsity}
grad_cosine/{arcface_vs_kl, arcface_vs_seed, cosine_vs_arcface}
```

The pairwise **gradient cosine similarities** are the direct empirical test for the conflicts hypothesized in F-07 (does `L_cos` oppose `L_ArcFace`?) and F-14 (does `L_KL` oppose `L_seed`?). Persistently negative cosine between two terms is the signal to apply gradient surgery (PCGrad) or reweight [16][17]. Costs one extra backward per logged step — restrict to every 50th step.

---

### 3.4 Self-Supervised Pretraining

---

#### **S-15 → F-18 — Either rename, or close the gap. Both are defensible; do one.**

**Option A (low cost, recommended if the revision deadline is near).** Rename to "DINO-style self-distillation" in the paper, the class names, and `dinov2_swinv2_pretrained.pth`. State plainly that stage 1 follows Caron et al. (2021) with a SwinV2 trunk. Nothing in the results changes; the reviewer risk disappears.

**Option B (moderate cost, better paper).** Add the two DINOv2 components that do **not** require patch tokens:

- **KoLeo regularizer** [19][20] — encourages uniform feature span within a batch; ~10 lines; DINOv2's ablation credits it with >8 % on instance retrieval and no cost elsewhere. Particularly apt for a *fine-grained* task where near-duplicate sub-varieties must not collapse onto each other.
- **Sinkhorn–Knopp centering** [19][21] — replaces softmax + EMA centering with 3 SK iterations over the batch. This directly addresses F-19(b): SK normalizes to a doubly-stochastic assignment *within the batch*, so it does not depend on a 65,536-dimensional running mean estimated from 320 samples.

The iBOT patch-level term requires token-level outputs and therefore pairs naturally with S-03. If you adopt S-03, Option B plus iBOT becomes reachable and the "DINOv2" name becomes accurate.

---

#### **S-16 → F-19, F-20, F-21 — Recalibrate stage 1 for a 9.4 k-image dataset.**

| Parameter | Current | Proposed | Rationale |
| --- | --- | --- | --- |
| `out_dim` | 65,536 | **8,192** | 7 prototypes/image → 0.88. Removes ~14.7 M parameters from the head (F-20). |
| `warmup_teacher_temp → teacher_temp` | 0.02 → 0.04 | **0.04 → 0.07** | Match DINO; current setting is ~2× sharper, and sharpening is the collapse-inducing force (F-19a) [22][23]. |
| `center_momentum` | 0.9 | **0.99** (or adopt Sinkhorn, S-15) | Effective window 10 → 100 steps ≈ 3,200 samples. Still under-determined at `out_dim=65536`; adequate at 8,192 (F-19b). |
| `momentum_teacher` | 0.996 constant | **cosine 0.996 → 1.0** | DINO's schedule; fast early adaptation, stable late targets (F-21.1) [22][24][25]. |
| `weight_decay` | 0.01 constant | **cosine 0.04 → 0.4** | DINO's schedule (F-21.2) [22][23]. |
| Effective batch | 16 | **≥ 64** via gradient accumulation | Every collapse guard in DINO is a batch statistic. |
| `DINOHead` norm | `BatchNorm1d` | **none or LayerNorm** | Removes teacher/student batch-statistic coupling and the EMA-buffer staleness in F-21.4. |

**Local-crop resolution (F-21.3).** Confirm whether `dynamic_img_size=True` genuinely permits variable input resolution for the configured SwinV2 variant. If yes, feed local crops at their native 101 px and delete `resize_local_to_global` — this removes the blur/resolution shortcut. If no, document the confound explicitly and consider matching the upsampling artifact across all views (e.g. downsample-then-upsample the globals identically) so the shortcut carries no discriminative signal.

**Same-view skip (F-21.5).** Replace positional matching with explicit view identifiers carried alongside the tensors, and add a test asserting exactly 10 cross-view terms with 2 global + 4 local crops.

---

### 3.5 Data and Evaluation Protocol

---

#### **S-17 → F-22 — Group-aware splitting. Do this before anything else in this document.**

**Step 1 — measure the exposure.** Count distinct source images / seed lots / capture sessions behind the 9,357 crops. If the filenames encode provenance (`IMG_0413_crop07.png`), the group key already exists on disk.

**Step 2 — if crops-per-source > 1, switch to grouped splitting:**

```python
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

groups = np.array([source_id(path) for path, _, _ in dataset.samples])
test_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
# then StratifiedGroupKFold over the remainder, stratifying on sub-variety
```

**Step 3 — quantify the leakage as a result, don't just fix it silently.** Run the full model under both protocols and report the delta. A "leakage ablation" table is a genuine methodological contribution and is now standard practice in careful applied-DL papers [3][4][27]. If the gap is large, you have turned a fatal reviewer objection into a section of the paper.

`groups` should also be persisted into `split_manifest.npz` alongside `test_indices`.

---

#### **S-18 → F-23, F-24 — Variance, and the right significance test.**

1. **`SEEDS = [42, 43, 44, 45, 46]`** in `run_ablations.py` / `run_baselines.py`; `VariantSpec` gains a `seed` field; `outputs/ablations/{variant}/seed{n}/`. `run_suite`'s failure isolation and `suite_manifest.json` already handle the extra runs.
2. **Report `mean ± std`** in `summary_metrics.csv` — `collect_run_summaries` already globs two directory levels, which is exactly the depth a `seed{n}/` subdirectory adds.
3. **McNemar's exact test** for each variant against `full_model`. All variants share the byte-identical test split (the suite's own strongest guarantee), so the predictions are paired and McNemar is both valid and more powerful than comparing independent intervals:

```python
from statsmodels.stats.contingency_tables import mcnemar
# n01 = full correct & variant wrong ; n10 = full wrong & variant correct
result = mcnemar([[n00, n01], [n10, n11]], exact=True)
```

Add a `p (vs full)` column to `REQUESTED_COLUMNS`. Apply Holm–Bonferroni across the six comparisons.

4. **Report `mean ± std` across folds, not the best fold** (F-24), keeping best-fold selection only for the artifact that gets profiled and shipped.

---

#### **S-19 → F-25, F-26, F-27, F-28 — Protocol and reporting hygiene.**

- **Augmentation:** default `horizontal_flip_prob: 0.5` and add `RandomResizedCrop(256, scale=(0.8, 1.0))` for stage-2 training only. Ablate it — "does stage-2 augmentation help when the encoder is frozen?" is a legitimate question and the answer belongs in the paper rather than in a config default.
- **Class balance:** add `sampler: {none, balanced}` and `class_weights: {none, effective_number}`. Report per-seed-type macro-F1 alongside overall, so rice does not silently dominate.
- **`Resize`:** make it `Resize((image_size, image_size))` explicitly, or `Resize(image_size) → CenterCrop(image_size)`. State which in the docs (F-26).
- **Benchmarking:** `iterations: 50`, report **median + IQR**, and tile `_resize_batch` from **distinct dataset samples** so batch-32 routing is realistic (F-27c). Record device, driver, and whether clocks were locked.
- **Efficiency framing:** state explicitly in the paper that the Top-4→Top-2 saving is ~2 % of total parameters and is **not** expected to appear in wall-clock latency, because a frozen 86.9 M backbone dominates and sparse dispatch trades a batched matmul for several small ones. An honest small number beats an implied large one.
- **`summary.json`:** add a `loss_flags` block (`use_kl_loss`, all seven λ, `cosine_mode`, `moe_sparsity_mode`, `detach_kl_seed_target`) so `wo_kl` is machine-distinguishable from `full_model` (F-28.1).
- **Docs:** correct the stratification rationale in [`01_DATA_PIPELINE.md`] §2 (F-28.2) and remove the "no expert stays unrouted for long" claim from [`03_MOE_MODULE.md`] (F-04).
- **Determinism:** set `torch.manual_seed`, `DataLoader(generator=...)`, `worker_init_fn`, and log `torch.backends.cudnn.deterministic`. `split_dataset` is deterministic; training is not.
- **Calibration:** report ECE with temperature scaling fitted on validation. `sub_scores` are currently `softmax(30·cos θ)` — extremely overconfident by construction (F-10), and a hierarchical classifier's practical value depends on trustworthy confidence.

---

#### **S-20 → F-32 — Baseline additions.**

| Variant | Definition | Question answered |
| --- | --- | --- |
| **`linear_probe`** | frozen DINO encoder → `Linear(384,4)` + `Linear(384,27)`, plain CE | Does any of the head machinery beat a linear layer on the same features? **Run this first.** |
| **`swinv2_supervised`** | ImageNet SwinV2-Base + the full hierarchical head, no DINO stage | Separates in-domain SSL from architecture. |
| **lr sweep** | `{1e-5, 3e-5, 1e-4}` per end-to-end baseline, report best | Removes the weak-baseline objection. |

`linear_probe` needs no new code — it is `FlatSupervisedBaseline` with `IdentityEncoder` swapped for the real encoder and the `EmbedDim` projection kept, which the existing dispatch in `build_model_and_encoder` almost supports already.

---

## 4. Actionable Implementation Plan

### 4.0 Critical path

Two ordering constraints dominate everything else:

1. **Phase 0 gates the value of every result in the paper.** If crop-level leakage exists (F-22), every number currently in hand is measuring the wrong thing and re-running anything before fixing the split is wasted compute.
2. **Batch the retrains.** Phases 1 and 2 both invalidate existing stage-2 results and both leave stage 1 untouched. Land them together and pay for one suite re-run, not two. Phase 3 is the only work that requires re-running the 300-epoch pretraining.

```
Phase 0 (1–2 days, no training)
        │
        ├─► Phase 1  loss + routing correctness  ─┐
        │                                          ├─► ONE stage-2 suite re-run
        └─► Phase 2  protocol + baselines        ─┘        (10 variants × 5 seeds)
                                     │
                                     └─► Phase 3  architecture (needs stage-1 re-pretrain)
                                                       │
                                                       └─► Phase 4  docs + paper
```

---

### 4.1 Phase 0 — Determine severity. No code changes. **1–2 days.**

| # | Check | How | Decides |
| --- | --- | --- | --- |
| 0.1 | **Crops per source image** | Parse provenance from filenames under `$SEED_DATA_ROOT`; if absent, perceptual-hash the crops and count near-duplicate clusters | Whether F-22 is fatal or void. **Highest priority item in this document.** |
| 0.2 | Are all images square? | `python -c "..."` over the tree, print the set of aspect ratios | F-26 |
| 0.3 | `zero_grad(set_to_none=?)` | grep `moe_finetune.py` | F-05 (whether unrouted experts skip weight decay) |
| 0.4 | Does `dynamic_img_size=True` actually work for `swinv2_base_window16_256`? | Forward a 101 px and a 256 px batch through `BackboneFeatureExtractor` | F-21.3 (whether the local-crop upsampling shortcut is removable) |
| 0.5 | Confirm degenerate attention empirically | Assert `attn_weights.allclose(1.0)` and `W_Q.grad is None` after one backward | F-03 — turns an analytical claim into a test |
| 0.6 | Measure current expert utilization | `expert_utilization_counts` on any existing `test_predictions.npz` | F-01 — is collapse already happening? |

**Deliverable:** a one-page note answering 0.1–0.6, which determines how much of Phases 1–3 is actually needed.

---

### 4.2 Phase 0.5 — Free wins. No retraining, no risk. **1 day.**

These run against artifacts you already have on disk.

| Item | File | Action |
| --- | --- | --- |
| Expert-specialization NMI | `src/utils/metrics.py` | Add `expert_label_nmi(expert_indices, labels)`; compute from existing `test_predictions.npz` (S-05) |
| `loss_flags` in summaries | `src/utils/evaluation.py` | Add block to `RunSummary` (F-28.1) |
| Benchmark statistics | `src/utils/efficiency.py` | `iterations: 10 → 50`, add median/IQR, tile from distinct samples (F-27c/d) |
| Determinism | `src/trainers/moe_finetune.py` | Seed generators + `worker_init_fn`; log cuDNN flags (S-19) |
| Doc corrections | `01_DATA_PIPELINE.md` §2, `03_MOE_MODULE.md` §1 | Strike the false stratification rationale (F-28.2) and the "no expert stays unrouted for long" claim (F-04) |
| Attention-degeneracy note | `04_HIERARCHICAL_FUSION.md` §4, `03_MOE_MODULE.md` §1 | Promote the existing honest footnote into an explicit statement of the dead-parameter consequence (F-03) |

---

### 4.3 Phase 1 — Loss and routing correctness. **1–2 weeks + suite re-run.**

| # | Change | Solution | Files | New tests |
| --- | --- | --- | --- | --- |
| 1.1 | Log-space KL aggregation via `logsumexp` | S-10 | `src/losses/hierarchical.py` | Recompute KL by hand for a confident-disagreement case; assert non-zero gradient where the old clamp produced zero |
| 1.2 | `detach_kl_seed_target: true` default; add `kl_mode: {forward, jsd}` | S-11 | `src/losses/hierarchical.py`, `conf/model/loss/arcface_kl.yaml` | Assert no gradient reaches `seed_type_classifier` through the KL path when detached |
| 1.3 | Add `tau_kl`, decouple from `arcface_scale` | S-10 | same | Assert `P_sub` entropy responds to `tau_kl` |
| 1.4 | Switch-form load-balancing loss (`load_mode: {entropy, switch}`) | S-01 | `src/losses/moe.py` | **The F-01 counterexample as a test:** assert the new loss is far from optimal for `G=(0.3,0.3,0.1,0.1,0.1,0.1)` while the old one scores −0.917 |
| 1.5 | Router z-loss, `λ_z = 1e-3` | S-01 | `src/losses/moe.py`, `conf/model/loss/arcface_kl.yaml` | Assert the term grows with router logit magnitude |
| 1.6 | Noisy top-k with annealed ε; near-zero router init; dead-expert counter | S-01 | `src/models/components/moe_layer.py` | Assert noise → 0 at the end of the schedule; assert the counter fires on a forced-collapse fixture |
| 1.7 | `λ_sparsity → 0.0` (keep as ablation axis) | S-02 | `conf/model/loss/arcface_kl.yaml` | — |
| 1.8 | Optimizer parity for unrouted experts | S-04 | `src/models/components/moe_layer.py` / trainer | **Assert parameter state matches after one step under sparse vs. dense dispatch** — the current test only checks the forward pass |
| 1.9 | EMA-smoothed utilization for the aux loss | S-04 | `src/losses/moe.py` | Assert the EMA buffer moves with `.to(device)` and is excluded from the optimizer |
| 1.10 | `arcface_scale: 4.6` (or dynamic AdaCos) + margin warm-up | S-09 | `src/models/components/arcface_head.py`, config | Assert `m=0` at epoch 0 and `m=0.5` after warm-up; assert `logits == margin_logits` at inference (existing invariant — keep) |
| 1.11 | LayerScale-gated residual; retire `cosine_mode="residual"` as default; EMA-centroid intra-class mode | S-06, S-13 | `src/models/builder.py`, `src/losses/cosine.py` | Assert `γ` is trainable and initialized at `1e-4`; assert every sample contributes to the intra-class term |
| 1.12 | Uncertainty weighting for `{seed, arcface, kl}` | S-12 | `src/losses/hierarchical.py` | Assert `log σ²` appear in `model.parameters()` and are logged |

**Invariants to preserve — do not regress these.** They are the tree's real assets:
`encoder(images).shape[-1] == 384` · MoE routes on `z` · `sub_logits` (no margin) drives prediction and KL · disabled toggles set attributes to `None`, never `nn.Identity` · one shared encoder checkpoint per suite · one trainer for every variant · `HierarchicalOutput` emitted by every model.

---

### 4.4 Phase 2 — Evaluation protocol and baselines. **1 week + the same suite re-run.**

| # | Change | Solution | Files |
| --- | --- | --- | --- |
| 2.1 | `StratifiedGroupKFold` / `GroupShuffleSplit`; persist `groups` in `split_manifest.npz` | S-17 | `src/trainers/moe_finetune.py`, `src/datasets/dataset.py` |
| 2.2 | **Leakage-ablation run:** full model under grouped vs. ungrouped splits, delta reported | S-17 | `scripts/run_ablations.py` |
| 2.3 | `SEEDS = [42..46]`; `VariantSpec.seed`; `outputs/{group}/{variant}/seed{n}/` | S-18 | `src/trainers/runner.py`, both suite scripts |
| 2.4 | `mean ± std` aggregation in the CSV; `p (vs full)` column via McNemar exact | S-18 | `src/utils/evaluation.py`, `scripts/generate_plots.py` |
| 2.5 | Fold aggregation reports mean ± std, not best fold | S-18 / F-24 | `src/trainers/moe_finetune.py` |
| 2.6 | Split the confounded ablations: add `wo_moe_capacity_matched`, `moe_fixed_router`, `wo_margin_only` (NormFace), `wo_residual_keep_cosine` | S-05, S-09 | `scripts/run_ablations.py`, `src/models/components/` |
| 2.7 | Add `linear_probe` and `swinv2_supervised` baselines; lr sweep for `resnet50`/`swin_tiny` | S-20 | `scripts/run_baselines.py`, `conf/experiment/baseline_*.yaml` |
| 2.8 | Stage-2 augmentation defaults + `sampler: balanced` option + per-seed-type macro-F1 | S-19 | `src/datasets/transforms.py`, `src/utils/metrics.py` |
| 2.9 | ECE with validation-fitted temperature scaling | S-19 | `src/utils/metrics.py` |
| 2.10 | Explicit `Resize((H, W))` | S-19 | `src/datasets/transforms.py` |

**Compute estimate.** 10 variants (6 ablations + 4 baselines) + 4 new controls + `full_model` = 15 configurations × 5 seeds = **75 stage-2 runs**. Stage 2 trains a head against a frozen encoder, so these are cheap relative to pretraining. `run_suite`'s subprocess isolation and `--stop-on-failure=false` already make an unattended 75-run sweep safe.

---

### 4.5 Phase 3 — Architecture. **3–5 weeks; requires stage-1 re-pretraining.**

Sequence these strictly, one at a time, each against the Phase 1+2 protocol so effects are attributable.

| # | Change | Solution | Depends on |
| --- | --- | --- | --- |
| 3.1 | **Stage-1 recalibration:** `out_dim 8192`, teacher temp `0.04→0.07`, cosine momentum `0.996→1.0`, cosine WD `0.04→0.4`, `center_momentum 0.99`, effective batch ≥64, drop BN in `DINOHead` | S-16 | — |
| 3.2 | Local crops at native resolution (or documented confound) | S-16 | Phase 0.4 |
| 3.3 | KoLeo + Sinkhorn–Knopp centering — **or** rename to "DINO-style" | S-15 | 3.1 |
| 3.4 | **Token-grid head:** stop pooling; MoE + cross-attention over `[B, 64, 384]`; attention/GeM pooling after the head | S-03 | 3.1 |
| 3.5 | FiLM conditioning replacing the additive `P(p_s)` residual (`fusion_mode: {additive, film}`) | S-07 | 3.4 |
| 3.6 | Seed-conditioned router (`gate([z, p_s.detach()])`); report NMI before/after | S-08 | 3.4 |
| 3.7 | iBOT patch-level objective (now reachable) | S-15 | 3.4 |
| 3.8 | Optional: sub-center ArcFace `K=3` | S-09 | 3.4 |

**Expected effect ranking (my inference, to be tested — not established):** 3.4 > 3.1 > 3.6 > 3.5 > 3.3 > 3.8. 3.4 is ranked first because it is the only change that converts three inert modules into functioning ones *and* addresses the mean-pooling/fine-grained mismatch at the same time.

---

### 4.6 Phase 4 — Documentation and paper. **1 week, concurrent with Phase 3.**

| Target | Change |
| --- | --- |
| `03_MOE_MODULE.md` | Replace §3's load-balancing derivation with the `f·P` form; add router z-loss and noisy gating; document the F-01 counterexample as the *reason* for the change; remove the "no expert stays unrouted for long" claim |
| `04_HIERARCHICAL_FUSION.md` | State the length-1 attention degeneracy and its parameter consequence explicitly in §4; document FiLM fusion in §3; add the NormFace control to §7 |
| `05_LOSS_FUNCTIONS.md` | Rewrite §4 for log-space aggregation; document `tau_kl`, `kl_mode`, and the detach default flip; rewrite §6 for LayerScale + EMA-centroid compactness |
| `06_ABLATION_ENGINE.md` | Add a **"what each toggle actually changes"** table (F-31) — this is the single most valuable documentation addition in the list; add the new controls and the multi-seed protocol |
| `01_DATA_PIPELINE.md` | Group-aware splitting; correct the stratification rationale; document the leakage ablation |
| `07_EFFICIENCY_AND_EVALUATION.md` | Benchmark statistics; state the parameter-vs-latency distinction; McNemar and mean±std reporting; NMI and ECE metrics |
| `02_BACKBONE_AND_SSL.md` | DINO vs DINOv2 naming resolution; recalibrated hyperparameters with rationale; local-crop resolution note |
| `PAPER_AUDIT.md` | Add F-01, F-03, F-07, F-12 as new audit entries — they are the same *class* of finding this file already tracks so well |
| **Paper** | Eq. 11–12 framing; "DINO-style" or full DINOv2; efficiency claim stated as parameters/FLOPs not latency; ablation table with mean±std and p-values; leakage-ablation subsection; remove any attention-map figure derived from `attn_weights` unless 3.4 lands |

---

## 5. Uncertainties, Limitations, and What Would Change My Mind

**What is established vs. what I infer.** Findings F-01, F-02, F-03, F-07, F-08, F-12, F-28.2, and F-31 are **derivations** from the documented code — they follow from the equations and configuration values as written, and each has a stated counterexample or computation you can verify without running anything. F-10's scale figure, F-19's centering ratio, F-20's prototype density, F-23's confidence intervals, and F-27's parameter counts are **arithmetic** from documented constants; the assumptions are stated inline and any of them can be recomputed. Everything about *expected accuracy impact* is **inference** and is labelled as such — I have no results to calibrate against.

**Assumptions that could flip conclusions:**

| Assumption | If wrong |
| --- | --- |
| Crops derive from a smaller set of source images | F-22 is void, and the leakage argument disappears entirely. **This is the single most consequential unknown.** |
| `zero_grad(set_to_none=True)` | F-05's weight-decay asymmetry inverts (dense would skip decay instead) |
| Images are square | F-26 is void |
| `nn.MultiheadAttention` is used with default `batch_first=True` and no attention mask | F-03's degeneracy argument is exact as stated; a mask or custom implementation could change it |
| Sub-varieties are roughly balanced | F-16's expected-nonzero-class computation shifts; the qualitative conclusion (most of the batch contributes zero) holds under any realistic imbalance |
| The docs match the code at the cited file:line anchors | The whole audit is downstream of this. The suite's precision suggests they do, but I read documentation, not source. |

**What I did not examine:** actual accuracy numbers (none in the suite), `src/utils/training/`, the tracker integration, `scripts/extract_features.py` / `dry_run.py`, the 280 tests themselves (only their described assertions), `REVISION_NOTES.md`, `PAPER_AUDIT.md`, and the paper PDF.

**Conflicting-evidence note.** On multi-task loss weighting, GradNorm's authors report that uncertainty weighting made the *inverse* decision to theirs on classification tasks and "often performed quite poorly" [16], while FairMOT found fixed grid-searched weights beat both on their primary metric [17]-adjacent evidence. I recommended uncertainty weighting anyway (S-12) on integration-cost grounds, not because the evidence is one-sided — this is a genuine trade-off and worth an ablation rather than a default.

**The strongest thing in this repository, stated plainly:** the discipline of `PAPER_AUDIT.md` + `REVISION_NOTES.md` + 280 invariant-pinning tests + a shared encoder checkpoint + byte-identical splits is better than most published ML code. Several findings above (F-03's attention degeneracy, F-30's cosine interaction) are things the documentation *already noticed and wrote down honestly* — the gap is that the consequences were not propagated to the parameter counts, the ablation semantics, and the paper's claims. That is a much easier gap to close than the alternative.

---

## 6. Sources

Every load-bearing empirical or formulaic claim above is keyed to one of these. Bracketed numbers are literal labels, not list positions.

**Mixture-of-Experts routing and stability**

- **[1]** Shazeer et al. (2017), *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*, arXiv:1701.06538 — noisy top-k gating, importance/load losses, the self-reinforcing collapse mechanism. Collapse dynamics summarized at https://www.ibm.com/think/topics/mixture-of-experts
- **[2]** Fedus et al. (2021), *Switch Transformers*, arXiv:2101.03961 — the `alpha·N·sum(f_i·P_i)` auxiliary loss now standard across MoE implementations.
- **[5]** Zoph et al. (2022), *ST-MoE: Designing Stable and Transferable Sparse Expert Models* — https://arxiv.org/pdf/2202.08906 — Appendix A gives the `f`/`P` load-balance definitions (Eqs. 7-9); introduces the router z-loss.
- **[6]** Muennighoff et al. (2024), *OLMoE: Open Mixture-of-Experts Language Models* — https://arxiv.org/pdf/2409.02060 — router z-loss formula and the standard weight beta = 0.001.
- **[7]** *Routing Is the Hard Part: A Practitioner's Guide to Mixture-of-Experts* (2026) — https://frontiercheckpoint.com/explainers/moe-routing-practitioners-guide/ — why dropping gating noise requires a hard-dispatch-aware controller in exchange.
- **[31]** *GatePro: Parameter-Free Expert Selection Optimization for MoE* — https://arxiv.org/pdf/2510.13079 — survey of routing alternatives including GShard's random secondary routing and BASE-layer assignment.

**Angular-margin losses**

- **[8]** Zhang et al. (2019), *AdaCos: Adaptively Scaling Cosine Logits*, CVPR — https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_AdaCos_Adaptively_Scaling_Cosine_Logits_for_Effectively_Learning_Deep_Face_CVPR_2019_paper.pdf — analysis of how scale and margin modulate the predicted probability.
- **[9]** *A Survey of Face Recognition* — https://arxiv.org/pdf/2212.13038 — Eq. 40 states the AdaCos fixed scale `s = sqrt(2)·log(N-1)` explicitly.
- **[10]** Huang et al. (2020), *CurricularFace* — https://arxiv.org/pdf/2004.00288 — ArcFace divergence at m = 0.5 on small backbones; curriculum over sample difficulty.
- **[11]** Kim et al. (2022), *AdaFace: Quality Adaptive Margin* — https://ar5iv.labs.arxiv.org/html/2204.00964 — survey of adaptive and scheduled margin strategies.
- **[18]** Deng et al. (2020), *Sub-center ArcFace* — overview: https://www.emergentmind.com/topics/sub-center-arcface
- **[29]** Deng et al., *ArcFace: Additive Angular Margin Loss* (incl. sub-center) — https://arxiv.org/pdf/1801.07698
- **[30]** Zhang et al. (2019), *AdaCos* — https://arxiv.org/abs/1905.00292

**Hierarchical classification consistency**

- **[12]** Garg et al. (2022), *Learning Hierarchy Aware Features for Reducing Mistake Severity* (HAF) — https://arxiv.org/pdf/2207.12646 — JSD between coarse predictions and marginalized fine predictions; notes that CE at all levels compromises fine-grained accuracy.
- **[13]** *Hierarchical Classification for Improved Histopathology Image Analysis* — https://arxiv.org/pdf/2603.00504 — JSD-based hierarchical consistency loss.
- **[14]** *Hierarchy-Aware Fine-Tuning of Vision-Language Models* — https://arxiv.org/pdf/2512.21529 — tree-path KL alone causes severe accuracy drops by over-penalizing fine-grained confusion.

**Multi-task loss weighting**

- **[15]** *Uncertainty Multi-Task Loss (Kendall weighting)* — https://distilledpatterns.org/patterns/uncertainty-multi-task-loss/ — formulation, and why fixed coefficients let one task dominate.
- **[16]** Chen et al. (2018), *GradNorm* — https://arxiv.org/pdf/1711.02257 — includes the comparison where uncertainty weighting underperforms on classification tasks.
- **[17]** *Multi-Loss Weighting with Coefficient of Variations* — https://arxiv.org/pdf/2009.01717 — side-by-side of GradNorm / MGDA / uncertainty weighting.

**Self-supervised pretraining**

- **[19]** Oquab et al. (2023), *DINOv2: Learning Robust Visual Features without Supervision* — https://arxiv.org/html/2304.07193v2 — iBOT patch objective, Sinkhorn-Knopp centering, KoLeo regularizer, untied heads, and their ablation magnitudes.
- **[20]** *Understanding DINOv2* (Lightly) — https://www.lightly.ai/blog/dinov2
- **[21]** *Self-Supervised Training — facebookresearch/dinov2* — https://deepwiki.com/facebookresearch/dinov2/6.1-self-supervised-training
- **[22]** *DINO: Emerging Properties in Self-Supervised Vision Transformers* — https://www.abhik.ai/papers/dino — cosine momentum schedule 0.996 -> 1.0 and why it matters.
- **[23]** *DINO hyperparameter summary* — https://medium.com/@kdk199604/dino-unlocking-emergent-visual-intelligence-in-self-supervised-vision-transformers-fbb2be1d7344 — weight decay 0.04 -> 0.4; teacher temperature 0.04 -> 0.07 over the first 30 epochs.
- **[24]** *Weighted Ensemble Self-Supervised Learning* — https://arxiv.org/pdf/2211.09981 — independent confirmation of DINO's momentum and weight-decay schedules.
- **[25]** *Emerging Properties in Self-Supervised ViT — paper summary* — https://medium.com/@anuj.dutt9/emerging-properties-in-self-supervised-vision-transformers-dino-paper-summary-4c7a6ed68161

**Data leakage under non-grouped splitting**

- **[3]** *Inflation of test accuracy due to data leakage in deep learning-based classification of OCT images* — https://arxiv.org/pdf/2202.12267 — per-image vs. per-volume/subject splitting inflates mean MCC by 0.07-0.43 across four datasets.
- **[4]** *The frame-level leakage trap* — https://arxiv.org/pdf/2605.06359 — frame- vs. scene-level splits inflate test PSNR by 1.6-2.0 dB, p < 0.01, architecture-independent.
- **[26]** *Investigation of Neural Network Methods for Reconstruction and Classification of Texture Images* — https://arxiv.org/pdf/2204.14224 — overlapping sliding-window patches produced 93.7 % accuracy that the authors excluded as a leakage artifact.
- **[27]** *A Leakage-Aware Comparative Benchmark of ML, DL, and Transformer Models for Reliable Leukemia Detection* — https://arxiv.org/pdf/2606.24944 — controlled ablation isolating image-level vs. group-wise splitting.
- **[28]** Chiu et al. (2020), *Agriculture-Vision* — https://arxiv.org/pdf/2001.01306 — assigns every crop to the split of the source farmland image; the reference pattern for S-17.

**Named methods referenced but not independently re-verified in this pass** *(standard and well-established; cited by author/year in the text rather than by link)*: FiLM (Perez et al., 2018), LayerScale (Touvron et al., 2021), center loss (Wen et al., 2016), NormFace (Wang et al., 2017), GeM pooling (Radenovic et al., 2018), Hash Layers (Roller et al., 2021), PCGrad (Yu et al., 2020), iBOT (Zhou et al., 2022), SwAV (Caron et al., 2020), McNemar's exact test.

---


## Appendix A — Findings Index

| ID | Sev | Finding | Doc | Solution |
| --- | --- | --- | --- | --- |
| F-01 | **C** | `L_load` regularizes the soft gate, not the top-k dispatch; 92 %-optimal while 4/6 experts are dead | 03, 06 | S-01 |
| F-02 | H | `L_sparsity` has a null space w.r.t. the output under `renormalize_top_k` | 03 | S-02 |
| F-03 | **C** | Length-1 attention ⇒ affine map; ≈2.07 M dead parameters; `attn_weights ≡ 1` | 03, 04 | S-03 |
| F-04 | H | No exploration mechanism; deterministic top-k on a frozen encoder | 03 | S-01 |
| F-05 | H | Sparse vs. dense dispatch differ under AdamW; 32 routing slots per step | 03 | S-04 |
| F-06 | H | `DenseExpertBlock` is not capacity-matched to Top-2 | 03, 06 | S-05 |
| F-07 | **C** | `L_cos(residual)` minimizers delete the residual; relative weight grows over training | 05 | S-06 |
| F-08 | H | Residual → 4-entry codebook; gradient to stage 1 vanishes as `p_s` saturates | 04 | S-07 |
| F-09 | M | MoE and seed head both read `z`; the "cascade" is a rank-≤4 coupling | 00, 04 | S-08 |
| F-10 | H | `s = 30` is 6.5× the AdaCos scale for C = 27; ArcFace ≈13:1 at init | 04, 05 | S-09 |
| F-11 | H | `LinearSubVarietyHead` changes 4 factors; `wo_arcface` is not a margin ablation | 04 | S-09 |
| F-12 | **C** | `clamp_min(1e-8) → log` zeroes the KL gradient in the confident-disagreement case | 05 | S-10 |
| F-13 | H | `s = 30` saturates `P_sub`; KL and ArcFace secretly coupled | 05 | S-10 |
| F-14 | H | Non-detached, mode-covering KL lets the coarse head chase the fine head | 05 | S-11 |
| F-15 | H | Seven fixed λ over a ~13× initial scale spread; no adaptive weighting | 05 | S-12 |
| F-16 | M | `intra_class` cosine: ~3.2 eligible classes per batch of 16 | 05 | S-13 |
| F-17 | M | No per-term gradient telemetry in stage 2 | 07 | S-14 |
| F-18 | H | "DINOv2" implements DINOv1 (no iBOT / KoLeo / Sinkhorn / untied heads) | 02 | S-15 |
| F-19 | H | Sharpening 2× stronger and centering 64× weaker than reference, simultaneously | 02 | S-16 |
| F-20 | H | 65,536 prototypes for 9,357 images; 16.8 M params in one layer | 02 | S-16 |
| F-21 | M | Missing schedules; local-crop upsampling shortcut; BN teacher/student coupling; fragile view skip | 02 | S-16 |
| F-22 | **C** | No group key in `split_dataset`; probable crop-level leakage | 01 | S-17 |
| F-23 | **C** | Single seed / single fold; ±1.40 pp noise floor on ablation gaps | 01 | S-18 |
| F-24 | H | Best-fold checkpoint selection is optimistically biased at `num_folds > 1` | 07 | S-18 |
| F-25 | M | Stage-2 augmentation effectively nil; no class balancing | 01 | S-19 |
| F-26 | L | `Resize(int)` preserves aspect ratio — latent collate failure or a missing comment | 01 | S-19 |
| F-27 | H | Efficiency delta ≈2 %, invisible in latency; batch-32 benchmark routes degenerately | 07 | S-19 |
| F-28 | L | `component_flags` omits `use_kl_loss`; stratification rationale is provably false | 01, 07 | S-19 |
| F-29 | H | `wo_moe` flips 3 factors | 06 | S-05 |
| F-30 | H | `wo_residual` flips 2 factors (Eq. 9 + `L_cos`) | 06 | S-06 |
| F-31 | H | Only 2 of 6 ablations are single-factor | 06 | S-05, S-09 |
| F-32 | M | No linear probe; baselines likely under-tuned; `hierarchical_cce` is not independent | 06 | S-20 |

*End of audit.*
