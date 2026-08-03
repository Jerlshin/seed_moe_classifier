# AUDIT_RESPONSE.md — disposition of every finding in `CHANGES.md`

**Audit:** `CHANGES.md`, 2026-07-29, F-01 … F-32.
**This document:** what was measured, what changed, what did not, and why.

Read this alongside `CHANGES.md`. Where the audit reasoned from documentation, we
measured; two of its numbers moved and one of its findings turned out to be void
in the code (though not in the docs). Those corrections are marked **[correction]**
and are stated before the fix that follows from them, because in three places the
measurement changes what the right fix is.

---

## 1. Phase 0 — the checks that gate everything else

`CHANGES.md` §4.1 asks six questions before any code changes. All six are now
answered from the actual tree and the actual dataset, and `tests/conftest.py`
carries the numbers as constants so a test can assert against them.

| # | Check | Answer | Consequence |
| --- | --- | --- | --- |
| 0.1 | Crops per source image | **9,357 crops from 81 source photographs; mean 115.5, range 37–297** | F-22 confirmed, and worse than its worst case. See §2. |
| 0.2 | Are all images square? | **No — 3.4 % are.** Aspect ratios 0.17–3.48; median size **52 × 51 px**; **100 %** have both sides under 256 | F-26 is **void in the code** and real in the docs. See §3. |
| 0.3 | `zero_grad(set_to_none=?)` | `set_to_none=True` (`moe_finetune.py`) | F-05's weight-decay asymmetry holds exactly as stated. Fixed. |
| 0.4 | Does `dynamic_img_size=True` work for SwinV2? | **No.** `swinv2_*_window16_256` raises `AssertionError: Input height (101) doesn't match model (256)` | F-21.3's local-crop shortcut is **not removable**. Documented and mitigable instead. |
| 0.5 | Is length-1 attention degenerate? | **Yes, measured.** Q/K gradient is zero to float precision; `attn_weights` is exactly 1.0 | F-03 confirmed. Now a test, not an argument. |
| 0.6 | Current expert utilisation | No prior `test_predictions.npz` on disk | Not answerable retrospectively; dead-expert count is now a first-class metric. |

### 0.1 in full — this is the finding that constrains the paper

```
total crops                    9,357
distinct source photographs       81
mean crops per source          115.5
sources per sub-variety          1 : 5 classes
                                 2 : 4 classes
                                 3 : 8 classes
                                 4 : 6 classes
                                 5 : 4 classes
```

`CHANGES.md` F-22 wrote: *"If it is 1, this finding is void. If it is 10–50, the
headline accuracy is not measuring what the paper says it measures."* It is
**115.5**.

**[correction]** The audit could not know, and we initially miscounted, how many
sub-varieties come from a *single* photograph. It is **5** — `Baryard`,
`Browntop`, `FingerMillet`, `PearlMillet`, `ProsaMillet` — not 8. The
distinction matters because those five bound what any split protocol can achieve.

---

## 2. F-22 — the finding that changes what the paper can claim

Confirmed, and it does not fully go away.

**What was done.** `split_protocol: grouped` is now the default.
`HierarchicalSeedDataset.source_groups()` derives a provenance key from the
filename (`IMG_0502_bbox137.png` → `Millet/Baryard/IMG_0502`), and
`split_dataset` uses `GroupShuffleSplit` / `StratifiedGroupKFold` so no source
photograph appears on both sides of a boundary. Groups are persisted into
`split_manifest.npz` so a reviewer can verify the grouping rather than trust the
protocol name.

**What the protocol cannot fix, and must therefore be reported.** Five
sub-varieties have exactly one source photograph. No grouped split can place any
of their crops in both train and test, so grouped stratification degrades to
"that class is entirely in one partition" for those five. On the real tree a
30 % grouped test split leaves three sub-varieties out of training altogether;
the trainer logs this as a warning and records it in `summary.json`.

The honest statement for the paper is: *for those classes, no protocol available
on this dataset can measure across-photograph generalisation.* That is a dataset
limitation, not a modelling one, and stating it is cheaper than having a reviewer
derive it.

**Turning it into a result rather than a correction.** `run_ablations.py` gains
`leakage_ungrouped`, which is the full model under the submitted crop-level
protocol and nothing else. The delta against `full_model` is a direct measurement
of what the leak was worth, which is the form the literature the audit cites
(OCT, intrinsic decomposition, leukemia cytology) reports it in.

---

## 3. F-26 — void in the code, real in the documentation **[correction]**

The audit inferred from `01_DATA_PIPELINE.md` that stage 2 called
`Resize(image_size)` with an integer, which resizes the shorter side and would
crash `default_collate` on non-square input. The code already passed a tuple,
`T.Resize((image_size, image_size))`, so there was never a latent crash. The
documentation was wrong, not the code.

The measurement does surface something neither the docs nor the audit mention:
**the crops are tiny.** Median 52 × 51 px, and every one of the 9,357 has both
sides under 256, so the pipeline upsamples by roughly 5× before the backbone sees
anything. "Fine-grained texture" in this dataset means texture the sensor
resolved at ~50 px. That belongs in the paper's data section and is now in
`src/datasets/transforms.py` and `architecture/01_DATA_PIPELINE.md`.

The tuple form also squashes aspect ratio, which is a real distortion chosen over
a real crash. It is now stated at the call site rather than left implicit.

---

## 4. Findings resolved in code

Severity column reproduces `CHANGES.md`. "Where" cites the module that now owns
the fix; every entry has at least one test.

### 4.1 Routing and sparse computation

| ID | Sev | Resolution | Where |
| --- | --- | --- | --- |
| **F-01** | C | `moe_load_mode: switch` is the default: `E·Σ fᵢPᵢ` with a zero floor at uniform routing. The entropy form is retained as `load_mode="entropy"` so the two can be compared on one split. The audit's counterexample `G=(.3,.3,.1,.1,.1,.1)` is a **test**: the entropy form scores −0.9172 while the Switch form scores 0.8 above its floor, and the gradient is shown to push down exactly the over-dispatched experts. | `src/losses/moe.py` |
| **F-02** | H | `lambda_moe_sparsity: 0.0`. Option A of S-02: `renormalize_top_k` already delivers the convex combination, so the L1 term has a null space w.r.t. the output and only cuts router entropy. Kept as an ablation axis, documented as redundant. | `conf/model/loss/arcface_kl.yaml` |
| **F-03** | C | Two-sided fix. `token_mode="grid"` (default) keeps SwinV2's 8×8 grid through the MoE and cross-attention, so attention is genuine, `num_heads` matters and `attn_weights` is a plottable map. `token_mode="pooled"` **does not allocate** the Q/K projections at all — it substitutes the single `nn.Linear` that spans the identical function class. Either way no unreachable parameter is built or counted. | `moe_layer.py`, `cross_attention.py`, `builder.py` |
| **F-04** | H | Noisy Top-K gating (`router_noise_std: 0.3`) annealed to **exactly** zero over the first 30 % of training; near-zero router init (`gate_init_std=1e-3`); a per-step **dead-expert counter** logged as a first-class metric. The docs' "no expert stays unrouted for long" claim is struck. | `moe_layer.py`, `moe_finetune.py` |
| **F-05** | H | `MixtureOfExperts.materialize_zero_grads()` is called between `backward()` and `step()`. Tested at the level that matters: **parameter state after one AdamW step** is identical under sparse and dense dispatch, and the counterfactual test shows they diverge without it. Estimator noise is addressed at the source by grid routing (32 → 2,048 routing slots per step) and by an EMA over the routing statistics. | `moe_layer.py`, `moe_finetune.py` |
| **F-06** | H | `dense_capacity_multiplier` widens the `use_moe=False` block to match Top-K's active capacity. Both controls run: `wo_moe` (naive, historical) and `wo_moe_capacity_matched`. | `moe_layer.py`, `run_ablations.py` |

### 4.2 Hierarchical fusion

| ID | Sev | Resolution | Where |
| --- | --- | --- | --- |
| **F-07** | C | `cosine_mode` defaults to `intra_class` with **EMA class centroids**; the residual is controlled *structurally* by `LayerScale(init=1e-4)` plus an optional magnitude hinge that is exactly zero below `τ = 0.5`, so nothing rewards `P(p_s) → 0`. `mode="residual"` is retained for ablation, and a test asserts it still collapses — the confound is reproducible on purpose. | `cosine.py`, `projections.py` |
| **F-08** | H | `fusion_mode: film` conditions on the seed classifier's **pre-softmax hidden state** (192-D), not on the saturating 4-simplex `p_s`. Initialised at `γ = 1, β = 0`, so it is a strict superset of the additive form and the two are comparable from step 0. A test measures the additive path's gradient collapsing by >1000× as the coarse head saturates. | `projections.py`, `classifiers.py` |
| **F-09** | M | `gate_conditioning: true` feeds the **detached** `p_s` to the router. The experts still consume `z` — tested — so `PAPER_AUDIT.md` §2.1's invariant is intact; only the gate sees the coarse prediction. `expert_label_nmi` makes the resulting specialisation a measurement. | `moe_layer.py`, `builder.py`, `metrics.py` |
| **F-10** | H | `arcface_scale: "auto"` → the AdaCos value `√2·log(C−1) = 4.61` for `C = 27`. Margin warm-up `0 → m` over the first 15 % of training. `arcface_dynamic` (AdaCos proper) and `arcface_sub_centers` available. | `arcface_head.py`, `moe_finetune.py` |
| **F-11** | M | `NormFaceHead` added: normalised embedding and centres, `m = 0`. `wo_margin_only` is now the single-factor margin control; the four-factor linear-head variant is honestly renamed `wo_angular_head`. | `arcface_head.py`, `run_ablations.py` |

### 4.3 Loss stack

| ID | Sev | Resolution | Where |
| --- | --- | --- | --- |
| **F-12** | C | `aggregate_sub_log_probs` marginalises with `logsumexp` over each parent's children. Exact, epsilon-free, well-conditioned everywhere. Two tests: the KL term now has non-zero gradient in the confident-disagreement case, and the old `clamp → log` composition is shown to be gradient-dead on the same input. | `hierarchical.py` |
| **F-13** | H | `tau_kl` decouples the hierarchy term from the ArcFace scale, which were one hyperparameter in disguise. At the AdaCos scale `tau_kl = 1.0` is the right value. | `hierarchical.py`, config |
| **F-14** | H | `detach_kl_seed_target` defaults to **true**. `kl_mode: jsd` offers the symmetric alternative (bounded by `log 2`, no zero-avoidance); `kl_jsd` runs it as an ablation. | `hierarchical.py` |
| **F-15** | H | Two changes. The ~13:1 initial imbalance is mostly removed by the analytic ArcFace scale (measured at ~4:1 in the dry run). `weighting_mode: uncertainty` learns the three task weights (Kendall et al.) and exposes them as a diagnostic; the three regularisers keep fixed λ deliberately. | `hierarchical.py` |
| **F-16** | M | EMA class centroids: every sample contributes, and each centroid is estimated from the whole training history rather than from two in-batch samples. A test shows an all-singleton batch scores exactly 0 per-batch and non-zero under EMA. | `cosine.py` |
| **F-17** | M | `per_term_gradient_norms` logs `grad_norm/{term}` and `grad_cosine/{pair}` at the shared trunk every 50 steps — the direct empirical test for F-07, F-10, F-14 and F-15. | `moe_finetune.py` |

### 4.4 Stage-1 self-supervision

| ID | Sev | Resolution | Where |
| --- | --- | --- | --- |
| **F-18** | H | **Renamed, and partially closed.** Stage 1 is described everywhere as "DINO-style self-distillation with a SwinV2 trunk", plus the two DINOv2 components that do not need patch tokens: `koleo_regularizer` and `sinkhorn_knopp`. iBOT and untied heads are **not** implemented and the docs say so. Option A + most of Option B of S-15. | `dino.py`, configs |
| **F-19** | H | Both guards corrected. Teacher temperature `0.04 → 0.07` over 30 epochs (was `0.02 → 0.04` over 5). `centering: sinkhorn` removes the dependence on a 65,536-dimensional running mean entirely; `center_momentum` raised to 0.99 for the `ema` path. `collapse_metrics` logs teacher entropy and prototype-marginal KL to uniform — the numbers a plausible-looking loss curve hides. | `dino.py`, configs |
| **F-20** | H | `out_dim: 8192` (0.88 prototypes per image, from 7.00). Removes ~14.7 M parameters from the head. | `conf/model/head/dino.yaml` |
| **F-21** | M | 1–2: cosine schedules for teacher momentum (`0.996 → 1.0`) and weight decay (`0.04 → 0.4`). 3: **not removable** — see Phase 0.4; documented as a confound with `match_view_lowpass` available to neutralise it. 4: head normalisation defaults to `layer`, so teacher/student BN statistics can no longer diverge. 5: explicit view identifiers replace positional matching. Gradient accumulation ×4 raises the effective batch to 64. | `dino.py`, `swinv2_dino.py`, `transforms.py`, `contrastive_pretrain.py` |

### 4.5 Data and evaluation protocol

| ID | Sev | Resolution | Where |
| --- | --- | --- | --- |
| **F-22** | C | See §2. Grouped splitting, persisted groups, per-run leakage report, and `leakage_ungrouped` as a measured ablation. | `dataset.py`, `moe_finetune.py` |
| **F-23** | C | `DEFAULT_SEEDS = (42…46)`; `VariantSpec.seed`; runs land in `{group}/{variant}/seed{n}/`. `aggregate_by_variant` reports **mean ± SD**; `paired_significance` runs **McNemar's exact test** against `full_model` with Holm–Bonferroni across the family. Implemented on `scipy.stats.binomtest`, so no new dependency. | `runner.py`, `evaluation.py`, `metrics.py` |
| **F-24** | H | Per-fold test evaluation, aggregated as `fold_metrics: {mean, std, min, max, folds}` in `summary.json`. Best-fold selection survives only for the artifact that gets profiled and shipped. | `moe_finetune.py` |
| **F-25** | M | Stage-2 defaults: `horizontal_flip_prob: 0.5` and `RandomResizedCrop(scale=(0.8,1.0))`, with `wo_stage2_augmentation` reproducing the submitted deterministic pipeline as an ablation. `balanced_sampler` available; `per_seed_type_breakdown` reports each branch separately so 13 rice classes stop setting the impression. | `transforms.py`, `moe_finetune.py`, `metrics.py` |
| **F-26** | L | Void in code — see §3. Docs corrected; the crop-size finding added. | `transforms.py`, docs |
| **F-27** | H | (a) The saving is now *stated* as ~2 % of total parameters, in `EfficiencyReport`'s own notes. (b) The notes say explicitly that it is a parameter/FLOP saving and **not** a wall-clock one, and why. (c) The latency sweep tiles from **real, distinct test images**, and annotates the report when it cannot. (d) `iterations: 50`, reported as **median + IQR** with min/max. | `efficiency.py`, config |
| **F-28** | L | 1: `loss_flags` and `split` blocks added to `summary.json`, and `component_flags()` now reports every axis a variant can move — `wo_kl` is machine-distinguishable from `full_model`. 2: the false stratification rationale is struck and replaced with the bijection argument; the code returns the simpler key that produces the identical partition. | `evaluation.py`, `builder.py`, `moe_finetune.py` |

### 4.6 Ablation and baseline validity

| ID | Sev | Resolution | Where |
| --- | --- | --- | --- |
| **F-29** | H | Split into `wo_moe` (kept, now labelled as three-factor), `wo_moe_capacity_matched`, `moe_fixed_router` and `moe_uniform_router`. | `run_ablations.py` |
| **F-30** | H | Dissolved rather than patched: with `cosine_mode="intra_class"` the compactness term no longer rides on the residual, so `wo_residual` removes Eq. 9 and only Eq. 9. `wo_layer_scale` isolates the gate. A test asserts compactness survives the toggle *and* that the submitted formulation still collapses. | `cosine.py`, `run_ablations.py` |
| **F-31** | H | 18 variants, each documented with the factors it actually moves; `architecture/06_ABLATION_ENGINE.md` carries the "what each toggle actually changes" table. | `run_ablations.py`, docs |
| **F-32** | M | `linear_probe` added (the control to run first) and `swinv2_supervised` (separates in-domain SSL from architecture). `--lr-sweep` runs `{1e-5, 3e-5, 1e-4}` per end-to-end baseline. The `spec_checkpoint = None` discipline the audit praised is kept and extended to `swinv2_supervised`. | `baselines.py`, `run_baselines.py`, configs |

---

## 5. What was deliberately not done

**iBOT patch-level objective (S-15 Option B, phase 3.7).** Not implemented. It
needs token-level student/teacher outputs plus a masking pipeline, which is a
substantially larger change than the rest of stage 1 combined, and the token-grid
work it depends on landed in stage 2 rather than stage 1. Stage 1 is named
accordingly: the tree does not claim DINOv2.

**PCGrad / GradNorm (S-12's rejected alternative).** Uncertainty weighting is
implemented; GradNorm is not. The audit's own conflicting-evidence note applies,
and gradient surgery needs per-task gradient access every step, which would
intrude on the single-code-path trainer that is this repository's best property.
The per-term gradient telemetry (F-17) is what would justify adding it later.

**Sub-centre ArcFace beyond the flag.** `arcface_sub_centers` works and is
tested, but it is not a suite variant. It is a data-cleaning tool for suspected
label noise, and there is no evidence of that here yet.

**Re-running training.** Nothing in this repository has been retrained. Every
number quoted in this document is either a static measurement of the dataset, a
closed-form parameter count, or a dry-run diagnostic. The suites must be re-run
before any of the paper's tables can be updated.

---

## 6. What the dry run already shows

Two runs of `scripts/dry_run.py` on synthetic data, differing only in
`token_mode`, with 10 samples and 2 epochs:

| | `pooled` (submitted) | `grid` (revision) |
| --- | --- | --- |
| Routing slots per step | 10 × 2 = 20 | 10 × 64 × 2 = 1,280 |
| Dead experts per step | 1 | 0 |
| Expert utilisation | 0.10 – 0.30 | 0.155 – 0.184 |

That is the F-01/F-05 estimator-noise argument reproducing itself at toy scale on
the first try, and it is why grid routing is the default rather than an option.

`L_ArcFace : L_seed` at initialisation is ~4:1 in the dry run, against the
audit's computed 12.7:1 at `s = 30` — most of F-15's imbalance is a consequence
of F-10 and goes away with it.

---

## 7. Verification

```bash
python -m pytest tests/ -q       # 341 tests
python scripts/dry_run.py        # synthetic end-to-end, grid routing
python scripts/dry_run.py --token-mode pooled --weighting-mode uncertainty
```

The tests that specifically pin this audit's findings:

- `test_entropy_load_balancing_scores_a_collapsed_router_as_nearly_perfect` — the F-01 counterexample, to four decimal places
- `test_switch_load_balancing_sees_the_collapse_the_entropy_form_misses`
- `test_length_one_attention_is_provably_affine` — F-03, measured
- `test_kl_gradient_survives_confident_disagreement` and
  `test_probability_space_aggregation_is_what_lost_the_gradient` — F-12 and its counterfactual
- `test_sparse_and_dense_dispatch_agree_after_an_optimizer_step` and
  `test_without_materialization_the_two_paths_diverge` — F-05
- `test_wo_residual_removes_only_equation_nine` — F-30
- `test_grouped_splitting_keeps_source_photographs_on_one_side` — F-22
- `test_additive_fusion_gradient_vanishes_as_the_coarse_head_saturates` — F-08

---

## 8. What still gates the paper

1. **Re-run the suites.** 18 ablation variants + 5 baselines × 5 seeds. Nothing
   in the current results tables survives the split-protocol change.
2. **Report the leakage delta** (`leakage_ungrouped` vs `full_model`) as a
   methods result, and state the five single-source sub-varieties as a dataset
   limitation.
3. **Run `linear_probe` first.** If the head does not clear it by a seed-stable
   margin, that outcome determines what the paper can claim.
4. **Rename stage 1 in the manuscript** to match the code.
5. **Restate the efficiency claim** as ~2 % of parameters and FLOPs, explicitly
   not wall-clock.
6. **Remove any attention-map figure** derived from the pooled path; regenerate
   from `token_mode="grid"`, where the map is real.
