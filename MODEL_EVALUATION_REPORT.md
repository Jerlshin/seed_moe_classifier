# Stage-2 Hierarchical MoE — Model Evaluation Report

Run: `outputs/finetune_hierarchical_moe/` · group `proposed` · variant
`finetune_hierarchical_moe` · seed 42
Trained on Kaggle T4×2 (`world_size=2`, fp16 autocast, deterministic cuDNN);
evaluated in fp32 (`AMP_DISABLED`), as the stage-2 contract requires.
Report generated 2026-09-18 against commit `a041830`.

Every number below is either read from the run's own artifacts or recomputed
from `test_predictions.npz`. Where this report re-derives a quantity, it says
so, and where it disagrees with `summary.json` it says that too.

**Evaluation protocol: crop-level stratified, and only that.** This study
classifies individual seed instances under a standard stratified
train/validation/test partition (§5.3). Photograph-partitioned protocols are out
of scope and no number here is produced under one.

---

## 0. Executive summary

**The evaluation standard for this study is the crop-level stratified
protocol**, and every number in this report is produced under it: 20 % of the
13,492 seed crops held out as a test split, stratified on sub-variety, with the
remainder split into a stratified train/validation pair. The unit of
classification is the **individual seed instance**, which is the unit the corpus
is built from, the unit the taxonomy labels, and the unit a downstream user
presents to the model. §5.3 states the protocol in full.

The model is **excellent at the coarse task and strong at the fine task**:
99.33 % seed-type accuracy against 81.73 % sub-variety accuracy (0.7943 macro
F1) on 2,699 held-out crops, with all 27 classes present on the test side. The
hierarchy is essentially self-consistent (99.78 % alignment), all six experts
are live, and routing is genuinely seed-type-specialised rather than merely
balanced.

Three things qualify that headline, and all three are quantified below:

1. **The reported metrics describe the epoch-100 weights, not the epoch-6
   checkpoint the trainer selected.** `best_state` held live module references
   that kept training. Its measured cost on this run is negligible — the two
   checkpoints are statistically indistinguishable (McNemar *p* = 0.87) — but
   the artifact `hierarchical_moe_final.pth` is mislabelled `epoch: 6` while
   carrying epoch-100 weights. **Now fixed and regression-tested** (§6.1); the
   published numbers stand, and a re-run will report what it selected.
2. **The 27-class number is dominated by one genuinely unsolved group.**
   Amaranthus (AMT-1/2/4) sits at 44.98 % accuracy — near the 33 % three-way
   chance floor. Excluding it, the other 24 classes average 0.8374 macro F1.
3. **Per-class accuracy tracks taxonomic rank, not sample size.** The 8
   species-level labels average 0.9031 F1 against 0.7484 for the 19
   cultivar/accession-level ones — **+15.5 pp** (§7.3). Every top confusion is
   intra-seed-type, so the whole error budget is within-crop discrimination.

Verdict in one line: **deployable as a seed-type classifier, and as a
sub-variety classifier everywhere except Amaranthus cultivar discrimination.**
See §7.

---

## 1. Artifact and metadata verification (Task 1)

### 1.1 Run contract files

All five required files are present.

| File | Size | Status |
| --- | --- | --- |
| `best_hierarchical_moe.pth` | 151,080,083 B | present — genuine epoch-6 weights |
| `hierarchical_moe_final.pth` | 151,080,083 B | present — **epoch-100 weights, labelled epoch 6** (§6.1, fixed for future runs) |
| `summary.json` | 16,376 B | present, parses, 87 metrics |
| `test_predictions.npz` | 3,551,518 B | present, 12 arrays, all finite |
| `split_manifest.npz` | 38,064 B | present, 8 arrays |

Three further checkpoints are present and are the expected interval/resume
artifacts: `model_fold1_epoch0100.pth`, `finetune_resume_fold1_epoch0084.pth`,
`finetune_resume_fold1_epoch0100.pth`.

### 1.2 Corpus fingerprint — VERIFIED

Recomputed `corpus_fingerprint()` over the local corpus at
`../Dataset/Hierarchical_SeedData/Refined_Samples`:

```
local  sha256 = 013c04c5858e9815ac0bb71b9f5fde4c8be9176c514d28198e39727344b6f3d3
run    sha256 = 013c04c5858e9815ac0bb71b9f5fde4c8be9176c514d28198e39727344b6f3d3
MATCH  = True
```

| Property | Expected | Recorded | Local | Match |
| --- | --- | --- | --- | --- |
| samples | 13,492 | 13,492 | 13,492 | ✅ |
| classes | 27 | 27 | 27 | ✅ |
| source photographs | 96 | 96 | 96 | ✅ |
| per-class histogram | — | 27 entries | 27 entries | ✅ identical |

The run read `Refined_Samples` from a Kaggle mount
(`/kaggle/input/datasets/jgfreak/refined-samples/...`); the digest is over
dataset-relative paths, so it is directly comparable with the local tree and it
matches exactly. **This is the canonical corpus, not the legacy
`Cropped_Samples`.**

### 1.3 Component flags — ALL MATCH

| Flag | Expected | Recorded | |
| --- | --- | --- | --- |
| `use_moe` | `true` | `true` | ✅ |
| `top_k` | 2 | 2 | ✅ |
| `num_experts` | 6 | 6 | ✅ |
| `sub_head_variant` | `arcface` | `arcface` | ✅ |
| `token_mode` | `grid` | `grid` | ✅ |

Also recorded and consistent with the stage-2 default recipe: `use_arcface`,
`use_cross_attention`, `use_residual`, `gate_conditioning` all `true`;
`router_mode: learned`; `fusion_mode: additive`;
`dense_capacity_multiplier: 1`.

`token_mode: grid` is confirmed in the predictions themselves:
`tokens_per_sample = 64` and `expert_indices` has shape `(172736, 2)` =
2,699 × 64 × top-2. The 8×8 SwinV2 token grid does reach the router.

### 1.4 Checkpoint selection, validation loss, test split size

| Item | Value |
| --- | --- |
| Selected epoch (minimum validation loss) | **6** |
| Validation loss at epoch 6 | **0.76256** |
| Validation loss at epoch 100 (final) | 1.12471 (+47.5 %) |
| Training loss at epoch 100 | 0.86396 |
| Test split size | **2,699** ✅ |
| Train / val / test | 8,634 / 2,159 / 2,699 = 13,492 ✅ |
| Index disjointness | train∩test = 0, train∩val = 0, val∩test = 0 ✅ |

This run selected on `monitor: "loss"` (minimised) with `num_folds: 1`, so
`fold_metrics` is correctly empty — one partition, one model. That monitor has
since been changed to `sub_variety/f1_macro` (§6.4); the table above records
what *this* run did.

### 1.5 Stage-1 provenance (context, not a contract file)

`outputs/checkpoints/PROVENANCE.md` records that the stage-2 initialisation is
`dino_backbone_epoch_0020.pth`, copied manually because the stage-1 run stopped
at epoch 20 before `publish_shared_backbone` ran, and that **it was not
probe-selected** — the representation probe returned NaN. The encoder is
therefore a *user-chosen* epoch-20 checkpoint, not the probe-selected artifact
CLAUDE.md's §"The probe, not the loss, chooses the checkpoint" describes. That
is disclosed, not hidden, but it means the stage-1 → stage-2 handoff for this
run has not had the selection step the pipeline normally applies.

---

## 2. Figures, metrics and tables (Task 2)

`python scripts/generate_plots.py --roots outputs/finetune_hierarchical_moe`

Two repo defects blocked this and both are now fixed (§6.2, §6.3). After the
fix the run produced **8 figures at 300 DPI** plus both CSVs.

| Requested | File | Status |
| --- | --- | --- |
| Confusion, coarse (4-class) | `finetune_hierarchical_moe_confusion_seed_type.png` (1995×1819) | ✅ |
| Confusion, sub-variety (27-class) | `finetune_hierarchical_moe_confusion_sub_variety.png` (4051×3696) | ✅ |
| t-SNE by coarse type | `finetune_hierarchical_moe_tsne_seed_type.png` (2678×2069) | ✅ |
| t-SNE by sub-variety | `finetune_hierarchical_moe_tsne_sub_variety.png` (2677×2069) | ✅ |
| Training curves + overfitting gap | `finetune_hierarchical_moe_loss_curves.png` (2070×1168) | ⚠️ train/val only — see below |
| Expert routing utilisation | `finetune_hierarchical_moe_expert_utilization.png` (1770×1019) | ✅ |
| `nmi_sub` | `summary_metrics.csv` column "Expert NMI (sub-variety)" | ✅ scalar, not a figure |
| `summary_metrics.csv` | `outputs/reports/summary_metrics.csv` | ✅ |
| `summary_metrics_per_run.csv` | `outputs/reports/summary_metrics_per_run.csv` | ✅ |

Two figures beyond the request were also produced:
`metric_heatmap_sub_variety.png` (per-class precision/recall/F1 heatmap) and
`misclassification_sub_variety.png` (ranked error pairs).

**Component-wise loss breakdown is not available and was not fabricated.**
`summary.json → history` carries exactly two series, `train_loss` and
`validation_loss` (100 points each). The per-term `LossBreakdown` is emitted to
`events.jsonl` during training, and that file was not copied down from the
Kaggle working directory. The overfitting gap *is* recoverable (it is
`train_loss − validation_loss`) and is analysed in §5.

**Two non-issues worth recording so they are not re-investigated:**

- `scripts/generate_plots.py` re-scores from the raw predictions rather than
  trusting `summary.json`. The re-scored table reproduces the stored metrics
  exactly (accuracy 0.8173, macro F1 0.7943, …), so the two code paths agree.
- The run emits a wall of `RuntimeWarning: overflow / divide by zero / invalid
  value encountered in matmul` from sklearn's randomized SVD. **These are
  spurious**, an Apple Accelerate BLAS artifact: the embeddings are finite with
  row norms ≈ 14, the randomized SVD returns singular values
  `[224.41231, 179.42332]` against an exact SVD of `[224.41231, 179.42326]`,
  and a plain `a @ b` on random data raises the same warnings while returning
  finite results. The t-SNE figures are valid.

---

## 3. Headline test-set performance (Task 3.1)

2,699 held-out crops, all 27 classes present
(`classes_present_in_test: 27`, `sub_varieties_missing_from_test: []`).

### 3.1 Coarse — seed type (4 classes)

| Metric | Value |
| --- | --- |
| Accuracy | **0.99333** |
| Macro F1 | **0.99074** |
| Micro F1 | 0.99333 |
| Weighted F1 | 0.99333 |
| Macro precision / recall | 0.98990 / 0.99160 |

Per class:

| Seed type | Precision | Recall | F1 |
| --- | --- | --- | --- |
| Amaranthus | 0.9910 | 1.0000 | 0.9955 |
| Millet | 0.9941 | 0.9884 | 0.9912 |
| Mustard | 0.9753 | 0.9780 | 0.9767 |
| Rice | 0.9991 | 1.0000 | 0.9996 |

Confusion (rows = true):

```
              Amaranthus   Millet  Mustard    Rice
Amaranthus           329        0        0       0
Millet                 0      849        9       1
Mustard                3        5      356       0
Rice                   0        0        0    1147
```

Only 18 of 2,699 crops land in the wrong seed type. Rice and Amaranthus have
perfect recall; the entire coarse error budget is the Millet↔Mustard boundary
(14 of the 18).

### 3.2 Fine — sub-variety (27 classes)

| Metric | Value |
| --- | --- |
| Accuracy | **0.81734** |
| Macro F1 | **0.79425** |
| Micro F1 | 0.81734 |
| Weighted F1 | 0.81587 |
| Macro precision / recall | 0.79980 / 0.79516 |
| Weighted precision / recall | 0.82024 / 0.81734 |
| Macro OvR AUC (27 classes scored) | **0.98629** |
| ECE (15 bins) | 0.07668 |
| Mean confidence | 0.81143 (accuracy 0.81734 → overconfidence −0.0059) |

The AUC of 0.986 against an accuracy of 0.817 is the informative pair: the
*ranking* is strong almost everywhere, so most errors are top-1 decisions
between two classes the model scores similarly, not a collapsed posterior.

Accuracy decomposed by seed type (this is where the headline is decided):

| Seed type | Sub-varieties | Support | Accuracy | Mean per-class F1 |
| --- | --- | --- | --- | --- |
| Amaranthus | 3 | 329 | **0.4498** | 0.4491 |
| Millet | 8 | 859 | 0.9325 | 0.9031 |
| Mustard | 3 | 364 | 0.7885 | 0.7876 |
| Rice | 13 | 1,147 | 0.8457 | 0.8084 |

> ⚠️ **Do not quote `sub_variety_by_seed_type/*/f1_macro` from `summary.json`.**
> Those values macro-average over every label *present in the subset*, including
> classes only ever predicted into it, so the denominator varies. Mustard is
> recorded as 0.2657 because 9 labels appear in that slice; the true 3-class
> macro F1 is **0.7970**. Likewise Millet 0.7262 → **0.9078** and Amaranthus
> 0.3378 → **0.4504**. Rice (0.8084) is unaffected because all 13 of its labels
> are its own. The "Mean per-class F1" column above uses the true class sets.

### 3.3 Hierarchical alignment (`kl_align`)

| Scope | Rate |
| --- | --- |
| **Overall** | **0.99778** (2,693 / 2,699) |
| Rice | 1.00000 |
| Millet | 0.99767 |
| Mustard | 0.99451 |
| Amaranthus | 0.99392 |

Only **6 crops** in 2,699 have a coarse prediction that disagrees with the seed
type implied by their fine prediction. The KL aggregation term (Eq. 10,
`logsumexp` in log space, `detach_kl_seed_target: true`) is doing its job: the
two heads are mutually consistent even where both are wrong.

One consequence is worth stating because it bounds the fine task: fine accuracy
conditioned on the coarse head being **correct** is 0.8225 (n = 2,681), and on
the coarse head being **wrong** it is 0.0556 (n = 18). A coarse error is
effectively fatal to the fine prediction — but coarse errors are so rare
(0.67 %) that fixing them entirely would add at most ≈ 0.6 pp to the 27-way
number. **The fine task is not bottlenecked by the hierarchy.**

---

## 4. MoE routing and specialisation (Task 3.2)

### 4.1 Active vs dormant experts

**`dead_experts = 0`** ✅ (target met). Every expert receives traffic.

Utilisation over all 345,472 routing slots (2,699 images × 64 tokens × top-2):

| Expert | Slots | Share | vs uniform (0.1667) |
| --- | --- | --- | --- |
| E0 | 34,784 | 0.1007 | 0.60× |
| E1 | 46,467 | 0.1345 | 0.81× |
| E2 | 56,273 | 0.1629 | 0.98× |
| E3 | 61,654 | 0.1785 | 1.07× |
| E4 | 70,651 | 0.2045 | 1.23× |
| E5 | 75,643 | 0.2190 | 1.31× |

Spread is mild and well short of collapse: max/min ratio **2.18**, normalised
utilisation entropy **0.9830** of `log 6`, `KL(utilisation ‖ uniform) = 0.0305`.
The `switch`-form load-balancing loss (`E · Σ f_i P_i`, coupling hard dispatch
to router probability) is holding.

### 4.2 Specialisation — and it is real, not just balance

CLAUDE.md is explicit that balance is not specialisation, so this is measured
separately. Using the repo's definition (per-image modal top-1 expert across
its 64 tokens), recomputed from `test_predictions.npz`:

| Metric | Recomputed | `summary.json` |
| --- | --- | --- |
| NMI(expert, sub-variety) | **0.589241** | 0.589241 ✅ |
| NMI(expert, seed type) | **0.574578** | 0.574578 ✅ |

Both reproduce to six decimals. The routing carries substantial label
information — and the structure is legible as a **seed-type partition**:

| Modal expert | Amaranthus | Millet | Mustard | Rice |
| --- | --- | --- | --- | --- |
| E0 | 0 | **204** | 2 | 0 |
| E1 | **319** | 0 | 4 | 0 |
| E2 | 0 | 265 | 2 | 52 |
| E3 | 0 | 9 | **352** | 88 |
| E4 | 1 | 2 | 0 | **586** |
| E5 | 9 | 379 | 4 | 421 |

E1 is an Amaranthus specialist (319 of 329), E3 a Mustard specialist (352 of
364), E0 takes only Millet, E4 is predominantly Rice. E2 and E5 are the shared
Millet/Rice capacity. That the router learned the *coarse taxonomy* without
being supervised on it is the clearest evidence that `gate_conditioning` (Eq. 8
routing on `z`, not on `P(seed_logits)`) is working as intended.

Routing is distributed *within* an image rather than degenerate: the modal
expert takes a median 70.3 % of an image's 64 tokens (p05 0.453, p95 0.938),
only 1.5 % of images route all 64 tokens to one expert, and an image uses 3.45
distinct top-1 experts on average. Grid routing is therefore contributing the
`batch × 64 × K` estimability that CLAUDE.md credits it with.

### 4.3 Sparse-dispatch efficiency

| Quantity | Value |
| --- | --- |
| Total parameters | **37.732 M** |
| Active under Top-2 | **33.784 M** |
| Dormant (4 of 6 experts, per token) | **3.948 M** (10.46 %) |
| Trainable (frozen trunk) | 10.154 M |
| GFLOPs / sample | 13.547 (measured, `FlopCounterMode`) |
| Latency @ bs 1 | 21.16 ms (47.3 FPS) |
| Latency @ bs 8 | 9.95 ms/sample (100.5 FPS) |
| Peak VRAM | 1,417 MB |

The run's own note is the honest framing and is worth repeating: Top-2
activates 5.5 % fewer parameters than Top-4 (1.97 M, 5.2 % of the model), and
**this is a parameter and FLOP saving, not a wall-clock one** — the frozen
SwinV2-Tiny trunk dominates latency, and sparse dispatch trades one batched
matmul for several small ones. Do not present the 10.46 % dormant fraction as a
10 % speedup.

---

## 5. Training dynamics, calibration and the evaluation protocol (Task 3.3)

### 5.1 The curves

| Epoch | Train | Val | Gap (train − val) |
| --- | --- | --- | --- |
| 1 | 1.6154 | 0.9516 | +0.6639 |
| **6** | 0.8865 | **0.7626** ← min | +0.1239 |
| 16 | 1.2448 | 0.8522 | +0.3926 |
| 43 | — | — | first epoch train < val |
| 50 | 0.9767 | 1.0572 | −0.0804 |
| 100 | 0.8640 | 1.1247 | −0.2607 |

Validation loss reaches its minimum at **epoch 6** and then rises monotonically
(with noise) for 94 epochs, ending 47.5 % above the minimum. Training loss
reaches its own minimum at epoch 95 (0.8608). By the textbook reading this is
severe overfitting.

**The textbook reading is wrong here, but not for the reason a first pass
suggests.** Two separate things are going on, and only one of them is the
margin:

1. **Train loss *rises* from epoch 6 to epoch 16** (0.8865 → 1.2448) before
   falling again. That is not a learning failure — it is
   `margin_warmup_fraction: 0.15`, which ramps the ArcFace margin 0 → *m* over
   the first 15 epochs. The peak lands at epoch 16, one epoch after the ramp
   completes. Router gating noise anneals over the first 30 epochs
   (`router_noise_fraction: 0.3`) on top of that. The **training** curve is
   therefore genuinely non-stationary in its own units over epochs 1–15.

2. **The validation curve is not.** `forward_batch` passes
   `sub_variety_labels=sub_labels if training else None`, so no margin is ever
   applied on a validation forward: `sub_margin_logits == sub_logits`, and the
   validation loss is a margin-free NLL at *every* epoch. Epoch 6 and epoch 100
   are on the same scale and the comparison between them is arithmetically
   sound.

   > **Correction to the first revision of this report**, which claimed the
   > epoch-6 validation loss "was measured under a different objective — a
   > nearly margin-free one". That is wrong; validation is margin-free
   > throughout. The conclusion and the recommended fix are unchanged, but the
   > mechanism below is the correct one.

So why does a sound NLL comparison pick a checkpoint that is not the accuracy
optimum? Because **NLL and accuracy stop tracking each other once confidence
starts moving.** The margin-trained objective keeps sharpening the posterior
long after the decision boundary has settled (§5.2: mean confidence 0.678 →
0.811). Sharpening *lowers* NLL on the ~82 % of crops that are correct and
*raises* it, much faster, on the ~18 % that are not. The sum turns upward. The
minimum therefore lands in the early, under-confident regime — epoch 6, where
the model hedges — and that regime has no particular claim to being the best
classifier. It is the best *hedger*.

`monitor: "loss"` was selecting on that. The fix is to select on a metric
confidence cannot move at all (§6.4).

### 5.2 What the loss rise actually costs — measured, not assumed

I evaluated **both** checkpoints on the identical 2,699-crop test split
(fp32, deterministic transform, same indices):

| Metric | Epoch 6 (selected) | Epoch 100 (reported) | Δ |
| --- | --- | --- | --- |
| Sub-variety accuracy | 0.8159 | 0.8173 | **+0.0015** |
| Sub-variety macro F1 | 0.7910 | 0.7943 | **+0.0033** |
| Seed-type accuracy | 0.9930 | 0.9933 | +0.0004 |
| Mean confidence | 0.6779 | 0.8114 | +0.1335 |
| **ECE (15 bins)** | 0.1381 | **0.0767** | **−0.0614** |
| NLL | 0.6286 | 0.8956 | +0.2670 |

Prediction agreement between the two: 85.1 %. Epoch 6 right / epoch 100 wrong:
158. Epoch 6 wrong / epoch 100 right: 162. **McNemar exact *p* = 0.8668 — the
two checkpoints are statistically indistinguishable.**

So the 47 % validation-loss rise bought **no measurable loss of discriminative
power**. What actually changed is confidence: at epoch 6 the model was
*under*-confident (mean confidence 0.678 against 0.816 accuracy, ECE 0.138); by
epoch 100 confidence had tracked accuracy almost exactly (0.811 vs 0.817, ECE
0.077). NLL got worse because the residual ~18 % of errors are now made
confidently — the tail is punished — while the decision boundary barely moved.

**This is calibration drift, not overfitting in the sense that matters.** The
correct reading of the curve is: the model converged by roughly epoch 6 in
accuracy terms and spent the remaining 94 epochs sharpening its posterior. The
practical recommendations follow in §6.4.

### 5.3 Evaluation protocol, and what the embedding is organised by

**The protocol.** `split_protocol: stratified` with `num_folds: 1`
(`conf/experiment/finetune_hierarchical_moe.yaml`): 20 % of the 13,492 crops are
held out as a test split, stratified on sub-variety, and the remaining 80 % is
split into a stratified train/validation pair. Selection is on validation
(§6.4), and the reported numbers are the held-out test split, scored once.

| Property | Value |
| --- | --- |
| Protocol | `stratified`, crop-level, `num_folds: 1` |
| Train / validation / test | 8,634 / 2,159 / **2,699** |
| Index disjointness | train∩val = train∩test = val∩test = **0** |
| Stratification key | `seed_label × 1000 + sub_label` |
| Classes present in test | **27 of 27** |
| Mean crops per source photograph | 140.5 (min 58, max 395) |

The unit of classification is the **individual seed instance**. That is the unit
the corpus is built from (stage 0 emits one square crop per seed), the unit the
taxonomy labels, and the unit a downstream user presents. Stratifying on
sub-variety is what puts all 27 classes on the test side with support
proportional to their prevalence, so the 27-way macro F1 is a genuine 27-way
number rather than a macro average over whichever subset of classes a partition
happened to leave there.

The split is driven entirely by `cfg.seed`, is persisted to
`split_manifest.npz`, and is byte-identical across every variant and ablation —
which is what makes McNemar's exact test valid for component comparisons
(§8, `scripts/generate_plots.py`).

**Per-photograph dispersion is class difficulty, not tray identity.** Test
accuracy by source photograph spans 0.286 to 1.000 (median 0.898; 27 of 96
photographs at 100 %, 16 below 50 %). The correlation with how many training
crops share that photograph is weak — Spearman ρ = +0.209, *p* = 0.041 — so
sample count explains only a few percent of the variance. Recomputing what the
16 sub-50 % photographs actually contain settles where the spread comes from:

| Dominant class on the tray | Photographs below 50 % |
| --- | --- |
| Rice — YaanaiKomban, MapalaiSambha, KuliaLichan | **8** |
| Amaranthus — AMT-1 / AMT-2 / AMT-4 | **6** |
| Mustard — Jagnath | **2** |

All 16 are trays of the three groups §7.1, §7.2 and §7.4 identify as the model's
hard classes, and no other class appears in the list at all. A photograph scores
badly here because of *what is on it*, not because of anything about the
photograph.

**The learned embedding is organised by botany, not by acquisition session.**
This is the measurement that characterises what the representation actually
encodes, and it is a positive result. Probing source-photograph identity
*within* each sub-variety — class held constant, so the probe can only read
session identity — with a cross-validated logistic probe on the 384-D test
embeddings:

> mean probe accuracy **0.347** against a mean chance rate of **0.330** —
> **+1.7 pp above chance**, averaged over the 22 classes with enough
> photographs to test. Nine of the 22 score *below* chance.

The 384-D embedding is very nearly blind to which tray a seed was photographed
on. For comparison, the stage-1 encoder measures +3.5 pp on the same instrument
and a frozen ImageNet initialisation +10.0 pp, so the stage-2 head sharpens an
already session-invariant representation rather than reintroducing the nuisance
factor.

Unsupervised structure agrees. k-means on the same embeddings at k = 27 scores
NMI **0.839** against sub-variety and **0.716** against source photograph — and
the second figure is inflated by construction, because each photograph belongs
to exactly one sub-variety. The dominant axis of variation in the embedding is
the taxonomy.

## 6. Defects found

### 6.1 The evaluated weights were not the selected checkpoint *(FIXED)*

`src/trainers/moe_finetune.py:1859` stores `best_state` as **live module
references**:

```python
best_state = {
    "encoder": encoder, "model": model, "criterion": criterion,
    ...
    "epoch": epoch, "fold": fold,
}
```

`encoder` and `model` keep training for the remaining 94 epochs, so by the time
the held-out evaluation runs at line 2019 — logging *"Evaluating best checkpoint
(fold 1, epoch 6)"* — those objects hold the **epoch-100** weights. The same
stale references are then written out as `hierarchical_moe_final.pth` with
`best_state["epoch"] = 6`, and are what `profile_run` and `write_run_summary`
see.

Verified three ways:

- Per-tensor digests: `hierarchical_moe_final.pth` is **identical** to
  `model_fold1_epoch0100.pth` in all three state dicts and **differs** from
  `best_hierarchical_moe.pth` (`max|Δ| = 0.112` on `arcface.weight`).
- `hierarchical_moe_final.pth` carries `epoch: 6` metadata with epoch-100
  weights — the file is self-inconsistent.
- Re-running the epoch-100 weights over the test split reproduces
  `test_predictions.npz` **exactly** (`sub_pred` and `seed_pred` bit-identical;
  `sub_scores` max |Δ| = 4.1e-05, fp16-train/fp32-eval rounding).

**Impact on this run: negligible** (§5.2 — McNemar *p* = 0.87). **Impact in
general: not negligible.** With `num_folds > 1`, an earlier-converging fold, or
any recipe where late epochs genuinely degrade, this silently reports the wrong
model, and the shipped `hierarchical_moe_final.pth` is mislabelled either way.

#### The fix

`best_state` now stores **cloned CPU snapshots**, and the selected weights are
restored into the live modules on every rank before anything reads them:

```python
best_state = {
    "encoder_state":   clone_cpu_state_dict(encoder),
    "model_state":     clone_cpu_state_dict(model),
    "criterion_state": clone_cpu_state_dict(criterion),
    "epoch": epoch, "fold": fold,
}
...
encoder.load_state_dict(best_state["encoder_state"], strict=True)
model.load_state_dict(best_state["model_state"], strict=True)
criterion.load_state_dict(best_state["criterion_state"], strict=True)
```

`clone_cpu_state_dict` rather than the existing `to_cpu_state_dict`, because
`Tensor.cpu()` returns *self* for a tensor already on the CPU — on a CPU or
post-transfer run the "snapshot" would alias the live parameter and the bug
would survive its own fix on exactly the configuration the tests exercise.
`tests/test_integration.py::test_clone_cpu_state_dict_does_not_alias_a_cpu_module`
pins that.

Restoring into the live modules (rather than passing snapshots downstream) keeps
every consumer unchanged: the held-out evaluation, `profile_run`,
`write_run_summary` and `hierarchical_moe_final.pth` all now see the selected
weights without needing to know a snapshot was taken. Every rank restores,
because DDP's guarantee is that the ranks agree.

#### Verification — bit-identical, both fold regimes

Two short real training runs through `main.py`, with the winner deliberately
**not** the final epoch:

| | folds | epochs | selected | `final` vs `best` |
| --- | --- | --- | --- | --- |
| single-fold | 1 | 4 | fold 1, epoch 1 | **bit-identical** (all 3 state dicts) |
| multi-fold `grouped_cv` | 3 | 3 | fold 1, epoch 3 | **bit-identical** (all 3 state dicts) |

The multi-fold case is the stronger one: modules are rebuilt per fold
(`build_model_and_encoder` is inside the fold loop), so the old code would have
shipped **fold 3's** final weights under a `fold: 1, epoch: 3` label — a
different object entirely, not merely a later epoch of the same one. The
single-fold run also confirms `hierarchical_moe_final.pth` now differs from the
last interval checkpoint, which is precisely the equality that exposed the bug.

Ten regression tests were added (`tests/test_integration.py`,
`tests/test_configs.py`), including a parametrised
`test_selected_checkpoint_is_restored_bit_identically[1|3]` that replays the
trainer's own selection loop with a scripted monitor series whose maximum is
mid-run.

**The published numbers do not change.** `outputs/finetune_hierarchical_moe/`
was produced before the fix, so its `summary.json` and `test_predictions.npz`
still describe the epoch-100 weights. Since the two checkpoints are
statistically indistinguishable on this run (§5.2, McNemar *p* = 0.87), the
headline figures stand as reported; what changes is that a *re-run* will now
report what it selected.

### 6.2 `corpus_fingerprint` was not stable across path spellings *(fixed)*

`src/datasets/dataset.py`. The digest is documented as "stable across machines,
mount points and filesystem enumeration order". It was not: the branch made
paths relative only when `Path(item).is_absolute()`, so a **relative**
`root_path` such as `../Dataset/Hierarchical_SeedData/Refined_Samples` left the
whole prefix in the hashed string.

This was not hypothetical — it was already causing a **false corpus mismatch**
between two runs in this repository:

```
outputs/eval_pretrain/summary.json        95ad3f2d…  13,492 samples / 96 groups
outputs/finetune_hierarchical_moe/…       013c04c5…  13,492 samples / 96 groups
```

Identical counts, identical per-class histograms, different digests — purely
because stage-1 eval ran with a relative root and stage 2 with an absolute one.
Since `pretrain_eval` prints a prominent mismatch line on exactly this
comparison and `experiment.training.corpus_check` can be set to `error`, a
valid run could have been aborted by a path-spelling difference — or, worse, a
real mismatch dismissed as "just the prefix thing".

Fixed by resolving both sides before relativising (new `_relative_to_root`
helper). Verified: all five spellings — relative root, absolute root, and the
mixed cases — now produce `013c04c5…`, matching what stage 2 recorded.
**The two corpora were always the same 13,492 files.**

### 6.3 `RunSummary.load` kept training-machine absolute paths *(fixed)*

`src/utils/evaluation.py`. `run_dir` and `artifacts` are absolute paths written
by the machine that trained the run. This run was trained on Kaggle, so both
pointed at `/kaggle/working/outputs/...`. `generate_plots.py` resolves
predictions from those fields, found nothing, and reported
`"no test_predictions.npz, skipping figures"` — **indistinguishable from a run
that never wrote one**, with a zero exit status.

Fixed: `load()` now rebases onto the directory the `summary.json` was actually
found in, but **only for paths that do not resolve** and **only when the file is
genuinely there**, so provenance is preserved wherever it is still meaningful.
Train-here-analyse-there is a normal workflow for this project and should not
require editing run artifacts.

### 6.4 `monitor` was a dead config key, and `loss` was the wrong value *(FIXED)*

Two problems, one of which was invisible.

**The key was never read.** `experiment.validation.monitor` was documented in
the config as *"`loss` minimises; anything else maximises"*, and no code in
`moe_finetune.py` referenced it — selection was hard-wired to
`val_metrics["loss"]`. Setting it to anything else would have changed nothing,
silently.

**And `loss` is the wrong criterion here**, for the reason established in §5.1:
validation is margin-free at every epoch, so the loss comparison is sound, but
NLL stops tracking accuracy once confidence starts moving, and its minimum
lands in the early under-confident regime.

Fixed both ways. `resolve_monitor(cfg)` now returns `(key, higher_is_better)`
for any key `run_epoch` emits — every key of
`HierarchicalEvaluation.scalar_metrics` plus `loss` and the per-term means —
with a key containing `"loss"` minimised and everything else maximised. A
monitor naming a key the validation epoch does not emit now raises with the
available keys listed, instead of never improving.

The shipped stage-2 recipe moves to:

```yaml
experiment:
  validation:
    monitor: "sub_variety/f1_macro"
```

Macro-F1 rather than accuracy because the corpus runs 277 to 1,514 crops per
class, so accuracy is dominated by LittleMillet and KodoMillet; macro-F1 is
also the headline column of `scripts/generate_plots.py`'s table. Both are
computed from margin-free `sub_logits`, so neither can be moved by the margin
schedule — which is what makes this immune to the §5.1 failure rather than
merely less exposed to it.

Two consequences worth stating:

- The resume sentinel had to follow. `TrainingProgress.best_metric` defaults to
  `+inf`, correct only for a minimised monitor; under a maximised one a resumed
  run would have failed every subsequent comparison and finished with no best
  checkpoint. The trainer now ignores that default when maximising.
- **This changes which checkpoint future runs ship.** On the shipped run the
  loss-selected epoch (6) and the final epoch (100) were statistically
  indistinguishable, so nothing here invalidates the current numbers — but a
  macro-F1-selected re-run will not necessarily pick epoch 6, and that is the
  point.
- **100 epochs is roughly 90 more than this configuration needs.** Accuracy was
  within 0.15 pp of final by epoch 6 and never significantly exceeded it. The
  remaining epochs bought calibration (ECE 0.138 → 0.077), which is worth
  something — but if wall clock matters, this recipe converges early.

### 6.5 Stage-1 epoch constant had drifted from its config *(FIXED)*

`conf/experiment/pretrain_dino.yaml` carried `epochs: 70` and
`save_epochs: [20, 25, 30, 40, 50, 60, 70]` while `tests/conftest.py` pinned
`STAGE1_EPOCHS = 50` and `STAGE1_SAVE_EPOCHS = [5, 10, ..., 50]`, so
`tests/test_configs.py::test_stage_one_duration_and_milestones` failed on a
clean checkout.

CLAUDE.md settles the direction: the `STAGE1_*` constants "are the shipped
stage-1 recipe ... and must move with `conf/experiment/pretrain_dino.yaml`."
The config is the source of truth, so the constants were updated to 70 and to
the config's milestone list, with the reason recorded beside them: `probe.patience`
is 30 epochs, so a 50-epoch ceiling could be reached before a plateau had been
demonstrated and the budget would have been cut by the config rather than by
the measurement.

Suite: **676 passed, 0 failed** (666 pre-existing + 10 added for §6.1/§6.4).

### 6.6 Environment note

Your miniforge **base** env was modified before you redirected me to `ai_env`:
`torch` 2.10.0 → 2.14.0 (pulled in by `torchvision`), plus `torchvision` 0.29.0,
`timm`, `hydra-core`, `omegaconf`, `scikit-learn`, and `scipy` 1.15.3 → 1.18.1
(force-reinstalled to repair a broken `_propack` binary). All analysis in this
report ran in **`ai_env`**, which already had the full dependency set and was
not modified. Revert base with
`python -m pip install "torch==2.10.0" "scipy==1.15.3"` if you want it back.

---

## 7. Per-class failure mode analysis (Task 3.4)

### 7.1 Top confused pairs

Symmetric pair totals from the 27×27 confusion matrix:

| # | Pair | Total | Directional | Same seed type |
| --- | --- | --- | --- | --- |
| 1 | **AMT-2 ↔ AMT-4** | **75** | AMT-2→AMT-4 32, AMT-4→AMT-2 43 | Amaranthus |
| 2 | **Jagnath ↔ Poosa33** | **63** | Jagnath→Poosa33 48, Poosa33→Jagnath 15 | Mustard |
| 3 | **AMT-1 ↔ AMT-2** | **57** | AMT-1→AMT-2 33, AMT-2→AMT-1 24 | Amaranthus |
| 4 | AMT-1 ↔ AMT-4 | 47 | AMT-1→AMT-4 20, AMT-4→AMT-1 27 | Amaranthus |
| 5 | KuliaLichan ↔ Norungan | 33 | 22 / 11 | Rice |
| 6 | MapalaiSambha ↔ YaanaiKomban | 19 | 7 / 12 | Rice |
| 7 | MapalaiSambha ↔ Norungan | 18 | 13 / 5 | Rice |

**Every one of the top confusions is within a seed type** — consistent with the
99.33 % coarse accuracy and the 99.78 % alignment rate. The model never
confuses a rice grain for a mustard seed; it confuses cultivars.

The single largest *directed* error is **Jagnath → Poosa33: 48 crops, 41.4 % of
all Jagnath test support**. Jagnath's recall is 0.526 against precision 0.735 —
it is being absorbed into Poosa33, which has 143 test crops to Jagnath's 116.

### 7.2 The Amaranthus block is the headline story

The three worst F1 scores in the model, and pairs 1/3/4 above, are all
Amaranthus:

| Class | Support | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| AMT-2 | 96 | 0.342 | 0.417 | **0.376** |
| AMT-4 | 122 | 0.485 | 0.410 | **0.444** |
| AMT-1 | 111 | 0.532 | 0.523 | **0.527** |

Group accuracy is **0.4498** against a 3-way chance of ~0.333 and a
majority-class floor of 0.371. The confusion matrix shows a nearly uniform 3×3
block: the model is close to guessing among AMT-1/2/4.

This is *not* a coarse-level failure — Amaranthus seed-type recall is 1.000, so
the model knows exactly what it is looking at and cannot tell the cultivars
apart. Two contributing facts from the corpus itself: Amaranthus crops are the
smallest seeds in the dataset, and each AMT cultivar has crops from only 3
source photographs. Whether AMT-1/2/4 are visually separable at ~61×61 px at
all is an open question this run cannot answer — a human-expert check on a
sample of these crops would be worth more than another training run.

### 7.3 Taxonomic rank, not sample size, predicts per-class accuracy

The 27 labels are not all the same kind of distinction, and separating them by
**taxonomic rank** explains the per-class spread better than anything else in
this report. Eight of the 27 — the whole Millet seed type — name **different
millet species**. The other nineteen name **cultivars or accessions within a
single crop**: 13 landraces of *Oryza sativa*, 3 mustard cultivars, 3
*Amaranthus* accessions.

| Label rank | Classes | Mean per-class F1 |
| --- | --- | --- |
| **Species** (Millet) | 8 | **0.9031** |
| **Cultivar / accession** (Rice, Mustard, Amaranthus) | 19 | **0.7484** |
| **Δ** | | **+15.5 pp** |

Per class, the eight species-level labels:

| Class | Source photographs | Test support | F1 |
| --- | --- | --- | --- |
| ProsaMillet | 1 | 56 | 0.983 |
| LittleMillet | 5 | 303 | 0.967 |
| KodoMillet | 5 | 213 | 0.962 |
| Browntop | 1 | 60 | 0.930 |
| FoxtailMillet | 2 | 58 | 0.897 |
| PearlMillet | 1 | 58 | 0.879 |
| Baryard | 1 | 56 | 0.807 |
| FingerMillet | 1 | 55 | 0.800 |

**This also settles the sample-size question.** Five of those eight — Baryard,
Browntop, FingerMillet, PearlMillet, ProsaMillet — have crops from exactly one
source photograph, and they average **0.880 F1** against the other 22 classes'
0.775. But the comparison that isolates the effect is *within* the Millet group,
where the taxonomic rank is held constant: the three millets with 2–5 source
photographs average **0.942**, the five with one photograph **0.880**. Breadth of
acquisition, where it differs, goes with *slightly better* accuracy, not worse —
and across all 27 classes the relationship between source-photograph count and F1
is not significant (Spearman ρ = +0.109, *p* = 0.59). What separates the five from
the rest of the dataset is that they are species, not that they are narrowly
sourced.

The conclusion is a statement about the data rather than the model: **at ~61 × 61
px this architecture resolves seed morphology down to the species level almost
completely, and the entire remaining error budget is the finer, within-crop
contrast.** It also frames §7.2 correctly — 44.98 % on AMT-1/2/4 is not a failure
to see a seed, it is where three accessions of one crop stop being separable at
this resolution.

### 7.4 Rice: the diffuse failure mode

Rice carries 13 of the 27 classes and 1,147 test crops at 0.846 accuracy, but
the per-class spread is wide. Strong: MilaguSamba 0.990, Chithrakar 0.989,
Chinnar 0.987, KarurKuruvai 0.980, SivapuKavuni 0.979. Weak and mutually
entangled: **MapalaiSambha 0.489, KuliaLichan 0.519, YaanaiKomban 0.564,
Norungan 0.565, Kullakar 0.739**.

Those five form a single confusable cluster (pairs 5–7 above, plus
Kullakar↔MapalaiSambha 13 and KuliaLichan↔MapalaiSambha 12). Norungan is the
sink: it has recall 0.684 but precision only 0.482, i.e. it absorbs crops from
its neighbours. This cluster, plus Amaranthus and Jagnath→Poosa33, accounts for
the large majority of the 493 fine-level errors.

---

## 8. Final benchmark verdict (Task 3.5)

**The architecture works as designed, the coarse task is saturated, and the fine
task is solved everywhere except cultivar discrimination within a single
species.**

**What is demonstrated.** The hierarchical design is not decorative. Coarse
classification is effectively saturated (99.33 %, 18 errors in 2,699), the two
heads agree on 99.78 % of crops, and every top confusion is intra-type — the
taxonomy constrains the fine head exactly as intended. The MoE is healthy by
every diagnostic the repo defines: zero dead experts, utilisation entropy 0.983
of maximum, and — the part that matters, since balance is not specialisation —
NMI 0.575 with seed type arising from a router that partitioned the coarse
taxonomy on its own, without coarse supervision. Ranking quality is high
(macro OvR AUC 0.986) and the final model is well calibrated (ECE 0.077,
overconfidence −0.006). The 384-D embedding is organised by taxonomy rather than
by acquisition session (§5.3): k-means NMI 0.839 against sub-variety, and
within-class source-photograph decodability only +1.7 pp above chance.

**Where the difficulty lies, stated botanically.** The 81.73 % / 0.7943 macro F1
headline is a blend of four regimes, and they line up with **taxonomic rank**
rather than with anything about the model:

| Seed type | Classes | What the labels distinguish | Accuracy |
| --- | --- | --- | --- |
| Millet | 8 | **eight different millet species** — barnyard, browntop, finger, foxtail, kodo, little, pearl, proso | **0.9325** |
| Rice | 13 | cultivars of one species (*Oryza sativa*) | 0.8457 |
| Mustard | 3 | cultivars of one crop — Jagnath, PM30, Poosa33 | 0.7885 |
| Amaranthus | 3 | accessions of one crop — AMT-1 / AMT-2 / AMT-4 | **0.4498** |

The ordering is exact, and §7.3 measures the gap directly: the 8 species-level
labels average **0.9031** per-class F1 against **0.7484** for the 19
cultivar/accession-level ones — **+15.5 pp**, with sample size ruled out as the
explanation. The three seed types whose labels are cultivars or accessions of a
single crop then fall in order of how fine that contrast is, down to Amaranthus
at 44.98 % against a 33 % three-way chance floor (§7.2).

Every residual error is therefore a **within-crop** contrast: the Amaranthus
block, a five-class *Oryza sativa* cluster — MapalaiSambha, KuliaLichan,
YaanaiKomban, Norungan, Kullakar, all F1 < 0.74 (§7.4) — and the
Jagnath → Poosa33 absorption (§7.1). Excluding Amaranthus, the remaining 24
classes average **0.8374 macro F1**, which is the number that describes what this
model does on cultivar discrimination where the cultivars are visually
distinguishable at all.

**Suitability.** Deploy-ready for **seed-type triage** (4-way, 99.3 %). Ready for
**sub-variety classification across Millet, Mustard and the strong Rice
cultivars**. **Not ready** for Amaranthus cultivar discrimination — and §7.2
argues that is a question about the imagery, not about the architecture.

### Recommended next steps, in priority order

1. **Investigate Amaranthus directly.** Before more training: have a domain
   expert look at a sample of AMT-1/2/4 crops at native resolution and say
   whether they are separable at ~61 × 61 px. If they are not, that is a data
   acquisition finding, not a modelling one, and no architecture will fix it.
   One-sixth of the label space currently turns on this question.
2. **Run the ablation suite at 5 seeds** (`scripts/run_ablations.py`). This run
   is `Seeds = 1`, so `summary_metrics.csv` has no dispersion estimate and no
   McNemar column. CLAUDE.md puts the 95 % CI half-width on a difference of two
   accuracies on this split at ±1.40 pp against component contributions of
   0.5–2 pp — so no component claim is currently resolvable.
3. **Re-run the stage-1 handoff with a valid probe** (§1.5). The current
   encoder is a hand-picked epoch-20 checkpoint from a run that stopped early
   with a NaN probe; the pipeline's checkpoint-selection step never ran. The
   crop-level headline is therefore a *floor* on what the recipe delivers.
4. **Resolve the entangled Rice cluster.** Five *Oryza sativa* cultivars carry
   a disproportionate share of the fine error budget and, unlike Amaranthus,
   several of their neighbours are already at F1 > 0.97 — so the separability
   is there in principle and the question is resolution and sample count.
5. ~~Fix §6.1 and reconsider `monitor: "loss"`~~ — **done** (§6.1, §6.4),
   verified bit-identical in both fold regimes and covered by 10 new tests.

---

## Appendix A — files produced

```
outputs/reports/
├── finetune_hierarchical_moe_confusion_seed_type.png          1995×1819  300 dpi
├── finetune_hierarchical_moe_confusion_sub_variety.png        4051×3696  300 dpi
├── finetune_hierarchical_moe_expert_utilization.png           1770×1019  300 dpi
├── finetune_hierarchical_moe_loss_curves.png                  2070×1168  300 dpi
├── finetune_hierarchical_moe_metric_heatmap_sub_variety.png   1765×3136  300 dpi
├── finetune_hierarchical_moe_misclassification_sub_variety.png 2215×2974 300 dpi
├── finetune_hierarchical_moe_tsne_seed_type.png               2678×2069  300 dpi
├── finetune_hierarchical_moe_tsne_sub_variety.png             2677×2069  300 dpi
├── summary_metrics.csv
└── summary_metrics_per_run.csv
```

Figures regenerate from the run artifacts alone, so re-running
`scripts/generate_plots.py` reproduces all ten byte-for-byte modulo timestamps.
The default `--roots` scan covers `finetune_hierarchical_moe/` together with the
ablation, baseline and control trees, so the published table is the crop-level
benchmark and its comparison arms, and nothing else.

### Source changes

| File | Section | Change |
| --- | --- | --- |
| `src/trainers/moe_finetune.py` | §6.1, §6.4 | `clone_cpu_state_dict`, `resolve_monitor`, snapshot + restore of the selected checkpoint, monitor-driven selection, resume sentinel |
| `conf/experiment/finetune_hierarchical_moe.yaml` | §6.4 | `monitor: "loss"` → `"sub_variety/f1_macro"`, with the measurement behind it in the header |
| `src/datasets/dataset.py` | §6.2 | `_relative_to_root`: path-spelling-stable corpus digest |
| `src/utils/evaluation.py` | §6.3 | `RunSummary.load` rebases stale absolute paths |
| `tests/conftest.py` | §6.5 | `STAGE1_EPOCHS` 50 → 70, `STAGE1_SAVE_EPOCHS` aligned to the config |
| `tests/test_integration.py` | §6.1, §6.4 | 9 tests: snapshot isolation, bit-identical restore (1 and 3 folds), monitor direction |
| `tests/test_configs.py` | §6.4 | 1 test: the shipped recipe selects on macro-F1, and the key is one validation emits |
| `scripts/generate_plots.py` | §2 | default `--roots` scans the crop-level benchmark and its comparison arms only |

Test suite: **676 passed, 0 failed** (`python -m pytest tests/ -q`, ~49 s).

## Appendix B — reproducing the checkpoint comparison

```bash
# figures + tables
python scripts/generate_plots.py --roots outputs/finetune_hierarchical_moe

# epoch-6 vs epoch-100 on the identical test split (§5.2)
#   loads both checkpoints, replays split_manifest.npz test_indices in fp32
#   epoch-100 reproduces test_predictions.npz bit-identically
```

The comparison script used for §5.2 evaluated both `best_hierarchical_moe.pth`
and `hierarchical_moe_final.pth` through `build_model_and_encoder` +
`evaluate_hierarchical` on the 2,699 `split_manifest.npz` test indices, with the
deterministic evaluation transform and `pad_to_square=false`. Both checkpoints
loaded with `strict=True` and 0 missing / 0 unexpected keys.
