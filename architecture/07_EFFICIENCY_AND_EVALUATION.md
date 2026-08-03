# 07 — Efficiency Profiling and Evaluation Pipeline

Covers `src/utils/{efficiency,metrics,evaluation,visualization}.py`, the
epoch loop in `src/trainers/moe_finetune.py`, and `scripts/generate_plots.py`.

## 1. Parameter, FLOP, latency, and memory accounting (`src/utils/efficiency.py`)

The claim a sparse MoE makes is that capacity and compute can be decoupled:
the model *owns* six experts but *evaluates* only `K` per sample. A plain
parameter count cannot express that, so two numbers are always reported side
by side.

### `ParameterReport` (`efficiency.py:55-85`)

```python
total, trainable, active, dormant, num_experts, top_k
active = total - dormant
dormant_fraction = dormant / total
```

```python
count_parameters(*modules) -> (total, trainable)     # de-duplicated by parameter identity
count_dormant_parameters(*modules) -> (dormant, num_experts, top_k)
```

(`efficiency.py:176-216`) `count_dormant_parameters` walks every submodule
looking for `MixtureOfExperts` or `DenseExpertBlock` instances and sums their
`dormant_parameters()` (see [`03_MOE_MODULE.md`](03_MOE_MODULE.md)). A model
with no router at all reports `(0, 1, 1)` — nothing dormant, one always-on
path, the honest description of a dense network. **This is a closed-form
calculation, not a profiling estimate** — every expert shares one
architecture, so counting one and multiplying by `(E − K)` is exact.

### `top_k_saving(*modules, reference_top_k=4)`

Reports the active-parameter count at the configured Top-K against a reference
Top-4, plus the reduction between them. Returns an empty dict when the model has
no router, where the comparison is meaningless.

### What the efficiency claim is, and what it is not

The accounting machinery is exact — closed-form `dormant_parameters()`, real ATen
FLOP counting rather than a hand formula, `synchronize()` around the timed loop.
The framing needed to change.

**Magnitude.** Per-expert parameters for `TransformerExpert(384, mlp=512,
heads=8)` under **grid** routing (real attention):

```text
MultiheadAttention   591,360   (in_proj 442,368 + 1,152; out_proj 147,456 + 384)
LayerNorm x 2          1,536
MLP 384 -> 512 -> 384        394,112
                     ---------
per expert           987,008
6 experts          5,922,048
dormant @ K=2      3,948,032        dormant @ K=4  1,974,016
```

Against a full model of ~96.5 M (SwinV2-Base ~86.9 M + Eq. 4 projection ~0.39 M +
head ~9.2 M):

```text
dormant_fraction @ K=2  ~ 4.1 %
Top-4 -> Top-2 saving   ~ 1.97 M  =  2.0 % of total parameters
```

**A 2 % parameter delta is a thin foundation for a headline efficiency claim**,
and the paper should state it as the small number it is. Note also that under the
submitted **pooled** routing roughly 1.18 M of that "saving" was the dead Q/K
projections — parameters that were never computed in the first place. Those are
no longer allocated, so the number now means what it says.

**It will not appear in wall-clock, and the profiler is honest enough to show
that.** `profile_model` correctly measures encoder + head together, and the frozen
86.9 M SwinV2-Base dominates. Meanwhile `_sparse_forward` replaces one batched
matmul with up to six gather -> matmul -> scatter-add sequences; at these batch
sizes the kernel-launch overhead plausibly makes sparse dispatch *slower* than
dense.

`EfficiencyReport.notes` therefore says so explicitly, in the report itself:

```text
Top-2 activates 33.3% fewer parameters than Top-4 (1.97 M, 2.0% of the total
model). This is a parameter and FLOP saving, not a wall-clock one: the frozen
backbone dominates latency and sparse dispatch trades one batched matmul for
several small ones.
```

The "Inference Latency (ms)" column sits next to "Active Params (M)" in the
results table, and must not be read as caused by it.

### FLOPs (`estimate_flops`, `efficiency.py:273-310`)

```python
from torch.utils.flop_counter import FlopCounterMode
with torch.no_grad(), FlopCounterMode(display=False):
    forward_fn() if forward_fn else model(example_input)
```

Counts real dispatched ATen operations rather than a hand-derived formula —
critical for a sparsely-routed MoE, since a formula would have to model the
router's runtime behavior, whereas the dispatch counter simply observes that
only `K` experts actually ran. Degrades to `None` (with a note appended to
the report) if the installed torch build doesn't expose the counter, never
raising.

### Latency, throughput, and memory

```python
benchmark_latency(model, example_input, device, batch_sizes=(1, 8, 32),
                  warmup=5, iterations=50, forward_fn=None, sample_pool=None)
```

`synchronize(device)` is called around the timed region. Both CUDA and MPS
dispatch asynchronously, so timing without it measures how fast Python can
*enqueue* kernels rather than how fast they run — typically off by an order of
magnitude. `warmup` iterations are untimed, absorbing lazy kernel compilation and
allocator warm-up.

**Statistics.** Each iteration is timed individually and the report carries the
**median**, the **IQR**, min, max, and the iteration count. `warmup=3,
iterations=10` gave a mean with no dispersion estimate at all, on a measurement
that a table row was being built on; the median is robust to the outliers that
still appear just after warm-up, and the IQR says how much to trust it.

**Batch composition.** `_resize_batch` tiles to reach the target batch size.
Tiling *one* example to batch 32 gives 32 identical gate logits, hence 32
identical Top-K selections, hence exactly `K` expert kernels launched for the
whole batch instead of the up-to-`E` that diverse data produces. That
systematically **understates** the dispatch overhead of sparse routing at larger
batches — which is the one thing the batch-32 row exists to measure.

`sample_pool` supplies real, distinct images; the trainer passes the held-out test
indices. When it cannot, the report is annotated:

```text
Latency batches were tiled from too few distinct samples: identical rows produce
identical routing, so sparse-dispatch overhead is understated at the larger batch
sizes.
```

**Memory.** CUDA tracks a true high-water mark. MPS exposes only *current*
allocation, so the number there is a snapshot taken right after the forward pass;
the report says so rather than hiding it.

### `profile_model(model, example_input, device, extra_modules=[], ...)` (`efficiency.py:433-502`)

The front door, called once per run from
`profile_run()` in `moe_finetune.py:611-660`:

```python
def forward(batch):
    return model(encoder(batch))
profile_model(model, example, device, extra_modules=[encoder], forward_fn=forward, ...)
```

**Profiles encoder + head together**, since that is the only combination
whose latency a real deployment would ever observe — profiling the head
alone would report a number no user could see. `batch_sizes: [1, 8, 32]`
(`conf/experiment/finetune_hierarchical_moe.yaml:64`), `warmup: 3`,
`iterations: 10` by default; batch size 1 doubles as the FLOP-counting batch
to give a clean per-sample figure. Every measurement degrades to `None`
rather than raising, so a profiling failure never aborts a training run
(`profile_run` wraps the whole call in `try/except`, logging a warning).

`EfficiencyReport.summary_line()` produces the one-line form logged to the
training log; `.as_metrics()` flattens into tracker scalars
(`efficiency/total_parameters_m`, `efficiency/active_parameters_m`,
`efficiency/dormant_fraction`, `efficiency/gflops_per_sample`,
`efficiency/latency_ms_bs{N}`, `efficiency/throughput_fps_bs{N}`, etc.).

## 2. Metrics — one evaluation pass (`src/utils/metrics.py`)

| Paper artefact | Function |
| --- | --- |
| Accuracy, macro/micro/weighted F1 | `classification_metrics` |
| Table 2 — per-class precision/recall/F1 | `per_class_metrics` |
| "Area under the ROC curve" | `roc_auc_ovr` |
| Table 3 — KL alignment rate | `kl_alignment_rate` |
| Fig. 10 — confusion matrix | `confusion_matrices` |
| Fig. 12 — misclassification rates | `misclassification_rates` |
| §5.2 — expert utilization | `expert_utilization_counts` |
| Figs. 8-9 — t-SNE | `tsne_projection` |

### `classification_metrics(y_true, y_pred)` (`metrics.py:83-109`)

Returns `accuracy`, `f1_{macro,micro,weighted}`,
`precision_{macro,weighted}`, `recall_{macro,weighted}`. **`f1_micro`
is numerically identical to `accuracy` for single-label multi-class
predictions** — every error is simultaneously one false positive and one
false negative, so micro-averaged precision and recall both collapse to
accuracy. It is reported because the revision's requested table asks for it,
not because it adds information; the docstring says so explicitly.

### `kl_alignment_rate(...) -> AlignmentReport` (`metrics.py:184-232`)

The paper names this metric (95.94% overall in Table 3) but never defines
it. The implemented definition:

$$
\text{aligned}_i = \big(\text{parent}[\arg\max \, \text{sub\_logits}_i] == \arg\max \, \text{seed\_logits}_i\big)
$$

```python
aligned = parent[sub_pred] == seed_pred     # parent = subvariety_to_seed_type array
```

The **overall** rate is `aligned.mean()`. The **per-seed-type breakdown**
groups by the **true** seed-type label (`seed_type_labels`, when supplied) —
that is what makes a row like "Mustard: 0.7189" a statement about samples
whose true seed type is mustard, not about samples predicted as mustard. When
`seed_type_labels` is omitted the breakdown falls back to grouping by
predicted seed type instead.

### `roc_auc_ovr(y_true, y_score, num_classes)` (`metrics.py:148-181`)

`sklearn.metrics.roc_auc_score(..., multi_class="ovr")` requires every class
to appear in `y_true`, which a stratified validation fold over 27
fine-grained classes does not guarantee. This function instead computes the
binary AUC **per class that has both positive and negative samples present**,
macro-averages what survives, and reports `auc_classes_scored` so a partial
average is never silently mistaken for a full 27-class one.

### `confusion_matrices(y_true, y_pred, num_classes, normalize=False)` (`metrics.py:258-279`)

`normalize=True` divides each row by its support — necessary for the 27-class
sub-variety matrix, where raw counts would make a frequent class look
accurate purely by being frequent.

### `expert_utilization_counts(top_k_indices, num_experts)`

Fraction of routing slots each expert took, summing to 1, so a balanced router
gives `1/num_experts` everywhere regardless of Top-K. This is the **hard dispatch
fraction `f`** — and it is now also what the load-balancing loss controls.

Previously the figure measured `f` and the loss measured `P` (the batch-mean soft
gate), so the diagnostic and the objective disagreed about what "balanced" meant.
The diagnostic was the correct one. See
[`03_MOE_MODULE.md`](03_MOE_MODULE.md) §3.

### `expert_label_nmi(expert_indices, labels, tokens_per_sample=1)`

Normalised mutual information between the top-1 routed expert and the label.

Utilisation bars show **balance**, not **specialisation**: six experts each taking
a sixth of the traffic at random are perfectly balanced and perfectly
uninformative, and the submitted diagnostics could not tell that apart from expert
specialisation. NMI can, and it makes "expert specialisation" a measured claim
rather than an asserted one — including for `gate_conditioning`, whose whole
purpose is to raise it.

Computable from an existing `test_predictions.npz` with **no retraining**. Under
grid routing there are `tokens_per_sample` routing rows per label, so the per-image
expert is the modal top-1 choice across that image's tokens.

### `expected_calibration_error` / `fit_temperature`

`sub_scores` are `softmax(s cos theta)` and therefore overconfident by
construction — at the submitted `s = 30`, extremely so. A hierarchical
classifier's practical value depends on trustworthy confidence, so ECE is reported
alongside `overconfidence` (the signed confidence-minus-accuracy gap, which says
*which way* the model is wrong).

`fit_temperature` is fitted on **validation** logits and applied to test ones;
fitting on test would make the resulting ECE meaningless. `test_predictions.npz`
stores raw `sub_logits` for this, because recovering them from float32 softmax
output is not reliable.

### `mcnemar_test` / `holm_bonferroni`

McNemar's exact paired test between two classifiers on the same samples. Valid
precisely because the suite guarantees a byte-identical test split across
variants: the prediction vectors are **paired**, so the discordant counts
(`n01`, `n10`) are the whole of the evidence and the concordant ones carry none.
That makes it both valid and strictly more powerful than comparing independent
confidence intervals.

Implemented on `scipy.stats.binomtest`, so the reporting path adds no new
dependency. `holm_bonferroni` step-down-adjusts across the family — six ablations
against one reference is six chances to find a difference, and without correction
the family-wise error rate is around 26 % at nominal 0.05.

### `per_seed_type_breakdown`

Sub-variety accuracy and macro-F1 *within* each seed type. The hierarchy is 13
rice + 8 millet + 3 + 3, so an overall figure is structurally dominated by rice;
this makes each branch's difficulty visible separately. `class_distribution()`
existed and Fig. 1 plotted it, but nothing consumed it.

### `tsne_projection(embeddings, perplexity=30.0, max_samples=2000)` (`metrics.py:295-323`)

Returns `None` (rather than raising) when there are too few samples for a
meaningful perplexity — t-SNE requires `perplexity < n_samples`. Effective
perplexity is clamped to `max((n-1)/3, 1)` when the batch is small.

### `evaluate_hierarchical(...) -> HierarchicalEvaluation` (`metrics.py:357-399`)

Runs every function above over one set of predictions and returns a single
dataclass. `.scalar_metrics()` (`metrics.py:340-354`) flattens everything
into tracker-ready `prefix/name` keys (`seed_type/accuracy`,
`sub_variety/f1_macro`, `kl_alignment/overall`, `kl_alignment/{seed_name}`,
`moe/expert_{i}_utilization`, etc.) — this is the dict that becomes both the
per-epoch tracked metrics and (at test time) the contents of
`summary.json["metrics"]`.

## 3. The stage-2 epoch loop (`src/trainers/moe_finetune.py`)

### `EpochAccumulator` (`moe_finetune.py:255-303`)

Collects predictions, scores, embeddings, and expert indices across an
**entire** epoch before computing any metric:

```python
def update(self, output, loss_parts, seed_labels, sub_labels, keep_embeddings): ...
```

This is deliberate — averaging per-batch accuracy is *not* the same number
as accuracy computed over the whole epoch, and the paper reports the latter.
`keep_embeddings` caps how many embeddings are retained for the t-SNE panel
(`max_tsne_samples`, default 2000) so a large dataset doesn't force the whole
epoch's 384-D embeddings into memory.

### `forward_batch` (`moe_finetune.py:334-357`)

```python
features = encoder(images)
output = model(features, sub_variety_labels=sub_labels if training else None)
breakdown = criterion(output, seed_labels, sub_labels)
```

**The ArcFace margin is applied only during training** — at evaluation,
`sub_variety_labels=None` is passed, so `ArcFaceHead.forward` returns
`sub_margin_logits == sub_logits` and reported metrics are never inflated by
the training-time margin (see [`04_HIERARCHICAL_FUSION.md`](04_HIERARCHICAL_FUSION.md)).

### `run_epoch` (`moe_finetune.py:360-473`)

Runs one train or evaluation epoch: `is_train = optimizer is not None` gates
`backward()` + gradient clipping + `optimizer.step()`; both branches funnel
through the same `forward_batch` and `EpochAccumulator`. At the end, it calls
`evaluate_hierarchical(...)` once over the accumulated epoch's predictions and
logs the flattened metrics dict with `tracker.log_metrics(metrics, epoch,
prefix=phase)`.

### `LossHistory` — the overfitting diagnostic (`moe_finetune.py:306-331`)

```python
@property
def latest_gap(self):
    return self.validation[-1] - self.train[-1]
```

Logged every epoch as `epoch/overfitting_gap`. On a 27-class problem with a
few hundred samples per class, a validation loss that turns upward while
training loss keeps falling is the failure mode most worth watching for — the
side-by-side train/validation loss figure (`plot_loss_curves`) makes this
visible directly.

### `log_evaluation_artifacts` (`moe_finetune.py:479-591`)

Pushes every Section 6 figure/table to the tracker: confusion matrices at
both hierarchy levels, sub-variety misclassification bar chart, per-class
metric heatmap, expert-utilization bars (+ a routing histogram), per-class
precision/recall/F1 tables, the KL alignment table, and t-SNE projections
colored by both label levels. **The entire function body is wrapped in a
`try/except`** that logs a warning on failure — figures are diagnostics and
must never abort a training run.

### Efficiency + run artifacts, at the end of training

```python
efficiency = profile_run(best_state["encoder"], best_state["model"], dataset, cfg, device, logger)
summary_path = write_run_summary(cfg, output_dir, best_state["model"], test_evaluation,
                                  test_accumulator, dataset, efficiency, history, final_path, logger)
```

(`moe_finetune.py:938-958`) `profile_run` measures the **best** checkpoint
(lowest validation loss across all folds), not the final epoch's weights.
`write_run_summary` (`moe_finetune.py:973-1034`) writes both contract files
described below, even when there is no held-out test split (the metrics block
is simply empty in that case) — a run must always leave a machine-readable
trace.

## 4. The cross-run reporting contract (`src/utils/evaluation.py`)

Two files per run, written into `experiment.training.save_path`:

### `test_predictions.npz` (`save_test_predictions`, `evaluation.py:228-263`)

Raw held-out `seed_true/pred`, `sub_true/pred`, `sub_scores` (softmax probs),
384-D `embeddings`, `expert_indices` (routed Top-K indices), and class-name
arrays. Keeping the **raw arrays**, not only figures, means a reviewer asking
for a differently-normalized confusion matrix or a re-colored t-SNE costs a
second of replotting rather than a full retrain.

### `summary.json` (`RunSummary`)

```text
name, group, run_dir
metrics          # every scalar from HierarchicalEvaluation.scalar_metrics()
efficiency       # EfficiencyReport.as_dict()
history          # train/validation loss series
component_flags  # every architectural axis a variant can move
loss_flags       # every loss-side setting a variant can move
split            # protocol, seed, provenance and leakage diagnostics
fold_metrics     # {metric: {mean, std, min, max, folds}} across CV folds
config           # backbone, head, token_mode, top_k, epochs, num_folds, lr, seed
artifacts        # checkpoint + predictions paths
```

Three of these blocks are new, and each closes a specific hole:

**`loss_flags`.** `component_flags()` used to report only the four architectural
booleans, deliberately excluding `use_kl_loss`. So a `wo_kl` run's
`component_flags` field was **byte-identical to `full_model`'s**, and only the
variant *name* separated them. For a tree whose design philosophy is "a run must
always leave a machine-readable trace", that was the wrong omission.
`loss_flags()` now carries `use_kl_loss`, `kl_mode`, `tau_kl`,
`detach_kl_seed_target`, `weighting_mode`, `cosine_mode`, `moe_load_mode`,
`moe_sparsity_mode` and all nine lambdas.

**`split`.** The headline accuracy is uninterpretable without knowing whether
crops of the same source photograph could appear on both sides. This records the
protocol, the number of source photographs, the crops-per-source ratio, how many
groups straddle the boundary, and any sub-variety missing from either partition.

**`fold_metrics`.** `profile_run` measures the **best** checkpoint (lowest
validation loss across all folds), and `write_run_summary` used to report that
checkpoint's test evaluation. Taking a maximum over `K` folds and reporting the
corresponding test score is a selection procedure whose expected value exceeds a
single fold's — so the moment anyone ran `num_folds=5` for the variance the
protocol needs, the numbers would become optimistic **and** incomparable with the
`num_folds=1` numbers already collected. Each fold is now scored on the held-out
split and aggregated as mean ± std; best-fold selection survives only for the
artifact that gets profiled and shipped.

### Multi-seed aggregation and paired testing

`collect_run_summaries` globs three directory levels, which is what the
`{group}/{variant}/seed{n}/` layout needs. `aggregate_by_variant` collapses
repeated seeds into one row carrying `{column}` (the mean), `{column} SD` and
`Seeds`. `paired_significance` runs McNemar against `full_model` and fills the
`p (vs full)` and `p (Holm)` columns.

`generate_plots.py` writes both tables: `summary_metrics.csv` (aggregated) and
`summary_metrics_per_run.csv` (one row per run).

### `save_publication_figures` (`evaluation.py:275-378`)

Re-derives every figure at `dpi=300` (print resolution, vs. 150 for live
training-log figures) from a `HierarchicalEvaluation` plus the raw embeddings
— confusion matrices (row-normalized), the metric heatmap, misclassification
bars, expert utilization, loss curves, and both t-SNE panels.

## 5. Figures (`src/utils/visualization.py`)

All functions return a `matplotlib.figure.Figure` (the `Agg` backend is
forced at import — training runs are headless); the caller decides whether to
save, push to TensorBoard, or log to W&B.

| Function | Notes |
| --- | --- |
| `plot_confusion_matrix` | Annotates cells only when `matrix.shape[0] <= annotate_threshold` (default 12) — a 27×27 grid of 729 numbers is unreadable. Canvas size (`_figure_size`) grows with label count so 27 tick labels stay legible; labels are never abbreviated. |
| `plot_metric_heatmap` | Precision/recall/F1 per class, `viridis` colormap. |
| `plot_misclassification_rates` | Horizontal bars of `1 - recall`, sorted descending. |
| `plot_expert_utilization` | Draws a dashed reference line at `1/num_experts` — the entropy load-balancing term ([`03_MOE_MODULE.md`](03_MOE_MODULE.md)) is exactly the pressure pulling the bars toward it, so collapse is visible at a glance. |
| `plot_tsne(annotate_clusters=True)` | Prints each class name at its cluster's **median** position (not mean — t-SNE routinely strands a few points far from a class's main mass, and a mean would drag the label into empty space between clusters), atop the color legend. Skips the overlay for clusters with `< min_annotated_cluster` (3) points. |
| `plot_loss_curves` | One line per named series — used for both DINO pretraining loss (Fig. 6) and the train/validation overfitting diagnostic. |

## 6. `scripts/generate_plots.py` — the offline report builder

```bash
python scripts/generate_plots.py
python scripts/generate_plots.py --roots outputs/ablations outputs/baselines
python scripts/generate_plots.py --dpi 600 --no-figures
```

Reads only `summary.json` and `test_predictions.npz` from every run under
`--roots` (default: `ablations/`, `baselines/`, `finetune_hierarchical_moe/`)
— **nothing here retrains or reloads a model**, so re-plotting at a different
DPI or normalization costs seconds. Critically, `regenerate_figures`
(`generate_plots.py:76-119`) **re-scores from the raw predictions** via
`evaluate_hierarchical(...)` rather than trusting the metrics already baked
into `summary.json` — this is what guarantees the CSV table and the
figures are always computed by exactly the same code, at plot time, even if
the metrics implementation has evolved since the run finished. Output:
`outputs/reports/summary_metrics.csv` plus `{variant}_*.png` for every figure
type in §5.
