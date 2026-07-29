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

### `top_k_saving(*modules, reference_top_k=4)` (`efficiency.py:233-260`)

Reports the configured Top-K's active-parameter count against a
`reference_top_k` (the submitted manuscript's Top-4), plus the absolute and
fractional saving:

```python
reference_active = total - (num_experts - reference_k) * per_expert
saved = reference_active - active
```

This is the mechanism behind the revision's central efficiency claim: moving
`top_k` from 4 to 2 doubles `dormant_parameters()`, and this function
surfaces exactly how many parameters that saves in the cost table.

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

### Latency, throughput, and memory (`benchmark_latency`, `peak_memory_mb`, `synchronize`)

```python
def benchmark_latency(model, example_input, device, batch_sizes=(1,8,32), warmup=3, iterations=10):
    for batch_size in batch_sizes:
        batch = _resize_batch(example_input, batch_size)   # tile or slice to the target size
        for _ in range(warmup): run()
        synchronize(device)
        started = time.perf_counter()
        for _ in range(iterations): run()
        synchronize(device)
        ...
```

(`efficiency.py:357-417`) Two details that are easy to get wrong and are
called out explicitly in the module docstring:

* **`synchronize(device)`** (`efficiency.py:316-328`) blocks on
  `torch.cuda.synchronize()` (CUDA) or `torch.mps.synchronize()` (MPS) before
  and after the timed loop. Both backends dispatch asynchronously; timing
  without this measures how fast Python can *enqueue* kernels, not how fast
  they *run* — typically off by an order of magnitude.
* **Warm-up iterations are untimed**, absorbing lazy kernel compilation and
  allocator warm-up that would otherwise dominate a short benchmark.

`peak_memory_mb(device)` (`efficiency.py:337-351`): CUDA reports a true
high-water mark (`torch.cuda.max_memory_allocated()`); MPS exposes only
*current* allocation (`torch.mps.current_allocated_memory()`), so the number
recorded there is a post-forward-pass snapshot, not a true peak — a note is
appended to the report so this is never silently mistaken for a peak. CPU
allocations are not tracked at all (returns `None`).

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

### `expert_utilization_counts(top_k_indices, num_experts)` (`metrics.py:282-292`)

Fraction of routing slots (not samples — each sample fills `top_k` slots)
taken by each expert; sums to 1. A perfectly balanced router gives `1 /
num_experts` everywhere regardless of `top_k`.

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

### `summary.json` (`RunSummary`, `evaluation.py:82-170`)

```python
RunSummary(name, group, run_dir, metrics, efficiency, history, component_flags, config, artifacts)
```

`metrics` holds the flattened output of `HierarchicalEvaluation.scalar_metrics()`.
`.as_row()` (`evaluation.py:137-170`) produces one CSV row — **the headline
accuracy/precision/recall/F1 columns describe the sub-variety task**
(`sub_variety/accuracy`, not `seed_type/accuracy`), since that is the 27-class
problem the architecture exists to solve and the one where variants actually
separate; seed-type numbers are appended as extra columns rather than
averaged in, which would blend two tasks of very different difficulty into a
meaningless composite figure.

`REQUESTED_COLUMNS` (`evaluation.py:51-62`, in this exact order):

```text
Model/Variant, Accuracy, Precision, Recall, Macro F1, Micro F1,
KL Alignment Rate (%), Total Params (M), Active Params (M), Inference Latency (ms)
```

`EXTRA_COLUMNS` (`evaluation.py:65-76`) appended after: seed-type accuracy/F1,
sub-variety AUC, expert/top-k counts, throughput, GFLOPs/sample, peak memory,
group, run directory.

`collect_run_summaries(roots)` (`evaluation.py:173-188`) globs
`summary.json` at up to two directory levels under each root, de-duplicates,
and sorts by `(group, name)`.

`write_summary_csv` (`evaluation.py:191-222`) writes **blank cells for
missing measurements**, never `nan` — a blank reads unambiguously as "not
measured," whereas `nan` in a results table invites a reader to mistake it
for a failed run.

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
