# `src/utils/` — metrics, efficiency, reporting, figures, infrastructure

| File | Contents |
| --- | --- |
| `metrics.py` | Every quantity the paper's Section 6 reports, for **one** set of predictions |
| `representation.py` | Stage-1 metrics: geometry, frozen-feature readouts, unsupervised structure |
| `efficiency.py` | Parameter accounting, FLOPs, latency, throughput, peak memory |
| `evaluation.py` | The cross-run layer: prediction dumps, `summary.json`, the comparison CSV, stage-1 event-stream recovery |
| `visualization.py` | Matplotlib figures for the paper's plots and the stage-1 report |
| `training/` | Tracker, checkpoints, logging, device selection, attention maps |

The split between `metrics` and `evaluation` is by scope: `metrics.py` scores one
evaluation pass in memory, `evaluation.py` defines the on-disk contract that lets
a *later* process compare runs that finished hours apart on different machines.

The split between `metrics` and `representation` is by *what exists*. `metrics.py`
scores the predictions of a trained classifier. After stage 1 there is no
classifier — DINO leaves an encoder and a 2,048-way prototype head whose classes
have no names — and the DINO loss is a cross-entropy against a teacher that moved
during training, so it is comparable to nothing. `representation.py` is the
instrument set for that situation:

| Family | Functions | Detects |
| --- | --- | --- |
| Label-free geometry | `spectral_report` (RankMe, participation ratio), `feature_statistics`, `alignment_uniformity`, `augmentation_consistency`, `linear_cka` | dimensional collapse, dead channels, whether the trained-for invariance actually holds |
| Frozen-feature readout | `linear_probe`, `select_regularisation`, `low_shot_probe`, `knn_classifier` | whether the classes are linearly separable, and separable at all under plain cosine distance |
| Structure without labels | `kmeans_report`, `prototype_report`, `retrieval_report`, `class_separability`, `centroid_similarity_matrix` | whether the taxonomy emerged *before* any label was seen |

They fail independently, which is the point: a probe far above the k-NN says the
classes are linearly separable but not compactly clustered — the exact gap stage
2's ArcFace margin and compactness terms exist to close.
`architecture/08_STAGE1_REPRESENTATION_EVALUATION.md` states each definition and
why it was chosen.

## `metrics.py`

| Paper artefact | Function |
| --- | --- |
| Section 6.2.1 / 6.3 — accuracy, F1 | `classification_metrics` |
| Table 2 — per-class precision / recall / F1 | `per_class_metrics` |
| Section 6.2 — "area under the ROC curve" | `roc_auc_ovr` |
| Table 3 — KL alignment rate | `kl_alignment_rate` |
| Fig. 10 — confusion matrix | `confusion_matrices` |
| Fig. 12 — misclassification rates | `misclassification_rates` |
| Section 5.2 — expert utilisation | `expert_utilization_counts` |
| Figs. 8-9 — t-SNE | `tsne_projection` |

`evaluate_hierarchical` runs all of them and returns a `HierarchicalEvaluation`;
`.scalar_metrics()` flattens it into tracker-ready `prefix/name` keys.

### The alignment rate

The paper reports it (95.94% overall, per-seed-type in Table 3) but never defines
it. The implemented definition: a sample is aligned when the parent seed type of
the **predicted** sub-variety equals the **predicted** seed type.

```
alignedᵢ = parent[argmax sub_logitsᵢ] == argmax seed_logitsᵢ
```

The overall rate is the mean; the per-type breakdown groups by **true** seed
type, which is what makes a row like "Mustard: 0.7189" a statement about mustard
samples rather than about samples predicted as mustard.

### Why AUC is computed class-by-class

`sklearn`'s `roc_auc_score(..., multi_class="ovr")` requires every class to be
present in `y_true`. A stratified validation fold over 27 fine-grained classes
does not guarantee that. `roc_auc_ovr` computes the binary AUC for each class
that has both positive and negative samples, macro-averages what survives, and
reports `auc_classes_scored` so a partial average is never mistaken for a full
one.

### Micro F1 and accuracy always match

`classification_metrics` reports `f1_micro` because the revision's summary table
asks for it. For single-label multi-class predictions it is numerically identical
to accuracy — every error is simultaneously one false positive and one false
negative, so micro-averaged precision and recall both collapse to the accuracy.
It is a column because reviewers expect one, not because it adds information; the
docstring says so, rather than leaving a reader to wonder why two columns always
agree.

## `efficiency.py`

The claim a sparse MoE makes is that capacity and compute can be decoupled: the
model *owns* six experts but *evaluates* `K` of them per sample. A plain
parameter count cannot express that, so two numbers are reported side by side:

* `total_parameters` — everything the checkpoint stores.
* `active_parameters` — total minus the `(E − K)` experts that sit out each
  forward pass.

The difference is a **closed form, not an estimate**: all experts share one
architecture, so counting one and multiplying is exact. `top_k_saving()` reports
the Top-2 versus Top-4 comparison directly.

| Measurement | How |
| --- | --- |
| FLOPs | `torch.utils.flop_counter.FlopCounterMode` |
| Peak memory | CUDA allocator high-water mark; MPS current allocation |
| Latency / throughput | Timed loops with warm-up, after device synchronisation |

Three details that are easy to get wrong:

* **FLOPs are counted by dispatch interception**, not a hand-written formula. A
  formula would have to model the router's behaviour; the counter simply observes
  that only `K` experts ran.
* **Timing without `synchronize()`** measures how fast Python enqueues kernels,
  not how fast they run — typically off by an order of magnitude on CUDA and MPS.
* **Warm-up iterations are untimed**, because lazy kernel compilation makes the
  first call dominate a short benchmark.

MPS exposes only *current* allocation, not a peak, so `peak_memory_mb` there is a
snapshot taken after the forward pass. It is noted in `report.notes` rather than
silently reported as a peak. Every measurement degrades to `None` rather than
raising, so profiling never breaks a training run.

## `evaluation.py`

Defines two files per run, written into that run's save path:

* **`test_predictions.npz`** — raw held-out predictions, scores, 384-D
  embeddings, routed expert indices, class names. Keeping the raw arrays rather
  than only the figures means a reviewer asking for a differently-normalised
  confusion matrix costs a second of replotting instead of a full retrain.
* **`summary.json`** — scalar metrics, the efficiency report, the loss history,
  the component flags. `collect_run_summaries()` globs these into the comparison
  table.

`write_summary_csv` leads with exactly the columns the revision requests
(`REQUESTED_COLUMNS`) and appends the rest, so the requested table can be sliced
off the left without editing. Missing measurements are written as **blank** cells
rather than `nan`: a blank reads unambiguously as "not measured", whereas `nan`
in a results table invites a reader to treat it as a failed run.

The headline accuracy/precision/recall/F1 columns describe the **sub-variety**
task — the 27-class problem the architecture exists to solve, and the one where
variants separate. Seed-type numbers follow in the extra columns rather than
being averaged in, which would blend two tasks of very different difficulty into
one meaningless figure.

## `visualization.py`

Forces the `Agg` backend at import (training runs are headless) and returns
`matplotlib` figures rather than writing files, so the caller decides where they
go. `save_figure` persists one.

`plot_confusion_matrix` only annotates cells when the matrix is at most 12 wide —
729 numbers on a 27×27 grid are unreadable. Canvas size scales with the label
count so 27 tick labels stay legible, and the labels themselves are never
abbreviated.

`plot_expert_utilization` draws a dashed reference line at `1/num_experts`: the
entropy load-balancing term is exactly the pressure pulling the bars toward it,
so collapse is visible at a glance.

`plot_tsne(annotate_clusters=True)` prints each class name at its cluster's
**median** position, on top of the colour legend. With 27 classes a legend alone
forces the reader to match 27 similar colours by eye. The median rather than the
mean because t-SNE routinely strands a few points of a class far from its main
mass, and a mean would drag the label into empty space between clusters.

## `training/`

| File | Contents |
| --- | --- |
| `tracker.py` | `ExperimentTracker` — the dual W&B + TensorBoard fan-out |
| `checkpoint.py` | `CheckpointManager` with prefix-based rolling pruning |
| `experiment_logging.py` | Console + file + JSONL structured logging |
| `snapshot.py` | Writes the resolved config, CLI args, and environment per run |
| `device.py` | `cuda` / `mps` / `cpu` selection with `auto`, plus device stats |
| `attention.py` | Extracts and normalises backbone attention maps |

### `ExperimentTracker`

One API, three sinks:

* **`events.jsonl`** — always on, dependency-free, flushed per write. The only
  sink guaranteed to survive a crashed run or an offline machine.
* **TensorBoard** — `tracking.tensorboard.enabled`.
* **W&B** — `tracking.wandb.enabled`, `mode: offline` by default so a run never
  blocks on network or credentials.

A missing optional dependency, or a W&B `init` that fails, degrades to a warning
rather than killing a training run that is otherwise fine.

Methods: `log_metrics`, `log_histogram`, `log_figure`, `log_table`,
`log_images`, `log_embeddings`, `log_gradient_norms`, `log_parameter_histograms`,
`log_gradient_histograms`, `log_artifact`, `log_hyperparameters`, `log_event`.
`log_figure` saves a PNG under `<run_dir>/figures/` in addition to pushing to the
trackers, and closes the figure so long runs do not leak canvases.

`log_metrics` silently drops non-scalar values, which is what lets callers pass a
mixed dict of metrics and metadata without filtering first.

### `CheckpointManager`

Keeps named artifacts ("best", "final") indefinitely and prunes rolling interval
checkpoints by filename prefix to `keep_last_n`. The defaults across this repo
(`keep_last_n_checkpoints: 1`, no optimizer state, no teacher weights) are tuned
for a 16 GB rented disk; a long run with them turned off is how the disk
fills.

### `select_device`

Supports `cuda` / `mps` / `cpu` with `auto` detection, so the code runs on Apple
Silicon. AMP is gated on `device.type == "cuda"`, and `pin_memory` is likewise
forced off unless CUDA.

## `representation.py` — two additions after the stage-1 audit

**`nuisance_decodability`** asks the mirror image of every other measurement in
the module: with class identity held constant, how well can the **source
photograph** be recovered? That is exactly the confound the photograph-disjoint
protocol punishes, and what an in-domain SSL stage should remove. Restricted to
photographs of the same sub-variety, one class at a time, so an encoder that
separates the classes perfectly and the photographs not at all scores exactly
chance. Nearest-class-mean in cosine space over stratified folds — no fitted
hyperparameter, so the number cannot be moved by a regularisation choice, and it
is comparable across encoders of different width.

It matters because it is the one axis on which the shipped stage-1 run
demonstrably worked: ImageNet at +10.0 pp above chance, DINO at +3.5 pp, a 65 %
reduction that no other metric reported. **Read it jointly with the readout** —
an encoder that discards everything scores chance, so low alone is not good.

**`handcrafted_image_features`** returns ten numbers per crop: `log(w·h)`,
`log(w/h)`, mean and std RGB, mean and std grey. They score **0.5360** 27-way
out-of-fold under the identical protocol — 15.6 pp *above* an untrained 48.96 M
trunk and 9.2 pp below the shipped encoder. Computed from the file at its
**native** resolution, deliberately: absolute crop size is part of what they
encode and the resized tensor the encoders see has thrown it away.

## `evaluation.py` — the stage-1 dynamics summary

`PretrainDynamics.summary()` reports `teacher_student_kl_{initial,final,min}` and
`teacher_entropy_share_of_loss_{final,improvement}` alongside the loss figures,
because the reported DINO loss is ~95 % target entropy and only the KL is the
student learning. Runs recorded before the decomposition existed come back as
**NaN** rather than as a silently wrong number.

`loop_blocked_fraction_mean` falls back to `epoch/data_wait_fraction` for those
runs, and `gpu_busy_fraction_mean` is NaN unless the run opted into the
synchronised measurement.
