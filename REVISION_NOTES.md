# Revision notes

What changed in this tree relative to the **submitted manuscript**
(`../paper/sn-article.pdf`), and how to reproduce either version.

[`PAPER_AUDIT.md`](PAPER_AUDIT.md) is a different document: it records where the
*original code* diverged from the *paper as written*. This file records where
the *revision* deliberately departs from the submitted paper.

---

## 1. Routing width: Top-4 → Top-2

**Paper, Section 5.2:** six experts, "only the top-4 most relevant experts" are
activated per sample.

**This tree:** `K = 2` of `E = 6`.

Total capacity is unchanged — the checkpoint still stores six experts — but each
forward pass evaluates two of them, so the *active* parameter count drops. The
saving is exact rather than estimated, because every expert shares one
architecture:

```
dormant = (E − K) × parameters_per_expert
active  = total − dormant
```

Halving `K` from 4 to 2 therefore doubles the dormant count. `top_k_saving()` in
`src/utils/efficiency.py` reports both figures side by side, and
`tests/test_efficiency.py::test_top_2_activates_fewer_parameters_than_top_4`
asserts the identity.

**One consequence worth stating explicitly.** With sparse dispatch, an expert
that no sample in a batch routed to receives *no gradient that step*. This is
the defining property of a sparse MoE, not a defect — but it is much more
visible at `K = 2` than at `K = 4`. A batch of 12 fills only 24 routing slots
across six experts, so an expert sitting out a whole batch is routine. The
entropy load-balancing term keeps pulling utilisation back toward uniform over
an epoch, so no expert stays unrouted for long.
`tests/test_models.py::test_only_the_routed_experts_receive_gradient` pins the
precise invariant: routed ⇔ has gradient.

**To reproduce the submitted configuration:**

```bash
python main.py finetune model.head.top_k=4
python scripts/run_ablations.py -- model.head.top_k=4
```

---

## 2. Backbone: SwinV2 only

**Paper, Section 6.1:** reports a comparative DINOv2 ViT-S/14 baseline alongside
the SwinV2 encoder.

**This tree:** SwinV2 is the only encoder on the DINOv2 path. The ViT-S/14
config, the `torch_hub` model source, and every documentation reference to it
have been removed. `validate_swinv2_name()` in `src/models/builder.py` rejects
any backbone name that does not start with `swinv2`, and is called from both
`BackboneFeatureExtractor` and the DINO student/teacher wrapper — so a stale
name fails in the first second rather than after an epoch of self-distillation
against the wrong encoder.

`tests/test_configs.py::test_only_swinv2_is_available_as_a_dinov2_backbone`
asserts that `conf/model/backbone/` contains exactly one entry.

Supervised comparison backbones (ResNet-50, Swin-T) still exist, but as
*baselines* in `src/models/baselines.py` — a separate class reached through
`conf/experiment/baseline_*.yaml`, never through the DINOv2 backbone group. The
separation is the point: those models are not DINOv2-pretrained, so putting them
in the same config group would invite exactly the mix-up this change removes.

---

## 3. `z ∈ ℝ³⁸⁴` is now an encoder invariant

**Paper, Eq. 4:** "the DINOv2 encoder extracts a 384-dimensional feature vector".

No SwinV2 variant emits 384 channels — Base emits 1024, Tiny and Small emit 768
— so a learned projection is required for Eq. 4 to hold at all. Previously that
projection lived inside the hierarchical head, which meant "the encoder output
is 384-D" was true only of the head's first layer.

`DinoV2SwinV2Encoder` now owns it, so `encoder(images).shape[-1] == 384` holds
unconditionally and every consumer — the head, the t-SNE feature dump, the
efficiency profiler, the baselines — observes the same 384-D space. The head's
own input projection collapses to `nn.Identity` when its input is already `z`.

Two practical consequences:

* The projection is **trainable even when the backbone is frozen**, so it must
  be in the optimizer. `build_optimizer()` takes a list of modules for exactly
  this reason; omitting the encoder would silently freeze the one layer adapting
  1024 backbone channels to the head's 384, and the head would train against a
  random projection.
* Stage-2 checkpoints now store `encoder_state_dict` alongside
  `model_state_dict`. A head-only checkpoint would be unusable.

---

## 4. Component toggles

Five booleans, all under `model.head`, each removing one architectural
ingredient:

| Flag | Effect when `false` |
| --- | --- |
| `use_moe` | Top-2 router replaced by one dense transformer block |
| `use_arcface` | ArcFace replaced by a linear head; objective becomes plain CE |
| `use_residual` | `h' = h`; the Eq. 9 fusion is not built |
| `use_cross_attention` | `h'' = h'`; Eqs. 11–12 skipped |
| `use_kl_loss` | Eq. 10 not computed (read by `model/loss` via interpolation) |

Three design decisions inside this that are easy to get wrong:

**Disabled blocks are not allocated.** `seed_projection` and `cross_attention`
become `None` rather than `nn.Identity`, so an ablation's parameter count
describes the model actually trained rather than the full model with dead
weights.

**`wo_moe` keeps one dense block.** Deleting the experts outright would also
delete a transformer block's worth of capacity, and the resulting gap would
confound *routing* with *depth*. `DenseExpertBlock` is architecturally identical
to a single expert and always on, so the only removed ingredient is the gate. It
returns a `MoEOutput` describing a one-expert router, on which both MoE
regularisers evaluate to exactly zero — no downstream caller needs a special
case.

**`wo_arcface` needs no loss-side branch.** `LinearSubVarietyHead` returns its
logits unchanged as `sub_margin_logits`. The ArcFace term is a cross-entropy
over `sub_margin_logits`, so with no margin present it *is* the categorical
cross-entropy the ablation calls for. Sharing one code path means the ablation
cannot drift from the full model by way of a second loss implementation.

---

## 5. Shared-checkpoint discipline

Every DINOv2-path run — the full model, all six ablations, and the
`hierarchical_cce` baseline — reads
`outputs/checkpoints/dinov2_swinv2_pretrained.pth`, published once by the
pretrain stage.

`ensure_pretrained_checkpoint()` refuses to start a suite when that file is
absent, and the error message says how to produce it. The alternative — training
some variants from a random encoder — would yield a comparison table that looks
completely normal and means nothing.
`tests/test_configs.py::test_all_variants_share_one_pretrained_encoder` asserts
the configs resolve to a single path.

The two end-to-end supervised baselines are deliberately *not* given this
checkpoint: they own ImageNet backbones of a different architecture, where a
SwinV2 DINO state dict would at best be ignored and at worst partially loaded.

---

## 6. New measurements

**Efficiency** (`src/utils/efficiency.py`, wired into every run): total vs.
active parameters, FLOPs, per-sample latency, throughput, peak memory. Profiles
encoder + head together, because that is the only combination whose latency a
deployment would observe.

**Overfitting diagnostics**: `LossHistory` records train and validation loss per
epoch; `epoch/overfitting_gap` (validation − train) is logged as a scalar and the
two curves are drawn side by side. On a 27-class problem with a few hundred
samples per class, a validation loss turning upward while training loss keeps
falling is the failure mode most worth watching for.

**Micro F1** is now reported. For single-label multi-class predictions it is
numerically identical to accuracy — every error is simultaneously one false
positive and one false negative — so it carries no information beyond
`accuracy`. It is a column because reviewers expect one, and
`classification_metrics()` says so in its docstring rather than leaving a reader
to wonder why two columns always match.

---

## 7. Reporting contract

Each run writes two files into its save path:

* `test_predictions.npz` — raw held-out predictions, scores, 384-D embeddings,
  routed expert indices, class names.
* `summary.json` — scalar metrics, the efficiency report, loss history,
  component flags.

`scripts/generate_plots.py` reads only those and produces
`outputs/reports/summary_metrics.csv` plus publication figures at 300 DPI. It
**re-scores** from the raw predictions rather than trusting `summary.json`, so
the table and the figures are computed by the same code at plot time.

`summary_metrics.csv` leads with exactly the requested columns —
`Model/Variant, Accuracy, Precision, Recall, Macro F1, Micro F1, KL Alignment
Rate (%), Total Params (M), Active Params (M), Inference Latency (ms)` — with
seed-type metrics, AUC, throughput, FLOPs and peak memory appended to the right,
so the requested table can be sliced off the left without editing.

The headline accuracy/precision/recall/F1 columns describe the **sub-variety**
task: it is the 27-class problem the architecture exists to solve, and the one
where the variants separate. Blending it with the much easier 4-class task would
produce a figure that means nothing.

---

## 8. Verification performed

| Check | Result |
| --- | --- |
| `python -m pytest tests/ -q` | 280 passed |
| `python scripts/dry_run.py` | full pipeline, real SwinV2 encoder, synthetic data, zero errors |
| `scripts/run_ablations.py` on the real 9,357-image dataset | 6/6 variants completed |
| `scripts/run_baselines.py` on the real dataset | 3/3 baselines completed |
| `scripts/generate_plots.py` | 9 runs collected, 72 figures + `summary_metrics.csv` |
| Confusion matrix, 27 classes | all sub-varieties labelled unabbreviated on both axes |
| t-SNE | real class names, colour legend, cluster-centroid overlays |

The suite runs used a reduced configuration (SwinV2-Tiny, 1 epoch, 3 batches,
random encoder) to exercise the plumbing quickly. Accuracies from those runs are
near chance by construction and are not results.
