# 00 — System Overview

> Audit scope: this suite documents the implementation as it exists in `src/`,
> `conf/`, and `scripts/` at the time of writing — exact module names, tensor
> shapes, formulas, and configuration values, cross-referenced to file:line.
> It complements, and does not replace, the two documents the repository asks
> you to read before touching `src/models/` or `src/losses/`:
> [`PAPER_AUDIT.md`](../PAPER_AUDIT.md) (where the code diverged from the
> submitted paper) and [`REVISION_NOTES.md`](../REVISION_NOTES.md) (where the
> peer-review revision deliberately diverges from the submitted paper).

## 1. What this system is

Reference implementation of *"Hierarchical Deep Learning for Fine-Grained Seed
Classification: A Self-Supervised and Mixture-of-Experts Approach"*
(`../paper/sn-article.pdf`). It classifies seed images at two levels of
granularity — **4 seed types** and **27 sub-varieties**, both derived from a
directory tree — using:

1. A **DINOv2 self-supervised pretraining stage** over a Swin Transformer V2
   (SwinV2) backbone (paper Section 4).
2. A **hierarchical Mixture-of-Experts (MoE) head** that cascades a coarse
   seed-type classifier into a sparse Top-K expert router, a residual fusion
   of the two, a cross-attention refinement, and an ArcFace angular-margin
   classifier for the fine sub-variety label (paper Section 5).

The two stages are connected by exactly one artifact: a published encoder
checkpoint. Every downstream run — the full model, six ablations, and three
baselines — loads that same file, so architectural comparisons are never
confounded by a different self-supervised initialization.

### Revision status

This tree implements the **peer-review revision**. It departs from the submitted
manuscript in several deliberate, reversible ways, and each departure exists
because the submitted form was measurably doing something other than what it
claimed. [`AUDIT_RESPONSE.md`](../AUDIT_RESPONSE.md) records the full disposition
of the independent audit in `CHANGES.md`; [`REVISION_NOTES.md`](../REVISION_NOTES.md)
records the rationale.

| Departure | Submitted | This tree | Reproduce submitted |
| --- | --- | --- | --- |
| MoE routing width | Top-4 of 6 | **Top-2** of 6 | `model.head.top_k=4` |
| Routing granularity | pooled vector | **token grid** (`8x8`) | `model.head.token_mode=pooled` |
| Load-balancing loss | soft-gate entropy | **Switch `E.sum f.P`** | `model.loss.moe_load_mode=entropy` |
| Hierarchy KL | probability space + `clamp` | **log space (`logsumexp`)** | not reversible — the old form was gradient-dead |
| KL gradient | flows into both heads | **coarse head detached** | `model.loss.detach_kl_seed_target=false` |
| ArcFace scale | `s = 30` | **AdaCos `s = 4.61`** | `model.head.arcface_scale=30.0` |
| Compactness term | residual cosine | **intra-class, EMA centroids** | `model.loss.cosine_mode=residual` |
| Split protocol | crop level | **source-photograph groups** | `experiment.training.split_protocol=stratified` |
| Reporting | 1 seed, best fold | **5 seeds, mean ± SD, McNemar** | `--seeds 42` |
| Stage-1 method name | "DINOv2" | **"DINO-style self-distillation"** | not reversible — it was never DINOv2 |
| Backbone | SwinV2 + comparative ViT-S/14 | **SwinV2 only** | not reversible — ViT-S/14 path removed |

`src/models/components/moe_layer.py` defines `DEFAULT_NUM_EXPERTS = 6` and
`DEFAULT_TOP_K = 2` as the single source of truth for the routing values.

Most of these are exposed as ablation variants precisely so the comparison can be
made rather than asserted — see [`06_ABLATION_ENGINE.md`](06_ABLATION_ENGINE.md)
§3.

## 2. End-to-end data flow

```text
                                    STAGE 1 — self-supervised pretraining
                                    (src/trainers/contrastive_pretrain.py)

image ──DataAugmentationDINO──▶ 2 global + 4 local crops
                                       │
                          ┌────────────┴─────────────┐
                          ▼                           ▼
                 teacher (2 globals only)     student (all 6 views)
                 SwinV2 ──▶ DINOHead                SwinV2 ──▶ DINOHead
                          │                           │
                          └──────── CustomDINOLoss ───┘   (Eqs. 1-3)
                                       │
                          EMA update, momentum 0.996
                                       │
                     dino_pretrained_backbone.pth
                                       │
                     published to outputs/checkpoints/
                     dinov2_swinv2_pretrained.pth
                                       │
════════════════════════════════════════════════════════════════════════════
                                    STAGE 2 — hierarchical finetuning
                                    (src/trainers/moe_finetune.py)

image ──SwinV2 (frozen)──▶ 8×8 token grid ──EmbeddingProjection──▶ z ∈ ℝ^(64×384)   (Eq. 4)
                                                              │
                              ┌───────────────────────────────┴───────────────┐
                              ▼                                              ▼
               SeedTypeClassifier (Eq. 5)                        MixtureOfExperts (Eq. 8)
                    s ∈ ℝ⁴, hidden ∈ ℝ¹⁹²                       h = Σ_{i∈Top-2} Gᵢ Eᵢ(z)
                              │                                  routed PER TOKEN
                    p_s = softmax(s)  (Eq. 6)                     gate sees [z, p_s.detach()]
                              │                                              │
             SeedTypeProjection P(p_s) · γ  ────────────────(+)──────────────┘
             or FiLM(h ; hidden)                              │
                                                    h' = h + γ⊙P(p_s)   (Eq. 9)
                                                              │
                                     CrossAttention(Q=h', K=V=h)     (Eq. 11)
                                        over 64 keys — a real map    (Eq. 12)
                                                              │
                                          TokenPooling  ──▶ ℝ³⁸⁴
                                                              │
                                            SubVarietyEmbedding
                                                              │
                                    ArcFaceHead(e, labels) ──▶ (sub_logits, sub_margin_logits)  (Eq. 13)
```

**Pooling moved to after the head**, and that is the revision's largest
architectural change. Over a length-1 sequence `softmax(QKᵀ/√d)` is identically
1, so the submitted pooled path made Eqs. 11-12 an affine map with ~2.07 M
parameters that no configuration could reach — while counting them as active.
Keeping the token grid makes the attention real, raises routing slots per step
from `batch × K` to `batch × 64 × K`, and preserves the localised texture a
27-way fine-grained task depends on. See
[`03_MOE_MODULE.md`](03_MOE_MODULE.md) §1.

Both stages share the SwinV2 trunk architecture but train independently:
stage 1 discovers the representation, stage 2 (with the backbone frozen by
default) trains only the head plus the Eq. 4 projection. Full detail in
[`02_BACKBONE_AND_SSL.md`](02_BACKBONE_AND_SSL.md) and
[`04_HIERARCHICAL_FUSION.md`](04_HIERARCHICAL_FUSION.md).

## 3. The central data structure: `HierarchicalOutput`

`src/models/builder.py:107-157` defines a frozen dataclass carrying **every**
intermediate tensor the losses, metrics, and figures need:

```python
seed_type_logits    # s,  Eq. 5   [batch, num_seed_types]
seed_type_probs     # p_s, Eq. 6  [batch, num_seed_types]
embedding           # z,  Eq. 4   [batch, embed_dim]
moe_features        # h,  Eq. 8   [batch, embed_dim]
projected_seed      # P(p_s), Eq. 9 (zeros when use_residual=False)
refined_features    # h', Eq. 9
attended_features   # h'', Eq. 12
sub_embeddings      # ArcFace input
sub_logits          # no margin — used for prediction, ranking, KL
sub_margin_logits   # margin on target class — used only by ArcFace CE
gate_probs          # G, full MoE distribution  [batch, num_experts]
top_k_indices       # [batch, top_k]
top_k_weights       # [batch, top_k]
dispatch_weights    # top_k_weights scattered over all experts
attn_weights        # cross-attention map, or None
```

This is a deliberate fix to a historical bug (`PAPER_AUDIT.md` §1): components
previously returned bare tuples, and adding a field silently broke positional
unpacking at call sites. **Every model in this repository — the full head,
all six ablations, and the two end-to-end baselines — emits this exact
dataclass**, which is what lets `CombinedHierarchicalLoss`, the metrics stack,
and the figure generators run unmodified against any of them.

## 4. Component index

| Concern | Primary module(s) | Doc |
| --- | --- | --- |
| Dataset, label hierarchy, splits, augmentation | `src/datasets/dataset.py`, `src/datasets/transforms.py` | [`01_DATA_PIPELINE.md`](01_DATA_PIPELINE.md) |
| SwinV2 backbone, DINOv2 self-distillation | `src/models/backbones/swinv2_dino.py`, `src/models/builder.py` (`BackboneFeatureExtractor`, `DinoV2SwinV2Encoder`), `src/losses/dino.py` | [`02_BACKBONE_AND_SSL.md`](02_BACKBONE_AND_SSL.md) |
| Sparse Top-K Mixture-of-Experts | `src/models/components/moe_layer.py`, `src/losses/moe.py` | [`03_MOE_MODULE.md`](03_MOE_MODULE.md) |
| Seed-type classifier, residual fusion, cross-attention, ArcFace | `src/models/components/{classifiers,projections,cross_attention,arcface_head}.py`, `src/models/builder.py` (`HierarchicalSeedClassifier`) | [`04_HIERARCHICAL_FUSION.md`](04_HIERARCHICAL_FUSION.md) |
| Combined objective | `src/losses/hierarchical.py`, `src/losses/arcface.py`, `src/losses/cosine.py` | [`05_LOSS_FUNCTIONS.md`](05_LOSS_FUNCTIONS.md) |
| Component toggles, ablation/baseline suites | `conf/model/head/hierarchical_moe.yaml`, `src/trainers/runner.py`, `scripts/run_ablations.py`, `scripts/run_baselines.py` | [`06_ABLATION_ENGINE.md`](06_ABLATION_ENGINE.md) |
| Parameter/FLOP/latency accounting, metrics, reporting, figures | `src/utils/{efficiency,metrics,evaluation,visualization}.py` | [`07_EFFICIENCY_AND_EVALUATION.md`](07_EFFICIENCY_AND_EVALUATION.md) |

## 5. Repository layout

```text
conf/                Hydra config groups (data, model/{backbone,head,loss}, experiment, tracking)
src/
  datasets/           HierarchicalSeedDataset, DataAugmentationDINO
  models/
    backbones/        DINO student/teacher, DINOHead
    components/       MoE layer, projections, classifiers, cross-attention, ArcFace head
    builder.py         DinoV2SwinV2Encoder, HierarchicalSeedClassifier, HierarchicalOutput
    baselines.py       FlatSupervisedBaseline, IdentityEncoder
  losses/             CustomDINOLoss, CombinedHierarchicalLoss, MoERegularization, CosineSimilarityLoss
  trainers/           contrastive_pretrain.py, moe_finetune.py, runner.py
  utils/              metrics.py, efficiency.py, evaluation.py, visualization.py, training/
scripts/              run_ablations.py, run_baselines.py, generate_plots.py, extract_features.py, dry_run.py
tests/                280 pytest tests pinning paper constants and dataflow invariants
```

Per-directory READMEs (`src/README.md`, `src/models/README.md`, etc.) carry
additional narrative detail and are the first thing to read when touching a
package this documentation suite doesn't cover in full.

## 6. Config composition

`conf/config.yaml` composes six Hydra groups: `data`, `model/backbone`,
`model/head`, `model/loss`, `experiment`, `tracking`. Every group is genuinely
wired — overriding one on the command line changes the run. Experiment files
under `conf/experiment/` are `# @package _global_` and select the head/loss
their stage needs via `defaults: - override /model/head: ...`. Every ablation
and baseline experiment inherits from `finetune_hierarchical_moe.yaml`, so a
change there propagates to all of them — which is the point: variants must
differ only in the one thing under test.

```bash
python main.py pretrain      # == python -m src.trainers.contrastive_pretrain experiment=pretrain_swinv2_dino
python main.py finetune      # == python -m src.trainers.moe_finetune experiment=finetune_hierarchical_moe
python main.py ablation      # flat-classifier ablation (removes coarse-stage influence)
python main.py smoke         # 2-batch dry run of both stages
python scripts/dry_run.py    # synthetic end-to-end pipeline check, no dataset needed
python scripts/run_ablations.py   # six component-wise variants
python scripts/run_baselines.py   # ResNet-50, Swin-T, hierarchical CCE
python scripts/generate_plots.py  # figures + outputs/reports/summary_metrics.csv
```

## 7. Cross-cutting invariants worth internalizing before editing code

These are the facts that, if violated, produce a model that *runs* and *trains*
but silently measures the wrong thing. Each is pinned by at least one test.

1. **`encoder(images).shape[-1] == 384` always holds.** `DinoV2SwinV2Encoder`
   owns the projection from the backbone's native width (768 for Tiny/Small,
   1024 for Base) to `embed_dim = 384`. Under `token_mode="grid"` the output is
   `[B, 64, 384]`, so the invariant is on the **last axis**. This projection is
   trainable even when the trunk is frozen, so it **must** be in the optimizer.
2. **The encoder's `token_mode` must match the head's.** A grid encoder feeding a
   pooled head (or the reverse) is a silent misconfiguration that would quietly
   re-degenerate every attention module. `build_model_and_encoder` passes one
   value to both.
3. **The experts consume `z`, not the seed-type projection** (Eq. 8). The *gate*
   may additionally see the **detached** `p_s` — that is what makes the MoE
   hierarchical — but the experts always see the image.
4. **The residual adds `P(p_s)`, softmax probabilities, not `P(s)`, logits**
   (Eq. 9), scaled by a LayerScale gain.
5. **`sub_logits` (no margin) drive prediction and the KL term;
   `sub_margin_logits` drive only the ArcFace cross-entropy.** Evaluation passes
   no labels, so the two coincide and metrics are never inflated by the margin.
   The same holds during the margin warm-up.
6. **A disabled component toggle is never allocated** — the attribute is `None`,
   not `nn.Identity` — and neither is a block no configuration can reach. The
   affine token mixer and the unallocated noise gate are the two cases of the
   latter, and both exist because unreachable parameters were being reported as
   *active*.
7. **`use_moe=False` substitutes one dense block**, not nothing, so the `wo_moe`
   gap measures routing rather than a missing layer's capacity — and
   `dense_capacity_multiplier` matches it to Top-K's *active* capacity, because
   the naive substitution still confounded routing with a 2× FLOP cut.
8. **`materialize_expert_grads()` runs between `backward()` and `step()`.**
   Without it, unrouted experts have `grad is None`, AdamW skips them entirely
   (including weight decay), and sparse and dense dispatch train measurably
   different models despite identical forward passes.
9. **Every self-supervised-path run reads one shared encoder checkpoint.**
   `ensure_pretrained_checkpoint()` refuses to start a suite otherwise.
10. **One trainer serves the full model, every ablation, the linear probe and the
    baselines**, dispatching on `model.head.name`.
11. **Splits are group-aware by default and byte-identical across variants.**
    The first is what makes the accuracy mean something; the second is what makes
    McNemar's paired test valid.
12. **Every run leaves a complete machine-readable trace** — `component_flags`,
    `loss_flags`, `split`, `fold_metrics`. Two runs that differ in any axis must
    differ in `summary.json`, not only in their variant name.
