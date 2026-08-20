# Architecture

The one-page map. Each section says what a component *is*, what invariant it
carries, and which document holds the detail. The per-topic files under
[`architecture/`](architecture/) are the long form; the per-directory `README.md`
files are the code-level narrative; [`README.md`](README.md) is how to *run* it.

| Topic | Detail |
| --- | --- |
| Overview, revision record, cross-cutting invariants | [`architecture/00_OVERVIEW.md`](architecture/00_OVERVIEW.md) |
| Dataset, splits, multi-crop augmentation | [`architecture/01_DATA_PIPELINE.md`](architecture/01_DATA_PIPELINE.md) |
| **Backbone and stage-1 self-supervision** | [`architecture/02_BACKBONE_AND_SSL.md`](architecture/02_BACKBONE_AND_SSL.md) |
| Mixture-of-Experts routing | [`architecture/03_MOE_MODULE.md`](architecture/03_MOE_MODULE.md) |
| Hierarchical fusion and cross-attention | [`architecture/04_HIERARCHICAL_FUSION.md`](architecture/04_HIERARCHICAL_FUSION.md) |
| Loss functions | [`architecture/05_LOSS_FUNCTIONS.md`](architecture/05_LOSS_FUNCTIONS.md) |
| Ablation and baseline engine | [`architecture/06_ABLATION_ENGINE.md`](architecture/06_ABLATION_ENGINE.md) |
| Efficiency and evaluation | [`architecture/07_EFFICIENCY_AND_EVALUATION.md`](architecture/07_EFFICIENCY_AND_EVALUATION.md) |
| **Stage-1 representation evaluation** | [`architecture/08_STAGE1_REPRESENTATION_EVALUATION.md`](architecture/08_STAGE1_REPRESENTATION_EVALUATION.md) |

## One pipeline

There is one data group, one trunk, one stage-1 experiment, one stage-2
experiment, one split protocol. Everything else under `conf/experiment/` is a
named **control** (a comparison the results table needs) or an **ablation** (one
override off the primary), and each file's header says which.

```text
STAGE 1  (src/trainers/contrastive_pretrain.py)   experiment=pretrain_dino

  ImageNet-1k ─▶ SwinV2-Tiny ─▶ DINO self-distillation ─▶ domain-adapted encoder
                 27.58 M         2 global @256 +           the probe picks which
                 768-d, 8x8      4 local @160              epoch is published
                                 trunk UNFROZEN

STAGE 1.5  (src/trainers/pretrain_eval.py)        experiment=eval_pretrain
                                                  experiment=eval_frozen_reference
                                                  experiment=screen_backbones
  evaluation only, no training

  the encoder above ─▶ frozen features ─▶ probe / k-NN / geometry / clustering
  + its numbered milestones, its ImageNet initialisation, an untrained trunk and
  a ten-scalar handcrafted floor, so every number sits between a floor and a
  baseline

STAGE 2  (src/trainers/moe_finetune.py)           experiment=finetune_hierarchical_moe

  encoder (frozen) ─▶ z ∈ ℝ^(64x384) ─▶ SeedType (4) ─▶ MoE Top-2/6
                                     ─▶ residual fusion ─▶ cross-attention
                                     ─▶ ArcFace over 27 sub-varieties
```

The **only** thing crossing the stage boundary is
`student_backbone.state_dict()`. Not the DINO head, not the teacher, not the
optimizer. That is why the projection head can be resized on stage-1 evidence
alone, and why a stage-1 change is safe exactly as long as the trunk's key set and
token grid are unchanged.

## The interface between the stages

Five properties, all pinned by tests, all silent if broken:

1. **One trunk name.** `conf/model/backbone/swinv2.yaml` selects
   `swinv2_tiny_window16_256` and **both stages read it**. A stage-2 config that
   named a different depth would load the stage-1 encoder under
   `checkpoint_strict: false` and leave the unmatched tensors at their random
   initialisation — 204 of 423 for a Tiny encoder in a Small trunk — behind one
   log line. The two capacity/corpus controls override the name and the width
   *together*, and publish to their own paths.
2. **Token grid `8x8 = 64`.** Every `swinv2_*_window16_256` variant emits it at
   256 px, which is what makes a Tiny/Base swap invisible to stage 2's grid
   routing. Only the channel width changes.
3. **`encoder(images).shape[-1] == 384`.** `DinoV2SwinV2Encoder` owns the
   projection from the trunk's native width (768 Tiny/Small, 1024 Base) to the
   paper's `z`, so the head's own input projection is an identity. *The projection
   is trainable even when the backbone is frozen* — `build_optimizer()` must
   include the encoder, or nothing reports an error and the head trains against a
   random projection.
4. **Key names.** The published file is a bare trunk state dict with no
   `_orig_mod.` or `module.` prefix — which is why `torch.compile` and DDP
   wrappers are kept *off* the module tree, in `DINO._compiled` / `DINO._ddp`.
   A prefixed key set would match nothing, log one line, and train against a
   random encoder.
5. **Pooling agreement.** `DINO._pool` and `BackboneFeatureExtractor._pool` mean
   over the spatial grid identically; taking `features[:, 0]` in either would
   select a corner patch.

And one property of the *corpus* rather than the weights: `corpus_fingerprint()`
is written into stage 1's `summary.json` and read back by the evaluation, so
"which crops was this encoder trained on" is answerable from the artifacts.
`data.expected_num_samples` makes a mismatch fatal at startup instead.

## Stage 1 at a glance

| | |
| --- | --- |
| Objective | DINO (Caron et al., 2021) + KoLeo + Sinkhorn-Knopp centering. **Not** DINOv2 — no iBOT, no untied heads |
| Trunk | SwinV2-Tiny (27.58 M), ImageNet-1k init, unfrozen, `drop_path 0.1` (student only) |
| Views | 2 global @256 scale `(0.70,1.00)` + 4 local @160 scale `(0.30,0.70)`, aspect `(0.5,2.0)`, full dihedral group |
| Head | 768 → 1024 → 1024 → 256 → L2 → weight-norm 2048, discarded after stage 1 |
| Batch | 64 physical × 1 accumulation — physical batch is what the collapse guards estimate from |
| Schedule | 50 epochs, 5 linear warmup then cosine; LR derived as `0.0005 × B_eff/256` |
| Decay | 0.04 → 0.4 cosine on weight matrices only |
| Selection | an in-training representation probe, not the loss; `publish: best` |
| Artifacts | the probe-selected encoder + every milestone, a budget report, entropy diagnostics with their own bounds, five figures, four CSV families |

The three decisions that are specific to *this* corpus rather than inherited from
DINO, each with a measurement behind it:

- **View geometry.** `scale` is a fraction of the source area and the source is
  one seed at a median 52 × 51 px, so DINO's `(0.05, 0.40)` builds a local view
  from a median **598 native pixels** — 0.91 % real content — and 8 of the 10
  cross-view terms in Eq. 1 are anchored on one. `(0.30, 0.70)` at 160 px takes
  that to 1,419.
- **Colour is signal.** Mean RGB alone scores 0.3169 on the 27-way task; only
  ~26.6 % of the within-class colour variance is photograph-specific. Brightness
  and contrast (illumination = nuisance) stay at 0.4; saturation, hue, grayscale
  and solarization are cut.
- **The probe picks the epoch.** The loss is a cross entropy against a moving
  teacher — 94.8 % irreducible target entropy on a measured 100-epoch run — while
  the representation peaked at epoch 50 and the pipeline published epoch 100.

Controls that make the stage-1 cost defensible:

```bash
python main.py eval-frozen                                                   # the bar, no training
python -m src.trainers.moe_finetune experiment=control_imagenet_frozen        # A: no stage 1
python main.py pretrain && python main.py finetune                            # B: with stage 1
python -m src.trainers.contrastive_pretrain experiment=pretrain_dino_base     # capacity control
```

## Stage 2 at a glance

| | |
| --- | --- |
| Routing | Top-2 of 6 experts, over the `8x8` grid (`batch x 64 x K` slots per step) |
| Fusion | `h' = h + LayerScale(P(p_s))` — softmax probabilities, not logits |
| Attention | `Q = h'`, `K = V = h`; not allocated at all in `token_mode=pooled`, where it would be affine |
| Head | ArcFace with the AdaCos scale `sqrt(2) log(C-1) = 4.61`, margin ramped over the first 15 % |
| Loss | Seven weighted terms; `CombinedHierarchicalLoss` reads a dataclass, never a tuple |
| Split | **Crop-level stratified** (primary), with `grouped_cv` as the photograph-disjoint diagnostic |

The split is the one place where the honest number and the reported number differ
by a measured amount: 9,357 crops come from 81 photographs, and under an
identical frozen encoder the crop-level 27-way probe sits **+18.65 pp** above the
photograph-disjoint one. Every run reports `shared_source_groups`,
`leaked_test_fraction` and `classes_present_in_test`, and
`experiment=finetune_grouped_diagnostic` measures the gap on the encoder being
reported rather than quoting it from another one.

## Where the invariants live

| Invariant | Enforced by |
| --- | --- |
| SwinV2 only, on both paths | `validate_swinv2_name` |
| One trunk for both stages | `conf/model/backbone/swinv2.yaml`; `DINO.__init__` cross-checks `feature_dim` against the trunk's `num_features` |
| Stage 1 never freezes the trunk | `build_dino` raises; the trainer re-counts trainable parameters |
| The teacher has no stochastic depth | `disable_drop_path`, called on the deepcopy |
| The corpus is the declared one | `corpus_fingerprint`, `data.expected_num_samples`, `corpus_check` |
| Effective batch is world-size independent | `resolve_accumulation` refuses a non-dividing combination |
| The LR follows the batch | `resolve_learning_rate`, logged with its provenance |
| Weight decay skips vectors and declared exclusions | `build_param_groups` + `apply_weight_decay` |
| The published epoch is the best-probing one | `RepresentationProbe` + `CheckpointSelector`, with the stop decision broadcast from rank 0 |
| Sparse and dense dispatch train the same model | `materialize_expert_grads()` between `backward()` and `step()` |
| Every run leaves a machine-readable trace | `component_flags`, `loss_flags`, `summary.json`, `events.jsonl`, `csv/*`, `budget/*` |

`tests/` (640 tests, no network) pins all of the above. `tests/conftest.py` holds
the paper constants in **pairs** — submitted and revised/canonical — so a test
asserting a bare number cannot silently become a claim about whichever the reader
assumed.
