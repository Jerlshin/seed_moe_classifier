# Architecture

The one-page map. Each section says what a component *is*, what invariant it
carries, and which document holds the detail. The per-topic files under
[`architecture/`](architecture/) are the long form; the per-directory `README.md`
files are the code-level narrative.

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

## The two stages

```text
STAGE 1  (src/trainers/contrastive_pretrain.py)

  ImageNet-1k ─▶ SwinV2-Tiny ─▶ DINO self-distillation ─▶ domain-adapted encoder
                 27.58 M         2 global + 4 local        published once, read
                 768-d, 8x8      trunk UNFROZEN            by every stage-2 run

STAGE 2  (src/trainers/moe_finetune.py)

  encoder (frozen) ─▶ z ∈ ℝ^(64x384) ─▶ SeedType (4) ─▶ MoE Top-2/6
                                     ─▶ residual fusion ─▶ cross-attention
                                     ─▶ ArcFace over 27 sub-varieties
```

The **only** thing crossing the boundary is `student_backbone.state_dict()`. Not
the DINO head, not the teacher, not the optimizer. That is why the projection
head can be resized on stage-1 evidence alone, and why a stage-1 change is safe
exactly as long as the trunk's key set and token grid are unchanged.

## The interface between them

Four properties, all pinned by tests, all silent if broken:

1. **Token grid `8x8 = 64`.** Every `swinv2_*_window16_256` variant emits it at
   256 px, which is what makes the Tiny/Base swap invisible to stage 2's grid
   routing.
2. **`encoder(images).shape[-1] == 384`.** `DinoV2SwinV2Encoder` owns the
   projection from the trunk's native width (768 Tiny, 1024 Base) to the paper's
   `z`, so the head's own input projection is an identity.
3. **Key names.** The published file is a bare trunk state dict with no
   `_orig_mod.` or `module.` prefix — which is why `torch.compile` and DDP
   wrappers are kept *off* the module tree in `DINO._compiled` / `DINO._ddp`.
   Stage 2 loads with `checkpoint_strict: false`, so a prefixed key set would
   match nothing, log one line, and train against a random encoder.
4. **Pooling agreement.** `DINO._pool` and `BackboneFeatureExtractor._pool` mean
   over the spatial grid identically; taking `features[:, 0]` in either would
   select a corner patch.

## Stage 1 at a glance

| | |
| --- | --- |
| Objective | DINO (Caron et al., 2021) + KoLeo + Sinkhorn-Knopp. **Not** DINOv2 — no iBOT, no untied heads |
| Trunk | SwinV2-Tiny, ImageNet-1k init, unfrozen, `drop_path 0.1` (student only) |
| Head | 768 → 1024 → 1024 → 256 → L2 → weight-norm 2048, discarded after stage 1 |
| Batch | 32 physical x 1 accumulation — physical batch is what the collapse guards estimate from |
| Schedule | 100 epochs, 10 linear warmup then cosine; LR derived as `0.0005 x B_eff/256` |
| Decay | 0.04 → 0.4 cosine on weight matrices only |
| Artifacts | final encoder + milestone encoders at 25/50/100, a budget report, entropy diagnostics with their own bounds |

Controls that make the stage-1 cost defensible:

```bash
python -m src.trainers.moe_finetune experiment=control_imagenet_frozen        # A: no stage 1
python main.py pretrain && python main.py finetune                            # B: with stage 1
python -m src.trainers.contrastive_pretrain experiment=pretrain_swinv2_base_dino  # capacity control
```

## Stage 2 at a glance

| | |
| --- | --- |
| Routing | Top-2 of 6 experts, over the `8x8` grid (`batch x 64 x K` slots per step) |
| Fusion | `h' = h + LayerScale(P(p_s))` — softmax probabilities, not logits |
| Attention | `Q = h'`, `K = V = h`; not allocated at all in `token_mode=pooled`, where it would be affine |
| Head | ArcFace with the AdaCos scale `sqrt(2) log(C-1) = 4.61`, margin ramped over the first 15 % |
| Loss | Seven weighted terms; `CombinedHierarchicalLoss` reads a dataclass, never a tuple |
| Split | Grouped by source photograph — 9,357 crops come from 81 photographs |

## Where the invariants live

| Invariant | Enforced by |
| --- | --- |
| SwinV2 only, on both paths | `validate_swinv2_name` |
| Stage 1 never freezes the trunk | `build_dino` raises; the trainer re-counts trainable parameters |
| The teacher has no stochastic depth | `disable_drop_path`, called on the deepcopy |
| Effective batch is world-size independent | `resolve_accumulation` refuses a non-dividing combination |
| The LR follows the batch | `resolve_learning_rate`, logged with its provenance |
| Weight decay skips vectors and declared exclusions | `build_param_groups` + `apply_weight_decay` |
| Every run leaves a machine-readable trace | `component_flags`, `loss_flags`, `summary.json`, `events.jsonl`, `budget/*` |

`tests/` (458 tests, no network) pins all of the above. `tests/conftest.py` holds
the paper constants in **pairs** — submitted and revised — so a test asserting a
bare number cannot silently become a claim about whichever the reader assumed.
