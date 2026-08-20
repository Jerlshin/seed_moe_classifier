# `notebooks/` — exploratory work

Scratch space. Nothing here is imported by `src/`, exercised by `tests/`, or
required by either training stage.

| Path | Contents |
| --- | --- |
| `abalation_studies/tasks.md` | Original protocol sketch for the supervised-baseline comparison (historical) |
| `abalation_studies/resnet_comparison.ipynb` | Empty stub predating that implementation (historical) |

## Relationship to the paper's Limitations section

The paper states (Section 8) that *"comparisons with standard supervised
baselines such as ResNet and Vision Transformer were not conducted in the present
work"*. `tasks.md` sketches the protocol for closing that gap: a shared backbone
(ResNet-50 or ViT-B/16) with a 4-class coarse head and a 27-class fine head,
trained on `L = 0.5·L_coarse + 0.5·L_fine`, compared on accuracy / precision /
recall / F1 plus parameter count, GFLOPs, and per-image latency.

**That gap is now closed for real**, not in this directory: `scripts/run_baselines.py`
runs `resnet50` and `swin_tiny` as full Hydra experiments
(`conf/experiment/baseline_resnet50.yaml`, `baseline_swin_tiny.yaml`), each
trained end to end with the same coarse/fine heads and metrics `tasks.md`
specifies. The one deliberate deviation from the sketch: `swin_tiny` stands in
for the originally-proposed ViT-B/16, because it is in the same shifted-window
family as the paper's own encoder, which isolates DINOv2 pretraining as the only
variable (see the module docstring in `run_baselines.py`). `tasks.md` and the
stub notebook are kept here as the original design record, not as an open TODO.

Note that `tasks.md` names **Amaranthus** as the third seed type, matching the
paper; the dataset on disk has **Seasame**. Worth resolving before these results
are written up: the label set the code discovers comes from the directory tree,
so the paper and the artifacts currently disagree on one seed type's name.

## Architecture ablations belong in `conf/`, not here

Ablations that only change the model or loss are config overrides and need no
notebook. `conf/experiment/ablation_flat_classifier.yaml` is a worked example
that reuses the finetune trainer unchanged:

```bash
python main.py ablation                                      # flat classifier
python main.py finetune model.loss.lambda_kl=0.0             # no hierarchy KL
python main.py finetune model.loss.lambda_cosine=0.0         # no compactness
python main.py finetune model.head.top_k=1                   # single expert routing
python main.py finetune model.head.cross_attention_variant=gated
python main.py finetune model.head.seed_classifier_variant=se_gated
```

Each writes a fully-resolved config snapshot into its run directory, so the exact
configuration behind any number stays recoverable.

The supervised-baseline comparison sketched in `tasks.md` used neither the DINO
backbone nor the hierarchical head, which is why the original stub lived here
rather than as an experiment config — it needed real new code, now written as
`scripts/run_baselines.py` and its baseline experiment configs.
