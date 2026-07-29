# `notebooks/` — exploratory work

Scratch space. Nothing here is imported by `src/`, exercised by `tests/`, or
required by either training stage.

| Path | Contents |
| --- | --- |
| `abalation_studies/tasks.md` | Protocol for the supervised-baseline comparison |
| `abalation_studies/resnet_comparison.ipynb` | Stub for that comparison |

## Relationship to the paper's Limitations section

The paper states (Section 8) that *"comparisons with standard supervised
baselines such as ResNet and Vision Transformer were not conducted in the present
work"*. `tasks.md` sketches the protocol for closing that gap: a shared backbone
(ResNet-50 or ViT-B/16) with a 4-class coarse head and a 27-class fine head,
trained on `L = 0.5·L_coarse + 0.5·L_fine`, compared on accuracy / precision /
recall / F1 plus parameter count, GFLOPs, and per-image latency.

Note that `tasks.md` names **Amaranthus** as the third seed type, matching the
paper; the dataset on disk has **Seasame**. See `PAPER_AUDIT.md` §6.2 — worth
resolving before these results are written up.

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

The supervised-baseline comparison in `tasks.md` genuinely needs new code — it
uses neither the DINO backbone nor the hierarchical head — which is why it lives
here rather than as an experiment config.
