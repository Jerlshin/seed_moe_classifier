# `src/` — implementation

Five packages, split by responsibility. Nothing imports "upward": `models`
never imports `trainers`, `losses` never imports `datasets`.

| Package | Responsibility | README |
| --- | --- | --- |
| `models/` | Network definitions — SwinV2 encoder, hierarchical head, components, baselines | [→](models/README.md) |
| `losses/` | Every objective, for both stages | [→](losses/README.md) |
| `datasets/` | Image-folder dataset, label hierarchy, augmentation | [→](datasets/README.md) |
| `trainers/` | Hydra entry points and the ablation/baseline suite runner | [→](trainers/README.md) |
| `utils/` | Metrics, efficiency profiling, reporting, figures, tracking | [→](utils/README.md) |

## Dependency direction

```
trainers ──▶ models ──▶ components
    │           └─────▶ backbones
    ├──────▶ losses  ──▶ models.components   (ArcFaceHead, HierarchicalOutput)
    ├──────▶ datasets
    └──────▶ utils   ──▶ utils.training
```

`losses` depending on `models` is deliberate: `CombinedHierarchicalLoss` takes a
`HierarchicalOutput` rather than a dozen positional tensors, which is what stops
loss terms and forward-pass outputs from drifting apart.

## Two conventions worth knowing

**Structured returns.** `MixtureOfExperts`, `CrossAttention` and the hierarchical
head all return NamedTuples/dataclasses, not bare tuples. Positional unpacking of
these is what previously broke the model when a component gained a field.

**The criterion is stateless.** ArcFace's class centres live in the model, so
`model.parameters()` is the complete optimisation set and a saved
`model_state_dict` is sufficient for inference.

## Adding a loss term

1. Implement it in `src/losses/`, as a function plus (if it has state) a module.
2. Add its weight to `conf/model/loss/arcface_kl.yaml`.
3. Wire it into `CombinedHierarchicalLoss.forward` and add a field to
   `LossBreakdown` — it is then logged automatically by every tracker.
4. If it needs a tensor the head does not yet expose, add a field to
   `HierarchicalOutput`. No call site needs updating.

## Adding an ablation

Usually none of the above. If the variant is "the proposed model without block
X", add a boolean to `HierarchicalSeedClassifier.__init__`, bypass or replace the
block in `forward`, expose the flag in `component_flags()`, declare it in
`conf/model/head/hierarchical_moe.yaml`, and add a `VariantSpec` to
`scripts/run_ablations.py`. The trainer, the losses, the metrics and the
reporting all keep working, because `HierarchicalOutput` stays fully populated.

Two rules that keep an ablation honest:

* **Do not allocate the disabled block.** Set the attribute to `None` rather than
  `nn.Identity`, so the reported parameter count describes the model actually
  trained.
* **Replace, do not delete, when capacity is at stake.** `use_moe=False`
  substitutes one dense transformer block instead of removing the layer, so the
  measured gap is attributable to *routing* rather than to a missing block.
