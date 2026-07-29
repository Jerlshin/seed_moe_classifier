# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

This is a reference implementation of `../paper/sn-article.pdf`. Read
`PAPER_AUDIT.md` before changing anything in `src/models/` or `src/losses/` — it
records which design decisions are dictated by the paper and which are open — and
`REVISION_NOTES.md`, which records where this tree deliberately departs from the
submitted manuscript.

## Commands

```bash
python -m pip install -e ".[tracking,dev]"

python main.py pretrain      # == python -m src.trainers.contrastive_pretrain experiment=pretrain_swinv2_dino
python main.py finetune      # == python -m src.trainers.moe_finetune experiment=finetune_hierarchical_moe
python main.py ablation      # flat-classifier ablation
python main.py smoke         # 2-batch dry run of both stages

python scripts/dry_run.py         # synthetic end-to-end pipeline check, no dataset
python scripts/run_ablations.py   # six component-wise variants
python scripts/run_baselines.py   # ResNet-50, Swin-T, hierarchical CCE
python scripts/generate_plots.py  # figures + outputs/reports/summary_metrics.csv

scripts/train_distributed.sh pretrain|finetune|ablations|baselines|report

python -m pytest tests/ -q        # 280 tests, ~7s
```

Everything after the stage name is a Hydra override:

```bash
python main.py finetune data.batch_size=16 experiment.training.num_folds=5
python main.py finetune model.head.top_k=4          # submitted routing width
python main.py finetune model.head.use_moe=false    # single-component ablation
```

### Environment variables

- `SEED_DATA_ROOT` — dataset root
- `SEED_OUTPUT_DIR` — root for Hydra run dirs, checkpoints, metadata
- `SEED_PRETRAIN_BACKBONE` — the encoder checkpoint every downstream run loads;
  the **only** handoff between the two stages. Defaults to
  `$SEED_OUTPUT_DIR/checkpoints/dinov2_swinv2_pretrained.pth`.

## Architecture

Per-directory READMEs carry the detail: [`src/`](src/README.md),
[`models/`](src/models/README.md), [`losses/`](src/losses/README.md),
[`datasets/`](src/datasets/README.md), [`trainers/`](src/trainers/README.md),
[`utils/`](src/utils/README.md), [`conf/`](conf/README.md),
[`tests/`](tests/README.md), [`scripts/`](scripts/README.md). What follows is
only what is non-obvious from them.

### The head returns a dataclass, not a tuple

`HierarchicalSeedClassifier.forward` returns `HierarchicalOutput` with every
intermediate (`embedding`, `moe_features`, `refined_features`,
`attended_features`, `sub_logits`, `sub_margin_logits`, `gate_probs`, …).
`CombinedHierarchicalLoss.forward(output, seed_labels, sub_labels)` reads named
fields. This is deliberate: the repo was previously broken by exactly this —
components gained fields and positional unpacking at the call sites silently
went out of sync. **Do not reintroduce positional tuple returns.**

The baselines in `src/models/baselines.py` emit the same dataclass, which is what
lets the entire evaluation stack run against them unmodified.

### Three dataflow facts that are easy to get wrong

Tests in `tests/test_models.py` pin all three; a change that breaks one is a
divergence from the paper, not a failing test.

1. **The MoE is routed on `z`** (Eq. 8), the DINO embedding — not on the
   seed-type projection. The original code routed on `P(seed_logits)`, which
   meant the experts never saw the image.
2. **The residual adds `P(p_s)`** (Eq. 9) — softmax probabilities, not logits.
3. **`sub_logits` (no margin) drive prediction and the KL term;
   `sub_margin_logits` drive only the ArcFace loss.** Evaluation passes no
   labels, so the two coincide and metrics are not inflated by the margin.

### `z ∈ ℝ³⁸⁴` is an encoder invariant

`DinoV2SwinV2Encoder` owns the projection from the backbone's native width (1024
for Base, 768 for Tiny/Small) to the paper's 384, so
`encoder(images).shape[-1] == 384` always holds and the head's own input
projection collapses to `nn.Identity`.

Two consequences that bite if forgotten:

- **The projection is trainable even when the backbone is frozen.**
  `build_optimizer()` takes a list of modules and must include the encoder.
  Omitting it freezes the one layer adapting 1024 channels to 384, and nothing
  reports an error — the head just trains against a random projection.
- **Checkpoints carry `encoder_state_dict`** alongside `model_state_dict`.

### SwinV2 is the only DINOv2 backbone

`validate_swinv2_name()` rejects any non-SwinV2 name, and is called from both
`BackboneFeatureExtractor` and `DINO.__init__`. The comparative ViT-S/14 path was
removed in the revision. Supervised comparison backbones (ResNet-50, Swin-T) go
through `src/models/baselines.py` and `conf/experiment/baseline_*.yaml` — never
through `conf/model/backbone/`.

### Top-2 routing, and what it does to gradients

`DEFAULT_TOP_K = 2` in `src/models/components/moe_layer.py` is the single place
the routing width is defined (the paper used 4).

With sparse dispatch, **an expert no sample routed to receives no gradient that
step**. This is correct MoE behaviour, but it is far more visible at `K = 2`: a
batch of 12 fills only 24 routing slots across six experts. Do not "fix" it by
asserting every expert always has a gradient — the tests assert the real
invariant, `routed ⇔ has gradient`, and the load-balancing term is what keeps
utilisation uniform over an epoch.

### Component toggles

Five booleans under `model.head`: `use_moe`, `use_arcface`, `use_residual`,
`use_cross_attention`, `use_kl_loss`. Rules when touching them:

- **Do not allocate a disabled block** — set the attribute to `None`, not
  `nn.Identity`, so ablation parameter counts are honest.
- **`use_moe=False` substitutes one dense block**, not nothing, so the `wo_moe`
  gap measures routing rather than a missing layer's capacity.
- **`use_arcface=False` needs no loss-side branch.** `LinearSubVarietyHead`
  returns its logits unchanged as `sub_margin_logits`, so the ArcFace term
  degrades to plain CE by itself. Adding a second CE path would let the ablation
  drift from the full model.
- `use_kl_loss` lives on the head but is consumed by the loss, reached via
  `use_kl_loss: ${model.head.use_kl_loss}`.

### The criterion is stateless

ArcFace's class centres live in `model.arcface`. Consequences: the optimiser only
needs the model plus encoder, gradient clipping covers everything, and
`model_state_dict` alone reproduces the head. Do not move the centres into the
loss.

### One trainer, all variants

The full model, six ablations and three baselines all run through
`moe_finetune.py`, differing only by Hydra overrides. An ablation routed through
a second loop would differ in ways nobody chose. `build_model_and_encoder()`
dispatches on `model.head.name`.

### Suite runs must share one encoder

Every DINOv2-path variant reads
`outputs/checkpoints/dinov2_swinv2_pretrained.pth`, published once by the
pretrain stage. `ensure_pretrained_checkpoint()` refuses to start otherwise. Do
not add a per-variant pretraining step: the resulting table would partly measure
self-supervised initialisation noise, and would look completely normal while
doing so.

Suite variants run as subprocesses because Hydra can only be initialised once per
process; `GlobalHydra.instance().clear()` between variants leaves stale state.

### Labels come from the directory tree

`HierarchicalSeedDataset` walks `root/<seed_type>/<sub_variety>/*` and assigns
indices from **sorted** names. Sub-variety labels are **global** (0..26), not
per-seed-type. The sub-variety → seed-type map that builds the KL aggregation
matrix is derived from this tree at runtime. Adding or renaming a folder shifts
every index and invalidates checkpoints; the trainer refuses to start when the
discovered counts disagree with `data.num_seed_types` / `num_sub_varieties`.

Splits stratify on `seed_label * 1000 + sub_label`, are driven entirely by
`cfg.seed` (so every variant sees the identical partition), and are persisted to
`split_manifest.npz`.

### Config groups are genuinely composed

`conf/config.yaml` composes `data`, `model/backbone`, `model/head`,
`model/loss`, `experiment`, `tracking`. Experiment files are `# @package
_global_` and select their head/loss via `defaults: - override /model/head: ...`.
Every ablation and baseline inherits from `finetune_hierarchical_moe`.

> Historical note: an earlier revision duplicated model settings inline in each
> experiment file, leaving `conf/model/*` dead. If you find documentation
> claiming editing `conf/model/*` has no effect, it is stale.

### Frozen backbone is a config flag

`model.backbone.freeze: true` (default) trains only the head and the projection.
The paper fine-tunes the encoder in stage 2 (`PAPER_AUDIT.md` §6.1); flip to
`false` to match it, at much higher cost. `BackboneFeatureExtractor.train()` is
overridden so a frozen trunk can never be pulled back into train mode.

`checkpoint_strict: false` is the default, so `load_checkpoint` returns the
missing/unexpected key report and the trainer logs it. Check that line first when
metrics look wrong.

### Run artifacts are a contract

Every run writes `summary.json` and `test_predictions.npz` into its save path.
`scripts/generate_plots.py` reads only those, and re-scores from the raw
predictions rather than trusting the stored metrics — so the table and the
figures are always computed by the same code. Keep both files populated when
changing the trainer.

### Tracking and disk pressure

`ExperimentTracker` always appends to `events.jsonl`; TensorBoard and W&B are
both on by default, W&B in `offline` mode so runs never block on credentials.
A missing backend degrades to a warning.

Defaults are tuned for a 16 GB vast.ai root disk: parameter/gradient histograms
off, `keep_last_n_checkpoints: 1`, no optimizer state, no teacher weights.
Turning these on for a 300-epoch run is how the disk fills.

### Device handling

`select_device` supports `cuda` / `mps` / `cpu` with `auto`, so this runs on
Apple Silicon. AMP is gated on `device.type == "cuda"`; `pin_memory` likewise.

Efficiency timing calls `synchronize()` before and after the timed loop. Removing
it would report enqueue speed rather than execution speed — off by roughly an
order of magnitude.

### Notebook-compat escape hatch

`data.dataset_format=pickle_batches` swaps in `PickleBatchSeedDataset` for
pretraining on flattened-RGB pickle files.

## Conventions

- Paper equations are cited in docstrings by number. Keep that up when editing.
- Paper constants live in `tests/conftest.py`; assert against those, not literals.
  Routing width has two: `SUBMITTED_TOP_K = 4` and `REVISED_TOP_K = 2`.
- New loss terms: add the term, add its weight to `conf/model/loss/arcface_kl.yaml`,
  add a field to `LossBreakdown` (it is then logged everywhere automatically).
- New ablations: add a boolean flag, a config key, and a `VariantSpec` — no
  trainer changes should be needed.
