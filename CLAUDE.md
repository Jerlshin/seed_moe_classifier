# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

This is a reference implementation of `../paper/sn-article.pdf`. Read
`PAPER_AUDIT.md` before changing anything in `src/models/` or `src/losses/` — it
records which design decisions are dictated by the paper and which are open —
plus `REVISION_NOTES.md` (where this tree departs from the submitted manuscript,
with the override that reverses each departure) and `AUDIT_RESPONSE.md` (the
disposition of the independent audit in `CHANGES.md`, and the dataset
measurements that constrain the whole protocol).

## Commands

```bash
python -m pip install -e ".[tracking,dev]"

python main.py pretrain      # == python -m src.trainers.contrastive_pretrain experiment=pretrain_swinv2_dino
python main.py finetune      # == python -m src.trainers.moe_finetune experiment=finetune_hierarchical_moe
python main.py ablation      # flat-classifier ablation
python main.py smoke         # 2-batch dry run of both stages

python scripts/dry_run.py         # synthetic end-to-end pipeline check, no dataset
python scripts/run_ablations.py   # 18 variants x 5 seeds
python scripts/run_baselines.py   # linear probe, swinv2_supervised, ResNet-50, Swin-T, hierarchical CCE
python scripts/generate_plots.py  # figures + summary_metrics.csv (mean +- SD, McNemar p)

scripts/train_distributed.sh pretrain|finetune|ablations|baselines|report

python -m pytest tests/ -q        # 341 tests, ~8s
```

Everything after the stage name is a Hydra override:

```bash
python main.py finetune data.batch_size=16 experiment.training.num_folds=5
python main.py finetune model.head.top_k=4               # submitted routing width
python main.py finetune model.head.token_mode=pooled     # submitted pooled head
python main.py finetune experiment.training.split_protocol=stratified  # submitted split
python main.py finetune model.head.use_moe=false         # component ablation
```

`REVISION_NOTES.md` §0 tabulates every departure from the submitted manuscript
next to the override that reverses it.

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

### The dataset is 81 photographs, not 9,357 images

`Cropped_Samples` holds 9,357 crops cut from **81 source photographs** — a mean of
115.5 crops per source, encoded in the filenames (`IMG_0502_bbox137.png`).
`split_protocol: grouped` is therefore the default: crop-level splitting puts
near-duplicate views of the same physical seeds on both sides of the boundary,
and the accuracy becomes substantially a memorisation score.

Five sub-varieties have crops from exactly **one** photograph, so no grouped split
can place them on both sides. For those classes no protocol on this dataset
measures across-photograph generalisation. The trainer logs this at startup and
records it in `summary.json`; do not "fix" it by falling back to crop-level
splitting. `leakage_ungrouped` measures the delta between protocols, which is a
result worth reporting.

Related: the crops are **tiny** — median 52 x 51 px, all under 256 — so every
image is upsampled ~5x. And they are **not square** (3.4 % are), which is why
`get_supervised_transforms` passes `Resize((H, W))` as a tuple. Changing that to
an integer resizes only the shorter side and crashes `default_collate`.

### Routing granularity: `token_mode`

`grid` (default) keeps SwinV2's `8x8` token grid through the MoE and the
cross-attention and pools afterwards; `pooled` reproduces the submitted
architecture. This is not a tuning knob — it decides whether three modules are
functions or affine maps.

Over a length-1 sequence `softmax(QK^T/sqrt(d))` is identically 1, so `Q` and `K`
receive exactly zero gradient forever. Under `pooled` the head therefore **does
not allocate** those projections; it substitutes the single `nn.Linear` that spans
the identical function class. Never re-introduce a `MultiheadAttention` on a
length-1 path — ~2.07 M unreachable parameters were previously being counted in
the results table's "Active Params" column.

Grid routing also raises routing slots per step from `batch x K` to
`batch x 64 x K`, which is what makes the load-balancing statistic estimable at
batch 16.

### The load-balancing loss must see the hard dispatch

`moe_load_mode: switch` is `E * sum_i f_i P_i` — the hard dispatch fraction `f`
coupled to the differentiable router probability `P`. Do not revert to the
entropy form as a default. On `G = (.3, .3, .1, .1, .1, .1)` the entropy form
scores 92 % of "perfect balance" while Top-2 sends every sample to two experts,
and its **global optimum** produces maximally imbalanced hard routing because
`torch.topk` breaks ties toward the lowest indices. `tests/test_losses.py` pins
that counterexample.

`MoEOutput.gate_logits` exists because the router z-loss needs pre-softmax
values; applied to probabilities it is a constant.

### Sparse and dense dispatch are only equal if you make them equal

Forward-equality is not enough. Under sparse dispatch an unrouted expert has
`grad is None`, and AdamW skips such parameters **entirely** — including decoupled
weight decay. `materialize_expert_grads()` must be called between `backward()` and
`step()`; without it the two dispatch modes train measurably different models and
rarely-routed experts carry stale Adam moments.

### Aggregate the hierarchy KL in log space

`aggregate_sub_log_probs` uses `logsumexp` over each parent's children. The
previous `log(aggregated.clamp_min(1e-8))` had **zero gradient in the clamped
region**, and with `s = 30` that was the common case — the term was live when the
two heads agreed and dead when they disagreed confidently. Never reintroduce a
`clamp -> log` composition here; raising the epsilon does not fix it.

`detach_kl_seed_target` defaults to `true`: the coarse head is already supervised
by hard labels, so a non-detached term can be reduced by the coarse head becoming
*less* accurate.

### Compactness is intra-class, and the residual is controlled structurally

`cosine_mode` defaults to `intra_class` with **EMA centroids**. Do not revert to
`residual`: `1 - cos(h + P(p_s), h)` is minimised by `P(p_s) = 0`, which is
literally the `use_residual=False` ablation — the loss rewards deleting the
connection it regularises. Magnitude control lives in `LayerScale` (init `1e-4`)
and an optional hinge that is exactly zero in the healthy regime.

EMA centroids are not an optimisation. With batch 16 over 27 classes only ~3.2
classes have two or more members, so per-batch centroids leave ~10 of 16 samples
contributing exactly zero.

### ArcFace scale is analytic, not inherited

`arcface_scale: "auto"` resolves to AdaCos's `sqrt(2) log(C-1) = 4.61` for
`C = 27`. The submitted `30.0` is ArcFace's face-recognition value for 10^5-10^6
identities; here it put `L_ArcFace` at ~17.6 against `L_seed = 1.386`, saturated
`softmax(s cos)` into a near-one-hot distribution, and made calibration
meaningless. The margin ramps `0 -> m` over the first 15 % of training.

`sub_head_variant` has three values. `normface` (normalised, `m = 0`) is the
single-factor margin control; swapping straight to `linear` changes four things
at once, which is why that variant is named `wo_angular_head`.

### Ablations must be single-factor, and named for what they change

Of the six variants the first revision shipped, only `wo_kl` and `wo_cross_attn`
flipped one factor. `architecture/06_ABLATION_ENGINE.md` §2 tabulates what each
one *actually* changed and which control now isolates it. When adding a variant,
state its factors in the `VariantSpec` description.

Every variant runs at five seeds (`DEFAULT_SEEDS`) into
`{group}/{variant}/seed{n}/`. One run per variant cannot resolve the gaps the
table reports: the 95 % CI half-width on a *difference* of two accuracies on this
test split is +-1.40 pp, against component contributions of 0.5-2 pp.
`generate_plots.py` aggregates to mean +- SD and runs McNemar's exact test, which
is valid because the split is byte-identical across variants.

### Stage 1 is DINO, not DINOv2

`CustomDINOLoss` implements Caron et al. (2021) plus the two DINOv2 components
that do not need patch tokens (KoLeo, Sinkhorn-Knopp centering). iBOT and untied
heads are **not** implemented. Do not reintroduce "DINOv2" as a description of
what the code does.

`dynamic_img_size=True` does **not** let SwinV2 accept non-native resolutions —
timm accepts the flag and the attention still asserts 256. That is measured and
pinned by a test, so local crops must stay upsampled and the resulting blur
shortcut is documented rather than removed.

### Every run leaves a complete machine-readable trace

`component_flags()` reports every architectural axis a variant can move, and the
criterion contributes `loss_flags()`. Reporting only the four architectural
booleans made a `wo_kl` run byte-identical to `full_model` in `summary.json`.
`summary.json` also carries `split` (protocol + provenance diagnostics) and
`fold_metrics` (mean +- std across folds, because reporting the best fold's test
score is a selection procedure with an optimistic expectation).

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

With sparse dispatch, **an expert no token routed to receives no gradient that
step**. This is correct MoE behaviour. Do not "fix" it by asserting every expert
always has a gradient — the tests assert the real invariant, `routed ⇔ has
gradient`.

Do **not** claim, as this file previously did, that the load-balancing term keeps
utilisation uniform over an epoch. That claim is false for the entropy form (see
above) and is now a *measured* quantity: `dead_experts` is logged per step and
`expert_label_nmi` reports whether the routing carries any information about the
label. Balance is not specialisation.

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
The paper fine-tunes the encoder in stage 2 (`PAPER_AUDIT.md` §7.1); flip to
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
  Most now come in **pairs** — `SUBMITTED_TOP_K` / `REVISED_TOP_K`,
  `SUBMITTED_TEACHER_TEMP` / `REVISED_TEACHER_TEMP`, `SUBMITTED_ARCFACE_SCALE` /
  `ADACOS_SCALE_27` — so a test asserting a bare number cannot silently become a
  claim about whichever value the reader assumed. Dataset provenance constants
  (`DATASET_NUM_SOURCE_PHOTOGRAPHS`, `DATASET_SINGLE_SOURCE_SUBVARIETIES`) are
  there too, because they decide what any accuracy figure can mean.
- New loss terms: add the term, add its weight to `conf/model/loss/arcface_kl.yaml`,
  add a field to `LossBreakdown` (logged everywhere automatically) and to
  `loss_flags()` (so a variant using it is machine-distinguishable).
- New ablations: add a flag, a config key, and a `VariantSpec` whose description
  states **every** factor it changes — no trainer changes should be needed.
- Before adding a module: check it can receive gradient in every configuration it
  ships in. Two blocks in this tree could not, and both were counted as active
  parameters until someone did the arithmetic.
