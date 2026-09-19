# Kaggle Runbook — Stage-2 Ablations and Baselines on Dual Tesla T4

Operational runbook for producing every ablation and baseline row of the results
table on Kaggle's free `T4 x2` accelerator, inside the 12-hour session boundary,
with **zero pre-training compute**.

Scope: stage 2 only. Stage 0 (corpus extraction) and stage 1 (DINO
self-distillation) are finished and are never re-run here. Nothing in this
document can start stage 1, and `scripts/run_experiments_suite.py` cannot either
— by construction.

Companion documents, none of which this one replaces:
[`CLAUDE.md`](CLAUDE.md) (what is non-obvious and silent when broken),
[`SERVER_RUN_GUIDE.md`](SERVER_RUN_GUIDE.md) §3.3a–3.7 (the general server
guide, including the Kaggle stage-1 path), [`README.md`](README.md) (the
user-facing command sequence).

---

## 0. Read this first — five corrections to the plan as specified

Each of these was checked against the code and the shipped run's own
`summary.json`, not assumed. Four are cost or provenance corrections; one is a
confirmation.

### 0.1 The runtime profile is not a deviation — it *is* the shipped configuration ✅

`outputs/finetune_hierarchical_moe/summary.json`, the run that produced the
81.73 % headline, records:

```json
"runtime": { "amp": "fp16",
             "distributed": { "enabled": true, "world_size": 2, "backend": "nccl" } },
"config":  { "batch_size": 8, "epochs": 100, "seed": 42,
             "split_protocol": "stratified", "num_folds": 1 }
```

So `data.batch_size=8` on 2 DDP ranks at `amp=fp16` is exactly what the
reference row was trained under. Keeping it is what makes every arm here
paired against that row. **Do not change it.**

### 0.2 `linear_probe` is NOT a 1.5-hour run — budget ~2.9 h ⚠️

There is no frozen-feature cache in stage 2. `src/trainers/moe_finetune.py:677`
calls `self.encoder(images)` on every step of every epoch, and
`BackboneFeatureExtractor.forward` runs the frozen trunk under `torch.no_grad()`
(`src/models/builder.py:796`) — but it still *runs* it. The SwinV2 forward
dominates the step for every frozen arm; the head is the only difference, and
`linear_probe`'s head is two `nn.Linear` layers.

Consequence: `linear_probe` costs roughly what `wo_moe` costs, minus a small
head. The proposed Session 1 (`wo_moe + wo_margin_only + wo_residual +
linear_probe`) is therefore **~12.2 h, not ~10.5 h** — it overruns the session.
§4 and §5 re-derive the packing from this.

### 0.3 `full_model` is not in the list, and it is the McNemar reference ⚠️

`scripts/generate_plots.py` defaults to `--reference full_model`. The published
production run at `outputs/finetune_hierarchical_moe/` carries
`"name": "finetune_hierarchical_moe"`, `"group": "proposed"` — **not**
`full_model`. With the eleven arms as listed and nothing else, the `p (vs full)`
and `p (Holm)` columns come back empty.

Two ways out, both fine:

| Option | Cost | What you get |
| --- | --- | --- |
| **A (recommended)** — keep the published run as the reference: `generate_plots.py --reference finetune_hierarchical_moe` | free | McNemar against the exact 81.73 % row the manuscript reports |
| B — add `full_model` (seed 42) as a twelfth arm | +3.1 h | a reference trained inside this suite, under an identical command shape |

Option A is valid because the reference shares the byte-identical test split:
same corpus SHA-256, same `seed=42`, same `stratified` protocol, same
`test_size`. That is the whole precondition McNemar needs.

### 0.4 Every end-to-end baseline needs network access (or a warmed cache) ⚠️

Clean split, worth internalising:

- **Checkpoint-consuming arms** (the five `wo_*` ablations, `linear_probe`)
  build SwinV2 with `pretrained: false` and load your stage-1 `.pth`. They need
  **no internet at all**.
- **End-to-end baselines** (`vit_small`, `swinv2_tiny`, `convnext_tiny`,
  `resnet50`, `swinv2_supervised`) call `timm.create_model(..., pretrained=True)`
  and will reach for the Hugging Face Hub. With Kaggle's *Internet: off* they
  fail at model construction, minutes into the session.

Turn Internet **on** for Sessions 3–5, or pre-warm the cache (§2.4).

### 0.5 Two baseline trunks are not ImageNet-**1k** supervised ⚠️ *(manuscript-facing)*

timm's default pretrained tags, resolved from the installed timm 1.0.29:

| Arm | `BASELINE_MODELS` entry | Default tag actually loaded |
| --- | --- | --- |
| `swinv2_tiny` | `swinv2_tiny_window16_256` | `ms_in1k` — pure ImageNet-1k ✅ |
| `resnet50` | `resnet50` | `a1_in1k` — pure ImageNet-1k ✅ |
| `vit_small` | `vit_small_patch16_224` | **`augreg_in21k_ft_in1k`** — IN-21k → IN-1k |
| `convnext_tiny` | `convnext_tiny` | **`in12k_ft_in1k`** — IN-12k → IN-1k |

`CLAUDE.md` and `scripts/run_baselines.py` describe the `vit_small` /
`swinv2_tiny` pair as holding "same ImageNet-1k supervised initialisation"
fixed, with the trunk as the only difference. With the default tags that is
inexact: the ViT arm carries an extra IN-21k pretraining corpus. Either say so
in the manuscript, or pin a pure IN-1k tag
(`model.head.baseline_model=vit_small_patch16_224.augreg_in1k`) — a full timm
identifier passes straight through `BASELINE_MODELS.get(name, name)`
(`src/models/baselines.py:161`). This runbook uses the defaults and flags it;
changing it is a scientific decision, not an operational one.

---

## 1. Invariants

### 1.1 Canonical inputs

| Thing | Path | Notes |
| --- | --- | --- |
| Corpus | `/kaggle/input/datasets/jgfreak/refined-samples/refined_samples/Refined_Samples` | 13,492 crops, 27 classes, 96 source photographs |
| Corpus SHA-256 | `013c04c5858e9815ac0bb71b9f5fde4c8be9176c514d28198e39727344b6f3d3` | recorded in every `summary.json`; two digests in one table is a re-baseline, not a comparison |
| Stage-1 encoder | `/kaggle/input/datasets/jgfreak/refined-samples/dino_backbone_epoch_0020.pth` | a stage-1 milestone encoder; consumed only by the arms marked **YES** in §3 |
| Output root | `/kaggle/working/outputs` | `/kaggle/input` is read-only |

The corpus digest is computed from dataset-relative paths plus the sample, class
and source-group counts, so it is stable across mount points. If a run's
`summary.json` shows a different digest, that run does not belong in this table.

### 1.2 The evaluation protocol — fixed, for all arms

```
split_protocol: stratified     # crop-level, 20 % held out, stratified on sub-variety
num_folds:      1
test_size:      0.2            # 2,699 crops, all 27 classes present
monitor:        sub_variety/f1_macro
seed:           42
```

This is the sole protocol the study reports under. Every arm inherits it from
`conf/experiment/finetune_hierarchical_moe.yaml`; do not override it, and do not
mix in a `grouped` or `grouped_cv` row. The split is byte-identical across
variants because it is driven entirely by `cfg.seed` — that is what makes the
McNemar column valid.

### 1.3 Execution profile

| Knob | Value | Why |
| --- | --- | --- |
| `experiment.training.amp` | `fp16` | `sm_75` has no hardware bf16. `auto` resolves to the same thing on a T4; passing it explicitly puts it in `summary.json` unambiguously. Evaluation is fp32 regardless (`AMP_DISABLED`). |
| DDP ranks | 2 (`--gpus 2`) | reproduces the shipped run's `world_size: 2` |
| `data.batch_size` | 8 **per rank** | global batch 16, matching the reference row |
| `data.num_workers` | 2 | per *process*; 2 ranks × 2 = 4 workers on Kaggle's ~4 vCPUs. `auto` resolves to the same 2. |
| `experiment.training.resume` | `auto` | continues from the newest valid epoch-boundary checkpoint, starts fresh when there is none — so one command line serves first launch and every relaunch |
| `experiment.training.epochs` | 100 | unchanged from the reference row (see §4.3 before shortening it) |
| `seed` | 42 | the reference row's seed |

### 1.4 Stage 2 has no `effective_batch_size`

Unlike stage 1, the stage-2 trainer has no accumulation-authority knob:
`data.batch_size` is the per-rank micro-batch and DDP multiplies it by the world
size. `8 × 2 ranks = 16` reproduces the reference; `16 × 2` would not.

One consequence worth knowing: there is **no `SyncBatchNorm` conversion anywhere
in this tree** (verified: no `convert_sync_batchnorm` call). Under 2×8 DDP, a
BatchNorm trunk normalises over 8 samples per rank, not 16. Of the arms here,
only `resnet50` and (if you add it) `efficientnetv2_s` have BatchNorm; SwinV2,
ViT and ConvNeXt are LayerNorm and are unaffected. This is consistent across all
arms at a fixed `batch_size`, so the table stays internally comparable — just do
not compare a `resnet50` number here against one trained single-GPU at batch 16.

### 1.5 Single seed, and what the table can then claim

`DEFAULT_SEEDS` is `(42, 43, 44, 45, 46)` and the repo's own justification is
blunt: on this 2,699-crop test split the 95 % CI half-width on a *difference* of
two accuracies is ±1.40 pp, against component contributions of 0.5–2 pp.

At seed 42 only:

- the `± SD` columns in `summary_metrics.csv` come back **blank** (one run per
  variant has no dispersion);
- **McNemar's exact test is still valid and is still the right statistic** — it
  is paired on the byte-identical test split, so it does not need repeated
  seeds. Report `p (Holm)`, not overlapping intervals.

If compute later allows, adding seeds 43–46 costs 4× §4's totals and is the
thing that turns point estimates into mean ± SD. Say "single seed, paired
McNemar" in the manuscript rather than implying dispersion you did not measure.

---

## 2. Kaggle environment

### 2.1 Notebook settings

- **Accelerator:** `GPU T4 x2`
- **Internet:** **on** for Sessions 3–5 (timm weights). Optional for Sessions 1–2.
- **Persistence:** "Files only" is enough; §6 is the reliable mechanism.

### 2.2 Setup cell

```bash
%%bash
set -euo pipefail

# The repo. Replace with your own source (dataset, git clone, or upload).
cd /kaggle/working
[ -d seed-moe-classifier ] || git clone <your-repo-url> seed-moe-classifier
cd seed-moe-classifier

pip install -q -e ".[tracking]"

nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv
python -c "import torch, timm; print('torch', torch.__version__, '| timm', timm.__version__,
           '| cuda', torch.cuda.is_available(), '| devices', torch.cuda.device_count())"
```

### 2.3 Environment cell — run this at the top of **every** session

```bash
%%bash
export SEED_DATA_ROOT=/kaggle/input/datasets/jgfreak/refined-samples/refined_samples/Refined_Samples
export SEED_OUTPUT_DIR=/kaggle/working/outputs
export SEED_PRETRAIN_BACKBONE=/kaggle/input/datasets/jgfreak/refined-samples/dino_backbone_epoch_0020.pth

# Cheap pre-flight: the three paths that silently ruin a session if wrong.
test -d "$SEED_DATA_ROOT"          || { echo "MISSING CORPUS: $SEED_DATA_ROOT"; exit 1; }
test -f "$SEED_PRETRAIN_BACKBONE"  || { echo "MISSING ENCODER: $SEED_PRETRAIN_BACKBONE"; exit 1; }
mkdir -p "$SEED_OUTPUT_DIR"

echo "corpus  : $(find "$SEED_DATA_ROOT" -type f \( -name '*.png' -o -name '*.jpg' \) | wc -l) files (expect 13492)"
echo "encoder : $(sha256sum "$SEED_PRETRAIN_BACKBONE" | cut -c1-16)…"
```

`SEED_REFINED_DATA_ROOT` and `SEED_RAW_DATA_ROOT` are **not** set — they belong
to stage 0, which does not run here. `/kaggle/input` is read-only and nothing in
stage 2 writes to it.

Note that in a Kaggle `%%bash` cell, `export` does **not** survive to the next
cell. Either keep each experiment in a single cell that re-exports (the commands
in §7 are written that way), or set them once via `os.environ` in a Python cell.

### 2.4 Pre-warming timm weights (only if Internet must stay off)

```bash
%%bash
export HF_HOME=/kaggle/working/hf_cache   # persists in the working dir
python - <<'PY'
import timm
for name in ["vit_small_patch16_224", "swinv2_tiny_window16_256",
             "convnext_tiny", "resnet50"]:
    timm.create_model(name, pretrained=True, num_classes=0)
    print("cached", name)
PY
```

Run that once with Internet on, save `/kaggle/working/hf_cache` into the
carry-over dataset (§6), and export `HF_HOME` to point at it in later sessions.

### 2.5 Optional: quieten the tracker

`tracking.wandb.enabled` defaults to `true` in `offline` mode, which writes a
`wandb/` tree into each run's Hydra directory. On a disk-constrained session:

```
tracking.wandb.enabled=false
```

TensorBoard and `events.jsonl` are unaffected, and nothing the report reads
comes from W&B.

---

## 3. Experiment inventory and dependency matrix

Twelve rows: eleven as specified, plus the reference. All definitions taken from
`scripts/run_ablations.py`, `scripts/run_baselines.py` and the composed Hydra
configs.

### 3.1 The matrix

| # | Category | Variant | Experiment config | Single-factor override | Requires stage-1 checkpoint? | Trunk | LR | Est. wall clock |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | *Reference* | `finetune_hierarchical_moe` | `finetune_hierarchical_moe` | — | **YES** | SwinV2-T, frozen | 1e-4 | **already done** (3.13 h) |
| 1 | Architectural ablation | `wo_moe` | `finetune_hierarchical_moe` | `model.head.use_moe=false` | **YES** | SwinV2-T, frozen | 1e-4 | ~3.1 h |
| 2 | Architectural ablation | `wo_margin_only` | `finetune_hierarchical_moe` | `model.head.sub_head_variant=normface` | **YES** | SwinV2-T, frozen | 1e-4 | ~3.1 h |
| 3 | Architectural ablation | `wo_residual` | `finetune_hierarchical_moe` | `model.head.use_residual=false` | **YES** | SwinV2-T, frozen | 1e-4 | ~3.1 h |
| 4 | Architectural ablation | `wo_cross_attn` | `finetune_hierarchical_moe` | `model.head.use_cross_attention=false` | **YES** | SwinV2-T, frozen | 1e-4 | ~3.1 h |
| 5 | Architectural ablation | `wo_kl` | `finetune_hierarchical_moe` | `model.head.use_kl_loss=false` | **YES** | SwinV2-T, frozen | 1e-4 | ~3.1 h |
| 6 | Supervised baseline | `linear_probe` | `baseline_linear_probe` | *(config: `linear_probe` head + `flat_cce`)* | **YES** | SwinV2-T, frozen | 1e-3 | **~2.9 h** |
| 7 | Supervised baseline | `vit_small` | `baseline_vit_small` | *(config: `flat_supervised`, `vit_small`, `img_size=256`)* | **NO** — timm ImageNet | ViT-S/16, unfrozen | 3e-5 | ~4.5–5.5 h |
| 8 | Supervised baseline | `swinv2_tiny` | `baseline_swinv2_tiny` | *(config: `flat_supervised`, `swinv2_tiny`)* | **NO** — timm ImageNet | SwinV2-T, unfrozen | 3e-5 | ~4.5–5.5 h |
| 9 | Supervised baseline | `convnext_tiny` | `baseline_convnext_tiny` | *(config: `flat_supervised`, `convnext_tiny`)* | **NO** — timm ImageNet | ConvNeXt-T, unfrozen | 3e-5 | ~4.5–5.5 h |
| 10 | Supervised baseline | `resnet50` | `baseline_resnet50` | *(config: `flat_supervised`, `resnet50`)* | **NO** — timm ImageNet | ResNet-50, unfrozen | 3e-5 | ~4.5–5.5 h |
| 11 | Supervised baseline | `swinv2_supervised` | `baseline_swinv2_supervised` | *(config: full hierarchical head, `freeze=false`, `checkpoint_path=null`)* | **NO** — timm ImageNet | SwinV2-T, unfrozen | 3e-5 | ~4.5–5.5 h |

Strongly recommended twelfth row, not in the original list:

| # | Category | Variant | Experiment config | Requires checkpoint? | Trunk | Est. |
| --- | --- | --- | --- | --- | --- | --- |
| 12 | Control | `imagenet_frozen` | `control_imagenet_frozen` | **NO** — timm ImageNet | SwinV2-T, **frozen** | ~3.1 h |

`imagenet_frozen` is the lower arm of the stage-1 comparison: ImageNet SwinV2,
trunk frozen, full hierarchical head, no stage 1. Proposed-model minus this row
is *what in-domain self-supervised pretraining bought*, with head, split and
seed all held fixed. Stage 1 is the most expensive thing in the pipeline; this
is the row that justifies it. It is 3.1 h (frozen trunk) and it answers the
first question a reviewer asks about the whole two-stage design.

### 3.2 How to read "Requires checkpoint?"

The dependency is structural, not a policy:

- **YES** — `build_model_and_encoder` (`src/trainers/moe_finetune.py:429`) builds
  a real `DinoV2SwinV2Encoder` from `conf/model/backbone/swinv2.yaml`, whose
  `pretrained: false` and `checkpoint_path: ${oc.env:SEED_PRETRAIN_BACKBONE,…}`
  mean the weights come entirely from your `.pth`. Omit it and
  `BackboneFeatureExtractor.load_checkpoint` raises `FileNotFoundError` with the
  remedy in the message. These arms never touch the network.
- **NO (flat_supervised)** — `vit_small`, `swinv2_tiny`, `convnext_tiny`,
  `resnet50` dispatch to `FlatSupervisedBaseline`, which owns its timm backbone;
  the encoder slot is filled by `IdentityEncoder` and `model.backbone.*` is never
  read. The inherited `checkpoint_path` value is inert — it is never opened.
- **NO (explicit null)** — `swinv2_supervised` and `imagenet_frozen` *do* use the
  `model.backbone` path but set `checkpoint_path: null, pretrained: true` in
  their own config files. That is deliberate: their entire job is to **not** read
  stage 1.

`swinv2_tiny` is the dangerous one and the reason `END_TO_END_BASELINES` exists.
It is shape-compatible with the published encoder, so a suite that handed it the
stage-1 checkpoint under `checkpoint_strict: false` would load it, log one line,
and quietly turn the "ImageNet SwinV2" arm into a second self-supervised row.
Never pass `model.backbone.checkpoint_path` to it.

And `swinv2_supervised` is **not** the SwinV2 arm of the ViT-vs-SwinV2 pair — it
carries the full hierarchical head, so its gap to a ViT mixes backbone with head.
The decomposition is `swinv2_tiny` (ImageNet trunk, flat head) →
`swinv2_supervised` (ImageNet trunk, full head) → the reference row (stage-1
trunk, full head).

### 3.3 Output destinations

```
$SEED_OUTPUT_DIR/ablations/<variant>/seed42/      # variants 1–5
$SEED_OUTPUT_DIR/baselines/<variant>/seed42/      # variants 6–11
$SEED_OUTPUT_DIR/baselines/imagenet_frozen/seed42/  # variant 12 (see note)
$SEED_OUTPUT_DIR/finetune_hierarchical_moe/       # variant 0, already present
```

Each directory is self-contained: Hydra config snapshot, logs, TensorBoard
events, checkpoints, `summary.json`, `test_predictions.npz`.

> **Note on `imagenet_frozen`.** `VariantSpec.group_directory` is
> `"ablations" if group == "ablation" else "baselines"`, so `group: control`
> lands under `baselines/`, not `controls/`. The docstring in
> `scripts/run_baselines.py` that says `outputs/controls/imagenet_frozen/` is
> stale. Either destination works — `generate_plots.py` scans `ablations`,
> `baselines`, `controls` and `finetune_hierarchical_moe` by default — but pick
> one and be consistent, because the runner and a hand-written command must not
> disagree or the suite will re-run it.

---

## 4. Runtime sizing

### 4.1 What is measured and what is derived

**Measured**, from `outputs/finetune_hierarchical_moe/summary.json`:

| Quantity | Value |
| --- | --- |
| Wall clock, full MoE, 100 epochs, T4×2 | **3.13 h** *(your measurement)* |
| Total / active / trainable parameters | 37.73 M / 33.78 M / 10.15 M |
| GFLOPs per sample | 13.55 |
| Peak memory | 1,417 MB (of 16 GB — VRAM is not the constraint) |
| Inference latency | 21.16 ms @ bs 1, 9.95 ms/sample @ bs 8 |

**Derived**, from the code rather than from timing runs:

| Arm class | Estimate | Reasoning |
| --- | --- | --- |
| Frozen-trunk ablation (`wo_*`) | **~3.0–3.2 h** | Identical trunk forward under `no_grad`; backward is head-only. Head differences (one dense block vs Top-2 MoE, attention vs skip) move a small share of a trunk-dominated step. `wo_moe` and `wo_cross_attn` sit at the low end. |
| `linear_probe` | **~2.9 h** | Same trunk forward, no feature cache (§0.2), two `nn.Linear` layers instead of the hierarchical head. |
| `imagenet_frozen` | **~3.1 h** | Same as an ablation — frozen trunk, full head. |
| End-to-end unfrozen | **~4.5–5.5 h** | *Your estimate, adopted.* Backward now traverses the trunk; trunk parameter counts are comparable (measured with `num_classes=0`: ResNet-50 23.51 M, ConvNeXt-T 27.82 M, ViT-S/16 21.67 M, SwinV2-T 27.58 M), so treat them as one class. |

Headroom note: peak memory on the reference run was 1.4 GB. Nothing here is
close to the T4's 16 GB, so an OOM in these sessions means a misconfiguration,
not a capacity limit.

### 4.2 Price it on the actual machine before committing

```bash
%%bash
cd /kaggle/working/seed-moe-classifier
export SEED_DATA_ROOT=/kaggle/input/datasets/jgfreak/refined-samples/refined_samples/Refined_Samples
export SEED_OUTPUT_DIR=/kaggle/working/outputs
export SEED_PRETRAIN_BACKBONE=/kaggle/input/datasets/jgfreak/refined-samples/dino_backbone_epoch_0020.pth

python scripts/estimate_suite_cost.py --gpus 2 --seeds 42 \
    --arms wo_moe wo_margin_only wo_residual wo_cross_attn wo_kl \
           linear_probe vit_small swinv2_tiny convnext_tiny resnet50 \
           swinv2_supervised imagenet_frozen
```

This times a handful of real training steps per arm shape on *this* T4 and
extrapolates over 13,492 samples × 100 epochs. It costs a few minutes and
replaces every estimate in §4.1 with a measurement. Run it in Session 1 and
re-plan §5 from its output.

### 4.3 The epoch budget is the one lever that actually changes the totals ⚠️

From [`MODEL_EVALUATION_REPORT.md`](MODEL_EVALUATION_REPORT.md) §6.4, measured on
the shipped run:

> **100 epochs is roughly 90 more than this configuration needs.** Accuracy was
> within 0.15 pp of final by epoch 6 and never significantly exceeded it. The
> remaining epochs bought calibration (ECE 0.138 → 0.077) […] but if wall clock
> matters, this recipe converges early.

So `experiment.training.epochs=40` would take the twelve-arm total from ~46 h to
~18 h — two sessions instead of five.

**Two conditions, both hard:**

1. It must apply to **every** arm. A table mixing 40-epoch and 100-epoch rows
   measures the schedule as much as the component.
2. It therefore requires re-running the reference at 40 epochs too (~1.25 h),
   because the published 81.73 % row is a 100-epoch run. The headline number in
   the manuscript would then no longer be the reference the ablation table is
   paired against.

**Recommendation: keep 100 epochs.** Comparability with the published row is
worth more than three Kaggle sessions, and the ECE cost is real. Take the
40-epoch route only if compute forces it, and if you do, state the schedule
beside the table.

---

## 5. Session packing

### 5.1 The requested grouping, priced

Shown with the arithmetic so the overruns are visible:

| Requested session | Arms | Est. total | Verdict |
| --- | --- | --- | --- |
| 1 | `wo_moe` + `wo_margin_only` + `wo_residual` + `linear_probe` | 3.1+3.1+3.1+**2.9** = **12.2 h** | ✗ over the 12 h hard limit |
| 2 | `wo_cross_attn` + `wo_kl` + `vit_small` | 3.1+3.1+**5.0** = **11.2 h** | ✗ over the 10.5 h working budget |
| 3 | `convnext_tiny` + `swinv2_tiny` + `resnet50` | 5.0+5.0+5.0 = **15.0 h** | ✗✗ well over |

The gap is §0.2 (`linear_probe` is not 1.5 h) and three end-to-end runs not
fitting in one session at ~5 h each.

### 5.2 The corrected plan — five sessions

Working budget **10.5 h** (12 h hard, minus ~1 h safety and ~0.5 h for install,
dataset mount and the final artifact writes).

| Session | Arms | Est. | Internet | Checkpoint needed |
| --- | --- | --- | --- | --- |
| **1** | `wo_moe`, `wo_margin_only`, `wo_residual` | ~9.3 h | off | YES |
| **2** | `wo_cross_attn`, `wo_kl`, `linear_probe` | ~9.1 h | off | YES |
| **3** | `vit_small`, `swinv2_tiny` | ~10.0 h | **on** | no |
| **4** | `convnext_tiny`, `resnet50` | ~10.0 h | **on** | no |
| **5** | `swinv2_supervised`, `imagenet_frozen` | ~8.1 h | **on** | no |
|  | **Total** | **~46.5 GPU-h** |  |  |

Sessions 1–2 need no network at all, which also makes them the most robust ones
to run first. Session 5 carries the recommended twelfth arm from §3.1; drop
`imagenet_frozen` if you must, and Session 5 becomes ~5 h.

### 5.3 Perfect packing is not actually required

This is the point that makes the whole plan robust. With
`experiment.training.resume=auto` plus a per-run
`experiment.training.max_runtime_minutes`, a run that will not fit **stops itself
cleanly at an epoch boundary with a complete resume checkpoint** and the
identical command line continues it next session.

The mechanism, and why it is trustworthy (`src/trainers/moe_finetune.py:1962`
onward):

- the wall-clock budget is latched locally but the stop decision is **all-reduced
  across ranks** before acting — a rank deciding alone would enter the RNG gather
  (a collective) without its peer and hang at the timeout;
- an interrupted run `return`s at line 2050, **before** the test evaluation, so
  it writes **no** `summary.json` and **no** `test_predictions.npz`;
- consequently `inspect_run` (`src/trainers/suite.py:156`) classifies it as
  incomplete from the artifacts alone and the next launch resumes it.

Stage 2 resumes at **epoch** granularity (fold, epoch, global step, optimizer
moments, best-so-far, RNG), so at most one epoch of work is lost.

**Rule for setting `max_runtime_minutes` on the last run of a session:**

```
max_runtime_minutes = (minutes remaining in the 12 h session) − 25
```

The 25 minutes covers the final test evaluation, the efficiency sweep
(`batch_sizes: [1, 8, 32]`, 50 timed iterations each) and the artifact writes.
Being killed *inside* those is what produces the half-written state the suite has
to detect and redo.

---

## 6. Persistence between sessions

`/kaggle/working` does not survive a session. Carry `outputs/` forward as a
Kaggle dataset.

### 6.1 Prune before saving

A completed stage-2 run is ~880 MB, of which the report needs ~3.4 MB:

```
144M  best_hierarchical_moe.pth
144M  hierarchical_moe_final.pth
144M  model_fold1_epoch0100.pth
222M  finetune_resume_fold1_epoch0084.pth     <- resume state, dead once complete
222M  finetune_resume_fold1_epoch0100.pth     <- resume state, dead once complete
3.4M  test_predictions.npz                    <- REQUIRED
16K   summary.json                            <- REQUIRED
37K   split_manifest.npz
```

Completion is decided by `summary.json` + `test_predictions.npz` **only**
(`inspect_run` opens both, parses the JSON, and actually loads the `.npz` arrays
rather than stat'ing the file — a kill during the `.npz` write leaves a file that
exists and raises on read). So pruning `*.pth` from **completed** runs is safe
and takes the carry-over from ~10 GB to ~40 MB.

```bash
%%bash
cd /kaggle/working/outputs
python - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, "/kaggle/working/seed-moe-classifier")
from src.trainers.suite import inspect_run

freed = 0
for summary in Path(".").rglob("summary.json"):
    run = summary.parent
    result = inspect_run(run)
    if not result.complete:
        print(f"KEEP (incomplete: {result.reason}) {run}")
        continue
    for checkpoint in run.glob("*.pth"):
        freed += checkpoint.stat().st_size
        checkpoint.unlink()
    print(f"pruned {run}")
print(f"freed {freed / 1e9:.2f} GB")
PY

du -sh /kaggle/working/outputs
```

Incomplete runs keep their `.pth` files — that is exactly the resume state the
next session needs.

### 6.2 Save and re-mount

1. Notebook sidebar → **Output** → **Create Dataset** (name it e.g.
   `seed-moe-outputs`). On later sessions use **New Version**.
2. In the next session, add that dataset as an input and restore before doing
   anything else:

```bash
%%bash
mkdir -p /kaggle/working/outputs
cp -a /kaggle/input/seed-moe-outputs/outputs/. /kaggle/working/outputs/ 2>/dev/null || true
find /kaggle/working/outputs -name summary.json | wc -l   # runs carried forward
```

`cp -a`, not a symlink: the trainer writes into this tree, and `/kaggle/input`
is read-only.

### 6.3 What is cached and what is not

- **Cached / skipped:** any run whose `summary.json` and `test_predictions.npz`
  are both present and mutually consistent. `run_experiments_suite.py` skips it;
  a hand-run command would overwrite it, so check §7.0 first.
- **Resumed:** any run with a `finetune_resume_*.pth` and no summary.
- **Not cached:** `outputs/suite_status.json` is a *cache*, not the truth.
  Deleting it loses attempt counts and timings and nothing that decides what gets
  launched.

---

## 7. The commands, one experiment at a time

### 7.0 Command anatomy

Every command below has the same shape, mirroring `build_command` in
`src/trainers/runner.py:177` with DDP added:

```bash
python scripts/launch.py finetune --gpus 2 \
    experiment=<EXPERIMENT CONFIG> \
    experiment.variant=<VARIANT NAME> \
    experiment.group=<ablation|baseline|control> \
    experiment.training.save_path=$SEED_OUTPUT_DIR/<GROUP DIR>/<VARIANT>/seed42 \
    hydra.run.dir=$SEED_OUTPUT_DIR/<GROUP DIR>/<VARIANT>/seed42/hydra \
    <SINGLE-FACTOR OVERRIDES> \
    seed=42 \
    [model.backbone.checkpoint_path=$SEED_PRETRAIN_BACKBONE] \
    data.batch_size=8 data.num_workers=2 \
    experiment.training.amp=fp16 \
    experiment.training.resume=auto \
    experiment.training.max_runtime_minutes=<BUDGET>
```

Four things about it:

- **`scripts/launch.py finetune` injects `experiment=finetune_hierarchical_moe`
  itself**, so every command carries a second `experiment=` override. Hydra takes
  the rightmost value (verified by composing both orders), so this is harmless
  for ablations and correct for baselines. Writing it explicitly in all twelve
  commands keeps them uniform and copy-paste-safe.
- **`experiment.variant` is mandatory for the ablation arms.**
  `finetune_hierarchical_moe.yaml` has `variant: null`, and the trainer then
  names the run after the experiment. Omit it on five ablations and all five rows
  collide under `finetune_hierarchical_moe` in `summary_metrics.csv`. The
  `baseline_*.yaml` files set `variant` and `group` themselves, so those
  overrides are redundant there — but harmless, and uniformity is worth more.
- **`hydra.run.dir` keeps logs, config snapshot, TensorBoard events and figures
  inside the variant's own folder** instead of the global timestamped tree, so
  one variant is one self-contained directory.
- **`launch.py` pins `SEED_RUN_ID` before the ranks exist**, so both ranks
  compose the same output directory. (The trainers also broadcast rank 0's
  resolved directory, so a bare `torchrun` is correct too — this just makes it
  tidy as well.)

Before launching a run by hand into a directory that may already hold a finished
run:

```bash
python -c "
import sys; sys.path.insert(0,'/kaggle/working/seed-moe-classifier')
from src.trainers.suite import inspect_run
print(inspect_run('$SEED_OUTPUT_DIR/ablations/wo_moe/seed42'))"
```

---

### SESSION 1 — Core architectural ablations (~9.3 h, no Internet needed)

Every command in this session reads `$SEED_PRETRAIN_BACKBONE`. Prefix each cell
with the §2.3 environment block (shown in full for the first, abbreviated after).

#### 1.1 — `wo_moe` (~3.1 h)

No sparse routing: one dense block. Note this is **not** capacity-matched — it
changes routing, active capacity and the two MoE regularisers at once (both
evaluate to exactly zero on a degenerate one-expert gate), so it optimises a
strictly smaller objective than the reference. Kept as the historical
comparison.

```bash
%%bash
set -euo pipefail
cd /kaggle/working/seed-moe-classifier
export SEED_DATA_ROOT=/kaggle/input/datasets/jgfreak/refined-samples/refined_samples/Refined_Samples
export SEED_OUTPUT_DIR=/kaggle/working/outputs
export SEED_PRETRAIN_BACKBONE=/kaggle/input/datasets/jgfreak/refined-samples/dino_backbone_epoch_0020.pth

python scripts/launch.py finetune --gpus 2 \
    experiment=finetune_hierarchical_moe \
    experiment.variant=wo_moe \
    experiment.group=ablation \
    experiment.training.save_path=$SEED_OUTPUT_DIR/ablations/wo_moe/seed42 \
    hydra.run.dir=$SEED_OUTPUT_DIR/ablations/wo_moe/seed42/hydra \
    model.head.use_moe=false \
    seed=42 \
    model.backbone.checkpoint_path=$SEED_PRETRAIN_BACKBONE \
    data.batch_size=8 data.num_workers=2 \
    experiment.training.amp=fp16 \
    experiment.training.resume=auto \
    experiment.training.max_runtime_minutes=220
```

**Artifacts →** `/kaggle/working/outputs/ablations/wo_moe/seed42/`

#### 1.2 — `wo_margin_only` (~3.1 h)

NormFace: L2-normalised embedding and centres kept, **margin removed**. This is
the true single-factor margin control. Do not confuse it with `wo_angular_head`
(`sub_head_variant=linear`), which removes margin *and* normalisation *and* the
logit scale — four factors — and is what the submitted suite mislabelled
`wo_arcface`.

```bash
%%bash
set -euo pipefail
cd /kaggle/working/seed-moe-classifier
export SEED_DATA_ROOT=/kaggle/input/datasets/jgfreak/refined-samples/refined_samples/Refined_Samples
export SEED_OUTPUT_DIR=/kaggle/working/outputs
export SEED_PRETRAIN_BACKBONE=/kaggle/input/datasets/jgfreak/refined-samples/dino_backbone_epoch_0020.pth

python scripts/launch.py finetune --gpus 2 \
    experiment=finetune_hierarchical_moe \
    experiment.variant=wo_margin_only \
    experiment.group=ablation \
    experiment.training.save_path=$SEED_OUTPUT_DIR/ablations/wo_margin_only/seed42 \
    hydra.run.dir=$SEED_OUTPUT_DIR/ablations/wo_margin_only/seed42/hydra \
    model.head.sub_head_variant=normface \
    seed=42 \
    model.backbone.checkpoint_path=$SEED_PRETRAIN_BACKBONE \
    data.batch_size=8 data.num_workers=2 \
    experiment.training.amp=fp16 \
    experiment.training.resume=auto \
    experiment.training.max_runtime_minutes=220
```

**Artifacts →** `/kaggle/working/outputs/ablations/wo_margin_only/seed42/`

#### 1.3 — `wo_residual` (~3.1 h)

No seed-type fusion (Eq. 9). The residual magnitude term goes with it.

```bash
%%bash
set -euo pipefail
cd /kaggle/working/seed-moe-classifier
export SEED_DATA_ROOT=/kaggle/input/datasets/jgfreak/refined-samples/refined_samples/Refined_Samples
export SEED_OUTPUT_DIR=/kaggle/working/outputs
export SEED_PRETRAIN_BACKBONE=/kaggle/input/datasets/jgfreak/refined-samples/dino_backbone_epoch_0020.pth

python scripts/launch.py finetune --gpus 2 \
    experiment=finetune_hierarchical_moe \
    experiment.variant=wo_residual \
    experiment.group=ablation \
    experiment.training.save_path=$SEED_OUTPUT_DIR/ablations/wo_residual/seed42 \
    hydra.run.dir=$SEED_OUTPUT_DIR/ablations/wo_residual/seed42/hydra \
    model.head.use_residual=false \
    seed=42 \
    model.backbone.checkpoint_path=$SEED_PRETRAIN_BACKBONE \
    data.batch_size=8 data.num_workers=2 \
    experiment.training.amp=fp16 \
    experiment.training.resume=auto \
    experiment.training.max_runtime_minutes=200
```

**Artifacts →** `/kaggle/working/outputs/ablations/wo_residual/seed42/`

*End of session: run §6.1 (prune), then create the dataset version.*

---

### SESSION 2 — Refinement ablations and the probe (~9.1 h, no Internet needed)

Restore the carry-over dataset (§6.2) first.

#### 2.1 — `wo_cross_attn` (~3.1 h)

No Q/K/V refinement: `h'' = h'`, Eqs. 11–12 skipped.

```bash
%%bash
set -euo pipefail
cd /kaggle/working/seed-moe-classifier
export SEED_DATA_ROOT=/kaggle/input/datasets/jgfreak/refined-samples/refined_samples/Refined_Samples
export SEED_OUTPUT_DIR=/kaggle/working/outputs
export SEED_PRETRAIN_BACKBONE=/kaggle/input/datasets/jgfreak/refined-samples/dino_backbone_epoch_0020.pth

python scripts/launch.py finetune --gpus 2 \
    experiment=finetune_hierarchical_moe \
    experiment.variant=wo_cross_attn \
    experiment.group=ablation \
    experiment.training.save_path=$SEED_OUTPUT_DIR/ablations/wo_cross_attn/seed42 \
    hydra.run.dir=$SEED_OUTPUT_DIR/ablations/wo_cross_attn/seed42/hydra \
    model.head.use_cross_attention=false \
    seed=42 \
    model.backbone.checkpoint_path=$SEED_PRETRAIN_BACKBONE \
    data.batch_size=8 data.num_workers=2 \
    experiment.training.amp=fp16 \
    experiment.training.resume=auto \
    experiment.training.max_runtime_minutes=220
```

**Artifacts →** `/kaggle/working/outputs/ablations/wo_cross_attn/seed42/`

#### 2.2 — `wo_kl` (~3.1 h)

No stage-1/stage-2 KL hierarchy-consistency loss (Eq. 10). `use_kl_loss` lives
on the head but is consumed by the loss, reached via
`use_kl_loss: ${model.head.use_kl_loss}` — so this one override moves both.

```bash
%%bash
set -euo pipefail
cd /kaggle/working/seed-moe-classifier
export SEED_DATA_ROOT=/kaggle/input/datasets/jgfreak/refined-samples/refined_samples/Refined_Samples
export SEED_OUTPUT_DIR=/kaggle/working/outputs
export SEED_PRETRAIN_BACKBONE=/kaggle/input/datasets/jgfreak/refined-samples/dino_backbone_epoch_0020.pth

python scripts/launch.py finetune --gpus 2 \
    experiment=finetune_hierarchical_moe \
    experiment.variant=wo_kl \
    experiment.group=ablation \
    experiment.training.save_path=$SEED_OUTPUT_DIR/ablations/wo_kl/seed42 \
    hydra.run.dir=$SEED_OUTPUT_DIR/ablations/wo_kl/seed42/hydra \
    model.head.use_kl_loss=false \
    seed=42 \
    model.backbone.checkpoint_path=$SEED_PRETRAIN_BACKBONE \
    data.batch_size=8 data.num_workers=2 \
    experiment.training.amp=fp16 \
    experiment.training.resume=auto \
    experiment.training.max_runtime_minutes=220
```

**Artifacts →** `/kaggle/working/outputs/ablations/wo_kl/seed42/`

#### 2.3 — `linear_probe` (~2.9 h)

Frozen self-supervised encoder plus two linear heads, plain CE. The cheapest
possible read of what stage 1 alone learned, with no head machinery to credit or
blame. Its `learning_rate: 0.001` and its `linear_probe` head + `flat_cce` loss
come from the experiment file — **do not override them**.

`build_model_and_encoder` forces `token_mode="pooled"` for this head, so the
probe sees the pooled trunk feature rather than the 8×8 grid.

```bash
%%bash
set -euo pipefail
cd /kaggle/working/seed-moe-classifier
export SEED_DATA_ROOT=/kaggle/input/datasets/jgfreak/refined-samples/refined_samples/Refined_Samples
export SEED_OUTPUT_DIR=/kaggle/working/outputs
export SEED_PRETRAIN_BACKBONE=/kaggle/input/datasets/jgfreak/refined-samples/dino_backbone_epoch_0020.pth

python scripts/launch.py finetune --gpus 2 \
    experiment=baseline_linear_probe \
    experiment.variant=linear_probe \
    experiment.group=baseline \
    experiment.training.save_path=$SEED_OUTPUT_DIR/baselines/linear_probe/seed42 \
    hydra.run.dir=$SEED_OUTPUT_DIR/baselines/linear_probe/seed42/hydra \
    seed=42 \
    model.backbone.checkpoint_path=$SEED_PRETRAIN_BACKBONE \
    data.batch_size=8 data.num_workers=2 \
    experiment.training.amp=fp16 \
    experiment.training.resume=auto \
    experiment.training.max_runtime_minutes=200
```

**Artifacts →** `/kaggle/working/outputs/baselines/linear_probe/seed42/`

*End of session: prune, save dataset version. Sessions 1–2 are now done and the
five ablation rows plus the probe exist.*

---

### SESSION 3 — Vision transformers (~10.0 h, **Internet on**)

No `model.backbone.checkpoint_path` in any command from here on. These arms own
their backbones; handing them the stage-1 checkpoint is meaningless for a ViT and
actively wrong for `swinv2_tiny` (§3.2).

#### 3.1 — `vit_small` (~4.5–5.5 h)

ImageNet ViT-S/**16** at 256 px behind the flat two-head classifier.
`backbone_kwargs: {img_size: 256}` comes from the experiment file and is what
makes timm interpolate the position embedding — that is what holds input
resolution fixed across the backbone comparison, on a corpus whose crops are a
median 61×61 px and are therefore upsampled ~4×.

ViT-S/**16**, not /14: timm ships no supervised ImageNet-1k ViT-S/14; /14 is
DINOv2's patch size and those weights are self-supervised, which would move the
pretraining axis this row holds fixed. Say S/16 in the manuscript. See §0.5 for
the IN-21k tag caveat.

```bash
%%bash
set -euo pipefail
cd /kaggle/working/seed-moe-classifier
export SEED_DATA_ROOT=/kaggle/input/datasets/jgfreak/refined-samples/refined_samples/Refined_Samples
export SEED_OUTPUT_DIR=/kaggle/working/outputs
# NO SEED_PRETRAIN_BACKBONE — this arm must not read stage 1.

python scripts/launch.py finetune --gpus 2 \
    experiment=baseline_vit_small \
    experiment.variant=vit_small \
    experiment.group=baseline \
    experiment.training.save_path=$SEED_OUTPUT_DIR/baselines/vit_small/seed42 \
    hydra.run.dir=$SEED_OUTPUT_DIR/baselines/vit_small/seed42/hydra \
    seed=42 \
    data.batch_size=8 data.num_workers=2 \
    experiment.training.amp=fp16 \
    experiment.training.resume=auto \
    experiment.training.max_runtime_minutes=330
```

**Artifacts →** `/kaggle/working/outputs/baselines/vit_small/seed42/`
**Checkpoint:** none — timm downloads `timm/vit_small_patch16_224.augreg_in21k_ft_in1k`.

#### 3.2 — `swinv2_tiny` (~4.5–5.5 h)

The matched SwinV2 arm of the ViT-vs-SwinV2 pair: same flat head, same 256 px
input, same ImageNet-1k supervised initialisation, same optimiser, schedule,
split and seed. The trunk is the only difference.

```bash
%%bash
set -euo pipefail
cd /kaggle/working/seed-moe-classifier
export SEED_DATA_ROOT=/kaggle/input/datasets/jgfreak/refined-samples/refined_samples/Refined_Samples
export SEED_OUTPUT_DIR=/kaggle/working/outputs
# NO SEED_PRETRAIN_BACKBONE — shape-compatible with the stage-1 encoder, so
# passing it would silently load it and produce a second self-supervised row.

python scripts/launch.py finetune --gpus 2 \
    experiment=baseline_swinv2_tiny \
    experiment.variant=swinv2_tiny \
    experiment.group=baseline \
    experiment.training.save_path=$SEED_OUTPUT_DIR/baselines/swinv2_tiny/seed42 \
    hydra.run.dir=$SEED_OUTPUT_DIR/baselines/swinv2_tiny/seed42/hydra \
    seed=42 \
    data.batch_size=8 data.num_workers=2 \
    experiment.training.amp=fp16 \
    experiment.training.resume=auto \
    experiment.training.max_runtime_minutes=330
```

**Artifacts →** `/kaggle/working/outputs/baselines/swinv2_tiny/seed42/`
**Checkpoint:** none — timm downloads `timm/swinv2_tiny_window16_256.ms_in1k`.

---

### SESSION 4 — CNN baselines (~10.0 h, **Internet on**)

#### 4.1 — `convnext_tiny` (~4.5–5.5 h)

Modern CNN, parameter-matched to the proposed trunk (27.82 M vs 27.58 M, both measured), so its
gap is not a capacity gap the way ResNet-50's could be argued to be. See §0.5 for
the IN-12k tag caveat.

```bash
%%bash
set -euo pipefail
cd /kaggle/working/seed-moe-classifier
export SEED_DATA_ROOT=/kaggle/input/datasets/jgfreak/refined-samples/refined_samples/Refined_Samples
export SEED_OUTPUT_DIR=/kaggle/working/outputs
# NO SEED_PRETRAIN_BACKBONE.

python scripts/launch.py finetune --gpus 2 \
    experiment=baseline_convnext_tiny \
    experiment.variant=convnext_tiny \
    experiment.group=baseline \
    experiment.training.save_path=$SEED_OUTPUT_DIR/baselines/convnext_tiny/seed42 \
    hydra.run.dir=$SEED_OUTPUT_DIR/baselines/convnext_tiny/seed42/hydra \
    seed=42 \
    data.batch_size=8 data.num_workers=2 \
    experiment.training.amp=fp16 \
    experiment.training.resume=auto \
    experiment.training.max_runtime_minutes=330
```

**Artifacts →** `/kaggle/working/outputs/baselines/convnext_tiny/seed42/`
**Checkpoint:** none — timm downloads `timm/convnext_tiny.in12k_ft_in1k`.

#### 4.2 — `resnet50` (~4.5–5.5 h)

The conventional CNN comparison. This is the one BatchNorm trunk in the set — see
§1.4 on per-rank BN under 2×8 DDP.

```bash
%%bash
set -euo pipefail
cd /kaggle/working/seed-moe-classifier
export SEED_DATA_ROOT=/kaggle/input/datasets/jgfreak/refined-samples/refined_samples/Refined_Samples
export SEED_OUTPUT_DIR=/kaggle/working/outputs
# NO SEED_PRETRAIN_BACKBONE.

python scripts/launch.py finetune --gpus 2 \
    experiment=baseline_resnet50 \
    experiment.variant=resnet50 \
    experiment.group=baseline \
    experiment.training.save_path=$SEED_OUTPUT_DIR/baselines/resnet50/seed42 \
    hydra.run.dir=$SEED_OUTPUT_DIR/baselines/resnet50/seed42/hydra \
    seed=42 \
    data.batch_size=8 data.num_workers=2 \
    experiment.training.amp=fp16 \
    experiment.training.resume=auto \
    experiment.training.max_runtime_minutes=330
```

**Artifacts →** `/kaggle/working/outputs/baselines/resnet50/seed42/`
**Checkpoint:** none — timm downloads `timm/resnet50.a1_in1k`.

---

### SESSION 5 — The stage-1 decomposition (~8.1 h, **Internet on**)

These two rows are what let you say *what stage 1 bought*, with head, split and
seed held fixed:

```
imagenet_frozen      ImageNet trunk, FROZEN,   full head, no stage 1
swinv2_supervised    ImageNet trunk, UNFROZEN, full head, no stage 1
reference row        stage-1 trunk,  frozen,   full head
```

#### 5.1 — `swinv2_supervised` (~4.5–5.5 h)

The same ImageNet initialisation with the trunk **unfrozen**, still with no
self-supervised stage. Its own config sets `checkpoint_path: null,
pretrained: true, freeze: false`.

```bash
%%bash
set -euo pipefail
cd /kaggle/working/seed-moe-classifier
export SEED_DATA_ROOT=/kaggle/input/datasets/jgfreak/refined-samples/refined_samples/Refined_Samples
export SEED_OUTPUT_DIR=/kaggle/working/outputs
# NO SEED_PRETRAIN_BACKBONE — its entire job is to NOT read stage 1.

python scripts/launch.py finetune --gpus 2 \
    experiment=baseline_swinv2_supervised \
    experiment.variant=swinv2_supervised \
    experiment.group=baseline \
    experiment.training.save_path=$SEED_OUTPUT_DIR/baselines/swinv2_supervised/seed42 \
    hydra.run.dir=$SEED_OUTPUT_DIR/baselines/swinv2_supervised/seed42/hydra \
    seed=42 \
    data.batch_size=8 data.num_workers=2 \
    experiment.training.amp=fp16 \
    experiment.training.resume=auto \
    experiment.training.max_runtime_minutes=330
```

**Artifacts →** `/kaggle/working/outputs/baselines/swinv2_supervised/seed42/`
**Checkpoint:** none — timm downloads `timm/swinv2_tiny_window16_256.ms_in1k`.

#### 5.2 — `imagenet_frozen` (~3.1 h) — *recommended addition*

```bash
%%bash
set -euo pipefail
cd /kaggle/working/seed-moe-classifier
export SEED_DATA_ROOT=/kaggle/input/datasets/jgfreak/refined-samples/refined_samples/Refined_Samples
export SEED_OUTPUT_DIR=/kaggle/working/outputs
# NO SEED_PRETRAIN_BACKBONE.

python scripts/launch.py finetune --gpus 2 \
    experiment=control_imagenet_frozen \
    experiment.variant=imagenet_frozen \
    experiment.group=control \
    experiment.training.save_path=$SEED_OUTPUT_DIR/baselines/imagenet_frozen/seed42 \
    hydra.run.dir=$SEED_OUTPUT_DIR/baselines/imagenet_frozen/seed42/hydra \
    seed=42 \
    data.batch_size=8 data.num_workers=2 \
    experiment.training.amp=fp16 \
    experiment.training.resume=auto \
    experiment.training.max_runtime_minutes=200
```

**Artifacts →** `/kaggle/working/outputs/baselines/imagenet_frozen/seed42/`
(see the §3.3 note on why `group: control` still lands under `baselines/`).

#### 5.3 — *Optional* `full_model` (~3.1 h)

Only if you chose Option B in §0.3 rather than repointing `--reference`.

```bash
%%bash
set -euo pipefail
cd /kaggle/working/seed-moe-classifier
export SEED_DATA_ROOT=/kaggle/input/datasets/jgfreak/refined-samples/refined_samples/Refined_Samples
export SEED_OUTPUT_DIR=/kaggle/working/outputs
export SEED_PRETRAIN_BACKBONE=/kaggle/input/datasets/jgfreak/refined-samples/dino_backbone_epoch_0020.pth

python scripts/launch.py finetune --gpus 2 \
    experiment=finetune_hierarchical_moe \
    experiment.variant=full_model \
    experiment.group=ablation \
    experiment.training.save_path=$SEED_OUTPUT_DIR/ablations/full_model/seed42 \
    hydra.run.dir=$SEED_OUTPUT_DIR/ablations/full_model/seed42/hydra \
    seed=42 \
    model.backbone.checkpoint_path=$SEED_PRETRAIN_BACKBONE \
    data.batch_size=8 data.num_workers=2 \
    experiment.training.amp=fp16 \
    experiment.training.resume=auto \
    experiment.training.max_runtime_minutes=200
```

---

## 8. The resumable alternative — one command per session

Everything in §7 can be replaced by a single command that resolves the same
overrides, skips what is finished, resumes what is not, and stops **between**
runs before the session limit. On a preemptible platform this is the more robust
path; §7 exists for granular control and for re-running one arm.

```bash
%%bash
set -euo pipefail
cd /kaggle/working/seed-moe-classifier
export SEED_DATA_ROOT=/kaggle/input/datasets/jgfreak/refined-samples/refined_samples/Refined_Samples
export SEED_OUTPUT_DIR=/kaggle/working/outputs
export SEED_PRETRAIN_BACKBONE=/kaggle/input/datasets/jgfreak/refined-samples/dino_backbone_epoch_0020.pth

# See exactly what would run, and where, before spending anything:
python scripts/run_experiments_suite.py --seeds 42 --dry-run \
    --variants wo_moe wo_margin_only wo_residual wo_cross_attn wo_kl \
               linear_probe vit_small swinv2_tiny convnext_tiny resnet50 \
               swinv2_supervised imagenet_frozen

# The real thing. Relaunch this IDENTICAL line every session.
python scripts/run_experiments_suite.py --seeds 42 --kaggle \
    --variants wo_moe wo_margin_only wo_residual wo_cross_attn wo_kl \
               linear_probe vit_small swinv2_tiny convnext_tiny resnet50 \
               swinv2_supervised imagenet_frozen \
    -- data.batch_size=16 data.num_workers=2

# Where did it get to?
python scripts/run_experiments_suite.py --seeds 42 --status \
    --variants wo_moe wo_margin_only wo_residual wo_cross_attn wo_kl \
               linear_probe vit_small swinv2_tiny convnext_tiny resnet50 \
               swinv2_supervised imagenet_frozen
```

`--kaggle` sets `--amp fp16`, `--max-session-minutes 690`,
`--max-runtime-minutes 660` and `--gpus 0,1`.

### 8.1 One difference that matters, stated plainly

`--gpus 0,1` in this script means **one arm per device, two arms concurrently,
each a single-GPU process** — it does *not* mean DDP. Each arm then sees one GPU
and must use `data.batch_size=16` to reproduce the reference row's global batch
of 16, which is why the override is appended above.

So the two paths differ in how the global batch is assembled:

| Path | Per-arm processes | `data.batch_size` | Global batch | Matches the 81.73 % row? |
| --- | --- | --- | --- | --- |
| §7 (DDP) | 2 ranks | 8 | 16 | **yes, exactly** |
| §8 (sharded) | 1 | 16 | 16 | same gradient; BN arms differ (§1.4) |

For LayerNorm trunks (SwinV2, ViT, ConvNeXt) the two are equivalent. For
`resnet50`, per-rank BatchNorm over 8 samples is not the same as BatchNorm over
16. **Do not mix the two paths within one table** — pick §7 or §8 and use it for
every arm.

Throughput is close to a wash on Kaggle: the sharded path runs two arms at once
but each does twice the per-GPU work, and with ~4 vCPUs feeding four dataloader
workers across two independent training processes, it tends to become
dataloader-bound. **§7's DDP path is the recommendation here**, because it
reproduces the reference row exactly and leaves no BN question to answer.

### 8.2 What the suite checks that a bare loop does not

- Completion is decided from each run's own `summary.json` and
  `test_predictions.npz`, never from the recorded status. Exit code 0 is
  necessary but not sufficient: `mark_result` re-verifies and records
  `exit 0 but <reason>` as a failure.
- On every launch it groups completed runs by **corpus SHA-256** and by **split
  identity** and prints a loud warning if either disagrees. Two corpora in one
  table is the re-baseline; two splits makes McNemar undefined, not merely
  weaker.
- The shared encoder's SHA-256 is recorded once and compared on relaunch — so
  pointing `SEED_PRETRAIN_BACKBONE` at a different milestone halfway through is
  caught rather than silently absorbed.

---

## 9. Post-run aggregation and table generation

### 9.1 Generate the report

```bash
%%bash
set -euo pipefail
cd /kaggle/working/seed-moe-classifier
export SEED_OUTPUT_DIR=/kaggle/working/outputs

python scripts/generate_plots.py \
    --roots $SEED_OUTPUT_DIR/ablations \
            $SEED_OUTPUT_DIR/baselines \
            $SEED_OUTPUT_DIR/finetune_hierarchical_moe \
    --reference finetune_hierarchical_moe \
    --dpi 300
```

`--reference finetune_hierarchical_moe` is §0.3 Option A. Use
`--reference full_model` instead only if you ran §7 5.3.

Faster iteration while checking the table (seconds, no figures):

```bash
python scripts/generate_plots.py --roots ... --reference finetune_hierarchical_moe --no-figures
```

Nothing here retrains or reloads a model, so re-plotting at a different DPI costs
seconds.

### 9.2 What lands in `outputs/reports/`

| File | Contents |
| --- | --- |
| `summary_metrics.csv` | one row per variant; repeated seeds collapse to mean ± SD; carries `p (vs full)` and `p (Holm)` |
| `summary_metrics_per_run.csv` | un-aggregated, one row per run |
| `{variant}_confusion_seed_type.png` | 4-class matrix, row-normalised |
| `{variant}_confusion_sub_variety.png` | all 27 sub-varieties, unabbreviated tick labels |
| `{variant}_tsne_seed_type.png`, `_tsne_sub_variety.png` | 384-D test embeddings → 2-D |
| `{variant}_loss_curves.png` | training vs validation loss |
| per-class metric heatmaps, misclassification rates, expert utilisation | |

`summary_metrics.csv` columns, in order
(`REQUESTED_COLUMNS + EXTRA_COLUMNS`, `src/utils/evaluation.py:51`):

```
Model/Variant, Accuracy, Precision, Recall, Macro F1, Micro F1,
KL Alignment Rate (%), Total Params (M), Active Params (M),
Inference Latency (ms), Seeds, p (vs full), p (Holm),
Seed-Type Accuracy, Seed-Type Macro F1, Sub-Variety AUC (macro OvR), ECE,
Expert NMI (sub-variety), Dead Experts, Experts, Top-K,
Throughput (FPS), GFLOPs/sample, Peak Memory (MB),
Split Protocol, Group, Run Directory
```

That covers the accuracy, macro-F1, parameter-count and FLOP columns the
manuscript table needs, plus the efficiency and routing diagnostics.

**Blank cells mean "not measured", never "failed".** The writer emits an empty
string rather than `nan` precisely so a reader cannot mistake one for the other.

### 9.3 Verify the table before quoting it

```bash
%%bash
cd /kaggle/working/seed-moe-classifier
export SEED_OUTPUT_DIR=/kaggle/working/outputs

python - <<'PY'
import csv, json, os
from pathlib import Path

root = Path(os.environ["SEED_OUTPUT_DIR"])
report = root / "reports" / "summary_metrics.csv"

expected = {
    "wo_moe", "wo_margin_only", "wo_residual", "wo_cross_attn", "wo_kl",
    "linear_probe", "vit_small", "swinv2_tiny", "convnext_tiny", "resnet50",
    "swinv2_supervised", "imagenet_frozen", "finetune_hierarchical_moe",
}

rows = list(csv.DictReader(report.open(encoding="utf-8")))
present = {r["Model/Variant"] for r in rows}
print(f"{len(rows)} rows in {report}")
missing = expected - present
if missing:
    print("MISSING:", ", ".join(sorted(missing)))
extra = present - expected
if extra:
    print("unexpected rows:", ", ".join(sorted(extra)))

print(f"\n{'variant':<28}{'acc':>8}{'macroF1':>10}{'p(Holm)':>10}{'protocol':>14}")
for r in sorted(rows, key=lambda r: r["Model/Variant"]):
    print(f"{r['Model/Variant']:<28}{r['Accuracy']:>8}{r['Macro F1']:>10}"
          f"{r.get('p (Holm)', ''):>10}{r.get('Split Protocol', ''):>14}")

# Provenance: one corpus, one split, or the table is not internally comparable.
corpora, splits = set(), set()
for summary in root.rglob("summary.json"):
    data = json.loads(summary.read_text(encoding="utf-8"))
    split = data.get("split") or {}
    corpus = (split.get("corpus") or {}).get("sha256")
    if corpus:
        corpora.add(corpus)
    splits.add((split.get("protocol"), split.get("num_samples"),
                split.get("test_size"), split.get("seed")))
print("\ncorpus digests :", {c[:16] for c in corpora} or "none")
print("split identities:", splits)
if len(corpora) > 1:
    print("!!! TWO CORPORA — rows from different corpora are not comparable.")
if len(splits) > 1:
    print("!!! TWO SPLITS — McNemar's paired test is invalid across these rows.")
PY
```

Expected: **one** corpus digest (`013c04c5…`) and **one** split identity
(`('stratified', 13492, 2699, 42)`). Either check failing invalidates the table
rather than degrading it — fix it before writing anything up.

### 9.4 Save the report

Copy `outputs/reports/` into the carry-over dataset (§6.2) or download it from
the notebook's Output panel. It is a few MB.

---

## 10. Failure modes and what they mean

| Symptom | Cause | Fix |
| --- | --- | --- |
| `FileNotFoundError: Backbone checkpoint not found` | `SEED_PRETRAIN_BACKBONE` unset or wrong, on a **YES** arm | re-export it (it does not survive between `%%bash` cells) |
| All five ablation rows collapse into one `finetune_hierarchical_moe` row | `experiment.variant` omitted | §7.0 — it is mandatory for ablation arms |
| `p (vs full)` / `p (Holm)` blank for every row | reference variant not present | §0.3 — pass `--reference finetune_hierarchical_moe` |
| End-to-end baseline dies at model construction | Kaggle Internet off, timm cannot fetch weights | turn Internet on, or pre-warm `HF_HOME` (§2.4) |
| `swinv2_tiny` scores suspiciously like the proposed model | it was handed the stage-1 checkpoint and silently loaded it (shape-compatible) | remove `model.backbone.checkpoint_path`; check the load report in the run log |
| Suite reports a run done but it is absent from the table | killed between the `summary.json` and `.npz` writes | nothing to do — `inspect_run` catches it and the next launch re-runs it |
| Job hangs with no output, no error | a collective entered by one rank only | almost always a hand-edited stop/save path; use the §7 command shape unmodified |
| `/kaggle/working` full | ~880 MB per completed run | §6.1 prune |
| "missing/unexpected keys" in the log | `checkpoint_strict: false` is the default and reports rather than raises | **check this line first when metrics look wrong** — a near-total mismatch means a wrong checkpoint |
| Two corpus digests in §9.3 | a run used a different corpus mount | that row is a re-baseline, not a comparison; re-run it |

### 10.1 Two-minute plumbing check before the first real run

```bash
%%bash
set -euo pipefail
cd /kaggle/working/seed-moe-classifier
export SEED_DATA_ROOT=/kaggle/input/datasets/jgfreak/refined-samples/refined_samples/Refined_Samples
export SEED_OUTPUT_DIR=/kaggle/working/outputs
export SEED_PRETRAIN_BACKBONE=/kaggle/input/datasets/jgfreak/refined-samples/dino_backbone_epoch_0020.pth

# Numerical checks on THIS machine (SDPA parity, AMP resolution, device report).
python scripts/verify_runtime.py --gpus 2

# 2-batch, 1-epoch pass through the real stage-2 path with the real overrides.
python scripts/launch.py finetune --gpus 2 \
    experiment=finetune_hierarchical_moe \
    experiment.variant=smoke_check \
    experiment.group=ablation \
    experiment.training.save_path=/kaggle/working/smoke/seed42 \
    hydra.run.dir=/kaggle/working/smoke/seed42/hydra \
    seed=42 \
    model.backbone.checkpoint_path=$SEED_PRETRAIN_BACKBONE \
    data.batch_size=8 data.num_workers=2 \
    experiment.training.amp=fp16 \
    experiment.training.epochs=1 experiment.training.max_batches=2 \
    experiment.training.test_size=0.3 \
    experiment.efficiency.measure_latency=false \
    tracking.wandb.enabled=false

rm -rf /kaggle/working/smoke   # it must not reach the report
```

Delete the smoke directory. It writes a `summary.json` and would otherwise appear
as a row.

---

## 11. Pre-flight checklist

Per session:

- [ ] Accelerator is `GPU T4 x2` (`nvidia-smi` shows two devices, `compute_cap 7.5`)
- [ ] Internet on for Sessions 3–5; off is fine for 1–2
- [ ] Carry-over dataset mounted and copied into `/kaggle/working/outputs` (§6.2)
- [ ] `SEED_DATA_ROOT`, `SEED_OUTPUT_DIR` exported; `SEED_PRETRAIN_BACKBONE`
      exported **only** for checkpoint-consuming arms
- [ ] `pip install -e ".[tracking]"` completed
- [ ] `find $SEED_DATA_ROOT -type f | wc -l` → 13,492
- [ ] Encoder SHA-256 matches previous sessions
- [ ] Session start time noted, so `max_runtime_minutes` on the last run of the
      session follows the §5.3 rule

Per run:

- [ ] `experiment=` names the right config
- [ ] `experiment.variant=` set (mandatory for ablations)
- [ ] `save_path` and `hydra.run.dir` point at the same directory
- [ ] `model.backbone.checkpoint_path` present **iff** §3 says **YES**
- [ ] `data.batch_size=8`, `data.num_workers=2`, `amp=fp16`, `seed=42`
- [ ] `resume=auto` and a `max_runtime_minutes` that fits the remaining session

Before writing anything up:

- [ ] Every expected variant appears in `summary_metrics.csv` (§9.3)
- [ ] Exactly one corpus SHA-256 across all runs
- [ ] Exactly one split identity across all runs
- [ ] `Split Protocol` reads `stratified` on every row
- [ ] The manuscript says "single seed (42), paired McNemar" and does not imply
      dispersion (§1.5)
- [ ] The ViT-S/16 and the IN-21k / IN-12k tag caveats are either stated or
      resolved (§0.5)
