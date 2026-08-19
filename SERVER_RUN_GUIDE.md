# Server run guide

Verified locally before this pipeline was cleared for a server run: 344/344
tests pass, `scripts/dry_run.py` completes cleanly, and a real 2-batch,
1-epoch pass through both `main.py pretrain` and `main.py finetune` on the
actual 9,357-image dataset produced `summary.json`, `test_predictions.npz` and
a checkpoint with no NaN/Inf anywhere and no unhandled exceptions.

---

## 1. Environment setup

Python `>=3.11` (`pyproject.toml`); developed and tested against **3.12**.

```bash
# Fresh environment (conda shown; venv works identically)
conda create -n seed-moe python=3.12 -y
conda activate seed-moe

# CUDA build of PyTorch first, matched to the box's driver — check with
# `nvidia-smi` and pick the matching index URL from pytorch.org/get-started.
# Example for CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Project + extras (tracking = tensorboard/wandb/pynvml, dev = pytest)
cd seed-moe-classifier
python -m pip install -e ".[tracking,dev]"
```

`device: auto` (the default) resolves to `cuda` when available, else `mps`,
else `cpu`. AMP and `pin_memory` are gated on `device.type == "cuda"` — no
action needed on the server beyond having a CUDA build of torch installed.

Confirm the install:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pytest tests/ -q          # expect all passing in ~8-15s on CPU
python scripts/dry_run.py           # synthetic pipeline, no dataset needed
```

---

## 2. Dataset requirements

Expected layout, read by `HierarchicalSeedDataset` (`src/datasets/dataset.py`):

```
<SEED_DATA_ROOT>/
  Millet/
    Baryard/*.png
    Browntop/*.png
    FingerMillet/*.png
    KarurKuruvai/*.png
    PearlMillet/*.png
    ProsaMillet/*.png
    ...                       (8 sub-varieties total)
  Mustard/
    Jagnath/*.png
    PM30/*.png
    Unknown1/*.png            (3 sub-varieties)
  Rice/
    ...                       (13 sub-varieties)
  Seasame/
    VRI1/*.png
    VRI2/*.png
    VRI4/*.png                (3 sub-varieties)
```

- **4 seed types, 27 sub-varieties total** (13 rice + 8 millet + 3 + 3).
  Labels are assigned from **sorted directory names** — adding, removing or
  renaming a folder shifts every index and invalidates any checkpoint trained
  against the old tree. The trainer refuses to start if the discovered counts
  disagree with `data.num_seed_types` / `data.num_sub_varieties`.
- **9,357 image crops from 81 source photographs** (mean 115.5 crops/source).
  Filenames encode provenance: `IMG_0502_bbox137.png` is bbox 137 cut from
  source photograph `IMG_0502`. Do not rename crops — the grouped split
  protocol parses this pattern to keep every crop of one source photograph on
  one side of the train/test boundary.
- **5 sub-varieties have crops from exactly one source photograph**
  (`Baryard`, `Browntop`, `FingerMillet`, `PearlMillet`, `ProsaMillet`). No
  split protocol can put any of their crops on both sides; the trainer logs
  this at startup and records it in `summary.json`. This is a dataset
  property, not a bug to chase.
- Images are **small and non-square**: median 52x51 px, all under 256px on
  both sides, only 3.4% square. The pipeline resizes with a `(H, W)` tuple and
  upsamples ~5x before the backbone sees anything — expected.
- Supported extensions: `.png .jpg .jpeg .bmp .tif .tiff .webp`.

Point the trainer at the dataset with an environment variable (no code change
needed):

```bash
export SEED_DATA_ROOT=/workspace/data/Hierarchical_SeedData/Cropped_Samples
```

---

## 3. Execution commands

### 3.1 One-off sanity pass (always run this first on a new box)

```bash
python -m pytest tests/ -q
python scripts/dry_run.py --device cuda      # or leave --device out for auto
python scripts/verify_runtime.py --gpus 2    # drop --gpus on a single-GPU box
```

`verify_runtime.py` is the one that answers *"do the optimisations still compute
the intended function on this hardware?"* — capability probe, SDPA parity per
module in fp64, AMP dtype selection and fp32 pinning, exact resume, and DDP
gradient equality over the real NCCL backend. Run it after provisioning and after
any driver or torch upgrade.

### 3.2 Environment variables (set once per shell / put in the launch script)

```bash
export SEED_DATA_ROOT=/workspace/data/Hierarchical_SeedData/Cropped_Samples
export SEED_OUTPUT_DIR=/workspace/outputs
export SEED_PRETRAIN_BACKBONE="${SEED_OUTPUT_DIR}/checkpoints/dinov2_swinv2_pretrained.pth"
mkdir -p "${SEED_OUTPUT_DIR}"
```

`scripts/train_distributed.sh` sets these three with the same server-shaped
defaults (`/workspace/...`) if you don't export them yourself, and dispatches
single- or multi-GPU depending on `GPUS`.

### 3.2a The v2 pipeline, end to end

The current stage-1 recipe. [`STAGE1_V2.md`](STAGE1_V2.md) carries the reasoning
and the measurements; this is the command sequence. The v1 commands in the rest
of this section are unchanged and still reproduce every published number.

```bash
export SEED_DATA_ROOT=/workspace/data/Hierarchical_SeedData/Cropped_Samples
export SEED_OUTPUT_DIR=/workspace/outputs
mkdir -p "${SEED_OUTPUT_DIR}"

# --- pre-flight, no GPU needed, ~3 minutes total ---------------------------
python -m pytest tests/ -q                          # 644 tests
python scripts/report_view_geometry.py --compare hierarchical_seeds seed_crops_v2
python main.py pretrain-v2 \
    data.batch_size=4 data.num_workers=0 experiment.training.effective_batch_size=4 \
    experiment.training.epochs=2 experiment.training.max_batches=3 \
    'experiment.training.save_epochs=[1,2]' \
    experiment.training.probe.every_epochs=1 experiment.training.probe.max_samples=270 \
    model.backbone.pretrained=false device=cpu      # end-to-end on the real corpus

# --- the reference the run must beat (no training) -------------------------
python -m src.trainers.pretrain_eval experiment=eval_frozen_v2

# --- stage 1 ---------------------------------------------------------------
python main.py pretrain-v2 --gpus 2                 # ~50 epochs, probe may stop earlier
#   on a preemptible instance, the same line resumes:
python main.py pretrain-v2 --gpus 2 experiment.training.resume=auto \
    experiment.training.max_runtime_minutes=520

# --- score the milestones, confirm the probe chose correctly ---------------
python main.py eval-pretrain-v2

# --- stage 2, crop-level train / validation / test -------------------------
export SEED_PRETRAIN_BACKBONE="${SEED_OUTPUT_DIR}/checkpoints/dino_v2_swinv2_tiny.pth"
python main.py finetune-v2 --gpus 2

# --- the photograph-disjoint diagnostic, for the leakage delta -------------
python -m src.trainers.moe_finetune experiment=finetune_v2_grouped_diagnostic

# --- the arms, if the budget allows ---------------------------------------
python scripts/run_stage1_ablations.py --arms conf/stage1_arms/view_design.yaml
```

**Five things to check in the first two minutes of the stage-1 log**, in order:

1. `Corpus size verified: 9357 images` — if this raises instead, the dataset root
   is wrong and the run stops before spending anything. That is the intent.
2. `View geometry | local ... native px p5/p50/p95 = 493/1462/4704` — what the
   augmentation is actually building. A local median near 600 means the v2 data
   group did not take effect.
3. `deterministic-centre-crop fallback` under 15 % on the global views. Above it
   the trainer warns and names the fix (`crop_ratio`).
4. `Representation probe ON | 9357 of 9357 images ... Probing at epochs: ...`
5. `epoch/loop_blocked_fraction` under ~0.3 by the end of epoch 1. Above that the
   loader is the bottleneck; raise `data.num_workers`.

**Where the artifacts land.** `${SEED_OUTPUT_DIR}/pretrain_v2_swinv2_tiny/` holds
the milestones, `dino_best_encoder.pth` (the probe-selected weights, and what
gets published) and `summary.json`. The Hydra run directory holds
`events.jsonl`, `csv/` (seven wide CSVs including `probe_history.csv`,
`checkpoint_selection.csv` and `view_geometry.csv`), `tensorboard/`, the W&B
offline run, and `figures/stage1/` (five figures, PNG + PDF). Rebuild the figures
anywhere with `python scripts/plot_stage1_run.py <run dir>`.

**Batch geometry.** `data.batch_size: 64` with `gradient_accumulation_steps: 1`.
Sinkhorn and KoLeo are per-**micro**-batch statistics, so if the card cannot hold
64, lower `data.batch_size` **and** `experiment.training.effective_batch_size`
together rather than raising accumulation — the learning rate follows
automatically through `resolve_learning_rate`. SwinV2-Tiny at 6 views leaves
substantial headroom on a 24 GB card;
`python scripts/bench_pretrain_step.py --find-batch-size 32,48,64,96` measures it.

### 3.3 Stage 1 — DINO self-supervised pretraining (run once)

```bash
python main.py pretrain data.num_workers=16
# equivalent to:
python -m src.trainers.contrastive_pretrain experiment=pretrain_swinv2_dino
```

> **Set `data.num_workers` on a server.** The published 13.34-hour run carried
> `data.num_workers=0` on a 48-physical-core host and spent a mean **91.6 %** of
> its wall clock blocked in the dataloader. One sample costs six independent PIL
> chains, so the loader — not the GPU — set the throughput. `auto` now caps at
> **16** per rank (`data.num_workers_auto_cap`, raised from 8, which was chosen
> for a 4-vCPU Kaggle instance). This changes who computes an augmentation, never
> what it is: the objective and every reported number are unaffected. Watch
> `epoch/loop_blocked_fraction` in epoch 1; if it does not fall below ~0.3 the
> bottleneck is elsewhere and every cost estimate below has to be redone.

Produces and publishes the **one** encoder checkpoint every downstream run
reads: `${SEED_OUTPUT_DIR}/checkpoints/dinov2_swinv2_pretrained.pth`. Do not
run this per-variant — every ablation and baseline that consumes it must start
from byte-identical weights, or the comparison table partly measures
self-supervised initialisation noise instead of the architecture change under
test.

**100 epochs** from ImageNet-1k weights on SwinV2-Tiny (27.58 M, 13.32
GFLOPs/view — both measured), at a physical batch of 32 with accumulation 1.
Budget for a multi-hour run on a single modern GPU; the trainer prints a
compute/parameter budget at startup and again at the end, with measured and
estimated quantities labelled apart.

Encoders are additionally kept at epochs **25, 50 and 100**
(`dino_backbone_epoch_0025.pth` and friends, never pruned), so the question
"did 100 epochs earn their cost over 25?" can be answered by pointing stage 2 at
each in turn:

```bash
SEED_PRETRAIN_BACKBONE=$SEED_OUTPUT_DIR/pretrain_swinv2_dino/dino_backbone_epoch_0025.pth \
    python main.py finetune
```

**Measure the batch before committing.** Physical batch is what Sinkhorn and
KoLeo estimate from — accumulation averages gradients and buys those statistics
nothing — so it is worth finding the largest that fits:

```bash
python scripts/bench_pretrain_step.py --find-batch-size 16,24,32,48,64
```

It runs the real micro-step at each candidate in a fresh subprocess (a failed
attempt leaves the allocator fragmented, so reusing the process would understate
every later candidate), reports peak VRAM and img/s per size, and names the
largest that fits. Raise `data.batch_size` and
`experiment.training.effective_batch_size` **together** so accumulation stays 1;
the learning rate re-derives itself from the effective batch.

**Reading the log.** Two lines are easy to misread:

```
Step 1200 | epoch=10 batch=45 loss=5.67 | CE=5.67 = KL 0.29 + H 5.38 | ... loop_blocked=8.1%
```

- `loss` is a cross entropy, so `CE = H(teacher) + KL(teacher‖student)`. Under
  Sinkhorn centering `H` is set by the normaliser, `K`, `B_teacher` and the
  temperature schedule — **none of which is the student learning**. On the
  published run 80 % of the total loss drop was `H` falling and the final loss was
  94.8 % irreducible target entropy. **Read `KL`.** An arm that changes the
  centering moves `H` directly, so its raw loss is not comparable to another
  arm's.
- `loop_blocked` is the share of wall clock the *loop* spent inside the
  dataloader. Nothing synchronises inside the step, so the queued GPU work drains
  during that window — it upper-bounds idleness rather than measuring it. For the
  real figure, add `experiment.training.measure_gpu_busy=true` (one stall per
  logging interval; leave it off for the production run).

**What the run records about itself.** It writes `summary.json` beside the
checkpoints: the resolved augmentation, the view geometry, the effective batch,
the LR provenance, every objective-side flag, the final `KL`, and a **corpus
fingerprint** — a SHA-256 over the sorted dataset-relative path list plus the
sample, class and source-group counts. That last one exists because the published
encoder turned out to have been self-distilled on **8,173** crops while everything
downstream used 9,357, with nothing on disk recording it. `eval-pretrain` reads
the fingerprint back and prints a prominent mismatch line when the corpora differ.

**On two GPUs**, and on any platform that ends the session before the run does:

```bash
python main.py pretrain --gpus 2 \
    experiment.training.resume=auto \
    experiment.training.resume_every_minutes=20
```

Three things to understand about that command.

`--gpus 2` pins one `$SEED_RUN_ID` so both ranks share an output directory, then
launches under `torch.distributed.run`. `DistributedSampler` shards the *images*;
all six views of a sample stay on the rank that owns it, because the loss pairs a
student view against the teacher's output for that same image.

The **effective batch does not change**. `data.batch_size` is per-rank, and
`experiment.training.effective_batch_size` (32) derives the accumulation count
from it and the world size — 1 GPU at `32 x 1` and 2 at `16 x 1` are the same 32
images per optimizer step. A mismatch that does not divide exactly is refused
rather than rounded.

Note the trade that splitting makes, though: `16 x 2` keeps the *gradient*
identical to `32 x 1` and halves what Sinkhorn and KoLeo estimate from, since
both are computed per micro-batch. If the second card has the memory, the
statistically better use of it is `data.batch_size=32` on both ranks with
`effective_batch_size=64` — a different, larger run — or
`model.loss.distributed_sinkhorn=true`, which normalises over the concatenated
global batch and restores the 32-image estimate exactly (a different objective
from the single-GPU one, hence opt-in).

`resume=auto` continues from the newest valid checkpoint and starts fresh when
there is none, so the **identical command line** works for the first launch and
every relaunch. The resume checkpoint carries the teacher, optimizer moments,
scheduler, `GradScaler`, epoch, global step, micro-batch within the epoch and
every rank's RNG — a relaunch continues rather than restarting warm.

### 3.3a Kaggle T4 x 2

Two Turing cards, 16 GB each, ~4 vCPUs, and a session limit. Everything adapts
automatically except the two flags that tell the run about the limit:

```bash
!cd /kaggle/working/seed-moe-classifier && \
  SEED_DATA_ROOT=/kaggle/input/<dataset>/Cropped_Samples \
  SEED_OUTPUT_DIR=/kaggle/working/outputs \
  python scripts/launch.py pretrain --gpus 2 \
      experiment.training.resume=auto \
      experiment.training.max_runtime_minutes=520 \
      experiment.training.resume_every_minutes=15 \
      experiment.training.keep_last_n_checkpoints=2
```

What adapts by itself:

| Resolved | To | Why |
| --- | --- | --- |
| `amp: auto` | **fp16 + `GradScaler`** | `sm_75` has no hardware bf16. The Sinkhorn normaliser, the 2,048-way prototype log-softmax and the KoLeo distances are pinned to fp32 inside the autocast region, which is what makes fp16 safe here. |
| `compile.enabled: auto` | on (Triton supports `sm_75`) | Costs a few minutes of graph capture on the first step. Set `false` if a short session cannot amortise it. |
| `data.num_workers: auto` | 2 per rank | Per-process setting; the literal 8 would put 16 augmentation workers on 4 vCPUs. |
| SDPA backend | memory-efficient (no flash below `sm_80`) | Still avoids materialising the `[B*nW, heads, N, N]` matrices, which is the whole point of the rewrite. |

What does not adapt, and why these two flags matter: `max_runtime_minutes` stops
the run *cleanly with a complete checkpoint* before the platform kills it — more
reliable than being signalled, since not every platform signals first — and
`resume_every_minutes` bounds the worst case to minutes rather than to an epoch
interval, which on a session-limited platform can exceed the whole session.

Measure the batch geometry before committing. On 16 GB Turing cards the
configured batch of 32 is the thing to check first:

```bash
python scripts/bench_pretrain_step.py --find-batch-size 8,16,24,32
python scripts/bench_pretrain_step.py --scaling 1,2 --batch-size 16
```

If 32 does not fit, lower both together (`data.batch_size=16
experiment.training.effective_batch_size=16`) rather than raising accumulation.

### 3.3b Evaluate the stage-1 encoder before spending stage-2 compute

```bash
# plumbing check first: ~1 minute, exercises every analysis and figure
python main.py eval-pretrain experiment.evaluation.max_samples=270

# the real report: one forward pass over the dataset per encoder
python main.py eval-pretrain
```

Runs on a single device (there is no gradient to reduce, so `--gpus` is refused)
and needs no network access **except** for the `imagenet_init` control, which asks
timm for `swinv2_small_window16_256.ms_in1k`. On a box that will be offline later,
warm the cache once while it has network:

```bash
python -c "import timm; timm.create_model('swinv2_small_window16_256', pretrained=True)"
```

Those are the same weights stage 1 started from, so the download is ~200 MB once
per machine. Without them the `imagenet_init` row — the control that measures the
entire contribution of in-domain self-distillation — cannot be produced, and
`experiment.evaluation.encoders` has to be overridden with a list that omits it.

Cost on one A100/H100: ~4 minutes of forward passes per encoder plus ~5 minutes of
CPU analysis, so ~25 minutes for the default five encoders. Features are cached
under `outputs/eval_pretrain/features/`, so a re-run that only changes an analysis
or a figure takes seconds.

Read `outputs/eval_pretrain/tables/encoder_comparison.csv` first. The two rows that
decide whether stage 1 earned its cost are `dino_epoch100` against
`imagenet_init`; the two that decide whether the *epochs* earned theirs are
`dino_epoch25`/`dino_epoch50` against `dino_epoch100`. Interpretation and the
recommendations that follow: [`STAGE1_EVALUATION.md`](STAGE1_EVALUATION.md).

Three columns added after the stage-1 audit, and what they are for:

| Column | Read it as |
| --- | --- |
| `nuisance_photo_above_chance` | how much **photograph nuisance** survives, class held constant. The shipped run cut it from +10.0 pp (ImageNet) to +3.5 pp — the one axis stage 1 demonstrably moved. Read **jointly** with the readout: an encoder that discards everything scores chance |
| `oof_probe_sub_accuracy_at_stage3` | the same headline probe at `layers.2` rather than the pooled output stage 2 consumes. `layers.2` scored +3.25 pp in the audit, and the ordering held for the plain ImageNet weights too |
| the `handcrafted_floor` row | ten image statistics under the identical protocol. They reach 0.5360, which is 15.6 pp **above** an untrained 48.96 M trunk. A deep encoder that does not clear this comfortably is not doing much |

### 3.3c Two screens that need no training at all

Both are one forward pass per row, cached by checkpoint digest afterwards, and
neither writes a checkpoint or touches the published handoff.

```bash
# Which INITIALISATION transfers best? The evaluation's own decomposition is
# random 0.3804 -> +0.2449 ImageNet-1k -> +0.0031 DINO: the initialisation is
# worth 79x what the 13.34-hour run bought, and was never treated as a variable.
python main.py screen-backbones

# The reference every stage-1 arm must beat: the chosen trunk, frozen, no
# in-domain training. If no arm beats it by more than a fold SD, stage 1 as an
# OBJECTIVE is not earning its compute.
python main.py eval-frozen
```

`screen-backbones` downloads several timm checkpoints, so run it while the box has
network. The row to look at first is `base_in1k`: Base capacity at the *same*
IN-1k corpus, which separates **capacity** from **the IN-22k corpus**. ≈0.63 means
capacity (and Tiny is then the efficient choice — it is −0.32 pp pooled and
**+0.69 pp at `layers.2`** against Small for half the FLOPs); ≈0.61 means the
corpus. **Do not adopt Base before that row has run.**

### 3.3d Stage-1 arm suites

```bash
python scripts/run_stage1_ablations.py --arms conf/stage1_arms/phase1.yaml --dry-run
python scripts/run_stage1_ablations.py --arms conf/stage1_arms/phase1.yaml     --experiment pretrain_swinv2_tiny_dino
```

Each arm trains, then evaluates, into its own directory under
`${SEED_OUTPUT_DIR}/stage1_arms/<arm>/`, with `experiment.training.save_path`,
`shared_backbone_path` and `experiment.evaluation.save_path` **all** pinned per
arm. That is the whole reason the script exists: left at their defaults, every arm
publishes over `outputs/checkpoints/dinov2_swinv2_pretrained.pth` and the last one
to finish silently becomes the encoder every stage-2 run reads.

Do **not** shard arms across GPUs. A stage-1 arm saturates one device, so two
concurrent arms halve each other's throughput; use both devices *within* an arm
(`python main.py pretrain --gpus 2` with `effective_batch_size` pinned).

The suite writes `stage1_arms/stage1_arm_results.csv` and prints the same table.
Judge on `oof_probe_sub_accuracy_testable_classes` with the fold SD — a single arm
cannot resolve a difference below ~2 pp, which is why Phase 3 exists.

### 3.4 Stage 2 — single finetune run (smoke-test the head before the full suite)

```bash
python main.py finetune
# with overrides, e.g. the submitted (not revised) configuration:
python main.py finetune model.head.top_k=4 model.head.token_mode=pooled
```

### 3.5 Full ablation suite — 18 variants x 5 seeds = 90 runs

```bash
# Verify command construction first, no training:
python scripts/run_ablations.py --dry-run

# Real run, all variants, all 5 seeds (42-46):
python scripts/run_ablations.py

# Subset, single seed (quick check the suite plumbing works on this box):
python scripts/run_ablations.py --variants full_model wo_moe wo_kl --seeds 42
```

Requires the Stage 1 checkpoint to exist (`ensure_pretrained_checkpoint()`
refuses to start otherwise). Runs land in
`${SEED_OUTPUT_DIR}/ablations/{variant}/seed{n}/`, each self-contained
(Hydra snapshot, logs, checkpoint, `summary.json`, `test_predictions.npz`).

### 3.6 Full baseline suite — 5 models x 5 seeds = 25 runs (+ optional LR sweep)

```bash
python scripts/run_baselines.py --dry-run
python scripts/run_baselines.py
python scripts/run_baselines.py --lr-sweep   # +6 runs: resnet50/swin_tiny x {1e-5,3e-5,1e-4}
```

Run `linear_probe` first if you only have time for one — its outcome bounds
what the rest of the paper can claim. `resnet50`, `swin_tiny` and
`swinv2_supervised` own their own ImageNet backbones and deliberately do
**not** read the Stage 1 checkpoint; `linear_probe` and `hierarchical_cce` do.

### 3.7 Report generation

```bash
python scripts/generate_plots.py
# or restrict to specific run roots:
python scripts/generate_plots.py --roots outputs/ablations outputs/baselines
```

Re-scores every run from its raw `test_predictions.npz` rather than trusting
`summary.json`, so the table and the figures are computed by the same code.
Writes `outputs/reports/summary_metrics.csv`, `summary_metrics_per_run.csv`
and 300 DPI figures.

The stage-1 evaluation (§3.3b) writes its own `summary.json` and
`test_predictions.npz` in the same format, so
`--roots outputs/eval_pretrain` includes it in that table. Its own report lives
under `outputs/eval_pretrain/` and does not need `generate_plots.py`.

### 3.8 Running detached (nohup / tmux)

**tmux (recommended — lets you reattach and watch progress):**

```bash
tmux new -s seedmoe
export SEED_DATA_ROOT=/workspace/data/Hierarchical_SeedData/Cropped_Samples
export SEED_OUTPUT_DIR=/workspace/outputs
python main.py pretrain
# Ctrl-b d to detach; `tmux attach -t seedmoe` to come back
```

**nohup, for a scripted end-to-end sequence:**

```bash
export SEED_DATA_ROOT=/workspace/data/Hierarchical_SeedData/Cropped_Samples
export SEED_OUTPUT_DIR=/workspace/outputs
mkdir -p "${SEED_OUTPUT_DIR}/logs"

nohup bash -c '
  set -e
  python main.py pretrain
  python scripts/run_ablations.py
  python scripts/run_baselines.py
  python scripts/generate_plots.py
' > "${SEED_OUTPUT_DIR}/logs/full_run.log" 2>&1 &

echo "Launched PID $!"
disown
# tail -f "${SEED_OUTPUT_DIR}/logs/full_run.log" to watch
```

**Or, using the bundled launcher** (sets the three env vars for you if unset):

```bash
nohup scripts/train_distributed.sh pretrain   > logs/pretrain.log   2>&1 & disown
# wait for it to finish, then:
nohup scripts/train_distributed.sh ablations  > logs/ablations.log  2>&1 & disown
nohup scripts/train_distributed.sh baselines  > logs/baselines.log  2>&1 & disown
nohup scripts/train_distributed.sh report     > logs/report.log     2>&1 & disown
```

Each stage must finish before the next starts (ablations/baselines both read
the checkpoint pretrain publishes) — do not launch them concurrently in the
background expecting them to race safely.

---

## 4. Hyperparameter reference

### 4.1 Data / training loop

| Hyperparameter | Value | Where |
| --- | --- | --- |
| `data.batch_size` | 16 | `conf/data/hierarchical_seeds.yaml` |
| `data.image_size` | 256 | must match backbone window resolution |
| Stage 1 `learning_rate` | 0.0005 | `conf/experiment/pretrain_swinv2_dino.yaml` |
| Stage 1 `epochs` | 300 | — |
| Stage 1 `clip_grad` | 3.0 | — |
| Stage 1 `gradient_accumulation_steps` | 4 (effective batch 64) | — |
| Stage 1 teacher momentum | 0.996 -> 1.0 (cosine) | — |
| Stage 1 weight decay | 0.04 -> 0.4 (cosine) | — |
| Stage 2 `learning_rate` | 0.0001 | `conf/experiment/finetune_hierarchical_moe.yaml` |
| Stage 2 `epochs` | 100 | — |
| Stage 2 `weight_decay` | 0.0001 | — |
| Stage 2 `clip_grad` | 3.0 | — |
| `split_protocol` | `grouped` (default) / `stratified` (submitted) | — |
| `test_size` | 0.2 | — |
| `num_folds` | 1 (set >1 for `StratifiedGroupKFold`) | — |
| `margin_warmup_fraction` | 0.15 | ArcFace margin ramps 0 -> m |
| `router_noise_fraction` | 0.3 | gate noise anneals to 0 |
| `horizontal_flip_prob` (stage 2) | 0.5 | — |
| `random_resized_crop_scale` (stage 2) | [0.8, 1.0] | — |

### 4.2 Model / MoE head (`conf/model/head/hierarchical_moe.yaml`)

| Hyperparameter | Value | Notes |
| --- | --- | --- |
| `embed_dim` (z, Eq. 4) | 384 | encoder invariant, always 384 regardless of backbone width |
| `num_experts` | 6 | Eq. 8 |
| `top_k` | **2** (revised) / 4 (submitted) | `model.head.top_k=4` to reproduce the paper |
| `moe_hidden_dim` | 512 | per-expert transformer hidden size |
| `token_mode` | `grid` (revised) / `pooled` (submitted) | grid keeps the 8x8 token grid; pooled collapses attention to affine |
| `router_mode` | `learned` | `hash` / `uniform` are ablation-only |
| `gate_conditioning` | true | gate sees detached `p_s`; experts always see `z` |
| `router_noise_std` | 0.3 | annealed to 0 |
| `num_heads` (cross-attention) | 8 | only real under `token_mode=grid` |
| `fusion_mode` | `additive` (Eq. 9) / `film` (ablation) | — |
| `residual_layer_scale` | 0.0001 | LayerScale init |
| `sub_head_variant` | `arcface` | `normface` / `linear` for ablation |
| `arcface_scale` | `"auto"` -> 4.61 (AdaCos, C=27) | submitted value: 30.0 |
| `arcface_margin` | 0.5 | ramped 0 -> 0.5 over `margin_warmup_fraction` |
| `arcface_sub_centers` | 1 | data-cleaning tool, not a suite variant |

### 4.3 Loss (`conf/model/loss/arcface_kl.yaml`)

| Hyperparameter | Value | Notes |
| --- | --- | --- |
| `weighting_mode` | `fixed` (`uncertainty` learns 3 scalars) | — |
| `lambda_seed` | 1.0 | Eq. 7 |
| `lambda_arcface` | 1.0 | Eq. 13 |
| `lambda_kl` | 1.0 | Eq. 10 |
| `tau_kl` | 1.0 | decoupled from `arcface_scale` |
| `lambda_moe_load` | 0.01 | — |
| `lambda_moe_sparsity` | 0.0 | redundant under `renormalize_top_k` |
| `lambda_moe_z` | 0.001 | ST-MoE router z-loss |
| `lambda_cosine` | 0.1 | Section 1 compactness |
| `lambda_residual` | 0.01 | hinge, zero below `residual_tau` |
| `residual_tau` | 0.5 | — |
| `moe_load_mode` | `switch` (revised) / `entropy` (submitted) | `model.loss.moe_load_mode=entropy` to reproduce the paper |
| `cosine_mode` | `intra_class` (EMA centroids) / `residual` (submitted, collapses) | — |
| `centroid_momentum` | 0.9 | EMA over class centroids |
| `detach_kl_seed_target` | true | — |
| `kl_mode` | `forward` (Eq. 10) / `jsd` (ablation) | — |

### 4.4 Suite-level

| Hyperparameter | Value | Notes |
| --- | --- | --- |
| `seeds` | `(42, 43, 44, 45, 46)` — `DEFAULT_SEEDS` in `src/trainers/runner.py` | every variant repeats over all 5 |
| Ablation variants | 18 (see `scripts/run_ablations.py`) | `full_model` is the reference |
| Baseline models | 5: `linear_probe`, `swinv2_supervised`, `resnet50`, `swin_tiny`, `hierarchical_cce` | see `scripts/run_baselines.py` |
| LR sweep (optional) | `{1e-5, 3e-5, 1e-4}` for `resnet50`/`swin_tiny` | `--lr-sweep` flag |

To reproduce the **submitted manuscript's** configuration point-for-point
instead of the revision, chain the override table in `REVISION_NOTES.md` §0
(e.g. `model.head.top_k=4 model.head.token_mode=pooled
model.loss.moe_load_mode=entropy model.head.arcface_scale=30.0
model.loss.cosine_mode=residual experiment.training.split_protocol=stratified`).

---

## 5. Output contract

Every run — full model, every ablation variant, every baseline — writes to
its own directory under `${SEED_OUTPUT_DIR}` and leaves the same set of
artifacts, which is what lets `scripts/generate_plots.py` process all of them
uniformly without retraining anything:

```
${SEED_OUTPUT_DIR}/
  checkpoints/
    dinov2_swinv2_pretrained.pth      # the ONE shared Stage-1 encoder
  pretrain_swinv2_dino/
    dino_pretrained_final.pth
    dino_pretrained_backbone.pth
    dino_checkpoint_epoch_XXXX.pth    # rolling, keep_last_n_checkpoints=1
  finetune_hierarchical_moe/          # `main.py finetune` (single run)
    summary.json
    test_predictions.npz
    split_manifest.npz
    hierarchical_moe_final.pth
    best_hierarchical_moe.pth
  ablations/
    {variant}/seed{42..46}/
      summary.json
      test_predictions.npz
      split_manifest.npz
      hydra/                          # full config snapshot for that run
      *.pth
    suite_manifest.json                # one row per (variant, seed): status, duration, path
  baselines/
    {model}/seed{42..46}/              # same contract as above
    suite_manifest.json
  reports/
    summary_metrics.csv               # Model/Variant, Accuracy, Precision, Recall,
                                       # Macro F1, Micro F1, KL Alignment Rate (%),
                                       # Total Params (M), Active Params (M),
                                       # Inference Latency (ms), + AUC/throughput/
                                       # FLOPs/peak-memory appended to the right
    summary_metrics_per_run.csv       # unaggregated, one row per (variant, seed)
    *.png                             # confusion matrices, t-SNE, loss curves,
                                       # expert utilization, metric heatmaps — 300 DPI
  metadata/
    seed_dataset.csv                  # image_path, seed_type(_label), subvariety(_label)
  hydra/<date>/<time>/
    tensorboard/                      # per-run TensorBoard logs
  events.jsonl                        # append-only tracker event log, every run
```

Key points:

- **`summary.json`** is the source of truth per run: scalar metrics,
  `component_flags()` (every architectural axis, so `wo_kl` is
  machine-distinguishable from `full_model`), `loss_flags()`, `split`
  (protocol + leakage/provenance diagnostics), `fold_metrics`
  (mean +- std across folds, not best-fold), efficiency report, loss history.
- **`test_predictions.npz`** carries raw predictions, scores, 384-D
  embeddings, routed expert indices and class names — `generate_plots.py`
  re-scores from this rather than trusting `summary.json`.
- **`split_manifest.npz`** persists the exact split (indices, source groups,
  class mappings) so a reviewer can verify grouping rather than trust the
  protocol name.
- W&B runs in **offline mode** by default (never blocks on credentials);
  TensorBoard is always on. A missing tracking backend degrades to a logged
  warning, not a crash.
- Defaults are tuned for a 16 GB disk: no optimizer-state checkpoints, no
  teacher weights saved, `keep_last_n_checkpoints: 1`, histograms off. A
  100-epoch pretrain run with those turned on is how the disk fills — leave
  them off unless actively debugging.

---

## 6. Known, expected warnings (not bugs)

- `Grouped splitting left these sub-varieties out of training entirely: [...]`
  — expected for the 5 single-source-photograph sub-varieties (and possible
  for others by chance under a small `test_size`). Logged and recorded in
  `summary.json`, not a failure.
- A `pynvml` deprecation `FutureWarning` on import — cosmetic, from a
  transitive dependency.
- A `lr_scheduler.step() before optimizer.step()` warning **only** appears in
  artificially short smoke runs where `max_batches` is smaller than
  `gradient_accumulation_steps` (so the optimizer never actually steps within
  that truncated epoch). It does not occur in a real run, where an epoch
  contains many accumulation windows.
