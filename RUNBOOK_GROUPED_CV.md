# Runbook — photograph-disjoint stage-2 cross-validation *(DORMANT)*

> ## ⚠️ DORMANT — OUT OF SCOPE FOR THIS STUDY
>
> **This procedure is not part of the pipeline and its output does not appear in
> the paper, the report or any results table.**
>
> This study's sole evaluation standard is the **crop-level stratified
> protocol**, run by `python main.py finetune` and reported in
> `MODEL_EVALUATION_REPORT.md`. The unit of classification is the individual
> seed instance; photograph-partitioned cross-validation answers a different
> question (generalisation to a new acquisition session) that this study does
> not pose.
>
> The document is kept only as a legacy exploratory note, so that the launch
> procedure does not have to be reconstructed if that question is ever taken up
> as separate work. Nothing in the primary pipeline depends on it, no figure or
> table reads its output, and `scripts/generate_plots.py` does not scan for it.
>
> **Do not cite numbers from this procedure alongside the published results.**

---

The job it describes: `experiment=finetune_grouped_diagnostic`, 5
photograph-disjoint folds over all 13,492 crops, out-of-fold predictions
concatenated. It is the same stage-2 recipe with the split protocol swapped.

## 0. Why not on the Mac

Measured on this machine (Apple MPS, `data.batch_size=8`):

| | throughput |
| --- | --- |
| training | ~14 img/s |
| evaluation | ~20 img/s |

≈ 15 min/epoch × 100 epochs × 5 folds ≈ **5.2 days**. On a single T4 the same job
is roughly **20–30 h** (see §4 for how to pin that down in the first five
minutes rather than trusting this estimate).

---

## 1. Run it on ONE GPU, not two

Use a single T4 even if the box has two. This is not a performance claim, it is
a comparability one.

Stage 2's `data.batch_size` is **per rank** and `DistributedSampler` shards the
dataset, so `--gpus 2` gives a global batch of 16 where the stratified run used
8. Stage 2 has no `effective_batch_size` authority to hold that fixed the way
stage 1 does. The entire purpose of this diagnostic is that **the split protocol
is the only thing that differs** from `outputs/finetune_hierarchical_moe`; a run
that also changed the global batch would confound the protocol delta with an
optimisation change, and the resulting "leakage gap" would be partly a
batch-size effect.

If you have two GPUs free, the better use is to run this on GPU 0 and something
independent on GPU 1 (`CUDA_VISIBLE_DEVICES=1 python scripts/run_ablations.py`),
which has no gradient traffic at all.

---

## 2. The command

```bash
# --- environment -----------------------------------------------------------
export SEED_DATA_ROOT=/kaggle/input/datasets/jgfreak/refined-samples/refined_samples/Refined_Samples
export SEED_OUTPUT_DIR=/kaggle/working/outputs
export SEED_PRETRAIN_BACKBONE=$SEED_OUTPUT_DIR/checkpoints/dino_pretrained_encoder.pth
export CUDA_VISIBLE_DEVICES=0

# --- the run (same line every session; see §3) -----------------------------
python main.py finetune-grouped \
    experiment.training.resume=auto \
    experiment.training.max_runtime_minutes=500
```

`main.py finetune-grouped` expands to
`python -m src.trainers.moe_finetune experiment=finetune_grouped_diagnostic`,
which is `finetune_hierarchical_moe` with exactly four overrides:
`split_protocol: grouped_cv`, `num_folds: 5`, `test_size: 0.0`, and its own
`save_path`. Everything else — 100 epochs, batch 8, LR 1e-4, the augmentation,
the frozen trunk — is inherited, which is what makes the comparison clean.

**Do not pass anything else.** Every extra override is another difference
between this run and the one it is being compared with.

### Pre-flight (2 minutes, catches the three things that actually go wrong)

```bash
# 1. the encoder exists and is the one stage 2 expects
python - <<'PY'
import hashlib, os, pathlib
p = pathlib.Path(os.environ["SEED_PRETRAIN_BACKBONE"])
print(p, p.exists(), p.stat().st_size if p.exists() else "-")
print(hashlib.sha256(p.read_bytes()).hexdigest())
PY
# expect ba96c3e6e0acb5f22e60e517c840297fcfe1ef956c334fa920f8c653c4bc4795

# 2. the corpus is the one every published number used
python -c "
from src.datasets.dataset import corpus_fingerprint
import os; print(corpus_fingerprint(os.environ['SEED_DATA_ROOT'])['sha256'])"
# expect 013c04c5858e9815ac0bb71b9f5fde4c8be9176c514d28198e39727344b6f3d3

# 3. the wiring, on 2 batches, ~1 minute
python main.py smoke
```

If the corpus digest differs, **stop** — the comparison is void, and
`describe_fingerprint_mismatch` will tell you which classes moved.

---

## 3. Surviving the session limit

Kaggle kills the session at 9 h (12 h on some quotas), so this job spans 3–4
sessions. Two settings make that a continuation rather than a restart:

- **`experiment.training.resume=auto`** — continues from the newest valid
  `finetune_resume_*.pth` under `save_path`, and starts fresh when there is
  none. The *same command line* therefore serves the first launch and every
  relaunch; do not special-case the first one.
- **`experiment.training.max_runtime_minutes=500`** — stop cleanly with a
  checkpoint before the hard kill. Leave ~40 min of headroom: the limit is
  checked at epoch boundaries, and a stage-2 epoch here is several minutes.

Stage 2 resumes at **epoch** granularity and the resume payload carries the
fold, so a session that dies during fold 3 restarts at fold 3, not fold 1.
Folds already completed are skipped (`if fold < resume_fold: continue`).

`keep_last_n_checkpoints` is 1 by default in this recipe. On a preemptible box
raise it, because the newest file is the one a kill is most likely to have
interrupted:

```bash
    experiment.training.keep_last_n_checkpoints=2
```

Between sessions, persist `$SEED_OUTPUT_DIR/finetune_grouped_diagnostic/` — on
Kaggle, save `/kaggle/working` as a dataset version and mount it on the next
session, or the resume has nothing to find.

### Confirming a resume actually resumed

```
Resume | continuing at fold 3, epoch 41, step 54120 (best sub_variety/f1_macro 0.61…).
```

If you instead see `Checkpoint selection | monitor=…` followed by
`fold_1/train epoch 1`, it started over — check that `save_path` points at the
persisted directory.

---

## 4. Pin the estimate in the first five minutes

Do not trust §0. The first epoch tells you the real number:

```bash
grep "fold_1/train epoch 1" -A 1 <logfile>     # duration_seconds is logged per epoch
```

```
total ≈ (seconds per epoch) × 100 epochs × 5 folds
```

At 3 min/epoch that is ~25 h; at 6 min/epoch, ~50 h, and you should reconsider
the budget before burning three sessions on it. The defensible way to shorten it
is `experiment.training.epochs`, applied to **both** protocols so they stay
comparable — not to this run alone.

---

## 5. What it writes

```
$SEED_OUTPUT_DIR/finetune_grouped_diagnostic/
├── summary.json                 # protocol, corpus digest, OOF metrics, fold_metrics
├── test_predictions.npz         # the OOF predictions: every crop, scored once
├── split_manifest.npz           # the photograph-disjoint fold assignment
├── best_hierarchical_moe.pth
└── hierarchical_moe_final.pth
```

Two properties of `test_predictions.npz` here that differ from the crop-level
run, and both matter when reading it:

- It covers **all 13,492 crops**, not a 2,699-crop test split — `grouped_cv`
  holds out no separate test set, so every crop is scored exactly once by a
  model that never trained on it.
- **K different models contributed.** This estimates the *recipe*, not any one
  shipped artifact. `summary.json` says so, and `fold_metrics` carries the
  mean ± std across folds, which is the honest dispersion figure.

### Expected, so it is not mistaken for a bug

Five sub-varieties have crops from exactly one photograph — Baryard, Browntop,
FingerMillet, PearlMillet, ProsaMillet, 1,429 crops, 10.6 % of the corpus. Under
**any** photograph-disjoint split their whole class sits on one side of the
boundary, so they are unpredictable by construction, identically for every
encoder including an untrained one. Expect their F1 to collapse toward 0.

That is a property of the corpus — each of those five genuinely has one raw
photograph — and the fix is a camera, not a splitter. It is also the main reason
this protocol is not a meaningful standard for the 27-way task as the dataset
currently stands: five of the 27 classes cannot be scored under it at all.

## 6. If it is ever run

```bash
# copy the finished run directory into ./outputs/

python scripts/generate_plots.py \
    --roots outputs/finetune_grouped_diagnostic
```

Point `--roots` at it **explicitly**. The default scan covers the crop-level
benchmark and its ablation, baseline and control arms only, so a stray run
directory cannot leak a second protocol into the published table.

Paths inside a `summary.json` are absolute to the machine that trained the run,
so a Kaggle-trained directory analysed locally points at `/kaggle/working/...`.
`RunSummary.load` rebases those onto wherever the summary is actually found, so
no editing of run artifacts is needed — but the rebase only fires for paths that
do not resolve, so provenance is preserved when it is still valid.

Whatever it produces stays in its own directory and its own table. It is
exploratory work outside the scope of this study, it is not a correction of the
crop-level benchmark, and the two are not to be reported as alternative
estimates of one quantity.
