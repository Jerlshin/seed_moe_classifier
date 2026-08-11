# `src/trainers/` — training entry points

Two Hydra applications, one per paper stage, plus the machinery that drives the
ablation and baseline suites. They own the loops, the logging, and the
checkpointing; every model and loss decision lives elsewhere.

| File | Stage | Paper |
| --- | --- | --- |
| `contrastive_pretrain.py` | DINO self-supervised pretraining | Section 4, Table 1 |
| `moe_finetune.py` | Hierarchical finetuning, ablations **and** baselines | Section 5, Section 6 |
| `runner.py` | Suite orchestration for `scripts/run_{ablations,baselines}.py` | — |

```bash
python main.py pretrain      # == python -m src.trainers.contrastive_pretrain experiment=pretrain_swinv2_dino
python main.py finetune      # == python -m src.trainers.moe_finetune experiment=finetune_hierarchical_moe
python main.py smoke         # 2-batch dry run of both
```

Both honour `experiment.training.max_batches`, which caps batches per epoch.

## One trainer serves all three

The full model, the six ablations and the three baselines all run through
`moe_finetune.py`, differing only by Hydra overrides. That is deliberate: an
ablation routed through a second training loop would differ from the full model
in ways nobody intentionally chose — a different shuffle, a different metric
reduction, a different early-stopping rule — and the measured gap would include
all of it.

`build_model_and_encoder()` dispatches on `model.head.name`. For
`flat_supervised` it builds a baseline that owns its backbone and fills the
encoder slot with `IdentityEncoder`, so images reach the model untouched and the
loop itself never branches.

## The handoff between stages is one file

Stage 1 ends by writing `dino_pretrained_backbone.pth` — a bare
`student_backbone` state dict — and then **publishing a copy** to
`experiment.training.shared_backbone_path`, by default
`outputs/checkpoints/dinov2_swinv2_pretrained.pth`. Every downstream run reads
that one file.

That indirection is what makes the ablation table valid. The suites compare
architectures, so they must all start from byte-identical encoder weights; a
per-variant pretraining stage would make each row partly a function of its own
self-supervised seed. `ensure_pretrained_checkpoint()` in `runner.py` refuses to
start a suite when the file is absent, rather than letting some variants train
from a random encoder and produce a table that looks fine and means nothing.

`checkpoint_strict: false` is the default, so a wrong checkpoint would load
quietly. The trainer therefore logs the missing/unexpected key report and warns
when it is non-empty — check that line first when debugging bad metrics.

## `contrastive_pretrain.py`

Per step:

1. Build `2 + local_crops_number` views of each image.
2. Teacher forward on the 2 global crops only, under `no_grad`.
3. Student forward on all views.
4. DINO loss over every cross-view pair (Eq. 1).
5. `backward()`, clip gradients at 3.0 (Table 1), then
   `cancel_last_layer_gradients` for the first epoch (Section 6.1).
6. `optimizer.step()`, then EMA the teacher at momentum 0.996.

Order matters in step 5: gradient cancelling must happen after clipping and
before `step()`, or the frozen layer still moves.

Logged: loss, learning rate, current teacher temperature (so the Eq. 2 schedule
is visible in the run), gradient norms, and a loss curve figure (paper Fig. 6).

`publish_shared_backbone()` failing is a warning, not an error — the per-stage
copy is already safely on disk, and discarding a completed 300-epoch pretraining
run over a file-copy failure would be indefensible.

### Running it on more than one GPU

    torchrun --standalone --nproc_per_node=2 -m src.trainers.contrastive_pretrain \
        experiment=pretrain_swinv2_dino
    python main.py pretrain --gpus 2          # the same, run id pinned

`DistributedSampler` shards **images**; all six views of a sample stay on the
rank that owns it, because Eq. 1 pairs a student view against the teacher's
output for that same image. `set_epoch` is called every epoch — the permutation
is a function of `seed + epoch`, and skipping the call replays epoch 0 forever
without an error.

The effective batch is held fixed rather than multiplied:
`resolve_accumulation()` derives `gradient_accumulation_steps` from
`effective_batch_size` and the world size and refuses a combination that does not
divide exactly. Holding the per-rank micro-batch fixed is also what keeps
Sinkhorn and KoLeo computing the same function they compute on one GPU — both
are already per-micro-batch statistics.

Only the student is wrapped in DDP (the teacher takes no gradient and is
broadcast from rank 0 once), the wrapper lives in `DINO._ddp` off the module tree
so no state-dict key gains a `module.` prefix, and `model.no_sync()` wraps every
micro-batch that is not an accumulation boundary.

### Surviving an interrupted session

`experiment.training.resume=auto` continues from the newest valid checkpoint and
starts fresh when there is none, so the same command line serves the first launch
and every relaunch. Resume checkpoints carry the teacher, optimizer moments,
scheduler, `GradScaler`, epoch, global step, **micro-batch within the epoch** and
one RNG snapshot per rank; the loop replays the already-consumed micro-batches so
the resume lands where it stopped rather than at the start of the epoch.

Two triggers write one: `resume_every_minutes` and a SIGTERM/SIGINT handler that
finishes the current micro-batch first. `max_runtime_minutes` stops the run
cleanly before a hard session limit rather than relying on being signalled.

## `moe_finetune.py`

### Splits

A held-out test set is carved out first (`test_size`), then `num_folds`
train/validation splits over the remainder — `StratifiedKFold` when
`num_folds > 1`, a single stratified split otherwise. Stratification is on the
composite key `seed_label * 1000 + sub_label`, so both hierarchy levels stay
balanced.

The split is driven entirely by `cfg.seed`, so **every variant in a suite sees
the byte-identical partition**. Comparing variants trained on different splits
would confound the architecture change with the split. Indices and class
mappings are written to `split_manifest.npz` so an evaluation can be reproduced
later.

### The optimizer covers encoder *and* head

`build_optimizer()` takes a list of modules and de-duplicates parameters by
identity. The encoder must be included even in the frozen recipe, because it owns
the Eq. 4 projection to `z`, which is trainable. Omitting it would silently
freeze the one layer adapting 1024 backbone channels to the head's 384 — the head
would train against a random projection, and nothing would report an error.

### Epoch loop

`EpochAccumulator` collects predictions, scores, embeddings and expert indices
across the whole epoch, and metrics are computed once at the end. This is
deliberate: averaging per-batch accuracy is not the same number as accuracy over
the epoch, and the paper reports the latter.

The ArcFace margin is applied **only during training** — evaluation passes no
labels, so reported metrics are not inflated by the margin.

`LossHistory` records train and validation loss per epoch. The gap between them
(`epoch/overfitting_gap`) is logged as a scalar and the two curves are drawn side
by side: on a 27-class problem with a few hundred samples per class, a validation
loss turning upward while training loss keeps falling is the failure mode most
worth watching for.

### What gets logged

Every epoch: total loss and each weighted component, seed-type and sub-variety
accuracy / F1 / precision / recall / AUC, per-class F1, the KL alignment rate
overall and per seed type, per-expert utilisation, learning rate, epoch duration
and the overfitting gap.

Every `tracking.intervals.figure_every_epochs` (and on the final epoch, and for
the test split): confusion matrices at both levels, the sub-variety metric
heatmap, per-sub-variety misclassification rates, expert-utilisation bars,
train-vs-validation loss curves, and t-SNE projections coloured by seed type and
by sub-variety with class names overlaid.

`log_evaluation_artifacts` catches its own exceptions and logs a warning —
figures are diagnostics and must never abort a training run.

### Efficiency and run artifacts

After training, `profile_run()` measures the best checkpoint: total vs. active
parameters, FLOPs, latency, throughput, peak memory. It profiles **encoder and
head together**, because that is the only combination whose latency a deployment
would observe; profiling the head alone would report a number no user could ever
see.

`write_run_summary()` then writes `summary.json` and `test_predictions.npz`,
which is the contract `scripts/generate_plots.py` reads. Both are written even
when there is no held-out split, so a run always leaves a machine-readable trace.

### Checkpoints

`best_hierarchical_moe.pth` tracks the lowest validation loss across all folds;
the best model is then evaluated on the held-out test set and saved as
`hierarchical_moe_final.pth`. Each payload carries the model state, the **encoder
state** (the Eq. 4 projection is trained here, so a head-only checkpoint would be
unusable), the class mappings, and the sub-variety → seed-type map.

## `runner.py`

`VariantSpec` names a run and its overrides; `run_suite()` launches each as a
**subprocess** and `write_suite_manifest()` records what ran with what outcome.

Subprocesses rather than an in-process loop, for three reasons in order of
importance:

1. Hydra can only be initialised once per process. Six variants in one process
   would need `GlobalHydra.instance().clear()` between them, which leaves stale
   config state behind — precisely the sort of thing that makes an ablation table
   quietly wrong.
2. GPU memory is released completely when a process exits.
3. A crash in one variant cannot take the suite down; the runner records the
   failure and continues, unless `--stop-on-failure` is passed.

Each variant gets its own `experiment.training.save_path` *and* `hydra.run.dir`,
so one variant is one self-contained folder rather than a scatter across the
global timestamped tree. Caller-supplied overrides are appended **last**, because
Hydra takes the rightmost value — a command-line override must win over the
suite's defaults.

## Adding a stage

Add `conf/experiment/<name>.yaml` with `# @package _global_`, override the
`model/head` and `model/loss` groups it needs, and register it in `main.py` or in
a runner's variant list. `baseline_hierarchical_cce` is the extreme worked
example: it adds a whole baseline with **no Python at all**, purely by flipping
four component toggles on the existing head.
