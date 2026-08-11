# `src/utils/training/` — training-loop infrastructure

Everything a training run needs that is not a model, a loss, or a dataset.

| File | Contents |
| --- | --- |
| `tracker.py` | `ExperimentTracker` — W&B + TensorBoard + JSONL fan-out |
| `checkpoint.py` | `CheckpointManager`, `to_cpu_state_dict` |
| `resume.py` | Full-state checkpoints, atomic writes, RNG capture, `InterruptGuard` |
| `distributed.py` | `DistributedContext`, process-group lifecycle, collectives |
| `experiment_logging.py` | `setup_experiment_logger`, `JsonlLogHandler` |
| `snapshot.py` | `snapshot_run_configuration` |
| `device.py` | `select_device`, capability probing, AMP and compile resolution |
| `ema.py` | `TeacherEmaUpdater` — fused foreach teacher update |
| `attention.py` | `log_attention_maps` |

## `distributed.py`

One module owns every decision about whether this process is part of a
distributed job, which backend it speaks, and what a collective means for each
quantity the objective computes. Callers ask for a `DistributedContext` once and
branch on it; nothing else in the repository touches `torch.distributed`.

**Images are sharded; views are not.** `DistributedSampler` splits the dataset,
and each rank builds all `2 + local_crops_number` views of its own images.
That is forced by the objective: Eq. 1 scores every cross-view pair of one image
against the teacher's output for *that same image*, so a sample's views must be
resident on one device.

**Gradients are averaged; batch statistics are per-rank by default.** The DINO
loss is a mean over samples, so `mean_r(grad over shard r)` equals the gradient
over the concatenated batch exactly, and DDP's all-reduce is all the
synchronisation the gradient needs. Sinkhorn and KoLeo are *not* means over
samples — but they are already computed **per micro-batch** under gradient
accumulation, so holding the per-rank micro-batch fixed leaves both functions
exactly as they are on one GPU. That is why `effective_batch_size` derives the
accumulation count from the world size rather than changing the micro-batch.

`logsumexp_across_ranks` exists for the opt-in `distributed_sinkhorn` path,
where the assignment is instead made doubly stochastic over the whole global step
batch. It is exact: the prototype marginal reduces along the sharded axis and so
needs a cross-rank `logsumexp`, and `tests/test_distributed.py` pins the result
against a single-process reference on the concatenated batch.

**The EMA centre is not optional.** `centering="ema"` keeps a running buffer that
the teacher's targets read, so per-rank buffers would mean ranks training against
different targets and a checkpoint whose contents depend on which rank wrote it.
`update_center` all-reduces unconditionally.

`resolve_num_workers` divides the configured worker budget by the local rank
count. `num_workers` is per-process, so on a Kaggle T4x2 instance — 4 vCPUs,
2 ranks — the literal 8 would put 16 augmentation workers on 4 cores.

`strip_wrapper_prefixes` removes `module.` / `_orig_mod.` from every key of a
state dict. Both wrappers are kept off the module tree here so this repository's
own checkpoints never carry a prefix, but a checkpoint from elsewhere may — and
under `checkpoint_strict: false` a prefixed key set loads as **zero matches**,
one log line, and a full run against random weights.

## `resume.py`

A checkpoint that carries only weights is a snapshot, not a checkpoint.
Restarting from one restarts AdamW's moments at zero, the LR schedule at its
peak, the teacher momentum at 0.996 and the RNG wherever the new process seeded
— none of which errors, and all of which produce a different run.

`build_checkpoint_payload` collects every named component's `state_dict`, the
`TrainingProgress` (epoch, global step, **micro-batch within the epoch**), one
RNG snapshot **per rank** (Python, NumPy, torch CPU, every CUDA device, and the
dataloader's own generator), the resolved config and the distributed layout.

`atomic_save` writes to a sibling temp file and promotes it with `os.replace`.
`torch.save` truncates its destination first, so a session killed mid-save
otherwise leaves a zero-length file *where the good checkpoint used to be*.
`COMPLETE_KEY` is written last inside the payload as a second line of defence,
and `find_latest_checkpoint` walks newest-first skipping anything that fails to
load — which is why `keep_last_n` should be ≥ 2 on a preemptible platform.

`InterruptGuard` turns SIGTERM/SIGINT and a wall-clock budget into a
checkpoint-and-exit. **Nothing is saved from inside the handler**: a signal can
arrive mid-CUDA-call, and re-entering the allocator from a handler turns a clean
shutdown into a hang. The handler sets a flag; the loop acts on it at the next
accumulation boundary. `max_runtime_minutes` is the more reliable of the two,
because not every platform signals before it kills.

## `tracker.py`

See [`../README.md`](../README.md) for the full method list. The design
rule: **one call, three sinks**, and a missing optional backend is a warning, not
a crash. `events.jsonl` is always written and flushed per record, so it survives
a run that dies mid-epoch — which is exactly when you most want the metrics.

`log_metrics` drops non-scalar values silently, letting callers pass a mixed dict
without filtering. `log_figure` writes a PNG to `<run_dir>/figures/`, pushes to
both trackers, then closes the figure so a long run does not accumulate
canvases.

## `checkpoint.py`

`save(filename, payload, rolling_prefix=...)` writes the payload atomically (via
`resume.atomic_save`); when `rolling_prefix` is given it then prunes files
matching that prefix down to `keep_last_n`, newest-first by mtime, skipping
in-progress `.tmp` writes. Named artifacts saved *without* a prefix ("best",
"final") are never pruned.

`CheckpointManager(..., enabled=context.is_main)` makes every write a no-op that
still reports its path, so call sites need no rank branch of their own. Under
DDP every rank holds identical parameters, so a second writer is at best wasted
bandwidth and at worst two processes interleaving into one file.

`to_cpu_state_dict` moves tensors off the accelerator before saving, so a
checkpoint written on CUDA or MPS loads anywhere.

## `experiment_logging.py`

Configures a named logger with up to three handlers: a rotating-free file handler
(`training.log`), a console handler, and `JsonlLogHandler` (`training.log.jsonl`).

The JSONL handler serialises the standard record fields plus any custom `extra=`
keys, falling back to `str()` for anything non-serialisable. That is what makes
`logger.info("...", extra={"snapshots": {...}})` produce a machine-readable line
rather than dropping the payload.

Existing handlers are removed and closed on setup, so re-running in one process
(a notebook, a test) does not duplicate every log line.

## `snapshot.py`

Writes the **fully resolved** config (YAML and JSON), the CLI arguments, and
optionally the environment into `<run_dir>/snapshots/`. Resolved matters: it
captures what interpolations and `oc.env` lookups actually produced, which is
what makes a run reproducible from its artifacts alone.

Environment capture is off by default and redacts any key containing `KEY`,
`TOKEN`, `SECRET`, `PASSWORD`, `PASS` or `CREDENTIAL`.

## `device.py`

`select_device("auto")` prefers CUDA, then MPS, then CPU, so the code runs on
Apple Silicon unchanged. An explicit request that is unavailable falls back to
CPU rather than raising.

`collect_device_stats` returns platform info plus, on CUDA, allocated/reserved
memory and (via `pynvml`, if installed) GPU utilisation — logged periodically by
both trainers at `tracking.intervals.device_every_steps`.

## `attention.py`

`log_attention_maps` extracts the last block's CLS attention, averages over
heads, reshapes to a square map, upsamples to the input resolution, and
normalises per image before handing it to the tracker (paper Fig. 9).

It first tries `backbone.get_last_selfattention` (the DINO ViT API), then falls
back to a forward hook on `attn_drop`, temporarily disabling `fused_attn` because
PyTorch's fused kernel never materialises the attention matrix. Backbones that
expose neither path log a warning and are skipped — Swin's windowed attention
does not produce a single global CLS map, so this is primarily a ViT tool.

Off by default (`tracking.artifacts.log_attention_maps`); the hook forces the
slow attention path.
