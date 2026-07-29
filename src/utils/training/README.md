# `src/utils/training/` — training-loop infrastructure

Everything a training run needs that is not a model, a loss, or a dataset.

| File | Contents |
| --- | --- |
| `tracker.py` | `ExperimentTracker` — W&B + TensorBoard + JSONL fan-out |
| `checkpoint.py` | `CheckpointManager`, `to_cpu_state_dict` |
| `experiment_logging.py` | `setup_experiment_logger`, `JsonlLogHandler` |
| `snapshot.py` | `snapshot_run_configuration` |
| `device.py` | `select_device`, `collect_device_stats` |
| `attention.py` | `log_attention_maps` |

## `tracker.py`

See [`../README.md`](../README.md) for the full method list. The design
rule: **one call, three sinks**, and a missing optional backend is a warning, not
a crash. `events.jsonl` is always written and flushed per record, so it survives
a run that dies mid-epoch — which is exactly when you most want the metrics.

`log_metrics` drops non-scalar values silently, letting callers pass a mixed dict
without filtering. `log_figure` writes a PNG to `<run_dir>/figures/`, pushes to
both trackers, then closes the figure so a 300-epoch run does not accumulate
canvases.

## `checkpoint.py`

`save(filename, payload, rolling_prefix=...)` writes the payload; when
`rolling_prefix` is given it then prunes files matching that prefix down to
`keep_last_n`, newest-first by mtime. Named artifacts saved *without* a prefix
("best", "final") are never pruned.

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
