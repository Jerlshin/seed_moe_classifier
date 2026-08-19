"""Experiment tracking over one API: JSONL, CSV, TensorBoard and W&B.

Every call fans out to four sinks:

* ``events.jsonl`` in the Hydra run directory -- always on, dependency-free, and
  the only sink guaranteed to survive a crashed run or an offline machine. It is
  the *complete* record: it carries the structured events (corpus digests,
  shapes, budgets, checkpoint paths) that a table cannot hold.
* ``csv/metrics_*.csv`` -- always on, one wide file per metric family. Same
  numbers as the event stream, in the shape an analysis loads directly. See
  :mod:`src.utils.training.csv_metrics` for why both exist.
* TensorBoard -- opt-in via ``tracking.tensorboard.enabled``.
* Weights & Biases -- opt-in via ``tracking.wandb.enabled``. ``mode: offline``
  is the default so a run never blocks on network or credentials; sync later
  with ``wandb sync``.

A missing optional dependency degrades to a warning rather than killing a
training run that is otherwise fine. The two always-on sinks have no optional
dependency at all -- they are stdlib ``json`` and ``csv`` -- which is deliberate:
a run on a machine with no TensorBoard and no W&B credentials still leaves a
complete, machine-readable trace.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.utils.training.csv_metrics import CsvMetricWriter


class ExperimentTracker:
    """Fan-out logger for scalars, histograms, figures, tables and artifacts.

    Args:
        enabled: ``False`` turns every method into a no-op and opens no files or
            backends. Pass ``context.is_main`` under DDP: the ranks hold
            identical parameters and compute identical metrics, so a second
            writer produces a W&B run per GPU, interleaved TensorBoard events
            for the same steps, and two processes appending to one
            ``events.jsonl``. The non-main ranks still get their Python logger,
            so a crash on rank 1 is still visible.
    """

    def __init__(self, cfg: DictConfig, logger: logging.Logger, enabled: bool = True):
        self.cfg = cfg
        self.logger = logger
        self.enabled = bool(enabled)
        self.output_dir = Path(OmegaConf.select(cfg, "tracking.output_dir", default="outputs/run"))
        self.figure_dir = self.output_dir / "figures"
        self.csv_dir = Path(
            OmegaConf.select(cfg, "tracking.csv.directory", default=None)
            or self.output_dir / "csv"
        )
        self.events_path = self.output_dir / "events.jsonl"
        self._events_file = None
        self.writer = None
        self.wandb = None
        self.wandb_run = None
        self.csv = CsvMetricWriter(self.csv_dir, enabled=False, logger=logger)

        if not self.enabled:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._events_file = self.events_path.open("a", encoding="utf-8")
        self._init_csv()
        self._init_tensorboard()
        self._init_wandb()

    # ------------------------------------------------------------------ setup

    @property
    def enabled_for_images(self) -> bool:
        """True when at least one sink can receive images or figures."""
        return self.writer is not None or self.wandb_run is not None

    def _init_csv(self) -> None:
        if not OmegaConf.select(self.cfg, "tracking.csv.enabled", default=True):
            self.logger.info("CSV metric mirroring is disabled; events.jsonl is the only table.")
            return
        self.csv = CsvMetricWriter(
            self.csv_dir,
            enabled=True,
            elapsed_start=time.perf_counter(),
            logger=self.logger,
        )
        self.logger.info("CSV metrics enabled at %s", self.csv_dir)

    def _init_tensorboard(self) -> None:
        if not OmegaConf.select(self.cfg, "tracking.tensorboard.enabled", default=False):
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as exc:
            self.logger.warning("TensorBoard is enabled but unavailable: %s", exc)
            return

        log_dir = Path(
            OmegaConf.select(
                self.cfg,
                "tracking.tensorboard.log_dir",
                default=str(self.output_dir / "tensorboard"),
            )
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_dir))
        self.logger.info("TensorBoard logging enabled at %s", log_dir)

    def _init_wandb(self) -> None:
        if not OmegaConf.select(self.cfg, "tracking.wandb.enabled", default=False):
            return
        try:
            import wandb
        except Exception as exc:
            self.logger.warning("Weights & Biases is enabled but unavailable: %s", exc)
            return

        try:
            self.wandb_run = wandb.init(
                project=OmegaConf.select(self.cfg, "tracking.wandb.project", default=None),
                entity=OmegaConf.select(self.cfg, "tracking.wandb.entity", default=None),
                name=OmegaConf.select(self.cfg, "tracking.run_name", default=None),
                group=OmegaConf.select(self.cfg, "tracking.wandb.group", default=None),
                tags=list(OmegaConf.select(self.cfg, "tracking.wandb.tags", default=[]) or []),
                mode=OmegaConf.select(self.cfg, "tracking.wandb.mode", default="offline"),
                dir=str(self.output_dir),
                config=OmegaConf.to_container(self.cfg, resolve=True),
            )
        except Exception as exc:
            # An unreachable or unauthenticated W&B must not abort training.
            self.logger.warning("Unable to initialise Weights & Biases: %s", exc)
            return

        self.wandb = wandb
        self.logger.info("Weights & Biases logging enabled (mode=%s).", self.wandb_run.settings.mode)

    def close(self) -> None:
        """Flush and close every sink. Safe to call more than once."""
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None
        if self.wandb_run is not None:
            self.wandb.finish()
            self.wandb_run = None
        self.csv.close()
        if self._events_file is not None and not self._events_file.closed:
            self._events_file.close()

    def __enter__(self) -> "ExperimentTracker":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # ---------------------------------------------------------------- scalars

    def log_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Append a structured record to ``events.jsonl``."""
        if not self.enabled or self._events_file is None:
            return
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            **self._json_safe(payload),
        }
        self._events_file.write(json.dumps(event, sort_keys=True) + "\n")
        self._events_file.flush()

    def log_metrics(self, metrics: Mapping[str, Any], step: int, prefix: str | None = None) -> None:
        """Log scalar metrics to every sink, dropping non-scalar values."""
        clean = {
            f"{prefix}/{key}" if prefix else key: scalar
            for key, value in metrics.items()
            if (scalar := self._to_scalar(value)) is not None
        }
        if not clean:
            return

        self.log_event("metrics", {"step": step, "metrics": clean})
        # The CSV sink keeps the prefix as the *family* and strips it from the
        # column names, so `metrics_train.csv` has a `loss` column rather than a
        # `train/loss` one -- the prefix is already the filename.
        self.csv.log(
            {key.split("/", 1)[1] if prefix and key.startswith(f"{prefix}/") else key: value
             for key, value in clean.items()},
            step,
            prefix=prefix,
        )
        if self.writer is not None:
            for key, value in clean.items():
                self.writer.add_scalar(key, value, step)
        if self.wandb_run is not None:
            self.wandb.log(clean, step=step)

    def log_hyperparameters(self, params: Mapping[str, Any], metrics: Mapping[str, float]) -> None:
        """Record the hyperparameter/metric pairing TensorBoard's HParams tab uses."""
        if not self.enabled:
            return
        self.log_event("hyperparameters", {"params": dict(params), "metrics": dict(metrics)})
        if self.writer is None:
            return
        scalar_params = {
            key: value
            for key, value in params.items()
            if isinstance(value, (int, float, str, bool))
        }
        try:
            self.writer.add_hparams(scalar_params, dict(metrics))
        except Exception as exc:
            self.logger.warning("Unable to log hyperparameters to TensorBoard: %s", exc)

    # ------------------------------------------------------------- model dumps

    def log_model_watch(self, model: torch.nn.Module) -> None:
        """Ask W&B to track parameter and gradient distributions."""
        if not self.enabled or self.wandb_run is None:
            return
        try:
            self.wandb.watch(
                model,
                log="all",
                log_freq=int(OmegaConf.select(self.cfg, "tracking.intervals.log_every_steps", default=10)),
            )
        except Exception as exc:
            self.logger.warning("Unable to watch model with W&B: %s", exc)

    def log_parameter_histograms(self, model: torch.nn.Module, step: int, prefix: str = "parameters") -> None:
        if not self.enabled:
            return
        if not OmegaConf.select(self.cfg, "tracking.artifacts.log_parameter_histograms", default=False):
            return
        for name, parameter in model.named_parameters():
            if parameter.numel() == 0:
                continue
            self.log_histogram(f"{prefix}/{self._clean_name(name)}", parameter.detach(), step)

    def log_gradient_histograms(self, model: torch.nn.Module, step: int, prefix: str = "gradients") -> None:
        if not self.enabled:
            return
        if not OmegaConf.select(self.cfg, "tracking.artifacts.log_gradient_histograms", default=False):
            return
        for name, parameter in model.named_parameters():
            if parameter.grad is None or parameter.grad.numel() == 0:
                continue
            self.log_histogram(f"{prefix}/{self._clean_name(name)}", parameter.grad.detach(), step)

    def log_gradient_norms(self, model: torch.nn.Module, step: int, prefix: str = "grad_norm") -> float:
        """Log the per-parameter and total L2 gradient norms; return the total.

        Costs one host synchronisation **per parameter tensor**, so the rank
        guard here is a throughput decision as much as a tidiness one: without
        it every rank of a DDP job would stall ~440 times per logged step to
        produce numbers only rank 0 would write.
        """
        if not self.enabled:
            return 0.0
        total_sq = 0.0
        norms: dict[str, float] = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            norm = float(parameter.grad.detach().data.norm(2).item())
            total_sq += norm**2
            norms[f"{prefix}/{self._clean_name(name)}"] = norm
        total = total_sq**0.5
        norms[f"{prefix}/total"] = total
        self.log_metrics(norms, step)
        return total

    # ------------------------------------------------------------- rich media

    def log_histogram(self, tag: str, values: Any, step: int) -> None:
        """Log a distribution to whichever sinks accept histograms."""
        if not self.enabled:
            return
        tensor = values.detach().float().cpu() if isinstance(values, torch.Tensor) else torch.as_tensor(values).float()
        if tensor.numel() == 0:
            return
        if self.writer is not None:
            self.writer.add_histogram(tag, tensor, step)
        if self.wandb_run is not None:
            self.wandb.log({tag: self.wandb.Histogram(tensor.numpy())}, step=step)

    def log_images(self, tag: str, images: torch.Tensor, step: int) -> None:
        if not self.enabled:
            return
        images = images.detach().cpu()
        if self.writer is not None:
            self.writer.add_images(tag, images, step)
        if self.wandb_run is not None:
            self.wandb.log({tag: [self.wandb.Image(image) for image in images]}, step=step)

    def log_figure(self, tag: str, figure: Any, step: int, save: bool = True) -> str | None:
        """Log a matplotlib figure and optionally persist it under ``figures/``.

        Returns the saved path, or ``None`` when ``save`` is off. The figure is
        always closed afterwards so long runs do not leak canvases.
        """
        if not self.enabled:
            return None
        import matplotlib.pyplot as plt

        saved_path: str | None = None
        if save:
            from src.utils.visualization import save_figure

            filename = f"{tag.replace('/', '_')}_step{step:06d}.png"
            saved_path = save_figure(figure, self.figure_dir / filename, close=False)

        if self.writer is not None:
            self.writer.add_figure(tag, figure, step, close=False)
        if self.wandb_run is not None:
            self.wandb.log({tag: self.wandb.Image(figure)}, step=step)

        plt.close(figure)
        if saved_path:
            self.log_event("figure", {"tag": tag, "step": step, "path": saved_path})
        return saved_path

    def log_table(self, tag: str, columns: Sequence[str], rows: Sequence[Sequence[Any]], step: int) -> None:
        """Log a tabular artifact (per-class metrics, alignment breakdowns)."""
        if not self.enabled:
            return
        payload = [dict(zip(columns, row)) for row in rows]
        self.log_event("table", {"tag": tag, "step": step, "rows": payload})
        self.csv.write_table(tag.replace("/", "_"), payload, columns=list(columns))
        if self.wandb_run is not None:
            table = self.wandb.Table(columns=list(columns), data=[list(row) for row in rows])
            self.wandb.log({tag: table}, step=step)
        if self.writer is not None:
            self.writer.add_text(tag, self._markdown_table(columns, rows), step)

    def log_embeddings(
        self,
        tag: str,
        embeddings: torch.Tensor,
        step: int,
        metadata: Sequence[str] | None = None,
    ) -> None:
        """Log an embedding set to TensorBoard's projector and a W&B table."""
        if not self.enabled:
            return
        if not OmegaConf.select(self.cfg, "tracking.artifacts.log_embeddings", default=False):
            return
        max_samples = int(OmegaConf.select(self.cfg, "tracking.artifacts.max_embedding_samples", default=256))
        vectors = embeddings.detach().float().cpu()[:max_samples]
        if vectors.numel() == 0:
            return
        labels = list(metadata[: len(vectors)]) if metadata is not None else None

        if self.writer is not None:
            self.writer.add_embedding(vectors, metadata=labels, global_step=step, tag=tag)
        if self.wandb_run is not None:
            table = self.wandb.Table(
                columns=["label", *[f"dim_{i}" for i in range(vectors.shape[1])]],
                data=[
                    [labels[i] if labels else str(i), *row.tolist()]
                    for i, row in enumerate(vectors)
                ],
            )
            self.wandb.log({tag: table}, step=step)

    def write_csv(
        self,
        name: str,
        rows: Sequence[Mapping[str, Any]],
        columns: Sequence[str] | None = None,
    ) -> str | None:
        """Write one standalone CSV table under ``csv/``.

        For records that are neither a time series nor a step-indexed metric --
        the view-geometry report, the checkpoint-selection ledger, the per-class
        probe breakdown. Returns the path, or ``None`` when CSV output is off.
        """
        if not self.enabled:
            return None
        path = self.csv.write_table(name, rows, columns=columns)
        if path:
            self.log_event("csv_table", {"name": name, "path": path, "rows": len(rows)})
        return path

    def log_artifact(self, path: str | Path, name: str, artifact_type: str = "model") -> None:
        """Upload a file to W&B as a versioned artifact when enabled."""
        if not self.enabled or self.wandb_run is None or not OmegaConf.select(
            self.cfg, "tracking.wandb.log_model", default=False
        ):
            return
        try:
            artifact = self.wandb.Artifact(name=name, type=artifact_type)
            artifact.add_file(str(path))
            self.wandb_run.log_artifact(artifact)
        except Exception as exc:
            self.logger.warning("Unable to log artifact %s to W&B: %s", name, exc)

    # ------------------------------------------------------------- conversion

    @staticmethod
    def _markdown_table(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
        header = "| " + " | ".join(str(column) for column in columns) + " |"
        divider = "| " + " | ".join("---" for _ in columns) + " |"
        body = [
            "| " + " | ".join(f"{value:.4f}" if isinstance(value, float) else str(value) for value in row) + " |"
            for row in rows
        ]
        return "\n".join([header, divider, *body])

    @staticmethod
    def _clean_name(name: str) -> str:
        return name.replace(".", "/")

    @staticmethod
    def _to_scalar(value: Any) -> float | int | None:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (float, int)):
            return value
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            return float(value.detach().cpu().item())
        return None

    @classmethod
    def _json_safe(cls, payload: Mapping[str, Any]) -> dict[str, Any]:
        safe: dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, Mapping):
                safe[key] = cls._json_safe(value)
            elif isinstance(value, (list, tuple)):
                safe[key] = [cls._json_value(item) for item in value]
            else:
                safe[key] = cls._json_value(value)
        return safe

    @staticmethod
    def _json_value(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item()) if value.numel() == 1 else value.detach().cpu().tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Mapping):
            return {str(k): ExperimentTracker._json_value(v) for k, v in value.items()}
        try:
            json.dumps(value)
            return value
        except TypeError:
            return str(value)
