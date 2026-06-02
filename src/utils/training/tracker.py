from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf


class ExperimentTracker:
    def __init__(self, cfg: DictConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.output_dir = Path(OmegaConf.select(cfg, "tracking.output_dir", default="outputs/run"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.output_dir / "events.jsonl"
        self._events_file = self.events_path.open("a", encoding="utf-8")
        self.writer = None
        self.wandb = None
        self.wandb_run = None

        self._init_tensorboard()
        self._init_wandb()

    @property
    def enabled_for_images(self) -> bool:
        return self.writer is not None or self.wandb_run is not None

    def _init_tensorboard(self) -> None:
        if not OmegaConf.select(self.cfg, "tracking.tensorboard.enabled", default=False):
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as exc:
            self.logger.warning("TensorBoard is enabled but unavailable: %s", exc)
            return

        log_dir = Path(OmegaConf.select(self.cfg, "tracking.tensorboard.log_dir"))
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

        self.wandb = wandb
        config = OmegaConf.to_container(self.cfg, resolve=True)
        self.wandb_run = wandb.init(
            project=OmegaConf.select(self.cfg, "tracking.wandb.project", default=None),
            entity=OmegaConf.select(self.cfg, "tracking.wandb.entity", default=None),
            name=OmegaConf.select(self.cfg, "tracking.run_name", default=None),
            group=OmegaConf.select(self.cfg, "tracking.wandb.group", default=None),
            tags=list(OmegaConf.select(self.cfg, "tracking.wandb.tags", default=[])),
            mode=OmegaConf.select(self.cfg, "tracking.wandb.mode", default="offline"),
            config=config,
        )
        self.logger.info("Weights & Biases logging enabled.")

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        if self.wandb_run is not None:
            self.wandb.finish()
        self._events_file.close()

    def log_event(self, event_type: str, payload: dict[str, Any]) -> None:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            **self._json_safe(payload),
        }
        self._events_file.write(json.dumps(event, sort_keys=True) + "\n")
        self._events_file.flush()

    def log_metrics(self, metrics: dict[str, Any], step: int, prefix: str | None = None) -> None:
        clean = {
            f"{prefix}/{key}" if prefix else key: self._to_scalar(value)
            for key, value in metrics.items()
            if self._to_scalar(value) is not None
        }
        if not clean:
            return

        self.log_event("metrics", {"step": step, "metrics": clean})
        if self.writer is not None:
            for key, value in clean.items():
                self.writer.add_scalar(key, value, step)
        if self.wandb_run is not None:
            self.wandb.log(clean, step=step)

    def log_model_watch(self, model: torch.nn.Module) -> None:
        if self.wandb_run is None:
            return
        try:
            self.wandb.watch(model, log="all", log_freq=int(OmegaConf.select(self.cfg, "tracking.intervals.log_every_steps", default=10)))
        except Exception as exc:
            self.logger.warning("Unable to watch model with W&B: %s", exc)

    def log_parameter_histograms(self, model: torch.nn.Module, step: int, prefix: str = "parameters") -> None:
        if not OmegaConf.select(self.cfg, "tracking.artifacts.log_parameter_histograms", default=True):
            return
        for name, parameter in model.named_parameters():
            if parameter.detach().numel() == 0:
                continue
            tag = f"{prefix}/{self._clean_name(name)}"
            values = parameter.detach().float().cpu()
            if self.writer is not None:
                self.writer.add_histogram(tag, values, step)
            if self.wandb_run is not None:
                self.wandb.log({tag: self.wandb.Histogram(values.numpy())}, step=step)

    def log_gradient_histograms(self, model: torch.nn.Module, step: int, prefix: str = "gradients") -> None:
        if not OmegaConf.select(self.cfg, "tracking.artifacts.log_gradient_histograms", default=True):
            return
        for name, parameter in model.named_parameters():
            if parameter.grad is None or parameter.grad.detach().numel() == 0:
                continue
            tag = f"{prefix}/{self._clean_name(name)}"
            values = parameter.grad.detach().float().cpu()
            if self.writer is not None:
                self.writer.add_histogram(tag, values, step)
            if self.wandb_run is not None:
                self.wandb.log({tag: self.wandb.Histogram(values.numpy())}, step=step)

    def log_gradient_norms(self, model: torch.nn.Module, step: int, prefix: str = "grad_norm") -> float:
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

    def log_images(self, tag: str, images: torch.Tensor, step: int) -> None:
        images = images.detach().cpu()
        if self.writer is not None:
            self.writer.add_images(tag, images, step)
        if self.wandb_run is not None:
            self.wandb.log({tag: [self.wandb.Image(image) for image in images]}, step=step)

    def log_embeddings(
        self,
        tag: str,
        embeddings: torch.Tensor,
        step: int,
        metadata: list[str] | None = None,
    ) -> None:
        if not OmegaConf.select(self.cfg, "tracking.artifacts.log_embeddings", default=True):
            return
        max_samples = int(OmegaConf.select(self.cfg, "tracking.artifacts.max_embedding_samples", default=256))
        vectors = embeddings.detach().float().cpu()[:max_samples]
        labels = metadata[: len(vectors)] if metadata is not None else None

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

    def log_curves(self, curves: dict[str, Any], step: int, prefix: str = "validation") -> None:
        self.log_event("curves", {"step": step, "curves": curves})
        if self.wandb_run is not None:
            for name, curve in curves.items():
                self.wandb.log({f"{prefix}/{name}": curve}, step=step)

    @staticmethod
    def _clean_name(name: str) -> str:
        return name.replace(".", "/")

    @staticmethod
    def _to_scalar(value: Any) -> float | int | None:
        if isinstance(value, (float, int)):
            return value
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            return float(value.detach().cpu().item())
        return None

    @classmethod
    def _json_safe(cls, payload: dict[str, Any]) -> dict[str, Any]:
        safe: dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, dict):
                safe[key] = cls._json_safe(value)
            elif isinstance(value, (list, tuple)):
                safe[key] = [cls._json_value(item) for item in value]
            else:
                safe[key] = cls._json_value(value)
        return safe

    @staticmethod
    def _json_value(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.detach().cpu().item())
            return value.detach().cpu().tolist()
        if hasattr(value, "tolist"):
            return value.tolist()
        try:
            json.dumps(value)
            return value
        except TypeError:
            return str(value)
