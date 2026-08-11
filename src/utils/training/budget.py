"""Stage-1 compute and parameter budget, in a form a paper table can consume.

A self-supervised stage is judged partly on what it cost, and "cost" is four
different quantities that are routinely conflated: parameters resident,
parameters trained, arithmetic performed, and wall clock. This module reports
all four, and -- the part that matters more than the numbers -- **labels which
were measured and which were derived**.

Measured
    Parameter counts (exact, by traversal). GFLOPs per view, from torch's
    dispatch-level ``FlopCounterMode`` on one real forward at the run's own
    resolution, so it reflects the model that will actually train rather than a
    formula's idea of it. Peak allocated and reserved VRAM, wall clock,
    throughput -- all read from the run itself.

Estimated
    Per-iteration and whole-run FLOPs. These multiply the measured per-view
    figure by view counts and by a **backward multiplier of 2**, which is the
    standard convention (one backward computes the input gradient and the
    parameter gradient, each about the cost of the forward) and is an
    approximation, not a measurement: it ignores recomputation under gradient
    checkpointing, the optimizer's own arithmetic, and the loss. Anything
    derived this way carries ``estimated`` in its name and an explicit note in
    the printed table.

The distinction is the whole point of the module. A table that prints
``1.2 EFLOPs`` next to ``27.58 M parameters`` in the same typeface invites a
reader to trust both equally, and only one of them is a fact.

Nothing here is on the training path. :func:`measure_gflops_per_view` runs once
at startup under ``no_grad``; everything else is arithmetic over numbers the
trainer already has.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn as nn

#: Backward cost as a multiple of the forward. The conventional figure: the
#: backward pass computes an input gradient and a parameter gradient, each of
#: roughly the forward's cost. Named rather than inlined because every
#: "estimated" number in this module rests on it, and a reader deserves to see
#: which assumption they are trusting.
BACKWARD_MULTIPLIER = 2.0

BYTES_PER_GIB = 1024.0**3


def measure_gflops_per_view(
    module: nn.Module,
    image_size: int,
    device: torch.device,
    channels: int = 3,
) -> float | None:
    """Forward GFLOPs for **one** view through ``module``, or ``None``.

    Counts real ATen dispatch rather than applying a closed form, so it stays
    correct across the SDPA rewrite, ``channels_last`` and any timm version
    change -- none of which a hand-derived formula would notice.

    Returns ``None`` rather than raising if the counter is unavailable (older
    torch) or the forward fails: a budget line is never worth aborting a
    training run for. Batch 1 keeps the result a genuine per-view figure with no
    division, and the probe runs under ``no_grad`` in eval mode so nothing here
    perturbs the model that is about to train.
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except Exception:  # pragma: no cover - depends on the torch build
        return None

    try:
        counter = FlopCounterMode(display=False)
    except TypeError:  # pragma: no cover - older signature
        counter = FlopCounterMode()

    was_training = module.training
    module.eval()
    try:
        probe = torch.zeros(1, channels, int(image_size), int(image_size), device=device)
        with torch.no_grad(), counter:
            module(probe)
        return float(counter.get_total_flops()) / 1e9
    except Exception:
        return None
    finally:
        module.train(was_training)


@dataclass
class StageOneBudget:
    """Everything the stage-1 cost table needs, measured and estimated apart.

    Built in two halves. Construct it after the model exists (parameters, FLOPs,
    configuration) and call :meth:`record_runtime` when the run finishes; the
    runtime fields stay ``None`` until then, so an interrupted run still prints
    a valid parameter and compute report.
    """

    # ------------------------------------------------------------- measured
    backbone_parameters: int = 0
    dino_head_parameters: int = 0
    student_parameters: int = 0
    student_trainable_parameters: int = 0
    teacher_parameters: int = 0
    prototype_layer_parameters: int = 0

    gflops_per_view: float | None = None
    image_size: int = 256

    # ----------------------------------------------------------- configured
    views_per_image: int = 6
    global_views_per_image: int = 2
    epochs: int = 100
    physical_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    world_size: int = 1
    effective_batch_size: int = 32
    steps_per_epoch: int = 0
    images_per_epoch: int = 0
    precision: str = "off"
    optimizer: str = "AdamW"
    learning_rate: float = 0.0
    weight_decay: float = 0.0
    backbone_name: str = ""
    pretrained_init: str = "random"
    drop_path_rate: float = 0.0
    prototypes: int = 0

    # -------------------------------------------------------------- runtime
    gpu_name: str | None = None
    gpu_memory_gb: float | None = None
    peak_allocated_gb: float | None = None
    peak_reserved_gb: float | None = None
    training_seconds: float | None = None
    images_per_second: float | None = None
    views_per_second: float | None = None

    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------ derived

    @property
    def views_per_iteration(self) -> int:
        """Student + teacher views behind one optimizer step, across all ranks.

        The teacher's globals are counted, because they are a real forward even
        though they take no gradient -- omitting them understates the step by
        about a quarter at 2 global + 4 local.
        """
        images = self.effective_batch_size
        return images * (self.views_per_image + self.global_views_per_image)

    @property
    def estimated_gflops_per_iteration(self) -> float | None:
        """Forward+backward for the student, forward only for the teacher.

        ESTIMATE. See :data:`BACKWARD_MULTIPLIER`.
        """
        if self.gflops_per_view is None:
            return None
        images = self.effective_batch_size
        student = images * self.views_per_image * (1.0 + BACKWARD_MULTIPLIER)
        teacher = images * self.global_views_per_image  # no_grad: forward only
        return self.gflops_per_view * (student + teacher)

    @property
    def estimated_total_flops(self) -> float | None:
        """Whole-run FLOPs. ESTIMATE, and the softest number in the report."""
        per_iteration = self.estimated_gflops_per_iteration
        if per_iteration is None or not self.steps_per_epoch:
            return None
        return per_iteration * 1e9 * self.steps_per_epoch * self.epochs

    @property
    def estimated_exaflops(self) -> float | None:
        total = self.estimated_total_flops
        return None if total is None else total / 1e18

    def format_total_flops(self) -> str:
        """The whole-run total in whichever unit does not round it to zero.

        A 100-epoch run lands in the 0.1-1 EFLOPs range, but a two-batch smoke
        run is ~13 orders of magnitude smaller, and printing ``0.0000 EFLOPs``
        for it reads as "the estimate is broken" rather than "the run was tiny".
        """
        total = self.estimated_total_flops
        if total is None:
            return "not available"
        for scale, unit in ((1e18, "EFLOPs"), (1e15, "PFLOPs"), (1e12, "TFLOPs"), (1e9, "GFLOPs")):
            if total >= scale:
                return f"{total / scale:.4g} {unit}   [ESTIMATED]"
        return f"{total:.4g} FLOPs   [ESTIMATED]"

    # ------------------------------------------------------------- capture

    @classmethod
    def from_model(
        cls,
        parameter_summary: dict[str, int],
        **fields: Any,
    ) -> "StageOneBudget":
        """Build from :meth:`~src.models.backbones.swinv2_dino.DINO.parameter_summary`."""
        return cls(
            backbone_parameters=int(parameter_summary.get("backbone", 0)),
            dino_head_parameters=int(parameter_summary.get("dino_head", 0)),
            student_parameters=int(parameter_summary.get("student_total", 0)),
            student_trainable_parameters=int(parameter_summary.get("student_trainable", 0)),
            teacher_parameters=int(parameter_summary.get("teacher_total", 0)),
            prototype_layer_parameters=int(parameter_summary.get("prototype_layer", 0)),
            **fields,
        )

    def record_runtime(
        self,
        device: torch.device,
        training_seconds: float,
        images_processed: int,
        peak_allocated_gb: float | None = None,
        peak_reserved_gb: float | None = None,
    ) -> None:
        """Attach the measured runtime half once training has finished.

        ``peak_reserved`` is reported next to ``peak_allocated`` because they
        answer different questions: allocated is what the model needed, reserved
        is what the caching allocator took from the driver and therefore what
        another process on the same card actually sees missing. A run that fits
        on allocated but not on reserved still OOMs its neighbour.

        Pass the two peaks explicitly when the caller has been tracking them
        across epochs. The trainer resets CUDA's peak counters every epoch to
        report per-epoch memory, so reading them here would report the *last*
        epoch's peak rather than the run's -- which is lower, and wrong in the
        direction that gets a relaunch OOM-killed.
        """
        self.training_seconds = float(training_seconds)
        if training_seconds > 0:
            self.images_per_second = images_processed / training_seconds
            self.views_per_second = (
                images_processed * (self.views_per_image + self.global_views_per_image)
                / training_seconds
            )
        if device.type == "cuda" and torch.cuda.is_available():
            index = device.index if device.index is not None else torch.cuda.current_device()
            self.gpu_name = torch.cuda.get_device_name(index)
            self.gpu_memory_gb = torch.cuda.get_device_properties(index).total_memory / BYTES_PER_GIB
            self.peak_allocated_gb = (
                float(peak_allocated_gb)
                if peak_allocated_gb is not None
                else torch.cuda.max_memory_allocated(index) / BYTES_PER_GIB
            )
            self.peak_reserved_gb = (
                float(peak_reserved_gb)
                if peak_reserved_gb is not None
                else torch.cuda.max_memory_reserved(index) / BYTES_PER_GIB
            )
        else:
            self.notes.append(
                f"Peak VRAM is not tracked on device type {device.type!r}; "
                "memory figures are unavailable for this run."
            )

    # -------------------------------------------------------------- output

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "views_per_iteration": self.views_per_iteration,
                "estimated_gflops_per_iteration": self.estimated_gflops_per_iteration,
                "estimated_total_flops": self.estimated_total_flops,
                "estimated_exaflops": self.estimated_exaflops,
                "backward_multiplier": BACKWARD_MULTIPLIER,
            }
        )
        return payload

    def as_metrics(self, prefix: str = "budget") -> dict[str, float]:
        """Flatten into scalars for W&B / TensorBoard.

        Estimated quantities keep ``estimated_`` in their key, so a chart axis
        cannot quietly present a derived number as a measured one.
        """
        metrics: dict[str, float] = {
            f"{prefix}/backbone_parameters_m": self.backbone_parameters / 1e6,
            f"{prefix}/dino_head_parameters_m": self.dino_head_parameters / 1e6,
            f"{prefix}/student_parameters_m": self.student_parameters / 1e6,
            f"{prefix}/student_trainable_parameters_m": self.student_trainable_parameters / 1e6,
            f"{prefix}/teacher_parameters_m": self.teacher_parameters / 1e6,
            f"{prefix}/prototype_layer_parameters_m": self.prototype_layer_parameters / 1e6,
            f"{prefix}/views_per_image": float(self.views_per_image),
            f"{prefix}/views_per_iteration": float(self.views_per_iteration),
            f"{prefix}/epochs": float(self.epochs),
            f"{prefix}/physical_batch_size": float(self.physical_batch_size),
            f"{prefix}/gradient_accumulation_steps": float(self.gradient_accumulation_steps),
            f"{prefix}/effective_batch_size": float(self.effective_batch_size),
            f"{prefix}/world_size": float(self.world_size),
            f"{prefix}/learning_rate": float(self.learning_rate),
            f"{prefix}/weight_decay": float(self.weight_decay),
            f"{prefix}/drop_path_rate": float(self.drop_path_rate),
            f"{prefix}/prototypes": float(self.prototypes),
        }
        optional = {
            f"{prefix}/gflops_per_view": self.gflops_per_view,
            f"{prefix}/estimated_gflops_per_iteration": self.estimated_gflops_per_iteration,
            f"{prefix}/estimated_total_exaflops": self.estimated_exaflops,
            f"{prefix}/gpu_memory_gb": self.gpu_memory_gb,
            f"{prefix}/peak_allocated_gb": self.peak_allocated_gb,
            f"{prefix}/peak_reserved_gb": self.peak_reserved_gb,
            f"{prefix}/training_hours": (
                self.training_seconds / 3600.0 if self.training_seconds else None
            ),
            f"{prefix}/images_per_second": self.images_per_second,
            f"{prefix}/views_per_second": self.views_per_second,
        }
        metrics.update({key: float(value) for key, value in optional.items() if value is not None})
        return metrics

    def format_table(self) -> str:
        """The copy-pasteable report. Measured and estimated are marked inline."""
        width = 60
        rule = "=" * width

        def row(label: str, value: str) -> str:
            return f"{label:<22}: {value}"

        def maybe(value: float | None, template: str, missing: str = "not available") -> str:
            return missing if value is None else template.format(value)

        lines = [
            rule,
            "STAGE-1 COMPUTE / PARAMETER BUDGET",
            rule,
            row("Backbone", f"{self.backbone_name} ({self.pretrained_init} init)"),
            "",
            "-- Parameters (measured) " + "-" * (width - 25),
            row("Backbone params", f"{self.backbone_parameters / 1e6:.2f} M"),
            row("DINO head params", f"{self.dino_head_parameters / 1e6:.2f} M"),
            row("  of which prototypes", f"{self.prototype_layer_parameters / 1e6:.2f} M"),
            row("Student params", f"{self.student_parameters / 1e6:.2f} M"),
            row("  trainable", f"{self.student_trainable_parameters / 1e6:.2f} M"),
            row("Teacher params", f"{self.teacher_parameters / 1e6:.2f} M (EMA, no gradient)"),
            "",
            "-- Compute " + "-" * (width - 11),
            row(
                "GFLOPs / view",
                maybe(self.gflops_per_view, "{:.2f}   [measured, fwd @ %d px]" % self.image_size),
            ),
            row("Views / image", f"{self.views_per_image} student + {self.global_views_per_image} teacher"),
            row("Views / iteration", f"{self.views_per_iteration}"),
            row(
                "GFLOPs / iteration",
                maybe(self.estimated_gflops_per_iteration, "{:,.0f}   [ESTIMATED]"),
            ),
            row("Total FLOPs", self.format_total_flops()),
            "",
            "-- Training configuration " + "-" * (width - 26),
            row("Epochs", f"{self.epochs}"),
            row("Physical batch", f"{self.physical_batch_size} per rank"),
            row("Grad accumulation", f"{self.gradient_accumulation_steps}"),
            row("World size", f"{self.world_size}"),
            row("Effective batch", f"{self.effective_batch_size}"),
            row("Steps / epoch", f"{self.steps_per_epoch}"),
            row("Precision", self.precision),
            row("Optimizer", self.optimizer),
            row("Learning rate", f"{self.learning_rate:.6g}"),
            row("Weight decay", f"{self.weight_decay:.4g} (start of schedule)"),
            row("Drop path", f"{self.drop_path_rate:.3g} (student only)"),
            row("Prototypes (K)", f"{self.prototypes}"),
            "",
            "-- Runtime (measured) " + "-" * (width - 22),
            row("GPU", self.gpu_name or "n/a"),
            row("GPU memory", maybe(self.gpu_memory_gb, "{:.2f} GB")),
            row("Peak allocated VRAM", maybe(self.peak_allocated_gb, "{:.2f} GB")),
            row("Peak reserved VRAM", maybe(self.peak_reserved_gb, "{:.2f} GB")),
            row(
                "Training time",
                maybe(
                    self.training_seconds / 3600.0 if self.training_seconds else None,
                    "{:.2f} h",
                    "in progress",
                ),
            ),
            row("Images / second", maybe(self.images_per_second, "{:.1f}")),
            row("Views / second", maybe(self.views_per_second, "{:.1f}")),
            rule,
            f"[ESTIMATED] assumes backward = {BACKWARD_MULTIPLIER:g}x forward and counts the",
            "teacher's global views as forward-only. Everything else is measured.",
        ]
        lines.extend(f"NOTE: {note}" for note in self.notes)
        lines.append(rule)
        return "\n".join(lines)
