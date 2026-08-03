"""Computational-efficiency measurements for the revision's cost table.

The claim a sparse MoE makes is that capacity and compute can be decoupled: the
model *owns* six experts but *evaluates* only ``K`` of them per sample. A plain
parameter count cannot express that, so this module reports two numbers side by
side:

``total_parameters``
    Everything the checkpoint stores.

``active_parameters``
    What a single forward pass actually touches: total, minus the ``(E - K)``
    experts that sit out each sample.

The difference is a closed form, not an estimate. Every expert shares one
architecture, so counting one and multiplying is exact -- see
:meth:`~src.models.components.moe_layer.MixtureOfExperts.dormant_parameters`.
Moving from ``K = 4`` to ``K = 2`` doubles the dormant count, which is precisely
the efficiency argument the revision needs to substantiate.

The remaining measurements are conventional:

* **FLOPs** via ``torch.utils.flop_counter.FlopCounterMode``, which tallies real
  dispatched matmul and convolution work rather than a hand-written formula.
  Because the counter observes actual dispatch, a sparsely-routed MoE is counted
  at its true cost automatically.
* **Peak memory** via the active accelerator's allocator statistics.
* **Latency and throughput** by timed loops with warm-up, after synchronising the
  device -- without which an asynchronous CUDA or MPS queue would be timed as
  approximately zero.

Every measurement degrades to ``None`` rather than raising when a backend cannot
supply it, so profiling never breaks a training run.
"""

from __future__ import annotations

import platform
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn as nn

from src.models.components.moe_layer import DenseExpertBlock, MixtureOfExperts

BYTES_PER_MB = 1024.0 * 1024.0


# --------------------------------------------------------------------- reports


@dataclass
class ParameterReport:
    """Total, trainable, active and dormant parameter counts."""

    total: int
    trainable: int
    active: int
    dormant: int
    num_experts: int
    top_k: int

    @property
    def total_millions(self) -> float:
        return self.total / 1e6

    @property
    def active_millions(self) -> float:
        return self.active / 1e6

    @property
    def dormant_fraction(self) -> float:
        """Share of parameters idle during any one forward pass."""
        return self.dormant / self.total if self.total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "total_millions": self.total_millions,
            "active_millions": self.active_millions,
            "dormant_fraction": self.dormant_fraction,
        }


@dataclass
class LatencyReport:
    """Timing for a single batch size.

    Reports the **median** and the interquartile range rather than a bare mean.
    A mean over ten iterations on a contended device has no dispersion estimate
    at all, and the first few iterations after warm-up are still the most likely
    to be outliers; the median is robust to both and the IQR says how much to
    trust it.
    """

    batch_size: int
    latency_ms_per_batch: float
    latency_ms_per_sample: float
    throughput_fps: float
    latency_ms_iqr: float = 0.0
    latency_ms_min: float = 0.0
    latency_ms_max: float = 0.0
    iterations: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EfficiencyReport:
    """Everything the efficiency table needs for one model."""

    name: str
    device: str
    parameters: ParameterReport
    flops: int | None = None
    flops_batch_size: int | None = None
    peak_memory_mb: float | None = None
    latencies: list[LatencyReport] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def primary_latency(self) -> LatencyReport | None:
        """The smallest benchmarked batch size, used as the headline number."""
        return min(self.latencies, key=lambda entry: entry.batch_size) if self.latencies else None

    @property
    def gflops_per_sample(self) -> float | None:
        if self.flops is None or not self.flops_batch_size:
            return None
        return self.flops / self.flops_batch_size / 1e9

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "device": self.device,
            "parameters": self.parameters.as_dict(),
            "flops": self.flops,
            "flops_batch_size": self.flops_batch_size,
            "gflops_per_sample": self.gflops_per_sample,
            "peak_memory_mb": self.peak_memory_mb,
            "latencies": [entry.as_dict() for entry in self.latencies],
            "notes": list(self.notes),
        }

    def as_metrics(self, prefix: str = "efficiency") -> dict[str, float]:
        """Flatten into scalars for the experiment tracker."""
        metrics: dict[str, float] = {
            f"{prefix}/total_parameters_m": self.parameters.total_millions,
            f"{prefix}/active_parameters_m": self.parameters.active_millions,
            f"{prefix}/dormant_fraction": self.parameters.dormant_fraction,
            f"{prefix}/top_k": float(self.parameters.top_k),
        }
        if self.gflops_per_sample is not None:
            metrics[f"{prefix}/gflops_per_sample"] = self.gflops_per_sample
        if self.peak_memory_mb is not None:
            metrics[f"{prefix}/peak_memory_mb"] = self.peak_memory_mb
        for entry in self.latencies:
            metrics[f"{prefix}/latency_ms_bs{entry.batch_size}"] = entry.latency_ms_per_sample
            metrics[f"{prefix}/throughput_fps_bs{entry.batch_size}"] = entry.throughput_fps
        return metrics

    def summary_line(self) -> str:
        """One-line human-readable form for the training log."""
        latency = self.primary_latency
        parts = [
            f"{self.name}: {self.parameters.total_millions:.2f}M total",
            f"{self.parameters.active_millions:.2f}M active "
            f"(top-{self.parameters.top_k}/{self.parameters.num_experts} experts)",
        ]
        if self.gflops_per_sample is not None:
            parts.append(f"{self.gflops_per_sample:.2f} GFLOPs/sample")
        if latency is not None:
            parts.append(f"{latency.latency_ms_per_sample:.2f} ms/sample")
            parts.append(f"{latency.throughput_fps:.1f} FPS")
        if self.peak_memory_mb is not None:
            parts.append(f"{self.peak_memory_mb:.0f} MB peak")
        return " | ".join(parts)


# ------------------------------------------------------------------ parameters


def count_parameters(*modules: nn.Module) -> tuple[int, int]:
    """Return ``(total, trainable)`` parameter counts across ``modules``.

    Parameters shared between modules are counted once, identified by object
    identity, so passing an encoder and a head that tie weights does not
    double-count them.
    """
    seen: dict[int, nn.Parameter] = {}
    for module in modules:
        if module is None:
            continue
        for parameter in module.parameters():
            seen[id(parameter)] = parameter
    total = sum(parameter.numel() for parameter in seen.values())
    trainable = sum(parameter.numel() for parameter in seen.values() if parameter.requires_grad)
    return total, trainable


def count_dormant_parameters(*modules: nn.Module) -> tuple[int, int, int]:
    """Return ``(dormant, num_experts, top_k)`` summed over every MoE found.

    Walks the module tree for :class:`MixtureOfExperts` and
    :class:`DenseExpertBlock` instances. A model with no router reports
    ``(0, 1, 1)``: nothing is dormant and there is a single always-on path,
    which is the honest description of a dense network.
    """
    dormant = 0
    num_experts = 0
    top_k = 0
    found = False

    for module in _iter_modules(*modules):
        if isinstance(module, (MixtureOfExperts, DenseExpertBlock)):
            found = True
            dormant += module.dormant_parameters()
            num_experts += int(module.num_experts)
            top_k += int(module.top_k)

    if not found:
        return 0, 1, 1
    return dormant, num_experts, top_k


def parameter_report(*modules: nn.Module) -> ParameterReport:
    """Full :class:`ParameterReport` for one or more modules."""
    total, trainable = count_parameters(*modules)
    dormant, num_experts, top_k = count_dormant_parameters(*modules)
    return ParameterReport(
        total=total,
        trainable=trainable,
        active=total - dormant,
        dormant=dormant,
        num_experts=num_experts,
        top_k=top_k,
    )


def top_k_saving(
    *modules: nn.Module,
    reference_top_k: int = 4,
) -> dict[str, float]:
    """Active-parameter saving of the configured Top-K against ``reference_top_k``.

    The revision's central efficiency claim is Top-2 versus Top-4, so this
    reports both active counts and the reduction between them. Returns an empty
    dict when the model has no router, where the comparison is meaningless.
    """
    report = parameter_report(*modules)
    per_expert = 0
    for module in _iter_modules(*modules):
        if isinstance(module, MixtureOfExperts):
            per_expert += module.parameters_per_expert()
    if per_expert == 0 or report.num_experts <= 1:
        return {}

    reference_k = min(int(reference_top_k), report.num_experts)
    reference_active = report.total - (report.num_experts - reference_k) * per_expert
    saved = reference_active - report.active
    return {
        "active_parameters": float(report.active),
        f"active_parameters_top{reference_k}": float(reference_active),
        "parameters_saved": float(saved),
        "parameters_saved_fraction": float(saved / reference_active) if reference_active else 0.0,
        "parameters_per_expert": float(per_expert),
    }


def _iter_modules(*modules: nn.Module):
    for module in modules:
        if module is None:
            continue
        yield from module.modules()


# ----------------------------------------------------------------------- FLOPs


def estimate_flops(
    model: nn.Module,
    example_input: torch.Tensor,
    forward_fn=None,
) -> int | None:
    """Total forward FLOPs for one batch, or ``None`` if unavailable.

    Uses ``torch.utils.flop_counter.FlopCounterMode``, which intercepts real ATen
    dispatch. That matters for a sparse MoE: a hand-derived formula would have to
    model the router's behaviour, whereas the dispatch counter simply observes
    that only ``K`` experts ran.

    Args:
        model: Module to profile.
        example_input: One representative batch.
        forward_fn: Callable invoked instead of ``model(example_input)``, for
            models whose forward takes extra arguments.
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except Exception:  # pragma: no cover - depends on the installed torch
        return None

    try:
        counter = FlopCounterMode(display=False)
    except TypeError:  # pragma: no cover - older signature
        counter = FlopCounterMode()

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), counter:
            forward_fn() if forward_fn is not None else model(example_input)
        return int(counter.get_total_flops())
    except Exception:
        return None
    finally:
        model.train(was_training)


# ---------------------------------------------------------------------- memory


def synchronize(device: torch.device) -> None:
    """Block until queued work on ``device`` has finished.

    Both CUDA and MPS dispatch asynchronously. Timing without this measures how
    fast Python can enqueue kernels, not how fast they run -- typically off by an
    order of magnitude.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        synchronizer = getattr(torch.mps, "synchronize", None)
        if synchronizer is not None:
            synchronizer()


def reset_peak_memory(device: torch.device) -> None:
    """Zero the allocator's peak-usage counter where the backend supports it."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb(device: torch.device) -> float | None:
    """Peak allocated memory in MB, or ``None`` on backends without the statistic.

    CUDA tracks a true high-water mark. MPS exposes only *current* allocation, so
    the number returned there is a snapshot taken right after the forward pass,
    not a peak; it is still the useful comparative figure across models on the
    same machine. CPU allocations are not tracked by torch at all.
    """
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / BYTES_PER_MB
    if device.type == "mps" and hasattr(torch, "mps"):
        current = getattr(torch.mps, "current_allocated_memory", None)
        if current is not None:
            return float(current()) / BYTES_PER_MB
    return None


# --------------------------------------------------------------------- latency


def benchmark_latency(
    model: nn.Module,
    example_input: torch.Tensor,
    device: torch.device,
    batch_sizes: Sequence[int] = (1, 8, 32),
    warmup: int = 5,
    iterations: int = 50,
    forward_fn=None,
    sample_pool: torch.Tensor | None = None,
) -> list[LatencyReport]:
    """Time inference at each batch size and return per-sample latency and FPS.

    ``example_input`` supplies the sample shape; each batch size is materialised
    by tiling or slicing, so the caller only has to provide one representative
    tensor.

    **Tile from distinct samples where possible.** Repeating one row to reach
    batch 32 gives 32 identical gate logits, hence 32 identical Top-K selections,
    hence exactly ``K`` expert kernels launched for the whole batch instead of
    the up-to-``E`` that diverse data produces. That systematically *understates*
    the dispatch overhead of sparse routing at larger batches, which is the one
    thing the batch-32 row exists to measure. ``sample_pool`` supplies real,
    distinct samples; when it is absent the tiled batch is still used but the
    report is annotated so nobody reads the number as realistic.

    Args:
        model: Module to benchmark, moved to ``device`` by the caller.
        example_input: Representative input batch.
        device: Device the model lives on.
        batch_sizes: Batch sizes to time.
        warmup: Untimed iterations, which absorb lazy kernel compilation and
            allocator warm-up. Skipping these inflates the first measurement
            enough to dominate a short benchmark.
        iterations: Timed iterations per batch size. Each is timed individually
            so a median and an IQR are available.
        forward_fn: Callable taking the batch tensor, for models whose forward
            takes extra arguments.
        sample_pool: Distinct real samples to tile from; see above.
    """
    was_training = model.training
    model.eval()
    reports: list[LatencyReport] = []
    source = sample_pool if sample_pool is not None and sample_pool.shape[0] > 0 else example_input

    try:
        for batch_size in batch_sizes:
            batch = _resize_batch(source, batch_size).to(device)
            run = (lambda: forward_fn(batch)) if forward_fn is not None else (lambda: model(batch))

            timings: list[float] = []
            with torch.no_grad():
                for _ in range(max(warmup, 0)):
                    run()
                synchronize(device)

                for _ in range(max(iterations, 1)):
                    started = time.perf_counter()
                    run()
                    synchronize(device)
                    timings.append((time.perf_counter() - started) * 1000.0)

            ordered = sorted(timings)
            count = len(ordered)
            median_ms = ordered[count // 2] if count % 2 else 0.5 * (ordered[count // 2 - 1] + ordered[count // 2])
            q1 = ordered[max(int(0.25 * count) - 1, 0)]
            q3 = ordered[min(int(0.75 * count), count - 1)]

            per_sample_ms = median_ms / batch_size
            reports.append(
                LatencyReport(
                    batch_size=batch_size,
                    latency_ms_per_batch=median_ms,
                    latency_ms_per_sample=per_sample_ms,
                    throughput_fps=1000.0 / per_sample_ms if per_sample_ms > 0 else float("inf"),
                    latency_ms_iqr=float(q3 - q1),
                    latency_ms_min=float(ordered[0]),
                    latency_ms_max=float(ordered[-1]),
                    iterations=count,
                )
            )
    finally:
        model.train(was_training)

    return reports


def _resize_batch(example: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Return a tensor with ``batch_size`` rows, tiling or slicing ``example``."""
    if example.shape[0] == batch_size:
        return example
    if example.shape[0] > batch_size:
        return example[:batch_size]
    repeats = -(-batch_size // example.shape[0])  # ceil division
    return example.repeat(repeats, *([1] * (example.ndim - 1)))[:batch_size]


# ------------------------------------------------------------------ front door


def profile_model(
    model: nn.Module,
    example_input: torch.Tensor,
    device: torch.device,
    name: str = "model",
    extra_modules: Sequence[nn.Module] = (),
    batch_sizes: Sequence[int] = (1, 8, 32),
    warmup: int = 5,
    iterations: int = 50,
    forward_fn=None,
    measure_flops: bool = True,
    measure_latency: bool = True,
    sample_pool: torch.Tensor | None = None,
) -> EfficiencyReport:
    """Run every measurement and return one :class:`EfficiencyReport`.

    A framing note that belongs with the numbers rather than only in the paper:
    the Top-4 -> Top-2 change saves ``2 x parameters_per_expert``, which against
    a ~96 M model dominated by a frozen 87 M SwinV2-Base trunk is roughly **2 %
    of total parameters**. That saving is real but it is in *parameters and
    FLOPs*, not in wall-clock: sparse dispatch replaces one batched matmul with
    several small gather/matmul/scatter sequences, so at these batch sizes it can
    be *slower* than dense evaluation. The "Inference Latency (ms)" column sits
    next to "Active Params (M)" in the results table and must not be read as
    caused by it. :meth:`EfficiencyReport.summary_line` says so explicitly.

    Args:
        model: The module to profile.
        example_input: A representative input batch, already on ``device``.
        device: Device to measure on.
        name: Label for the report (variant or baseline name).
        extra_modules: Additional modules whose parameters belong to this model,
            such as a frozen encoder held outside it.
        batch_sizes: Batch sizes for the latency sweep.
        warmup / iterations: Latency loop configuration.
        forward_fn: Callable taking a batch, for non-standard forward signatures.
        measure_flops: Run the FLOP counter.
        measure_latency: Run the latency sweep.
        sample_pool: Distinct real samples for the latency sweep to tile from;
            see :func:`benchmark_latency`.
    """
    modules = [model, *extra_modules]
    report = EfficiencyReport(
        name=name,
        device=str(device),
        parameters=parameter_report(*modules),
        notes=[f"platform={platform.platform()}", f"torch={torch.__version__}"],
    )

    reset_peak_memory(device)

    if measure_flops:
        flops_fn = (lambda: forward_fn(example_input)) if forward_fn is not None else None
        report.flops = estimate_flops(model, example_input, forward_fn=flops_fn)
        report.flops_batch_size = int(example_input.shape[0])
        if report.flops is None:
            report.notes.append("FLOP counting unavailable on this torch build.")

    if measure_latency:
        report.latencies = benchmark_latency(
            model,
            example_input,
            device,
            batch_sizes=batch_sizes,
            warmup=warmup,
            iterations=iterations,
            forward_fn=forward_fn,
            sample_pool=sample_pool,
        )
        if sample_pool is None or sample_pool.shape[0] < max(batch_sizes, default=1):
            report.notes.append(
                "Latency batches were tiled from too few distinct samples: identical rows "
                "produce identical routing, so sparse-dispatch overhead is understated at "
                "the larger batch sizes."
            )

    report.peak_memory_mb = peak_memory_mb(device)
    if report.peak_memory_mb is None:
        report.notes.append(f"Peak memory is not tracked on device type {device.type!r}.")
    elif device.type == "mps":
        report.notes.append("MPS reports current allocation, not a true peak.")

    saving = top_k_saving(*modules)
    if saving:
        total = report.parameters.total or 1
        report.notes.append(
            f"Top-{report.parameters.top_k} activates "
            f"{saving['parameters_saved_fraction'] * 100:.1f}% fewer parameters than Top-4 "
            f"({saving['parameters_saved'] / 1e6:.2f} M, "
            f"{saving['parameters_saved'] / total * 100:.1f}% of the total model). "
            "This is a parameter and FLOP saving, not a wall-clock one: the frozen backbone "
            "dominates latency and sparse dispatch trades one batched matmul for several small ones."
        )

    return report
