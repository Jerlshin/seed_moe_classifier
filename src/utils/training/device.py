"""Device selection, backend tuning and mixed-precision plumbing.

Three things live here, and the second and third are what make a run fast:

* :func:`select_device` -- ``cuda`` / ``mps`` / ``cpu`` with ``auto``.
* :func:`configure_backend` -- TF32, cuDNN autotuning, matmul precision, the
  expandable-segments allocator and SDPA backend selection. Called once, before
  the first tensor reaches the GPU.
* :func:`resolve_amp` / :func:`autocast_context` -- one place that decides
  whether autocast runs, in which dtype, and whether a ``GradScaler`` is needed.

Why the AMP decision is centralised
-----------------------------------

bf16 and fp16 are *not* interchangeable. bf16 carries fp32's exponent range, so
it needs no loss scaling and no ``GradScaler``; fp16 has ~5 exponent bits and
underflows small gradients to zero without one. Ampere and later support bf16 in
hardware; Turing and Volta do not. Every caller that guessed this independently
guessed it differently, so :func:`resolve_amp` is the single decision point:
``"auto"`` picks bf16 where the hardware has it, fp16 plus a scaler where it does
not, and off on CPU/MPS.

The DINO objective needs one further guarantee, which :class:`AmpConfig` exists
to carry: the prototype softmax, the Sinkhorn iterations and the KoLeo distances
all run in fp32 *inside* the autocast region (``src/losses/dino.py`` casts
explicitly), because a log-space normaliser over 8,192 prototypes at a
temperature of 0.04 is exactly the arithmetic fp16 cannot hold.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from typing import Any

import torch


def select_device(requested: str = "auto") -> torch.device:
    requested = (requested or "auto").lower()

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ------------------------------------------------------------ capability probing


#: Compute capability at which each hardware feature this repository depends on
#: first appears. Named rather than inlined because the *reason* a T4 takes a
#: different path is one of these numbers, and a bare ``>= 8`` in three places is
#: how the three drift apart.
SM_BF16 = (8, 0)          # Ampere: hardware bfloat16, hence no GradScaler
SM_TF32 = (8, 0)          # Ampere: TF32 tensor cores for fp32 matmuls
SM_FLASH_SDPA = (8, 0)    # PyTorch's FlashAttention backend; Turing gets mem-efficient
SM_TRITON = (7, 0)        # Volta and later, which is inductor's floor


@dataclass(frozen=True)
class AcceleratorReport:
    """What this machine can actually do, resolved once and logged.

    Every downstream decision -- which autocast dtype, whether a ``GradScaler``
    is needed, whether ``torch.compile`` can run, which SDPA backend the window
    attention will land on -- follows from these fields, so they are gathered in
    one place and recorded in ``summary.json``. A run whose numbers look wrong is
    then diagnosable from its own artifacts rather than by re-probing the machine
    it ran on.

    The T4 case is the one worth naming: ``sm_75`` has no bf16, no TF32 and no
    FlashAttention backend, so the same ``amp: auto`` config that yields bf16 on
    an A100 yields fp16 plus loss scaling here, and SDPA lands on the
    memory-efficient kernel rather than flash. All three are automatic; this
    report is how a reader confirms which one happened.
    """

    device_type: str
    device_count: int = 0
    name: str = ""
    capability: tuple[int, int] | None = None
    total_memory_gb: float = 0.0
    supports_bf16: bool = False
    supports_tf32: bool = False
    supports_flash_sdpa: bool = False
    supports_mem_efficient_sdpa: bool = False
    compile_available: bool = False
    compile_reason: str = ""
    torch_version: str = ""
    platform: str = ""

    @property
    def capability_str(self) -> str:
        return f"{self.capability[0]}.{self.capability[1]}" if self.capability else "n/a"

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "device_type": self.device_type,
            "device_count": self.device_count,
            "name": self.name,
            "compute_capability": self.capability_str,
            "total_memory_gb": round(self.total_memory_gb, 2),
            "supports_bf16": self.supports_bf16,
            "supports_tf32": self.supports_tf32,
            "supports_flash_sdpa": self.supports_flash_sdpa,
            "supports_mem_efficient_sdpa": self.supports_mem_efficient_sdpa,
            "compile_available": self.compile_available,
            "torch_version": self.torch_version,
            "platform": self.platform,
        }
        if self.compile_reason:
            payload["compile_reason"] = self.compile_reason
        return payload

    def summary_line(self) -> str:
        if self.device_type != "cuda":
            return (
                f"{self.device_type} | torch {self.torch_version} | "
                f"compile={'yes' if self.compile_available else 'no'}"
            )
        return (
            f"{self.device_count}x {self.name} (sm_{self.capability_str.replace('.', '')}, "
            f"{self.total_memory_gb:.1f} GB) | bf16={'yes' if self.supports_bf16 else 'no'} "
            f"tf32={'yes' if self.supports_tf32 else 'no'} "
            f"flash_sdpa={'yes' if self.supports_flash_sdpa else 'no'} "
            f"compile={'yes' if self.compile_available else 'no'}"
        )


def compile_available() -> tuple[bool, str]:
    """Whether ``torch.compile`` can produce a compiled kernel here, and why not.

    Three independent things have to hold, and each fails on a platform this
    repository is asked to run on:

    * ``torch.compile`` exists (torch >= 2.0);
    * inductor's GPU backend needs **Triton**, which ships with the Linux CUDA
      wheels and does not ship with the Windows or macOS ones;
    * the GPU must be Volta or later. A T4 (``sm_75``) qualifies; anything
      older does not.

    Returning the reason rather than a bare boolean matters because the
    interesting case is a run that quietly stayed eager: the reason is logged and
    written into ``summary.json``, so "why is Windows slower" has an answer in
    the run's own artifacts.
    """
    if not hasattr(torch, "compile"):
        return False, f"torch {torch.__version__} has no torch.compile"

    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        if capability < SM_TRITON:
            return False, f"compute capability {capability[0]}.{capability[1]} is below Triton's floor 7.0"
        try:
            import triton  # noqa: F401
        except Exception as exc:
            return False, f"Triton is unavailable ({type(exc).__name__}); inductor cannot emit GPU kernels"
        return True, ""

    # CPU inductor compiles through the C++ backend and needs a working
    # compiler. It is legal but rarely worth it, so it is reported as available
    # and left to the caller's config.
    return True, "CPU inductor backend"


def describe_accelerator(device: torch.device | None = None) -> AcceleratorReport:
    """Probe the machine once and return everything the run's decisions need."""
    compile_ok, compile_reason = compile_available()
    base = {
        "compile_available": compile_ok,
        "compile_reason": compile_reason,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
    }

    if device is not None and device.type != "cuda":
        return AcceleratorReport(device_type=device.type, **base)
    if not torch.cuda.is_available():
        return AcceleratorReport(
            device_type=device.type if device is not None else "cpu", **base
        )

    index = device.index if device is not None and device.index is not None else torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(index)
    properties = torch.cuda.get_device_properties(index)
    return AcceleratorReport(
        device_type="cuda",
        device_count=torch.cuda.device_count(),
        name=torch.cuda.get_device_name(index),
        capability=capability,
        total_memory_gb=properties.total_memory / 1024**3,
        supports_bf16=supports_bf16(),
        supports_tf32=capability >= SM_TF32,
        supports_flash_sdpa=capability >= SM_FLASH_SDPA,
        # The memory-efficient kernel covers everything back to Maxwell, which is
        # what keeps the SDPA rewrite worth doing on a T4: the N^2 attention
        # matrices stop being materialised there too, just through a different
        # kernel than an A100 would pick.
        supports_mem_efficient_sdpa=capability >= (5, 0),
        **base,
    )


# --------------------------------------------------------------- backend tuning


def enable_expandable_segments(logger=None) -> bool:
    """Switch the CUDA caching allocator to expandable segments.

    Multi-crop DINO allocates a different set of activation shapes for the
    teacher's ``2B`` views and the student's ``6B``, and (with grad checkpointing
    or a chunked forward) a third set again. Under the default allocator those
    interleave into segments that can never be coalesced, so a run that fits in
    principle dies of fragmentation after a few hundred steps with "tried to
    allocate 200 MiB" while ``memory_reserved`` sits gigabytes above
    ``memory_allocated``. Expandable segments let one segment grow instead.

    The knob is read when the allocator initialises, so setting the environment
    variable only works before the first CUDA allocation; after that we fall back
    to the runtime setter. Returns whether either path succeeded.
    """
    if not torch.cuda.is_available():
        return False

    setting = "expandable_segments:True"
    if not torch.cuda.is_initialized():
        existing = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if "expandable_segments" not in existing:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
                f"{existing},{setting}" if existing else setting
            )
        return True

    try:  # torch >= 2.1 can change it after initialisation
        torch.cuda.memory._set_allocator_settings(setting)
        return True
    except Exception as exc:  # pragma: no cover - depends on the torch build
        if logger is not None:
            logger.warning("Could not enable expandable_segments: %s", exc)
        return False


def enable_fused_attention(module: torch.nn.Module) -> int:
    """Flip timm's ``fused_attn`` flag wherever the model exposes it.

    Returns the number of attention modules switched, which the trainer logs --
    because on the stock ``timm`` SwinV2 that number is **zero**, and it is worth
    seeing rather than assuming. SwinV2's window attention is *cosine* attention:
    it L2-normalises ``q`` and ``k``, multiplies by a clamped learned logit scale
    and adds a continuous relative-position bias before the softmax, and timm
    offers no fused path for that combination, so this flag finds nothing to
    switch. The function *is* expressible with SDPA all the same -- fold the
    per-head scale into the normalised ``q``, pass the bias as ``attn_mask``,
    disable SDPA's own ``1/sqrt(d)`` -- which is what
    ``src/models/backbones/sdpa_attention.py`` does for the stage-1 trunks; on
    this backbone that conversion, not this flag, is the fused-attention route.

    The call is still made because ``timm`` does expose ``fused_attn`` on other
    trunks in this repository's baseline set (Swin-T, ViT-style heads), and on
    those it is a real win.
    """
    switched = 0
    for submodule in module.modules():
        if getattr(submodule, "fused_attn", None) is False:
            submodule.fused_attn = True
            switched += 1
    return switched


def configure_backend(
    device: torch.device,
    *,
    allow_tf32: bool = True,
    cudnn_benchmark: bool = True,
    deterministic: bool = False,
    matmul_precision: str = "high",
    expandable_segments: bool = True,
    logger=None,
) -> dict[str, Any]:
    """Apply the process-wide backend settings and report what was applied.

    Args:
        allow_tf32: Let cuDNN and cuBLAS use TF32 tensor cores for fp32 matmuls.
            Ampere and later only; a no-op elsewhere. Costs ~3 mantissa bits on
            the accumulate, which is far inside the noise of an SSL objective.
        cudnn_benchmark: Autotune convolution algorithms. Worth it whenever the
            input shape is *fixed* -- which it is here, because ``drop_last`` and
            the fixed 256 px window guarantee every batch has the same shape.
            Mutually exclusive with ``deterministic``.
        deterministic: Force deterministic cuDNN algorithms. Stage 1 produces a
            single checkpoint rather than a row in a comparison table, so it does
            not need this; stage 2 keeps it, because that is where variants must
            be comparable.
        matmul_precision: ``torch.set_float32_matmul_precision`` value.
        expandable_segments: See :func:`enable_expandable_segments`.

    Returns:
        The settings actually in force, for logging into ``summary.json``.
    """
    applied: dict[str, Any] = {"device": str(device)}

    if deterministic:
        cudnn_benchmark = False

    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    applied["cudnn.deterministic"] = bool(deterministic)
    applied["cudnn.benchmark"] = bool(cudnn_benchmark)

    if device.type == "cuda" and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
        applied["allow_tf32"] = bool(allow_tf32)
        try:
            torch.set_float32_matmul_precision(str(matmul_precision))
            applied["float32_matmul_precision"] = str(matmul_precision)
        except Exception as exc:  # pragma: no cover
            if logger is not None:
                logger.warning("Could not set float32 matmul precision: %s", exc)
        if expandable_segments:
            applied["expandable_segments"] = enable_expandable_segments(logger)

        capability = torch.cuda.get_device_capability()
        applied["compute_capability"] = f"{capability[0]}.{capability[1]}"
        applied["bf16_supported"] = supports_bf16()

    report = describe_accelerator(device)
    applied["accelerator"] = report.as_dict()

    if logger is not None:
        logger.info("Accelerator | %s", report.summary_line())
        logger.info(
            "Backend | tf32=%s cudnn.benchmark=%s deterministic=%s matmul=%s "
            "expandable_segments=%s",
            applied.get("allow_tf32"),
            applied["cudnn.benchmark"],
            applied["cudnn.deterministic"],
            applied.get("float32_matmul_precision"),
            applied.get("expandable_segments"),
        )
        if device.type == "cuda" and not report.supports_flash_sdpa:
            logger.info(
                "No FlashAttention SDPA backend on %s (needs sm_80+); the converted window "
                "attention will use the memory-efficient kernel, which still avoids "
                "materialising the [B*nW, heads, N, N] matrices.",
                report.name or "this GPU",
            )
    return applied


# ------------------------------------------------------------ mixed precision


def supports_bf16() -> bool:
    """True when the current CUDA device has hardware bf16 (Ampere, sm_80+)."""
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_bf16_supported())
    except Exception:  # pragma: no cover - old torch builds
        return torch.cuda.get_device_capability()[0] >= 8


@dataclass(frozen=True)
class AmpConfig:
    """Resolved autocast settings for one run.

    Attributes:
        enabled: Whether to enter an autocast region at all.
        dtype: ``torch.bfloat16`` or ``torch.float16`` when enabled.
        device_type: What to pass to ``torch.autocast``.
        needs_scaler: True only for fp16. bf16 has fp32's exponent range, so
            gradients do not underflow and loss scaling would be pure overhead.
    """

    enabled: bool
    dtype: torch.dtype | None
    device_type: str
    needs_scaler: bool

    @property
    def label(self) -> str:
        if not self.enabled:
            return "off"
        return "bf16" if self.dtype is torch.bfloat16 else "fp16"


def resolve_amp(device: torch.device, requested: Any = "auto", logger=None) -> AmpConfig:
    """Decide the autocast dtype for ``device``.

    Args:
        requested: ``"auto"`` (default), ``"bf16"``, ``"fp16"``, ``"off"``, or a
            bare boolean. ``True`` is treated as ``"auto"``, which is what makes
            the legacy ``experiment.training.amp: true`` flag keep working.

    ``"auto"`` gives bf16 on Ampere and later, fp16 with a ``GradScaler``
    elsewhere on CUDA, and off on CPU/MPS -- autocast on MPS exists but silently
    changes numerics on a path nothing in this repository has validated.

    An explicit ``bf16`` on hardware without it (a T4, ``sm_75``) is **downgraded
    to fp16 with a scaler**, not honoured. Emulated bf16 there is neither fast
    nor accurate, and the alternative -- refusing to start -- would mean a config
    that runs on the development box cannot run on Kaggle at all. The downgrade
    is logged, and it is safe here for a reason specific to this objective: the
    three pieces fp16 genuinely cannot hold (the Sinkhorn log-space normaliser,
    the 8,192-way prototype log-softmax, the KoLeo pairwise distances) are pinned
    to fp32 *inside* the autocast region by ``src/losses/dino.py``.
    """
    if isinstance(requested, bool):
        requested = "auto" if requested else "off"
    mode = str(requested or "off").lower()

    if mode in {"off", "false", "none", "no", "fp32", "float32"} or device.type != "cuda":
        return AmpConfig(enabled=False, dtype=None, device_type=device.type, needs_scaler=False)

    if mode in {"bf16", "bfloat16"}:
        dtype = torch.bfloat16
        if not supports_bf16():
            dtype = torch.float16
            if logger is not None:
                capability = torch.cuda.get_device_capability()
                logger.warning(
                    "amp=bf16 requested but %s (sm_%s%s) has no hardware bfloat16; using fp16 "
                    "with a GradScaler instead. The loss terms that fp16 cannot hold are "
                    "pinned to fp32 inside the autocast region.",
                    torch.cuda.get_device_name(), capability[0], capability[1],
                )
    elif mode in {"fp16", "float16", "half"}:
        dtype = torch.float16
    elif mode in {"auto", "true", "yes", "on"}:
        dtype = torch.bfloat16 if supports_bf16() else torch.float16
    else:
        raise ValueError(f"Unsupported amp mode: {requested!r}")

    return AmpConfig(
        enabled=True,
        dtype=dtype,
        device_type="cuda",
        needs_scaler=dtype is torch.float16,
    )


def resolve_compile(requested: Any, device: torch.device, logger=None) -> bool:
    """Decide whether to compile, honouring ``"auto"`` and the platform's limits.

    ``"auto"`` compiles wherever :func:`compile_available` says a kernel can
    actually be produced, and stays eager otherwise -- which is what lets one
    config file run unchanged on a Linux CUDA box, on Kaggle's T4s, on Windows
    (no Triton) and on CPU. An explicit ``true`` on a machine that cannot compile
    is downgraded with a warning rather than left to fail inside the first step,
    where the traceback names inductor rather than the config.
    """
    if isinstance(requested, str):
        mode = requested.strip().lower()
        if mode in {"auto", ""}:
            available, reason = compile_available()
            enabled = available and device.type == "cuda"
            if logger is not None:
                if enabled:
                    logger.info("compile=auto -> enabled.")
                else:
                    logger.info(
                        "compile=auto -> disabled (%s).",
                        reason or f"device is {device.type}, not cuda",
                    )
            return enabled
        requested = mode not in {"false", "no", "off", "0"}

    if not bool(requested):
        return False

    available, reason = compile_available()
    if not available:
        if logger is not None:
            logger.warning("torch.compile was requested but is unavailable: %s. Staying eager.", reason)
        return False
    return True


def autocast_context(amp: AmpConfig):
    """An autocast region for ``amp``, or a no-op context when it is disabled."""
    if not amp.enabled:
        return torch.autocast(device_type=amp.device_type, enabled=False)
    return torch.autocast(device_type=amp.device_type, dtype=amp.dtype, enabled=True)


def build_grad_scaler(amp: AmpConfig):
    """A ``GradScaler`` for fp16, or ``None``. Never needed for bf16."""
    if not amp.needs_scaler:
        return None
    try:  # torch >= 2.4 moved it out of torch.cuda.amp
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):  # pragma: no cover - older torch
        return torch.cuda.amp.GradScaler()


# -------------------------------------------------------------------- compile


def maybe_compile(module: torch.nn.Module, enabled: bool, mode: str = "default", logger=None):
    """Return ``torch.compile(module)``, or the module unchanged.

    **The return value must not be assigned back onto an ``nn.Module``
    attribute.** ``torch.compile`` wraps the module in an ``OptimizedModule``
    whose ``state_dict`` keys all gain an ``_orig_mod.`` prefix, and the bare
    ``student_backbone`` state dict is the only handoff to stage 2 -- a prefixed
    checkpoint loads into the stage-2 encoder as 0 matched keys and, because
    ``checkpoint_strict: false`` is the default, trains happily against a random
    trunk. :class:`~src.models.backbones.swinv2_dino.DINO` therefore keeps the
    compiled callables in a plain dict off the module tree.

    Failures degrade to eager with a warning: a compiler error on a long
    unattended run is not worth losing the run over.
    """
    if not enabled:
        return module
    if not hasattr(torch, "compile"):  # pragma: no cover - torch < 2.0
        if logger is not None:
            logger.warning("torch.compile is unavailable in torch %s.", torch.__version__)
        return module
    try:
        compiled = torch.compile(module, mode=mode)
        if logger is not None:
            logger.info("Compiled %s with mode=%s.", type(module).__name__, mode)
        return compiled
    except Exception as exc:  # pragma: no cover - backend dependent
        if logger is not None:
            logger.warning("torch.compile failed (%s); continuing eagerly.", exc)
        return module


# ---------------------------------------------------------------- diagnostics


def collect_device_stats(device: torch.device) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "device/type": device.type,
        "system/platform": platform.platform(),
    }

    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        stats.update(
            {
                "gpu/name": torch.cuda.get_device_name(index),
                "gpu/memory_allocated_mb": torch.cuda.memory_allocated(index) / 1024**2,
                "gpu/memory_reserved_mb": torch.cuda.memory_reserved(index) / 1024**2,
                "gpu/max_memory_allocated_mb": torch.cuda.max_memory_allocated(index) / 1024**2,
            }
        )
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            stats["gpu/utilization_percent"] = utilization.gpu
            stats["gpu/memory_utilization_percent"] = utilization.memory
        except Exception:
            stats["gpu/utilization_percent"] = None

    elif device.type == "mps":
        stats.update(
            {
                "gpu/name": "Apple Metal Performance Shaders",
                "gpu/utilization_percent": None,
            }
        )

    return stats


def _nvml():
    """The NVML binding, under either of its two package names, or ``None``.

    ``pynvml`` was renamed ``nvidia-ml-py``; both install a module named
    ``pynvml``, but recent NVIDIA wheels also expose ``pynvml`` as a deprecation
    shim that warns on import. Both are optional extras here -- a machine without
    either loses the utilisation number and nothing else.
    """
    try:
        import pynvml

        return pynvml
    except Exception:  # pragma: no cover - depends on the extras installed
        return None


class GpuUtilizationSampler:
    """Background sampler for SM and memory utilisation over a timed region.

    Peak memory and throughput answer "did it fit" and "how fast"; neither
    answers "was the GPU busy". A step that is 60 % utilised is not compute
    bound, and the fix for it is in the dataloader or the launch overhead rather
    than in the model -- which is precisely the distinction
    ``data_wait_fraction`` is trying to make, from the other side.

    Sampling runs on a daemon thread at a fixed interval and never touches the
    CUDA context, so it cannot perturb what it measures (NVML reads the driver,
    not the process). Degrades to empty results without NVML rather than failing.
    """

    def __init__(self, device_index: int = 0, interval_seconds: float = 0.05):
        self.device_index = int(device_index)
        self.interval = float(interval_seconds)
        self.gpu_samples: list[float] = []
        self.memory_samples: list[float] = []
        self._thread = None
        self._stop = None
        self._nvml = _nvml()

    @property
    def available(self) -> bool:
        return self._nvml is not None and torch.cuda.is_available()

    def __enter__(self) -> "GpuUtilizationSampler":
        if not self.available:
            return self
        import threading

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._thread = None

    def _run(self) -> None:  # pragma: no cover - timing dependent
        import time as _time

        try:
            self._nvml.nvmlInit()
            handle = self._nvml.nvmlDeviceGetHandleByIndex(self.device_index)
        except Exception:
            return
        while not self._stop.is_set():
            try:
                rates = self._nvml.nvmlDeviceGetUtilizationRates(handle)
                self.gpu_samples.append(float(rates.gpu))
                self.memory_samples.append(float(rates.memory))
            except Exception:
                break
            _time.sleep(self.interval)

    def summary(self) -> dict[str, float | None]:
        """Mean and peak utilisation over the sampled region."""
        if not self.gpu_samples:
            return {"gpu_utilization_mean": None, "gpu_utilization_peak": None,
                    "memory_bandwidth_utilization_mean": None, "samples": 0}
        return {
            "gpu_utilization_mean": sum(self.gpu_samples) / len(self.gpu_samples),
            "gpu_utilization_peak": max(self.gpu_samples),
            "memory_bandwidth_utilization_mean": (
                sum(self.memory_samples) / len(self.memory_samples) if self.memory_samples else None
            ),
            "samples": len(self.gpu_samples),
        }
