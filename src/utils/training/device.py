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

    if logger is not None:
        logger.info(
            "Backend | tf32=%s cudnn.benchmark=%s deterministic=%s matmul=%s "
            "expandable_segments=%s",
            applied.get("allow_tf32"),
            applied["cudnn.benchmark"],
            applied["cudnn.deterministic"],
            applied.get("float32_matmul_precision"),
            applied.get("expandable_segments"),
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


def resolve_amp(device: torch.device, requested: Any = "auto") -> AmpConfig:
    """Decide the autocast dtype for ``device``.

    Args:
        requested: ``"auto"`` (default), ``"bf16"``, ``"fp16"``, ``"off"``, or a
            bare boolean. ``True`` is treated as ``"auto"``, which is what makes
            the legacy ``experiment.training.amp: true`` flag keep working.

    ``"auto"`` gives bf16 on Ampere and later, fp16 with a ``GradScaler``
    elsewhere on CUDA, and off on CPU/MPS -- autocast on MPS exists but silently
    changes numerics on a path nothing in this repository has validated.
    """
    if isinstance(requested, bool):
        requested = "auto" if requested else "off"
    mode = str(requested or "off").lower()

    if mode in {"off", "false", "none", "no", "fp32", "float32"} or device.type != "cuda":
        return AmpConfig(enabled=False, dtype=None, device_type=device.type, needs_scaler=False)

    if mode in {"bf16", "bfloat16"}:
        dtype = torch.bfloat16
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
