from __future__ import annotations

import platform
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
