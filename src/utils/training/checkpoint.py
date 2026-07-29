from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class CheckpointManager:
    """
    Small-disk checkpoint helper for rented GPU instances.

    It keeps named artifacts such as "best" and "final", and prunes rolling
    interval checkpoints by prefix. This prevents long runs from filling a
    16 GB vast.ai root disk.
    """

    def __init__(self, output_dir: str | Path, keep_last_n: int = 1):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = max(int(keep_last_n), 0)

    def save(
        self,
        filename: str,
        payload: dict[str, Any],
        rolling_prefix: str | None = None,
    ) -> str:
        path = self.output_dir / filename
        torch.save(payload, path)
        if rolling_prefix is not None:
            self.prune(rolling_prefix=rolling_prefix)
        return str(path)

    def prune(self, rolling_prefix: str) -> None:
        if self.keep_last_n <= 0:
            checkpoints = sorted(self.output_dir.glob(f"{rolling_prefix}*.pth"))
        else:
            checkpoints = sorted(
                self.output_dir.glob(f"{rolling_prefix}*.pth"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )[self.keep_last_n :]

        for path in checkpoints:
            path.unlink(missing_ok=True)


def to_cpu_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    cpu_state: dict[str, Any] = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            cpu_state[key] = value.detach().cpu()
        else:
            cpu_state[key] = value
    return cpu_state
