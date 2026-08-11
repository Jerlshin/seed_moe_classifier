"""Named and rolling checkpoint files, written atomically, from one rank only.

Two concerns live here, and both are about the *file*, not its contents (which
:mod:`src.utils.training.resume` owns):

**Disk pressure.** Rented instances have small root disks -- 16 GB is the vast.ai
default this repository is tuned for -- and a 300-epoch run that keeps every
interval checkpoint fills it. :meth:`CheckpointManager.prune` keeps the newest
``keep_last_n`` per rolling prefix and leaves named artifacts ("best", "final")
alone.

**Torn writes.** ``torch.save`` truncates the destination before writing it, so a
session killed mid-save destroys the checkpoint it was replacing. Every write
goes through :func:`~src.utils.training.resume.atomic_save`: a temp file in the
same directory, then ``os.replace``. The consequence for pruning is that
``keep_last_n`` should be **at least 2** on a preemptible platform -- not because
a write can tear, but because the newest file is the one a kill is most likely
to have interrupted *before* it completed, and the previous one is the fallback
:func:`~src.utils.training.resume.find_latest_checkpoint` walks back to.

**One writer.** Under DDP every rank holds identical parameters, so every rank
writing the same path is at best wasted bandwidth and at worst a partially
overwritten file. ``CheckpointManager(..., enabled=ctx.is_main)`` turns the
non-main ranks into no-ops that still return the path they would have written,
so call sites need no rank branch of their own.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.utils.training.resume import TEMP_SUFFIX, atomic_save


class CheckpointManager:
    """Small-disk, crash-safe checkpoint writer for rented GPU instances.

    Args:
        output_dir: Directory every file is written into.
        keep_last_n: Rolling checkpoints retained per prefix. ``2`` or more on a
            platform that stops sessions; see the module docstring.
        enabled: ``False`` makes every write a no-op that still reports its
            path. Set it to ``context.is_main`` under DDP.
    """

    def __init__(self, output_dir: str | Path, keep_last_n: int = 1, enabled: bool = True):
        self.output_dir = Path(output_dir)
        self.enabled = bool(enabled)
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = max(int(keep_last_n), 0)

    def save(
        self,
        filename: str,
        payload: dict[str, Any],
        rolling_prefix: str | None = None,
    ) -> str:
        """Write ``payload`` to ``filename`` atomically and prune older siblings."""
        path = self.output_dir / filename
        if not self.enabled:
            return str(path)
        atomic_save(payload, path)
        if rolling_prefix is not None:
            self.prune(rolling_prefix=rolling_prefix)
        return str(path)

    def prune(self, rolling_prefix: str) -> None:
        """Delete all but the newest ``keep_last_n`` files matching the prefix.

        In-progress writes (``*.tmp``) are never candidates: deleting one would
        race the writer that is still filling it.
        """
        if not self.enabled:
            return
        candidates = [
            path
            for path in self.output_dir.glob(f"{rolling_prefix}*.pth")
            if path.is_file() and not path.name.endswith(TEMP_SUFFIX)
        ]
        if self.keep_last_n <= 0:
            stale = sorted(candidates)
        else:
            stale = sorted(
                candidates,
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )[self.keep_last_n :]

        for path in stale:
            path.unlink(missing_ok=True)


def to_cpu_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    cpu_state: dict[str, Any] = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            cpu_state[key] = value.detach().cpu()
        else:
            cpu_state[key] = value
    return cpu_state
