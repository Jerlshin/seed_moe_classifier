"""Resumable checkpoints: complete state, atomic writes, automatic discovery.

A checkpoint that carries only weights is not a checkpoint, it is a snapshot.
Restarting a 300-epoch DINO run from one restarts the optimizer's moments at
zero, the LR schedule at its peak, the teacher momentum at 0.996 and the RNG at
whatever the new process seeded -- so the "resumed" run is a different run that
happens to share an initialisation. On a platform that stops the session every
few hours, that is the difference between a job that finishes and a job that
cannot.

What a checkpoint here contains
-------------------------------

Everything needed to make the next step identical to the step that would have
come next:

* every named component's ``state_dict`` -- student, teacher, projection heads,
  optimizer, scheduler, ``GradScaler``, and the criterion (whose centering
  buffer and learnable log-variances are state);
* :class:`TrainingProgress` -- epoch, global optimizer step, and **micro-batches
  consumed within the current epoch**, which is what makes resume land mid-epoch
  instead of at its start;
* :class:`RngSnapshot` for Python, NumPy, torch CPU, every CUDA device, and the
  dataloader's own generator, captured **per rank**;
* the resolved config and the distributed layout the run had.

Why the write is atomic
-----------------------

``torch.save`` to the destination path truncates it first. A session killed
during that window leaves a zero-length or half-written file *where the good
checkpoint used to be*, and the next resume finds the newest checkpoint and
fails to load it -- having already destroyed the one before. Every write here
goes to a temporary file in the same directory and is promoted with
``os.replace``, which is atomic on POSIX and on Windows for a same-directory
rename. A kill mid-write leaves the previous checkpoint untouched and an
obvious ``.tmp`` file that :func:`find_latest_checkpoint` ignores.

:data:`COMPLETE_KEY` is written last inside the payload as a second line of
defence: a file that loads but lacks it was produced by an older or interrupted
writer and is skipped rather than trusted.

Why resume searches rather than being told
------------------------------------------

``resume: auto`` scans the save directory newest-first and returns the first
file that actually loads and is complete. On a preempted instance the most
recent file is exactly the one most likely to be damaged, so "newest" and
"valid" have to be separate tests -- and the fallback to the previous
checkpoint is the entire point of keeping more than one.
"""

from __future__ import annotations

import glob
import logging
import os
import random
import signal
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from src.utils.training.distributed import (
    DistributedContext,
    gather_objects,
    single_process_context,
    strip_wrapper_prefixes,
)

LOGGER = logging.getLogger(__name__)

#: Bumped when the payload layout changes in a way a reader must notice.
CHECKPOINT_FORMAT_VERSION = 2

#: Written last. Its absence means the writer did not finish.
COMPLETE_KEY = "checkpoint_complete"

#: Suffix of an in-progress write. Never a resume candidate.
TEMP_SUFFIX = ".tmp"


# --------------------------------------------------------------------- progress


@dataclass
class TrainingProgress:
    """Where the run is, precisely enough to continue rather than restart.

    Attributes:
        epoch: Zero-based index of the epoch **in progress**. A checkpoint taken
            at the end of epoch ``k`` stores ``epoch=k+1, micro_step=0``.
        global_step: Optimizer steps taken, which is what the LR schedule, the
            momentum schedule and every logged x-axis are counted in.
        micro_step: Micro-batches already consumed *within* ``epoch``. Non-zero
            only for a checkpoint written mid-epoch, which is the case a
            time-budgeted or preempted run always hits.
        best_metric: Best monitored value so far, so the "is this the best?"
            comparison survives a restart instead of re-saving on the first
            epoch after resume.
        completed: The run reached its final epoch. A resume against a completed
            checkpoint is a no-op rather than an error.
    """

    epoch: int = 0
    global_step: int = 0
    micro_step: int = 0
    best_metric: float = float("inf")
    completed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "TrainingProgress":
        if not payload:
            return cls()
        known = {field_name: payload[field_name] for field_name in cls().as_dict() if field_name in payload}
        return cls(**known)


# ------------------------------------------------------------------ RNG capture


@dataclass
class RngSnapshot:
    """Every random stream that influences the next step, for one rank.

    The dataloader generator is included because it is the one nothing else
    covers: it drives the shuffling order, and without it a mid-epoch resume
    would continue with correct weights against a re-rolled sample order --
    reproducible in aggregate, not reproducible step by step.
    """

    python: Any = None
    numpy: Any = None
    torch_cpu: Any = None
    torch_cuda: Any = None
    loader_generator: Any = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def capture_rng(generator: torch.Generator | None = None) -> dict[str, Any]:
    """Snapshot every random stream this process owns."""
    snapshot = RngSnapshot(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch_cpu=torch.get_rng_state(),
        torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        loader_generator=generator.get_state() if generator is not None else None,
    )
    return snapshot.as_dict()


def restore_rng(
    state: Mapping[str, Any] | None,
    generator: torch.Generator | None = None,
    *,
    logger: logging.Logger | None = None,
) -> bool:
    """Restore the streams in ``state``. Returns whether anything was restored.

    A failure to restore one stream is logged and skipped rather than raised: a
    resume that continues with a re-seeded RNG is a small loss of reproducibility,
    and a resume that refuses to start because the checkpoint came from a machine
    with a different CUDA device count is a large one.
    """
    if not state:
        return False
    log = logger or LOGGER

    def attempt(name: str, action) -> None:
        try:
            action()
        except Exception as exc:  # pragma: no cover - platform/version dependent
            log.warning("Could not restore the %s RNG state: %s", name, exc)

    if state.get("python") is not None:
        attempt("Python", lambda: random.setstate(tuple(state["python"])))
    if state.get("numpy") is not None:
        attempt("NumPy", lambda: np.random.set_state(state["numpy"]))
    if state.get("torch_cpu") is not None:
        attempt("torch CPU", lambda: torch.set_rng_state(_as_byte_tensor(state["torch_cpu"])))

    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        # Device counts differ between the machine that saved and the machine
        # that resumes far more often than anything else here does.
        if len(cuda_state) == torch.cuda.device_count():
            attempt(
                "CUDA",
                lambda: torch.cuda.set_rng_state_all([_as_byte_tensor(item) for item in cuda_state]),
            )
        else:
            attempt(
                "CUDA device 0",
                lambda: torch.cuda.set_rng_state(_as_byte_tensor(cuda_state[0])),
            )

    if generator is not None and state.get("loader_generator") is not None:
        attempt("dataloader generator", lambda: generator.set_state(_as_byte_tensor(state["loader_generator"])))
    return True


def _as_byte_tensor(value: Any) -> torch.Tensor:
    """RNG states must be CPU ``uint8`` tensors, whatever ``torch.load`` produced."""
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return tensor.cpu().to(torch.uint8)


# ------------------------------------------------------------------- the payload


def build_checkpoint_payload(
    *,
    components: Mapping[str, Any],
    progress: TrainingProgress,
    context: DistributedContext,
    config: Any = None,
    rng_states: Sequence[Mapping[str, Any]] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the full payload from live objects.

    Args:
        components: ``name -> object exposing ``state_dict()``. ``None`` values
            are skipped, so an absent ``GradScaler`` (bf16, or CPU) simply does
            not appear rather than needing a branch at every call site.
        rng_states: One snapshot per rank, rank-ordered. Collected by
            :func:`collect_rng_states`; the whole list is stored so a job that
            resumes at the same world size restores each rank's own stream.
        extra: Anything stage-specific -- the class mappings stage 2 stores, the
            resolved effective batch, the backbone name.
    """
    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "progress": progress.as_dict(),
        "distributed": context.as_dict(),
        "rng_states": list(rng_states) if rng_states is not None else [],
        "torch_version": torch.__version__,
        "saved_at": time.time(),
    }

    for name, component in components.items():
        if component is None:
            continue
        state = component.state_dict() if hasattr(component, "state_dict") else component
        if isinstance(state, dict):
            payload[name] = _to_cpu(state)
        else:  # pragma: no cover - state_dict contracts return dicts
            payload[name] = state

    if config is not None:
        payload["config"] = _config_to_container(config)
    if extra:
        payload.update(dict(extra))

    # Last, deliberately: a reader that finds it knows every key above it exists.
    payload[COMPLETE_KEY] = True
    return payload


def _to_cpu(state: Mapping[str, Any]) -> dict[str, Any]:
    """Move tensors to host memory so a checkpoint never pins a device index."""
    moved: dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.detach().cpu()
        elif isinstance(value, Mapping):
            moved[key] = _to_cpu(value)
        else:
            moved[key] = value
    return moved


def _config_to_container(config: Any) -> Any:
    """Resolved plain-Python copy of a Hydra config, or the object unchanged."""
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(config):
            return OmegaConf.to_container(config, resolve=True)
    except Exception:  # pragma: no cover - omegaconf always present in practice
        pass
    return config


def collect_rng_states(
    context: DistributedContext,
    generator: torch.Generator | None = None,
) -> list[dict[str, Any]]:
    """Every rank's RNG snapshot, rank-ordered, on every rank.

    Gathered rather than saved per-rank-file because a single file is the only
    thing that survives being copied off a preempted instance by hand.
    """
    return gather_objects(capture_rng(generator), context)


def restore_rng_states(
    payload: Mapping[str, Any],
    context: DistributedContext,
    generator: torch.Generator | None = None,
    *,
    logger: logging.Logger | None = None,
) -> None:
    """Restore this rank's stream from a checkpoint's gathered list.

    World size may differ from the saving run's -- resuming a 2-GPU job on 1 GPU
    is the ordinary Kaggle case. Each rank then takes ``rank % saved`` and the
    run stays deterministic *going forward* without pretending to reproduce a
    different topology's sample order.
    """
    states = payload.get("rng_states") or []
    if not states:
        return
    log = logger or LOGGER
    if len(states) != context.world_size:
        log.warning(
            "Checkpoint carries %s RNG state(s) but this job has %s rank(s); rank %s "
            "restores state %s. Streams are deterministic from here, but the sample "
            "order is not a continuation of the saved topology's.",
            len(states), context.world_size, context.rank, context.rank % len(states),
        )
    restore_rng(states[context.rank % len(states)], generator, logger=log)


# --------------------------------------------------------------------- restoring


def restore_components(
    payload: Mapping[str, Any],
    components: Mapping[str, Any],
    *,
    strict: bool = True,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Load each named component from ``payload``; report what was missing.

    Module state dicts pass through :func:`strip_wrapper_prefixes` first, so a
    checkpoint written by a DDP- or ``torch.compile``-wrapped module loads into a
    bare one. Without that, ``strict=False`` loading matches zero keys and
    reports success.
    """
    log = logger or LOGGER
    report: dict[str, Any] = {"loaded": [], "missing": [], "failed": {}}

    for name, component in components.items():
        if component is None:
            continue
        if name not in payload:
            report["missing"].append(name)
            continue
        state = payload[name]
        if isinstance(state, dict):
            state = strip_wrapper_prefixes(state)
        try:
            if isinstance(component, torch.nn.Module):
                incompatible = component.load_state_dict(state, strict=strict)
                if getattr(incompatible, "missing_keys", None) or getattr(
                    incompatible, "unexpected_keys", None
                ):
                    log.warning(
                        "%s loaded with %s missing and %s unexpected keys.",
                        name, len(incompatible.missing_keys), len(incompatible.unexpected_keys),
                    )
            else:
                component.load_state_dict(state)
            report["loaded"].append(name)
        except Exception as exc:
            report["failed"][name] = str(exc)
            log.error("Could not restore %s from the checkpoint: %s", name, exc)

    if report["missing"]:
        log.warning("Checkpoint had no state for: %s", ", ".join(sorted(report["missing"])))
    return report


# ------------------------------------------------------------------- disk access


def atomic_save(payload: Mapping[str, Any], path: str | Path) -> Path:
    """``torch.save`` to a sibling temp file, then ``os.replace`` onto ``path``.

    The rename is the only operation that touches ``path``, and it is atomic, so
    the destination is either the old checkpoint or the new one and never a
    truncated hybrid. The temp file is unlinked on failure so a crashed write
    does not accumulate.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + TEMP_SUFFIX)

    try:
        with temporary.open("wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            # The rename is atomic with respect to *ordering*, not durability:
            # without the fsync a power loss can reorder the rename ahead of the
            # data. Cheap next to a multi-hundred-megabyte save.
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def is_valid_checkpoint(path: str | Path, *, logger: logging.Logger | None = None) -> bool:
    """True when ``path`` loads and carries the completion sentinel."""
    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as exc:
        (logger or LOGGER).warning("Ignoring unreadable checkpoint %s: %s", path, exc)
        return False
    return isinstance(payload, Mapping) and bool(payload.get(COMPLETE_KEY, False))


def find_latest_checkpoint(
    directory: str | Path,
    patterns: Iterable[str] = ("*.pth", "*.pt"),
    *,
    logger: logging.Logger | None = None,
) -> Path | None:
    """Newest *loadable* checkpoint under ``directory``, or ``None``.

    Newest-first by modification time, skipping ``.tmp`` files and anything that
    fails :func:`is_valid_checkpoint`. On a preempted instance the newest file is
    the most likely to be damaged, which is exactly why validity is checked
    rather than assumed -- and why ``keep_last_n_checkpoints`` should be at least
    2 on a platform that stops sessions.
    """
    root = Path(directory)
    if not root.is_dir():
        return None

    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(
            path for path in root.glob(pattern) if path.is_file() and not path.name.endswith(TEMP_SUFFIX)
        )
    if not candidates:
        return None

    for path in sorted(set(candidates), key=lambda item: item.stat().st_mtime, reverse=True):
        if is_valid_checkpoint(path, logger=logger):
            return path
    return None


def resolve_resume_path(
    requested: Any,
    directory: str | Path,
    *,
    patterns: Iterable[str] = ("*.pth", "*.pt"),
    logger: logging.Logger | None = None,
) -> Path | None:
    """Turn the ``resume`` config value into a path, or ``None`` for a fresh run.

    Accepted values:

    ``false`` / ``null``
        Start fresh. Any existing checkpoints are left alone.
    ``"auto"`` / ``true``
        Continue from the newest valid checkpoint in ``directory`` if there is
        one, otherwise start fresh. This is the setting a preemptible or
        session-limited platform wants, because the same command works for the
        first launch and every relaunch after it.
    a path
        Continue from that file, and fail loudly if it is absent -- an explicit
        path that silently falls back to a fresh run is how a week of compute
        gets thrown away.
    """
    log = logger or LOGGER
    if requested is None or requested is False:
        return None
    if isinstance(requested, str) and requested.strip().lower() in {"", "false", "no", "off", "none"}:
        return None

    if requested is True or (isinstance(requested, str) and requested.strip().lower() in {"auto", "true", "latest"}):
        found = find_latest_checkpoint(directory, patterns, logger=log)
        if found is None:
            log.info("resume=auto: no valid checkpoint under %s; starting a fresh run.", directory)
        else:
            log.info("resume=auto: continuing from %s", found)
        return found

    path = Path(str(requested))
    if not path.exists():
        raise FileNotFoundError(
            f"resume={requested!r} was requested but {path} does not exist. "
            "Pass resume=auto to continue from the newest valid checkpoint in the save "
            "directory, or resume=false to start fresh."
        )
    return path


def load_checkpoint_payload(
    path: str | Path,
    map_location: Any = "cpu",
    *,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Read a checkpoint, warning when it predates the completion sentinel."""
    log = logger or LOGGER
    payload = torch.load(str(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a checkpoint dictionary.")
    if not payload.get(COMPLETE_KEY, False):
        log.warning(
            "%s has no completion marker (format_version=%s). It was written by an older "
            "or interrupted writer; resuming from it may restore partial state.",
            path, payload.get("format_version"),
        )
    return payload


# ------------------------------------------------------------------- interrupts


@dataclass
class StopRequest:
    """Why the loop was asked to stop, for the log line and the exit code."""

    requested: bool = False
    reason: str = ""


class InterruptGuard:
    """Turn SIGTERM/SIGINT and a wall-clock budget into a checkpoint-and-exit.

    Kaggle, Colab and every preemptible instance end a session by signalling the
    process. The default handler kills it where it stands, discarding whatever
    has happened since the last periodic save. This sets a flag instead; the
    training loop polls :meth:`should_stop` at its next safe point -- an
    accumulation boundary, where no gradient is half-accumulated -- writes a
    checkpoint and exits cleanly.

    **Nothing is saved from inside the handler.** A signal can arrive in the
    middle of a CUDA call, and re-entering the allocator or launching a kernel
    from a handler is how a clean shutdown becomes a hang.

    ``max_runtime_minutes`` is the more reliable of the two guards, because a
    platform that stops a session does not always signal first. Set it a few
    minutes below the session limit and the run stops itself with a complete
    checkpoint every time.

    Args:
        max_runtime_minutes: Wall-clock budget from construction. ``None`` or 0
            disables it.
        signals: Which signals to trap. ``SIGUSR1`` is included where it exists
            because Slurm's ``--signal`` sends it before a preemption.
    """

    def __init__(
        self,
        max_runtime_minutes: float | None = None,
        *,
        signals: Sequence[int] | None = None,
        logger: logging.Logger | None = None,
    ):
        self.logger = logger or LOGGER
        self.started = time.monotonic()
        self.max_runtime_seconds = (
            float(max_runtime_minutes) * 60.0 if max_runtime_minutes else None
        )
        self.stop = StopRequest()
        self._previous: dict[int, Any] = {}

        if signals is None:
            names = ["SIGINT", "SIGTERM", "SIGUSR1"]
            signals = [getattr(signal, name) for name in names if hasattr(signal, name)]
        self._signals = list(signals)

    def install(self) -> "InterruptGuard":
        """Trap the signals. Also reachable as a context manager."""
        for number in self._signals:
            try:
                self._previous[number] = signal.getsignal(number)
                signal.signal(number, self._handle)
            except (ValueError, OSError, RuntimeError):
                # Only the main thread of the main interpreter may install
                # handlers; a worker or an embedded interpreter simply does not
                # get this feature, and that must not be fatal.
                self._previous.pop(number, None)
        return self

    def restore(self) -> None:
        """Put the previous handlers back. Safe to call more than once."""
        for number, handler in self._previous.items():
            try:
                signal.signal(number, handler)
            except (ValueError, OSError, RuntimeError):  # pragma: no cover
                pass
        self._previous.clear()

    def __enter__(self) -> "InterruptGuard":
        return self.install()

    def __exit__(self, *exc_info: Any) -> None:
        self.restore()

    def _handle(self, signum: int, _frame: Any) -> None:
        name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        if self.stop.requested:
            # A second signal means the operator wants out now. Restore the
            # default and re-raise so the process actually dies.
            self.logger.warning("Received %s again; exiting immediately.", name)
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
            return
        self.stop = StopRequest(True, f"signal {name}")
        self.logger.warning(
            "Received %s. Finishing the current micro-batch, then checkpointing and exiting.",
            name,
        )

    @property
    def elapsed_minutes(self) -> float:
        return (time.monotonic() - self.started) / 60.0

    def should_stop(self) -> StopRequest:
        """Poll for a stop request. Cheap enough for every accumulation boundary."""
        if self.stop.requested:
            return self.stop
        if self.max_runtime_seconds is not None and (
            time.monotonic() - self.started
        ) >= self.max_runtime_seconds:
            self.stop = StopRequest(
                True, f"runtime budget of {self.max_runtime_seconds / 60.0:.1f} min reached"
            )
            self.logger.warning("Runtime budget reached; checkpointing and exiting cleanly.")
        return self.stop


class PeriodicSaver:
    """Decide when a time-based checkpoint is due.

    Epoch-interval saving is the wrong cadence for a session-limited platform:
    ``save_interval: 50`` on a 300-epoch run means a session that dies at epoch
    49 has produced nothing. This adds a wall-clock trigger on top, so the worst
    case is bounded by minutes rather than by epochs.
    """

    def __init__(self, every_minutes: float | None = None):
        self.every_seconds = float(every_minutes) * 60.0 if every_minutes else None
        self.last = time.monotonic()

    def due(self) -> bool:
        if self.every_seconds is None:
            return False
        if (time.monotonic() - self.last) < self.every_seconds:
            return False
        return True

    def mark(self) -> None:
        self.last = time.monotonic()


@dataclass
class ResumeState:
    """What a resume produced, for the trainer to branch on."""

    path: Path | None = None
    progress: TrainingProgress = field(default_factory=TrainingProgress)
    report: dict[str, Any] = field(default_factory=dict)

    @property
    def resumed(self) -> bool:
        return self.path is not None

    def describe(self) -> str:
        if not self.resumed:
            return "fresh run"
        return (
            f"resumed from {self.path} at epoch {self.progress.epoch + 1}, "
            f"step {self.progress.global_step}, micro-batch {self.progress.micro_step}"
        )


def default_context() -> DistributedContext:
    """Single-process context, for callers that never touch distributed."""
    return single_process_context(torch.device("cpu"))


def checkpoint_glob(directory: str | Path, prefix: str) -> list[Path]:
    """Rolling checkpoints for ``prefix``, newest first, temp files excluded."""
    matches = [
        Path(item)
        for item in glob.glob(str(Path(directory) / f"{prefix}*"))
        if not item.endswith(TEMP_SUFFIX) and Path(item).is_file()
    ]
    return sorted(matches, key=lambda item: item.stat().st_mtime, reverse=True)
