"""Process-group lifecycle and the collectives the two stages need.

One module owns every decision about *whether* this process is part of a
distributed run, *which* backend it speaks, and *what* a collective means for
each quantity the objective computes. Callers ask for a
:class:`DistributedContext` once and then branch on it; nothing else in the
repository reads ``torch.distributed`` directly.

What is distributed, and what is not
------------------------------------

**Images are sharded; views are not.** ``DistributedSampler`` splits the
*dataset* across ranks, and each rank then builds all ``2 + local_crops_number``
views of its own images. That is forced by the objective rather than chosen:
Eq. 1 scores every cross-view pair of one image against the teacher's output for
that same image, so a sample's views must be resident on one device. Splitting
the *view* axis across ranks would make the cross-view pairing a collective, and
the loss curve of a run that got that wrong looks entirely normal.

**Gradients are averaged; batch statistics are per-rank by default.** The DINO
loss is a mean over samples, so ``mean_r(grad over shard r) == grad over the
concatenated batch`` exactly, and DDP's all-reduce is all the synchronisation the
gradient needs. The batch *statistics* -- Sinkhorn's doubly-stochastic
normaliser, KoLeo's nearest-neighbour distances -- are not means over samples,
and they are already computed **per micro-batch** under gradient accumulation.
Holding the per-rank micro-batch fixed while distributing therefore leaves both
functions exactly as they are on one GPU, which is why
``effective_batch_size`` derives the accumulation count from the world size
rather than the micro-batch. :func:`logsumexp_across_ranks` exists for the
opt-in ``distributed_sinkhorn`` path, where the assignment is instead made
doubly stochastic over the whole global step batch.

**The EMA centre is not optional.** ``centering="ema"`` keeps a running buffer
that the teacher's targets read, so per-rank buffers would diverge into ranks
training against different targets and a checkpoint whose contents depend on
which rank wrote it. :meth:`~src.losses.dino.CustomDINOLoss.update_center`
all-reduces unconditionally.

Backend selection
-----------------

NCCL on CUDA where it is built (Linux); Gloo everywhere else -- Windows has no
NCCL, and CPU debugging runs need a backend that works without a GPU. Both are
shipped with the official PyTorch wheels; nothing here requires an extra
package.
"""

from __future__ import annotations

import datetime
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import torch
import torch.distributed as dist

LOGGER = logging.getLogger(__name__)

#: Environment variables ``torchrun`` sets. Their presence is what distinguishes
#: a rank of a distributed job from an ordinary single-process run; nothing in
#: this repository ever spawns processes behind the caller's back.
TORCHRUN_KEYS = ("RANK", "WORLD_SIZE", "LOCAL_RANK")


@dataclass(frozen=True)
class DistributedContext:
    """Everything a caller needs to know about its place in the job.

    A single-process run gets ``enabled=False, rank=0, world_size=1``, so call
    sites read the same way whether or not the job is distributed --
    ``if ctx.is_main`` is correct in both.

    Attributes:
        enabled: A process group is initialised and ``world_size > 1``.
        rank: Global rank in ``[0, world_size)``.
        local_rank: Rank within this node; also the CUDA device index.
        world_size: Total processes.
        local_world_size: Processes on this node, which is what the dataloader
            worker budget has to be divided by.
        backend: ``"nccl"``, ``"gloo"`` or ``"none"``.
        device: The device this rank owns.
    """

    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    local_world_size: int
    backend: str
    device: torch.device

    @property
    def is_main(self) -> bool:
        """True on exactly one process of the whole job. Guards all shared IO."""
        return self.rank == 0

    @property
    def is_local_main(self) -> bool:
        """True once per node. Guards node-local IO such as a scratch cache."""
        return self.local_rank == 0

    def as_dict(self) -> dict[str, Any]:
        """Serialisable description, recorded in checkpoints and ``summary.json``."""
        return {
            "enabled": self.enabled,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "local_world_size": self.local_world_size,
            "backend": self.backend,
            "device": str(self.device),
        }


def single_process_context(device: torch.device) -> DistributedContext:
    """The context a non-distributed run uses."""
    return DistributedContext(
        enabled=False,
        rank=0,
        local_rank=0,
        world_size=1,
        local_world_size=1,
        backend="none",
        device=device,
    )


def launched_distributed() -> bool:
    """True when ``torchrun`` (or an equivalent) put this process in a job.

    A ``WORLD_SIZE`` of 1 counts as *not* distributed: initialising a
    single-member process group buys nothing and adds a failure mode (a stale
    ``MASTER_PORT`` from a previous run) to a path that does not need one.
    """
    if not all(key in os.environ for key in TORCHRUN_KEYS):
        return False
    try:
        return int(os.environ["WORLD_SIZE"]) > 1
    except ValueError:
        return False


def resolve_backend(device: torch.device) -> str:
    """``"nccl"`` where it exists and there is a GPU, otherwise ``"gloo"``.

    NCCL is the only backend with a fast GPU-to-GPU path, but it is not built on
    Windows and cannot carry CPU tensors, so the fallback is not a degraded mode
    -- it is the only correct choice on those platforms.
    """
    if device.type == "cuda" and dist.is_nccl_available():
        return "nccl"
    return "gloo"


def setup_distributed(
    requested_device: str = "auto",
    *,
    timeout_minutes: float = 30.0,
    logger: logging.Logger | None = None,
) -> DistributedContext:
    """Join the process group if there is one, and claim this rank's device.

    Idempotent: a second call returns a context describing the group that is
    already up, which is what lets a benchmark and a trainer share the helper.

    ``torch.cuda.set_device`` runs **before** ``init_process_group`` because NCCL
    binds its communicator to the current device; leaving every rank on device 0
    is the classic way to get a job that initialises, runs, and deadlocks at the
    first all-reduce.

    Args:
        requested_device: Passed through to
            :func:`~src.utils.training.device.select_device` on the
            single-process path. Ignored when distributed, where the device is
            ``cuda:{local_rank}``.
        timeout_minutes: Collective timeout. The default is generous because the
            first step of a compiled run can spend minutes in graph capture
            while its peers sit in a barrier.
    """
    from src.utils.training.device import select_device

    log = logger or LOGGER

    if not launched_distributed():
        if dist.is_available() and dist.is_initialized():
            # Someone else brought the group up (a test, or a nested launcher).
            return _context_from_live_group()
        return single_process_context(select_device(requested_device))

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))

    if torch.cuda.is_available():
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but this node has "
                f"{torch.cuda.device_count()} visible CUDA device(s). Launch with "
                f"--nproc_per_node <= {torch.cuda.device_count()}, or set CUDA_VISIBLE_DEVICES."
            )
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    backend = resolve_backend(device)
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            timeout=datetime.timedelta(minutes=float(timeout_minutes)),
        )

    context = DistributedContext(
        enabled=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        local_world_size=max(local_world_size, 1),
        backend=backend,
        device=device,
    )
    log.info(
        "Distributed | rank %s/%s (local %s of %s) backend=%s device=%s",
        rank, world_size, local_rank, context.local_world_size, backend, device,
    )
    return context


def _context_from_live_group() -> DistributedContext:
    """Describe a process group that something else initialised."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available() and dist.get_backend() == "nccl":
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return DistributedContext(
        enabled=world_size > 1,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        local_world_size=int(os.environ.get("LOCAL_WORLD_SIZE", world_size)),
        backend=dist.get_backend(),
        device=device,
    )


def shutdown_distributed(context: DistributedContext) -> None:
    """Leave the process group, tolerating a group that is already down.

    Called from a ``finally`` block, which is exactly the situation where one
    rank has raised and the others are somewhere else entirely; a barrier here
    would hang the job instead of letting it report the original exception.
    """
    if not context.enabled or not dist.is_available() or not dist.is_initialized():
        return
    try:
        dist.destroy_process_group()
    except Exception as exc:  # pragma: no cover - depends on how the job died
        LOGGER.debug("destroy_process_group failed during shutdown: %s", exc)


# ------------------------------------------------------------------ collectives


def barrier(context: DistributedContext) -> None:
    """Synchronise every rank. A no-op on one process."""
    if not context.enabled or not dist.is_initialized():
        return
    if dist.get_backend() == "nccl":
        # NCCL barriers are enqueued on the current stream, so the device has to
        # be named or rank 0's barrier can be attributed to device 0 on every
        # rank -- which is a deadlock, not an error message.
        dist.barrier(device_ids=[context.local_rank])
    else:
        dist.barrier()


def all_reduce_mean(tensor: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    """Average ``tensor`` across ranks, in place. Returns it for chaining."""
    if not context.enabled or not dist.is_initialized():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor.div_(context.world_size)
    return tensor


def all_reduce_sum(tensor: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    """Sum ``tensor`` across ranks, in place. Returns it for chaining."""
    if not context.enabled or not dist.is_initialized():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def all_reduce_max(tensor: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    """Element-wise max across ranks, in place. Returns it for chaining.

    The reduction that turns per-rank *opinions* into a job-wide decision: "any
    rank wants to stop" and "any rank's save timer fired" are both a max over a
    0/1 flag. That matters because those conditions are wall-clock based and
    therefore genuinely differ between ranks, and acting on a local answer means
    one rank entering a collective its peers never reach.
    """
    if not context.enabled or not dist.is_initialized():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor


def gather_objects(payload: Any, context: DistributedContext) -> list[Any]:
    """Collect one picklable object per rank, ordered by rank, on every rank.

    Used for the things that are *not* tensors: per-rank RNG states at
    checkpoint time, and per-rank prediction lists at evaluation time.
    """
    if not context.enabled or not dist.is_initialized():
        return [payload]
    gathered: list[Any] = [None] * context.world_size
    dist.all_gather_object(gathered, payload)
    return gathered


def broadcast_object(payload: Any, context: DistributedContext, src: int = 0) -> Any:
    """Send ``src``'s copy of a picklable object to every rank."""
    if not context.enabled or not dist.is_initialized():
        return payload
    holder = [payload if context.rank == src else None]
    dist.broadcast_object_list(holder, src=src)
    return holder[0]


def broadcast_module_state(
    module: torch.nn.Module,
    context: DistributedContext,
    src: int = 0,
) -> None:
    """Force every rank's copy of ``module`` to match ``src``'s, parameters and buffers.

    DDP does this for the module it wraps, at construction. It does **not** do it
    for anything else, and stage 1 has two things it must be done for anyway: the
    EMA teacher (outside DDP by construction, because it takes no gradient) and
    the criterion's centering buffer. Both feed the *targets* the student is
    trained against, so a divergence there is not a slow drift -- it is ranks
    optimising different objectives while every log looks healthy.
    """
    if not context.enabled or not dist.is_initialized():
        return
    with torch.no_grad():
        for tensor in list(module.parameters()) + list(module.buffers()):
            dist.broadcast(tensor.data, src=src)


def logsumexp_across_ranks(
    tensor: torch.Tensor,
    context: DistributedContext,
    dim: int | None = None,
    keepdim: bool = False,
) -> torch.Tensor:
    """``logsumexp`` over the tensor **concatenated across ranks** along ``dim``.

    Exactly equal to running ``torch.logsumexp`` on one process over the
    concatenated input, up to floating-point reduction order: the max-shift is
    taken globally (an all-reduce MAX), every rank's shifted sum is added (an
    all-reduce SUM), and the log is taken once at the end. Doing it any other way
    -- summing local ``logsumexp`` results, or shifting by a local max --
    silently gives a different number.

    ``dim=None`` reduces over every element of every rank, which is the
    normalisation Sinkhorn opens with.

    This is the whole mechanism behind ``distributed_sinkhorn``: the prototype
    marginal is a reduction along the *batch* axis, and the batch axis is what is
    sharded.
    """
    if dim is None:
        flat = tensor.reshape(-1)
        result = logsumexp_across_ranks(flat, context, dim=0, keepdim=False)
        return result

    if not context.enabled or not dist.is_initialized():
        return torch.logsumexp(tensor, dim=dim, keepdim=keepdim)

    local_max = tensor.amax(dim=dim, keepdim=True)
    # An all-empty shard would carry -inf into the shift and produce nan; the
    # sampler's drop_last makes that impossible here, but the guard costs one
    # pointwise op and removes a silent nan source.
    local_max = torch.where(torch.isfinite(local_max), local_max, torch.zeros_like(local_max))
    dist.all_reduce(local_max, op=dist.ReduceOp.MAX)

    shifted_sum = (tensor - local_max).exp().sum(dim=dim, keepdim=True)
    dist.all_reduce(shifted_sum, op=dist.ReduceOp.SUM)

    result = local_max + shifted_sum.log()
    return result if keepdim else result.squeeze(dim)


# ------------------------------------------------------------------- utilities


@contextmanager
def main_process_first(context: DistributedContext) -> Iterator[None]:
    """Run the body on rank 0 first, then on everyone else.

    For work that writes a shared file and is then *read* by every rank -- the
    dataset CSV, a downloaded weight file. Without the ordering, ranks race to
    write the same path and the losers read a half-written file.
    """
    if not context.enabled:
        yield
        return
    if not context.is_main:
        barrier(context)
    try:
        yield
    finally:
        if context.is_main:
            barrier(context)


def available_cpus() -> int:
    """CPU count this process may actually use.

    ``os.cpu_count()`` reports the *machine*, which on Kaggle and inside any
    cgroup-limited container is several times what the process is allowed to
    schedule on. The affinity mask is the honest number where it exists.
    """
    getaffinity = getattr(os, "sched_getaffinity", None)
    if getaffinity is not None:
        try:
            return max(len(getaffinity(0)), 1)
        except OSError:  # pragma: no cover - platform dependent
            pass
    return max(os.cpu_count() or 1, 1)


#: Ceiling on what ``resolve_num_workers("auto")`` returns, per rank.
#:
#: Raised from 8 to 16 after the stage-1 audit. The 8 was chosen for a 4-vCPU
#: Kaggle T4x2 instance and became the binding limit on every larger host: the
#: shipped 13.34-hour run sat at a mean ``data_wait_fraction`` of 0.916 on a
#: 48-physical-core machine. Callers pass ``data.num_workers_auto_cap`` so the
#: ceiling is configurable rather than a constant a reader has to find in the
#: source.
DEFAULT_NUM_WORKERS_AUTO_CAP = 16


def resolve_num_workers(
    requested: Any,
    context: DistributedContext,
    *,
    auto_cap: int = DEFAULT_NUM_WORKERS_AUTO_CAP,
    logger: logging.Logger | None = None,
) -> int:
    """Per-rank dataloader worker count, budgeted across the ranks on this node.

    ``num_workers`` is a **per-process** setting, so a value tuned on one GPU
    becomes ``local_world_size`` times as many processes the moment the job is
    distributed. On a Kaggle T4x2 instance -- 4 vCPUs, 2 ranks -- the configured
    8 would spawn 16 workers onto 4 cores, and the augmentation pipeline would
    spend its time context-switching rather than decoding.

    ``"auto"`` resolves to the affinity-aware core count per rank, capped at
    ``auto_cap``; an explicit integer is divided by ``local_world_size`` with a
    log line saying so, because silently honouring it is the failure above and
    silently ignoring it is worse.

    Nothing here touches the objective. Workers change *who* computes an
    augmentation, never what it is: the per-view RNG streams are seeded from the
    loader generator either way, so a run at 0 workers and one at 16 optimise the
    identical function.
    """
    log = logger or LOGGER
    cpus = available_cpus()
    per_rank_cpus = max(cpus // max(context.local_world_size, 1), 1)
    cap = max(int(auto_cap), 0)

    if isinstance(requested, str) and requested.strip().lower() == "auto":
        workers = min(per_rank_cpus, cap) if cap else per_rank_cpus
        log.info(
            "Dataloader workers: auto -> %s per rank (%s usable CPUs / %s local ranks, cap %s).",
            workers, cpus, context.local_world_size, cap or "none",
        )
        return workers

    workers = max(int(requested), 0)
    if context.local_world_size > 1 and workers > 0:
        shared = max(workers // context.local_world_size, 1)
        if shared != workers:
            log.info(
                "Dataloader workers: %s configured -> %s per rank, so the %s local ranks "
                "together use the configured budget rather than %sx it.",
                workers, shared, context.local_world_size, context.local_world_size,
            )
        return shared
    return workers


def buffer_sync_kwarg(enabled: bool = False) -> dict[str, Any]:
    """The right "do not re-broadcast buffers every forward" flag for this torch.

    ``broadcast_buffers`` is deprecated in favour of ``forward_sync_buffers``,
    and the two differ in more than spelling: ``broadcast_buffers=False``
    suppresses the *initial* sync as well, while ``forward_sync_buffers=False``
    keeps it. Either is correct here -- the only buffers on this path are
    SwinV2's ``relative_coords_table`` and ``relative_position_index``, which are
    constants derived from the window geometry and identical on every rank by
    construction -- so the newer flag is used where it exists.

    What is *not* optional is turning the per-forward sync off. Leaving it on
    adds a collective over several megabytes of constants to every step, and on
    two T4s without NVLink that is paid over PCIe.
    """
    from torch.nn.parallel import DistributedDataParallel
    import inspect

    parameters = inspect.signature(DistributedDataParallel.__init__).parameters
    if "forward_sync_buffers" in parameters:
        return {"forward_sync_buffers": bool(enabled)}
    return {"broadcast_buffers": bool(enabled)}


def unwrap_model(module: torch.nn.Module) -> torch.nn.Module:
    """Strip ``DistributedDataParallel`` and ``torch.compile`` wrappers.

    Both wrappers rename every key of ``state_dict`` (``module.`` and
    ``_orig_mod.`` respectively). This repository keeps both off the module tree
    precisely so that never happens to a saved checkpoint, but a checkpoint that
    arrived from somewhere else may still carry the prefixes -- see
    :func:`strip_wrapper_prefixes`.
    """
    seen: set[int] = set()
    while id(module) not in seen:
        seen.add(id(module))
        inner = getattr(module, "module", None)
        if isinstance(inner, torch.nn.Module):
            module = inner
            continue
        inner = getattr(module, "_orig_mod", None)
        if isinstance(inner, torch.nn.Module):
            module = inner
            continue
        break
    return module


#: Prefixes that a wrapper adds to every key of a ``state_dict``.
WRAPPER_PREFIXES = ("module.", "_orig_mod.")


def strip_wrapper_prefixes(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Remove ``module.`` / ``_orig_mod.`` prefixes from every key, repeatedly.

    Loading a DDP-saved checkpoint into a bare module matches **zero** keys, and
    because ``checkpoint_strict: false`` is this repository's default, the result
    is a run that logs one line about missing keys and then trains happily
    against random weights. That failure has already happened here once, with
    ``torch.compile``'s prefix; this is the general fix.
    """
    cleaned = dict(state_dict)
    changed = True
    while changed:
        changed = False
        for prefix in WRAPPER_PREFIXES:
            if cleaned and all(key.startswith(prefix) for key in cleaned):
                cleaned = {key[len(prefix):]: value for key, value in cleaned.items()}
                changed = True
    return cleaned


def reduce_metrics(
    metrics: dict[str, float],
    context: DistributedContext,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Average a flat scalar metric dict across ranks.

    One collective for the whole dict rather than one per key: the keys are
    sorted so every rank packs the same order, which is a requirement rather
    than a tidiness preference -- a dict that iterates differently on two ranks
    would average unrelated numbers together without erroring.
    """
    if not context.enabled or not dist.is_initialized() or not metrics:
        return dict(metrics)
    keys = sorted(metrics)
    packed = torch.tensor(
        [float(metrics[key]) for key in keys],
        dtype=torch.float64,
        device=device if device is not None else context.device,
    )
    all_reduce_mean(packed, context)
    return {key: float(value) for key, value in zip(keys, packed.tolist())}


def concat_across_ranks(values: Sequence[Any], context: DistributedContext) -> list[Any]:
    """Flatten one list per rank into a single rank-ordered list on every rank."""
    if not context.enabled or not dist.is_initialized():
        return list(values)
    gathered = gather_objects(list(values), context)
    return [item for shard in gathered for item in shard]
