"""Multi-process correctness: the claims DDP makes that are not self-evident.

Every test here runs real processes over a real Gloo process group on CPU, which
is what makes them meaningful *and* what makes them runnable anywhere -- no GPU,
no NCCL, no launcher. The arithmetic being checked is backend-independent: an
all-reduce is an all-reduce.

Four claims are pinned, and each one has a plausible-looking failure mode that
produces no error:

1. **DDP's averaged gradient equals the single-process gradient** on the batch
   the ranks jointly hold -- including across ``no_sync`` accumulation. A wrong
   answer here trains a different model at a loss curve that looks correct.
2. **Distributed Sinkhorn equals single-process Sinkhorn** on the concatenated
   batch. The tempting implementation -- reduce the local ``logsumexp`` results
   -- is off by a per-rank shift and produces an assignment that is *nearly*
   doubly stochastic, which nothing downstream would notice.
3. **Per-rank Sinkhorn is untouched by the presence of a process group**, which
   is what makes the default setting reproduce the single-GPU numbers.
4. **No wrapper prefix reaches a state dict.** ``module.`` on the stage-1
   backbone keys would load into stage 2 as zero matched keys and, under this
   repository's ``checkpoint_strict: false`` default, train a full run against a
   random encoder while logging one line about it.
"""

from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.losses.dino import CustomDINOLoss, sinkhorn_knopp
from src.utils.training.distributed import (
    DistributedContext,
    all_reduce_mean,
    logsumexp_across_ranks,
    resolve_num_workers,
    single_process_context,
    strip_wrapper_prefixes,
    unwrap_model,
)

pytestmark = pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="torch.distributed with the Gloo backend is required",
)

WORLD_SIZE = 2
#: Kept small: these tests spawn processes, and every second here is paid on
#: every run of the suite.
FEATURES = 8
BATCH_PER_RANK = 4


# --------------------------------------------------------------- test harness


def _context(rank: int, world_size: int) -> DistributedContext:
    return DistributedContext(
        enabled=world_size > 1,
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        local_world_size=world_size,
        backend="gloo",
        device=torch.device("cpu"),
    )


def _init(rank: int, world_size: int, init_file: str) -> DistributedContext:
    """Join a file-backed Gloo group. No free port needed, so no flakiness."""
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        world_size=world_size,
        rank=rank,
    )
    return _context(rank, world_size)


def _tiny_model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(FEATURES, 16), nn.ReLU(), nn.Linear(16, 3))


def _full_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """The batch the whole job holds, deterministic across processes."""
    generator = torch.Generator().manual_seed(1234)
    inputs = torch.randn(WORLD_SIZE * BATCH_PER_RANK, FEATURES, generator=generator)
    targets = torch.randint(0, 3, (WORLD_SIZE * BATCH_PER_RANK,), generator=generator)
    return inputs, targets


def _run(worker, world_size: int, tmp_path: Path, *args) -> list:
    """Spawn ``world_size`` ranks, run ``worker``, and collect what each saved.

    Results come back through files rather than a queue: a rank that dies mid-run
    leaves the parent with a missing file and a clear assertion, where a queue
    would leave it blocked.
    """
    init_file = tmp_path / "gloo_init"
    mp.spawn(
        worker,
        args=(world_size, str(init_file), str(tmp_path), *args),
        nprocs=world_size,
        join=True,
    )
    outputs = []
    for rank in range(world_size):
        path = tmp_path / f"result_{rank}.pt"
        assert path.exists(), f"rank {rank} produced no result"
        outputs.append(torch.load(path, map_location="cpu", weights_only=False))
    return outputs


# ------------------------------------------------------- 1. gradient equality


def _ddp_gradient_worker(rank: int, world_size: int, init_file: str, tmp_dir: str, accum: int) -> None:
    """One rank: DDP over half the batch, accumulated over ``accum`` micro-batches."""
    from torch.nn.parallel import DistributedDataParallel

    context = _init(rank, world_size, init_file)
    model = _tiny_model()
    ddp = DistributedDataParallel(model)

    inputs, targets = _full_batch()
    # Rank-major sharding, matching DistributedSampler's contiguous-after-shuffle
    # contract closely enough for the arithmetic under test.
    shard = slice(rank * BATCH_PER_RANK, (rank + 1) * BATCH_PER_RANK)
    rank_inputs, rank_targets = inputs[shard], targets[shard]

    micro = BATCH_PER_RANK // accum
    ddp.zero_grad(set_to_none=True)
    for step in range(accum):
        piece = slice(step * micro, (step + 1) * micro)
        is_last = step + 1 == accum
        # `no_sync` on every micro-batch but the last: the gradient crosses the
        # interconnect once per optimizer step, not once per micro-batch.
        with nullcontext() if is_last else ddp.no_sync():
            loss = nn.functional.cross_entropy(ddp(rank_inputs[piece]), rank_targets[piece])
            (loss / accum).backward()

    torch.save(
        {name: parameter.grad.clone() for name, parameter in model.named_parameters()},
        Path(tmp_dir) / f"result_{rank}.pt",
    )
    dist.destroy_process_group()


@pytest.mark.parametrize("accum", [1, 2])
def test_ddp_gradient_equals_the_single_process_gradient(tmp_path, accum):
    """``mean_r(grad on shard r)`` is the gradient on the concatenated batch.

    This is the property that lets a DDP run be called equivalent to a
    single-GPU one at all, and it holds because the loss is a **mean over
    samples** with equal shard sizes. It is checked across ``no_sync``
    accumulation too, because that is where the divisor is easiest to get wrong:
    dividing by ``accum`` on each rank and letting DDP divide by the world size
    has to compose to a single division by the global batch, and an
    implementation that reduced on every micro-batch instead would still produce
    this number while paying ``accum`` times the communication.
    """
    per_rank = _run(_ddp_gradient_worker, WORLD_SIZE, tmp_path, accum)

    inputs, targets = _full_batch()
    reference = _tiny_model()
    loss = nn.functional.cross_entropy(reference(inputs), targets)
    loss.backward()

    for name, parameter in reference.named_parameters():
        for rank, grads in enumerate(per_rank):
            torch.testing.assert_close(
                grads[name], parameter.grad, rtol=1e-5, atol=1e-6,
                msg=f"rank {rank} disagrees with the single-process gradient for {name}",
            )


def _ddp_state_dict_worker(rank: int, world_size: int, init_file: str, tmp_dir: str) -> None:
    from torch.nn.parallel import DistributedDataParallel

    _init(rank, world_size, init_file)
    model = _tiny_model()
    wrapped = DistributedDataParallel(model)
    torch.save(
        {
            "inner": sorted(model.state_dict()),
            "wrapper": sorted(wrapped.state_dict()),
            "unwrapped_is_inner": unwrap_model(wrapped) is model,
        },
        Path(tmp_dir) / f"result_{rank}.pt",
    )
    dist.destroy_process_group()


def test_the_wrapped_module_keeps_its_own_unprefixed_state_dict(tmp_path):
    """Saving the inner module -- as both trainers do -- yields no ``module.``.

    The wrapper's own ``state_dict`` does carry the prefix, which is exactly why
    this repository never saves it. That failure has already happened here once
    with ``torch.compile``'s ``_orig_mod.``: the stage-1 backbone is the only
    handoff to stage 2, ``checkpoint_strict: false`` is the default there, and a
    prefixed key set loads as zero matches, one log line, and a full run against
    a random encoder.
    """
    for result in _run(_ddp_state_dict_worker, WORLD_SIZE, tmp_path):
        assert not any(key.startswith("module.") for key in result["inner"])
        assert all(key.startswith("module.") for key in result["wrapper"])
        assert result["unwrapped_is_inner"]


def test_strip_wrapper_prefixes_handles_both_wrappers_and_nesting():
    """A checkpoint from elsewhere may carry either prefix, or both."""
    assert strip_wrapper_prefixes({"module.a": 1, "module.b": 2}) == {"a": 1, "b": 2}
    assert strip_wrapper_prefixes({"_orig_mod.a": 1}) == {"a": 1}
    assert strip_wrapper_prefixes({"module._orig_mod.a": 1}) == {"a": 1}
    # A prefix on only *some* keys is not a wrapper prefix and must be left alone.
    mixed = {"module.a": 1, "b": 2}
    assert strip_wrapper_prefixes(mixed) == mixed


# --------------------------------------------------------- 2 & 3. Sinkhorn


def _sinkhorn_worker(rank: int, world_size: int, init_file: str, tmp_dir: str) -> None:
    context = _init(rank, world_size, init_file)
    logits = _sinkhorn_logits()
    shard = logits[rank * BATCH_PER_RANK : (rank + 1) * BATCH_PER_RANK]
    torch.save(
        {
            "distributed": sinkhorn_knopp(shard, temperature=0.07, iterations=3, context=context),
            "local": sinkhorn_knopp(shard, temperature=0.07, iterations=3),
            "logsumexp_all": logsumexp_across_ranks(shard.reshape(-1), context, dim=0),
        },
        Path(tmp_dir) / f"result_{rank}.pt",
    )
    dist.destroy_process_group()


def _sinkhorn_logits() -> torch.Tensor:
    generator = torch.Generator().manual_seed(7)
    return torch.randn(WORLD_SIZE * BATCH_PER_RANK, 32, generator=generator)


def test_distributed_sinkhorn_matches_the_single_process_result(tmp_path):
    """The doubly-stochastic assignment is over the *global* batch, exactly.

    The prototype marginal reduces along the batch axis, which is the axis that
    was sharded, so it needs a cross-rank ``logsumexp``; the sample marginal
    reduces along prototypes, which every rank holds in full, and stays local.
    Getting only the second one right yields an assignment whose per-prototype
    mass is right within a rank and wrong across the job -- and since the result
    is detached and used as a training *target*, nothing downstream errors.
    """
    results = _run(_sinkhorn_worker, WORLD_SIZE, tmp_path)
    reference = sinkhorn_knopp(_sinkhorn_logits(), temperature=0.07, iterations=3)

    stacked = torch.cat([result["distributed"] for result in results], dim=0)
    torch.testing.assert_close(stacked, reference, rtol=1e-5, atol=1e-6)

    # The sample marginal is the last normalisation the algorithm applies, so
    # each row summing to 1 is exact -- and it is the contract the softmax path
    # also satisfies, which is what lets the two centering modes be swapped.
    # (The prototype marginal is only approximately uniform after finitely many
    # iterations, in the distributed and single-process paths alike; that is
    # Sinkhorn, not a distribution artefact.)
    torch.testing.assert_close(stacked.sum(dim=1), torch.ones(stacked.shape[0]), rtol=1e-4, atol=1e-5)

    # The test has teeth: normalising per rank and concatenating -- the
    # implementation someone reaches for first -- gives a materially different
    # assignment, so the agreement above is not an accident of small numbers.
    naive = torch.cat([result["local"] for result in results], dim=0)
    assert (naive - reference).abs().max() > 1e-3


def test_per_rank_sinkhorn_is_what_reproduces_the_single_gpu_numbers(tmp_path):
    """Without ``context``, each rank computes exactly what one GPU would.

    This is why ``distributed_sinkhorn`` defaults to false. Sinkhorn is already
    applied per micro-batch under gradient accumulation, so a rank holding the
    same number of images computes the identical function -- the objective does
    not change when the job is distributed unless someone asks for it to.
    """
    results = _run(_sinkhorn_worker, WORLD_SIZE, tmp_path)
    logits = _sinkhorn_logits()
    for rank, result in enumerate(results):
        shard = logits[rank * BATCH_PER_RANK : (rank + 1) * BATCH_PER_RANK]
        torch.testing.assert_close(
            result["local"], sinkhorn_knopp(shard, temperature=0.07, iterations=3)
        )


def test_logsumexp_across_ranks_matches_the_concatenated_reduction(tmp_path):
    """The primitive the distributed normaliser is built on, checked alone."""
    results = _run(_sinkhorn_worker, WORLD_SIZE, tmp_path)
    expected = torch.logsumexp(_sinkhorn_logits().reshape(-1), dim=0)
    for result in results:
        torch.testing.assert_close(result["logsumexp_all"], expected, rtol=1e-6, atol=1e-7)


# ----------------------------------------------------------- EMA centering


def _center_worker(rank: int, world_size: int, init_file: str, tmp_dir: str) -> None:
    context = _init(rank, world_size, init_file)
    criterion = CustomDINOLoss(
        out_dim=16, num_crops=2, warmup_teacher_temp=0.04, teacher_temp=0.07,
        warmup_teacher_temp_epochs=1, num_epochs=1, centering="ema", center_momentum=0.5,
        lambda_koleo=0.0, context=context,
    )
    logits = _center_logits()
    shard = logits[rank * BATCH_PER_RANK : (rank + 1) * BATCH_PER_RANK]
    criterion.update_center(shard)
    torch.save({"center": criterion.center.clone()}, Path(tmp_dir) / f"result_{rank}.pt")
    dist.destroy_process_group()


def _center_logits() -> torch.Tensor:
    generator = torch.Generator().manual_seed(11)
    # Deliberately different scales per shard: a per-rank centre would diverge
    # visibly, which is the failure this pins.
    logits = torch.randn(WORLD_SIZE * BATCH_PER_RANK, 16, generator=generator)
    logits[BATCH_PER_RANK:] += 5.0
    return logits


def test_the_ema_centre_is_one_global_statistic(tmp_path):
    """Every rank's centering buffer is the same, and is the global mean.

    Not configurable, unlike the Sinkhorn choice: ``center`` is *state* that the
    teacher's targets subtract and that the checkpoint stores. Per-rank centres
    mean ranks optimising against different targets, and a checkpoint whose
    contents depend on which rank happened to write it.
    """
    results = _run(_center_worker, WORLD_SIZE, tmp_path)
    expected = 0.5 * _center_logits().mean(dim=0, keepdim=True)  # zero-init * m + mean * (1 - m)

    for rank, result in enumerate(results):
        torch.testing.assert_close(result["center"], expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(results[0]["center"], results[1]["center"])


# ------------------------------------------------------- single-process paths


def test_helpers_are_no_ops_without_a_process_group():
    """Every call site reads the same on one process; nothing needs a branch."""
    context = single_process_context(torch.device("cpu"))
    assert context.is_main and context.world_size == 1 and not context.enabled

    tensor = torch.tensor([2.0, 4.0])
    torch.testing.assert_close(all_reduce_mean(tensor.clone(), context), tensor)
    torch.testing.assert_close(
        logsumexp_across_ranks(tensor, context, dim=0), torch.logsumexp(tensor, dim=0)
    )


def test_worker_budget_is_shared_between_local_ranks():
    """``num_workers`` is per process, so N ranks would otherwise spawn N times it.

    On a Kaggle T4x2 instance -- 4 vCPUs, 2 ranks -- honouring the configured 8
    literally puts 16 augmentation workers on 4 cores.
    """
    solo = single_process_context(torch.device("cpu"))
    assert resolve_num_workers(8, solo) == 8

    paired = DistributedContext(
        enabled=True, rank=0, local_rank=0, world_size=2, local_world_size=2,
        backend="gloo", device=torch.device("cpu"),
    )
    assert resolve_num_workers(8, paired) == 4
    # Never zero: a 0 would move the whole augmentation pipeline onto the
    # training process, which is a different and much slower failure.
    assert resolve_num_workers(1, paired) == 1
    # 0 is an explicit request for in-process loading and must survive.
    assert resolve_num_workers(0, paired) == 0

    assert resolve_num_workers("auto", paired) >= 1


# ------------------------------- 6. stage-1 changes under a real 2-rank job


def _koleo_worker(rank: int, world_size: int, init_file: str, tmp_dir: str) -> None:
    """One rank: per-view KoLeo over its own shard of the global views."""
    from src.losses.dino import grouped_koleo

    _init(rank, world_size, init_file)

    # The full job's two global views, view-major and identical on every rank.
    generator = torch.Generator().manual_seed(99)
    batch = WORLD_SIZE * BATCH_PER_RANK
    view0 = torch.randn(batch, FEATURES, generator=generator)
    view1 = view0 + 0.4 * torch.randn(batch, FEATURES, generator=generator)

    # Images shard, views do not: a rank owns BOTH views of its own images,
    # because Eq. 1 pairs a student view against the teacher's output for that
    # same image.
    shard = slice(rank * BATCH_PER_RANK, (rank + 1) * BATCH_PER_RANK)
    local = torch.cat([view0[shard], view1[shard]], dim=0)

    torch.save(
        {
            "per_view": grouped_koleo(local, WORLD_SIZE, "per_view").clone(),
            "all_views": grouped_koleo(local, WORLD_SIZE, "all_views").clone(),
        },
        Path(tmp_dir) / f"result_{rank}.pt",
    )
    dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="torch.distributed with the Gloo backend is required",
)
def test_per_view_koleo_is_a_purely_local_statistic(tmp_path):
    """Each rank's KoLeo depends only on its own shard, and nothing all-reduces it.

    That is deliberate and matches the reference implementation: KoLeo is a
    nearest-neighbour statistic over the local view block, and it is already
    computed per *micro-batch* under accumulation — so a rank holding the same
    number of images computes the identical function a single-GPU run computes on
    its micro-batch. Unlike Sinkhorn, it has no `distributed_*` switch.

    The corollary is the one worth stating: splitting the batch across ranks
    halves what KoLeo estimates from. If a second GPU is available, raise
    `data.batch_size` per rank rather than splitting the configured batch.
    """
    from src.losses.dino import grouped_koleo

    results = _run(_koleo_worker, WORLD_SIZE, tmp_path)

    generator = torch.Generator().manual_seed(99)
    batch = WORLD_SIZE * BATCH_PER_RANK
    view0 = torch.randn(batch, FEATURES, generator=generator)
    view1 = view0 + 0.4 * torch.randn(batch, FEATURES, generator=generator)

    for rank, payload in enumerate(results):
        shard = slice(rank * BATCH_PER_RANK, (rank + 1) * BATCH_PER_RANK)
        local = torch.cat([view0[shard], view1[shard]], dim=0)
        # Recomputed in a single process from that rank's shard alone.
        assert payload["per_view"] == pytest.approx(
            float(grouped_koleo(local, WORLD_SIZE, "per_view")), abs=1e-6
        )
        assert payload["all_views"] == pytest.approx(
            float(grouped_koleo(local, WORLD_SIZE, "all_views")), abs=1e-6
        )
    # And the two scopes genuinely differ on this data, so the check is not vacuous.
    assert results[0]["per_view"] != pytest.approx(results[0]["all_views"], abs=1e-3)


def _decomposition_worker(rank: int, world_size: int, init_file: str, tmp_dir: str) -> None:
    """One rank: the loss decomposition over its own shard, under Sinkhorn."""
    from src.losses.dino import CustomDINOLoss

    context = _init(rank, world_size, init_file)
    criterion = CustomDINOLoss(
        out_dim=FEATURES, num_crops=4, warmup_teacher_temp=0.04, teacher_temp=0.04,
        warmup_teacher_temp_epochs=0, num_epochs=1, centering="sinkhorn",
        lambda_koleo=0.0, context=context,
    )
    generator = torch.Generator().manual_seed(7 + rank)
    student = [torch.randn(BATCH_PER_RANK, FEATURES, generator=generator) for _ in range(4)]
    teacher = [torch.randn(BATCH_PER_RANK, FEATURES, generator=generator) for _ in range(2)]

    total = criterion(student, teacher, epoch=0)
    metrics = criterion.last_metrics
    torch.save(
        {
            "loss": float(total),
            "ce": metrics["dino_cross_entropy"],
            "entropy": metrics["teacher_entropy_cross_view"],
            "kl": metrics["teacher_student_kl"],
        },
        Path(tmp_dir) / f"result_{rank}.pt",
    )
    dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="torch.distributed with the Gloo backend is required",
)
def test_the_loss_decomposition_holds_on_every_rank(tmp_path):
    """`CE = H + KL` is a per-rank identity, so the trainer's device-tensor
    accumulators are meaningful before the epoch-boundary all-reduce.

    The trainer sums `teacher_student_kl` per micro-batch on the device and
    all-reduces the *epoch mean* once. That is only the global mean if the
    identity holds locally on each rank, which is what this pins.
    """
    for payload in _run(_decomposition_worker, WORLD_SIZE, tmp_path):
        assert payload["ce"] == pytest.approx(payload["loss"], abs=1e-5)
        assert payload["entropy"] + payload["kl"] == pytest.approx(payload["ce"], abs=1e-4)
        assert payload["kl"] > -1e-5
