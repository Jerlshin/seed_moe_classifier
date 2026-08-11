"""Resume must continue the run, not restart it warm.

The distinction is the whole point. A checkpoint that carries only weights lets a
relaunched job *look* like a continuation while the optimizer's moments restart
at zero, the LR schedule restarts at its peak and the RNG restarts wherever the
new process seeded -- none of which produces an error, and all of which produce a
different run. On a platform that ends the session every few hours, that is the
difference between a 300-epoch run finishing and never finishing.

The headline test is
:func:`test_resume_continues_the_run_exactly`: training ``n`` steps in one
process is compared parameter-by-parameter against training ``k``, saving,
restoring into **freshly constructed** objects, and training ``n - k``. Anything
the checkpoint forgets shows up there as a numerical difference, which is why it
is written against fresh objects rather than the same ones -- reusing them would
let state survive through Python rather than through the file.

The rest cover the file itself: a write that cannot tear, and a reader that
distrusts the newest file precisely because a killed session damages that one
first.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.training.checkpoint import CheckpointManager
from src.utils.training.distributed import single_process_context
from src.utils.training.resume import (
    COMPLETE_KEY,
    TEMP_SUFFIX,
    TrainingProgress,
    atomic_save,
    build_checkpoint_payload,
    capture_rng,
    find_latest_checkpoint,
    is_valid_checkpoint,
    load_checkpoint_payload,
    resolve_resume_path,
    restore_components,
    restore_rng,
    restore_rng_states,
)

CONTEXT = single_process_context(torch.device("cpu"))


# --------------------------------------------------------------- the artifacts


def _model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(6, 12), nn.Tanh(), nn.Linear(12, 3))


def _training_objects(seed: int = 0):
    """A model plus every piece of state a real step mutates."""
    model = _model(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)
    return model, optimizer, scheduler


def _components(model, optimizer, scheduler) -> dict:
    return {"model": model, "optimizer": optimizer, "scheduler": scheduler}


def _step(model, optimizer, scheduler, step_index: int) -> None:
    """One optimizer step on data drawn from the *live* RNG.

    Drawing from the global RNG rather than from a fixed tensor is deliberate:
    it makes the RNG part of the state under test, so a checkpoint that restores
    the weights but not the random stream fails
    :func:`test_resume_continues_the_run_exactly` instead of passing it.
    """
    inputs = torch.randn(4, 6)
    targets = torch.randn(4, 3)
    optimizer.zero_grad(set_to_none=True)
    nn.functional.mse_loss(model(inputs), targets).backward()
    optimizer.step()
    scheduler.step()


def _parameters(model) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


# ------------------------------------------------------------- exact resume


def test_resume_continues_the_run_exactly(tmp_path):
    """Split training at step 3 and it lands where the unsplit run did.

    Every piece of state has to survive for this to hold: the weights, AdamW's
    two moment buffers (restart them and the first post-resume step takes a
    wildly mis-scaled update), the scheduler's ``last_epoch`` (restart it and the
    learning rate jumps back to its peak), and the RNG (restart it and the
    resumed run sees different data). All four are silent failures.
    """
    total_steps, split_at = 6, 3

    torch.manual_seed(99)
    random.seed(99)
    np.random.seed(99)
    reference_model, reference_optimizer, reference_scheduler = _training_objects()
    for index in range(total_steps):
        _step(reference_model, reference_optimizer, reference_scheduler, index)
    expected = _parameters(reference_model)

    # --- the interrupted run ------------------------------------------------
    torch.manual_seed(99)
    random.seed(99)
    np.random.seed(99)
    model, optimizer, scheduler = _training_objects()
    for index in range(split_at):
        _step(model, optimizer, scheduler, index)

    payload = build_checkpoint_payload(
        components=_components(model, optimizer, scheduler),
        progress=TrainingProgress(epoch=0, global_step=split_at, micro_step=split_at),
        context=CONTEXT,
        rng_states=[capture_rng()],
    )
    path = atomic_save(payload, tmp_path / "resume.pth")

    # --- the relaunch, against objects that share nothing with the above ----
    del model, optimizer, scheduler
    torch.manual_seed(0)  # a different seed, so nothing can pass by luck
    fresh_model, fresh_optimizer, fresh_scheduler = _training_objects(seed=5)
    loaded = load_checkpoint_payload(path)
    report = restore_components(
        loaded, _components(fresh_model, fresh_optimizer, fresh_scheduler), strict=True
    )
    restore_rng_states(loaded, CONTEXT)
    progress = TrainingProgress.from_dict(loaded["progress"])

    assert not report["failed"] and not report["missing"]
    assert progress.global_step == split_at

    for index in range(progress.global_step, total_steps):
        _step(fresh_model, fresh_optimizer, fresh_scheduler, index)

    for name, tensor in _parameters(fresh_model).items():
        torch.testing.assert_close(
            tensor, expected[name], rtol=0, atol=0,
            msg=f"{name} diverged: the resumed run is not a continuation",
        )


def test_a_weights_only_checkpoint_is_detectably_not_enough(tmp_path):
    """The control for the test above: dropping the optimizer changes the run.

    Without this, ``test_resume_continues_the_run_exactly`` could pass against an
    implementation that saved nothing at all, as long as the arithmetic happened
    to be insensitive at this scale. It is not.
    """
    torch.manual_seed(99)
    model, optimizer, scheduler = _training_objects()
    for index in range(3):
        _step(model, optimizer, scheduler, index)

    payload = build_checkpoint_payload(
        components={"model": model},  # weights only
        progress=TrainingProgress(global_step=3),
        context=CONTEXT,
        rng_states=[capture_rng()],
    )
    path = atomic_save(payload, tmp_path / "weights_only.pth")

    torch.manual_seed(99)
    reference_model, reference_optimizer, reference_scheduler = _training_objects()
    for index in range(6):
        _step(reference_model, reference_optimizer, reference_scheduler, index)

    fresh_model, fresh_optimizer, fresh_scheduler = _training_objects(seed=5)
    loaded = load_checkpoint_payload(path)
    restore_components(loaded, {"model": fresh_model}, strict=True)
    restore_rng_states(loaded, CONTEXT)
    for index in range(3, 6):
        _step(fresh_model, fresh_optimizer, fresh_scheduler, index)

    differences = [
        (tensor - _parameters(reference_model)[name]).abs().max().item()
        for name, tensor in _parameters(fresh_model).items()
    ]
    assert max(differences) > 1e-6, (
        "a weights-only resume produced the same parameters, so this suite cannot "
        "distinguish a complete checkpoint from an incomplete one"
    )


def test_the_grad_scaler_state_survives(tmp_path):
    """fp16's loss scale is state; restarting it costs skipped steps every resume.

    A ``GradScaler`` that restarts at 65536 spends its first steps overflowing and
    halving, and on a T4 -- where fp16 is the automatic choice, not an option --
    a resume every 30 minutes turns that into a recurring tax.
    """
    scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=1024.0, growth_interval=1)
    # The scale tensor is created lazily on the first `scale()` call, so a
    # scaler that has never scaled anything has no state to save yet.
    scaler.scale(torch.zeros(1))
    scaler.update(new_scale=512.0)

    payload = build_checkpoint_payload(
        components={"scaler": scaler},
        progress=TrainingProgress(),
        context=CONTEXT,
    )
    path = atomic_save(payload, tmp_path / "scaler.pth")

    restored = torch.amp.GradScaler("cpu", enabled=True, init_scale=1024.0)
    restore_components(load_checkpoint_payload(path), {"scaler": restored}, strict=True)
    assert restored.get_scale() == pytest.approx(512.0)


# ------------------------------------------------------------------ RNG state


def test_rng_capture_and_restore_reproduce_the_stream():
    """All three streams, because the pipeline draws from all three."""
    torch.manual_seed(3)
    random.seed(3)
    np.random.seed(3)
    state = capture_rng()

    expected = (torch.randn(4), random.random(), float(np.random.rand()))
    # Advance every stream well past where it was.
    torch.randn(100)
    [random.random() for _ in range(100)]
    np.random.rand(100)

    restore_rng(state)
    torch.testing.assert_close(torch.randn(4), expected[0])
    assert random.random() == expected[1]
    assert float(np.random.rand()) == expected[2]


def test_a_dataloader_generator_is_part_of_the_state():
    """Without it, a mid-epoch resume continues with a re-rolled sample order.

    The weights would be right and the data order wrong -- reproducible in
    aggregate, not reproducible step by step, and invisible in any log.
    """
    generator = torch.Generator().manual_seed(17)
    state = capture_rng(generator)
    expected = torch.randperm(20, generator=generator)

    torch.randperm(20, generator=generator)  # advance it
    restore_rng(state, generator)
    torch.testing.assert_close(torch.randperm(20, generator=generator), expected)


def test_a_checkpoint_from_a_different_world_size_still_restores(tmp_path):
    """Resuming a 2-GPU job on 1 GPU is the ordinary Kaggle case.

    Each rank takes ``rank % saved``. The stream is deterministic from there,
    which is what a resume needs; it does not pretend to reproduce the other
    topology's sample order, and the trainer says so in a warning.
    """
    payload = build_checkpoint_payload(
        components={},
        progress=TrainingProgress(global_step=5),
        context=CONTEXT,
        rng_states=[capture_rng(), capture_rng()],  # as a 2-rank job would write
    )
    path = atomic_save(payload, tmp_path / "two_ranks.pth")
    loaded = load_checkpoint_payload(path)
    assert len(loaded["rng_states"]) == 2

    restore_rng_states(loaded, CONTEXT)  # world_size 1; must not raise
    assert TrainingProgress.from_dict(loaded["progress"]).global_step == 5


# ----------------------------------------------------------------- the file


def test_a_failed_write_leaves_the_previous_checkpoint_intact(tmp_path):
    """``torch.save`` truncates its destination; ``atomic_save`` never does.

    This is the failure that costs a run: a session killed during a save leaves a
    zero-length file *where the good checkpoint used to be*, so the next resume
    finds it, fails to load it, and has nothing to fall back to.
    """
    destination = tmp_path / "checkpoint.pth"
    atomic_save({"value": 1, COMPLETE_KEY: True}, destination)

    class Unpicklable:
        def __reduce__(self):
            raise RuntimeError("simulated failure part-way through the write")

    with pytest.raises(RuntimeError):
        atomic_save({"value": Unpicklable()}, destination)

    assert torch.load(destination, map_location="cpu", weights_only=False)["value"] == 1
    assert not list(tmp_path.glob(f"*{TEMP_SUFFIX}")), "a failed write left its temp file behind"


def test_find_latest_checkpoint_skips_what_it_cannot_load(tmp_path):
    """Newest *and* valid are separate tests, and have to be.

    On a preempted instance the newest file is the one most likely to be damaged,
    so falling back to the previous one is the entire reason for keeping more
    than one.
    """
    good = tmp_path / "resume_0001.pth"
    atomic_save({"value": "good", COMPLETE_KEY: True}, good)

    # Newer, and unreadable.
    corrupt = tmp_path / "resume_0002.pth"
    corrupt.write_bytes(b"not a checkpoint")
    # Newer still, and an in-progress write.
    (tmp_path / f"resume_0003.pth{TEMP_SUFFIX}").write_bytes(b"half written")

    assert not is_valid_checkpoint(corrupt)
    assert find_latest_checkpoint(tmp_path, ("resume_*.pth",)) == good


def test_a_checkpoint_without_the_completion_marker_is_not_valid(tmp_path):
    """A loadable file is not necessarily a finished one."""
    path = tmp_path / "partial.pth"
    torch.save({"value": 1}, path)
    assert not is_valid_checkpoint(path)


def test_resume_path_resolution_covers_the_three_intents(tmp_path):
    """``false`` starts fresh, ``auto`` continues if it can, a path must exist."""
    assert resolve_resume_path(False, tmp_path) is None
    assert resolve_resume_path("false", tmp_path) is None
    assert resolve_resume_path(None, tmp_path) is None

    # `auto` with nothing to resume from is a fresh run, not an error -- that is
    # what lets one command line serve the first launch and every relaunch.
    assert resolve_resume_path("auto", tmp_path) is None

    checkpoint = tmp_path / "resume_0001.pth"
    atomic_save({COMPLETE_KEY: True}, checkpoint)
    assert resolve_resume_path("auto", tmp_path) == checkpoint

    # An explicit path that is absent must fail loudly. Falling back to a fresh
    # run there is how a week of compute gets silently thrown away.
    with pytest.raises(FileNotFoundError):
        resolve_resume_path(str(tmp_path / "nope.pth"), tmp_path)


def test_restore_accepts_a_checkpoint_written_by_a_wrapped_module(tmp_path):
    """``module.``-prefixed keys load into a bare module.

    This repository keeps the wrappers off the module tree so its own checkpoints
    never carry the prefix, but a checkpoint from elsewhere may -- and with
    ``strict=False`` such a load matches zero keys and reports success.
    """
    model = _model()
    prefixed = {f"module.{key}": value for key, value in model.state_dict().items()}
    path = atomic_save({"model": prefixed, COMPLETE_KEY: True}, tmp_path / "prefixed.pth")

    target = _model(seed=5)
    report = restore_components(load_checkpoint_payload(path), {"model": target}, strict=True)
    assert report["loaded"] == ["model"]
    for name, parameter in target.named_parameters():
        torch.testing.assert_close(parameter, dict(model.named_parameters())[name])


def test_checkpoint_manager_prunes_by_prefix_and_spares_named_artifacts(tmp_path):
    """Rolling checkpoints are bounded; ``best`` and ``final`` are not touched."""
    manager = CheckpointManager(tmp_path, keep_last_n=2)
    manager.save("best_model.pth", {"kind": "best"})
    for step in range(4):
        manager.save(f"resume_{step:04d}.pth", {"step": step}, rolling_prefix="resume_")

    remaining = sorted(path.name for path in tmp_path.glob("*.pth"))
    assert remaining == ["best_model.pth", "resume_0002.pth", "resume_0003.pth"]


def test_a_stop_on_the_last_micro_batch_is_recorded_as_the_next_epoch():
    """Otherwise the resume re-enters an epoch with nothing left in it.

    ``is_last_batch`` forces an optimizer step, so the final micro-batch of every
    epoch is always a checkpoint opportunity -- this is a state a real run reaches
    routinely, not an exotic one. Recorded as "epoch k, fully consumed", the
    resume skips every batch of epoch k, finds zero to process, and raises where
    it should simply have moved on.
    """
    from src.trainers.contrastive_pretrain import resume_position

    middle = resume_position(
        epoch=2, global_step=40, batch_idx=17, is_last_batch=False, epochs=10
    )
    assert (middle.epoch, middle.micro_step, middle.completed) == (2, 18, False)

    boundary = resume_position(
        epoch=2, global_step=40, batch_idx=17, is_last_batch=True, epochs=10
    )
    assert (boundary.epoch, boundary.micro_step, boundary.completed) == (3, 0, False)

    # The final batch of the final epoch is the run being finished.
    final = resume_position(
        epoch=9, global_step=99, batch_idx=17, is_last_batch=True, epochs=10
    )
    assert (final.epoch, final.micro_step, final.completed) == (10, 0, True)


def test_a_disabled_manager_writes_nothing_but_still_reports_its_path(tmp_path):
    """Non-main ranks need no ``if is_main`` at the call site.

    Every rank holds identical parameters, so a second writer is at best wasted
    bandwidth and at worst two processes interleaving into one file.
    """
    manager = CheckpointManager(tmp_path / "unwritten", keep_last_n=1, enabled=False)
    path = manager.save("model.pth", {"value": 1})
    assert path.endswith("model.pth")
    assert not Path(path).exists()
    assert not (tmp_path / "unwritten").exists()
