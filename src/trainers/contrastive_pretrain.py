"""Stage 1: DINO-style self-supervised pretraining (paper Section 4, Table 1).

DINO (Caron et al., 2021) with a SwinV2 trunk, plus the two DINOv2 components
that do not require patch tokens -- KoLeo and Sinkhorn-Knopp centering. It is
**not** DINOv2: there is no iBOT patch objective and no untied heads. See
``src/losses/dino.py``.

    python main.py pretrain
    python main.py pretrain --gpus 2                     # DDP over 2 GPUs
    python main.py pretrain data.batch_size=2 experiment.training.epochs=1 \
        experiment.training.max_batches=2

The regime::

    ImageNet-1k -> SwinV2-Tiny -> DINO self-distillation (trunk unfrozen)
        -> domain-adapted encoder -> stage 2

Per training step:

1. Build ``2 + local_crops_number`` augmented views of each image.
2. The **teacher** sees only the 2 global crops; the **student** sees all views.
3. The DINO loss scores every cross-view pair (Eq. 1).
4. Backprop into the student only; clip gradients at 3.0 (Table 1); cancel the
   projection head's final-layer gradients during the first epoch.
5. Advance the teacher by EMA at a cosine-scheduled momentum (0.996 -> 1.0).

The **physical** batch is what the collapse guards see. Sinkhorn's assignment and
KoLeo's nearest-neighbour distances are computed inside each micro-batch, so
accumulation averages *gradients* and buys those statistics nothing: 16x4 and
32x1 are the same effective batch and not the same run. The configuration
therefore prefers physical batch to accumulation
(``data.batch_size=32``, ``gradient_accumulation_steps=1``) and derives the
learning rate from the effective batch it ends up with -- see
:func:`resolve_learning_rate`, which applies the linear scaling rule rather than
quoting DINO's rate at a batch 8x smaller than the one it belongs to.

The run ends by writing three files: ``dino_pretrained_final.pth`` (full state),
``dino_pretrained_backbone.pth`` (a bare ``student_backbone`` state dict) and
``summary.json``. The second is the **only** weight handoff to stage 2; the third
is the machine-readable record of what produced it -- resolved augmentation, view
geometry, effective batch, LR provenance, centering/``K``/``lambda_koleo``, the
**corpus fingerprint**, the final ``KL(q||p)`` and teacher entropy with its
bounds, wall clock and peak VRAM. ``experiment.training.save_epochs``
additionally keeps a permanent encoder at each listed epoch
(``dino_backbone_epoch_0025.pth`` and friends), which is what makes "was 100
epochs necessary?" a question stage 2 can answer instead of a claim.

Reading the training log
========================

**The loss is not a learning curve.** ``loss`` is a cross entropy, so
``loss = H(teacher) + KL(teacher||student)``, and under Sinkhorn centering
``H(teacher)`` is fixed by the normaliser, ``K``, ``B_teacher`` and the
temperature schedule. Measured on the shipped 100-epoch run: 80 % of the total
loss drop was ``H`` falling, the final loss was 94.8 % irreducible target
entropy, and ``KL`` -- the only learnable part -- was still improving at epoch 93
while the raw curve had been flat since epoch 20. Every step and epoch record
therefore carries ``teacher_student_kl`` and ``teacher_entropy_cross_view``
alongside ``loss``, and the loss figure plots the KL first.

**``loop_blocked_fraction`` is not a GPU-idle fraction.** It is the share of wall
clock the training loop spent inside the dataloader's ``__next__``. Nothing
synchronises inside the step, so queued GPU work drains during that window: the
metric *upper-bounds* idleness, and turning ``1 - loop_blocked_fraction`` into a
GPU-busy time measures CPU enqueue time instead. Set
``experiment.training.measure_gpu_busy=true`` for a genuinely synchronised
measurement (``gpu_busy_fraction``), at the cost of one stall per logging
interval. ``data_wait_fraction`` remains as an alias of the same number so
figures written against the shipped run keep working.

**The corpus is recorded and checked.** The stage-1 -> stage-2 handoff is a bare
``state_dict``, and until the audit nothing recorded what it was self-distilled
on -- the shipped encoder was trained on 8,173 crops while everything downstream
used 9,357. A SHA-256 over the sorted relative path list, the sample and class
counts, the source-photograph count and the per-class histogram go into
``events.jsonl`` and ``summary.json``, and ``pretrain_eval`` compares its own
corpus against them.

**KoLeo is applied per global view.** Applied across the two views concatenated
-- the pre-audit behaviour, still reachable at
``model.loss.koleo_scope=all_views`` -- it makes the two views of one image each
other's nearest neighbour and pushes them apart. See ``src/losses/dino.py``.

How a step is executed
======================

The arithmetic above is the paper's. What follows is how it is scheduled, and it
is the difference between a run that finishes and one that does not: at 6 views
per image and a 256 px window, a naive implementation of this loop spends most
of its wall clock not computing.

**One backbone call, not six.** All ``6B`` student views go through
``forward_student_views`` as a single stacked tensor, and the teacher's ``2B``
globals as another. A SwinV2 block at one micro-batch does not fill a GPU -- its kernels
are launch-bound -- and the per-view loop paid that overhead six times for the
same total arithmetic. Peak activation memory is unchanged, because the loop
already kept all six autograd graphs alive until backward.

**Local crops cross the PCIe bus at their own size.** ``resize_local_to_global``
upsamples 101 px crops to 256 px so SwinV2's fixed windows accept them. Doing
that on the dataloader worker means four of every six views are collated,
pinned and copied at **6.4x their information content**; doing it with one
``F.interpolate`` after the copy is the same function (bicubic, applied after
normalisation, exactly as the CPU pipeline ordered it) for a fraction of the
CPU and bus cost. Views also arrive as ``uint8`` and are normalised here, which
is another 4x off all three.

**The un-augmented view is not built.** ``_originals`` was constructed for every
sample of every epoch -- a resize, a float conversion, a collate and 768 KB of
transfer each -- and then dropped on the floor by this loop.

**Nothing synchronises inside the step.** Every ``float(tensor)`` blocks the CPU
until the queued backward drains, so the next batch cannot start being enqueued.
The loss accumulator stays a device tensor, the loss diagnostics stay device
tensors (``CustomDINOLoss.metrics_enabled``), the KL decomposition is accumulated
as device tensors too, and the whole lot is converted once per logging interval.
``loop_blocked_fraction`` in the step log is the direct measurement of whether any
of this is working: it is the share of wall clock the loop spent blocked in the
dataloader, and if it is high the bottleneck is the CPU pipeline, not the GPU. It
is *not* a GPU-idle fraction -- see "Reading the training log" above.

**The EMA is two kernels, not ~880.** See
:class:`~src.utils.training.ema.TeacherEmaUpdater`.

**Window attention is SDPA, by algebraic rewrite.** timm runs SwinV2's cosine
attention eagerly, and per block autograd saves two full ``[B*nW, heads, N, N]``
matrices for backward -- several GB of bf16 at 192 student views, the step's largest
memory and bandwidth consumer and the binding constraint on the physical batch.
``experiment.training.sdpa_attention`` rebinds each attention module to an
algebraically identical ``F.scaled_dot_product_attention`` form, parity-checked
per module at conversion time. See ``src/models/backbones/sdpa_attention.py``.

Precision and compilation are configured under ``experiment.training`` -- see
``conf/experiment/pretrain_swinv2_dino.yaml``. ``amp: auto`` selects bf16 on
Ampere and later, fp16 with a ``GradScaler`` on older CUDA cards (a T4 is one),
and off on CPU/MPS. The Sinkhorn normaliser, the prototype log-softmax and the
KoLeo distances are pinned to fp32 inside the autocast region by
``src/losses/dino.py``; those three are the parts of this objective that fp16
cannot hold.

Running on more than one GPU
============================

    torchrun --standalone --nproc_per_node=2 -m src.trainers.contrastive_pretrain \
        experiment=pretrain_swinv2_dino

or ``python main.py pretrain --gpus 2``, which is the same thing with the run id
pinned so every rank shares one output directory.

**The effective batch is held fixed, not multiplied.** ``data.batch_size`` is the
*per-rank* micro-batch, so launching on two GPUs would silently double the global
batch -- and with it the LR/momentum regime the schedules are tuned to.
``experiment.training.effective_batch_size`` is therefore the authority:
:func:`resolve_accumulation` derives ``gradient_accumulation_steps`` from it and
the world size, and refuses to start on a combination that does not divide
exactly. One GPU at ``32 x 1`` and two at ``16 x 1`` are the same 32 images per
optimizer step.

**Holding the per-rank micro-batch fixed is also what preserves the objective.**
Sinkhorn centering and KoLeo are batch statistics, not per-sample means, and both
are already computed per *micro-batch* under accumulation. A rank that sees the
same number of images per micro-batch therefore computes the identical function
to a single-GPU run's micro-batch.

The corollary is worth stating, because it cuts against the usual reading of
"same effective batch, same run": splitting the configured batch of 32 across
two ranks as ``16 x 2`` keeps the *gradient* identical to ``32 x 1`` and halves
what Sinkhorn and KoLeo estimate from. If a second GPU is available, the
statistically better use of it is ``data.batch_size=32`` on both ranks with
``effective_batch_size=64`` -- a different, larger run -- or
``model.loss.distributed_sinkhorn=true``, which normalises the assignment over
the concatenated global batch and restores the 32-image estimate exactly (a
different objective from the single-GPU one, hence opt-in).

What genuinely must be synchronised is
synchronised: gradients (by DDP, at accumulation boundaries only -- see
:meth:`~src.models.backbones.swinv2_dino.DINO.no_sync`), the EMA centering buffer
under ``centering="ema"``, and the teacher's weights at construction. Making the
Sinkhorn assignment doubly stochastic over the *global* batch instead is
available and exact (``model.loss.distributed_sinkhorn=true``), but it is a
different objective from the single-GPU one and so is opt-in.

Surviving an interrupted session
================================

``experiment.training.resume=auto`` continues from the newest valid checkpoint in
the save directory, and starts fresh when there is none -- so the same command
line works for the first launch and every relaunch after a Kaggle or vast.ai
session ends. The resume checkpoint carries the teacher, the optimizer moments,
the scheduler, the ``GradScaler``, the epoch, the global step, the micro-batch
within the epoch and every rank's RNG state, which is what makes the continuation
a continuation rather than a warm restart.

Two triggers write it: ``resume_every_minutes`` (wall clock, the one that matters
on a session-limited platform, where an epoch interval can easily be longer than
the session) and a SIGTERM/SIGINT handler that finishes the current micro-batch
and then saves. ``max_runtime_minutes`` stops the run cleanly before a hard
session limit rather than relying on being signalled at all.
"""

from __future__ import annotations

import math
import os
import random
import shutil
import sys
import time
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.datasets.dataset import (
    MultiCropBatch,
    corpus_fingerprint,
    get_pretrain_dataloader,
    raw_photograph_coverage,
)
from src.datasets.transforms import get_dino_transforms
from src.losses.dino import CustomDINOLoss
from src.models.backbones.swinv2_dino import DINO, build_dino
from src.utils.training import (
    CheckpointManager,
    DistributedContext,
    ExperimentTracker,
    InterruptGuard,
    PeriodicSaver,
    ResumeState,
    StageOneBudget,
    TeacherEmaUpdater,
    TrainingProgress,
    all_reduce_max,
    all_reduce_mean,
    autocast_context,
    barrier,
    broadcast_object,
    build_checkpoint_payload,
    build_grad_scaler,
    collect_device_stats,
    collect_rng_states,
    configure_backend,
    log_attention_maps,
    load_checkpoint_payload,
    measure_gflops_per_view,
    resolve_amp,
    resolve_compile,
    resolve_num_workers,
    resolve_resume_path,
    restore_components,
    restore_rng_states,
    setup_distributed,
    setup_experiment_logger,
    shutdown_distributed,
    snapshot_run_configuration,
    to_cpu_state_dict,
)
from src.utils.evaluation import RunSummary
from src.utils.visualization import plot_loss_curves

#: Rolling resume checkpoints. Two, not one: on a preempted instance the newest
#: file is the one a kill is most likely to have interrupted, and the previous
#: one is what ``find_latest_checkpoint`` walks back to.
RESUME_PREFIX = "dino_resume_step"
RESUME_KEEP_LAST = 2


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch.

    cuDNN's determinism and autotuning flags are **not** set here -- they belong
    to :func:`~src.utils.training.device.configure_backend`, which the caller
    invokes with the run's ``deterministic`` setting. Stage 1 produces a single
    checkpoint rather than a row in a comparison table, so it defaults to
    autotuning; stage 2, where variants must be comparable, does not.

    Every rank seeds **identically**, and that is deliberate: the student's
    initialisation must match across ranks or DDP's construction-time broadcast
    would be silently overwriting half the job's idea of the model. Divergence
    where it is wanted -- the sample order, the augmentation stream -- comes from
    the ``DistributedSampler`` and from a rank-offset dataloader generator, not
    from the weights.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_value(start: float, end: float, step: int, total: int) -> float:
    """Cosine interpolation from ``start`` to ``end`` over ``total`` steps.

    DINO schedules both the teacher momentum (0.996 -> 1.0) and the weight decay
    (0.04 -> 0.4) this way. The point of the momentum schedule is a fast-adapting
    teacher early and a stable target late; a constant value gives neither end of
    that trade-off, which is what the submitted configuration did.
    """
    if total <= 1 or start == end:
        return float(end if total <= 1 else start)
    progress = min(max(step, 0), total) / total
    return float(end + (start - end) * (1.0 + math.cos(math.pi * progress)) / 2.0)


def resolve_accumulation(
    cfg: DictConfig,
    context: DistributedContext,
    logger,
) -> tuple[int, int]:
    """Return ``(accumulation_steps, effective_batch)``, honouring the world size.

    ``experiment.training.effective_batch_size`` is the authority when it is set:
    the accumulation count is *derived* from it, so the number of images behind
    one optimizer step is identical on 1 GPU and on 8. That is not tidiness. The
    learning rate, the teacher momentum schedule and every collapse guard in DINO
    are tuned to a batch, and a run launched on two GPUs with the accumulation
    left alone trains at twice the intended batch while every log line looks
    normal.

    A combination that does not divide exactly is refused rather than rounded.
    Rounding would silently change the very number this function exists to pin,
    and the fix -- change the micro-batch, or the target -- is one the operator
    has to choose.

    Falls back to the literal ``gradient_accumulation_steps`` when
    ``effective_batch_size`` is null, warning if that makes the effective batch
    world-size dependent.
    """
    micro_batch = int(cfg.data.batch_size)
    configured = max(
        int(OmegaConf.select(cfg, "experiment.training.gradient_accumulation_steps", default=1)), 1
    )
    target = OmegaConf.select(cfg, "experiment.training.effective_batch_size", default=None)

    if target is None:
        effective = micro_batch * configured * context.world_size
        if context.world_size > 1:
            logger.warning(
                "effective_batch_size is null, so the effective batch is %s x %s x %s ranks = "
                "%s images -- %sx the single-GPU value. Set "
                "experiment.training.effective_batch_size to pin it instead.",
                micro_batch, configured, context.world_size, effective, context.world_size,
            )
        return configured, effective

    target = int(target)
    per_step = micro_batch * context.world_size
    if target % per_step != 0:
        raise ValueError(
            f"effective_batch_size={target} is not divisible by "
            f"data.batch_size={micro_batch} x world_size={context.world_size} = {per_step}.\n"
            "One optimizer step must consume a whole number of micro-batches on every rank. "
            f"Use a multiple of {per_step}, or change data.batch_size."
        )
    accumulation = max(target // per_step, 1)
    logger.info(
        "Effective batch %s = %s per rank x %s ranks x %s accumulation steps.",
        target, micro_batch, context.world_size, accumulation,
    )
    if accumulation != configured:
        logger.info(
            "gradient_accumulation_steps: %s configured -> %s derived from "
            "effective_batch_size at this world size.",
            configured, accumulation,
        )
    return accumulation, target


def resume_position(
    epoch: int,
    global_step: int,
    batch_idx: int,
    is_last_batch: bool,
    epochs: int,
) -> TrainingProgress:
    """Where a run stopping *here* should be recorded as having stopped.

    The one non-obvious rule: a stop that lands on an epoch's **last**
    micro-batch is recorded as the *start of the next epoch*, not as "this epoch,
    fully consumed". Both describe the same position, but only the first one
    survives a resume — the second re-enters an epoch with nothing left in it,
    and the loop then finds zero batches where it expected the tail of one.

    That state is not exotic: ``is_last_batch`` forces an optimizer step, so the
    final micro-batch of every epoch is always a checkpoint opportunity.
    """
    if is_last_batch:
        return TrainingProgress(
            epoch=epoch + 1,
            global_step=global_step,
            micro_step=0,
            completed=epoch + 1 >= epochs,
        )
    return TrainingProgress(
        epoch=epoch, global_step=global_step, micro_step=batch_idx + 1, completed=False
    )


#: Optimizer-group key marking the group the weight-decay schedule may move.
#: The no-decay group carries ``False`` and the epoch-end ramp skips it; without
#: the flag the ramp would walk every group and undo the split the moment the
#: first epoch ended.
WEIGHT_DECAY_FLAG = "apply_weight_decay"


def resolve_learning_rate(
    cfg: DictConfig,
    effective_batch: int,
    logger,
) -> tuple[float, dict[str, Any]]:
    """Return ``(lr, provenance)`` for the batch this run actually assembled.

    Section 6.1 states 0.0005. That is DINO's rate **at its reference batch of
    256**, and quoting it next to a batch of 16 or 32 is a 16x or 8x
    overstatement of the intended step size -- the linear scaling rule
    (Goyal et al., 2017) is what connects the two, and nothing in the previous
    configuration applied it. Worse, the literal value was immune to every knob
    that changes the batch: raising ``data.batch_size``, changing the
    accumulation, or launching on a second GPU all moved the effective batch and
    left the rate alone.

    So the rate is derived here from ``effective_batch`` -- the number
    :func:`resolve_accumulation` just finished pinning, which already accounts
    for the micro-batch, the world size and the accumulation count:

        lr = lr_base * effective_batch / lr_reference_batch_size

    ``experiment.training.learning_rate`` remains an override: set a float and it
    is used verbatim. ``lr_scaling: "none"`` uses ``lr_base`` at any batch. Both
    paths return the same provenance dict, which is logged and written into the
    event stream, so a run's own artifacts say which rule produced its rate.
    """
    training = "experiment.training"
    configured = OmegaConf.select(cfg, f"{training}.learning_rate", default=None)
    base = float(OmegaConf.select(cfg, f"{training}.lr_base", default=5e-4))
    reference = int(OmegaConf.select(cfg, f"{training}.lr_reference_batch_size", default=256) or 256)
    mode = str(OmegaConf.select(cfg, f"{training}.lr_scaling", default="linear")).lower()

    if configured is not None:
        learning_rate = float(configured)
        rule = "configured"
    elif mode == "linear":
        if reference <= 0:
            raise ValueError(f"lr_reference_batch_size must be positive, got {reference}")
        learning_rate = base * effective_batch / reference
        rule = "linear"
    elif mode == "none":
        learning_rate = base
        rule = "lr_base"
    else:
        raise ValueError(f"lr_scaling must be 'linear' or 'none', got {mode!r}")

    provenance = {
        "learning_rate": learning_rate,
        "rule": rule,
        "lr_base": base,
        "lr_reference_batch_size": reference,
        "effective_batch_size": int(effective_batch),
    }
    if rule == "linear":
        logger.info(
            "Learning rate %.6g = lr_base %.6g x effective_batch %s / reference %s "
            "(linear scaling; set experiment.training.learning_rate to override).",
            learning_rate, base, effective_batch, reference,
        )
    else:
        logger.info("Learning rate %.6g (%s; no batch scaling applied).", learning_rate, rule)
    return learning_rate, provenance


def _declared_no_decay(module: torch.nn.Module) -> set[str]:
    """Names the module itself says must not be decayed, or an empty set.

    timm models expose ``no_weight_decay()``, and on SwinV2 it names every
    ``cpb_mlp`` -- the small MLP that generates the continuous relative-position
    bias. Those are ordinary 2-D ``Linear`` weights, so no shape rule finds them,
    and the SwinV2 authors excluded them deliberately: the MLP outputs a *bias*,
    and decaying its weights pulls the learned position structure toward a
    constant. Asking the model rather than guessing is both more correct and
    less to maintain.
    """
    declare = getattr(module, "no_weight_decay", None)
    if not callable(declare):
        return set()
    try:
        return {str(name) for name in declare()}
    except Exception:  # pragma: no cover - a model with a broken declaration
        return set()


def _is_no_decay(name: str, parameter: torch.nn.Parameter, declared: set[str]) -> bool:
    """Whether ``name`` belongs in the decay-free group.

    Three rules, in order of authority:

    1. **The model's own declaration** (:func:`_declared_no_decay`).
    2. **Biases**, by name.
    3. **Anything with at most one non-singleton dimension.** This is the shape
       rule DINO's ``get_params_groups`` writes as ``len(param.shape) == 1``,
       widened for a reason: SwinV2's per-head ``logit_scale`` has shape
       ``[heads, 1, 1]`` and is 3-D, so the literal 1-D test decays it. It is the
       learned temperature of cosine attention, and decaying it toward zero
       drives ``exp(logit_scale)`` toward 1 -- flattening every attention map in
       the trunk, gradually, with nothing in the loss to say so. Counting
       non-singleton axes classifies it as what it is: a per-head vector.
    """
    if name in declared or any(name.startswith(f"{prefix}.") for prefix in declared):
        return True
    if name.endswith(".bias"):
        return True
    return sum(1 for size in parameter.shape if size > 1) <= 1


def build_param_groups(
    *modules: torch.nn.Module,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """Split trainable parameters into decayed and non-decayed groups.

    Biases, normalisation gains and every other vector-shaped parameter go into
    the second group at ``weight_decay = 0``, as in DINO's own
    ``get_params_groups``; see :func:`_is_no_decay` for the exact rules and why
    the shape test is not the literal 1-D one.

    Why this matters more here than usual: the weight decay is **scheduled to
    0.4**, ten times its starting value. Decaying a LayerNorm gain toward zero
    is not regularisation, it is a gradual deletion of the layer, and at 0.4 the
    normalisation the trunk depends on erodes over the run while the loss curve
    stays entirely plausible.

    Parameters are de-duplicated by identity, so a module passed twice does not
    receive two updates per step. Returns **two** groups always -- an empty one
    is harmless and keeps ``param_groups[0]`` meaning "the decayed group"
    regardless of the model.
    """
    decayed: dict[int, torch.nn.Parameter] = {}
    plain: dict[int, torch.nn.Parameter] = {}
    for module in modules:
        if module is None:
            continue
        declared = _declared_no_decay(module)
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            bucket = plain if _is_no_decay(name, parameter, declared) else decayed
            bucket[id(parameter)] = parameter

    return [
        {
            "params": list(decayed.values()),
            "weight_decay": float(weight_decay),
            WEIGHT_DECAY_FLAG: True,
        },
        {
            "params": list(plain.values()),
            "weight_decay": 0.0,
            WEIGHT_DECAY_FLAG: False,
        },
    ]


def apply_weight_decay(optimizer: optim.Optimizer, value: float) -> None:
    """Set the scheduled weight decay on the decayed group only."""
    for group in optimizer.param_groups:
        if group.get(WEIGHT_DECAY_FLAG, True):
            group["weight_decay"] = float(value)


def build_optimizer(
    param_groups,
    cfg: DictConfig,
    device: torch.device,
    logger,
    learning_rate: float | None = None,
) -> optim.Optimizer:
    """AdamW over the student's parameter groups (Section 6.1), fused where available.

    The fused implementation runs the whole update as a handful of multi-tensor
    kernels instead of ~440 small ones. That matters more here than the FLOP
    count suggests: the optimizer fires once per accumulation window, and the
    update's cost is almost entirely kernel launches on tensors far too small to
    hide them.

    Falls back to the reference implementation if the build does not support it,
    which is a performance difference and never a numerical one.

    ``param_groups`` is what :func:`build_param_groups` returns, so the per-group
    ``weight_decay`` already set there wins over the value passed here -- which
    is exactly the point of the split.
    """
    kwargs = {
        "lr": float(
            learning_rate
            if learning_rate is not None
            else OmegaConf.select(cfg, "experiment.training.learning_rate", default=5e-4)
        ),
        "weight_decay": float(cfg.experiment.training.weight_decay),
    }
    if device.type == "cuda" and bool(
        OmegaConf.select(cfg, "experiment.training.fused_optimizer", default=True)
    ):
        try:
            return optim.AdamW(param_groups, fused=True, **kwargs)
        except (RuntimeError, TypeError, ValueError) as exc:
            logger.warning("Fused AdamW unavailable (%s); using the default implementation.", exc)
    return optim.AdamW(param_groups, **kwargs)


def resolve_warmup_epochs(cfg: DictConfig) -> int:
    """Warmup length in epochs, clamped so it cannot consume the whole run.

    A smoke run at ``epochs=1`` composes the same config as a 100-epoch run, and
    an unclamped 10-epoch warmup there would train the entire job at a tenth of
    the rate while reporting a warmup that never finished.
    """
    epochs = int(cfg.experiment.training.epochs)
    requested = int(OmegaConf.select(cfg, "experiment.training.warmup_epochs", default=0) or 0)
    return max(min(requested, max(epochs - 1, 0)), 0)


def build_scheduler(optimizer: optim.Optimizer, cfg: DictConfig, logger=None):
    """Linear warmup then cosine decay, as **one** scheduler object, or ``None``.

    Section 6.1 specifies cosine decay; DINO additionally warms the rate up over
    the first 10 epochs, which the submitted configuration omitted entirely --
    so step 0 hit a freshly-initialised prototype layer at the full rate, which
    is the moment a self-distillation run is least able to absorb it.

    Implemented as ``SequentialLR([LinearLR, CosineAnnealingLR])`` rather than as
    a warmup branch inside the training loop. One object owns the whole
    trajectory, which matters for three reasons: the loop keeps its single
    unconditional ``scheduler.step()``; ``state_dict()`` round-trips both phases
    *and the milestone*, so a resume inside warmup resumes inside warmup; and
    there is no second scheduler holding a reference to the same optimizer,
    which is the usual way a hand-rolled warmup ends up multiplying the rate by
    two schedules at once.

    The cosine spans ``epochs - warmup_epochs`` by default, so the decay reaches
    ``eta_min`` on the final epoch. Using the full ``epochs`` would leave the
    rate at a few percent of peak at the end and make the warmup silently
    truncate the decay; an explicit ``scheduler.t_max`` overrides it.
    """
    name = OmegaConf.select(cfg, "experiment.training.scheduler.name", default=None)
    if name is None:
        return None
    if name != "cosine":
        raise ValueError(f"Unsupported scheduler: {name}")

    epochs = int(cfg.experiment.training.epochs)
    warmup_epochs = resolve_warmup_epochs(cfg)
    eta_min = float(OmegaConf.select(cfg, "experiment.training.scheduler.eta_min", default=0.0))
    t_max = OmegaConf.select(cfg, "experiment.training.scheduler.t_max", default=None)
    cosine_epochs = int(t_max) if t_max else max(epochs - warmup_epochs, 1)

    # Read the target rate BEFORE constructing anything: every LRScheduler
    # applies its own factor to `param_groups` in ``__init__``, so a peak read
    # after ``LinearLR`` exists is already the warmup's first step, not the peak.
    peak = float(optimizer.param_groups[0]["lr"])

    cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=eta_min)
    if warmup_epochs <= 0:
        if logger is not None:
            logger.info(
                "LR schedule: cosine %.4g -> %.4g over %s epochs (no warmup).",
                peak, eta_min, cosine_epochs,
            )
        return cosine

    # start_factor = 1/warmup_epochs rather than 0: LinearLR rejects 0, and a
    # first epoch at 1/10th of the target is the same ramp DINO's per-iteration
    # warmup describes, sampled once per epoch.
    warmup = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0 / warmup_epochs,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    if logger is not None:
        logger.info(
            "LR schedule: linear warmup %.4g -> %.4g over %s epochs (peak reached at the start "
            "of epoch %s), then cosine over %s epochs to eta_min=%.4g.",
            peak / warmup_epochs, peak, warmup_epochs, warmup_epochs + 1, cosine_epochs, eta_min,
        )
    return optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )


class _StudentViewPass(nn.Module):
    """One student view, end to end, for the FLOP probe and nothing else.

    ``FlopCounterMode`` intercepts dispatch under a module call, and the quantity
    the cost table wants is per *view* through the path a training step actually
    runs -- backbone, pooling, projection head. Measuring the backbone alone
    would omit the head, which the student evaluates on all six views, not once
    per image.
    """

    def __init__(self, model: DINO):
        super().__init__()
        self.model = model

    def forward(self, views: torch.Tensor) -> torch.Tensor:
        return self.model.forward_student_views(views)


# --------------------------------------------------------------- view assembly


class ViewBatcher:
    """Turn a :class:`~src.datasets.dataset.MultiCropBatch` into model inputs.

    Owns the two pieces of per-view work that were moved off the dataloader:
    normalisation of ``uint8`` views, and the local-crop upsample. Both are
    applied here in the same order the CPU pipeline used --
    ``/255 -> (x - mean) / std -> bicubic resize`` -- so the tensors the backbone
    sees are the same function of the augmented crop either way.

    The order matters and is not interchangeable. Normalisation is affine and
    bicubic interpolation is a partition-of-unity kernel, so the two commute in
    exact arithmetic; but resizing *first* would mean resizing ``uint8``, which
    clamps the bicubic overshoot into [0, 255] and quantises it. That is a
    different transform, and it is why ``output_uint8`` forces
    ``defer_local_upsample`` in ``src/datasets/transforms.py``.

    ``antialias=True`` on the interpolate call is equally load-bearing, and it is
    the trap in this refactor. ``torchvision.transforms.Resize`` defaults to
    ``antialias=True``, and -- contrary to the usual intuition that antialiasing
    only matters when *down*sampling -- torch's antialiased bicubic kernel is not
    the same kernel as the plain one on the way up either. Dropping the flag
    reproduces the CPU pipeline to within ~0.2 in normalised units, which is far
    too small to look like a bug and easily large enough to change what the model
    learns from a 101 px crop. With the flag, the two paths agree bitwise.

    Args:
        image_size: Resolution the backbone requires (256 for
            ``swinv2_*_window16_256``).
        mean / std: Channel statistics, held as ``[1, 3, 1, 1]`` device tensors
            so normalisation is two fused broadcasts rather than a per-channel
            loop.
        device: Where the model lives.
    """

    def __init__(
        self,
        image_size: int,
        mean: tuple[float, ...],
        std: tuple[float, ...],
        device: torch.device,
    ):
        self.image_size = int(image_size)
        self.device = device
        self.mean = torch.tensor(mean, device=device, dtype=torch.float32).view(1, -1, 1, 1)
        self.std = torch.tensor(std, device=device, dtype=torch.float32).view(1, -1, 1, 1)

    def _to_device(self, views: torch.Tensor) -> torch.Tensor:
        """``[V, B, C, H, W]`` -> ``[V * B, C, H, W]`` float, on device.

        ``flatten(0, 1)`` on the collated tensor is a view, not a copy, and it
        preserves the **view-major** block order the loss chunks on.
        """
        views = views.to(self.device, non_blocking=True)
        views = views.flatten(0, 1)
        if views.dtype == torch.uint8:
            views = views.float().div_(255.0).sub_(self.mean).div_(self.std)
        return views

    def __call__(self, batch: MultiCropBatch) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return ``(student_views, teacher_views, batch_size)``.

        ``teacher_views`` is the ``[2B, ...]`` global block and is the same
        storage as the first ``2B`` rows of ``student_views``, so the globals are
        normalised and transferred once.
        """
        batch_size = batch.global_views.shape[1]
        globals_ = self._to_device(batch.global_views)

        if batch.local_views is None:
            return globals_, globals_, batch_size

        locals_ = self._to_device(batch.local_views)
        if locals_.shape[-1] != self.image_size or locals_.shape[-2] != self.image_size:
            # `antialias=True` is what makes this identical to the
            # `T.Resize(..., BICUBIC)` it replaces -- see the class docstring.
            locals_ = F.interpolate(
                locals_,
                size=(self.image_size, self.image_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        student_views = torch.cat([globals_, locals_], dim=0)
        return student_views, globals_, batch_size


# ------------------------------------------------------------------ checkpoints


class GpuBusyMeter:
    """Optional, genuinely synchronised measurement of GPU-busy time per step.

    ``loop_blocked_fraction`` is what the loop spends waiting for data, and it
    **upper-bounds** GPU idleness rather than measuring it: nothing synchronises
    inside the step, so the queued backward drains while the loop blocks. Turning
    ``1 - loop_blocked_fraction`` into a GPU-busy time -- which is what produced
    the shipped run's "GPU-busy 1.12 h of 13.34 h" -- measures CPU *enqueue*
    time instead.

    This measures the real thing with a pair of ``torch.cuda.Event``s around each
    micro-batch's compute. The events are queued asynchronously and **drained in
    batches**, at logging steps and at the epoch boundary, so the stall is paid
    once per logging interval rather than once per micro-batch. It is still a
    stall, which is why the whole thing is off unless
    ``experiment.training.measure_gpu_busy=true``.

    A no-op on CPU/MPS, where there is no separate device queue to measure.
    """

    def __init__(self, device: torch.device, enabled: bool):
        self.enabled = bool(enabled) and device.type == "cuda"
        self._pending: list[tuple[Any, Any]] = []
        self.seconds = 0.0

    def start(self):
        """Context manager around one micro-batch's compute."""
        if not self.enabled:
            return nullcontext()
        return self._Region(self)

    def drain(self) -> float:
        """Synchronise on the queued events and fold them into :attr:`seconds`."""
        if not self._pending:
            return self.seconds
        for start, end in self._pending:
            end.synchronize()
            self.seconds += start.elapsed_time(end) / 1000.0
        self._pending.clear()
        return self.seconds

    class _Region:
        def __init__(self, meter: "GpuBusyMeter"):
            self.meter = meter
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)

        def __enter__(self):
            self.start_event.record()
            return self

        def __exit__(self, *exc):
            self.end_event.record()
            self.meter._pending.append((self.start_event, self.end_event))
            return False


def _corpus_fingerprint_of(dataset: Any, cfg: DictConfig, logger) -> dict[str, Any]:
    """Corpus identity for whatever dataset the loader ended up with.

    ``PretrainImageFolderDataset`` knows its own sample list, which is the honest
    answer -- it describes what will actually be read rather than what the
    directory contains. The pickle-batch escape hatch has no per-file paths at
    all, so it reports its length and no digest rather than a digest of nothing.
    """
    method = getattr(dataset, "corpus_fingerprint", None)
    if callable(method):
        return dict(method())
    try:
        return dict(corpus_fingerprint(str(cfg.data.root_path)))
    except OSError as error:  # pragma: no cover - unreadable root
        logger.warning("Could not fingerprint the corpus at %s (%s).", cfg.data.root_path, error)
        return {}


def _check_corpus(cfg: DictConfig, fingerprint: Mapping[str, Any], logger) -> None:
    """Compare the corpus against what the config declares, and act on the verdict.

    ``experiment.training.corpus_check`` is ``"warn"`` (default), ``"error"`` or
    ``"off"``. The check is the class-count one -- stage 2's label indices come
    from *sorted directory names*, so a corpus with a different number of
    sub-variety directories produces an encoder whose downstream indices refer to
    different classes, with nothing anywhere to say so.

    Default ``warn`` rather than ``error`` because stage 1 is label-free and a
    legitimate corpus (a subsample, a smoke tree, a partner-dataset screen) can
    have any class count; ``error`` is what a publication run should set.
    """
    mode = str(OmegaConf.select(cfg, "experiment.training.corpus_check", default="warn") or "off")
    if mode == "off":
        return
    expected = int(OmegaConf.select(cfg, "data.num_sub_varieties", default=0) or 0)
    found = int(fingerprint.get("num_classes", 0))
    if not expected or found == expected:
        return
    message = (
        f"Corpus has {found} class directories but data.num_sub_varieties={expected}. Stage-2 "
        "label indices come from sorted directory names, so an encoder pretrained on a corpus "
        "with a different class set is not the encoder the downstream indices assume. Set "
        "experiment.training.corpus_check=off if this is deliberate."
    )
    if mode == "error":
        raise RuntimeError(message)
    logger.warning(message)


def save_dino_checkpoint(
    model: DINO,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    checkpoint_manager: CheckpointManager,
    filename: str,
    include_optimizer: bool = True,
    include_teacher: bool = True,
    rolling_prefix: str | None = None,
) -> str:
    """Save student weights, and optionally the teacher and optimizer state.

    The lightweight *artifact* writer: what a later run needs in order to read
    the encoder. :func:`save_resume_checkpoint` is the other one, and it saves
    everything needed to *continue*.

    Reads the *uncompiled, unwrapped* modules deliberately: ``DINO`` keeps both
    the compiled callables and the DDP wrapper off the module tree precisely so
    these keys stay free of the ``_orig_mod.`` and ``module.`` prefixes those
    wrappers would otherwise introduce. That state dict is the only handoff to
    stage 2, where ``checkpoint_strict: false`` would turn a prefixed key set
    into zero matches, one log line, and a full run against a random encoder.
    """
    payload = {
        "epoch": epoch,
        "student_backbone": to_cpu_state_dict(model.student_backbone.state_dict()),
        "student_head": to_cpu_state_dict(model.student_head.state_dict()),
    }
    if include_teacher:
        payload["teacher_backbone"] = to_cpu_state_dict(model.teacher_backbone.state_dict())
        payload["teacher_head"] = to_cpu_state_dict(model.teacher_head.state_dict())
    if include_optimizer:
        payload["optimizer"] = optimizer.state_dict()
        payload["scheduler"] = scheduler.state_dict() if scheduler is not None else None
    return checkpoint_manager.save(filename, payload, rolling_prefix=rolling_prefix)


def resume_components(
    model: DINO,
    criterion: CustomDINOLoss,
    optimizer: optim.Optimizer,
    scheduler,
    scaler,
) -> dict[str, Any]:
    """The named state a resume must carry, in one place used by save and load.

    Keeping the mapping in a function rather than writing it out twice is what
    stops the two sides drifting -- a component saved but never restored is a
    resume that looks complete and quietly reinitialises something. The names
    match :func:`save_dino_checkpoint`'s so one payload satisfies both readers.

    The teacher is here because it *is* the training target: resuming without it
    would rebuild it from the current student and throw away the EMA's entire
    history. ``criterion`` is here for the centering buffer under
    ``centering="ema"``, and the ``GradScaler`` because its scale is state --
    restarting it at 65536 means a handful of skipped steps at every resume.
    """
    return {
        "student_backbone": model.student_backbone,
        "student_head": model.student_head,
        "teacher_backbone": model.teacher_backbone,
        "teacher_head": model.teacher_head,
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "scaler": scaler,
    }


def save_resume_checkpoint(
    checkpoint_manager: CheckpointManager,
    filename: str,
    *,
    components: dict[str, Any],
    progress: TrainingProgress,
    context: DistributedContext,
    cfg: DictConfig,
    loader_generator: torch.Generator | None,
    extra: dict[str, Any] | None = None,
    rolling_prefix: str | None = RESUME_PREFIX,
) -> str:
    """Write a fully resumable checkpoint. Collective: every rank must call it.

    The RNG gather is a collective, so this cannot be guarded by ``if is_main``
    at the call site -- the non-main ranks have to reach the gather or rank 0
    blocks in it until the timeout. The *write* is guarded instead, inside
    :class:`~src.utils.training.checkpoint.CheckpointManager`.
    """
    payload = build_checkpoint_payload(
        components=components,
        progress=progress,
        context=context,
        config=cfg,
        rng_states=collect_rng_states(context, loader_generator),
        extra={"epoch": progress.epoch, **(extra or {})},
    )
    return checkpoint_manager.save(filename, payload, rolling_prefix=rolling_prefix)


def write_stage1_summary(
    save_path: Path,
    cfg: DictConfig,
    *,
    criterion: CustomDINOLoss,
    model: DINO,
    transform,
    fingerprint: Mapping[str, Any],
    raw_coverage: Mapping[str, Any],
    parameters: Mapping[str, int],
    budget: StageOneBudget,
    dynamics: Mapping[str, float],
    runtime: Mapping[str, Any],
    artifacts: Mapping[str, str],
    context: DistributedContext,
    logger,
) -> str:
    """Write ``summary.json`` beside the stage-1 checkpoints.

    Stage 2 has written one since the beginning and ``scripts/generate_plots.py``
    reads nothing else; stage 1 wrote checkpoints, ``events.jsonl`` and a table
    printed into a log, so "what recipe produced this encoder?" meant parsing
    3.7 MB of JSONL. A four-arm stage-1 comparison is only tractable when each
    arm leaves exactly one machine-readable file, which is why
    ``scripts/run_stage1_ablations.py`` reads this and nothing else.

    Uses :class:`~src.utils.evaluation.RunSummary`, so the cross-run table
    renders a stage-1 arm and a stage-2 variant without a special case. The three
    fields that carry the stage-1-specific weight:

    * ``loss_flags`` -- every objective-side setting an arm can move, from the
      criterion itself, so a ``koleo_scope=all_views`` control is machine-
      distinguishable from the default rather than distinguished only by its
      directory name.
    * ``split`` -- the **corpus fingerprint**, which is the field that would have
      caught the shipped encoder having been trained on 8,173 of 9,357 crops.
    * ``metrics`` -- the final ``KL(q||p)`` and teacher entropy with its bounds,
      not the raw loss alone; see A4.
    """
    summary = RunSummary(
        name=str(cfg.experiment.name),
        group=str(OmegaConf.select(cfg, "experiment.group", default="stage1_pretraining")),
        run_dir=str(save_path),
        metrics=dict(dynamics),
        efficiency={},
        history={},
        component_flags={
            "stage": "pretrain",
            "backbone": str(cfg.model.backbone.name),
            "feature_stage_published": "final",
            "pretrained_init": (
                "imagenet"
                if bool(OmegaConf.select(cfg, "model.backbone.pretrained", default=False))
                else "random"
            ),
            "drop_path_rate": model.drop_path_rate,
            "aux_stage": model.aux_stage,
            "aux_weight": model.aux_weight if model.aux_stage is not None else None,
            "koleo_space": str(
                OmegaConf.select(cfg, "model.loss.koleo_space", default="bottleneck")
            ),
            **{f"params_{key}": int(value) for key, value in parameters.items()},
        },
        loss_flags=criterion.loss_flags(),
        split={
            "corpus": dict(fingerprint),
            "raw_photograph_coverage": dict(raw_coverage),
            # Stage 1 is TRANSDUCTIVE on this dataset: it self-distils on every
            # crop, including crops of the photographs the evaluation and stage 2
            # hold out. No labels leak and this is the standard in-domain SSL
            # setting, but a paper claiming photograph-disjoint generalisation
            # has to say so, and the direction is conservative -- the setup
            # favours DINO.
            "stage1_transductive": True,
            "transductive_note": (
                "Stage 1 self-distilled on every crop, including the photographs the readout "
                "and stage 2 hold out. Labels never leak; the encoder is not photograph-disjoint "
                "from the test split."
            ),
        },
        fold_metrics={},
        runtime={
            "world_size": context.world_size,
            "effective_batch_size": budget.effective_batch_size,
            "physical_batch_size": budget.physical_batch_size,
            "gradient_accumulation_steps": budget.gradient_accumulation_steps,
            **dict(runtime),
        },
        config={
            "backbone": str(cfg.model.backbone.name),
            "image_size": int(cfg.data.image_size),
            "local_crop_size": int(cfg.data.local_crop_size),
            "num_crops": transform.num_crops,
            "view_sizes": list(transform.view_sizes),
            "view_ids": list(transform.view_ids),
            "global_view_ids": list(transform.global_view_ids),
            "epochs": int(cfg.experiment.training.epochs),
            "seed": int(cfg.seed),
            "augmentation": OmegaConf.to_container(cfg.data.augmentation, resolve=True),
            "budget": budget.as_dict(),
        },
        artifacts=dict(artifacts),
    )
    path = summary.save(save_path)
    logger.info("Wrote %s", path)
    return path


def publish_shared_backbone(cfg: DictConfig, backbone_file: Path, logger) -> Path | None:
    """Copy the pretrained backbone to the shared path all downstream runs read.

    Configured by ``experiment.training.shared_backbone_path``. Returns the
    destination, or ``None`` when publishing is disabled or fails -- a failure
    here must not discard a completed pretraining run, since the per-stage copy
    at ``backbone_file`` is already safely on disk.
    """
    destination = OmegaConf.select(cfg, "experiment.training.shared_backbone_path", default=None)
    if not destination:
        return None

    target = Path(str(destination))
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(backbone_file, target)
    except OSError as exc:
        logger.warning("Could not publish the shared backbone to %s: %s", target, exc)
        return None

    logger.info("Published shared backbone for downstream runs: %s", target)
    return target


# ------------------------------------------------------------------- main loop


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Before anything else: a rank has to know which device it owns before the
    # first tensor is created, and NCCL binds its communicator to whatever device
    # is current when the group comes up.
    context = setup_distributed(
        str(cfg.device),
        timeout_minutes=float(
            OmegaConf.select(cfg, "experiment.training.ddp.timeout_minutes", default=30)
        ),
    )
    device = context.device

    # Every rank must agree on the output directory. Hydra's `${now:...}` is
    # evaluated per process, so two ranks launched in the same second can still
    # land in different directories -- and one that did would write half the
    # run's artifacts somewhere nobody looks. `SEED_RUN_ID` pins it at the
    # launcher; this broadcast makes it hold even when it was not set.
    output_dir = broadcast_object(str(cfg.tracking.output_dir), context)
    logger = setup_experiment_logger(
        log_dir=output_dir,
        name="seed_moe.dino_pretrain",
        level=cfg.tracking.log_level,
        console=bool(cfg.tracking.console) and context.is_main,
        structured_jsonl=cfg.tracking.structured_jsonl,
        rank=context.rank,
        world_size=context.world_size,
    )
    tracker = ExperimentTracker(cfg, logger, enabled=context.is_main)
    training_started = time.perf_counter()
    interrupted = False

    try:
        logger.info("========== DINO pretraining: %s ==========", cfg.experiment.name)
        if context.enabled:
            logger.info(
                "Distributed run | %s ranks, backend=%s, this rank owns %s.",
                context.world_size, context.backend, device,
            )
        seed_everything(int(cfg.seed))
        if context.is_main:
            snapshot_paths = snapshot_run_configuration(cfg, output_dir)
            logger.info(
                "Saved run configuration snapshots.",
                extra={"snapshots": {key: str(value) for key, value in snapshot_paths.items()}},
            )

        backend = configure_backend(
            device,
            allow_tf32=bool(OmegaConf.select(cfg, "experiment.training.allow_tf32", default=True)),
            cudnn_benchmark=bool(
                OmegaConf.select(cfg, "experiment.training.cudnn_benchmark", default=True)
            ),
            deterministic=bool(
                OmegaConf.select(cfg, "experiment.training.deterministic", default=False)
            ),
            matmul_precision=str(
                OmegaConf.select(cfg, "experiment.training.matmul_precision", default="high")
            ),
            logger=logger,
        )
        tracker.log_event("backend", {**backend, "distributed": context.as_dict()})
        logger.info("Selected training device: %s", device)
        tracker.log_metrics(collect_device_stats(device), step=0)

        amp = resolve_amp(
            device, OmegaConf.select(cfg, "experiment.training.amp", default="auto"), logger=logger
        )
        scaler = build_grad_scaler(amp)
        logger.info(
            "Mixed precision: %s (grad scaler: %s). bf16 needs no loss scaling; fp16 does.",
            amp.label,
            "on" if scaler is not None else "off",
        )

        # ------------------------------------------------------------- data
        transform = get_dino_transforms(
            int(cfg.data.image_size),
            int(cfg.data.local_crop_size),
            cfg.data.augmentation,
            # This loop never reads the un-augmented view; building it would cost
            # a resize and a 256x256x3 float per sample per epoch for nothing.
            return_original=False,
        )
        loader_generator = torch.Generator()
        # Offset by rank so the augmentation streams are independent while rank 0
        # stays byte-identical to a single-process run.
        loader_generator.manual_seed(int(cfg.seed) + context.rank)
        num_workers = resolve_num_workers(
            OmegaConf.select(cfg, "data.num_workers", default=8),
            context,
            auto_cap=int(
                OmegaConf.select(cfg, "data.num_workers_auto_cap", default=16) or 16
            ),
            logger=logger,
        )
        dataloader = get_pretrain_dataloader(
            data_dir=cfg.data.root_path,
            transform=transform,
            batch_size=int(cfg.data.batch_size),
            num_workers=num_workers,
            dataset_format=str(cfg.data.dataset_format),
            image_size=int(cfg.data.image_size),
            pin_memory=bool(cfg.data.pin_memory) and device.type == "cuda",
            drop_last=bool(cfg.data.drop_last),
            persistent_workers=bool(
                OmegaConf.select(cfg, "data.persistent_workers", default=True)
            ),
            prefetch_factor=int(OmegaConf.select(cfg, "data.prefetch_factor", default=4)),
            cache_images=bool(OmegaConf.select(cfg, "data.cache_images", default=True)),
            cache_limit_mb=float(OmegaConf.select(cfg, "data.cache_limit_mb", default=4096)),
            generator=loader_generator,
            logger=logger,
            world_size=context.world_size,
            rank=context.rank,
            same_photo_local_views=int(
                OmegaConf.select(cfg, "data.augmentation.same_photo_local_views", default=0) or 0
            ),
            seed=int(cfg.seed),
        )
        if len(dataloader) == 0:
            raise RuntimeError(
                "Dataloader is empty. Lower data.batch_size, or -- on a distributed run -- "
                "note that the dataset is sharded first, so each rank needs at least one "
                "full batch of its own share."
            )
        sampler = dataloader.sampler if context.enabled else None
        num_crops = transform.num_crops

        # ------------------------------------------------------- corpus provenance
        #
        # The stage-1 -> stage-2 handoff is a bare `state_dict`, and until this
        # block existed nothing anywhere recorded WHAT it was self-distilled on.
        # That was not hypothetical: the shipped 100-epoch encoder was trained on
        # 8,173 crops while the evaluation, stage 2 and every published number use
        # 9,357, and the discrepancy was recoverable only by cross-reading two log
        # lines against `metrics.json`.
        #
        # The digest is over the sorted list of dataset-relative paths, so it is
        # stable across machines and mount points and changes the instant a file
        # is added, removed or renamed. It travels into `events.jsonl`, into
        # `summary.json` beside the checkpoints, and -- via that file --
        # into `pretrain_eval`, which compares it against its own corpus and
        # prints a prominent mismatch line.
        fingerprint = _corpus_fingerprint_of(dataloader.dataset, cfg, logger)
        coverage = {}
        raw_root = OmegaConf.select(cfg, "data.raw_photographs_root", default=None)
        if raw_root:
            coverage = raw_photograph_coverage(str(raw_root), str(cfg.data.root_path))
        if fingerprint:
            logger.info(
                "Corpus | %s images in %s classes from %s source photographs, digest %s "
                "(root %s).",
                fingerprint["num_samples"], fingerprint["num_classes"],
                fingerprint["num_source_groups"], str(fingerprint["sha256"])[:16],
                fingerprint["root"],
            )
            tracker.log_event("corpus", {**fingerprint, "raw_photograph_coverage": coverage})
            _check_corpus(cfg, fingerprint, logger)
        if coverage:
            logger.warning(
                "Raw photographs | %s of %s source photographs under %s were never cropped, so "
                "this corpus covers %s scenes rather than %s. Scene count -- not crop count -- is "
                "the binding constraint on photograph-disjoint generalisation. See "
                "scripts/report_raw_photographs.py.",
                coverage["num_unused_photographs"], coverage["num_raw_photographs"],
                coverage["raw_root"], coverage["num_used_photographs"],
                coverage["num_raw_photographs"],
            )

        logger.info(
            "Loaded %s batches of %s images, %s views each (2 global + %s local).",
            len(dataloader),
            cfg.data.batch_size,
            num_crops,
            cfg.data.augmentation.local_crops_number,
        )
        same_photo_views = int(
            OmegaConf.select(cfg, "data.augmentation.same_photo_local_views", default=0) or 0
        )
        if same_photo_views:
            logger.warning(
                "Provenance-derived positives ON: %s of %s local views are other crops of the "
                "SAME source photograph. This changes what invariance is being learned. Gate the "
                "result on the evaluation's within-class photograph decodability -- an arm that "
                "raises it has learned the photograph confound, not the variety.",
                same_photo_views, cfg.data.augmentation.local_crops_number,
            )
        logger.info(
            "View geometry | emitted sizes=%s dtype=%s local upsample on %s",
            transform.view_sizes,
            "uint8" if transform.output_uint8 else "float32",
            "device" if transform.upsample_locals_on_device else "CPU",
        )
        batcher = ViewBatcher(
            image_size=int(cfg.data.image_size),
            mean=transform.normalize_mean,
            std=transform.normalize_std,
            device=device,
        )

        # ------------------------------------------------------------ model
        pretrained_init = bool(OmegaConf.select(cfg, "model.backbone.pretrained", default=False))
        logger.info(
            "Initialising DINO with backbone %s (%s initialisation).",
            cfg.model.backbone.name,
            "ImageNet-1k pretrained" if pretrained_init else "random",
        )
        model = build_dino(
            backbone_cfg=cfg.model.backbone,
            head_cfg=cfg.model.head,
            freeze_last_layer_epochs=int(cfg.experiment.training.freeze_last_layer_epochs),
        ).to(device)

        # ------------------------------------------------- startup verification
        #
        # Three things that are silent when wrong and expensive to discover late.
        #
        # (1) The trunk must be TRAINABLE. `build_dino` already refuses
        #     `freeze=true`, but nothing stops a future caller from freezing
        #     parameters afterwards, and a run that self-distils into a frozen
        #     trunk trains the projection head alone and publishes an unadapted
        #     encoder. Counted, not assumed.
        # (2) The teacher must NOT have stochastic depth. It is a deepcopy of the
        #     student and its outputs are the targets; DropPath there is noise in
        #     the label.
        # (3) The shapes, end to end. The token grid in particular: stage 2's
        #     grid routing consumes an 8x8 final stage, and a backbone swap that
        #     changed it would surface much later as a reshape deep in the head.
        parameters = model.parameter_summary()
        if parameters["student_backbone_trainable"] != parameters["backbone"]:
            raise RuntimeError(
                "The SwinV2 trunk is not fully trainable "
                f"({parameters['student_backbone_trainable']:,} of {parameters['backbone']:,} "
                "parameters require gradient). Stage 1 fine-tunes the encoder; a frozen trunk "
                "would train the projection head against fixed features and publish an "
                "unadapted encoder. Check model.backbone.freeze."
            )
        shapes = model.shape_report(image_size=int(cfg.data.image_size), device=device)
        logger.info(
            "Shapes | input %s -> tokens %s (grid %sx%s = %s) -> pooled %s -> "
            "head in %s -> bottleneck %s -> student prototypes %s / teacher %s",
            shapes["input"], shapes["backbone_tokens"],
            shapes["token_grid"][0], shapes["token_grid"][1], shapes["tokens_per_image"],
            shapes["pooled_features"], shapes["head_input_dim"], shapes["head_bottleneck"],
            shapes["student_prototypes"], shapes["teacher_prototypes"],
        )
        logger.info(
            "Parameters | backbone %.2f M (%.2f M trainable) + DINO head %.2f M = student "
            "%.2f M | teacher %.2f M (EMA only) | drop_path=%.3g on the student, disabled on "
            "%s teacher modules",
            parameters["backbone"] / 1e6,
            parameters["student_backbone_trainable"] / 1e6,
            parameters["dino_head"] / 1e6,
            parameters["student_total"] / 1e6,
            parameters["teacher_total"] / 1e6,
            model.drop_path_rate,
            model.teacher_drop_paths_disabled,
        )
        tracker.log_event(
            "model_shapes",
            {
                **{key: list(value) if isinstance(value, tuple) else value
                   for key, value in shapes.items()},
                "pretrained_init": "imagenet1k" if pretrained_init else "random",
                "drop_path_rate": model.drop_path_rate,
                "teacher_drop_paths_disabled": model.teacher_drop_paths_disabled,
                **{f"params_{key}": value for key, value in parameters.items()},
            },
        )

        # GFLOPs per view, measured on the EAGER model before torch.compile and
        # DDP touch it -- neither changes the arithmetic, and both make the
        # dispatch counter's job harder. One forward at batch 1, under no_grad.
        gflops_per_view = None
        if bool(OmegaConf.select(cfg, "experiment.budget.enabled", default=True)) and bool(
            OmegaConf.select(cfg, "experiment.budget.measure_flops", default=True)
        ):
            gflops_per_view = measure_gflops_per_view(
                _StudentViewPass(model), int(cfg.data.image_size), device
            )
            if gflops_per_view is None:
                logger.warning("FLOP counting is unavailable here; the budget omits GFLOPs.")
            else:
                logger.info(
                    "Measured %.2f GFLOPs per %s px view (student backbone + head, forward).",
                    gflops_per_view, int(cfg.data.image_size),
                )

        runtime = model.configure_runtime(
            compile_enabled=resolve_compile(
                OmegaConf.select(cfg, "experiment.training.compile.enabled", default="auto"),
                device,
                logger,
            ),
            compile_mode=str(
                OmegaConf.select(cfg, "experiment.training.compile.mode", default="default")
            ),
            grad_checkpointing=bool(
                OmegaConf.select(cfg, "experiment.training.grad_checkpointing", default=False)
            ),
            channels_last=bool(
                OmegaConf.select(cfg, "experiment.training.channels_last", default=False)
            ),
            # Default true: every module parity-checks itself against its own
            # stock forward at conversion time and falls back on disagreement,
            # so the failure mode of a drifted timm is current performance, not
            # a wrong model.
            sdpa_attention=bool(
                OmegaConf.select(cfg, "experiment.training.sdpa_attention", default=True)
            ),
            logger=logger,
        )
        tracker.log_event("model_runtime", runtime)
        forward_chunk_size = OmegaConf.select(
            cfg, "experiment.training.forward_chunk_size", default=None
        )
        forward_chunk_size = int(forward_chunk_size) if forward_chunk_size else None

        tracker.log_model_watch(model)
        student_parameters = model.student_parameters()
        tracker.log_metrics(
            {"model/student_parameters": sum(p.numel() for p in student_parameters)}, step=0
        )
        ema = TeacherEmaUpdater(model.ema_pairs())
        logger.info("EMA covers %s teacher tensors, updated with fused foreach kernels.", len(ema))

        criterion = CustomDINOLoss(
            out_dim=int(cfg.model.head.out_dim),
            num_crops=num_crops,
            warmup_teacher_temp=float(cfg.model.loss.warmup_teacher_temp),
            teacher_temp=float(cfg.model.loss.teacher_temp),
            warmup_teacher_temp_epochs=int(cfg.model.loss.warmup_teacher_temp_epochs),
            num_epochs=int(cfg.experiment.training.epochs),
            student_temp=float(cfg.model.loss.student_temp),
            center_momentum=float(cfg.model.loss.center_momentum),
            num_global_crops=2,
            centering=str(OmegaConf.select(cfg, "model.loss.centering", default="sinkhorn")),
            sinkhorn_iterations=int(
                OmegaConf.select(cfg, "model.loss.sinkhorn_iterations", default=3)
            ),
            lambda_koleo=float(OmegaConf.select(cfg, "model.loss.lambda_koleo", default=0.0)),
            koleo_scope=str(
                OmegaConf.select(cfg, "model.loss.koleo_scope", default="per_view")
            ),
            koleo_reduction=str(
                OmegaConf.select(cfg, "model.loss.koleo_reduction", default="mean")
            ),
            context=context,
            distributed_sinkhorn=bool(
                OmegaConf.select(cfg, "model.loss.distributed_sinkhorn", default=False)
            ),
        ).to(device)
        koleo_space = str(OmegaConf.select(cfg, "model.loss.koleo_space", default="bottleneck"))
        if koleo_space not in {"bottleneck", "backbone"}:
            raise ValueError(
                f"model.loss.koleo_space must be 'bottleneck' or 'backbone', got {koleo_space!r}"
            )
        logger.info(
            "Stage 1 objective: DINO self-distillation | centering=%s koleo=%s (scope=%s, "
            "space=%s) prototypes=%s distributed_sinkhorn=%s",
            criterion.centering,
            criterion.lambda_koleo,
            criterion.koleo_scope,
            koleo_space,
            int(cfg.model.head.out_dim),
            criterion.distributed_sinkhorn,
        )
        if criterion.lambda_koleo > 0 and criterion.koleo_scope == "all_views":
            logger.warning(
                "koleo_scope=all_views applies the nearest-neighbour term ACROSS the two global "
                "views, so the two views of one image are each other's closest pair and the "
                "gradient pushes them apart -- an anti-alignment force on exactly the pair Eq. 1 "
                "pulls together. This is the pre-audit behaviour, kept only as a control."
            )

        # The auxiliary stage head (the C4 arm). Off unless model.head.aux_stage
        # is set, in which case a second DINO objective supervises `layers.2`
        # directly instead of reaching it through two blocks optimised for the
        # 2,048-prototype task at `layers.3`.
        aux_enabled = model.aux_stage is not None
        aux_criterion = None
        if aux_enabled:
            aux_criterion = CustomDINOLoss(
                out_dim=int(
                    OmegaConf.select(cfg, "model.head.aux_out_dim", default=None)
                    or cfg.model.head.out_dim
                ),
                num_crops=num_crops,
                warmup_teacher_temp=float(cfg.model.loss.warmup_teacher_temp),
                teacher_temp=float(cfg.model.loss.teacher_temp),
                warmup_teacher_temp_epochs=int(cfg.model.loss.warmup_teacher_temp_epochs),
                num_epochs=int(cfg.experiment.training.epochs),
                student_temp=float(cfg.model.loss.student_temp),
                center_momentum=float(cfg.model.loss.center_momentum),
                num_global_crops=2,
                centering=str(OmegaConf.select(cfg, "model.loss.centering", default="sinkhorn")),
                sinkhorn_iterations=int(
                    OmegaConf.select(cfg, "model.loss.sinkhorn_iterations", default=3)
                ),
                # KoLeo is applied once, by the primary criterion, on the space
                # `koleo_space` names. A second copy on the auxiliary head would
                # double an unrelated regulariser as a side effect of enabling
                # the arm, which would make the arm uninterpretable.
                lambda_koleo=0.0,
                context=context,
                distributed_sinkhorn=bool(
                    OmegaConf.select(cfg, "model.loss.distributed_sinkhorn", default=False)
                ),
            ).to(device)
            logger.warning(
                "Auxiliary stage-%s DINO head ON at weight %.3g (%.2f M parameters). The "
                "objective now supervises layers.%s directly as well as the trunk output; this "
                "confounds a simultaneous model.backbone.feature_stage change, so run it after "
                "that decision, not beside it.",
                model.aux_stage, model.aux_weight,
                parameters["dino_aux_head"] / 1e6, model.aux_stage,
            )

        # DDP goes on *after* the runtime options, so the compiled callables and
        # the SDPA rebinding are already in place when the reducer is built.
        distributed_runtime = model.configure_distributed(
            context,
            gradient_as_bucket_view=bool(
                OmegaConf.select(cfg, "experiment.training.ddp.gradient_as_bucket_view", default=True)
            ),
            static_graph=bool(
                OmegaConf.select(cfg, "experiment.training.ddp.static_graph", default=False)
            ),
            find_unused_parameters=bool(
                OmegaConf.select(
                    cfg, "experiment.training.ddp.find_unused_parameters", default=False
                )
            ),
            bucket_cap_mb=OmegaConf.select(
                cfg, "experiment.training.ddp.bucket_cap_mb", default=None
            ),
            logger=logger,
        )
        tracker.log_event("distributed_runtime", distributed_runtime)

        # The accumulation count is resolved BEFORE the optimizer, because the
        # learning rate is derived from the effective batch it pins. Deriving
        # the rate from `data.batch_size` alone, or from a literal, is how a
        # two-GPU relaunch silently trains at twice the intended step size.
        accumulation_steps, effective_batch = resolve_accumulation(cfg, context, logger)
        learning_rate, lr_provenance = resolve_learning_rate(cfg, effective_batch, logger)
        tracker.log_event("learning_rate", lr_provenance)

        # Two groups: everything with more than one dimension decays, biases and
        # 1-D normalisation parameters do not. See build_param_groups.
        param_groups = build_param_groups(
            model.student_backbone,
            model.student_head,
            weight_decay=float(cfg.experiment.training.weight_decay),
        )
        logger.info(
            "Optimizer groups | %s decayed tensors (wd %.3g -> %.3g), %s excluded "
            "(biases and 1-D norm parameters, wd 0 throughout).",
            len(param_groups[0]["params"]),
            float(cfg.experiment.training.weight_decay),
            float(OmegaConf.select(cfg, "experiment.training.weight_decay_final", default=0.0) or 0.0),
            len(param_groups[1]["params"]),
        )
        optimizer = build_optimizer(param_groups, cfg, device, logger, learning_rate=learning_rate)
        scheduler = build_scheduler(optimizer, cfg, logger=logger)

        save_path = Path(cfg.experiment.training.save_path)
        if context.is_main:
            save_path.mkdir(parents=True, exist_ok=True)
        barrier(context)
        checkpoint_manager = CheckpointManager(
            save_path,
            keep_last_n=int(cfg.experiment.training.keep_last_n_checkpoints),
            enabled=context.is_main,
        )
        resume_manager = CheckpointManager(
            save_path, keep_last_n=RESUME_KEEP_LAST, enabled=context.is_main
        )

        # ------------------------------------------------------------- resume
        components = resume_components(model, criterion, optimizer, scheduler, scaler)
        resume = ResumeState()
        resume_path = resolve_resume_path(
            OmegaConf.select(cfg, "experiment.training.resume", default=False),
            save_path,
            patterns=(f"{RESUME_PREFIX}*.pth",),
            logger=logger,
        )
        # Rank 0 decides; every rank obeys. Two ranks scanning the same directory
        # can disagree the moment one of them sees a file mid-write, and a job
        # where half the ranks resumed is worse than one where none did.
        resume_path = broadcast_object(str(resume_path) if resume_path else None, context)
        if resume_path:
            payload = load_checkpoint_payload(resume_path, map_location=device, logger=logger)
            report = restore_components(payload, components, strict=True, logger=logger)
            restore_rng_states(payload, context, loader_generator, logger=logger)
            resume = ResumeState(
                path=Path(resume_path),
                progress=TrainingProgress.from_dict(payload.get("progress")),
                report=report,
            )
            logger.info("Resume | %s", resume.describe())
            tracker.log_event(
                "resume",
                {
                    "path": str(resume_path),
                    "saved_world_size": (payload.get("distributed") or {}).get("world_size"),
                    "world_size": context.world_size,
                    **resume.progress.as_dict(),
                },
            )

        # ------------------------------------------------------- loop config
        epochs = int(cfg.experiment.training.epochs)
        max_batches = OmegaConf.select(cfg, "experiment.training.max_batches", default=None)
        momentum_start = float(cfg.experiment.training.momentum_teacher)
        momentum_final = OmegaConf.select(
            cfg, "experiment.training.momentum_teacher_final", default=None
        )
        weight_decay_start = float(cfg.experiment.training.weight_decay)
        weight_decay_final = OmegaConf.select(
            cfg, "experiment.training.weight_decay_final", default=None
        )
        # Every collapse guard in DINO is a batch statistic, and it is the
        # PER-MICRO-BATCH size those statistics actually see -- Sinkhorn and
        # KoLeo run inside each micro-batch, so accumulation buys gradient
        # averaging and nothing else. Both numbers are logged for that reason.
        logger.info(
            "Effective batch: %s per rank x %s ranks x %s accumulation = %s images "
            "(%s teacher views/step, %s student views/step). Sinkhorn/KoLeo see %s images "
            "and %s teacher views per estimate.",
            int(cfg.data.batch_size),
            context.world_size,
            accumulation_steps,
            effective_batch,
            effective_batch * 2,
            effective_batch * num_crops,
            int(cfg.data.batch_size),
            int(cfg.data.batch_size) * 2,
        )
        entropy_min, entropy_max = criterion.entropy_bounds(int(cfg.data.batch_size) * 2)
        logger.info(
            "Teacher entropy bounds for this configuration: H in [%.3f, %.3f] nats "
            "(K=%s prototypes, %s teacher views, centering=%s). The floor is structural, "
            "not a model property -- read H against it, not against 0.",
            entropy_min, entropy_max, int(cfg.model.head.out_dim),
            int(cfg.data.batch_size) * 2, criterion.centering,
        )
        # Only ask the student pass for the pooled trunk feature when something
        # will read it: returning it otherwise keeps a [6B, 768] tensor alive for
        # the whole step for nothing.
        need_backbone_koleo = koleo_space == "backbone" and criterion.lambda_koleo > 0
        measure_gpu_busy = bool(
            OmegaConf.select(cfg, "experiment.training.measure_gpu_busy", default=False)
        )
        busy_meter = GpuBusyMeter(device, measure_gpu_busy)
        if measure_gpu_busy and not busy_meter.enabled:
            logger.info(
                "measure_gpu_busy has no effect on %s: there is no separate device queue to "
                "time.", device.type,
            )
        elif busy_meter.enabled:
            logger.warning(
                "measure_gpu_busy is ON. It synchronises once per logging interval, so the "
                "reported throughput is slightly pessimistic; use it to establish the real GPU "
                "busy fraction, then turn it off for the production run."
            )

        clip_grad = cfg.experiment.training.clip_grad
        save_interval = int(cfg.experiment.training.save_interval)
        # Permanently-kept milestone epochs. Named artifacts, so
        # `keep_last_n_checkpoints` never prunes them -- the point of the list is
        # that all of 25/50/100 survive to be evaluated against each other.
        save_epochs = sorted(
            {
                int(epoch)
                for epoch in (
                    OmegaConf.select(cfg, "experiment.training.save_epochs", default=None) or []
                )
                if 0 < int(epoch) <= epochs
            }
        )
        if save_epochs:
            logger.info(
                "Milestone checkpoints (kept permanently) at epochs: %s",
                ", ".join(str(epoch) for epoch in save_epochs),
            )
        save_full_checkpoints = bool(cfg.experiment.training.save_full_checkpoints)
        save_teacher = bool(cfg.experiment.training.save_teacher_in_checkpoints)
        fast_forward = bool(
            OmegaConf.select(cfg, "experiment.training.resume_fast_forward", default=True)
        )

        intervals = cfg.tracking.intervals
        artifacts = cfg.tracking.artifacts
        log_every_steps = int(intervals.log_every_steps)
        device_every_steps = int(intervals.device_every_steps)
        # Per-parameter gradient norms cost one host synchronisation *per
        # tensor* -- ~440 of them for this student. The clipped total norm is
        # free (it is computed anyway), so the per-tensor breakdown runs on its
        # own, much rarer, interval.
        gradient_norm_every_steps = int(
            OmegaConf.select(cfg, "tracking.intervals.gradient_norm_every_steps", default=200)
        )
        global_step = resume.progress.global_step
        loss_history: list[float] = []
        kl_history: list[float] = []

        momentum = momentum_start
        logger.info(
            "Training for %s epochs at lr=%.6g (%s epochs of warmup), teacher momentum=%s -> %s.",
            epochs,
            learning_rate,
            resolve_warmup_epochs(cfg),
            momentum_start,
            momentum_final if momentum_final is not None else "(constant)",
        )

        # ---------------------------------------------------------- budget
        # Assembled now (everything it needs is resolved), printed twice: here
        # without the runtime half, and again at the end with it. An interrupted
        # run therefore still leaves a complete parameter/compute report.
        batches_per_epoch = (
            min(len(dataloader), int(max_batches)) if max_batches is not None else len(dataloader)
        )
        budget = StageOneBudget.from_model(
            parameters,
            gflops_per_view=gflops_per_view,
            image_size=int(cfg.data.image_size),
            views_per_image=num_crops,
            global_views_per_image=2,
            epochs=epochs,
            physical_batch_size=int(cfg.data.batch_size),
            gradient_accumulation_steps=accumulation_steps,
            world_size=context.world_size,
            effective_batch_size=effective_batch,
            steps_per_epoch=-(-batches_per_epoch // accumulation_steps),
            images_per_epoch=batches_per_epoch * int(cfg.data.batch_size) * context.world_size,
            precision=amp.label,
            optimizer=str(cfg.experiment.training.optimizer.name),
            learning_rate=learning_rate,
            weight_decay=weight_decay_start,
            backbone_name=str(cfg.model.backbone.name),
            pretrained_init="ImageNet-1k" if pretrained_init else "random",
            drop_path_rate=model.drop_path_rate,
            prototypes=int(cfg.model.head.out_dim),
        )
        budget_enabled = bool(OmegaConf.select(cfg, "experiment.budget.enabled", default=True))
        if budget_enabled and context.is_main:
            logger.info("Stage-1 budget (pre-run)\n%s", budget.format_table())
            tracker.log_metrics(budget.as_metrics(), step=0)
            tracker.log_event("stage1_budget", {"phase": "start", **budget.as_dict()})

        # Whole-run memory high-water marks. The epoch loop resets CUDA's peak
        # counters so it can report per-epoch memory, so the run-wide maximum has
        # to be accumulated here or the final report would show only the last
        # epoch's peak -- lower than the truth, in the direction that gets a
        # relaunch OOM-killed.
        run_peak_allocated_gb = 0.0
        run_peak_reserved_gb = 0.0
        images_processed = 0

        periodic = PeriodicSaver(
            OmegaConf.select(cfg, "experiment.training.resume_every_minutes", default=None)
        )
        # How often the ranks compare notes about stopping and saving. Every
        # rank reaches the same `global_step`, so this is the one schedule they
        # can agree on without a collective per step.
        resume_check_every = max(
            int(OmegaConf.select(cfg, "experiment.training.resume_check_every_steps", default=20)), 1
        )
        guard_ctx = InterruptGuard(
            OmegaConf.select(cfg, "experiment.training.max_runtime_minutes", default=None),
            logger=logger,
        )

        with guard_ctx as guard:
            for epoch in range(resume.progress.epoch, epochs):
                epoch_started = time.perf_counter()
                # Accumulated on the device. Summing `float(loss)` here would put
                # a full pipeline stall in every micro-batch.
                total_loss = torch.zeros((), device=device, dtype=torch.float32)
                # A4: the reported loss is 95 % target entropy, so the epoch
                # summary carries the decomposition rather than the sum alone.
                # Both are device tensors read from `last_metric_tensors`, which
                # does not synchronise -- the conversion happens once, at the
                # epoch boundary, exactly as `total_loss` already does.
                total_kl = torch.zeros((), device=device, dtype=torch.float32)
                total_entropy = torch.zeros((), device=device, dtype=torch.float32)
                # The primary cross-entropy term alone. `total_loss` is the whole
                # objective and can carry KoLeo and the auxiliary head on top, so
                # only THIS decomposes exactly as `CE = H + KL`.
                total_ce = torch.zeros((), device=device, dtype=torch.float32)
                batches_seen = 0
                data_wait = 0.0
                busy_meter.seconds = 0.0
                model.train()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()

                # The sampler's permutation is a function of `seed + epoch`.
                # Without this call every epoch replays the first one -- no
                # error, no visible change in the loss magnitude, and 100 epochs
                # of one epoch's data.
                if sampler is not None:
                    sampler.set_epoch(epoch)

                momentum = (
                    cosine_value(momentum_start, float(momentum_final), epoch, epochs)
                    if momentum_final is not None
                    else momentum_start
                )

                optimizer.zero_grad(set_to_none=True)
                micro_in_window = 0
                skip_batches = 0
                if epoch == resume.progress.epoch and resume.progress.micro_step > 0:
                    skip_batches = int(resume.progress.micro_step)
                    if not fast_forward:
                        logger.warning(
                            "resume_fast_forward=false: restarting epoch %s from its first "
                            "batch instead of continuing at micro-batch %s.",
                            epoch + 1, skip_batches,
                        )
                        skip_batches = 0

                iterator = enumerate(dataloader)
                if skip_batches:
                    logger.info(
                        "Skipping %s already-consumed micro-batches of epoch %s.",
                        skip_batches, epoch + 1,
                    )

                wait_started = time.perf_counter()
                for batch_idx, batch in iterator:
                    if batch_idx < skip_batches:
                        # Consumed and discarded, which is what makes the resume
                        # land mid-epoch rather than restart it: the sample order
                        # comes from the sampler's `seed + epoch` permutation (or
                        # the restored generator state on one process), so the
                        # batches after the skip are the ones the interrupted run
                        # would have seen next. The cost is real augmentation
                        # work thrown away, bounded by the save interval.
                        wait_started = time.perf_counter()
                        continue
                    if max_batches is not None and batch_idx >= max_batches:
                        break
                    data_wait += time.perf_counter() - wait_started

                    is_last_batch = (
                        batch_idx + 1 == len(dataloader)
                        or (max_batches is not None and batch_idx + 1 >= max_batches)
                    )
                    is_step = (micro_in_window + 1) % accumulation_steps == 0 or is_last_batch
                    will_log = is_step and (global_step % log_every_steps == 0)
                    # The collapse diagnostics are only ever read on logging
                    # steps, so on every other step they are not computed at all.
                    criterion.metrics_enabled = will_log

                    student_views, teacher_views, batch_size = batcher(batch)

                    # Suppress DDP's all-reduce except on the boundary. Without
                    # this the full gradient crosses the interconnect once per
                    # micro-batch instead of once per optimizer step, for an
                    # identical result -- which on two T4s over PCIe is enough
                    # to make the two-GPU run slower than the one-GPU run.
                    sync_context = nullcontext() if is_step else model.no_sync()
                    with sync_context, busy_meter.start():
                        with autocast_context(amp):
                            # One fused forward for the teacher's 2B globals ...
                            teacher_result = model.forward_teacher_views(
                                teacher_views,
                                chunk_size=forward_chunk_size,
                                return_aux=aux_enabled,
                            )
                            teacher_out, teacher_aux = (
                                teacher_result if aux_enabled else (teacher_result, None)
                            )
                            # ... and one for the student's 6B views, view-major.
                            student_result = model.forward_student_views(
                                student_views,
                                return_bottleneck=True,
                                chunk_size=forward_chunk_size,
                                return_features=need_backbone_koleo,
                                return_aux=aux_enabled,
                            )
                            student_out, bottleneck, *rest = student_result
                            trunk_features = rest.pop(0) if need_backbone_koleo else None
                            student_aux = rest.pop(0) if aux_enabled else None
                            # KoLeo measures uniformity of the *representation*.
                            # `bottleneck` is the head's 256-D L2-normalised
                            # embedding, which stage 2 never sees;
                            # `koleo_space=backbone` regularises the pooled trunk
                            # feature instead -- the space that actually ships,
                            # and the one DINOv2 applies KoLeo to. Either way the
                            # rows handed over are the leading 2B GLOBAL views in
                            # view-major order, which is what makes the per-view
                            # chunking inside the loss meaningful.
                            koleo_source = (
                                trunk_features if koleo_space == "backbone" else bottleneck
                            )
                            student_embeddings = (
                                koleo_source[: 2 * batch_size]
                                if criterion.lambda_koleo > 0
                                else None
                            )
                            loss = criterion(
                                student_out,
                                teacher_out,
                                epoch=epoch,
                                # Explicit view identifiers. Matching by position
                                # is only correct while the student's first two
                                # views are the two globals in the teacher's
                                # order -- an invariant nothing enforced, and one
                                # whose violation would silently skip a
                                # global-local pair and include a same-view pair.
                                student_view_ids=transform.view_ids,
                                teacher_view_ids=transform.global_view_ids,
                                student_embeddings=student_embeddings,
                            )
                            if aux_enabled:
                                aux_criterion.metrics_enabled = will_log
                                aux_loss = aux_criterion(
                                    student_aux,
                                    teacher_aux,
                                    epoch=epoch,
                                    student_view_ids=transform.view_ids,
                                    teacher_view_ids=transform.global_view_ids,
                                )
                                loss = loss + model.aux_weight * aux_loss

                        scaled = loss / accumulation_steps
                        if scaler is not None:
                            scaler.scale(scaled).backward()
                        else:
                            scaled.backward()

                    gradient_norm = None
                    clipped_norm = None

                    if is_step:
                        if scaler is not None:
                            # Gradients must be back on their true scale before
                            # anything inspects or clips them.
                            scaler.unscale_(optimizer)

                        if (
                            artifacts.log_gradient_norms
                            and gradient_norm_every_steps > 0
                            and global_step % gradient_norm_every_steps == 0
                        ):
                            gradient_norm = tracker.log_gradient_norms(model, global_step)

                        if clip_grad is not None and float(clip_grad) > 0:
                            # Every rank clips the same all-reduced gradient, so
                            # every rank computes the same scale factor and the
                            # parameters stay bit-identical across the job.
                            clipped_norm = torch.nn.utils.clip_grad_norm_(
                                student_parameters, max_norm=float(clip_grad)
                            )

                        # Section 6.1: last layer frozen for the first epoch.
                        # Cancelled *after* clipping and *before* step(),
                        # matching the reference implementation's ordering.
                        model.student_head.cancel_last_layer_gradients(current_epoch=epoch)

                        if (
                            artifacts.log_gradient_histograms
                            and int(intervals.gradient_histogram_every_epochs) > 0
                            and (epoch + 1) % int(intervals.gradient_histogram_every_epochs) == 0
                            and batch_idx == 0
                        ):
                            tracker.log_gradient_histograms(model, global_step)

                        if scaler is not None:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                        # EMA teacher update, cosine-scheduled 0.996 -> 1.0
                        # (DINO). Every rank runs it on identical students, so
                        # the teachers stay identical without a collective.
                        ema.update(momentum)
                        micro_in_window = 0
                    else:
                        micro_in_window += 1

                    total_loss += loss.detach().float()
                    decomposition = criterion.last_metric_tensors
                    total_kl += decomposition["teacher_student_kl"].float()
                    total_entropy += decomposition["teacher_entropy_cross_view"].float()
                    total_ce += decomposition["dino_cross_entropy"].float()
                    batches_seen += 1

                    if will_log:
                        lr = optimizer.param_groups[0]["lr"]
                        elapsed = max(time.perf_counter() - epoch_started, 1e-9)
                        # One collective, on logging steps only, so the reported
                        # loss is the global one rather than this rank's shard.
                        logged_loss = all_reduce_mean(loss.detach().float().clone(), context)
                        step_metrics = {
                            # The single synchronisation point of the step.
                            "loss": float(logged_loss),
                            "lr": lr,
                            "teacher_temp": criterion.teacher_temperature(epoch),
                            "teacher_momentum": momentum,
                            "weight_decay": optimizer.param_groups[0]["weight_decay"],
                            "images_per_second": (
                                batches_seen * batch_size * context.world_size / elapsed
                            ),
                            "views_per_second": (
                                batches_seen * batch_size * num_crops * context.world_size / elapsed
                            ),
                            # Wall clock the TRAINING LOOP spent blocked inside
                            # the dataloader's `__next__`, as a share of the
                            # epoch so far.
                            #
                            # It is NOT a GPU-idle fraction, and reading it as
                            # one produces a wrong throughput number: nothing
                            # synchronises inside the step (deliberately -- see
                            # the module docstring), so the queued GPU work
                            # drains *during* this window. The shipped run's
                            # 0.916 was turned into "GPU-busy time = 13.34 h x
                            # (1 - 0.916) = 1.12 h", which is the CPU *enqueue*
                            # time. The metric's direction and its operational
                            # conclusion ("the loader is the bottleneck") are
                            # right; the derived FLOP/s figure is not.
                            #
                            # `experiment.training.measure_gpu_busy=true` adds a
                            # genuinely synchronised measurement below, at the
                            # cost of one stall per logging step.
                            "loop_blocked_fraction": data_wait / elapsed,
                            # Retained under its historical name so a figure or a
                            # parser written against the shipped run keeps
                            # working. Same number, better name above.
                            "data_wait_fraction": data_wait / elapsed,
                            # The loss curve of a partially collapsed run looks
                            # perfectly plausible. These are the numbers that do
                            # not.
                            **criterion.last_metrics,
                        }
                        if busy_meter.enabled:
                            busy_seconds = busy_meter.drain()
                            step_metrics["gpu_busy_fraction"] = busy_seconds / elapsed
                            step_metrics["gpu_busy_seconds"] = busy_seconds
                        if aux_criterion is not None:
                            step_metrics.update(
                                {f"aux_{key}": value for key, value in aux_criterion.last_metrics.items()}
                            )
                        if gradient_norm is not None:
                            step_metrics["gradient_norm"] = gradient_norm
                        if clipped_norm is not None:
                            step_metrics["clipped_gradient_norm"] = float(clipped_norm)
                        if scaler is not None:
                            step_metrics["grad_scale"] = float(scaler.get_scale())
                        tracker.log_metrics(step_metrics, global_step, prefix="train")
                        logger.info(
                            "Step %s | epoch=%s batch=%s loss=%.5f | CE=%.5f = KL %.5f + H %.5f "
                            "| lr=%.6g tau_t=%.4f img/s=%.1f loop_blocked=%.1f%%",
                            global_step, epoch + 1, batch_idx + 1, step_metrics["loss"],
                            step_metrics.get("dino_cross_entropy", float("nan")),
                            step_metrics.get("teacher_student_kl", float("nan")),
                            step_metrics.get("teacher_entropy_cross_view", float("nan")),
                            lr, criterion.teacher_temperature(epoch),
                            step_metrics["images_per_second"],
                            step_metrics["loop_blocked_fraction"] * 100.0,
                        )

                    if is_step and global_step % device_every_steps == 0:
                        tracker.log_metrics(collect_device_stats(device), global_step)

                    if (
                        artifacts.log_embeddings
                        and int(intervals.embedding_every_epochs) > 0
                        and (epoch + 1) % int(intervals.embedding_every_epochs) == 0
                        and batch_idx == 0
                    ):
                        tracker.log_embeddings(
                            "dino/student_projection",
                            student_out[:batch_size].detach(),
                            global_step,
                            metadata=[str(item) for item in batch.paths],
                        )

                    if (
                        artifacts.log_attention_maps
                        and int(intervals.attention_every_epochs) > 0
                        and (epoch + 1) % int(intervals.attention_every_epochs) == 0
                        and batch_idx == 0
                    ):
                        log_attention_maps(
                            tracker, model.student_backbone, student_views[:batch_size], global_step,
                            logger=logger, max_images=int(artifacts.max_attention_images),
                        )

                    if is_step:
                        global_step += 1

                        # Checkpoints are written only here, at an accumulation
                        # boundary with the gradients already applied and zeroed.
                        # Saving mid-window would store a half-accumulated
                        # gradient that nothing restores, so a resume would drop
                        # part of one step's data -- silently.
                        #
                        # Latched every step (a monotonic clock comparison), but
                        # *acted on* only at a step index every rank agrees on.
                        # Both triggers are wall-clock based and so genuinely
                        # differ between ranks: acting locally would put one rank
                        # into the checkpoint's RNG gather -- a collective --
                        # while its peers ran on, and the job would hang at the
                        # collective timeout with no error to point at.
                        stop = guard.should_stop()
                        due = periodic.due()
                        stop_now, stop_reason = stop.requested, stop.reason

                        if context.enabled and global_step % resume_check_every == 0:
                            flags = torch.tensor(
                                [1.0 if due else 0.0, 1.0 if stop.requested else 0.0],
                                device=device,
                                dtype=torch.float32,
                            )
                            all_reduce_max(flags, context)
                            due, stop_now = bool(flags[0] > 0), bool(flags[1] > 0)
                            if stop_now and not stop_reason:
                                stop_reason = "another rank requested a stop"
                        elif context.enabled:
                            # Not a decision point for the job: keep the latch,
                            # take no collective.
                            due, stop_now = False, False

                        if due or stop_now:
                            progress = resume_position(
                                epoch, global_step, batch_idx, is_last_batch, epochs
                            )
                            path = save_resume_checkpoint(
                                resume_manager,
                                f"{RESUME_PREFIX}{global_step:09d}.pth",
                                components=components,
                                progress=progress,
                                context=context,
                                cfg=cfg,
                                loader_generator=loader_generator,
                                extra={"effective_batch_size": effective_batch},
                            )
                            periodic.mark()
                            tracker.log_event(
                                "resume_checkpoint",
                                {"path": path, "reason": stop_reason or "interval", **progress.as_dict()},
                            )
                            logger.info(
                                "Wrote resume checkpoint at step %s (%s): %s",
                                global_step, stop_reason or "interval", path,
                            )
                        if stop_now:
                            interrupted = True
                            logger.warning("Stopping after step %s: %s", global_step, stop_reason)
                            break

                    wait_started = time.perf_counter()

                if interrupted:
                    break

                if batches_seen == 0:
                    if skip_batches == 0:
                        raise RuntimeError(
                            "No batches were processed. Check max_batches and the dataloader."
                        )
                    # Every micro-batch of this epoch was already consumed before
                    # the interruption, and fewer are available now than were
                    # then -- which happens when the job resumes at a smaller
                    # world size, since each rank's share of the epoch grows.
                    # The epoch is over; fall through so the scheduler and the
                    # weight-decay ramp still advance for it.
                    logger.warning(
                        "Epoch %s had %s micro-batches to skip but only %s exist at this "
                        "world size; treating it as complete and continuing.",
                        epoch + 1, skip_batches, len(dataloader),
                    )
                    average_loss = float("nan")
                    average_kl = float("nan")
                    average_entropy = float("nan")
                    average_ce = float("nan")
                else:
                    # Both are per-rank sums over an identical batch count, so
                    # the mean of the means is the global mean.
                    average_loss = float(all_reduce_mean(total_loss / batches_seen, context))
                    average_kl = float(all_reduce_mean(total_kl / batches_seen, context))
                    average_entropy = float(
                        all_reduce_mean(total_entropy / batches_seen, context)
                    )
                    average_ce = float(all_reduce_mean(total_ce / batches_seen, context))
                    loss_history.append(average_loss)
                    kl_history.append(average_kl)
                epoch_seconds = time.perf_counter() - epoch_started

                if scheduler is not None:
                    scheduler.step()
                new_lr = optimizer.param_groups[0]["lr"]

                # Weight decay is cosine-scheduled 0.04 -> 0.4 in DINO. The
                # submitted constant 0.01 sat below even the schedule's starting
                # value.
                #
                # `apply_weight_decay` moves the DECAYED group only. Walking
                # every group here -- which is what the previous version did --
                # would push 0.4 onto the biases and LayerNorm gains that
                # `build_param_groups` deliberately excluded, so the exclusion
                # would survive exactly one epoch.
                if weight_decay_final is not None:
                    decay = cosine_value(
                        weight_decay_start, float(weight_decay_final), epoch + 1, epochs
                    )
                    apply_weight_decay(optimizer, decay)

                epoch_metrics = {
                    "loss": average_loss,
                    # The decomposition, at epoch resolution. `loss` is a cross
                    # entropy, so `loss = teacher_entropy_cross_view +
                    # teacher_student_kl` exactly, and only the second term is
                    # the student learning. Read THIS as the learning curve --
                    # on the shipped run the raw loss was flat from epoch 20
                    # while the KL kept falling to epoch 93.
                    "teacher_student_kl": average_kl,
                    "teacher_entropy_cross_view": average_entropy,
                    # `dino_cross_entropy == teacher_entropy_cross_view +
                    # teacher_student_kl` exactly. `loss` equals it too unless
                    # KoLeo or an auxiliary head is on, in which case `loss` is
                    # the full objective and this is its Eq. 1 term.
                    "dino_cross_entropy": average_ce,
                    "duration_seconds": epoch_seconds,
                    "batches": batches_seen * context.world_size,
                    "lr": new_lr,
                    "teacher_temp": criterion.teacher_temperature(epoch),
                    "images_per_second": (
                        batches_seen * int(cfg.data.batch_size) * context.world_size / epoch_seconds
                    ),
                    "loop_blocked_fraction": data_wait / epoch_seconds,
                    # Same number under its historical name; see the step metrics.
                    "data_wait_fraction": data_wait / epoch_seconds,
                }
                if busy_meter.enabled:
                    epoch_metrics["gpu_busy_fraction"] = busy_meter.drain() / epoch_seconds
                images_processed += batches_seen * int(cfg.data.batch_size) * context.world_size
                if device.type == "cuda":
                    epoch_metrics["peak_memory_mb"] = torch.cuda.max_memory_allocated() / 1024**2
                    epoch_metrics["peak_reserved_mb"] = torch.cuda.max_memory_reserved() / 1024**2
                    # Run-wide maxima, kept here because the next epoch resets
                    # the counters these are read from.
                    run_peak_allocated_gb = max(
                        run_peak_allocated_gb, epoch_metrics["peak_memory_mb"] / 1024.0
                    )
                    run_peak_reserved_gb = max(
                        run_peak_reserved_gb, epoch_metrics["peak_reserved_mb"] / 1024.0
                    )
                tracker.log_metrics(epoch_metrics, epoch + 1, prefix="epoch")
                logger.info(
                    "Epoch %s/%s | loss=%.5f | CE %.5f = KL %.5f + H %.5f | batches=%s "
                    "duration=%.2fs img/s=%.1f loop_blocked=%.1f%% peak_mem=%.0fMB",
                    epoch + 1, epochs, average_loss, average_ce, average_kl, average_entropy,
                    batches_seen, epoch_seconds,
                    epoch_metrics["images_per_second"],
                    epoch_metrics["loop_blocked_fraction"] * 100.0,
                    epoch_metrics.get("peak_memory_mb", 0.0),
                )

                if (
                    artifacts.log_parameter_histograms
                    and int(intervals.histogram_every_epochs) > 0
                    and (epoch + 1) % int(intervals.histogram_every_epochs) == 0
                ):
                    tracker.log_parameter_histograms(model, global_step)

                figure_every = int(
                    OmegaConf.select(cfg, "tracking.intervals.figure_every_epochs", default=5)
                )
                if figure_every > 0 and (epoch + 1) % figure_every == 0 and len(loss_history) > 1:
                    tracker.log_figure(
                        "pretrain/loss_curve",
                        plot_loss_curves(
                            {
                                # KL first so it owns the legend order: it is the
                                # learnable half and the raw cross entropy is
                                # mostly the teacher's entropy.
                                "KL(teacher||student)": kl_history,
                                "DINO loss (CE)": loss_history,
                            },
                            title="DINO pretraining: cross entropy and its learnable part",
                        ),
                        epoch + 1,
                    )

                if save_interval > 0 and (epoch + 1) % save_interval == 0:
                    checkpoint_file = save_dino_checkpoint(
                        model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch + 1,
                        checkpoint_manager=checkpoint_manager,
                        filename=f"dino_checkpoint_epoch_{epoch + 1:04d}.pth",
                        include_optimizer=save_full_checkpoints,
                        include_teacher=save_teacher,
                        rolling_prefix="dino_checkpoint_epoch_",
                    )
                    tracker.log_event("checkpoint", {"epoch": epoch + 1, "path": checkpoint_file})
                    logger.info("Saved interval checkpoint: %s", checkpoint_file)

                # Milestone checkpoints, written with NO rolling prefix so
                # `CheckpointManager.prune` never considers them. That is the
                # difference from the interval series above and the reason both
                # exist: "is 100 epochs of stage 1 better than 25" is answerable
                # only if the epoch-25 encoder is still on disk at the end.
                #
                # Each one is published as a bare backbone too, because that --
                # not the training state -- is what stage 2 consumes, and asking
                # a reader to re-derive it from the full checkpoint is how the
                # comparison never gets run.
                if (epoch + 1) in save_epochs:
                    milestone_file = save_dino_checkpoint(
                        model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch + 1,
                        checkpoint_manager=checkpoint_manager,
                        filename=f"dino_milestone_epoch_{epoch + 1:04d}.pth",
                        include_optimizer=save_full_checkpoints,
                        include_teacher=save_teacher,
                        rolling_prefix=None,
                    )
                    milestone_backbone = save_path / f"dino_backbone_epoch_{epoch + 1:04d}.pth"
                    if context.is_main:
                        torch.save(
                            to_cpu_state_dict(model.student_backbone.state_dict()),
                            milestone_backbone,
                        )
                    tracker.log_event(
                        "milestone_checkpoint",
                        {
                            "epoch": epoch + 1,
                            "path": milestone_file,
                            "backbone": str(milestone_backbone),
                        },
                    )
                    logger.info(
                        "Milestone checkpoint at epoch %s (kept permanently): %s | encoder: %s",
                        epoch + 1, milestone_file, milestone_backbone,
                    )

                # End-of-epoch resume point. `micro_step=0` means the next epoch
                # starts from its first batch with nothing to fast-forward.
                save_resume_checkpoint(
                    resume_manager,
                    f"{RESUME_PREFIX}{global_step:09d}.pth",
                    components=components,
                    progress=TrainingProgress(
                        epoch=epoch + 1,
                        global_step=global_step,
                        micro_step=0,
                        completed=epoch + 1 >= epochs,
                    ),
                    context=context,
                    cfg=cfg,
                    loader_generator=loader_generator,
                    extra={"effective_batch_size": effective_batch},
                )
                periodic.mark()

        if interrupted:
            logger.warning(
                "Run stopped early at step %s. Relaunch the identical command with "
                "experiment.training.resume=auto to continue.",
                global_step,
            )
            tracker.log_event(
                "training_interrupted",
                {"global_step": global_step, "duration_seconds": time.perf_counter() - training_started},
            )
            # The runtime half of the budget, for the segment that did run. On a
            # session-limited platform this is the *only* path that ever
            # executes, so skipping it here would mean the throughput and peak
            # VRAM of a Kaggle or vast.ai run were never reported at all. The
            # numbers describe this segment, not the whole schedule.
            budget.record_runtime(
                device,
                training_seconds=time.perf_counter() - training_started,
                images_processed=images_processed,
                peak_allocated_gb=run_peak_allocated_gb or None,
                peak_reserved_gb=run_peak_reserved_gb or None,
            )
            budget.notes.append(
                f"Interrupted at step {global_step} of epoch {epoch + 1}/{epochs}; the runtime "
                "figures cover this segment only."
            )
            if budget_enabled and context.is_main:
                logger.info("Stage-1 budget (interrupted segment)\n%s", budget.format_table())
                tracker.log_event("stage1_budget", {"phase": "interrupted", **budget.as_dict()})
            if context.is_main:
                # A summary for the segment that ran, flagged as incomplete. On a
                # preemptible platform this is the only path that ever executes,
                # so skipping it would leave the arm with no machine-readable
                # trace at all -- and `run_stage1_ablations.py` reads exactly
                # this file to decide whether an arm produced anything.
                write_stage1_summary(
                    save_path,
                    cfg,
                    criterion=criterion,
                    model=model,
                    transform=transform,
                    fingerprint=fingerprint,
                    raw_coverage=coverage,
                    parameters=parameters,
                    budget=budget,
                    dynamics={
                        "final_loss": loss_history[-1] if loss_history else float("nan"),
                        "final_teacher_student_kl": kl_history[-1] if kl_history else float("nan"),
                        "epochs_completed": float(len(loss_history)),
                        "global_step": float(global_step),
                    },
                    runtime={
                        "amp": amp.label,
                        "device": str(device),
                        "completed": False,
                        "interrupted_at_step": global_step,
                        "note": (
                            "Interrupted run. Every figure here describes the segment that "
                            "executed, not the configured schedule."
                        ),
                    },
                    artifacts={"events": str(Path(output_dir) / "events.jsonl")},
                    context=context,
                    logger=logger,
                )
            return

        # ------------------------------------------------------- final saves
        # Rank 0 alone writes the artifacts; the barrier keeps the others from
        # tearing down the process group while it is still writing.
        final_file = save_dino_checkpoint(
            model=model, optimizer=optimizer, scheduler=scheduler, epoch=epochs,
            checkpoint_manager=checkpoint_manager, filename="dino_pretrained_final.pth",
            include_optimizer=save_full_checkpoints, include_teacher=save_teacher,
        )
        backbone_file = save_path / "dino_pretrained_backbone.pth"
        if context.is_main:
            # The bare backbone state dict is the handoff to stage 2.
            torch.save(to_cpu_state_dict(model.student_backbone.state_dict()), backbone_file)

            # Publish the same weights at the shared, stage-independent path that
            # every downstream run reads. The ablation and baseline suites compare
            # architectures, so they must all start from *one* set of encoder
            # weights; a per-run pretraining stage would make each variant's
            # result partly a function of its own self-supervised seed.
            shared_file = publish_shared_backbone(cfg, backbone_file, logger)
            if shared_file is not None:
                tracker.log_event("shared_backbone", {"path": str(shared_file)})

            if loss_history:
                tracker.log_figure(
                    "pretrain/loss_curve",
                    plot_loss_curves(
                        {
                            "KL(teacher||student)": kl_history,
                            "DINO loss (CE)": loss_history,
                        },
                        title="DINO pretraining: cross entropy and its learnable part",
                    ),
                    epochs,
                )
            tracker.log_artifact(backbone_file, name="dino_pretrained_backbone", artifact_type="model")
        barrier(context)

        # ---------------------------------------------------- run artifacts
        # `summary.json` is written before the budget's runtime half is folded
        # in below, so the wall-clock figures are added by a second write. One
        # file, two writes, rather than a partial file if the second half fails.

        total_seconds = time.perf_counter() - training_started
        tracker.log_event(
            "training_complete",
            {
                "duration_seconds": total_seconds,
                "checkpoint": final_file,
                "student_backbone": str(backbone_file),
                "amp": amp.label,
                "effective_batch_size": effective_batch,
                "world_size": context.world_size,
                **{f"runtime_{key}": value for key, value in runtime.items()},
            },
        )

        # The budget again, now with the runtime half measured. Printed as one
        # block so it can be pasted into the paper's cost table without being
        # reassembled from a hundred log lines.
        budget.record_runtime(
            device,
            training_seconds=total_seconds,
            images_processed=images_processed,
            peak_allocated_gb=run_peak_allocated_gb or None,
            peak_reserved_gb=run_peak_reserved_gb or None,
        )
        if budget_enabled and context.is_main:
            logger.info("Stage-1 budget (final)\n%s", budget.format_table())
            tracker.log_metrics(budget.as_metrics(), step=global_step)
            tracker.log_event("stage1_budget", {"phase": "final", **budget.as_dict()})

        if context.is_main:
            summary_path = write_stage1_summary(
                save_path,
                cfg,
                criterion=criterion,
                model=model,
                transform=transform,
                fingerprint=fingerprint,
                raw_coverage=coverage,
                parameters=parameters,
                budget=budget,
                dynamics={
                    "final_loss": loss_history[-1] if loss_history else float("nan"),
                    "final_teacher_student_kl": kl_history[-1] if kl_history else float("nan"),
                    "initial_teacher_student_kl": kl_history[0] if kl_history else float("nan"),
                    "min_teacher_student_kl": min(kl_history) if kl_history else float("nan"),
                    "initial_loss": loss_history[0] if loss_history else float("nan"),
                    "teacher_entropy_floor": entropy_min,
                    "teacher_entropy_ceiling": entropy_max,
                    "epochs_completed": float(epochs),
                    "global_step": float(global_step),
                    "wall_clock_seconds": total_seconds,
                    "peak_allocated_gb": run_peak_allocated_gb,
                    "learning_rate": learning_rate,
                },
                runtime={
                    "amp": amp.label,
                    "device": str(device),
                    "num_workers": num_workers,
                    "wall_clock_seconds": total_seconds,
                    "peak_allocated_gb": run_peak_allocated_gb,
                    "peak_reserved_gb": run_peak_reserved_gb,
                    **{f"runtime_{key}": value for key, value in runtime.items()},
                    **distributed_runtime,
                    "learning_rate_provenance": lr_provenance,
                },
                artifacts={
                    "final_checkpoint": str(final_file),
                    "student_backbone": str(backbone_file),
                    "events": str(Path(output_dir) / "events.jsonl"),
                },
                context=context,
                logger=logger,
            )
            tracker.log_event("stage1_summary", {"path": summary_path})

        logger.info(
            "Pretraining complete in %.2fs. Final: %s. Backbone for stage 2: %s",
            total_seconds, final_file, backbone_file,
        )

    except Exception:
        logger.exception("DINO pretraining failed.")
        tracker.log_event("exception", {"stage": "dino_pretraining", "rank": context.rank})
        raise
    finally:
        tracker.log_event("training_end", {"duration_seconds": time.perf_counter() - training_started})
        tracker.close()
        # No barrier here: this block also runs when one rank has raised, and a
        # barrier would replace the traceback with a collective timeout.
        shutdown_distributed(context)


if __name__ == "__main__":
    main()
