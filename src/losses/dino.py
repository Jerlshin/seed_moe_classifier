"""Self-distillation loss for stage 1 (paper Section 4, Eqs. 1-3 and Algorithm 1).

Cross-view objective (Eq. 1)::

    L_DINO = -(1/N) sum_i sum_{v != q} q_v . log p_v

``q_v`` is the teacher's soft target for view ``v``, ``p_v`` the student's
prediction. Only *cross-view* pairs contribute: a student view is never scored
against the teacher's output for that same view, which is what forces the
representation to be invariant to the augmentation rather than to memorise it.

Naming
------

This module implements **DINO (Caron et al., 2021) with a SwinV2 trunk**, not
DINOv2. DINOv2 is defined by four additions on top of DINO -- an iBOT
patch-level masked-prediction objective, the KoLeo regulariser, Sinkhorn-Knopp
centering in place of softmax-with-EMA-centering, and untied image/patch head
weights. The revision implements the two that do not require patch tokens
(:func:`koleo_regularizer`, :func:`sinkhorn_knopp`) and does **not** implement
iBOT or untied heads, so the honest description of stage 1 is "DINO-style
self-distillation with the KoLeo and Sinkhorn components of DINOv2". The class,
the checkpoint name and the paper text all say that now.

Collapse control
----------------

DINO prevents collapse by balancing two opposing forces: **sharpening** (a low
teacher temperature pulls targets toward one-hot) and **centering** (subtracting
a running mean pushes them toward uniform). The submitted configuration set both
in the collapsing direction at once:

* Sharpening was ~2x stronger than reference -- ``0.02 -> 0.04`` against DINO's
  ``0.04 -> 0.07``, i.e. a converged teacher roughly twice as sharp as DINO's.
* Centering was noise-dominated. ``C`` lives in ``R^65536`` and is an EMA at
  ``m = 0.9`` (an effective window of ~10 steps) over ``2 global crops x batch
  16 = 32`` teacher vectors per step. That is ~320 effective samples for 65,536
  dimensions -- **0.005 samples per estimated dimension**, against DINO's 0.31
  at batch 1024. The counterweight to sharpening was essentially noise.

Both are corrected in ``conf/model/loss/dino.yaml`` and
``conf/model/head/dino.yaml``. ``centering="sinkhorn"`` removes the dependence on
that running mean entirely: Sinkhorn-Knopp normalises the teacher's assignment to
be doubly stochastic **within the batch**, so nothing has to be estimated across
steps.

A partially collapsed run has a perfectly plausible-looking loss curve, so the
loss figure will not reveal any of this. :meth:`CustomDINOLoss.collapse_metrics`
reports the diagnostics that will: teacher output entropy, the KL between the
teacher's mean assignment and uniform, and how many prototypes the batch
effectively used.

**Entropy is a conditional quantity, not a score.** It scales with ``log K``, so
it is not comparable across prototype counts, and under Sinkhorn centering it
cannot fall below ``log(K / B_teacher)`` no matter what the model does -- 3.47 of
a 7.62 maximum at the configured ``K = 2048`` and 64 teacher views. The metrics
therefore ship with their own bounds (:meth:`CustomDINOLoss.entropy_bounds`), a
normalised form, and ``K`` and ``B_teacher`` alongside, so a number in a log line
can be read without the config next to it. Nothing here labels high entropy
"good" or low entropy "collapse"; both readings are wrong at the edges.

KoLeo is per view, not across views
-----------------------------------

The KoLeo regulariser is a **uniformity** term over distinct instances. Applied
to the two global views concatenated -- which is what this repository shipped
until the stage-1 audit -- the two augmented views of one crop become each
other's nearest neighbour, so the term's gradient pushes them apart. That is an
explicit anti-alignment force acting on precisely the pair Eq. 1 exists to pull
together, and on the shipped 100-epoch run it coincided with ``alignment``
getting *worse* (0.638 -> 1.111) and ``same_image_minus_same_class`` landing at
-0.028.

:func:`grouped_koleo` chunks by view, as DINOv2's ``ssl_meta_arch.py`` does
(*"we don't apply koleo loss between cls tokens of a same image"*).
``model.loss.koleo_scope=all_views`` restores the old behaviour as a control, and
``model.loss.lambda_koleo=0`` is the control that says whether the term is worth
anything here at all.

The loss is 95 % target entropy, so it is logged decomposed
------------------------------------------------------------

``CE(q, p) = H(q) + KL(q || p)``, and under Sinkhorn centering ``H(q)`` is a
property of the normaliser, ``K``, ``B_teacher`` and the temperature schedule
rather than of the student. Measured on the shipped run: the reported loss fell
7.765 -> 5.671 and **80.0 % of that was ``H(q)`` falling**; the final loss is
94.8 % irreducible target entropy; and ``KL(q || p)`` -- the only learnable part
-- was still improving at epoch 93 while the raw curve had been flat since epoch
20. :meth:`CustomDINOLoss.compute_dino_loss` therefore always emits
``dino_cross_entropy``, ``teacher_entropy_cross_view`` and
``teacher_student_kl`` as device tensors, and the trainer logs and epoch-averages
all three. **Read ``train/teacher_student_kl``, not ``train/loss``.**

Two things this module does *not* do any more
---------------------------------------------

**It does not recompute the student's log-softmax twelve times.** Eq. 1 pairs 2
teacher views against 6 student views. Written as a double loop, the inner
``F.log_softmax(student_logits)`` is evaluated once per *pair*, so each student
view's softmax over ``out_dim`` prototypes is computed twice and back-propagated
twice.
:meth:`CustomDINOLoss.compute_dino_loss` now takes one log-softmax over the whole
``[6B, out_dim]`` block and contracts every pair in a single ``einsum``. Same
arithmetic, ~6x less of it, and one kernel where there were ~36.

**It does not synchronise with the GPU on every step.** The diagnostics used to
be stored as Python ``float``s, and every ``float(tensor)`` is a
``cudaStreamSynchronize``: the CPU stops and waits for the entire queued backward
to drain before it can enqueue the next batch. Four of them per micro-batch --
teacher entropy, its max, the KL, KoLeo -- is four full pipeline stalls per
micro-batch, for numbers the trainer logs once every ten *steps*. They are kept
as device tensors now, computation is gated on
:attr:`CustomDINOLoss.metrics_enabled`, and :attr:`CustomDINOLoss.last_metrics`
converts on access -- which the trainer only does when it is about to log.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.utils.training.distributed import DistributedContext

CENTERING_MODES = ("ema", "sinkhorn")

#: How :func:`koleo_regularizer` is applied across the student's global views.
#:
#: ``per_view`` (the default, and the DINOv2 reference behaviour) computes the
#: nearest-neighbour term **inside each global view's block separately**;
#: ``all_views`` computes it over the two blocks concatenated, which is what this
#: repository shipped before the stage-1 audit and is retained only as the
#: control that measures what the change was worth. See :func:`koleo_regularizer`
#: for why the difference is not cosmetic.
KOLEO_SCOPES = ("per_view", "all_views")

#: Reduction over the per-view KoLeo terms under ``koleo_scope="per_view"``.
#:
#: ``mean`` keeps ``lambda_koleo`` on the same scale as ``all_views``, so the two
#: are a single-factor comparison. ``sum`` reproduces DINOv2's
#: ``sum(self.koleo_loss(p) for p in student_cls_tokens.chunk(2))``, which at two
#: global views is the identical term at twice the weight.
KOLEO_REDUCTIONS = ("mean", "sum")


def _at_least_float32(tensor: torch.Tensor) -> torch.Tensor:
    """Promote half precision to fp32, and leave anything wider alone.

    Every numerically delicate step in this module -- the Sinkhorn log-space
    normaliser, the log-softmax over the prototypes, the KoLeo pairwise
    distances -- needs at least fp32, and under autocast the inputs arrive as
    bf16 or fp16. A bare ``.float()`` would do that, but it would also silently
    *downcast* an fp64 input, which is exactly what a numerical test written in
    double precision would be trying to rule out.
    """
    return tensor.to(torch.promote_types(tensor.dtype, torch.float32))


def koleo_regularizer(features: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Kozachenko-Leonenko differential-entropy regulariser (DINOv2).

        L_koleo = -(1/n) sum_i log( min_{j != i} ||z_i - z_j|| )

    Encourages a uniform span of the feature sphere within a batch, which stops
    distinct-but-similar samples collapsing onto each other. That is precisely
    the failure mode a *fine-grained* task cannot tolerate: 27 sub-varieties of
    the same four crops are near-duplicates by construction. DINOv2's ablation
    credits KoLeo with >8 % on instance retrieval at no cost elsewhere.

    The distance matrix stays :func:`torch.cdist` rather than the faster
    ``2 - 2 * z z^T`` identity, and that is deliberate. The Gram form loses
    catastrophic precision exactly where this loss does its work: for two nearly
    identical unit vectors, ``2 - 2cos`` subtracts two numbers that agree to
    within float epsilon, and the result can come out **negative**, get clamped,
    and hand the optimiser ``-log(eps) ~ 18`` as the gradient signal for a pair
    that should have contributed almost nothing. The matrix is ``[2B, 2B]`` --
    32 x 32 here -- so its cost is not measurable either way; there is nothing to
    buy by being clever.

    Args:
        features: ``[batch, dim]``. L2-normalised internally, so pass the
            bottleneck embedding rather than the wide prototype logits.
            Always evaluated in fp32: under fp16 autocast the squared distances
            of near-duplicate crops underflow to exactly zero.
    """
    if features.ndim != 2 or features.shape[0] < 2:
        return features.new_zeros(())
    normalized = F.normalize(_at_least_float32(features), dim=-1, eps=epsilon)
    distances = torch.cdist(normalized, normalized, p=2)
    # Exclude the self-distance without perturbing any real pair. `masked_fill`
    # keeps this out-of-place (the old `+ eye * 1e6` allocated a fresh identity
    # matrix on every call) and uses `inf`, which cannot be mistaken for a real
    # distance the way a finite sentinel can.
    diagonal = torch.eye(distances.shape[0], device=distances.device, dtype=torch.bool)
    nearest = distances.masked_fill(diagonal, float("inf")).amin(dim=-1)
    return -torch.log(nearest.clamp_min(epsilon)).mean()


def grouped_koleo(
    features: torch.Tensor,
    num_groups: int,
    scope: str = "per_view",
    reduction: str = "mean",
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """KoLeo over ``num_groups`` view-major blocks, applied per block or across all.

    **The per-block default is a correctness fix, not a cosmetic one.**
    ``features`` arrives view-major, so rows ``[0:B]`` are global view 0 of every
    image and rows ``[B:2B]`` are global view 1 of *the same* images in the same
    order. Handing that whole block to :func:`koleo_regularizer` masks only the
    diagonal, so row ``i`` and row ``B + i`` -- two augmented views of one crop --
    are eligible neighbours of each other. Under a working DINO they are the
    *closest* pair in the block, so they are the argmin, and the term's gradient
    pushes them apart: an explicit **anti-alignment** force acting on exactly the
    pair Eq. 1 exists to pull together.

    DINOv2's reference does
    ``sum(self.koleo_loss(p) for p in student_cls_tokens.chunk(2))`` with the
    in-source comment *"we don't apply koleo loss between cls tokens of a same
    image"*. ``scope="per_view"`` is that behaviour.

    ``scope="all_views"`` restores the pre-audit behaviour. It is kept because
    "the fix was worth X" is only a measurable claim if the unfixed version is
    still runnable at a flag, not because either is a tuning knob.

    Args:
        features: ``[num_groups * B, dim]``, **view-major**. Any other layout
            makes the grouping meaningless, which is why the trainer passes the
            leading ``num_global_crops * B`` rows of the view-major bottleneck
            and nothing else.
        num_groups: Number of equal view blocks in ``features``.
        scope: ``"per_view"`` or ``"all_views"``; see :data:`KOLEO_SCOPES`.
        reduction: ``"mean"`` or ``"sum"`` over the per-view terms; see
            :data:`KOLEO_REDUCTIONS`. Ignored under ``all_views``.
        epsilon: Floor on the nearest-neighbour distance before the log.

    Returns:
        A scalar tensor. Zero when there are fewer than two rows per block, which
        is the degenerate case a batch of 1 produces.
    """
    if scope not in KOLEO_SCOPES:
        raise ValueError(f"koleo_scope must be one of {KOLEO_SCOPES}, got {scope!r}")
    if reduction not in KOLEO_REDUCTIONS:
        raise ValueError(f"koleo_reduction must be one of {KOLEO_REDUCTIONS}, got {reduction!r}")
    if features.ndim != 2 or features.shape[0] < 2:
        return features.new_zeros(())

    groups = max(int(num_groups), 1)
    if scope == "all_views" or groups <= 1:
        return koleo_regularizer(features, epsilon=epsilon)
    if features.shape[0] % groups != 0:
        raise ValueError(
            f"KoLeo received {features.shape[0]} rows, which is not divisible by "
            f"num_groups={groups}; the global views must be stacked view-major."
        )

    terms = [koleo_regularizer(block, epsilon=epsilon) for block in features.chunk(groups, dim=0)]
    total = torch.stack(terms).sum()
    return total if reduction == "sum" else total / float(len(terms))


@torch.no_grad()
def sinkhorn_knopp(
    logits: torch.Tensor,
    temperature: float = 0.05,
    iterations: int = 3,
    context: "DistributedContext | None" = None,
) -> torch.Tensor:
    """Doubly-stochastic assignment over a batch (SwaV/DINOv2 centering).

    Alternately normalises prototype and sample marginals, giving an assignment
    matrix whose prototype marginals are uniform **by construction within the
    batch**. Unlike subtracting an EMA centre, nothing is estimated across steps,
    so it does not degrade when the batch is small relative to the prototype
    count -- which is the exact regime this dataset is in.

    Computed entirely in **log space** with ``logsumexp``. The direct form
    ``exp(logits / temperature)`` overflows immediately at the temperatures this
    is used at: unit-scale logits at ``tau = 0.04`` reach ``exp(25)``, and a
    single ``inf`` turns the whole assignment into ``nan`` -- silently, since the
    result is detached and only shows up later as a ``nan`` loss.

    Args:
        logits: Teacher outputs, ``[batch, out_dim]``. Under DDP this is the
            **local** shard.
        temperature: Sharpening temperature applied before normalisation.
        iterations: Sinkhorn iterations (3 is the reference value).
        context: Pass an enabled :class:`~src.utils.training.distributed.\
DistributedContext` to normalise over the batch **concatenated across ranks**
            instead of the local shard.

            Which one is wanted is a real choice, not an implementation detail.
            The two axes of the normaliser behave differently under sharding: the
            *sample* marginal (``dim=0``, over prototypes) is per-column and
            therefore already local, while the *prototype* marginal (``dim=1``,
            over the batch) reduces along the axis that was split, so it needs
            :func:`~src.utils.training.distributed.logsumexp_across_ranks` to
            mean what it means on one GPU. With ``context`` supplied, the result
            is equal to running this function on one process over the
            concatenated batch, up to floating-point reduction order --
            ``tests/test_distributed.py`` pins that.

            Left ``None`` (the default), each rank normalises its own shard, and
            *that* is what reproduces the single-GPU numbers: this function is
            already applied per micro-batch under gradient accumulation, so a
            per-rank shard of the same size computes the identical function to
            the one a single-GPU run computes on its micro-batch.

    Returns:
        ``[batch, out_dim]`` in **fp32**, whatever dtype came in. Under bf16
        autocast the teacher logits arrive with ~3 decimal digits of mantissa;
        dividing them by 0.04 and exponentiating a doubly-stochastic normaliser
        in that precision is not something to do to a training target, and the
        cast is free next to the backbone forward that produced them.
    """
    from src.utils.training.distributed import (
        all_reduce_sum,
        logsumexp_across_ranks,
    )

    distributed = context is not None and context.enabled
    log_assignments = (_at_least_float32(logits) / float(temperature)).t()  # [out_dim, batch]
    num_prototypes, local_samples = log_assignments.shape

    if distributed:
        # Counted rather than assumed to be `local * world_size`: the marginal
        # this divides by is what makes the assignment doubly stochastic, and an
        # uneven final batch would otherwise scale it wrongly on every rank.
        counter = torch.tensor([float(local_samples)], device=log_assignments.device, dtype=torch.float64)
        num_samples = int(all_reduce_sum(counter, context).item())
        total = logsumexp_across_ranks(log_assignments.reshape(-1), context, dim=0)
    else:
        num_samples = local_samples
        total = torch.logsumexp(log_assignments.reshape(-1), dim=0)

    log_assignments = log_assignments - total

    for _ in range(max(int(iterations), 1)):
        # Prototype marginals -> 1 / num_prototypes. Reduces along the batch
        # axis, which is the sharded one.
        prototype_marginal = (
            logsumexp_across_ranks(log_assignments, context, dim=1, keepdim=True)
            if distributed
            else torch.logsumexp(log_assignments, dim=1, keepdim=True)
        )
        log_assignments = log_assignments - prototype_marginal - math.log(num_prototypes)
        # Sample marginals -> 1 / num_samples. Reduces along the prototype axis,
        # which every rank holds in full, so this stays local either way.
        log_assignments = (
            log_assignments
            - torch.logsumexp(log_assignments, dim=0, keepdim=True)
            - math.log(num_samples)
        )

    # Rescale so each sample's row sums to 1, matching the softmax path's contract.
    log_assignments = log_assignments + math.log(num_samples)
    return log_assignments.t().exp()


class CustomDINOLoss(nn.Module):
    """Self-distillation loss with teacher temperature warmup and centering.

    Args:
        out_dim: Width of the projection head output.
        num_crops: Total student views per image (paper: 2 global + 4 local = 6).
        warmup_teacher_temp: Teacher temperature at epoch 0.
        teacher_temp: Teacher temperature after warmup.
        warmup_teacher_temp_epochs: Warmup length in epochs.
        num_epochs: Total training epochs, used to size the schedule.
        student_temp: Fixed student temperature.
        center_momentum: ``m`` in Eq. 3. Only used by ``centering="ema"``.
        num_global_crops: Views the teacher sees (paper: 2).
        centering: ``"sinkhorn"`` (default) or ``"ema"``; see the module
            docstring.
        sinkhorn_iterations: Iterations for the Sinkhorn variant.
        lambda_koleo: Weight on the KoLeo regulariser. ``0`` disables it, which
            is the control that says whether KoLeo is worth anything here at all.
        koleo_scope: ``"per_view"`` (default) applies KoLeo **inside each global
            view's block**, as DINOv2 does; ``"all_views"`` applies it over the
            concatenated blocks, which makes the two views of one image each
            other's nearest neighbour and turns the term into an anti-alignment
            force. See :func:`grouped_koleo`.
        koleo_reduction: ``"mean"`` (default) or ``"sum"`` over the per-view
            terms. Mean keeps ``lambda_koleo`` on the same scale as
            ``all_views``, so the two are a single-factor comparison.
        context: This rank's place in a distributed job, or ``None`` for a
            single process. Used for two things and no others:

            * ``centering="ema"`` -- the centre is a **shared** running buffer
              that the teacher's targets subtract, so :meth:`update_center`
              all-reduces unconditionally. Per-rank centres would leave the
              ranks training against different targets, and the checkpoint's
              contents would depend on which rank wrote it.
            * ``distributed_sinkhorn`` -- see below.
        distributed_sinkhorn: Normalise the Sinkhorn assignment over the batch
            concatenated across ranks rather than over each rank's own shard.

            Default ``False``, which is the setting that **preserves the
            single-GPU numbers**: Sinkhorn is already applied per micro-batch
            under gradient accumulation, so a per-rank shard of the same size is
            the same function of the same amount of data. Turning it on makes
            the assignment doubly stochastic over the whole global step batch --
            defensible, arguably better, and a different objective, which is why
            it is opt-in rather than implied by launching on two GPUs.

            KoLeo is deliberately **not** given the same switch. It is a
            nearest-neighbour statistic over the local view block, applied
            per-rank in the reference DINOv2 implementation for the same reason:
            per-rank matches what a single GPU computes per micro-batch.
    """

    def __init__(
        self,
        out_dim: int,
        num_crops: int,
        warmup_teacher_temp: float,
        teacher_temp: float,
        warmup_teacher_temp_epochs: int,
        num_epochs: int,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
        num_global_crops: int = 2,
        centering: str = "sinkhorn",
        sinkhorn_iterations: int = 3,
        lambda_koleo: float = 0.1,
        koleo_scope: str = "per_view",
        koleo_reduction: str = "mean",
        context: "DistributedContext | None" = None,
        distributed_sinkhorn: bool = False,
    ):
        super().__init__()
        if num_crops < 2:
            raise ValueError(f"num_crops must be >= 2 to form cross-view pairs, got {num_crops}")
        if not 1 <= num_global_crops <= num_crops:
            raise ValueError(f"num_global_crops must be in [1, {num_crops}], got {num_global_crops}")
        if centering not in CENTERING_MODES:
            raise ValueError(f"centering must be one of {CENTERING_MODES}, got {centering!r}")
        if koleo_scope not in KOLEO_SCOPES:
            raise ValueError(f"koleo_scope must be one of {KOLEO_SCOPES}, got {koleo_scope!r}")
        if koleo_reduction not in KOLEO_REDUCTIONS:
            raise ValueError(
                f"koleo_reduction must be one of {KOLEO_REDUCTIONS}, got {koleo_reduction!r}"
            )

        self.student_temp = float(student_temp)
        self.center_momentum = float(center_momentum)
        self.num_crops = int(num_crops)
        self.num_global_crops = int(num_global_crops)
        self.centering = centering
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.lambda_koleo = float(lambda_koleo)
        self.koleo_scope = koleo_scope
        self.koleo_reduction = koleo_reduction
        self.register_buffer("center", torch.zeros(1, out_dim))

        #: Plain attributes, not buffers: the process group is not model state
        #: and must never reach a checkpoint, where it would pin a run to the
        #: topology that produced it.
        self.context = context
        self.distributed_sinkhorn = bool(distributed_sinkhorn)

        #: Device-resident diagnostics from the most recent :meth:`forward`.
        #: Read through :attr:`last_metrics`, which is what actually pays the
        #: synchronisation cost of moving them to the host.
        self._metric_tensors: dict[str, torch.Tensor] = {}
        self._metric_scalars: dict[str, float] = {}
        #: When False, :meth:`collapse_metrics` is skipped entirely. The trainer
        #: turns it on only for the steps it is about to log.
        self.metrics_enabled: bool = True
        #: Cached cross-view mask, keyed by the view-id pairing and device.
        self._pair_mask_key: tuple | None = None
        self._pair_mask: torch.Tensor | None = None

        # Eq. 2: linear ramp then constant.
        warmup_epochs = max(min(int(warmup_teacher_temp_epochs), int(num_epochs)), 0)
        steady_epochs = max(int(num_epochs) - warmup_epochs, 0)
        self.teacher_temp_schedule = np.concatenate(
            (
                np.linspace(warmup_teacher_temp, teacher_temp, warmup_epochs),
                np.full(steady_epochs, teacher_temp, dtype=float),
            )
        )
        if self.teacher_temp_schedule.size == 0:
            self.teacher_temp_schedule = np.array([teacher_temp], dtype=float)

    def teacher_temperature(self, epoch: int) -> float:
        """Teacher temperature for ``epoch`` under the Eq. 2 schedule."""
        index = min(max(int(epoch), 0), len(self.teacher_temp_schedule) - 1)
        return float(self.teacher_temp_schedule[index])

    @property
    def last_metric_tensors(self) -> dict[str, torch.Tensor]:
        """Diagnostics from the last forward, still on the device.

        Reading this does **not** synchronise, which is the whole point: the
        trainer accumulates ``teacher_student_kl`` and ``dino_cross_entropy``
        into device-resident epoch sums and converts once, at the epoch boundary.
        Use :attr:`last_metrics` when host floats are actually wanted.
        """
        return dict(self._metric_tensors)

    @property
    def last_metrics(self) -> dict[str, float]:
        """Diagnostics from the last forward, as host floats.

        **Accessing this synchronises with the GPU.** Every value is a 0-d device
        tensor until something asks for it, so reading this property inside the
        training loop reintroduces exactly the stall the tensors exist to avoid.
        Read it on logging steps only.
        """
        metrics = dict(self._metric_scalars)
        metrics.update({name: float(value) for name, value in self._metric_tensors.items()})
        return metrics

    def forward(
        self,
        student_output: torch.Tensor | Sequence[torch.Tensor],
        teacher_output: torch.Tensor | Sequence[torch.Tensor],
        epoch: int,
        student_view_ids: Sequence[int] | None = None,
        teacher_view_ids: Sequence[int] | None = None,
        student_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the mean cross-view loss and update the centering buffer.

        Args:
            student_output: ``num_crops`` tensors of shape ``[batch, out_dim]``,
                or a single pre-concatenated tensor.
            teacher_output: ``num_global_crops`` tensors, same convention.
            epoch: Zero-based epoch index, used to pick the teacher temperature.
            student_view_ids / teacher_view_ids: Explicit identifiers naming
                which augmented view each chunk came from. The submitted code
                skipped same-view pairs with ``if student_index == teacher_index``,
                which is only correct while the student's first two views happen
                to be the two globals in the teacher's order -- an invariant
                nothing enforced. Passing identifiers makes the pairing explicit;
                omitting them falls back to positional matching.
            student_embeddings: Pre-prototype bottleneck features for the KoLeo
                term. Omitted, the term is skipped.
        """
        student_output = self._concat_outputs(student_output)
        teacher_output = self._concat_outputs(teacher_output)
        loss = self.compute_dino_loss(
            student_output,
            teacher_output,
            epoch,
            student_view_ids=student_view_ids,
            teacher_view_ids=teacher_view_ids,
        )

        if self.lambda_koleo > 0.0 and student_embeddings is not None:
            koleo = grouped_koleo(
                student_embeddings,
                num_groups=self.num_global_crops,
                scope=self.koleo_scope,
                reduction=self.koleo_reduction,
            )
            # Always a device tensor, never a host float: `last_metrics` pays the
            # synchronisation only when the trainer is about to log.
            self._metric_tensors["koleo"] = koleo.detach()
            loss = loss + self.lambda_koleo * koleo

        if self.centering == "ema":
            self.update_center(teacher_output)
        return loss

    def _cross_view_mask(
        self,
        teacher_ids: Sequence[int],
        student_ids: Sequence[int],
        device: torch.device,
    ) -> torch.Tensor:
        """``[T, V]`` float mask that is 1 on cross-view pairs and 0 elsewhere.

        Cached: the view ids are fixed for the whole run, so rebuilding this
        every step would add a host-to-device copy per micro-batch for a tensor
        of 12 elements.
        """
        key = (tuple(teacher_ids), tuple(student_ids), str(device))
        if self._pair_mask_key != key or self._pair_mask is None:
            mask = torch.tensor(
                [[0.0 if s == t else 1.0 for s in student_ids] for t in teacher_ids],
                device=device,
                dtype=torch.float32,
            )
            self._pair_mask_key = key
            self._pair_mask = mask
        return self._pair_mask

    def teacher_targets(self, teacher_output: torch.Tensor, epoch: int) -> torch.Tensor:
        """Sharpened, centred teacher distribution over prototypes."""
        teacher_temp = self.teacher_temperature(epoch)
        if self.centering == "sinkhorn":
            return sinkhorn_knopp(
                teacher_output,
                temperature=teacher_temp,
                iterations=self.sinkhorn_iterations,
                context=self.context if self.distributed_sinkhorn else None,
            )
        center = self.center.to(device=teacher_output.device, dtype=teacher_output.dtype)
        return F.softmax((teacher_output - center) / teacher_temp, dim=-1)

    def compute_dino_loss(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        epoch: int,
        student_view_ids: Sequence[int] | None = None,
        teacher_view_ids: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Eq. 1 over every cross-view (teacher view, student view) pair.

        Computed as a single contraction rather than a double loop. With
        ``P = log_softmax(student / tau_s)`` reshaped to ``[V, B, D]`` and the
        teacher targets to ``[T, B, D]``, the per-pair cross entropy is

            CE[t, v] = mean_b sum_d -Q[t, b, d] * P[v, b, d]

        which is one ``einsum`` -- a batched matmul over the prototype axis --
        followed by a masked mean over the ``T x V`` pairs. Identical value and
        identical gradient to the loop it replaces (``tests/test_losses.py``
        checks it against a hand-written double loop), but the student's
        log-softmax over the prototypes is evaluated once per view instead of
        once per pair.
        """
        student_ids = list(student_view_ids) if student_view_ids is not None else list(range(self.num_crops))
        teacher_ids = (
            list(teacher_view_ids) if teacher_view_ids is not None else list(range(self.num_global_crops))
        )
        if len(student_ids) != self.num_crops or len(teacher_ids) != self.num_global_crops:
            raise ValueError(
                f"view id counts must match the crop counts: got {len(student_ids)} student ids for "
                f"{self.num_crops} crops and {len(teacher_ids)} teacher ids for {self.num_global_crops}"
            )
        if student_output.shape[0] % self.num_crops != 0:
            raise ValueError(
                f"student output has {student_output.shape[0]} rows, which is not divisible by "
                f"num_crops={self.num_crops}; the views must be stacked view-major."
            )
        if teacher_output.shape[0] % self.num_global_crops != 0:
            raise ValueError(
                f"teacher output has {teacher_output.shape[0]} rows, which is not divisible by "
                f"num_global_crops={self.num_global_crops}."
            )

        dim = student_output.shape[-1]
        batch = student_output.shape[0] // self.num_crops

        # fp32 throughout: `student_temp` is 0.1, so the logits are multiplied by
        # 10 before a softmax over `out_dim` classes. Autocast already promotes
        # log_softmax, but the division ahead of it happens in whatever dtype
        # arrives, and fp16 has ~3 decimal digits to lose there.
        student_log_probs = F.log_softmax(
            _at_least_float32(student_output) / self.student_temp, dim=-1
        ).view(self.num_crops, batch, dim)
        teacher_probs_all = _at_least_float32(self.teacher_targets(teacher_output, epoch).detach())
        teacher_probs = teacher_probs_all.view(self.num_global_crops, batch, dim)

        # [T, V]: the mean over the batch of each pair's cross entropy.
        pair_losses = -torch.einsum("tbd,vbd->tvb", teacher_probs, student_log_probs).mean(dim=-1)

        mask = self._cross_view_mask(teacher_ids, student_ids, pair_losses.device)
        # Counted on the host from the id lists, not as `mask.sum().item()` --
        # the latter would be a GPU synchronisation on every micro-batch to
        # recover a number that is constant for the entire run.
        loss_terms = sum(1 for t in teacher_ids for s in student_ids if s != t)
        if loss_terms == 0:
            raise RuntimeError(
                "DINO loss received no cross-view pairs; check num_crops, num_global_crops and the view ids."
            )

        cross_entropy = (pair_losses * mask).sum() / loss_terms

        # ------------------------------------------------------------------
        # The loss is not a learning curve, and this is what makes it readable.
        #
        # `CE(q, p) = H(q) + KL(q || p)`. Under Sinkhorn centering `H(q)` is set
        # by the normaliser, `K`, `B_teacher` and the temperature schedule --
        # none of which is the student learning anything. On the shipped
        # 100-epoch run, 80 % of the total loss drop was `H(q)` and the final
        # loss was 94.8 % irreducible target entropy, while the learnable part
        # was still falling at epoch 93 with the raw curve flat since epoch 20.
        #
        # Computed unconditionally rather than under `metrics_enabled`: these are
        # two reductions over an already-materialised `[T*B, K]` tensor, they
        # stay device tensors (no synchronisation), and the epoch-level mean has
        # to see every micro-batch to be an epoch mean at all. The heavier
        # prototype diagnostics below stay gated.
        #
        # `H` is weighted by each teacher view's share of the cross-view pairs,
        # so the decomposition is exact for any mask -- not only the balanced
        # 2 x 6 one. `teacher_entropy` keeps its historical definition (the plain
        # mean over every teacher row) so a number logged before this change and
        # one logged after mean the same thing; the two coincide whenever every
        # teacher view takes part in the same number of pairs.
        row_entropy = (
            -teacher_probs * teacher_probs.clamp_min(1e-12).log()
        ).sum(dim=-1).mean(dim=-1)  # [T]
        pair_entropy = (row_entropy.unsqueeze(1) * mask).sum() / loss_terms
        self._metric_tensors["dino_cross_entropy"] = cross_entropy.detach()
        self._metric_tensors["teacher_entropy_cross_view"] = pair_entropy.detach()
        self._metric_tensors["teacher_student_kl"] = (cross_entropy - pair_entropy).detach()

        if self.metrics_enabled:
            self._metric_tensors.update(self.collapse_metrics(teacher_probs_all))
            self._metric_scalars["cross_view_terms"] = float(loss_terms)
        return cross_entropy

    def entropy_bounds(self, teacher_batch: int) -> tuple[float, float]:
        """``(H_min, H_max)`` for one teacher row, given the batch it came from.

        Entropy is **not** interpretable on its own, and the reason is structural
        rather than statistical. ``H_max = log K`` is the usual ceiling. The floor
        is the part that gets misread: an *exactly* doubly-stochastic assignment
        gives every prototype column ``B_teacher / K`` of the mass, so no single
        row (mass 1) can concentrate on fewer than ``K / B_teacher`` prototypes:

            H_min = log(K / B_teacher)     [sinkhorn, B_teacher < K]

        At ``K = 2048`` and ``2 x 32 = 64`` teacher views that is **3.47 against a
        7.62 maximum** -- nearly half the nominal range is structurally
        unreachable. A run reporting ``H = 3.6`` there is close to the sharpest
        the normaliser permits, while the same 3.6 read against ``[0, log K]``
        looks like comfortable headroom. Under ``centering="ema"`` nothing
        constrains the rows and the floor is 0.

        **The floor is a reference, not a guarantee, at the shipped iteration
        count.** ``sinkhorn_iterations: 3`` does not converge: measured on random
        logits at ``K = 128``, ``B = 16``, the prototype column masses after 3
        iterations span 0.071 to 0.333 around a 0.125 target, and the observed
        entropy sits ~4 % *below* ``log(K / B_teacher)``. It approaches the bound
        from below as iterations rise and crosses it once the assignment is
        genuinely doubly stochastic (~200 iterations on that example). So read
        ``H_min`` as "where a converged Sinkhorn would floor this", expect the
        measured ``H`` to sit slightly under it, and treat a *large* gap -- not a
        small one -- as the signal. ``tests/test_losses.py`` pins both halves of
        this behaviour so the caveat cannot quietly stop being true.

        Raising the physical batch lowers the floor (more prototypes reachable
        per row) and lowering ``out_dim`` raises the normalised entropy; the two
        move together, which is why they were changed together.
        """
        prototypes = int(self.center.shape[-1])
        maximum = float(np.log(prototypes))
        if self.centering != "sinkhorn" or teacher_batch <= 0 or teacher_batch >= prototypes:
            return 0.0, maximum
        return float(np.log(prototypes / teacher_batch)), maximum

    @torch.no_grad()
    def collapse_metrics(self, teacher_probs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Diagnostics that reveal collapse when the loss curve does not.

        None of these is good or bad on its own, and the logging deliberately
        does not say which direction is which. What each one *is*:

        ``teacher_entropy`` (``H``)
            Mean entropy of a teacher row, in nats. Compare it against
            ``teacher_entropy_min`` and ``_max`` from :meth:`entropy_bounds`, not
            against zero -- the floor is a property of the centering and the
            batch, not of the model. Note that ``H`` sitting a few percent
            *below* ``teacher_entropy_min`` is expected at
            ``sinkhorn_iterations: 3``; see :meth:`entropy_bounds`.
        ``teacher_entropy_normalized``
            ``H / log K``. Comparable across prototype counts, which a bare ``H``
            is not: halving ``out_dim`` moves ``H`` by ``log 2`` for free.
        ``prototype_kl_to_uniform``
            KL of the batch's mean assignment from uniform. Rising means the
            batch is concentrating on a shrinking subset of prototypes -- a
            different failure from per-row sharpening, and the one Sinkhorn is
            supposed to prevent.
        ``prototype_perplexity`` / ``prototype_utilization``
            ``exp(H(marginal))`` and that divided by ``K``: the effective number
            of prototypes the batch actually used, and its share of the total.
            More directly readable than the KL, and defined for any centering
            mode. Note the ceiling is ``min(K, ...)`` in principle but is reached
            in practice only when the marginal is flat.
        ``prototype_dim`` / ``teacher_batch_size``
            ``K`` and ``B_teacher``, logged as metrics because every number above
            is conditional on them and a run's own artifacts should not require
            the config to interpret.

        Returns **device tensors** for the computed quantities. Converting here
        would put a ``cudaStreamSynchronize`` in the middle of every forward
        pass to produce numbers logged once in ten steps; :attr:`last_metrics`
        does the conversion when a caller actually wants them. The bounds and
        counts are Python floats -- they depend only on shapes.

        Cost: two extra reductions over an already-materialised ``[B, K]`` tensor
        (a mean and a log-sum), on logging steps only. Nothing here is per-step.
        """
        probs = _at_least_float32(teacher_probs).clamp_min(1e-12)
        prototypes = probs.shape[-1]
        teacher_batch = int(probs.shape[0])
        entropy = (-probs * probs.log()).sum(dim=-1).mean()
        marginal = probs.mean(dim=0)
        uniform = 1.0 / marginal.numel()
        kl_uniform = (marginal * (marginal / uniform).log()).sum()
        # Effective prototypes in use: the perplexity of the mean assignment.
        # Robust in a way a threshold count is not -- no cutoff to choose, and it
        # degrades smoothly rather than stepping as mass crosses a boundary.
        marginal_entropy = -(marginal * marginal.log()).sum()
        perplexity = marginal_entropy.exp()

        maximum = float(np.log(prototypes))
        minimum, _ = self.entropy_bounds(teacher_batch)
        self._metric_scalars.update(
            {
                "teacher_entropy_max": maximum,
                "teacher_entropy_min": minimum,
                "prototype_dim": float(prototypes),
                "teacher_batch_size": float(teacher_batch),
            }
        )
        return {
            "teacher_entropy": entropy,
            "teacher_entropy_normalized": entropy / maximum if maximum > 0 else entropy,
            "prototype_kl_to_uniform": kl_uniform,
            "prototype_perplexity": perplexity,
            "prototype_utilization": perplexity / float(prototypes),
        }

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor) -> None:
        """Eq. 3: EMA update of the centering vector (``centering="ema"`` only).

        The batch mean is all-reduced before it enters the EMA, so the buffer is
        one statistic over the global batch rather than ``world_size`` statistics
        drifting apart. Unlike the Sinkhorn choice this is not configurable:
        ``center`` is *state* the teacher's targets read and the checkpoint
        stores, so ranks disagreeing about it means ranks optimising different
        objectives, and a checkpoint whose contents depend on which rank wrote it.
        """
        from src.utils.training.distributed import all_reduce_mean

        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        if self.context is not None and self.context.enabled:
            batch_center = all_reduce_mean(batch_center.contiguous(), self.context)
        self.center = (
            self.center.to(batch_center.device) * self.center_momentum
            + batch_center * (1.0 - self.center_momentum)
        )

    @staticmethod
    def _concat_outputs(outputs: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
        """Normalise list/stacked view outputs into one ``[views * batch, dim]`` tensor."""
        if isinstance(outputs, torch.Tensor):
            if outputs.ndim == 3:
                return outputs.reshape(-1, outputs.shape[-1])
            return outputs
        return torch.cat(list(outputs), dim=0)

    def loss_flags(self) -> dict[str, Any]:
        """Every loss-side setting a stage-1 arm can move.

        Written into ``summary.json``. Without it a ``koleo_scope=all_views``
        control and the fixed default leave byte-identical machine-readable
        traces, which is exactly the failure ``loss_flags()`` was added to stage
        2 to prevent.
        """
        return {
            "objective": "dino_self_distillation",
            "centering": self.centering,
            "sinkhorn_iterations": self.sinkhorn_iterations,
            "distributed_sinkhorn": self.distributed_sinkhorn,
            "student_temp": self.student_temp,
            "center_momentum": self.center_momentum,
            "lambda_koleo": self.lambda_koleo,
            "koleo_scope": self.koleo_scope,
            "koleo_reduction": self.koleo_reduction,
            "num_crops": self.num_crops,
            "num_global_crops": self.num_global_crops,
            "prototypes": int(self.center.shape[-1]),
            "teacher_temp_start": float(self.teacher_temp_schedule[0]),
            "teacher_temp_final": float(self.teacher_temp_schedule[-1]),
        }

    def extra_repr(self) -> str:
        return (
            f"num_crops={self.num_crops}, num_global_crops={self.num_global_crops}, "
            f"student_temp={self.student_temp}, centering={self.centering}, "
            f"lambda_koleo={self.lambda_koleo}, koleo_scope={self.koleo_scope}"
        )
