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
reports the diagnostics that will: teacher output entropy and the KL between the
teacher's mean assignment and uniform.

Two things this module does *not* do any more
---------------------------------------------

**It does not recompute the student's log-softmax twelve times.** Eq. 1 pairs 2
teacher views against 6 student views. Written as a double loop, the inner
``F.log_softmax(student_logits)`` is evaluated once per *pair*, so each student
view's softmax over 8,192 prototypes is computed twice and back-propagated twice.
:meth:`CustomDINOLoss.compute_dino_loss` now takes one log-softmax over the whole
``[6B, 8192]`` block and contracts every pair in a single ``einsum``. Same
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CENTERING_MODES = ("ema", "sinkhorn")


def _at_least_float32(tensor: torch.Tensor) -> torch.Tensor:
    """Promote half precision to fp32, and leave anything wider alone.

    Every numerically delicate step in this module -- the Sinkhorn log-space
    normaliser, the log-softmax over 8,192 prototypes, the KoLeo pairwise
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
            bottleneck embedding rather than the 65k-wide prototype logits.
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


@torch.no_grad()
def sinkhorn_knopp(
    logits: torch.Tensor,
    temperature: float = 0.05,
    iterations: int = 3,
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
        logits: Teacher outputs, ``[batch, out_dim]``.
        temperature: Sharpening temperature applied before normalisation.
        iterations: Sinkhorn iterations (3 is the reference value).

    Returns:
        ``[batch, out_dim]`` in **fp32**, whatever dtype came in. Under bf16
        autocast the teacher logits arrive with ~3 decimal digits of mantissa;
        dividing them by 0.04 and exponentiating a doubly-stochastic normaliser
        in that precision is not something to do to a training target, and the
        cast is free next to the backbone forward that produced them.
    """
    log_assignments = (_at_least_float32(logits) / float(temperature)).t()  # [out_dim, batch]
    num_prototypes, num_samples = log_assignments.shape
    log_assignments = log_assignments - torch.logsumexp(log_assignments.reshape(-1), dim=0)

    for _ in range(max(int(iterations), 1)):
        # Prototype marginals -> 1 / num_prototypes.
        log_assignments = (
            log_assignments
            - torch.logsumexp(log_assignments, dim=1, keepdim=True)
            - math.log(num_prototypes)
        )
        # Sample marginals -> 1 / num_samples.
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
        lambda_koleo: Weight on the KoLeo regulariser. ``0`` disables it.
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
    ):
        super().__init__()
        if num_crops < 2:
            raise ValueError(f"num_crops must be >= 2 to form cross-view pairs, got {num_crops}")
        if not 1 <= num_global_crops <= num_crops:
            raise ValueError(f"num_global_crops must be in [1, {num_crops}], got {num_global_crops}")
        if centering not in CENTERING_MODES:
            raise ValueError(f"centering must be one of {CENTERING_MODES}, got {centering!r}")

        self.student_temp = float(student_temp)
        self.center_momentum = float(center_momentum)
        self.num_crops = int(num_crops)
        self.num_global_crops = int(num_global_crops)
        self.centering = centering
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.lambda_koleo = float(lambda_koleo)
        self.register_buffer("center", torch.zeros(1, out_dim))

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
            koleo = koleo_regularizer(student_embeddings)
            if self.metrics_enabled:
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
        log-softmax over 8,192 prototypes is evaluated once per view instead of
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
        # 10 before a softmax over 8,192 classes. Autocast already promotes
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

        if self.metrics_enabled:
            self._metric_tensors.update(self.collapse_metrics(teacher_probs_all))
            self._metric_scalars["cross_view_terms"] = float(loss_terms)
        return (pair_losses * mask).sum() / loss_terms

    @torch.no_grad()
    def collapse_metrics(self, teacher_probs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Diagnostics that reveal collapse when the loss curve does not.

        ``teacher_entropy`` falling toward 0 means the targets have sharpened to
        one-hot; ``prototype_kl_to_uniform`` rising means the batch is using a
        shrinking subset of the prototypes. Either alone is the signature the
        loss curve hides.

        Returns **device tensors**, not floats. Converting here would put a
        ``cudaStreamSynchronize`` in the middle of every forward pass, three
        times over, to produce numbers that get logged once in ten steps;
        :attr:`last_metrics` does the conversion when a caller actually wants
        them. ``teacher_entropy_max`` stays a Python float because it depends
        only on the prototype count.
        """
        probs = _at_least_float32(teacher_probs).clamp_min(1e-12)
        entropy = (-probs * probs.log()).sum(dim=-1).mean()
        marginal = probs.mean(dim=0)
        uniform = 1.0 / marginal.numel()
        kl_uniform = (marginal * (marginal / uniform).log()).sum()
        self._metric_scalars["teacher_entropy_max"] = float(np.log(probs.shape[-1]))
        return {
            "teacher_entropy": entropy,
            "prototype_kl_to_uniform": kl_uniform,
        }

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor) -> None:
        """Eq. 3: EMA update of the centering vector (``centering="ema"`` only)."""
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
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

    def extra_repr(self) -> str:
        return (
            f"num_crops={self.num_crops}, num_global_crops={self.num_global_crops}, "
            f"student_temp={self.student_temp}, centering={self.centering}, "
            f"lambda_koleo={self.lambda_koleo}"
        )
