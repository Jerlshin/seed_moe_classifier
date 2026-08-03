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
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CENTERING_MODES = ("ema", "sinkhorn")


def koleo_regularizer(features: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Kozachenko-Leonenko differential-entropy regulariser (DINOv2).

        L_koleo = -(1/n) sum_i log( min_{j != i} ||z_i - z_j|| )

    Encourages a uniform span of the feature sphere within a batch, which stops
    distinct-but-similar samples collapsing onto each other. That is precisely
    the failure mode a *fine-grained* task cannot tolerate: 27 sub-varieties of
    the same four crops are near-duplicates by construction. DINOv2's ablation
    credits KoLeo with >8 % on instance retrieval at no cost elsewhere.

    Args:
        features: ``[batch, dim]``. L2-normalised internally, so pass the
            bottleneck embedding rather than the 65k-wide prototype logits.
    """
    if features.ndim != 2 or features.shape[0] < 2:
        return features.new_zeros(())
    normalized = F.normalize(features.float(), dim=-1, eps=epsilon)
    distances = torch.cdist(normalized, normalized, p=2)
    # Exclude the self-distance without perturbing any real pair.
    distances = distances + torch.eye(distances.shape[0], device=distances.device) * 1e6
    nearest = distances.min(dim=-1).values
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
    """
    import math

    log_assignments = (logits.float() / float(temperature)).t()  # [out_dim, batch]
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
    return log_assignments.t().exp().to(logits.dtype)


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
        #: Populated by :meth:`forward`; read by the trainer for the collapse
        #: diagnostics the loss curve cannot show.
        self.last_metrics: dict[str, float] = {}

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
            self.last_metrics["koleo"] = float(koleo.detach())
            loss = loss + self.lambda_koleo * koleo

        if self.centering == "ema":
            self.update_center(teacher_output)
        return loss

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
        """Eq. 1 over every cross-view (teacher view, student view) pair."""
        student_chunks = (student_output / self.student_temp).chunk(self.num_crops)
        teacher_probs_all = self.teacher_targets(teacher_output, epoch).detach()
        teacher_chunks = teacher_probs_all.chunk(self.num_global_crops)

        student_ids = list(student_view_ids) if student_view_ids is not None else list(range(self.num_crops))
        teacher_ids = (
            list(teacher_view_ids) if teacher_view_ids is not None else list(range(self.num_global_crops))
        )
        if len(student_ids) != self.num_crops or len(teacher_ids) != self.num_global_crops:
            raise ValueError(
                f"view id counts must match the crop counts: got {len(student_ids)} student ids for "
                f"{self.num_crops} crops and {len(teacher_ids)} teacher ids for {self.num_global_crops}"
            )

        total_loss = student_output.new_zeros(())
        loss_terms = 0
        for teacher_id, teacher_probs in zip(teacher_ids, teacher_chunks):
            for student_id, student_logits in zip(student_ids, student_chunks):
                if student_id == teacher_id:
                    # Same view on both sides carries no cross-view signal.
                    continue
                loss = torch.sum(-teacher_probs * F.log_softmax(student_logits, dim=-1), dim=-1)
                total_loss = total_loss + loss.mean()
                loss_terms += 1

        if loss_terms == 0:
            raise RuntimeError(
                "DINO loss received no cross-view pairs; check num_crops, num_global_crops and the view ids."
            )
        self.last_metrics.update(self.collapse_metrics(teacher_probs_all))
        self.last_metrics["cross_view_terms"] = float(loss_terms)
        return total_loss / loss_terms

    @torch.no_grad()
    def collapse_metrics(self, teacher_probs: torch.Tensor) -> dict[str, float]:
        """Diagnostics that reveal collapse when the loss curve does not.

        ``teacher_entropy`` falling toward 0 means the targets have sharpened to
        one-hot; ``prototype_kl_to_uniform`` rising means the batch is using a
        shrinking subset of the prototypes. Either alone is the signature the
        loss curve hides.
        """
        probs = teacher_probs.float().clamp_min(1e-12)
        entropy = float((-probs * probs.log()).sum(dim=-1).mean())
        marginal = probs.mean(dim=0)
        uniform = 1.0 / marginal.numel()
        kl_uniform = float((marginal * (marginal / uniform).log()).sum())
        return {
            "teacher_entropy": entropy,
            "teacher_entropy_max": float(np.log(probs.shape[-1])),
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
