"""DINO self-distillation loss (paper Section 4, Eqs. 1-3 and Algorithm 1).

Cross-view objective (Eq. 1)::

    L_DINO = -(1/N) sum_i sum_{v != q} q_v . log p_v

``q_v`` is the teacher's soft target for view ``v``, ``p_v`` the student's
prediction. Only *cross-view* pairs contribute: a student view is never scored
against the teacher's output for that same view, which is what forces the
representation to be invariant to the augmentation rather than to memorise it.

Two stabilisers keep the objective from collapsing onto a constant embedding:

*Temperature scheduling* (Eq. 2) ramps the teacher temperature linearly from
``warmup_teacher_temp`` to ``teacher_temp`` over the first
``warmup_teacher_temp_epochs`` epochs, then holds it. Paper Table 1 uses
0.02 -> 0.04 over 5 epochs. A cold teacher early on produces sharp targets that
would let the student collapse before it has learned anything.

*Centering* (Eq. 3) subtracts a running mean from the teacher logits::

    C_t = m C_{t-1} + (1 - m) qbar

with ``m = 0.9`` and ``qbar`` the batch-mean teacher output. Sharpening (low
temperature) and centering counteract each other: centering alone would push
towards a uniform output, sharpening alone towards a one-hot one.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomDINOLoss(nn.Module):
    """Self-distillation loss with teacher temperature warmup and centering.

    Args:
        out_dim: Width of the projection head output (paper: 65,536).
        num_crops: Total student views per image (paper: 2 global + 4 local = 6).
        warmup_teacher_temp: Teacher temperature at epoch 0 (paper: 0.02).
        teacher_temp: Teacher temperature after warmup (paper: 0.04).
        warmup_teacher_temp_epochs: Warmup length in epochs (paper: 5).
        num_epochs: Total training epochs, used to size the schedule.
        student_temp: Fixed student temperature.
        center_momentum: ``m`` in Eq. 3 (paper: 0.9).
        num_global_crops: Views the teacher sees (paper: 2).
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
    ):
        super().__init__()
        if num_crops < 2:
            raise ValueError(f"num_crops must be >= 2 to form cross-view pairs, got {num_crops}")
        if not 1 <= num_global_crops <= num_crops:
            raise ValueError(f"num_global_crops must be in [1, {num_crops}], got {num_global_crops}")

        self.student_temp = float(student_temp)
        self.center_momentum = float(center_momentum)
        self.num_crops = int(num_crops)
        self.num_global_crops = int(num_global_crops)
        self.register_buffer("center", torch.zeros(1, out_dim))

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
    ) -> torch.Tensor:
        """Return the mean cross-view loss and update the centering buffer.

        Args:
            student_output: ``num_crops`` tensors of shape ``[batch, out_dim]``,
                or a single pre-concatenated tensor.
            teacher_output: ``num_global_crops`` tensors, same convention.
            epoch: Zero-based epoch index, used to pick the teacher temperature.
        """
        student_output = self._concat_outputs(student_output)
        teacher_output = self._concat_outputs(teacher_output)
        loss = self.compute_dino_loss(student_output, teacher_output, epoch)
        self.update_center(teacher_output)
        return loss

    def compute_dino_loss(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        """Eq. 1 over every cross-view (teacher view, student view) pair."""
        teacher_temp = self.teacher_temperature(epoch)
        center = self.center.to(device=teacher_output.device, dtype=teacher_output.dtype)

        student_chunks = (student_output / self.student_temp).chunk(self.num_crops)
        teacher_chunks = F.softmax((teacher_output - center) / teacher_temp, dim=-1)
        teacher_chunks = teacher_chunks.detach().chunk(self.num_global_crops)

        total_loss = student_output.new_zeros(())
        loss_terms = 0
        for teacher_index, teacher_probs in enumerate(teacher_chunks):
            for student_index, student_logits in enumerate(student_chunks):
                if student_index == teacher_index:
                    # Same view on both sides carries no cross-view signal.
                    continue
                loss = torch.sum(-teacher_probs * F.log_softmax(student_logits, dim=-1), dim=-1)
                total_loss = total_loss + loss.mean()
                loss_terms += 1

        if loss_terms == 0:
            raise RuntimeError(
                "DINO loss received no cross-view pairs; check num_crops and num_global_crops."
            )
        return total_loss / loss_terms

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor) -> None:
        """Eq. 3: EMA update of the centering vector."""
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
            f"student_temp={self.student_temp}, center_momentum={self.center_momentum}"
        )
