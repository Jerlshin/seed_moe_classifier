"""Fused EMA update for the DINO teacher.

``lightly.models.utils.update_momentum`` walks ``zip(student.parameters(),
teacher.parameters())`` in Python and issues two CUDA kernels per parameter. For
a SwinV2-Base student plus its projection head that is ~440 parameter tensors,
so ~880 kernel launches every optimizer step -- each one a few microseconds of
launch latency against a few microseconds of work, on tensors far too small to
saturate anything. At batch 16 that overhead is not lost in the noise; it is
measurable next to the step itself.

:class:`TeacherEmaUpdater` resolves the parameter pairing **once** at
construction and then applies the whole update with two ``torch._foreach_``
calls, which the CUDA implementation fuses into a handful of multi-tensor
kernels. The arithmetic is identical:

    teacher <- m * teacher + (1 - m) * student

Buffers are deliberately not touched, matching ``update_momentum``: SwinV2's
buffers (``relative_coords_table``, ``relative_position_index``) are constants,
and the projection head's normalisation is LayerNorm by default precisely so
that no running statistic needs carrying across the EMA. Switching the head to
``use_batch_norm: "batch"`` reintroduces buffers that nothing propagates -- which
is the coupling ``conf/model/head/dino.yaml`` documents, not something to paper
over here.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TeacherEmaUpdater:
    """Momentum update of one or more (student, teacher) module pairs.

    Args:
        pairs: ``(student_module, teacher_module)`` tuples. Both sides must
            expose parameters in the same order, which is guaranteed here
            because every teacher is a deep copy of its student.

    Raises:
        ValueError: If a pair's parameter lists disagree in length or shape --
            the failure mode that a silent ``zip`` would truncate away, leaving
            the tail of the teacher frozen at initialisation forever.
    """

    def __init__(self, pairs: list[tuple[nn.Module, nn.Module]]):
        self._student: list[torch.Tensor] = []
        self._teacher: list[torch.Tensor] = []

        for student, teacher in pairs:
            student_params = list(student.parameters())
            teacher_params = list(teacher.parameters())
            if len(student_params) != len(teacher_params):
                raise ValueError(
                    f"EMA pair mismatch: student has {len(student_params)} parameters, "
                    f"teacher has {len(teacher_params)}."
                )
            for index, (source, target) in enumerate(zip(student_params, teacher_params)):
                if source.shape != target.shape:
                    raise ValueError(
                        f"EMA pair mismatch at parameter {index}: {tuple(source.shape)} "
                        f"vs {tuple(target.shape)}."
                    )
                self._student.append(source.detach())
                self._teacher.append(target.detach())

    def __len__(self) -> int:
        return len(self._teacher)

    @torch.no_grad()
    def update(self, momentum: float) -> None:
        """Advance every teacher tensor toward its student by ``1 - momentum``."""
        momentum = float(momentum)
        if not self._teacher:
            return
        if momentum >= 1.0:
            # The cosine schedule reaches exactly 1.0 at the final step, where
            # the update is the identity. Skipping it avoids a no-op kernel.
            return
        torch._foreach_mul_(self._teacher, momentum)
        torch._foreach_add_(self._teacher, self._student, alpha=1.0 - momentum)
