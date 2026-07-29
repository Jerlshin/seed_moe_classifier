"""ArcFace objective for sub-variety classification (paper Section 5.5, Eq. 13).

    L = -sum_i log( e^{s cos(theta_i + m)}
                    / (e^{s cos(theta_i + m)} + sum_{j != i} e^{s cos theta_j}) )

That expression is exactly a softmax cross-entropy over logits in which the
target class's cosine has been shifted by the angular margin ``m``. The shift
itself is produced by :class:`~src.models.components.arcface_head.ArcFaceHead`,
which owns the learnable class centres.

Two entry points:

``arcface_loss``
    Functional form, taking the margin logits the head already produced. This is
    what :class:`~src.losses.hierarchical.CombinedHierarchicalLoss` uses, and it
    keeps the criterion **stateless** -- the class centres live in the model, so
    a saved ``model_state_dict`` is sufficient for inference.

``ArcFaceLoss``
    Self-contained module that owns its own head, for ablations and for testing
    the head in isolation. It holds parameters, so any optimizer must be told
    about them.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.arcface_head import ArcFaceHead


def arcface_loss(
    margin_logits: torch.Tensor,
    labels: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Cross-entropy over margin-adjusted, scaled cosine logits.

    Args:
        margin_logits: ``s * cos(theta + m)`` on the target class and
            ``s * cos(theta)`` elsewhere, shape ``[batch, num_classes]``.
        labels: Integer targets, shape ``[batch]``.
        label_smoothing: Passed through to ``F.cross_entropy``.
    """
    return F.cross_entropy(margin_logits, labels, label_smoothing=label_smoothing)


class ArcFaceLoss(nn.Module):
    """Standalone ArcFace loss that owns its class centres.

    Args:
        feature_dim: Embedding width.
        num_classes: Number of classes (paper: 27 sub-varieties).
        scale: Feature scale ``s``.
        margin: Angular margin ``m`` in radians.
        easy_margin: Use the softer out-of-range fallback.
        label_smoothing: Passed through to the cross-entropy.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        scale: float = 30.0,
        margin: float = 0.5,
        easy_margin: bool = False,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.head = ArcFaceHead(
            feature_dim=feature_dim,
            num_classes=num_classes,
            scale=scale,
            margin=margin,
            easy_margin=easy_margin,
        )
        self.label_smoothing = float(label_smoothing)

    @property
    def weight(self) -> nn.Parameter:
        """The learnable class centres, exposed for inspection and checkpointing."""
        return self.head.weight

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        _, margin_logits = self.head(embeddings, labels)
        return arcface_loss(margin_logits, labels, label_smoothing=self.label_smoothing)
