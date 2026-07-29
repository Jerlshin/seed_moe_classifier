"""ArcFace classification head (paper Section 5.6, Eq. 15).

    L = -sum_i log( e^{s cos(theta_i + m)} / (e^{s cos(theta_i + m)} + sum_{j != i} e^{s cos theta_j}) )

The head owns the learnable class centres, so a saved model is self-sufficient for
inference: ``logits`` (no margin) rank the classes, while ``margin_logits`` are what
the cross-entropy is taken over during training.

``cos(theta + m)`` is expanded as ``cos theta cos m - sin theta sin m`` rather than
routed through ``acos``, which keeps the gradient finite near ``cos theta = +-1``.
For ``theta + m > pi`` the true cosine stops being monotonic in ``theta``; ArcFace
handles that region with a linear penalty (``cos theta - m sin m``), or by falling
back to the unmodified cosine when ``easy_margin`` is set.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    """Additive angular margin head over L2-normalised embeddings.

    Args:
        feature_dim: Embedding width.
        num_classes: Number of classes (paper: 27 sub-varieties).
        scale: Feature scale ``s`` applied to the cosine logits.
        margin: Angular margin ``m`` in radians.
        easy_margin: Use the softer out-of-range fallback.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        scale: float = 30.0,
        margin: float = 0.5,
        easy_margin: bool = False,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.scale = float(scale)
        self.margin = float(margin)
        self.easy_margin = bool(easy_margin)

        self.weight = nn.Parameter(torch.empty(num_classes, feature_dim))
        nn.init.xavier_uniform_(self.weight)

        self.register_buffer("cos_m", torch.tensor(math.cos(self.margin)), persistent=False)
        self.register_buffer("sin_m", torch.tensor(math.sin(self.margin)), persistent=False)
        # Beyond this cosine, theta + m exceeds pi and cos(theta + m) is no longer monotonic.
        self.register_buffer("threshold", torch.tensor(math.cos(math.pi - self.margin)), persistent=False)
        self.register_buffer(
            "margin_penalty",
            torch.tensor(math.sin(math.pi - self.margin) * self.margin),
            persistent=False,
        )

    def cosine_similarity(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Cosine between each embedding and every class centre, shape ``[batch, num_classes]``."""
        normalized_embeddings = F.normalize(embeddings.float(), p=2, dim=1)
        normalized_weight = F.normalize(self.weight.float(), p=2, dim=1)
        return F.linear(normalized_embeddings, normalized_weight).clamp(-1.0 + 1e-7, 1.0 - 1e-7)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits, margin_logits)``, both scaled by ``s``.

        ``logits`` never carry the margin and are the correct quantity for
        prediction, ranking and hierarchy alignment. ``margin_logits`` equal
        ``logits`` when ``labels`` is ``None`` (i.e. at inference).
        """
        cosine = self.cosine_similarity(embeddings)
        logits = self.scale * cosine
        if labels is None:
            return logits, logits

        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp_min(1e-9))
        target_cosine = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            target_cosine = torch.where(cosine > 0, target_cosine, cosine)
        else:
            target_cosine = torch.where(
                cosine > self.threshold,
                target_cosine,
                cosine - self.margin_penalty,
            )

        one_hot = torch.zeros_like(cosine).scatter_(1, labels.view(-1, 1), 1.0)
        margin_cosine = one_hot * target_cosine + (1.0 - one_hot) * cosine
        return logits, self.scale * margin_cosine

    def extra_repr(self) -> str:
        return (
            f"feature_dim={self.feature_dim}, num_classes={self.num_classes}, "
            f"scale={self.scale}, margin={self.margin}, easy_margin={self.easy_margin}"
        )
