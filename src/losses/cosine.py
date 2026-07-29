"""Cosine similarity loss for feature compactness (paper Section 1).

The introduction states: "we introduce cosine similarity loss within SwinV2's
residual connections, promoting feature compactness and improving robustness
against environmental variations."

The residual connection the hierarchical head owns is Eq. 9, ``h' = h + P(p_s)``.
Two readings of "compactness" are supported:

``mode="residual"`` (default)
    Keep the seed-type-conditioned feature ``h'`` angularly aligned with the raw
    MoE feature ``h``::

        L_cos = 1 - mean_i cos(h'_i, h_i)

    This lets the seed-type prior *shift* the representation without rotating it
    away from what the experts extracted, which is what stops the residual from
    overwriting the MoE output when stage 1 is confidently wrong.

``mode="intra_class"``
    Class-level compactness: pull every embedding toward its own class centroid::

        L_cos = 1 - mean_i cos(e_i, centroid(class of i))

    Complements ArcFace's inter-class separation with an explicit intra-class
    term. Requires labels.

Both are bounded in ``[0, 2]`` and reach ``0`` at perfect alignment.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

COSINE_MODES = ("residual", "intra_class")


def residual_cosine_loss(refined: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    """``1 - mean cos(refined, original)``, bounded in ``[0, 2]``."""
    return 1.0 - F.cosine_similarity(refined, original, dim=-1, eps=1e-8).mean()


def intra_class_cosine_loss(embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """``1 - mean cos(e_i, centroid of its class)``, bounded in ``[0, 2]``.

    Classes with a single sample in the batch contribute exactly ``0`` (the
    embedding is its own centroid), so they neither help nor distort the term.
    """
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError(
            f"embeddings and labels disagree on batch size: "
            f"{embeddings.shape[0]} vs {labels.shape[0]}"
        )
    if embeddings.numel() == 0:
        return embeddings.new_zeros(())

    normalized = F.normalize(embeddings, p=2, dim=-1, eps=1e-8)
    num_classes = int(labels.max().item()) + 1
    index = labels.view(-1, 1).expand(-1, normalized.shape[-1])

    sums = torch.zeros(num_classes, normalized.shape[-1], device=normalized.device, dtype=normalized.dtype)
    sums = sums.scatter_add(0, index, normalized)
    counts = torch.zeros(num_classes, device=normalized.device, dtype=normalized.dtype)
    counts = counts.scatter_add(0, labels, torch.ones_like(labels, dtype=normalized.dtype))

    centroids = F.normalize(sums / counts.clamp_min(1.0).unsqueeze(-1), p=2, dim=-1, eps=1e-8)
    similarity = (normalized * centroids.index_select(0, labels)).sum(dim=-1)
    return 1.0 - similarity.mean()


class CosineSimilarityLoss(nn.Module):
    """Compactness term over the Eq. 9 residual connection.

    Args:
        mode: ``"residual"`` or ``"intra_class"``. See the module docstring.
    """

    def __init__(self, mode: str = "residual"):
        super().__init__()
        if mode not in COSINE_MODES:
            raise ValueError(f"mode must be one of {COSINE_MODES}, got {mode!r}")
        self.mode = mode

    def forward(
        self,
        refined: torch.Tensor,
        original: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.mode == "residual":
            return residual_cosine_loss(refined, original)
        if labels is None:
            raise ValueError("mode='intra_class' requires labels")
        return intra_class_cosine_loss(refined, labels)

    def extra_repr(self) -> str:
        return f"mode={self.mode}"
