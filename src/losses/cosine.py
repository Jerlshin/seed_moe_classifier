"""Feature-compactness terms (paper Section 1).

The introduction states: "we introduce cosine similarity loss within SwinV2's
residual connections, promoting feature compactness and improving robustness
against environmental variations."

``mode="intra_class"`` (revision default)
    Compactness is an **intra-class** property, so this is the reading that
    matches the words. Every embedding is pulled toward its own class centroid::

        L_cos = 1 - mean_i cos(e_i, c_{y_i})

    Centroids are maintained as an **EMA over training** rather than recomputed
    per batch. That is not a refinement, it is what makes the term exist: with
    ``batch_size=16`` over 27 classes the expected number of classes with two or
    more members is

        27 * [1 - (1-p)^16 - 16 p (1-p)^15] = 3.2,   p = 1/27

    so roughly 6 of 16 embeddings would contribute a non-zero term and the
    "centroid" each was pulled toward would be estimated from two samples. With
    EMA centroids every sample contributes and every centroid is estimated from
    the whole training history. This is centre loss (Wen et al., 2016) adapted to
    the hypersphere, and it supplies exactly the intra-class complement that
    ArcFace's inter-class margin does not.

``mode="residual"`` (submitted; retained for ablation, no longer the default)
    Keep ``h' = h + P(p_s)`` angularly aligned with ``h``::

        L_cos = 1 - cos(h + P(p_s), h)

    **Its minimisers destroy the residual it regularises.** ``L_cos = 0`` iff
    ``h'`` is a positive multiple of ``h``, i.e. iff

        P(p_s) = alpha(x) * h(x)   for some alpha >= -1, including alpha = 0

    So every global minimum either zeroes the residual outright -- which *is* the
    ``use_residual=False`` ablation -- or collapses it to a scalar rescaling
    carrying one degree of freedom of seed-type information instead of 384.
    Cosine is invariant to magnitude, so it constrains only the residual's
    direction; the cheapest way to preserve direction is to make the residual
    vanish. The weighting makes this worse over time: ``L_ArcFace`` decays toward
    0 as the model fits while ``L_cos`` does not, so the cosine term's share of
    the gradient budget grows monotonically -- weakest while the residual is
    forming, strongest when it could be dismantled.

    The structural replacement is :class:`LayerScale` on the residual branch
    (see :mod:`src.models.components.projections`), plus
    :func:`residual_magnitude_loss` if an explicit penalty is still wanted.

All terms are bounded below by 0 and reach 0 at perfect alignment.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

COSINE_MODES = ("intra_class", "residual")


def residual_cosine_loss(refined: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    """``1 - mean cos(refined, original)``, bounded in ``[0, 2]``."""
    return 1.0 - F.cosine_similarity(refined, original, dim=-1, eps=1e-8).mean()


def residual_magnitude_loss(
    residual: torch.Tensor,
    base: torch.Tensor,
    tau: float = 0.5,
) -> torch.Tensor:
    """Hinge on the residual-to-base magnitude ratio::

        L_res = mean_i max(0, ||P(p_s)_i|| / ||h_i|| - tau)^2

    Bounds how far the seed-type prior can move the representation **without
    rewarding its disappearance**: the term is exactly 0 for every residual
    smaller than ``tau``, so it is inactive in the healthy regime and has no
    gradient pushing ``P`` toward zero. That is the property
    :func:`residual_cosine_loss` lacks.
    """
    ratio = residual.norm(dim=-1) / base.norm(dim=-1).clamp_min(1e-8)
    return (ratio - float(tau)).clamp_min(0.0).pow(2).mean()


def intra_class_cosine_loss(embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Batch-centroid compactness, ``1 - mean cos(e_i, centroid(class of i))``.

    Classes with a single sample in the batch contribute exactly ``0`` -- the
    embedding is its own centroid. See :class:`CosineSimilarityLoss` for why the
    EMA variant is used instead of this by default; this is kept because it is
    the honest per-batch baseline for that comparison.
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
    """Compactness term, defaulting to EMA-centroid intra-class compactness.

    Args:
        mode: ``"intra_class"`` (default) or ``"residual"``. See the module
            docstring; ``"residual"`` is retained only so the submitted
            formulation can be ablated against the replacement.
        num_classes: Number of sub-varieties, needed to size the centroid buffer.
        embed_dim: Embedding width, likewise.
        centroid_momentum: EMA factor for the class centroids. ``0`` falls back
            to per-batch centroids.
    """

    def __init__(
        self,
        mode: str = "intra_class",
        num_classes: int | None = None,
        embed_dim: int | None = None,
        centroid_momentum: float = 0.9,
    ):
        super().__init__()
        if mode not in COSINE_MODES:
            raise ValueError(f"mode must be one of {COSINE_MODES}, got {mode!r}")
        self.mode = mode
        self.centroid_momentum = float(centroid_momentum)
        self.use_ema_centroids = (
            mode == "intra_class"
            and num_classes is not None
            and embed_dim is not None
            and self.centroid_momentum > 0.0
        )

        if self.use_ema_centroids:
            # Persistent: the centroids are training state that a resumed run
            # must not silently reset, and they are buffers rather than
            # parameters so the optimizer never sees them.
            self.register_buffer("centroids", torch.zeros(int(num_classes), int(embed_dim)))
            self.register_buffer("centroid_initialized", torch.zeros(int(num_classes), dtype=torch.bool))

    @torch.no_grad()
    def _update_centroids(self, normalized: torch.Tensor, labels: torch.Tensor) -> None:
        momentum = self.centroid_momentum
        for class_index in labels.unique():
            members = normalized[labels == class_index]
            batch_mean = F.normalize(members.mean(dim=0), p=2, dim=-1, eps=1e-8)
            if not bool(self.centroid_initialized[class_index]):
                # A class's first appearance must seed the centroid outright;
                # blending against a zero vector would halve its magnitude and
                # make the first few updates meaningless.
                self.centroids[class_index] = batch_mean
                self.centroid_initialized[class_index] = True
            else:
                blended = momentum * self.centroids[class_index] + (1.0 - momentum) * batch_mean
                self.centroids[class_index] = F.normalize(blended, p=2, dim=-1, eps=1e-8)

    def _ema_intra_class(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(embeddings, p=2, dim=-1, eps=1e-8)
        if self.centroids.device != normalized.device:
            self.centroids = self.centroids.to(normalized.device)
            self.centroid_initialized = self.centroid_initialized.to(normalized.device)
        if self.training:
            self._update_centroids(normalized.detach(), labels)
        targets = self.centroids.index_select(0, labels).detach()
        # Unseen classes still have a zero centroid; their cosine is 0, which is
        # the same "no information yet" contribution a per-batch singleton makes.
        return 1.0 - (normalized * targets).sum(dim=-1).mean()

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
        if self.use_ema_centroids:
            return self._ema_intra_class(refined, labels)
        return intra_class_cosine_loss(refined, labels)

    def extra_repr(self) -> str:
        return f"mode={self.mode}, ema_centroids={getattr(self, 'use_ema_centroids', False)}"
