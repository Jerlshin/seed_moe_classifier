"""Angular-margin classification heads (paper Section 5.6, Eq. 15).

    L = -sum_i log( e^{s cos(theta_i + m)} / (e^{s cos(theta_i + m)} + sum_{j != i} e^{s cos theta_j}) )

The head owns the learnable class centres, so a saved model is self-sufficient for
inference: ``logits`` (no margin) rank the classes, while ``margin_logits`` are what
the cross-entropy is taken over during training.

``cos(theta + m)`` is expanded as ``cos theta cos m - sin theta sin m`` rather than
routed through ``acos``, which keeps the gradient finite near ``cos theta = +-1``.
For ``theta + m > pi`` the true cosine stops being monotonic in ``theta``; ArcFace
handles that region with a linear penalty (``cos theta - m sin m``), or by falling
back to the unmodified cosine when ``easy_margin`` is set.

The feature scale, revised
--------------------------

The submitted configuration used ``s = 30``, the value ArcFace tuned for face
recognition over 10^5--10^6 identities. AdaCos (Zhang et al., CVPR 2019) derives
the fixed optimal scale for a ``C``-class cosine-softmax as ``sqrt(2) log(C-1)``,
which for ``C = 27`` is

    sqrt(2) * ln 26 = 1.4142 * 3.2581 = 4.61

so ``s = 30`` is **6.5x too large for this problem**, with two measurable
consequences. At initialisation ``cos theta ~ 0`` and ``m = 0.5`` give a target
logit of ``30 * (-sin 0.5) = -14.38``, hence ``L_ArcFace = ln(1 + 26 e^{14.38})
= 17.64`` against ``L_seed = ln 4 = 1.386`` -- a **12.7 : 1** ratio at equal
lambda, so the angular-margin term consumes essentially the whole gradient budget
early in training. And ``sub_logits = 30 cos theta`` in ``[-30, 30]`` makes
``softmax(sub_logits)`` near-one-hot, which saturates the hierarchy KL term and
makes any calibration analysis meaningless without temperature scaling.

:func:`adacos_scale` computes the analytic value; ``scale="auto"`` uses it.
``dynamic=True`` instead re-derives ``s`` each step from the running median
target angle (AdaCos proper), which removes the hyperparameter rather than
retuning it.

The margin warm-up
------------------

Applying the full margin from step 0 is a documented convergence hazard on small
backbones and small datasets -- CurricularFace reports outright divergence at
``m = 0.5`` on MobileFaceNet/CASIA-WebFace where ``m = 0.45`` converges. The
trainer calls :meth:`ArcFaceHead.set_margin_scale` once per epoch to ramp
``m: 0 -> margin`` over the first ``margin_warmup_fraction`` of training.

Two heads, three ablations
--------------------------

* :class:`ArcFaceHead` -- normalised embedding, normalised centres, margin.
* :class:`NormFaceHead` -- normalised embedding, normalised centres, **no**
  margin. This is the single-factor control for the margin (``wo_margin_only``).
* :class:`~src.models.components.classifiers.LinearSubVarietyHead` -- plain
  linear. Changes normalisation *and* margin *and* logit scale at once, so it is
  labelled ``wo_angular_head`` rather than ``wo_arcface``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def adacos_scale(num_classes: int) -> float:
    """AdaCos fixed scale ``sqrt(2) * log(C - 1)`` (Zhang et al., CVPR 2019)."""
    if num_classes < 3:
        return 1.0
    return math.sqrt(2.0) * math.log(num_classes - 1)


def resolve_scale(scale: float | str | None, num_classes: int) -> float:
    """Resolve a configured scale, honouring the ``"auto"`` sentinel."""
    if scale is None or (isinstance(scale, str) and str(scale).lower() == "auto"):
        return adacos_scale(num_classes)
    return float(scale)


class ArcFaceHead(nn.Module):
    """Additive angular margin head over L2-normalised embeddings.

    Args:
        feature_dim: Embedding width.
        num_classes: Number of classes (paper: 27 sub-varieties).
        scale: Feature scale ``s``. Pass ``"auto"`` for the AdaCos value.
        margin: Angular margin ``m`` in radians, at full strength.
        easy_margin: Use the softer out-of-range fallback.
        sub_centers: Prototypes per class (Deng et al., sub-centre ArcFace). ``1``
            is standard ArcFace; ``3`` lets one class span several appearance
            modes, which is plausible when a "sub-variety" covers multiple
            growing conditions or imaging sessions.
        dynamic: Re-derive ``s`` each training step from the running median
            target angle (AdaCos) instead of holding it fixed.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        scale: float | str = "auto",
        margin: float = 0.5,
        easy_margin: bool = False,
        sub_centers: int = 1,
        dynamic: bool = False,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.sub_centers = max(int(sub_centers), 1)
        self.scale = resolve_scale(scale, num_classes)
        self.margin = float(margin)
        self.easy_margin = bool(easy_margin)
        self.dynamic = bool(dynamic)

        self.weight = nn.Parameter(torch.empty(num_classes * self.sub_centers, feature_dim))
        nn.init.xavier_uniform_(self.weight)

        # Effective margin, ramped by the trainer. Non-persistent so it is always
        # rebuilt from the schedule rather than restored stale from a checkpoint.
        self.register_buffer("margin_scale", torch.tensor(1.0), persistent=False)
        # AdaCos running scale, used only when `dynamic`.
        self.register_buffer("dynamic_scale", torch.tensor(float(self.scale)), persistent=False)

    # ---------------------------------------------------------------- schedule

    def set_margin_scale(self, value: float) -> None:
        """Set the margin multiplier in ``[0, 1]`` (0 at step 0, 1 after warm-up)."""
        self.margin_scale.fill_(min(max(float(value), 0.0), 1.0))

    @property
    def effective_margin(self) -> float:
        return self.margin * float(self.margin_scale)

    @property
    def effective_scale(self) -> float:
        return float(self.dynamic_scale) if self.dynamic else self.scale

    # ------------------------------------------------------------------ logits

    def cosine_similarity(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Cosine between each embedding and every class, shape ``[batch, num_classes]``.

        With ``sub_centers > 1`` the per-class value is the maximum over that
        class's prototypes, which is what makes sub-centre ArcFace tolerant of
        multi-modal classes.
        """
        normalized_embeddings = F.normalize(embeddings.float(), p=2, dim=1)
        normalized_weight = F.normalize(self.weight.float(), p=2, dim=1)
        cosine = F.linear(normalized_embeddings, normalized_weight)
        if self.sub_centers > 1:
            cosine = cosine.view(-1, self.num_classes, self.sub_centers).max(dim=-1).values
        return cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

    @torch.no_grad()
    def _update_dynamic_scale(self, cosine: torch.Tensor, labels: torch.Tensor) -> None:
        """AdaCos: set ``s`` from the median target angle of this batch."""
        target_cos = cosine.gather(1, labels.view(-1, 1)).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        median_angle = torch.acos(target_cos).median()
        theta = torch.clamp(median_angle, max=math.pi / 4.0)
        # B_avg, the non-target logit mass, estimated from the current scale.
        one_hot = torch.zeros_like(cosine).scatter_(1, labels.view(-1, 1), 1.0)
        b_avg = torch.exp(self.dynamic_scale * cosine)[one_hot == 0].sum() / cosine.shape[0]
        scale = torch.log(b_avg.clamp_min(1e-8)) / torch.cos(theta).clamp_min(1e-8)
        self.dynamic_scale.fill_(float(scale.clamp(1.0, 64.0)))

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
        if labels is not None and self.dynamic and self.training:
            self._update_dynamic_scale(cosine, labels)

        scale = self.effective_scale
        logits = scale * cosine
        if labels is None:
            return logits, logits

        margin = self.effective_margin
        if margin <= 0.0:
            # During warm-up the head *is* a cosine-softmax (NormFace). Returning
            # the plain logits keeps that exact rather than approximating it with
            # cos(theta + 0) computed through the trigonometric expansion.
            return logits, logits

        cos_m = math.cos(margin)
        sin_m = math.sin(margin)
        threshold = math.cos(math.pi - margin)
        margin_penalty = math.sin(math.pi - margin) * margin

        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp_min(1e-9))
        target_cosine = cosine * cos_m - sine * sin_m
        if self.easy_margin:
            target_cosine = torch.where(cosine > 0, target_cosine, cosine)
        else:
            target_cosine = torch.where(cosine > threshold, target_cosine, cosine - margin_penalty)

        one_hot = torch.zeros_like(cosine).scatter_(1, labels.view(-1, 1), 1.0)
        margin_cosine = one_hot * target_cosine + (1.0 - one_hot) * cosine
        return logits, scale * margin_cosine

    def extra_repr(self) -> str:
        return (
            f"feature_dim={self.feature_dim}, num_classes={self.num_classes}, "
            f"scale={self.scale:.3f}, margin={self.margin}, sub_centers={self.sub_centers}, "
            f"dynamic={self.dynamic}, easy_margin={self.easy_margin}"
        )


class NormFaceHead(nn.Module):
    """Cosine-softmax head: normalised embedding and centres, **no margin**.

    This is the single-factor control for the angular margin. Swapping
    :class:`ArcFaceHead` for :class:`~src.models.components.classifiers.LinearSubVarietyHead`
    changes four things at once -- margin, embedding normalisation, centre
    normalisation and logit scale -- so the gap that substitution measures is not
    attributable to the margin. This head changes exactly one.

    Args:
        feature_dim: Embedding width.
        num_classes: Number of sub-varieties.
        scale: Feature scale ``s``; ``"auto"`` uses the AdaCos value, which is
            what keeps it identical to the ArcFace head's scale.
    """

    def __init__(self, feature_dim: int, num_classes: int, scale: float | str = "auto"):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.scale = resolve_scale(scale, num_classes)
        self.weight = nn.Parameter(torch.empty(num_classes, feature_dim))
        nn.init.xavier_uniform_(self.weight)

    def set_margin_scale(self, value: float) -> None:  # noqa: ARG002 - schedule parity
        """No-op: this head has no margin to ramp."""

    def cosine_similarity(self, embeddings: torch.Tensor) -> torch.Tensor:
        normalized_embeddings = F.normalize(embeddings.float(), p=2, dim=1)
        normalized_weight = F.normalize(self.weight.float(), p=2, dim=1)
        return F.linear(normalized_embeddings, normalized_weight).clamp(-1.0 + 1e-7, 1.0 - 1e-7)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor | None = None,  # noqa: ARG002 - signature parity
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.scale * self.cosine_similarity(embeddings)
        return logits, logits

    def extra_repr(self) -> str:
        return f"feature_dim={self.feature_dim}, num_classes={self.num_classes}, scale={self.scale:.3f}"
