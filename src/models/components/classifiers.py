"""Stage-1 seed-type classifier and the stage-2 sub-variety embedding network.

Paper Section 5.1 (Eqs. 5-6) defines stage 1 as a plain MLP over the DINOv2
embedding::

    s   = g(z),          s in R^4
    p_s = softmax(s)

``SeedTypeClassifier`` implements that as ``variant="mlp"`` (the default). The
repository previously shipped a heavier block with a squeeze-excitation branch
and a scalar feature gate; that behaviour is preserved verbatim under
``variant="se_gated"`` so the two can be compared in an ablation, but it is not
what the paper specifies.

Section 5.5 applies ArcFace to the cross-attention output. ArcFace needs an
embedding to measure angles in, so ``SubVarietyEmbedding`` produces that
embedding; the ArcFace class centres live in
:class:`~src.models.components.arcface_head.ArcFaceHead`. The paper does not
specify this network's depth, so ``variant="identity"`` (feed ``h''`` straight
to ArcFace) is available alongside the default residual MLP.
"""

from __future__ import annotations

import torch
import torch.nn as nn

SEED_CLASSIFIER_VARIANTS = ("mlp", "se_gated")
SUB_EMBEDDING_VARIANTS = ("mlp", "identity")


class SeedTypeClassifier(nn.Module):
    """``g`` from Eq. 5: pooled embedding ``z`` -> seed-type logits ``s``.

    Args:
        feature_dim: Width of ``z`` (paper: 384).
        num_seed_types: Number of coarse classes (paper: 4).
        dropout_rate: Dropout applied after the hidden activation.
        hidden_ratio: Hidden width relative to ``feature_dim``.
        variant: ``"mlp"`` for the paper's plain MLP, ``"se_gated"`` for the
            squeeze-excitation + gated block kept for ablation.
    """

    def __init__(
        self,
        feature_dim: int,
        num_seed_types: int,
        dropout_rate: float = 0.3,
        hidden_ratio: float = 0.5,
        variant: str = "mlp",
    ):
        super().__init__()
        if variant not in SEED_CLASSIFIER_VARIANTS:
            raise ValueError(f"variant must be one of {SEED_CLASSIFIER_VARIANTS}, got {variant!r}")
        self.variant = variant
        hidden_dim = max(int(feature_dim * hidden_ratio), 1)

        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, num_seed_types)

        if variant == "se_gated":
            squeeze_dim = max(hidden_dim // 4, 1)
            self.se = nn.Sequential(
                nn.Linear(hidden_dim, squeeze_dim),
                nn.ReLU(inplace=True),
                nn.Linear(squeeze_dim, hidden_dim),
                nn.Sigmoid(),
            )
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
            self.stochastic_depth = nn.Dropout(p=min(dropout_rate, 0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return seed-type logits ``s`` of shape ``[batch, num_seed_types]``."""
        x = self.dropout1(self.act1(self.norm1(self.fc1(x))))
        if self.variant == "mlp":
            return self.fc2(x)
        x = x * self.se(x)
        x = self.gate(x) * x
        return self.fc2(self.stochastic_depth(x))

    def extra_repr(self) -> str:
        return f"variant={self.variant}"


class SubVarietyEmbedding(nn.Module):
    """Embedding network feeding the ArcFace head (paper Section 5.5).

    Produces the vector whose angle against each class centre ArcFace penalises.
    The output keeps ``feature_dim`` width so the ArcFace centres live in the
    same space as ``h''``.

    Args:
        feature_dim: Input and output width.
        dropout_rate: Dropout inside the MLP.
        variant: ``"mlp"`` for the residual/highway MLP, ``"identity"`` to pass
            ``h''`` through unchanged.
        use_highway: Blend MLP output with the input through a learned gate.
            Only used by ``variant="mlp"``.
        expansion: Widest hidden layer relative to ``feature_dim``.
    """

    def __init__(
        self,
        feature_dim: int,
        dropout_rate: float = 0.3,
        variant: str = "mlp",
        use_highway: bool = True,
        expansion: int = 4,
    ):
        super().__init__()
        if variant not in SUB_EMBEDDING_VARIANTS:
            raise ValueError(f"variant must be one of {SUB_EMBEDDING_VARIANTS}, got {variant!r}")
        self.variant = variant
        self.feature_dim = feature_dim

        if variant == "identity":
            return

        wide_dim = feature_dim * max(expansion // 2, 1)
        widest_dim = feature_dim * expansion
        self.norm = nn.LayerNorm(feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, wide_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(wide_dim, widest_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(widest_dim, wide_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(wide_dim, feature_dim),
            nn.Dropout(dropout_rate),
        )
        self.use_highway = use_highway
        if use_highway:
            self.highway_gate = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.Sigmoid())
        self.drop_path = nn.Identity() if dropout_rate == 0 else nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the sub-variety embedding, shape ``[batch, feature_dim]``."""
        if self.variant == "identity":
            return x
        x = self.norm(x)
        mlp_out = self.mlp(x)
        if self.use_highway:
            gate = self.highway_gate(x)
            embeddings = gate * mlp_out + (1.0 - gate) * x
        else:
            embeddings = mlp_out
        return self.drop_path(embeddings)

    def extra_repr(self) -> str:
        return f"feature_dim={self.feature_dim}, variant={self.variant}"


class LinearSubVarietyHead(nn.Module):
    """Plain linear classifier replacing ArcFace (``use_arcface=False``).

    Mirrors :class:`~src.models.components.arcface_head.ArcFaceHead`'s
    ``(logits, margin_logits)`` return so the model body needs no branch. There
    is no angular margin, so both outputs are the same tensor -- which means the
    combined objective's ArcFace term degrades to exactly the categorical
    cross-entropy the ``wo_arcface`` ablation calls for, with no change to the
    loss code.

    Args:
        feature_dim: Embedding width.
        num_classes: Number of sub-varieties (27).
    """

    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.classifier = nn.Linear(self.feature_dim, self.num_classes)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor | None = None,  # noqa: ARG002 - signature parity with ArcFaceHead
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.classifier(embeddings)
        return logits, logits

    def extra_repr(self) -> str:
        return f"feature_dim={self.feature_dim}, num_classes={self.num_classes}"
