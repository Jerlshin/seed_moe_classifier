"""Projection blocks that carry information between the paper's hierarchy levels.

Two distinct projections appear in Section 5:

``EmbeddingProjection``
    Aligns the backbone's pooled output with the embedding width the paper works
    in. Section 5.1 states the encoder "extracts a 384-dimensional feature
    vector" (Eq. 4, ``z in R^384``), but no SwinV2 variant emits 384 channels --
    Base emits 1024, Tiny and Small emit 768 -- so this projection is what makes
    Eq. 4 hold at all.

    It belongs to :class:`~src.models.builder.DinoV2SwinV2Encoder`, which is why
    ``encoder(images).shape[-1] == 384`` is an invariant rather than a
    configuration coincidence. Inside the hierarchical head the same class is
    used only when the head is fed raw backbone features; when its input is
    already ``z`` the head substitutes ``nn.Identity`` instead.

``SeedTypeProjection``
    ``P`` from Eq. 9, mapping the 4-D seed-type probability vector ``p_s`` back
    into the 384-D feature space so it can be added onto the MoE output:

        h' = h + P(p_s)

    The paper calls this an "MLP projection layer"; the depth is configurable
    because the paper does not specify it.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EmbeddingProjection(nn.Module):
    """Map a backbone feature of width ``in_dim`` onto the paper's ``z in R^{out_dim}``.

    Args:
        in_dim: Width of the pooled backbone feature.
        out_dim: Target embedding width (paper: 384).
        hidden_dim: Width of the optional hidden layer. ``None`` uses a single
            linear layer, which is the default and keeps the projection cheap.
        dropout: Dropout applied after the non-linearity of the hidden variant.
        use_norm: Apply LayerNorm to the projected embedding.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        use_norm: bool = True,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.is_identity = self.in_dim == self.out_dim and hidden_dim is None and not use_norm

        if self.is_identity:
            self.project: nn.Module = nn.Identity()
            return

        if hidden_dim is None:
            layers: list[nn.Module] = [nn.Linear(self.in_dim, self.out_dim)]
        else:
            layers = [
                nn.Linear(self.in_dim, int(hidden_dim)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(int(hidden_dim), self.out_dim),
            ]
        if use_norm:
            layers.append(nn.LayerNorm(self.out_dim))
        self.project = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_dim:
            raise ValueError(f"EmbeddingProjection expects in_dim={self.in_dim}, got {x.shape[-1]}")
        return self.project(x)

    def extra_repr(self) -> str:
        return f"in_dim={self.in_dim}, out_dim={self.out_dim}"


class SeedTypeProjection(nn.Module):
    """``P`` from Eq. 9: seed-type probabilities -> feature space.

    Args:
        num_seed_types: Width of ``p_s`` (paper: 4).
        embed_dim: Width of the MoE feature ``h`` (paper: 384).
        depth: Number of ``Linear -> GELU -> LayerNorm -> Dropout`` stages before
            the final linear layer. ``depth=1`` is the minimal MLP the paper
            describes; the repository's earlier head used ``depth=3``.
        dropout: Dropout inside each stage.
        bottleneck_ratio: Width of the middle stages relative to ``embed_dim``.
            Only used when ``depth > 1``.
    """

    def __init__(
        self,
        num_seed_types: int,
        embed_dim: int,
        depth: int = 2,
        dropout: float = 0.1,
        bottleneck_ratio: float = 0.5,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.num_seed_types = int(num_seed_types)
        self.embed_dim = int(embed_dim)

        bottleneck_dim = max(int(self.embed_dim * bottleneck_ratio), 1)
        widths = [self.embed_dim if index == 0 else bottleneck_dim for index in range(depth)]

        layers: list[nn.Module] = []
        current = self.num_seed_types
        for width in widths:
            layers.extend(
                [
                    nn.Linear(current, width),
                    nn.GELU(),
                    nn.LayerNorm(width),
                    nn.Dropout(dropout),
                ]
            )
            current = width
        layers.append(nn.Linear(current, self.embed_dim))
        self.project = nn.Sequential(*layers)

    def forward(self, seed_type_probs: torch.Tensor) -> torch.Tensor:
        """Project ``p_s`` of shape ``[batch, num_seed_types]`` to ``[batch, embed_dim]``."""
        if seed_type_probs.shape[-1] != self.num_seed_types:
            raise ValueError(
                f"SeedTypeProjection expects {self.num_seed_types} seed types, "
                f"got {seed_type_probs.shape[-1]}"
            )
        return self.project(seed_type_probs)

    def extra_repr(self) -> str:
        return f"num_seed_types={self.num_seed_types}, embed_dim={self.embed_dim}"
