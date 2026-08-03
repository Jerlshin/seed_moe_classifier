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

        h' = h + gamma * P(p_s)

    The paper calls this an "MLP projection layer"; the depth is configurable
    because the paper does not specify it. ``gamma`` is :class:`LayerScale` --
    see below for why the residual needs a structural magnitude control rather
    than a loss-side one.

Three further blocks the revision adds:

``LayerScale``
    A learned per-channel gain initialised at ``1e-4`` (Touvron et al., 2021).
    It gives the residual branch the "start negligible, grow only if it helps"
    behaviour that ``L_cos(residual)`` was reaching for, without a loss term
    whose global minimum is ``P(p_s) = 0``.

``FiLMFusion``
    The alternative to the additive residual. Eq. 9 conditions on ``p_s``, a
    4-simplex point, through a 2-layer MLP. Once the seed head fits -- and with
    only 4 classes it fits early -- ``p_s -> e_c`` one-hot, so ``P(p_s)`` takes
    one of **four fixed vectors**: at convergence the fusion is a 4-entry
    codebook implemented by a 75k-parameter MLP. Worse, the softmax Jacobian
    ``diag(p) - p p^T`` vanishes as ``p`` saturates, so the fine branch's
    gradient into the coarse classifier dies exactly when the coarse head becomes
    confident. FiLM conditions on the seed classifier's **pre-softmax hidden
    state** instead, which neither saturates nor quantises, and modulates
    multiplicatively::

        (gamma, beta) = MLP(g_hidden(z)),   h' = gamma * h + beta

    ``gamma = 1 + tanh(.)`` is bounded by construction, so the residual's
    magnitude cannot track the coarse head's confidence -- the same hazard
    ``PAPER_AUDIT.md`` 2.2 addressed by projecting probabilities rather than
    logits, handled structurally. Setting ``gamma = 1`` recovers the additive
    form exactly, so this is a strict superset and the two are ablatable against
    each other.

``TokenPooling``
    Collapses SwinV2's ``[B, L, 384]`` token grid to ``[B, 384]`` **after** the
    head rather than before it, which is what lets the MoE and the
    cross-attention see more than one token.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

TOKEN_POOLING_MODES = ("attention", "gem", "mean")
FUSION_MODES = ("additive", "film")


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


class LayerScale(nn.Module):
    """Learned per-channel gain on a residual branch (Touvron et al., 2021).

    Initialised near zero so the branch starts inert and has to earn its
    magnitude. This replaces the loss-side attempt to control the Eq. 9 residual:
    a cosine penalty on ``(h + P(p_s), h)`` is minimised by ``P(p_s) = 0``, so it
    rewards deleting the very connection it exists to regularise. A gain the
    optimiser can raise has no such fixed point.

    Args:
        dim: Channel width.
        init_value: Initial gain (paper-standard LayerScale value: 1e-4).
    """

    def __init__(self, dim: int, init_value: float = 1e-4):
        super().__init__()
        self.dim = int(dim)
        self.init_value = float(init_value)
        self.gamma = nn.Parameter(torch.full((self.dim,), self.init_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, init_value={self.init_value}"


class FiLMFusion(nn.Module):
    """Feature-wise linear modulation of ``h`` by the coarse branch (Perez et al., 2018).

        (gamma, beta) = MLP(condition),   h' = gamma * h + beta

    ``gamma = 1 + tanh(.)`` lies in ``(0, 2)``, so the modulation is bounded
    however confident the conditioning branch becomes. See the module docstring
    for why this is conditioned on the seed classifier's hidden state rather than
    on ``p_s``.

    Args:
        condition_dim: Width of the conditioning vector (the seed classifier's
            hidden activation, 192-D at ``hidden_ratio=0.5``).
        embed_dim: Width of the feature being modulated.
        hidden_dim: Hidden width of the modulation MLP. Defaults to
            ``condition_dim``.
        dropout: Dropout inside the modulation MLP.
    """

    def __init__(
        self,
        condition_dim: int,
        embed_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.condition_dim = int(condition_dim)
        self.embed_dim = int(embed_dim)
        hidden = int(hidden_dim) if hidden_dim else self.condition_dim

        self.project = nn.Sequential(
            nn.Linear(self.condition_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2 * self.embed_dim),
        )
        # Start as the identity: gamma = 1 + tanh(0) = 1, beta = 0. The fusion
        # therefore begins exactly at "no modulation" and is a strict superset of
        # the additive form from step 0.
        nn.init.zeros_(self.project[-1].weight)
        nn.init.zeros_(self.project[-1].bias)

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(modulated_features, residual)``.

        ``residual = h' - h`` is returned so the reporting contract and the
        magnitude penalty see the same quantity they do in the additive case.
        """
        if condition.shape[-1] != self.condition_dim:
            raise ValueError(
                f"FiLMFusion expects condition width {self.condition_dim}, got {condition.shape[-1]}"
            )
        gamma_beta = self.project(condition)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = 1.0 + torch.tanh(gamma)

        if features.ndim == 3:  # broadcast across a token grid
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)

        modulated = gamma * features + beta
        return modulated, modulated - features

    def extra_repr(self) -> str:
        return f"condition_dim={self.condition_dim}, embed_dim={self.embed_dim}"


class TokenPooling(nn.Module):
    """Collapse ``[batch, tokens, dim]`` to ``[batch, dim]`` after the head.

    Args:
        dim: Token width.
        mode: ``"attention"`` learns a query over the tokens, ``"gem"`` is
            generalised-mean pooling with a learned exponent (Radenovic et al.,
            2018), ``"mean"`` is the plain average the pooled path uses.
    """

    def __init__(self, dim: int, mode: str = "attention"):
        super().__init__()
        if mode not in TOKEN_POOLING_MODES:
            raise ValueError(f"mode must be one of {TOKEN_POOLING_MODES}, got {mode!r}")
        self.mode = mode
        self.dim = int(dim)

        if mode == "attention":
            self.score = nn.Sequential(nn.LayerNorm(self.dim), nn.Linear(self.dim, 1))
        elif mode == "gem":
            self.power = nn.Parameter(torch.tensor(3.0))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim == 2:
            return tokens
        if tokens.ndim != 3:
            raise ValueError(f"TokenPooling expects [batch, tokens, dim], got {tuple(tokens.shape)}")

        if self.mode == "mean":
            return tokens.mean(dim=1)
        if self.mode == "gem":
            power = self.power.clamp(1.0, 8.0)
            return tokens.clamp_min(1e-6).pow(power).mean(dim=1).pow(1.0 / power)
        weights = F.softmax(self.score(tokens), dim=1)
        return (weights * tokens).sum(dim=1)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, mode={self.mode}"
