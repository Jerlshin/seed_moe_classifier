"""Sparse Mixture-of-Experts block (paper Section 5.2, Eq. 8).

Six experts with a Top-K gate that activates the K most relevant experts per
routing token:

    h = sum_{i in Top-K} G_i * E_i(z)

**Two routing granularities.** ``token_mode`` on the head decides what a routing
token *is*:

``pooled``
    One token per image -- the pooled DINO embedding ``z``. This is what the
    submitted manuscript describes.

``grid`` (revision default)
    SwinV2's final-stage ``8x8`` token grid, projected to 384. Routing then
    happens per spatial token, which raises the routing slots per optimiser step
    from ``batch x K`` to ``batch x 64 x K``. That matters twice over: the
    load-balancing statistic stops being an entropy estimated from 16 samples
    over 6 bins, and the experts' self-attention stops being degenerate (below).

**Attention over a length-1 sequence is an affine map.** With a key/value
sequence of length 1, ``softmax(QK^T/sqrt(d))`` evaluates to the scalar ``1``
whatever ``Q`` and ``K`` are, so the output is ``W_O W_V x`` and the ``Q``/``K``
projections receive exactly zero gradient forever. Under ``token_mode="pooled"``
this module therefore does **not** allocate a ``MultiheadAttention`` at all: it
substitutes ``nn.Linear``, which spans exactly the same function class
(``W_O(W_V x + b_V) + b_O`` is affine) at 147,840 parameters per expert instead
of 591,360. That removes 295,680 provably dead parameters per expert from the
reported totals rather than counting them as "active". Under
``token_mode="grid"`` the attention is real and the full module is built.

**K = 2, revised.** The submitted manuscript specifies ``K = 4`` of ``E = 6``.
``DEFAULT_TOP_K`` below is the single place that value is defined.

**Routing controls.** ``router_mode`` selects between the learned gate and two
controls the ablation suite needs to attribute the MoE's contribution:
``"hash"`` routes by a fixed hash of the token index (sparse capacity with no
learned routing) and ``"uniform"`` gives every expert weight ``1/E`` (dense
ensembling with no sparsity). Neither has trainable routing, which is the point.

:class:`DenseExpertBlock` is the ``use_moe=False`` ablation. Its
``capacity_multiplier`` exists because the full model activates ``K`` experts per
token while a dense block activates one: at ``capacity_multiplier=top_k`` the
block's feed-forward width matches Top-K's active capacity, so the ``wo_moe``
gap measures routing rather than a 2x cut in active FLOPs.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_NUM_EXPERTS = 6
DEFAULT_TOP_K = 2

#: How each expert mixes information across its input tokens.
TOKEN_MIXING_MODES = ("attention", "affine")

#: Router implementations. Only ``learned`` has trainable routing parameters.
ROUTER_MODES = ("learned", "hash", "uniform")


class MoEOutput(NamedTuple):
    """Everything downstream code and the regularisers need from one MoE pass."""

    features: torch.Tensor
    """``h`` from Eq. 8, same shape as the input (``[B, D]`` or ``[B, L, D]``)."""

    gate_logits: torch.Tensor
    """Pre-softmax router output, shape ``[num_tokens, num_experts]``. The router
    z-loss is taken over these, so they must not be softmaxed first."""

    gate_probs: torch.Tensor
    """Full gate distribution ``G``, shape ``[num_tokens, num_experts]``."""

    top_k_indices: torch.Tensor
    """Indices of the selected experts, shape ``[num_tokens, top_k]``."""

    top_k_weights: torch.Tensor
    """Renormalised weights of the selected experts, shape ``[num_tokens, top_k]``."""

    dispatch_weights: torch.Tensor
    """``top_k_weights`` scattered back over all experts, zero elsewhere."""

    tokens_per_sample: int = 1
    """Routing tokens contributed by each image. ``1`` in pooled mode, ``H*W`` in
    grid mode. Consumers that need per-image routing statistics reshape
    ``top_k_indices`` by this factor rather than assuming one row per image."""


class TransformerExpert(nn.Module):
    """A single expert: post-norm block whose token mixer depends on the mode.

    Args:
        embed_dim: Token width.
        num_heads: Attention heads. Ignored under ``token_mixing="affine"``.
        mlp_dim: Hidden width of the feed-forward block.
        dropout: Dropout inside the expert.
        token_mixing: ``"attention"`` builds a real ``MultiheadAttention``, valid
            only when the input carries more than one token. ``"affine"`` builds
            the single ``nn.Linear`` that a length-1 attention is mathematically
            equal to, so no parameter in the expert is unreachable by gradient.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
        token_mixing: str = "affine",
    ):
        super().__init__()
        if token_mixing not in TOKEN_MIXING_MODES:
            raise ValueError(f"token_mixing must be one of {TOKEN_MIXING_MODES}, got {token_mixing!r}")
        self.token_mixing = token_mixing

        if token_mixing == "attention":
            self.attn: nn.Module | None = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.mix: nn.Module | None = None
        else:
            # Not allocated: a length-1 attention can never use Q or K.
            self.attn = None
            self.mix = nn.Linear(embed_dim, embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block over ``[tokens, seq_len, embed_dim]``."""
        if self.attn is not None:
            mixed, _ = self.attn(x, x, x, need_weights=False)
        else:
            mixed = self.mix(x)
        x = self.norm1(x + mixed)
        return self.norm2(x + self.mlp(x))

    def extra_repr(self) -> str:
        return f"token_mixing={self.token_mixing}"


class MixtureOfExperts(nn.Module):
    """Top-K sparse MoE over ``num_experts`` experts.

    Args:
        embed_dim: Width of the tokens the experts operate on.
        num_experts: Expert count (6).
        mlp_dim: Hidden width inside each expert's feed-forward block.
        top_k: Experts activated per routing token (2 after the revision).
        num_heads: Attention heads inside each expert.
        dropout: Dropout applied inside the experts.
        renormalize_top_k: Rescale the selected gate weights to sum to 1. Note
            that with this on, an L1 penalty on the off-Top-K mass cannot change
            the module's function -- see :func:`~src.losses.moe.l1_sparsity_loss`.
        sparse_dispatch: Run each expert only on the tokens routed to it. Set
            ``False`` to evaluate every expert densely. The two are equal in the
            forward pass; they are *not* equal under AdamW unless the trainer
            calls :meth:`materialize_zero_grads`.
        token_mixing: Passed to :class:`TransformerExpert`.
        router_mode: ``"learned"``, ``"hash"`` or ``"uniform"``; see the module
            docstring.
        gate_condition_dim: Extra width concatenated onto the router input, used
            by the seed-conditioned router (the coarse prediction ``p_s``). Zero
            keeps the paper's unconditioned gate.
        noise_std: Initial scale of Shazeer-style Gaussian gating noise. The
            trainer anneals it to zero with :meth:`set_noise_scale`; noise is
            applied in training mode only.
        gate_init_std: Standard deviation of the router's weight init. Near-zero
            makes early routing close to uniform, which is what stops the
            rich-get-richer race from being decided by initialisation.
    """

    def __init__(
        self,
        embed_dim: int,
        num_experts: int = DEFAULT_NUM_EXPERTS,
        mlp_dim: int = 512,
        top_k: int = DEFAULT_TOP_K,
        num_heads: int = 8,
        dropout: float = 0.0,
        renormalize_top_k: bool = True,
        sparse_dispatch: bool = True,
        token_mixing: str = "affine",
        router_mode: str = "learned",
        gate_condition_dim: int = 0,
        noise_std: float = 0.0,
        gate_init_std: float = 1e-3,
    ):
        super().__init__()
        if num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {num_experts}")
        if not 1 <= top_k <= num_experts:
            raise ValueError(f"top_k must be in [1, {num_experts}], got {top_k}")
        if router_mode not in ROUTER_MODES:
            raise ValueError(f"router_mode must be one of {ROUTER_MODES}, got {router_mode!r}")

        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.renormalize_top_k = renormalize_top_k
        self.sparse_dispatch = sparse_dispatch
        self.router_mode = router_mode
        self.gate_condition_dim = int(gate_condition_dim)
        self.experts = nn.ModuleList(
            TransformerExpert(embed_dim, num_heads, mlp_dim, dropout, token_mixing=token_mixing)
            for _ in range(num_experts)
        )

        gate_in_dim = embed_dim + self.gate_condition_dim
        if router_mode == "learned":
            self.gate: nn.Module | None = nn.Linear(gate_in_dim, num_experts)
            nn.init.normal_(self.gate.weight, std=float(gate_init_std))
            nn.init.zeros_(self.gate.bias)
            # Learned noise scale, as in Shazeer et al. (2017). Allocated only
            # when noise is actually scheduled: at `noise_std = 0` it would never
            # enter the graph, so it would be a block of parameters that no
            # configuration can reach -- the same defect as the dead Q/K
            # projections, and it would be counted as "active" the same way.
            self.noise_gate: nn.Module | None = None
            if float(noise_std) > 0.0:
                self.noise_gate = nn.Linear(gate_in_dim, num_experts)
                nn.init.zeros_(self.noise_gate.weight)
                nn.init.zeros_(self.noise_gate.bias)
        else:
            # A control router owns no parameters: that is what makes it a
            # control for *learned* routing rather than for sparse capacity.
            self.gate = None
            self.noise_gate = None

        self.register_buffer("noise_scale", torch.tensor(float(noise_std)), persistent=False)

    # ------------------------------------------------------------------ schedule

    def set_noise_scale(self, value: float) -> None:
        """Set the gating-noise scale (the trainer anneals this to 0)."""
        self.noise_scale.fill_(max(float(value), 0.0))

    # ------------------------------------------------------------------- routing

    def _router_logits(
        self,
        tokens: torch.Tensor,
        condition: torch.Tensor | None,
    ) -> torch.Tensor:
        """Pre-softmax router output for ``[num_tokens, embed_dim]`` input."""
        if self.router_mode == "uniform":
            return tokens.new_zeros(tokens.shape[0], self.num_experts)
        if self.router_mode == "hash":
            # Fixed assignment by token position: no parameters, no learning, but
            # the same sparse capacity as Top-K. Large negative logits elsewhere
            # keep the softmax a valid distribution over the chosen experts.
            positions = torch.arange(tokens.shape[0], device=tokens.device)
            logits = tokens.new_full((tokens.shape[0], self.num_experts), -1e4)
            for offset in range(self.top_k):
                chosen = (positions * (offset + 1) + offset) % self.num_experts
                logits.scatter_(1, chosen.unsqueeze(1), 0.0)
            return logits

        gate_input = tokens if condition is None else torch.cat([tokens, condition], dim=-1)
        logits = self.gate(gate_input)
        if self.noise_gate is not None and self.training and float(self.noise_scale) > 0.0:
            # Shazeer et al. (2017): the only exploration mechanism deterministic
            # Top-K has. `topk` is flat almost everywhere, so nothing else in the
            # objective can say "this token should have gone to expert 4".
            spread = F.softplus(self.noise_gate(gate_input))
            logits = logits + torch.randn_like(logits) * spread * self.noise_scale
        return logits

    def forward(
        self,
        x: torch.Tensor,
        gate_condition: torch.Tensor | None = None,
    ) -> MoEOutput:
        """Route and combine.

        Args:
            x: ``[batch, embed_dim]`` (pooled) or ``[batch, tokens, embed_dim]``
                (grid). Every token is routed independently.
            gate_condition: Optional ``[batch, gate_condition_dim]`` vector
                concatenated onto the router input -- the seed-type posterior in
                the hierarchical configuration. Broadcast across tokens.
        """
        if x.ndim not in (2, 3):
            raise ValueError(f"MoE expects [batch, embed_dim] or [batch, tokens, embed_dim], got {tuple(x.shape)}")
        if x.shape[-1] != self.embed_dim:
            raise ValueError(f"MoE expects embed_dim={self.embed_dim}, got {x.shape[-1]}")

        batch = x.shape[0]
        tokens_per_sample = 1 if x.ndim == 2 else x.shape[1]
        flat = x.reshape(-1, self.embed_dim)

        condition = None
        if gate_condition is not None:
            if self.gate_condition_dim == 0:
                raise ValueError("gate_condition supplied but gate_condition_dim=0")
            if gate_condition.shape[-1] != self.gate_condition_dim:
                raise ValueError(
                    f"gate_condition must have width {self.gate_condition_dim}, got {gate_condition.shape[-1]}"
                )
            condition = gate_condition.unsqueeze(1).expand(batch, tokens_per_sample, -1).reshape(
                -1, self.gate_condition_dim
            )
        elif self.gate_condition_dim > 0:
            raise ValueError("This MoE was built with gate_condition_dim > 0 but no gate_condition was passed")

        gate_logits = self._router_logits(flat, condition)
        gate_probs = F.softmax(gate_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        if self.renormalize_top_k:
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        dispatch_weights = torch.zeros_like(gate_probs).scatter(1, top_k_indices, top_k_weights)
        features = (
            self._sparse_forward(flat, top_k_indices, dispatch_weights)
            if self.sparse_dispatch
            else self._dense_forward(flat, dispatch_weights)
        )
        return MoEOutput(
            features=features.reshape(x.shape),
            gate_logits=gate_logits,
            gate_probs=gate_probs,
            top_k_indices=top_k_indices,
            top_k_weights=top_k_weights,
            dispatch_weights=dispatch_weights,
            tokens_per_sample=tokens_per_sample,
        )

    def _sparse_forward(
        self,
        x: torch.Tensor,
        top_k_indices: torch.Tensor,
        dispatch_weights: torch.Tensor,
    ) -> torch.Tensor:
        selection = torch.zeros_like(dispatch_weights, dtype=torch.bool)
        selection.scatter_(1, top_k_indices, True)

        output = torch.zeros_like(x)
        for expert_index, expert in enumerate(self.experts):
            rows = selection[:, expert_index].nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            expert_output = expert(x.index_select(0, rows).unsqueeze(1)).squeeze(1)
            weights = dispatch_weights.index_select(0, rows)[:, expert_index].unsqueeze(-1)
            output = output.index_add(0, rows, expert_output * weights)
        return output

    def _dense_forward(self, x: torch.Tensor, dispatch_weights: torch.Tensor) -> torch.Tensor:
        token = x.unsqueeze(1)
        expert_outputs = torch.stack([expert(token).squeeze(1) for expert in self.experts], dim=1)
        return torch.sum(expert_outputs * dispatch_weights.unsqueeze(-1), dim=1)

    # -------------------------------------------------------- optimizer parity

    @torch.no_grad()
    def materialize_zero_grads(self) -> int:
        """Give every unrouted expert parameter an explicit zero gradient.

        Under sparse dispatch an expert no token reached never enters the
        autograd graph, so ``p.grad is None`` and AdamW skips it **entirely** --
        including decoupled weight decay and the moment-buffer decay. The dense
        path gives those same parameters a zero gradient, so decay *does* apply.
        The two dispatch modes therefore train measurably different models unless
        this is called between ``backward()`` and ``step()``.

        Returns the number of parameters that had to be materialised, which is a
        direct measure of how much of the model sat out the step.
        """
        materialized = 0
        for parameter in self.experts.parameters():
            if parameter.requires_grad and parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
                materialized += 1
        return materialized

    # ------------------------------------------------------------- diagnostics

    @torch.no_grad()
    def expert_utilization(self, top_k_indices: torch.Tensor) -> torch.Tensor:
        """Hard dispatch fraction ``f`` per expert, shape ``[num_experts]``.

        This is the quantity the load-balancing loss must control, and it is what
        ``plot_expert_utilization`` draws. Sums to 1 across experts.
        """
        flat = top_k_indices.reshape(-1)
        counts = torch.zeros(self.num_experts, device=top_k_indices.device)
        counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=counts.dtype))
        return counts / counts.sum().clamp_min(1.0)

    @torch.no_grad()
    def dead_expert_count(self, top_k_indices: torch.Tensor) -> int:
        """Experts that received no token at all: ``#{i : f_i = 0}``."""
        return int((self.expert_utilization(top_k_indices) == 0).sum())

    # ------------------------------------------------------- efficiency support

    def parameters_per_expert(self) -> int:
        """Parameter count of a single expert.

        All experts share one architecture, so counting the first is exact. This
        is what makes the active-parameter arithmetic in
        :mod:`src.utils.efficiency` a closed form rather than a measurement.
        """
        return sum(parameter.numel() for parameter in self.experts[0].parameters())

    def dormant_parameters(self) -> int:
        """Parameters that a single forward pass never touches.

        ``(E - K)`` experts sit out every routing token, so their weights
        contribute to the total parameter count but not to per-token compute.

        This counts only *routing* dormancy. Parameters that no configuration can
        ever reach are not built in the first place -- see the affine token mixer
        in :class:`TransformerExpert` -- so this number no longer includes any
        provably dead weights.
        """
        return (self.num_experts - self.top_k) * self.parameters_per_expert()

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, num_experts={self.num_experts}, "
            f"top_k={self.top_k}, router_mode={self.router_mode}, "
            f"sparse_dispatch={self.sparse_dispatch}"
        )


class DenseExpertBlock(nn.Module):
    """Single always-on block replacing the MoE (``use_moe=False``).

    The ``wo_moe`` ablation asks what sparse routing contributes. Deleting the
    experts outright would also delete a block's worth of capacity, so the gap
    would confound *routing* with *depth*.

    Keeping exactly one expert-sized block does not fix that on its own: the full
    model activates ``top_k`` experts per token and this activates one, so the
    naive substitution still confounds routing with a ``top_k``-fold cut in
    active capacity. ``capacity_multiplier`` scales the feed-forward width so the
    two can be matched; ``scripts/run_ablations.py`` runs both the naive and the
    capacity-matched control, because they answer different questions.

    Returns a :class:`MoEOutput` describing a one-expert router that always picks
    its single expert with weight 1. Note what that means for the objective: both
    MoE regularisers evaluate to zero on that degenerate gate, so this variant
    optimises a strictly smaller objective than the full model. That is recorded
    in the run's ``loss_flags`` rather than left implicit.
    """

    def __init__(
        self,
        embed_dim: int,
        mlp_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.0,
        token_mixing: str = "affine",
        capacity_multiplier: int = 1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = 1
        self.top_k = 1
        self.router_mode = "none"
        self.capacity_multiplier = max(int(capacity_multiplier), 1)
        self.expert = TransformerExpert(
            embed_dim,
            num_heads,
            mlp_dim * self.capacity_multiplier,
            dropout,
            token_mixing=token_mixing,
        )

    def forward(
        self,
        x: torch.Tensor,
        gate_condition: torch.Tensor | None = None,  # noqa: ARG002 - parity with MixtureOfExperts
    ) -> MoEOutput:
        if x.ndim not in (2, 3):
            raise ValueError(f"DenseExpertBlock expects 2-D or 3-D input, got {tuple(x.shape)}")
        if x.shape[-1] != self.embed_dim:
            raise ValueError(f"DenseExpertBlock expects embed_dim={self.embed_dim}, got {x.shape[-1]}")

        tokens_per_sample = 1 if x.ndim == 2 else x.shape[1]
        flat = x.reshape(-1, self.embed_dim)
        features = self.expert(flat.unsqueeze(1)).squeeze(1)

        ones = flat.new_ones(flat.shape[0], 1)
        indices = torch.zeros(flat.shape[0], 1, dtype=torch.long, device=flat.device)
        return MoEOutput(
            features=features.reshape(x.shape),
            gate_logits=flat.new_zeros(flat.shape[0], 1),
            gate_probs=ones,
            top_k_indices=indices,
            top_k_weights=ones,
            dispatch_weights=ones,
            tokens_per_sample=tokens_per_sample,
        )

    @torch.no_grad()
    def materialize_zero_grads(self) -> int:
        """Always zero: a dense block has no unrouted parameters."""
        return 0

    @torch.no_grad()
    def expert_utilization(self, top_k_indices: torch.Tensor) -> torch.Tensor:
        return torch.ones(1, device=top_k_indices.device)

    @torch.no_grad()
    def dead_expert_count(self, top_k_indices: torch.Tensor) -> int:  # noqa: ARG002
        return 0

    def parameters_per_expert(self) -> int:
        return sum(parameter.numel() for parameter in self.expert.parameters())

    def dormant_parameters(self) -> int:
        """Always zero: a dense block activates everything it owns."""
        return 0

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, dense=True, "
            f"capacity_multiplier={self.capacity_multiplier}"
        )
