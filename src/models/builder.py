"""The hierarchical seed classifier (paper Section 5) and its SwinV2 encoder.

Dataflow, with the paper's equation numbers::

    x  --SwinV2--> tokens --projection--> z                              (Eq. 4)
    z_pooled --seed_type_classifier-->  s in R^4                         (Eq. 5)
    p_s = softmax(s)                                                     (Eq. 6)
    h   = sum_{i in Top-K} G_i E_i(z)                                    (Eq. 8)
    h'  = h + gamma * P(p_s)     or    h' = FiLM(h ; g_hidden(z))        (Eq. 9)
    a   = softmax(Q K^T / sqrt(d)) V,   Q = h',  K = V = h               (Eq. 11)
    h'' = LayerNorm(a + Q)                                               (Eq. 12)
    e   = SubVarietyEmbedding(pool(h''))
    ArcFace(e, y) -> logits, margin_logits                               (Eq. 13)

Routing granularity is the head's most consequential setting
------------------------------------------------------------

``token_mode="grid"`` (revision default) keeps SwinV2's final-stage ``8x8`` token
grid all the way through the MoE and the cross-attention, and pools only
afterwards. ``token_mode="pooled"`` reproduces the submitted manuscript, which
mean-pools before the head. Four things follow from the choice:

* **Attention.** Over a length-1 sequence ``softmax(QK^T/sqrt(d))`` is
  identically ``1``, so Eqs. 11-12 reduce to an affine map and the ``Q``/``K``
  projections get exactly zero gradient forever. In pooled mode this head
  therefore does not *build* those projections -- see
  :mod:`src.models.components.cross_attention`. In grid mode the attention is
  real, ``num_heads`` matters, and ``attn_weights`` is a plottable spatial map.
* **Routing statistics.** Pooled mode fills ``batch x K`` routing slots per step
  (32 at the configured batch size); grid mode fills ``batch x 64 x K``.
* **Fine-grained evidence.** Mean-pooling a 64-token grid before a 27-way
  fine-grained task discards the localised texture that separates rice
  sub-varieties.
* **Parameter honesty.** Nothing in the head is allocated that no configuration
  can reach, so ``ParameterReport.active`` means what it says.

Three details are easy to get wrong and are worth stating explicitly:

* **The MoE consumes ``z``, not the projected seed-type vector.** Eq. 8 is
  written over ``z``, the DINO embedding. Routing the seed-type projection into
  the experts instead would mean the experts never see the image. What the
  *gate* sees is separately configurable (``gate_conditioning``); the experts
  always see the image.
* **The residual adds ``P(p_s)``, not ``P(s)``.** Eq. 9 projects the *softmax
  probabilities* from Eq. 6, so the residual stays bounded and scale-stable
  regardless of how confident stage 1 is.
* **``z`` is produced by the encoder, not the head.** :class:`DinoV2SwinV2Encoder`
  owns the width reduction from the backbone's native channels to the 384 of
  Eq. 4, so ``encoder(images).shape[-1] == 384`` holds for every SwinV2 variant.

**Backbone standardisation.** Stage-1 pretraining and the hierarchical head both
run exclusively on Swin Transformer V2; :func:`validate_swinv2_name` enforces it
at construction time. The comparative ViT-S/14 path from the submitted
manuscript has been removed, so no run can silently fall back to a different
encoder.

**Component toggles.** Booleans under ``model.head`` switch off one architectural
ingredient each, for the component-wise ablation suite:

======================  ==========================================================
Flag                    Effect when ``False``
======================  ==========================================================
``use_moe``             Top-2 router replaced by one dense block
``use_arcface``         ArcFace replaced by ``sub_head_variant``'s alternative
``use_residual``        ``h' = h``; the Eq. 9 seed-type fusion is not built
``use_cross_attention`` ``h'' = h'``; Eqs. 11-12 are skipped
``use_kl_loss``         Consumed by the loss builder, not the model
======================  ==========================================================

Each disabled branch is *not allocated*, so the parameter counts reported for an
ablation reflect the smaller model rather than dead weights. Note that several of
these change more than one factor at a time; ``component_flags()`` and the
suite's ``loss_flags`` record what actually differed, and
``architecture/06_ABLATION_ENGINE.md`` tabulates it.

:class:`HierarchicalOutput` carries every intermediate the losses and the
metrics need, so adding a term never requires changing a tuple's arity.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.arcface_head import ArcFaceHead, NormFaceHead
from src.models.components.classifiers import (
    LinearSubVarietyHead,
    SeedTypeClassifier,
    SubVarietyEmbedding,
)
from src.models.components.cross_attention import CrossAttention, resolve_attention_mode
from src.models.components.moe_layer import (
    DEFAULT_NUM_EXPERTS,
    DEFAULT_TOP_K,
    DenseExpertBlock,
    MixtureOfExperts,
)
from src.models.components.projections import (
    EmbeddingProjection,
    FiLMFusion,
    LayerScale,
    SeedTypeProjection,
    TokenPooling,
)

#: Width of ``z`` in Eq. 4. Every encoder in this repository emits exactly this.
PAPER_EMBED_DIM = 384

#: timm prefix identifying a Swin Transformer V2 model.
SWINV2_PREFIX = "swinv2"

#: Routing granularities. See the module docstring.
TOKEN_MODES = ("grid", "pooled")

#: Sub-variety head variants, and what each one isolates.
SUB_HEAD_VARIANTS = ("arcface", "normface", "linear")


def validate_swinv2_name(model_name: str) -> str:
    """Return ``model_name`` if it is a SwinV2 variant, else raise.

    Stage-1 pretraining and the hierarchical head are standardised on SwinV2, so
    a mistyped or leftover backbone name must fail loudly at construction rather
    than train for hours against the wrong encoder.
    """
    name = str(model_name)
    if not name.startswith(SWINV2_PREFIX):
        raise ValueError(
            f"Backbone must be a Swin Transformer V2 variant (a timm name starting with "
            f"{SWINV2_PREFIX!r}), got {name!r}. This project is standardised on SwinV2; "
            "supervised comparison backbones such as ResNet-50 or Swin-T belong in "
            "src/models/baselines.py, not on the self-supervised path."
        )
    return name


@dataclass(frozen=True)
class HierarchicalOutput:
    """Every tensor produced by one forward pass of the hierarchical head."""

    seed_type_logits: torch.Tensor
    """``s`` from Eq. 5, shape ``[batch, num_seed_types]``."""

    seed_type_probs: torch.Tensor
    """``p_s`` from Eq. 6, shape ``[batch, num_seed_types]``."""

    embedding: torch.Tensor
    """``z`` from Eq. 4, shape ``[batch, embed_dim]`` (pooled) or
    ``[batch, tokens, embed_dim]`` (grid)."""

    moe_features: torch.Tensor
    """``h`` from Eq. 8, same shape as ``embedding``."""

    projected_seed: torch.Tensor
    """The Eq. 9 residual actually added, ``h' - h``, same shape as ``h``. Zeros
    when ``use_residual=False``."""

    refined_features: torch.Tensor
    """``h'`` from Eq. 9, same shape as ``h``."""

    attended_features: torch.Tensor
    """``h''`` from Eq. 12, pooled to ``[batch, embed_dim]``."""

    sub_embeddings: torch.Tensor
    """ArcFace input embedding, shape ``[batch, embed_dim]``."""

    sub_logits: torch.Tensor
    """``s * cos(theta)`` with **no** margin. Use these for prediction and KL."""

    sub_margin_logits: torch.Tensor
    """``s * cos(theta + m)`` on the target class. Use these for the ArcFace CE.
    Equals ``sub_logits`` when no labels were supplied, always when the head is
    not ArcFace, and during the margin warm-up."""

    gate_logits: torch.Tensor
    """Pre-softmax router output, shape ``[routing_tokens, num_experts]``. The
    router z-loss needs logits, not probabilities."""

    gate_probs: torch.Tensor
    """Full gate distribution ``G``, shape ``[routing_tokens, num_experts]``."""

    top_k_indices: torch.Tensor
    """Selected expert indices, shape ``[routing_tokens, top_k]``."""

    top_k_weights: torch.Tensor
    """Renormalised weights of the selected experts."""

    dispatch_weights: torch.Tensor
    """``top_k_weights`` scattered over all experts, zero elsewhere."""

    seed_hidden: torch.Tensor | None = None
    """Pre-softmax activation of the seed classifier, ``[batch, hidden_dim]``.
    What FiLM fusion conditions on."""

    tokens_per_sample: int = 1
    """Routing tokens contributed by each image. ``routing_tokens = batch *
    tokens_per_sample``; consumers wanting per-image routing statistics reshape
    by this rather than assuming one row per image."""

    attn_weights: torch.Tensor | None = None
    """Cross-attention map, or ``None`` when not requested, disabled, or
    degenerate (pooled mode, where it would be a constant 1.0)."""


class HierarchicalSeedClassifier(nn.Module):
    """Coarse-to-fine seed classifier: seed type -> MoE -> fusion -> attention -> ArcFace.

    Args:
        feature_dim: Width of the features fed into ``forward``. With
            :class:`DinoV2SwinV2Encoder` this already equals ``embed_dim``, and
            the head's input projection collapses to an identity.
        num_seed_types: Coarse classes (4).
        num_sub_varieties: Fine classes (27).
        embed_dim: Width of ``z`` (384). Defaults to ``feature_dim``.
        num_experts: MoE expert count (6).
        top_k: Experts activated per routing token (2 after the revision).
        moe_hidden_dim: Feed-forward width inside each expert.
        num_heads: Attention heads in the experts and in cross-attention.
        dropout_rate: Dropout throughout the head.
        arcface_scale: ArcFace feature scale ``s``; ``"auto"`` uses the AdaCos
            value ``sqrt(2) log(C-1)``.
        arcface_margin: ArcFace angular margin ``m`` in radians, at full strength.
        arcface_easy_margin: Use ArcFace's softer out-of-range fallback.
        arcface_sub_centers: Prototypes per class (1 = standard ArcFace).
        arcface_dynamic: Re-derive ``s`` per step (AdaCos).
        seed_classifier_variant: ``"mlp"`` (paper) or ``"se_gated"`` (ablation).
        sub_embedding_variant: ``"mlp"`` or ``"identity"``.
        seed_projection_depth: Hidden stages inside ``P`` (Eq. 9).
        cross_attention_variant: ``"paper"`` (Eq. 12) or ``"gated"`` (ablation).
        moe_sparse_dispatch: Run each expert only on its routed tokens.
        input_projection_hidden_dim: Optional hidden width for the ``z`` projection.
        token_mode: ``"grid"`` or ``"pooled"``; see the module docstring.
        token_pooling: How the grid is collapsed after the head.
        fusion_mode: ``"additive"`` (Eq. 9) or ``"film"``.
        residual_layer_scale: LayerScale init for the additive residual. ``None``
            disables the gain, reproducing the submitted free residual.
        router_mode: ``"learned"``, ``"hash"`` or ``"uniform"``.
        gate_conditioning: Feed the detached ``p_s`` to the router, making the
            MoE genuinely hierarchical rather than a flat router beside a coarse
            head. The *experts* still consume ``z`` either way.
        router_noise_std: Initial gating-noise scale, annealed by the trainer.
        dense_capacity_multiplier: Feed-forward widening of the ``use_moe=False``
            block, so it can be capacity-matched to Top-K.
        sub_head_variant: ``"arcface"``, ``"normface"`` (margin removed, geometry
            kept) or ``"linear"``.
        use_moe / use_arcface / use_residual / use_cross_attention: Ablation
            toggles; see the module docstring.
    """

    def __init__(
        self,
        feature_dim: int,
        num_seed_types: int,
        num_sub_varieties: int,
        embed_dim: int | None = None,
        num_experts: int = DEFAULT_NUM_EXPERTS,
        top_k: int = DEFAULT_TOP_K,
        moe_hidden_dim: int = 512,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        arcface_scale: float | str = "auto",
        arcface_margin: float = 0.5,
        arcface_easy_margin: bool = False,
        arcface_sub_centers: int = 1,
        arcface_dynamic: bool = False,
        seed_classifier_variant: str = "mlp",
        sub_embedding_variant: str = "mlp",
        seed_projection_depth: int = 2,
        cross_attention_variant: str = "paper",
        moe_sparse_dispatch: bool = True,
        input_projection_hidden_dim: int | None = None,
        token_mode: str = "grid",
        token_pooling: str = "attention",
        fusion_mode: str = "additive",
        residual_layer_scale: float | None = 1e-4,
        router_mode: str = "learned",
        gate_conditioning: bool = True,
        router_noise_std: float = 0.0,
        dense_capacity_multiplier: int = 1,
        sub_head_variant: str | None = None,
        use_moe: bool = True,
        use_arcface: bool = True,
        use_residual: bool = True,
        use_cross_attention: bool = True,
    ):
        super().__init__()
        if token_mode not in TOKEN_MODES:
            raise ValueError(f"token_mode must be one of {TOKEN_MODES}, got {token_mode!r}")

        self.feature_dim = int(feature_dim)
        self.embed_dim = int(embed_dim) if embed_dim is not None else self.feature_dim
        self.num_seed_types = int(num_seed_types)
        self.num_sub_varieties = int(num_sub_varieties)
        self.token_mode = token_mode
        self.fusion_mode = str(fusion_mode)
        self.gate_conditioning = bool(gate_conditioning)

        if sub_head_variant is None:
            sub_head_variant = "arcface" if use_arcface else "linear"
        if sub_head_variant not in SUB_HEAD_VARIANTS:
            raise ValueError(f"sub_head_variant must be one of {SUB_HEAD_VARIANTS}, got {sub_head_variant!r}")
        self.sub_head_variant = sub_head_variant

        self.use_moe = bool(use_moe)
        self.use_arcface = sub_head_variant == "arcface"
        self.use_residual = bool(use_residual)
        self.use_cross_attention = bool(use_cross_attention)

        # Eq. 4. The encoder normally hands over a 384-D z already, in which case
        # this is a genuine identity and costs nothing; the projection stays for
        # callers that feed raw backbone features (feature dumps, notebooks).
        if self.feature_dim == self.embed_dim and input_projection_hidden_dim is None:
            self.input_projection: nn.Module = nn.Identity()
        else:
            self.input_projection = EmbeddingProjection(
                in_dim=self.feature_dim,
                out_dim=self.embed_dim,
                hidden_dim=input_projection_hidden_dim,
                dropout=dropout_rate,
                use_norm=True,
            )

        # Eqs. 5-6.
        self.seed_type_classifier = SeedTypeClassifier(
            feature_dim=self.embed_dim,
            num_seed_types=self.num_seed_types,
            dropout_rate=dropout_rate,
            variant=seed_classifier_variant,
        )

        # Attention is only non-degenerate when there is more than one token.
        token_mixing = resolve_attention_mode(token_mode)

        # Eq. 8, or its dense replacement.
        if self.use_moe:
            self.moe: nn.Module = MixtureOfExperts(
                embed_dim=self.embed_dim,
                num_experts=num_experts,
                mlp_dim=moe_hidden_dim,
                top_k=top_k,
                num_heads=num_heads,
                dropout=dropout_rate,
                sparse_dispatch=moe_sparse_dispatch,
                token_mixing=token_mixing,
                router_mode=router_mode,
                gate_condition_dim=self.num_seed_types if self.gate_conditioning else 0,
                noise_std=router_noise_std,
            )
        else:
            self.gate_conditioning = False
            self.moe = DenseExpertBlock(
                embed_dim=self.embed_dim,
                mlp_dim=moe_hidden_dim,
                num_heads=num_heads,
                dropout=dropout_rate,
                token_mixing=token_mixing,
                capacity_multiplier=dense_capacity_multiplier,
            )

        # Eq. 9, only allocated when the residual is in use.
        self.seed_projection: nn.Module | None = None
        self.residual_scale: nn.Module | None = None
        self.film: nn.Module | None = None
        if self.use_residual:
            if self.fusion_mode == "film":
                self.film = FiLMFusion(
                    condition_dim=self.seed_type_classifier.hidden_dim,
                    embed_dim=self.embed_dim,
                    dropout=dropout_rate,
                )
            elif self.fusion_mode == "additive":
                self.seed_projection = SeedTypeProjection(
                    num_seed_types=self.num_seed_types,
                    embed_dim=self.embed_dim,
                    depth=seed_projection_depth,
                    dropout=dropout_rate,
                )
                if residual_layer_scale is not None:
                    self.residual_scale = LayerScale(self.embed_dim, float(residual_layer_scale))
            else:
                raise ValueError(f"fusion_mode must be 'additive' or 'film', got {self.fusion_mode!r}")

        # Eqs. 11-12, likewise.
        self.cross_attention = (
            CrossAttention(
                dim=self.embed_dim,
                num_heads=num_heads,
                dropout=dropout_rate,
                variant=cross_attention_variant,
                mode=token_mixing,
            )
            if self.use_cross_attention
            else None
        )

        # Grid mode pools after the head; pooled mode has nothing to pool.
        self.token_pooling = TokenPooling(self.embed_dim, mode=token_pooling) if token_mode == "grid" else None

        # Eq. 13, or one of its two controls.
        self.sub_variety_embedding = SubVarietyEmbedding(
            feature_dim=self.embed_dim,
            dropout_rate=dropout_rate,
            variant=sub_embedding_variant,
        )
        if sub_head_variant == "arcface":
            self.arcface: nn.Module = ArcFaceHead(
                feature_dim=self.embed_dim,
                num_classes=self.num_sub_varieties,
                scale=arcface_scale,
                margin=arcface_margin,
                easy_margin=arcface_easy_margin,
                sub_centers=arcface_sub_centers,
                dynamic=arcface_dynamic,
            )
        elif sub_head_variant == "normface":
            self.arcface = NormFaceHead(
                feature_dim=self.embed_dim,
                num_classes=self.num_sub_varieties,
                scale=arcface_scale,
            )
        else:
            self.arcface = LinearSubVarietyHead(
                feature_dim=self.embed_dim,
                num_classes=self.num_sub_varieties,
            )

    # ---------------------------------------------------------------- schedules

    def set_margin_scale(self, value: float) -> None:
        """Ramp the ArcFace margin (0 at step 0, 1 after warm-up)."""
        setter = getattr(self.arcface, "set_margin_scale", None)
        if setter is not None:
            setter(value)

    def set_router_noise(self, value: float) -> None:
        """Set the gating-noise scale, annealed to 0 during training."""
        setter = getattr(self.moe, "set_noise_scale", None)
        if setter is not None:
            setter(value)

    def materialize_expert_grads(self) -> int:
        """Give unrouted expert parameters a zero grad; see the MoE docstring."""
        materializer = getattr(self.moe, "materialize_zero_grads", None)
        return int(materializer()) if materializer is not None else 0

    # ------------------------------------------------------------------- flags

    @property
    def num_experts(self) -> int:
        """Experts actually built: ``num_experts`` normally, 1 when ``use_moe=False``."""
        return int(self.moe.num_experts)

    @property
    def top_k(self) -> int:
        """Experts activated per routing token."""
        return int(self.moe.top_k)

    def component_flags(self) -> dict[str, Any]:
        """Everything the ablation summary needs to tell two runs apart.

        The submitted version reported only the four architectural booleans,
        which made a ``wo_kl`` run byte-identical to ``full_model`` in
        ``summary.json``. Every axis that a variant can move is recorded here, so
        a run always leaves a machine-readable trace of what it actually was.
        """
        return {
            "use_moe": self.use_moe,
            "use_arcface": self.use_arcface,
            "use_residual": self.use_residual,
            "use_cross_attention": self.use_cross_attention,
            "token_mode": self.token_mode,
            "fusion_mode": self.fusion_mode if self.use_residual else "none",
            "sub_head_variant": self.sub_head_variant,
            "router_mode": str(getattr(self.moe, "router_mode", "none")),
            "gate_conditioning": self.gate_conditioning,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "dense_capacity_multiplier": int(getattr(self.moe, "capacity_multiplier", 1)),
        }

    # ----------------------------------------------------------------- forward

    def forward(
        self,
        features: torch.Tensor,
        sub_variety_labels: torch.Tensor | None = None,
        need_attn_weights: bool = False,
    ) -> HierarchicalOutput:
        """Run the full coarse-to-fine cascade.

        Args:
            features: Encoder output, ``[batch, feature_dim]`` in pooled mode or
                ``[batch, tokens, feature_dim]`` in grid mode.
            sub_variety_labels: Targets used to place the ArcFace angular margin.
                Pass them during training; omit at inference, where
                ``sub_margin_logits`` then equals ``sub_logits``.
            need_attn_weights: Also return the cross-attention map. Only
                meaningful in grid mode.
        """
        if features.ndim not in (2, 3):
            raise ValueError(
                f"Expected [batch, feature_dim] or [batch, tokens, feature_dim], got {tuple(features.shape)}"
            )
        if self.token_mode == "pooled" and features.ndim == 3 and features.shape[1] != 1:
            # Caught here rather than several blocks later, where it would
            # surface as an opaque complaint from the affine attention path.
            raise ValueError(
                f"token_mode='pooled' expects one token per image, got {features.shape[1]}. "
                "Build the head with token_mode='grid' (and an encoder to match) to route "
                "over the token grid."
            )
        if self.token_mode == "grid" and features.ndim == 2:
            # Tolerated so a pooled feature dump can still be scored, but it
            # means the attention modules see one token and are affine.
            features = features.unsqueeze(1)

        embedding = self.input_projection(features)  # z, Eq. 4
        # The coarse task is a whole-image judgement, so Eq. 5 reads a pooled z
        # even in grid mode.
        pooled_embedding = embedding.mean(dim=1) if embedding.ndim == 3 else embedding

        seed_type_logits, seed_hidden = self.seed_type_classifier.forward_with_hidden(pooled_embedding)
        seed_type_probs = F.softmax(seed_type_logits, dim=-1)  # p_s, Eq. 6

        # Eq. 8 -- experts always consume z. The *gate* may additionally see the
        # detached coarse posterior, which is what makes the MoE hierarchical
        # without letting the router's gradient reshape the coarse head.
        gate_condition = seed_type_probs.detach() if self.gate_conditioning else None
        moe = self.moe(embedding, gate_condition=gate_condition)

        if not self.use_residual:
            # Zeros rather than None keeps HierarchicalOutput's contract total.
            projected_seed = torch.zeros_like(moe.features)
            refined_features = moe.features
        elif self.film is not None:
            refined_features, projected_seed = self.film(moe.features, seed_hidden)
        else:
            projection = self.seed_projection(seed_type_probs)  # P(p_s)
            if moe.features.ndim == 3:
                projection = projection.unsqueeze(1).expand_as(moe.features)
            if self.residual_scale is not None:
                projection = self.residual_scale(projection)
            projected_seed = projection
            refined_features = moe.features + projection  # h', Eq. 9

        if self.cross_attention is not None:
            # Eq. 11: Q = h' (seed-type aware), K = V = h (raw MoE feature).
            query = refined_features if refined_features.ndim == 3 else refined_features.unsqueeze(1)
            keys = moe.features if moe.features.ndim == 3 else moe.features.unsqueeze(1)
            attention = self.cross_attention(
                query=query,
                key=keys,
                value=keys,
                need_weights=need_attn_weights,
            )
            attended = attention.features  # h'', Eq. 12
            attn_weights = attention.attn_weights
        else:
            attended = refined_features if refined_features.ndim == 3 else refined_features.unsqueeze(1)
            attn_weights = None

        # Pool once, at the end, so the head worked on tokens rather than on a
        # single averaged vector.
        if self.token_pooling is not None:
            attended_features = self.token_pooling(attended)
        else:
            attended_features = attended.squeeze(1) if attended.ndim == 3 else attended

        sub_embeddings = self.sub_variety_embedding(attended_features)
        sub_logits, sub_margin_logits = self.arcface(sub_embeddings, sub_variety_labels)  # Eq. 13

        return HierarchicalOutput(
            seed_type_logits=seed_type_logits,
            seed_type_probs=seed_type_probs,
            embedding=embedding,
            moe_features=moe.features,
            projected_seed=projected_seed,
            refined_features=refined_features,
            attended_features=attended_features,
            sub_embeddings=sub_embeddings,
            sub_logits=sub_logits,
            sub_margin_logits=sub_margin_logits,
            gate_logits=moe.gate_logits,
            gate_probs=moe.gate_probs,
            top_k_indices=moe.top_k_indices,
            top_k_weights=moe.top_k_weights,
            dispatch_weights=moe.dispatch_weights,
            seed_hidden=seed_hidden,
            tokens_per_sample=moe.tokens_per_sample,
            attn_weights=attn_weights,
        )

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(seed_type_predictions, sub_variety_predictions)`` as index tensors."""
        output = self(features)
        return output.seed_type_logits.argmax(dim=-1), output.sub_logits.argmax(dim=-1)

    def extra_repr(self) -> str:
        flags = ", ".join(f"{key}={value}" for key, value in self.component_flags().items())
        return f"embed_dim={self.embed_dim}, {flags}"


class BackboneFeatureExtractor(nn.Module):
    """Wraps a timm SwinV2 backbone, emitting pooled features or its token grid.

    Emits the backbone's **native** width (1024 for SwinV2-Base, 768 for
    Tiny/Small). :class:`DinoV2SwinV2Encoder` wraps this to reach the 384 of
    Eq. 4; use this class directly only when the raw backbone feature is what
    you want.

    Stage 2 reuses the pretrained encoder. ``freeze=True`` (the default) keeps it
    in eval mode with gradients disabled, which is what this repository's
    two-stage recipe assumes; set ``freeze=False`` to fine-tune the encoder
    jointly with the head, as Section 4 describes.

    ``strict=False`` on checkpoint loading is convenient but silently tolerates a
    mismatched checkpoint. :meth:`load_checkpoint` therefore returns the
    missing/unexpected key report so callers can log it instead of discovering
    the mismatch as unexplained garbage metrics.

    A note on ``dynamic_img_size``: timm accepts the flag for SwinV2 but the
    shifted-window attention still asserts the native resolution, so a 101 px
    crop raises rather than running. That is measured, not assumed --
    ``tests/test_models.py`` pins it -- and it is why stage 1 must upsample local
    crops back to 256. See ``architecture/02_BACKBONE_AND_SSL.md``.

    Args:
        model_name: SwinV2 identifier understood by timm.
        checkpoint_path: Optional pretrained weights to load.
        pretrained: Ask timm for its own pretrained weights.
        dynamic_img_size: timm flag; see the caveat above.
        strict: Strict ``load_state_dict``.
        freeze: Disable gradients and force eval mode.
    """

    def __init__(
        self,
        model_name: str,
        checkpoint_path: str | None = None,
        pretrained: bool = False,
        dynamic_img_size: bool = True,
        strict: bool = False,
        freeze: bool = True,
    ):
        super().__init__()
        self.model_name = validate_swinv2_name(model_name)
        self.frozen = bool(freeze)

        self.backbone = timm.create_model(
            self.model_name,
            pretrained=pretrained,
            num_classes=0,
            dynamic_img_size=dynamic_img_size,
        )

        self.load_report: dict[str, list[str]] | None = None
        if checkpoint_path:
            self.load_report = self.load_checkpoint(checkpoint_path, strict=strict)

        if self.frozen:
            self.eval()
            for parameter in self.parameters():
                parameter.requires_grad = False

    @property
    def feature_dim(self) -> int | None:
        """Backbone output width, or ``None`` if the backbone does not expose it."""
        return getattr(self.backbone, "num_features", getattr(self.backbone, "embed_dim", None))

    def train(self, mode: bool = True):
        # A frozen backbone must never leave eval mode; otherwise BatchNorm running
        # statistics and dropout would keep mutating a module whose weights are
        # supposed to be fixed.
        return super().train(False) if self.frozen else super().train(mode)

    def load_checkpoint(self, checkpoint_path: str, strict: bool = False) -> dict[str, list[str]]:
        """Load pretrained weights and return the key mismatch report."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Backbone checkpoint not found: {path}\n"
                "Run the pretraining stage first (`python main.py pretrain`), or point "
                "model.backbone.checkpoint_path / $SEED_PRETRAIN_BACKBONE at an existing "
                "checkpoint. Pass model.backbone.checkpoint_path=null to start from scratch."
            )
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = self._extract_backbone_state_dict(checkpoint)
        incompatible = self.backbone.load_state_dict(state_dict, strict=strict)
        return {
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        }

    def forward(self, x: torch.Tensor, return_tokens: bool = False) -> torch.Tensor:
        """Return pooled ``[batch, feature_dim]`` or the ``[batch, tokens, feature_dim]`` grid."""
        if self.frozen:
            with torch.no_grad():
                return self._extract(x, return_tokens)
        return self._extract(x, return_tokens)

    def _extract(self, x: torch.Tensor, return_tokens: bool) -> torch.Tensor:
        if return_tokens:
            return self._flatten_tokens(self.backbone.forward_features(x))
        return self._pool(self.backbone(x))

    @staticmethod
    def _flatten_tokens(features: torch.Tensor) -> torch.Tensor:
        """Normalise timm's stage output into ``[batch, tokens, channels]``."""
        if features.ndim == 4:
            # timm Swin returns [batch, H, W, channels] from forward_features.
            batch, height, width, channels = features.shape
            return features.reshape(batch, height * width, channels)
        if features.ndim == 3:
            return features
        return features.unsqueeze(1)

    @staticmethod
    def _pool(features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 4:
            # timm Swin returns [batch, H, W, channels] when left unpooled.
            return features.mean(dim=(1, 2))
        if features.ndim == 3:
            return features.mean(dim=1)
        return features

    @staticmethod
    def _extract_backbone_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
        """Accept the several checkpoint layouts this project has produced."""
        if isinstance(checkpoint, Mapping):
            for key in ("student_backbone", "model_state_dict", "state_dict"):
                if key in checkpoint:
                    return checkpoint[key]
            if "student_model" in checkpoint:
                # nn.Sequential(backbone, head) layout: keep the "0." branch.
                student_state = checkpoint["student_model"]
                return {
                    key[2:]: value
                    for key, value in student_state.items()
                    if key.startswith("0.") and len(key) > 2
                }
        return checkpoint


class DinoV2SwinV2Encoder(nn.Module):
    """Self-supervised SwinV2 encoder emitting ``z`` at width 384 (Eq. 4).

    The single place the paper's embedding width is realised. SwinV2-Base emits
    1024 channels and Tiny/Small emit 768; none emits 384, so a learned
    projection is required for Eq. 4 to hold at all. Putting it here rather than
    inside the head means every consumer -- the hierarchical head, the feature
    dump used for t-SNE, the efficiency profiler -- observes the same 384-D
    space, and ``encoder(images).shape[-1] == 384`` is an invariant rather than
    a configuration coincidence.

    ``token_mode="grid"`` returns ``[batch, H*W, 384]`` instead of
    ``[batch, 384]``. The projection is shared across tokens, so the invariant
    on the last axis is unchanged.

    The backbone is frozen by default and the projection is always trainable, so
    ``trainable_parameters()`` is non-empty even in the frozen recipe. Callers
    must add it to the optimizer; :func:`build_optimizer` in the finetune trainer
    does this.

    Args:
        model_name: SwinV2 identifier understood by timm.
        embed_dim: Width of ``z`` (384).
        checkpoint_path: Pretrained backbone weights.
        pretrained: Ask timm for ImageNet weights (normally ``False``; stage 1
            supplies the weights).
        dynamic_img_size: timm flag; see :class:`BackboneFeatureExtractor`.
        strict: Strict ``load_state_dict``.
        freeze_backbone: Freeze the SwinV2 trunk.
        projection_hidden_dim: Optional hidden width inside the projection.
        projection_dropout: Dropout inside the projection's hidden variant.
        token_mode: ``"grid"`` to emit the token grid, ``"pooled"`` for a vector.
    """

    def __init__(
        self,
        model_name: str,
        embed_dim: int = PAPER_EMBED_DIM,
        checkpoint_path: str | None = None,
        pretrained: bool = False,
        dynamic_img_size: bool = True,
        strict: bool = False,
        freeze_backbone: bool = True,
        projection_hidden_dim: int | None = None,
        projection_dropout: float = 0.0,
        token_mode: str = "grid",
    ):
        super().__init__()
        if token_mode not in TOKEN_MODES:
            raise ValueError(f"token_mode must be one of {TOKEN_MODES}, got {token_mode!r}")
        self.encoder = BackboneFeatureExtractor(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            pretrained=pretrained,
            dynamic_img_size=dynamic_img_size,
            strict=strict,
            freeze=freeze_backbone,
        )
        backbone_dim = self.encoder.feature_dim
        if backbone_dim is None:
            raise ValueError(
                f"Backbone {model_name!r} does not expose num_features/embed_dim, so the "
                "projection width to z cannot be determined."
            )

        self.backbone_dim = int(backbone_dim)
        self.embed_dim = int(embed_dim)
        self.frozen_backbone = bool(freeze_backbone)
        self.token_mode = token_mode
        self.projection = EmbeddingProjection(
            in_dim=self.backbone_dim,
            out_dim=self.embed_dim,
            hidden_dim=projection_hidden_dim,
            dropout=projection_dropout,
            use_norm=True,
        )

    @property
    def load_report(self) -> dict[str, list[str]] | None:
        """Checkpoint key-mismatch report from the wrapped backbone."""
        return self.encoder.load_report

    @property
    def feature_dim(self) -> int:
        """Output width, which is always ``embed_dim``. Named for parity with
        :class:`BackboneFeatureExtractor`, whose ``feature_dim`` is the native
        backbone width."""
        return self.embed_dim

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Parameters the optimizer must own: the projection, plus the trunk when unfrozen."""
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return ``z``: ``[batch, 384]`` pooled, or ``[batch, tokens, 384]`` on the grid."""
        return self.projection(self.encoder(images, return_tokens=self.token_mode == "grid"))

    def extra_repr(self) -> str:
        return (
            f"backbone_dim={self.backbone_dim} -> embed_dim={self.embed_dim}, "
            f"token_mode={self.token_mode}, frozen_backbone={self.frozen_backbone}"
        )


def build_hierarchical_moe(cfg: Any) -> HierarchicalSeedClassifier:
    """Instantiate :class:`HierarchicalSeedClassifier` from a ``model.head`` node."""

    def get(key: str, default: Any) -> Any:
        value = getattr(cfg, key, default)
        return default if value is None else value

    hidden = getattr(cfg, "input_projection_hidden_dim", None)
    layer_scale = getattr(cfg, "residual_layer_scale", 1e-4)
    return HierarchicalSeedClassifier(
        feature_dim=int(cfg.feature_dim),
        num_seed_types=int(cfg.num_seed_types),
        num_sub_varieties=int(cfg.num_sub_varieties),
        embed_dim=int(get("embed_dim", cfg.feature_dim)),
        num_experts=int(get("num_experts", DEFAULT_NUM_EXPERTS)),
        top_k=int(get("top_k", DEFAULT_TOP_K)),
        moe_hidden_dim=int(get("moe_hidden_dim", 512)),
        num_heads=int(get("num_heads", 8)),
        dropout_rate=float(get("dropout_rate", 0.1)),
        arcface_scale=get("arcface_scale", "auto"),
        arcface_margin=float(get("arcface_margin", 0.5)),
        arcface_easy_margin=bool(get("arcface_easy_margin", False)),
        arcface_sub_centers=int(get("arcface_sub_centers", 1)),
        arcface_dynamic=bool(get("arcface_dynamic", False)),
        seed_classifier_variant=str(get("seed_classifier_variant", "mlp")),
        sub_embedding_variant=str(get("sub_embedding_variant", "mlp")),
        seed_projection_depth=int(get("seed_projection_depth", 2)),
        cross_attention_variant=str(get("cross_attention_variant", "paper")),
        moe_sparse_dispatch=bool(get("moe_sparse_dispatch", True)),
        input_projection_hidden_dim=int(hidden) if hidden is not None else None,
        token_mode=str(get("token_mode", "grid")),
        token_pooling=str(get("token_pooling", "attention")),
        fusion_mode=str(get("fusion_mode", "additive")),
        residual_layer_scale=None if layer_scale is None else float(layer_scale),
        router_mode=str(get("router_mode", "learned")),
        gate_conditioning=bool(get("gate_conditioning", True)),
        router_noise_std=float(get("router_noise_std", 0.0)),
        dense_capacity_multiplier=int(get("dense_capacity_multiplier", 1)),
        sub_head_variant=getattr(cfg, "sub_head_variant", None),
        use_moe=bool(get("use_moe", True)),
        use_arcface=bool(get("use_arcface", True)),
        use_residual=bool(get("use_residual", True)),
        use_cross_attention=bool(get("use_cross_attention", True)),
    )


def build_encoder(
    backbone_cfg: Any,
    embed_dim: int = PAPER_EMBED_DIM,
    token_mode: str = "grid",
) -> DinoV2SwinV2Encoder:
    """Instantiate :class:`DinoV2SwinV2Encoder` from a ``model.backbone`` node.

    Args:
        backbone_cfg: The ``model.backbone`` config node.
        embed_dim: Width of ``z``. Pass ``cfg.model.head.embed_dim`` so the
            encoder and the head cannot disagree.
        token_mode: Pass ``cfg.model.head.token_mode`` for the same reason -- an
            encoder emitting a grid into a pooled head (or the reverse) is a
            silent misconfiguration the head would tolerate.
    """
    return DinoV2SwinV2Encoder(
        model_name=str(backbone_cfg.name),
        embed_dim=int(embed_dim),
        checkpoint_path=getattr(backbone_cfg, "checkpoint_path", None),
        pretrained=bool(getattr(backbone_cfg, "pretrained", False)),
        dynamic_img_size=bool(getattr(backbone_cfg, "dynamic_img_size", True)),
        strict=bool(getattr(backbone_cfg, "checkpoint_strict", False)),
        freeze_backbone=bool(getattr(backbone_cfg, "freeze", True)),
        projection_hidden_dim=getattr(backbone_cfg, "projection_hidden_dim", None),
        projection_dropout=float(getattr(backbone_cfg, "projection_dropout", 0.0) or 0.0),
        token_mode=str(token_mode),
    )


def build_feature_extractor(cfg: Any) -> BackboneFeatureExtractor:
    """Instantiate the bare SwinV2 trunk from a ``model.backbone`` node.

    Returns the backbone at its native width. Prefer :func:`build_encoder`,
    which adds the Eq. 4 projection; this exists for callers that genuinely want
    the unprojected feature.
    """
    return BackboneFeatureExtractor(
        model_name=str(cfg.name),
        checkpoint_path=getattr(cfg, "checkpoint_path", None),
        pretrained=bool(getattr(cfg, "pretrained", False)),
        dynamic_img_size=bool(getattr(cfg, "dynamic_img_size", True)),
        strict=bool(getattr(cfg, "checkpoint_strict", False)),
        freeze=bool(getattr(cfg, "freeze", True)),
    )
