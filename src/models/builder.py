"""The hierarchical seed classifier (paper Section 5) and its SwinV2 encoder.

Dataflow, with the paper's equation numbers::

    x  --SwinV2--> pooled --projection--> z in R^384                     (Eq. 4)
    z  --seed_type_classifier-->  s in R^4                               (Eq. 5)
    p_s = softmax(s)                                                     (Eq. 6)
    h   = sum_{i in Top-K} G_i E_i(z)                                    (Eq. 8)
    h'  = h + P(p_s)                                                     (Eq. 9)
    a   = softmax(Q K^T / sqrt(d)) V,   Q = h',  K = V = h               (Eq. 11)
    h'' = LayerNorm(a + Q)                                               (Eq. 12)
    e   = SubVarietyEmbedding(h'')
    ArcFace(e, y) -> logits, margin_logits                               (Eq. 13)

Three details are easy to get wrong and are worth stating explicitly:

* **The MoE consumes ``z``, not the projected seed-type vector.** Eq. 8 is
  written over ``z``, the DINO embedding. Routing the seed-type projection into
  the experts instead would mean the experts never see the image.
* **The residual adds ``P(p_s)``, not ``P(s)``.** Eq. 9 projects the *softmax
  probabilities* from Eq. 6, so the residual stays bounded and scale-stable
  regardless of how confident stage 1 is.
* **``z`` is produced by the encoder, not the head.** :class:`DinoV2SwinV2Encoder`
  owns the width reduction from the backbone's native channels to the 384 of
  Eq. 4, so ``encoder(images).shape[-1] == 384`` holds for every SwinV2 variant
  and every consumer -- head, baselines, t-SNE, profiler -- sees one embedding
  space.

**Backbone standardisation.** DINOv2 pretraining and the hierarchical head both
run exclusively on Swin Transformer V2; :func:`validate_swinv2_name` enforces it
at construction time. The comparative ViT-S/14 path from the submitted
manuscript has been removed, so no run can silently fall back to a different
encoder.

**Component toggles.** Five booleans switch off one architectural ingredient
each, for the component-wise ablation suite:

======================  ==========================================================
Flag                    Effect when ``False``
======================  ==========================================================
``use_moe``             Top-2 router replaced by one dense transformer block
``use_arcface``         ArcFace replaced by a linear head (objective becomes CE)
``use_residual``        ``h' = h``; the Eq. 9 seed-type fusion is not built
``use_cross_attention`` ``h'' = h'``; Eqs. 11-12 are skipped
``use_kl_loss``         Consumed by the loss builder, not the model
======================  ==========================================================

Each disabled branch is *not allocated*, so the parameter counts reported for an
ablation reflect the smaller model rather than dead weights.

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

from src.models.components.arcface_head import ArcFaceHead
from src.models.components.classifiers import (
    LinearSubVarietyHead,
    SeedTypeClassifier,
    SubVarietyEmbedding,
)
from src.models.components.cross_attention import CrossAttention
from src.models.components.moe_layer import (
    DEFAULT_NUM_EXPERTS,
    DEFAULT_TOP_K,
    DenseExpertBlock,
    MixtureOfExperts,
)
from src.models.components.projections import EmbeddingProjection, SeedTypeProjection

#: Width of ``z`` in Eq. 4. Every encoder in this repository emits exactly this.
PAPER_EMBED_DIM = 384

#: timm prefix identifying a Swin Transformer V2 model.
SWINV2_PREFIX = "swinv2"


def validate_swinv2_name(model_name: str) -> str:
    """Return ``model_name`` if it is a SwinV2 variant, else raise.

    DINOv2 pretraining and the hierarchical head are standardised on SwinV2, so
    a mistyped or leftover backbone name must fail loudly at construction rather
    than train for hours against the wrong encoder.
    """
    name = str(model_name)
    if not name.startswith(SWINV2_PREFIX):
        raise ValueError(
            f"Backbone must be a Swin Transformer V2 variant (a timm name starting with "
            f"{SWINV2_PREFIX!r}), got {name!r}. This project is standardised on SwinV2; "
            "supervised comparison backbones such as ResNet-50 or Swin-T belong in "
            "src/models/baselines.py, not on the DINOv2 path."
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
    """``z`` from Eq. 4, shape ``[batch, embed_dim]``."""

    moe_features: torch.Tensor
    """``h`` from Eq. 8, shape ``[batch, embed_dim]``."""

    projected_seed: torch.Tensor
    """``P(p_s)`` from Eq. 9, shape ``[batch, embed_dim]``. Zeros when
    ``use_residual=False``, which makes the residual cosine term vanish."""

    refined_features: torch.Tensor
    """``h'`` from Eq. 9, shape ``[batch, embed_dim]``."""

    attended_features: torch.Tensor
    """``h''`` from Eq. 12, shape ``[batch, embed_dim]``."""

    sub_embeddings: torch.Tensor
    """ArcFace input embedding, shape ``[batch, embed_dim]``."""

    sub_logits: torch.Tensor
    """``s * cos(theta)`` with **no** margin. Use these for prediction and KL."""

    sub_margin_logits: torch.Tensor
    """``s * cos(theta + m)`` on the target class. Use these for the ArcFace CE.
    Equals ``sub_logits`` when no labels were supplied, and always when
    ``use_arcface=False``."""

    gate_probs: torch.Tensor
    """Full gate distribution ``G``, shape ``[batch, num_experts]``."""

    top_k_indices: torch.Tensor
    """Selected expert indices, shape ``[batch, top_k]``."""

    top_k_weights: torch.Tensor
    """Renormalised weights of the selected experts, shape ``[batch, top_k]``."""

    dispatch_weights: torch.Tensor
    """``top_k_weights`` scattered over all experts, zero elsewhere."""

    attn_weights: torch.Tensor | None = None
    """Cross-attention map, or ``None`` when not requested or disabled."""


class HierarchicalSeedClassifier(nn.Module):
    """Coarse-to-fine seed classifier: seed type -> MoE -> cross-attention -> ArcFace.

    Args:
        feature_dim: Width of the vector fed into ``forward``. With
            :class:`DinoV2SwinV2Encoder` this already equals ``embed_dim``, and
            the head's input projection collapses to an identity.
        num_seed_types: Coarse classes (4).
        num_sub_varieties: Fine classes (27).
        embed_dim: Width of ``z`` (384). Defaults to ``feature_dim``.
        num_experts: MoE expert count (6).
        top_k: Experts activated per sample (2 after the revision).
        moe_hidden_dim: Feed-forward width inside each expert.
        num_heads: Attention heads in the experts and in cross-attention.
        dropout_rate: Dropout throughout the head.
        arcface_scale: ArcFace feature scale ``s``.
        arcface_margin: ArcFace angular margin ``m`` in radians.
        arcface_easy_margin: Use ArcFace's softer out-of-range fallback.
        seed_classifier_variant: ``"mlp"`` (paper) or ``"se_gated"`` (ablation).
        sub_embedding_variant: ``"mlp"`` or ``"identity"``.
        seed_projection_depth: Hidden stages inside ``P`` (Eq. 9).
        cross_attention_variant: ``"paper"`` (Eq. 12) or ``"gated"`` (ablation).
        moe_sparse_dispatch: Run each expert only on its routed samples.
        input_projection_hidden_dim: Optional hidden width for the ``z`` projection.
        use_moe: Sparse Top-K routing. ``False`` substitutes one dense block.
        use_arcface: Angular-margin head. ``False`` substitutes a linear head.
        use_residual: Eq. 9 seed-type fusion.
        use_cross_attention: Eqs. 11-12 refinement.
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
        arcface_scale: float = 30.0,
        arcface_margin: float = 0.5,
        arcface_easy_margin: bool = False,
        seed_classifier_variant: str = "mlp",
        sub_embedding_variant: str = "mlp",
        seed_projection_depth: int = 2,
        cross_attention_variant: str = "paper",
        moe_sparse_dispatch: bool = True,
        input_projection_hidden_dim: int | None = None,
        use_moe: bool = True,
        use_arcface: bool = True,
        use_residual: bool = True,
        use_cross_attention: bool = True,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.embed_dim = int(embed_dim) if embed_dim is not None else self.feature_dim
        self.num_seed_types = int(num_seed_types)
        self.num_sub_varieties = int(num_sub_varieties)

        self.use_moe = bool(use_moe)
        self.use_arcface = bool(use_arcface)
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
            )
        else:
            self.moe = DenseExpertBlock(
                embed_dim=self.embed_dim,
                mlp_dim=moe_hidden_dim,
                num_heads=num_heads,
                dropout=dropout_rate,
            )

        # Eq. 9, only allocated when the residual is in use.
        self.seed_projection = (
            SeedTypeProjection(
                num_seed_types=self.num_seed_types,
                embed_dim=self.embed_dim,
                depth=seed_projection_depth,
                dropout=dropout_rate,
            )
            if self.use_residual
            else None
        )

        # Eqs. 11-12, likewise.
        self.cross_attention = (
            CrossAttention(
                dim=self.embed_dim,
                num_heads=num_heads,
                dropout=dropout_rate,
                variant=cross_attention_variant,
            )
            if self.use_cross_attention
            else None
        )

        # Eq. 13, or its plain-softmax replacement.
        self.sub_variety_embedding = SubVarietyEmbedding(
            feature_dim=self.embed_dim,
            dropout_rate=dropout_rate,
            variant=sub_embedding_variant,
        )
        if self.use_arcface:
            self.arcface: nn.Module = ArcFaceHead(
                feature_dim=self.embed_dim,
                num_classes=self.num_sub_varieties,
                scale=arcface_scale,
                margin=arcface_margin,
                easy_margin=arcface_easy_margin,
            )
        else:
            self.arcface = LinearSubVarietyHead(
                feature_dim=self.embed_dim,
                num_classes=self.num_sub_varieties,
            )

    @property
    def num_experts(self) -> int:
        """Experts actually built: ``num_experts`` normally, 1 when ``use_moe=False``."""
        return int(self.moe.num_experts)

    @property
    def top_k(self) -> int:
        """Experts activated per sample."""
        return int(self.moe.top_k)

    def component_flags(self) -> dict[str, bool]:
        """The five toggles, for logging and for the ablation summary table."""
        return {
            "use_moe": self.use_moe,
            "use_arcface": self.use_arcface,
            "use_residual": self.use_residual,
            "use_cross_attention": self.use_cross_attention,
        }

    def forward(
        self,
        features: torch.Tensor,
        sub_variety_labels: torch.Tensor | None = None,
        need_attn_weights: bool = False,
    ) -> HierarchicalOutput:
        """Run the full coarse-to-fine cascade.

        Args:
            features: Encoder output ``z``, shape ``[batch, feature_dim]``.
            sub_variety_labels: Targets used to place the ArcFace angular margin.
                Pass them during training; omit at inference, where
                ``sub_margin_logits`` then equals ``sub_logits``.
            need_attn_weights: Also return the cross-attention map.
        """
        if features.ndim != 2:
            raise ValueError(f"Expected [batch, feature_dim] features, got shape {tuple(features.shape)}")

        embedding = self.input_projection(features)  # z, Eq. 4

        seed_type_logits = self.seed_type_classifier(embedding)  # s, Eq. 5
        seed_type_probs = F.softmax(seed_type_logits, dim=-1)  # p_s, Eq. 6

        moe = self.moe(embedding)  # h, Eq. 8 -- routed on z, per the paper

        if self.seed_projection is not None:
            projected_seed = self.seed_projection(seed_type_probs)  # P(p_s)
            refined_features = moe.features + projected_seed  # h', Eq. 9
        else:
            # Zeros rather than None keeps HierarchicalOutput's contract total,
            # and makes the residual cosine term evaluate to 0 instead of NaN.
            projected_seed = torch.zeros_like(moe.features)
            refined_features = moe.features

        if self.cross_attention is not None:
            # Eq. 11: Q = h' (seed-type aware), K = V = h (raw MoE feature).
            attention = self.cross_attention(
                query=refined_features.unsqueeze(1),
                key=moe.features.unsqueeze(1),
                value=moe.features.unsqueeze(1),
                need_weights=need_attn_weights,
            )
            attended_features = attention.features.squeeze(1)  # h'', Eq. 12
            attn_weights = attention.attn_weights
        else:
            attended_features = refined_features
            attn_weights = None

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
            gate_probs=moe.gate_probs,
            top_k_indices=moe.top_k_indices,
            top_k_weights=moe.top_k_weights,
            dispatch_weights=moe.dispatch_weights,
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
    """Wraps a timm SwinV2 backbone and pools its output to ``[batch, dim]``.

    Emits the backbone's **native** width (1024 for SwinV2-Base, 768 for
    Tiny/Small). :class:`DinoV2SwinV2Encoder` wraps this to reach the 384 of
    Eq. 4; use this class directly only when the raw backbone feature is what
    you want.

    Stage 2 reuses the DINO-pretrained encoder. ``freeze=True`` (the default)
    keeps it in eval mode with gradients disabled, which is what this
    repository's two-stage recipe assumes; set ``freeze=False`` to fine-tune the
    encoder jointly with the head, as Section 4 describes.

    ``strict=False`` on checkpoint loading is convenient but silently tolerates a
    mismatched checkpoint. :meth:`load_checkpoint` therefore returns the
    missing/unexpected key report so callers can log it instead of discovering
    the mismatch as unexplained garbage metrics.

    Args:
        model_name: SwinV2 identifier understood by timm.
        checkpoint_path: Optional DINO-pretrained weights to load.
        pretrained: Ask timm for its own pretrained weights.
        dynamic_img_size: timm flag allowing non-native input resolutions.
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
        """Load DINO-pretrained weights and return the key mismatch report."""
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled features, shape ``[batch, feature_dim]``."""
        if self.frozen:
            with torch.no_grad():
                return self._pool(self.backbone(x))
        return self._pool(self.backbone(x))

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
    """DINOv2-pretrained SwinV2 encoder emitting ``z in R^384`` (Eq. 4).

    The single place the paper's embedding width is realised. SwinV2-Base emits
    1024 channels and Tiny/Small emit 768; none emits 384, so a learned
    projection is required for Eq. 4 to hold at all. Putting it here rather than
    inside the head means every consumer -- the hierarchical head, the feature
    dump used for t-SNE, the efficiency profiler -- observes the same 384-D
    space, and ``encoder(images).shape[-1] == 384`` is an invariant rather than
    a configuration coincidence.

    The backbone is frozen by default and the projection is always trainable, so
    ``trainable_parameters()`` is non-empty even in the frozen recipe. Callers
    must add it to the optimizer; :func:`build_optimizer` in the finetune trainer
    does this.

    Args:
        model_name: SwinV2 identifier understood by timm.
        embed_dim: Width of ``z`` (384).
        checkpoint_path: DINO-pretrained backbone weights.
        pretrained: Ask timm for ImageNet weights (normally ``False``; stage 1
            supplies the weights).
        dynamic_img_size: timm flag allowing non-native input resolutions.
        strict: Strict ``load_state_dict``.
        freeze_backbone: Freeze the SwinV2 trunk.
        projection_hidden_dim: Optional hidden width inside the projection.
        projection_dropout: Dropout inside the projection's hidden variant.
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
    ):
        super().__init__()
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
        """Return ``z``, shape ``[batch, 384]``."""
        return self.projection(self.encoder(images))

    def extra_repr(self) -> str:
        return (
            f"backbone_dim={self.backbone_dim} -> embed_dim={self.embed_dim}, "
            f"frozen_backbone={self.frozen_backbone}"
        )


def build_hierarchical_moe(cfg: Any) -> HierarchicalSeedClassifier:
    """Instantiate :class:`HierarchicalSeedClassifier` from a ``model.head`` node."""

    def get(key: str, default: Any) -> Any:
        value = getattr(cfg, key, default)
        return default if value is None else value

    hidden = getattr(cfg, "input_projection_hidden_dim", None)
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
        arcface_scale=float(get("arcface_scale", 30.0)),
        arcface_margin=float(get("arcface_margin", 0.5)),
        arcface_easy_margin=bool(get("arcface_easy_margin", False)),
        seed_classifier_variant=str(get("seed_classifier_variant", "mlp")),
        sub_embedding_variant=str(get("sub_embedding_variant", "mlp")),
        seed_projection_depth=int(get("seed_projection_depth", 2)),
        cross_attention_variant=str(get("cross_attention_variant", "paper")),
        moe_sparse_dispatch=bool(get("moe_sparse_dispatch", True)),
        input_projection_hidden_dim=int(hidden) if hidden is not None else None,
        use_moe=bool(get("use_moe", True)),
        use_arcface=bool(get("use_arcface", True)),
        use_residual=bool(get("use_residual", True)),
        use_cross_attention=bool(get("use_cross_attention", True)),
    )


def build_encoder(backbone_cfg: Any, embed_dim: int = PAPER_EMBED_DIM) -> DinoV2SwinV2Encoder:
    """Instantiate :class:`DinoV2SwinV2Encoder` from a ``model.backbone`` node.

    Args:
        backbone_cfg: The ``model.backbone`` config node.
        embed_dim: Width of ``z``. Pass ``cfg.model.head.embed_dim`` so the
            encoder and the head cannot disagree.
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
