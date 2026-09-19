"""Supervised comparison baselines for the revision's results table.

Three baselines are needed to isolate what the proposed framework contributes:

``resnet50``
    ImageNet-pretrained ResNet-50, trained end to end. The conventional CNN
    reference point.

``swin_tiny``
    ImageNet-pretrained Swin Transformer V1 Tiny, trained end to end. Shares the
    shifted-window inductive bias with the proposed SwinV2 encoder but is
    trained *supervised from ImageNet weights* rather than self-supervised on
    seed imagery.

``linear_probe`` -- **run this one first**
    The frozen self-supervised encoder with nothing but ``Linear(384, 4)`` and
    ``Linear(384, 27)`` on top, plain cross-entropy. This answers the question a
    reviewer asks before any other: *does any of the head machinery beat a linear
    layer on the same features?* If the full architecture does not clear it by a
    comfortable, seed-stable margin, that is the single most important number in
    the paper.

    Note that ``hierarchical_cce`` is **not** this control: it keeps
    ``use_residual=true``, so it retains the coarse-to-fine link and the
    ``SubVarietyEmbedding`` MLP. It is a composed point in the ablation lattice
    (``wo_moe`` + ``wo_angular_head`` + ``wo_cross_attn`` + ``wo_kl``), not an
    independent baseline.

``swinv2_supervised``
    ImageNet-initialised SwinV2-Base with the *full* hierarchical head and no
    self-supervised stage. The only variant that separates "in-domain
    self-supervised pretraining" from "the architecture". Configured through
    ``conf/experiment/baseline_swinv2_supervised.yaml`` rather than through
    ``model/backbone``, which ``validate_swinv2_name`` reserves for the
    self-supervised path.

**hierarchical CCE**
    Two-stage coarse-to-fine classification with plain cross-entropy at both
    levels and no MoE, no cross-attention and no ArcFace. This one needs *no
    code*: it is the proposed model with four toggles flipped, which is exactly
    what ``conf/experiment/baseline_hierarchical_cce.yaml`` does. Expressing it
    through the same class as the full model is what makes the comparison fair --
    identical data pipeline, identical optimiser, identical evaluation.

Every class here emits a :class:`~src.models.builder.HierarchicalOutput`, so the
entire evaluation stack (metrics, KL alignment, confusion matrices, t-SNE,
trackers) runs against a baseline unmodified.

**Why two heads.** A flat 27-way classifier would make the hierarchical
alignment rate identically 1.0 -- the seed type would be *derived* from the
sub-variety prediction, so the two could not disagree by construction, and the
column would be meaningless. Giving the baselines two independent linear heads
over a shared trunk makes their alignment rate a real measurement, directly
comparable with the proposed model's.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.builder import PAPER_EMBED_DIM, HierarchicalOutput

#: Short names for the supervised comparison backbones, mapped to timm ids.
#
# The first two are the historical pair. The five below them were added for the
# revision, and every one answers a comment on the submitted manuscript:
#
#   vit_small        Reviewer 2's first major comment. The abstract and the
#                    introduction say SwinV2; Table 1 says ViT-S/14. This is the
#                    ViT arm of the comparison they asked for, at ViT-S/16 --
#                    timm has no /14 ViT-S with ImageNet-1k supervised weights,
#                    and the patch size is the one axis of Table 1's claim that
#                    cannot be honoured exactly. State it as ViT-S/16.
#   swinv2_tiny      the *matched* SwinV2 arm of that same comparison: the
#                    proposed trunk, same flat head, same 256 px input, same
#                    schedule. `swinv2_supervised` is NOT this row -- it carries
#                    the full hierarchical head, so it cannot isolate a backbone.
#   deit3_small      the same ViT-S/16 geometry under a modern supervised
#                    recipe, so "ViT loses" cannot be read as "ViT's 2020
#                    training recipe loses".
#   convnext_tiny    Reviewer 1's "recent fine-grained approaches": a modern CNN
#                    at SwinV2-Tiny's parameter count (27.8 M vs 27.6 M).
#   efficientnetv2_s the efficiency-oriented reference point, for the cost table.
#
# Every one of these is ImageNet-supervised and trains end to end. None of them
# reads the stage-1 encoder, so none of them costs any pretraining compute.
BASELINE_MODELS = {
    "resnet50": "resnet50",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "vit_small": "vit_small_patch16_224",
    "deit3_small": "deit3_small_patch16_224",
    "convnext_tiny": "convnext_tiny",
    "efficientnetv2_s": "tf_efficientnetv2_s",
    "swinv2_tiny": "swinv2_tiny_window16_256",
}


class IdentityEncoder(nn.Module):
    """Pass-through stand-in for the encoder in end-to-end runs.

    The finetune trainer is structured as ``encoder(images) -> model(features)``
    because the proposed recipe freezes the encoder. An end-to-end baseline owns
    its own backbone, so it needs that first stage to do nothing. This keeps the
    training loop identical for both instead of branching it.
    """

    #: Declared so the trainer's dimension check has something to read.
    feature_dim = None
    load_report = None
    frozen = False

    def trainable_parameters(self) -> list[nn.Parameter]:
        return []

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images


class FlatSupervisedBaseline(nn.Module):
    """ImageNet-pretrained backbone with two independent linear heads.

    Architecture::

        images --backbone--> pooled --Linear+LayerNorm--> z in R^384
                                                          |-- Linear --> 4 seed types
                                                          |-- Linear --> 27 sub-varieties

    The projection to 384 is not required by the baseline itself; it is there so
    the embeddings this model produces live in the same space as the proposed
    model's, which makes the t-SNE panels and the embedding-dimension column of
    the results table comparable across rows.

    Args:
        model_name: timm identifier, or a key of :data:`BASELINE_MODELS`.
        num_seed_types: Coarse classes (4).
        num_sub_varieties: Fine classes (27).
        pretrained: Load ImageNet weights (the point of a supervised baseline).
        embed_dim: Shared embedding width (384).
        dropout_rate: Dropout before each classification head.
        backbone_kwargs: Extra keyword arguments for ``timm.create_model``,
            in practice ``{"img_size": 256}`` for the fixed-resolution ViT arms.
    """

    def __init__(
        self,
        model_name: str,
        num_seed_types: int,
        num_sub_varieties: int,
        pretrained: bool = True,
        embed_dim: int = PAPER_EMBED_DIM,
        dropout_rate: float = 0.1,
        backbone_kwargs: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        self.model_name = BASELINE_MODELS.get(str(model_name), str(model_name))
        self.num_seed_types = int(num_seed_types)
        self.num_sub_varieties = int(num_sub_varieties)
        self.embed_dim = int(embed_dim)
        self.backbone_kwargs = dict(backbone_kwargs or {})

        # `backbone_kwargs` exists for exactly one reason: `img_size`. A ViT
        # arrives pinned to the resolution it was trained at, and comparing a
        # 224 px ViT against a 256 px SwinV2 would answer "which backbone" with
        # a number that is partly "which input size" -- on a corpus whose crops
        # are a median 61 x 61 px, so the upsampling factor is the dominant
        # thing the resolution changes. timm interpolates the position
        # embedding when `img_size` is passed, which is what lets the ViT arm
        # run at the proposed model's 256 and makes the comparison a backbone
        # comparison. A CNN accepts any size and takes no kwarg at all.
        self.backbone = timm.create_model(
            self.model_name, pretrained=pretrained, num_classes=0, **self.backbone_kwargs
        )
        backbone_dim = getattr(self.backbone, "num_features", None)
        if backbone_dim is None:
            raise ValueError(f"timm model {self.model_name!r} does not expose num_features")
        self.backbone_dim = int(backbone_dim)

        self.projection = nn.Sequential(
            nn.Linear(self.backbone_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.seed_head = nn.Linear(self.embed_dim, self.num_seed_types)
        self.sub_head = nn.Linear(self.embed_dim, self.num_sub_varieties)

    # A one-expert, always-on router, so MoE-shaped consumers see valid tensors.
    @property
    def num_experts(self) -> int:
        return 1

    @property
    def top_k(self) -> int:
        return 1

    def component_flags(self) -> dict[str, Any]:
        flags = _baseline_component_flags(self.model_name)
        # The ViT arms run at an interpolated resolution, and that is a factor
        # of the comparison. Recording it here is what makes a 224 px row
        # machine-distinguishable from a 256 px one in `summary.json`, rather
        # than distinguishable only by reading the experiment file back.
        if self.backbone_kwargs:
            flags["backbone_kwargs"] = dict(self.backbone_kwargs)
        return flags

    def forward(
        self,
        images: torch.Tensor,
        sub_variety_labels: torch.Tensor | None = None,  # noqa: ARG002 - parity with the proposed head
        need_attn_weights: bool = False,  # noqa: ARG002
    ) -> HierarchicalOutput:
        """Return a :class:`HierarchicalOutput` so shared evaluation code applies."""
        pooled = self.backbone(images)
        if pooled.ndim == 4:
            pooled = pooled.mean(dim=(1, 2))
        elif pooled.ndim == 3:
            pooled = pooled.mean(dim=1)

        embedding = self.projection(pooled)
        hidden = self.dropout(embedding)
        return _degenerate_output(
            embedding=embedding,
            seed_type_logits=self.seed_head(hidden),
            sub_logits=self.sub_head(hidden),
        )

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self(images)
        return output.seed_type_logits.argmax(dim=-1), output.sub_logits.argmax(dim=-1)

    def extra_repr(self) -> str:
        parts = [
            f"model_name={self.model_name}",
            f"backbone_dim={self.backbone_dim}",
            f"embed_dim={self.embed_dim}",
        ]
        if self.backbone_kwargs:
            parts.append(f"backbone_kwargs={self.backbone_kwargs}")
        return ", ".join(parts)


class LinearProbeHead(nn.Module):
    """Two linear layers on the frozen encoder's ``z``: the honest floor.

    Unlike :class:`FlatSupervisedBaseline` this owns no backbone. It sits behind
    the *real* self-supervised encoder in the trainer's
    ``encoder(images) -> model(features)`` structure, so it is measured on
    byte-identical features to every other variant in the suite. That is what
    makes it the right control: any gap to the full head is attributable to the
    head, not to a different representation.

    Args:
        feature_dim: Width of ``z`` (384).
        num_seed_types: Coarse classes (4).
        num_sub_varieties: Fine classes (27).
    """

    def __init__(self, feature_dim: int, num_seed_types: int, num_sub_varieties: int):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_seed_types = int(num_seed_types)
        self.num_sub_varieties = int(num_sub_varieties)
        self.seed_head = nn.Linear(self.feature_dim, self.num_seed_types)
        self.sub_head = nn.Linear(self.feature_dim, self.num_sub_varieties)

    @property
    def num_experts(self) -> int:
        return 1

    @property
    def top_k(self) -> int:
        return 1

    def component_flags(self) -> dict[str, Any]:
        return _baseline_component_flags("linear_probe")

    def set_margin_scale(self, value: float) -> None:  # noqa: ARG002 - schedule parity
        """No-op: a linear probe has no margin."""

    def set_router_noise(self, value: float) -> None:  # noqa: ARG002 - schedule parity
        """No-op: a linear probe has no router."""

    def materialize_expert_grads(self) -> int:
        """No-op: a linear probe has no experts."""
        return 0

    def forward(
        self,
        features: torch.Tensor,
        sub_variety_labels: torch.Tensor | None = None,  # noqa: ARG002
        need_attn_weights: bool = False,  # noqa: ARG002
    ) -> HierarchicalOutput:
        # A grid-mode encoder hands over tokens; the probe is deliberately the
        # simplest thing that could work, so it mean-pools them.
        embedding = features.mean(dim=1) if features.ndim == 3 else features
        return _degenerate_output(
            embedding=embedding,
            seed_type_logits=self.seed_head(embedding),
            sub_logits=self.sub_head(embedding),
        )

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self(features)
        return output.seed_type_logits.argmax(dim=-1), output.sub_logits.argmax(dim=-1)

    def extra_repr(self) -> str:
        return f"feature_dim={self.feature_dim}, linear_probe=True"


def _baseline_component_flags(model_name: str) -> dict[str, Any]:
    """Component flags for a model with no MoE, no margin and no fusion."""
    return {
        "use_moe": False,
        "use_arcface": False,
        "use_residual": False,
        "use_cross_attention": False,
        "token_mode": "pooled",
        "fusion_mode": "none",
        "sub_head_variant": "linear",
        "router_mode": "none",
        "gate_conditioning": False,
        "num_experts": 1,
        "top_k": 1,
        "dense_capacity_multiplier": 1,
        "baseline_model": str(model_name),
    }


def _degenerate_output(
    embedding: torch.Tensor,
    seed_type_logits: torch.Tensor,
    sub_logits: torch.Tensor,
) -> HierarchicalOutput:
    """Fill :class:`HierarchicalOutput` for a model with no MoE and no margin.

    The one-expert router that always selects its single expert with weight 1 is
    what lets the MoE regularisers evaluate to exactly zero without any consumer
    needing a special case.
    """
    ones = embedding.new_ones(embedding.shape[0], 1)
    indices = torch.zeros(embedding.shape[0], 1, dtype=torch.long, device=embedding.device)
    return HierarchicalOutput(
        seed_type_logits=seed_type_logits,
        seed_type_probs=F.softmax(seed_type_logits, dim=-1),
        embedding=embedding,
        moe_features=embedding,
        projected_seed=torch.zeros_like(embedding),
        refined_features=embedding,
        attended_features=embedding,
        sub_embeddings=embedding,
        sub_logits=sub_logits,
        # No angular margin: the ArcFace term of the combined objective
        # therefore reduces to plain categorical cross-entropy.
        sub_margin_logits=sub_logits,
        gate_logits=embedding.new_zeros(embedding.shape[0], 1),
        gate_probs=ones,
        top_k_indices=indices,
        top_k_weights=ones,
        dispatch_weights=ones,
        seed_hidden=None,
        tokens_per_sample=1,
        attn_weights=None,
    )


def _as_plain_mapping(value: Any) -> dict[str, Any]:
    """A plain ``dict`` from an OmegaConf node, a mapping, or ``None``.

    ``timm.create_model(**node)`` on a ``DictConfig`` passes ``ValueNode``
    wrappers rather than ints, and timm compares ``img_size`` against an int.
    """
    if value is None:
        return {}
    try:  # OmegaConf is a hard dependency of the trainers, not of this module.
        from omegaconf import DictConfig, OmegaConf

        if isinstance(value, DictConfig):
            return dict(OmegaConf.to_container(value, resolve=True))  # type: ignore[arg-type]
    except ImportError:  # pragma: no cover - notebook use without Hydra
        pass
    return dict(value)


def build_baseline(cfg: Any) -> FlatSupervisedBaseline:
    """Instantiate :class:`FlatSupervisedBaseline` from a ``model.head`` node."""

    def get(key: str, default: Any) -> Any:
        value = getattr(cfg, key, default)
        return default if value is None else value

    return FlatSupervisedBaseline(
        model_name=str(cfg.baseline_model),
        num_seed_types=int(cfg.num_seed_types),
        num_sub_varieties=int(cfg.num_sub_varieties),
        pretrained=bool(get("pretrained", True)),
        embed_dim=int(get("embed_dim", PAPER_EMBED_DIM)),
        dropout_rate=float(get("dropout_rate", 0.1)),
        backbone_kwargs=_as_plain_mapping(get("backbone_kwargs", None)),
    )


def build_linear_probe(cfg: Any) -> LinearProbeHead:
    """Instantiate :class:`LinearProbeHead` from a ``model.head`` node."""
    return LinearProbeHead(
        feature_dim=int(getattr(cfg, "embed_dim", PAPER_EMBED_DIM)),
        num_seed_types=int(cfg.num_seed_types),
        num_sub_varieties=int(cfg.num_sub_varieties),
    )
