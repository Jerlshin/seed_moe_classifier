"""Supervised comparison baselines for the revision's results table.

Three baselines are needed to isolate what the proposed framework contributes:

``resnet50``
    ImageNet-pretrained ResNet-50, trained end to end. The conventional CNN
    reference point.

``swin_tiny``
    ImageNet-pretrained Swin Transformer V1 Tiny, trained end to end. Shares the
    shifted-window inductive bias with the proposed SwinV2 encoder but is
    trained *supervised from ImageNet weights* rather than self-supervised on
    seed imagery, so the gap against the full model isolates the contribution of
    DINOv2 pretraining plus the hierarchical head.

**hierarchical CCE**
    Two-stage coarse-to-fine classification with plain cross-entropy at both
    levels and no MoE, no cross-attention and no ArcFace. This one needs *no
    code*: it is the proposed model with four toggles flipped, which is exactly
    what ``conf/experiment/baseline_hierarchical_cce.yaml`` does. Expressing it
    through the same class as the full model is what makes the comparison fair --
    identical data pipeline, identical optimiser, identical evaluation.

Both classes here emit a :class:`~src.models.builder.HierarchicalOutput`, so the
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

from typing import Any

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.builder import PAPER_EMBED_DIM, HierarchicalOutput

BASELINE_MODELS = {
    "resnet50": "resnet50",
    "swin_tiny": "swin_tiny_patch4_window7_224",
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
    """

    def __init__(
        self,
        model_name: str,
        num_seed_types: int,
        num_sub_varieties: int,
        pretrained: bool = True,
        embed_dim: int = PAPER_EMBED_DIM,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.model_name = BASELINE_MODELS.get(str(model_name), str(model_name))
        self.num_seed_types = int(num_seed_types)
        self.num_sub_varieties = int(num_sub_varieties)
        self.embed_dim = int(embed_dim)

        self.backbone = timm.create_model(self.model_name, pretrained=pretrained, num_classes=0)
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

    def component_flags(self) -> dict[str, bool]:
        return {
            "use_moe": False,
            "use_arcface": False,
            "use_residual": False,
            "use_cross_attention": False,
        }

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

        seed_type_logits = self.seed_head(hidden)
        sub_logits = self.sub_head(hidden)

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
            gate_probs=ones,
            top_k_indices=indices,
            top_k_weights=ones,
            dispatch_weights=ones,
            attn_weights=None,
        )

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self(images)
        return output.seed_type_logits.argmax(dim=-1), output.sub_logits.argmax(dim=-1)

    def extra_repr(self) -> str:
        return (
            f"model_name={self.model_name}, backbone_dim={self.backbone_dim}, "
            f"embed_dim={self.embed_dim}"
        )


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
    )
