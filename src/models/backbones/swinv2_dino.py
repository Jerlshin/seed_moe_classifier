"""DINO student/teacher wrapper around a SwinV2 backbone (paper Section 4).

Paper: "DINOv2 adopts a teacher-student training paradigm in which the student
network is optimized through gradient descent, while the teacher network is
updated as an exponential moving average (EMA) of the student parameters."

Projection head, per Section 4: "The DINO projection head consists of a
multi-layer perceptron (MLP) with batch normalization and GELU activation
functions. The final embeddings are normalized to facilitate stable
self-distillation." So the head is
``Linear -> BN -> GELU -> ... -> Linear(bottleneck) -> L2 normalize ->
weight-normalised Linear(out_dim)``, with ``out_dim = 65,536`` from Table 1.

The final layer's weight-norm gain is frozen (``norm_last_layer``) and its
gradients are cancelled for the first epoch (``freeze_last_layer_epochs``, Section
6.1: "the last layer was frozen for the first epoch to stabilize the initial
training dynamics"). Both exist because that 65,536-wide layer is where a
collapsing run collapses first.
"""

from __future__ import annotations

import copy
from typing import Any

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightly.models.utils import deactivate_requires_grad
from timm.layers import trunc_normal_

from src.models.builder import validate_swinv2_name


def _weight_norm(module: nn.Module) -> nn.Module:
    """Apply weight norm, preferring the modern parametrization API."""
    try:
        return nn.utils.parametrizations.weight_norm(module)
    except (AttributeError, RuntimeError):
        return nn.utils.weight_norm(module)


class DINOHead(nn.Module):
    """MLP projection head with batch norm, GELU, and an L2-normalised bottleneck.

    Args:
        in_dim: Backbone output width.
        out_dim: Projection width (paper Table 1: 65,536).
        use_batch_norm: Insert BatchNorm after each hidden linear (paper: yes).
        norm_last_layer: Freeze the weight-norm gain of the final layer.
        num_layers: Total linear layers in the MLP, including the bottleneck.
        hidden_dim: Hidden width.
        bottleneck_dim: Width of the L2-normalised bottleneck.
        freeze_last_layer_epochs: Epochs during which the final layer's
            gradients are cancelled.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_batch_norm: bool = True,
        norm_last_layer: bool = True,
        num_layers: int = 3,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        freeze_last_layer_epochs: int = 1,
    ):
        super().__init__()
        num_layers = max(int(num_layers), 1)
        if num_layers == 1:
            self.mlp: nn.Module = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim)]
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)
        self.last_layer = _weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self._set_weight_norm_gain(1.0, requires_grad=not norm_last_layer)
        self.freeze_last_layer_epochs = int(freeze_last_layer_epochs)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _set_weight_norm_gain(self, value: float, requires_grad: bool) -> None:
        """Set the weight-norm magnitude, across both PyTorch weight-norm APIs."""
        gain = getattr(self.last_layer, "weight_g", None)  # legacy nn.utils.weight_norm
        if gain is None:
            parametrization = getattr(self.last_layer, "parametrizations", None)
            if parametrization is not None:
                gain = parametrization.weight.original0
        if gain is None:
            return
        with torch.no_grad():
            gain.fill_(value)
        gain.requires_grad = requires_grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)  # L2-normalised embedding, per Section 4
        return self.last_layer(x)

    def cancel_last_layer_gradients(self, current_epoch: int) -> None:
        """Zero the final layer's gradients during the initial frozen epochs."""
        if current_epoch >= self.freeze_last_layer_epochs:
            return
        for parameter in self.last_layer.parameters():
            parameter.grad = None


class DINO(nn.Module):
    """Student/teacher pair sharing an architecture, joined by an EMA update.

    The teacher is a deep copy of the student at initialisation with gradients
    disabled; the trainer advances it with
    ``lightly.models.utils.update_momentum`` at momentum 0.996 (Table 1).

    Args:
        backbone_name: timm model identifier.
        input_dim: Backbone output width, feeding the projection head.
        hidden_dim / bottleneck_dim / out_dim: Projection head widths.
        pretrained: Initialise the backbone from timm's pretrained weights.
        dynamic_img_size: Allow non-native input resolutions.
        projection_layers: Linear layers in the head.
        projection_use_batch_norm: Batch norm inside the head (paper: True).
        projection_norm_last_layer: Freeze the final weight-norm gain.
        freeze_last_layer_epochs: Epochs of cancelled final-layer gradients.
    """

    def __init__(
        self,
        backbone_name: str,
        input_dim: int,
        hidden_dim: int,
        bottleneck_dim: int,
        out_dim: int,
        pretrained: bool = False,
        dynamic_img_size: bool = True,
        projection_layers: int = 3,
        projection_use_batch_norm: bool = True,
        projection_norm_last_layer: bool = True,
        freeze_last_layer_epochs: int = 1,
    ):
        super().__init__()
        # DINOv2 pretraining is standardised on SwinV2. Validating here means a
        # stale or mistyped backbone name fails in the first second rather than
        # after an epoch of self-distillation against the wrong encoder.
        self.backbone_name = validate_swinv2_name(backbone_name)
        self.student_backbone = timm.create_model(
            self.backbone_name,
            pretrained=pretrained,
            num_classes=0,
            dynamic_img_size=dynamic_img_size,
        )
        backbone_dim = getattr(self.student_backbone, "num_features", None)
        if backbone_dim is not None and int(backbone_dim) != int(input_dim):
            raise ValueError(
                f"Backbone {backbone_name!r} emits {backbone_dim} channels but the DINO head "
                f"expects in_dim={input_dim}. Update model.backbone.feature_dim."
            )

        head_kwargs = {
            "in_dim": input_dim,
            "hidden_dim": hidden_dim,
            "bottleneck_dim": bottleneck_dim,
            "out_dim": out_dim,
            "num_layers": projection_layers,
            "use_batch_norm": projection_use_batch_norm,
            "norm_last_layer": projection_norm_last_layer,
            "freeze_last_layer_epochs": freeze_last_layer_epochs,
        }
        self.student_head = DINOHead(**head_kwargs)

        # Teacher starts identical to the student, then only ever moves by EMA.
        self.teacher_backbone = copy.deepcopy(self.student_backbone)
        self.teacher_head = DINOHead(**head_kwargs)
        self.teacher_head.load_state_dict(self.student_head.state_dict())

        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

    def student_parameters(self) -> list[nn.Parameter]:
        """Parameters the optimizer should own (the teacher is EMA-only)."""
        return list(self.student_backbone.parameters()) + list(self.student_head.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled student features, so a trained DINO can act as a plain encoder."""
        return self.forward_features(x)

    def forward_features(self, x: torch.Tensor, teacher: bool = False) -> torch.Tensor:
        backbone = self.teacher_backbone if teacher else self.student_backbone
        features = backbone(x)
        if features.ndim == 4:
            return features.mean(dim=(1, 2))
        if features.ndim == 3:
            return features[:, 0]
        return features

    def forward_student(self, x: torch.Tensor) -> torch.Tensor:
        return self.student_head(self.forward_features(x, teacher=False))

    def forward_teacher(self, x: torch.Tensor) -> torch.Tensor:
        return self.teacher_head(self.forward_features(x, teacher=True))


def build_dino(backbone_cfg: Any, head_cfg: Any, freeze_last_layer_epochs: int = 1) -> DINO:
    """Instantiate :class:`DINO` from the ``model.backbone`` and ``model.head`` nodes."""
    return DINO(
        backbone_name=str(backbone_cfg.name),
        input_dim=int(head_cfg.in_dim),
        hidden_dim=int(head_cfg.hidden_dim),
        bottleneck_dim=int(head_cfg.bottleneck_dim),
        out_dim=int(head_cfg.out_dim),
        pretrained=bool(getattr(backbone_cfg, "pretrained", False)),
        dynamic_img_size=bool(getattr(backbone_cfg, "dynamic_img_size", True)),
        projection_layers=int(getattr(head_cfg, "num_layers", 3)),
        projection_use_batch_norm=bool(getattr(head_cfg, "use_batch_norm", True)),
        projection_norm_last_layer=bool(getattr(head_cfg, "norm_last_layer", True)),
        freeze_last_layer_epochs=int(freeze_last_layer_epochs),
    )
