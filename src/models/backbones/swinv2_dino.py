"""Self-distillation student/teacher wrapper around a SwinV2 backbone (Section 4).

Paper: "adopts a teacher-student training paradigm in which the student network
is optimized through gradient descent, while the teacher network is updated as an
exponential moving average (EMA) of the student parameters."

Projection head, per Section 4: "The projection head consists of a multi-layer
perceptron (MLP) with batch normalization and GELU activation functions. The
final embeddings are normalized to facilitate stable self-distillation." So the
head is ``Linear -> norm -> GELU -> ... -> Linear(bottleneck) -> L2 normalize ->
weight-normalised Linear(out_dim)``.

Two revisions to that description
---------------------------------

**``out_dim``.** Table 1's 65,536 prototypes are DINO's value, set for
ImageNet-1k's 1.28 M images -- 0.051 prototypes per image. Against this dataset's
9,357 images that is **7.00 prototypes per image, a 137x higher density**, and
the prototype layer alone is 16.8 M parameters (71 % of the head, ~19 % of the
student). Total training exposure here is ``300 x 9,357 = 2.81 M`` image
presentations, about 2 % of DINO's 100-epoch ImageNet budget. The revision
default is 8,192; see ``conf/model/head/dino.yaml``.

**The head's normalisation.** ``use_batch_norm=True`` couples the teacher to the
student in a way the EMA does not cover: ``update_momentum`` EMAs *parameters*,
not *buffers*, so the two networks' BatchNorm running statistics diverge -- and
since the teacher sees 2 views per step while the student sees 6, their batch
statistics differ even in principle. DINO's official head defaults to
``use_bn=False`` for transformer backbones for exactly this reason. The revision
default is ``"layer"``; ``"batch"`` reproduces the submitted configuration and
``"none"`` matches the DINO reference.

The final layer's weight-norm gain is frozen (``norm_last_layer``) and its
gradients are cancelled for the first epoch (``freeze_last_layer_epochs``, Section
6.1: "the last layer was frozen for the first epoch to stabilize the initial
training dynamics"). Both exist because that wide prototype layer is where a
collapsing run collapses first.

:meth:`DINOHead.forward` can return the pre-prototype bottleneck alongside the
logits, which is what the KoLeo regulariser consumes -- applying it to the
65k-wide prototype output would measure the wrong space.

One forward per step, not six
-----------------------------

:meth:`DINO.forward_student_views` takes **all** the student's views as one
stacked tensor and issues a single backbone call. The obvious loop --
``[model.forward_student(view) for view in views]`` -- is six separate SwinV2-Base
forwards at batch 16, and a Swin block at batch 16 is nowhere near large enough
to saturate a GPU: each of its ~200 kernels spends more time being launched than
running, and that launch overhead is paid six times over. Stacking makes it one
forward at batch 96, which is the same arithmetic on tensors large enough to
matter, with a sixth of the launches.

It also costs almost nothing in memory. The loop already holds all six autograd
graphs simultaneously -- backward runs after the last view -- so peak activation
memory is unchanged; the only new allocation is the stacked *input*, and
``chunk_size`` splits even that back up if a smaller card needs it.

The teacher gets the same treatment over its two global views, under
``no_grad``.
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


HEAD_NORM_MODES = ("layer", "batch", "none")


def _resolve_head_norm(use_batch_norm: bool | str | None) -> str:
    """Accept the legacy boolean flag and the named modes interchangeably."""
    if isinstance(use_batch_norm, str):
        mode = use_batch_norm.lower()
        if mode not in HEAD_NORM_MODES:
            raise ValueError(f"head norm must be one of {HEAD_NORM_MODES}, got {use_batch_norm!r}")
        return mode
    return "batch" if bool(use_batch_norm) else "none"


class DINOHead(nn.Module):
    """MLP projection head with GELU and an L2-normalised bottleneck.

    Args:
        in_dim: Backbone output width.
        out_dim: Projection width.
        use_batch_norm: ``"layer"`` (default), ``"batch"`` or ``"none"``. A bare
            boolean is accepted for backward compatibility and maps to
            ``"batch"``/``"none"``; see the module docstring for why ``"batch"``
            couples the teacher's statistics to the student's.
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
        use_batch_norm: bool | str = "layer",
        norm_last_layer: bool = True,
        num_layers: int = 3,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        freeze_last_layer_epochs: int = 1,
    ):
        super().__init__()
        self.norm_mode = _resolve_head_norm(use_batch_norm)

        def norm_layer() -> list[nn.Module]:
            if self.norm_mode == "batch":
                return [nn.BatchNorm1d(hidden_dim)]
            if self.norm_mode == "layer":
                return [nn.LayerNorm(hidden_dim)]
            return []

        num_layers = max(int(num_layers), 1)
        if num_layers == 1:
            self.mlp: nn.Module = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), *norm_layer(), nn.GELU()]
            for _ in range(num_layers - 2):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), *norm_layer(), nn.GELU()])
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

    def forward(
        self,
        x: torch.Tensor,
        return_bottleneck: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return prototype logits, optionally with the bottleneck embedding.

        The KoLeo regulariser needs the bottleneck, not the prototype logits:
        uniformity is a property of the representation, and measuring it in the
        65k-wide prototype space would measure the head's output distribution
        instead.
        """
        bottleneck = F.normalize(self.mlp(x), dim=-1, p=2)  # per Section 4
        logits = self.last_layer(bottleneck)
        return (logits, bottleneck) if return_bottleneck else logits

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
        projection_use_batch_norm: Head normalisation, "layer" / "batch" /
            "none"; see :class:`DINOHead`.
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
        projection_use_batch_norm: bool | str = "layer",
        projection_norm_last_layer: bool = True,
        freeze_last_layer_epochs: int = 1,
    ):
        super().__init__()
        # Self-supervised pretraining is standardised on SwinV2. Validating here means a
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

        # Compiled callables live OFF the module tree, in a plain dict.
        #
        # `torch.compile(module)` returns an `OptimizedModule` wrapper, and
        # assigning that back onto `self.student_backbone` would register it as a
        # submodule -- at which point every key in `student_backbone.state_dict()`
        # gains an `_orig_mod.` prefix. That state dict is the *only* handoff to
        # stage 2, and `checkpoint_strict: false` is the default there, so the
        # result would be a stage-2 encoder that matched zero keys, loaded a
        # random trunk, logged one line about missing keys and trained to
        # completion looking entirely healthy.
        self._compiled: dict[str, Any] = {}
        self.grad_checkpointing = False

    # ------------------------------------------------------------- runtime

    def configure_runtime(
        self,
        *,
        compile_enabled: bool = False,
        compile_mode: str = "default",
        grad_checkpointing: bool = False,
        channels_last: bool = False,
        fused_attention: bool = True,
        logger=None,
    ) -> dict[str, Any]:
        """Apply the execution-level options and report what took effect.

        None of these change the function the model computes; they change how it
        is executed. Each is reported back (and logged into the run's event
        stream) because two of them can silently *fail* to apply:
        ``torch.compile`` falls back to eager on a backend error, and
        ``fused_attention`` finds nothing to switch on a stock SwinV2 trunk.

        Args:
            compile_enabled / compile_mode: See
                :func:`~src.utils.training.device.maybe_compile`. ``"default"``
                is the safe choice here. ``"reduce-overhead"`` adds CUDA graphs,
                which capture the parameter *pointers* -- fine for the student,
                but the teacher's parameters are rewritten in place by the EMA
                every step and there is no reason to take that risk for a trunk
                that runs under ``no_grad`` anyway.
            grad_checkpointing: Recompute student activations during backward
                instead of storing them. Buys roughly half the activation memory
                for roughly a third more compute, so it is a way to *afford a
                larger batch*, not a speedup on its own. Applied to the student
                only -- the teacher stores no activations to begin with.
            channels_last: NHWC memory format. Only the patch-embed convolution
                benefits; SwinV2's blocks are already channel-last internally.
                Off by default.
            fused_attention: Ask timm for SDPA where it offers it.
        """
        from src.utils.training.device import enable_fused_attention, maybe_compile

        applied: dict[str, Any] = {}

        if grad_checkpointing:
            setter = getattr(self.student_backbone, "set_grad_checkpointing", None)
            if callable(setter):
                setter(True)
                self.grad_checkpointing = True
                applied["grad_checkpointing"] = True
            elif logger is not None:
                logger.warning(
                    "%s does not expose set_grad_checkpointing; continuing without it.",
                    self.backbone_name,
                )

        if fused_attention:
            switched = enable_fused_attention(self.student_backbone)
            switched += enable_fused_attention(self.teacher_backbone)
            applied["fused_attention_modules"] = switched
            if logger is not None and switched == 0:
                logger.info(
                    "No fused-attention modules on %s. SwinV2 window attention is cosine "
                    "attention with a clamped logit scale and a continuous relative-position "
                    "bias, which SDPA cannot express; the speedup on this trunk comes from "
                    "autocast and inductor fusion instead.",
                    self.backbone_name,
                )

        if channels_last:
            self.to(memory_format=torch.channels_last)
            applied["channels_last"] = True

        if compile_enabled:
            if self.grad_checkpointing and logger is not None:
                logger.warning(
                    "torch.compile together with gradient checkpointing frequently causes "
                    "graph breaks that undo both. Prefer one or the other."
                )
            self._compiled["student_backbone"] = maybe_compile(
                self.student_backbone, True, compile_mode, logger
            )
            self._compiled["teacher_backbone"] = maybe_compile(
                self.teacher_backbone, True, compile_mode, logger
            )
            self._compiled["student_head"] = maybe_compile(
                self.student_head, True, compile_mode, logger
            )
            self._compiled["teacher_head"] = maybe_compile(
                self.teacher_head, True, compile_mode, logger
            )
            applied["compiled"] = sorted(self._compiled)
            applied["compile_mode"] = compile_mode

        return applied

    def _module(self, name: str) -> nn.Module:
        """The compiled callable for ``name`` when there is one, else the module."""
        return self._compiled.get(name, getattr(self, name))

    def student_parameters(self) -> list[nn.Parameter]:
        """Parameters the optimizer should own (the teacher is EMA-only)."""
        return list(self.student_backbone.parameters()) + list(self.student_head.parameters())

    def ema_pairs(self) -> list[tuple[nn.Module, nn.Module]]:
        """``(student, teacher)`` module pairs the EMA advances, in order."""
        return [
            (self.student_backbone, self.teacher_backbone),
            (self.student_head, self.teacher_head),
        ]

    # -------------------------------------------------------------- forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled student features, so a trained DINO can act as a plain encoder."""
        return self.forward_features(x)

    @staticmethod
    def _pool(features: torch.Tensor) -> torch.Tensor:
        # SwinV2 has no class token, so pooling must average the spatial grid --
        # taking `features[:, 0]` would silently select one corner patch, and
        # would disagree with BackboneFeatureExtractor._pool, which is what
        # stage 2 uses to read the very same weights.
        if features.ndim == 4:
            return features.mean(dim=(1, 2))
        if features.ndim == 3:
            return features.mean(dim=1)
        return features

    def _backbone_features(
        self,
        name: str,
        x: torch.Tensor,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Pooled features for ``x``, optionally in chunks of ``chunk_size`` images.

        Chunking exists purely as an OOM valve: it splits the *input* batch, so
        the peak activation of any one backbone call falls proportionally, while
        the autograd graphs of all chunks stay alive until backward. It therefore
        helps when a single forward's transient allocation is what overflows, and
        does nothing for the accumulated graph. Leave it unset unless a run OOMs.
        """
        backbone = self._module(name)
        if not chunk_size or chunk_size >= x.shape[0]:
            return self._pool(backbone(x))
        return torch.cat([self._pool(backbone(part)) for part in x.split(int(chunk_size))], dim=0)

    def forward_features(self, x: torch.Tensor, teacher: bool = False) -> torch.Tensor:
        return self._backbone_features("teacher_backbone" if teacher else "student_backbone", x)

    def forward_student(
        self,
        x: torch.Tensor,
        return_bottleneck: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.forward_student_views(x, return_bottleneck=return_bottleneck)

    def forward_teacher(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_teacher_views(x)

    def forward_student_views(
        self,
        views: torch.Tensor,
        return_bottleneck: bool = False,
        chunk_size: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Prototype logits for a stacked ``[V * B, C, H, W]`` view tensor.

        The blocks must be **view-major** -- all of view 0, then all of view 1 --
        because :class:`~src.losses.dino.CustomDINOLoss` chunks the output back
        into per-view groups. ``MultiCropCollate`` produces exactly that layout.
        """
        features = self._backbone_features("student_backbone", views, chunk_size)
        return self._module("student_head")(features, return_bottleneck=return_bottleneck)

    @torch.no_grad()
    def forward_teacher_views(
        self,
        views: torch.Tensor,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Prototype logits for the teacher's stacked global views, under ``no_grad``."""
        features = self._backbone_features("teacher_backbone", views, chunk_size)
        return self._module("teacher_head")(features)


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
        projection_use_batch_norm=getattr(head_cfg, "use_batch_norm", "layer"),
        projection_norm_last_layer=bool(getattr(head_cfg, "norm_last_layer", True)),
        freeze_last_layer_epochs=int(freeze_last_layer_epochs),
    )
