"""Self-distillation student/teacher wrapper around a SwinV2 backbone (Section 4).

Paper: "adopts a teacher-student training paradigm in which the student network
is optimized through gradient descent, while the teacher network is updated as an
exponential moving average (EMA) of the student parameters."

Projection head, per Section 4: "The projection head consists of a multi-layer
perceptron (MLP) with batch normalization and GELU activation functions. The
final embeddings are normalized to facilitate stable self-distillation." So the
head is ``Linear -> norm -> GELU -> ... -> Linear(bottleneck) -> L2 normalize ->
weight-normalised Linear(out_dim)``.

Three revisions to that description
-----------------------------------

**Initialisation.** The trunk starts from ImageNet-1k
(``model.backbone.pretrained=true``) rather than from noise, and it *trains* --
``build_dino`` refuses ``model.backbone.freeze=true`` outright, because
self-distillation against a frozen trunk fits the projection head to fixed
features and adapts nothing, with no error to say so. Stochastic depth
(``drop_path_rate``) applies to the student only; :func:`disable_drop_path`
silences the teacher copy, whose outputs are the targets.

**``out_dim``.** Table 1's 65,536 prototypes are DINO's value, set for
ImageNet-1k's 1.28 M images -- 0.051 prototypes per image. Against this dataset's
9,357 images that is **7.00 prototypes per image, a 137x higher density**, and
the prototype layer alone is 16.8 M parameters. The current default is 2,048,
which is a decision about the *batch* as much as the head: Sinkhorn-Knopp gives
each prototype ``B_teacher / out_dim`` of the assignment mass, so at 64 teacher
views 2,048 prototypes carry four times the evidence per column that 8,192 did.
See ``conf/model/head/dino.yaml``.

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
prototype output would measure the wrong space.

One forward per step, not six
-----------------------------

:meth:`DINO.forward_student_views` takes **all** the student's views as one
stacked tensor and issues a single backbone call. The obvious loop --
``[model.forward_student(view) for view in views]`` -- is six separate SwinV2
forwards at the micro-batch size, which is nowhere near large enough to saturate
a GPU: each of a Swin block's ~200 kernels spends more time being launched than
running, and that launch overhead is paid six times over. Stacking makes it one
forward at ``6 x batch`` (192 views at the configured batch of 32), which is the
same arithmetic on tensors large enough to matter, with a sixth of the launches.

It also costs almost nothing in memory. The loop already holds all six autograd
graphs simultaneously -- backward runs after the last view -- so peak activation
memory is unchanged; the only new allocation is the stacked *input*, and
``chunk_size`` splits even that back up if a smaller card needs it.

The teacher gets the same treatment over its two global views, under
``no_grad``.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Any

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightly.models.utils import deactivate_requires_grad
from timm.layers import trunc_normal_

from src.models.builder import validate_swinv2_name

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.utils.training.distributed import DistributedContext


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


def disable_drop_path(module: nn.Module) -> int:
    """Set every ``DropPath`` probability in ``module`` to zero. Returns the count.

    Stochastic depth belongs to the **student** alone. The teacher is a deep copy
    of the student and is advanced only by the EMA, so it inherits the student's
    ``drop_path_rate`` -- and ``model.train()`` puts it in training mode, where
    timm's ``DropPath`` is active. The teacher's outputs are the *targets* of
    Eq. 1, so leaving it on would randomly delete residual branches from the
    label rather than from the learner: a different, noisier objective, with a
    loss curve that looks entirely normal. DINO's reference implementation
    constructs its teacher without the flag for the same reason; here the copy
    already exists, so the probabilities are zeroed in place instead, which keeps
    ``teacher.state_dict()`` byte-identical to the student's at initialisation.

    ``drop_prob`` is a plain float attribute rather than a buffer, so it is not
    part of any state dict and a resumed run re-applies this at construction.
    """
    silenced = 0
    for child in module.modules():
        if type(child).__name__ == "DropPath" and getattr(child, "drop_prob", 0.0):
            child.drop_prob = 0.0
            silenced += 1
    return silenced


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
        hidden_dim: int = 1024,
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
        prototype space would measure the head's output distribution instead.
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


class _StudentPass(nn.Module):
    """The student's whole forward as one module, so DDP has one thing to wrap.

    ``DistributedDataParallel`` synchronises the parameters of the module it
    wraps, on the backward of the forward *it* was called through. The student
    pass here is two modules and a pooling step, and the teacher must stay
    outside -- it takes no gradient, so wrapping :class:`DINO` itself would hand
    DDP a set of parameters half of which never produce one.

    This adapter registers the *same* module objects the owning :class:`DINO`
    holds. Registering a module twice does not copy it: ``DINO.state_dict()`` is
    byte-identical with or without this wrapper existing, which is the property
    that matters, because ``student_backbone.state_dict()`` is the only handoff
    to stage 2 and a renamed key there is a silent failure rather than a loud one.

    ``resolve`` is the owner's :meth:`DINO._module`, called per forward rather
    than captured once, so a ``torch.compile`` applied before *or* after this
    wrapper is built is still the callable that runs. It is stored as a plain
    attribute (a bound method is not an ``nn.Module``, so it registers nothing).
    """

    def __init__(self, backbone: nn.Module, head: nn.Module, resolve, pool, owner=None, aux_head=None):
        super().__init__()
        self.backbone = backbone
        self.head = head
        # Registered so DDP owns the auxiliary head's parameters too. `None`
        # under the default configuration, in which case nothing is registered
        # and the wrapper is byte-identical to what it was before.
        if aux_head is not None:
            self.aux_head = aux_head
        self._resolve = resolve
        self._pool = pool
        #: Plain attribute: the owning DINO is already this wrapper's parent, and
        #: registering it would recurse and rename every state-dict key.
        self._owner = owner

    def forward(
        self,
        views: torch.Tensor,
        return_bottleneck: bool = False,
        chunk_size: int | None = None,
        return_features: bool = False,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        backbone = self._resolve("student_backbone")
        head = self._resolve("student_head")
        capture = self._owner.capture_aux("student") if return_aux else nullcontext()
        with capture:
            if not chunk_size or chunk_size >= views.shape[0]:
                features = self._pool(backbone(views))
            else:
                features = torch.cat(
                    [self._pool(backbone(part)) for part in views.split(int(chunk_size))], dim=0
                )
        outputs: list[torch.Tensor] = []
        logits = head(features, return_bottleneck=return_bottleneck)
        if return_bottleneck:
            outputs.extend(logits)
        else:
            outputs.append(logits)
        if return_features:
            outputs.append(features)
        if return_aux:
            outputs.append(self._owner.apply_aux_head("student"))
        return outputs[0] if len(outputs) == 1 else tuple(outputs)


class DINO(nn.Module):
    """Student/teacher pair sharing an architecture, joined by an EMA update.

    The teacher is a deep copy of the student at initialisation with gradients
    disabled; the trainer advances it with
    ``lightly.models.utils.update_momentum`` at momentum 0.996 (Table 1).

    Args:
        backbone_name: timm model identifier.
        input_dim: Backbone output width, feeding the projection head.
        hidden_dim / bottleneck_dim / out_dim: Projection head widths.
        pretrained: Initialise the backbone from timm's ImageNet weights. The
            primary stage-1 configuration sets this ``True``: DINO from a random
            trunk was budgeted at 300 epochs, and starting from ImageNet-1k
            changes what that budget is for.
        dynamic_img_size: Allow non-native input resolutions.
        drop_path_rate: Stochastic depth for the **student** trunk (timm's own
            mechanism; there is no second one). The teacher copy is silenced --
            see :func:`disable_drop_path`.
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
        drop_path_rate: float = 0.0,
        projection_layers: int = 3,
        projection_use_batch_norm: bool | str = "layer",
        projection_norm_last_layer: bool = True,
        freeze_last_layer_epochs: int = 1,
        aux_stage: int | None = None,
        aux_out_dim: int | None = None,
        aux_weight: float = 1.0,
    ):
        super().__init__()
        # Self-supervised pretraining is standardised on SwinV2. Validating here means a
        # stale or mistyped backbone name fails in the first second rather than
        # after an epoch of self-distillation against the wrong encoder.
        self.backbone_name = validate_swinv2_name(backbone_name)
        self.pretrained_init = bool(pretrained)
        self.drop_path_rate = float(drop_path_rate)
        self.student_backbone = timm.create_model(
            self.backbone_name,
            pretrained=pretrained,
            num_classes=0,
            dynamic_img_size=dynamic_img_size,
            drop_path_rate=self.drop_path_rate,
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

        # ------------------------------------------------- auxiliary stage head
        #
        # The C4 arm of STAGE1_CHANGES.md. `aux_stage=2` attaches a SECOND DINO
        # head to the mean-pooled output of `layers.2`, with its own prototypes,
        # and the trainer sums the two losses at `aux_weight`.
        #
        # The reason is measurable rather than aesthetic. Reading `layers.2`
        # downstream (`model.backbone.feature_stage=stage3`) fixes *where* stage 2
        # reads; it does not fix the fact that the stage-1 gradient reaches
        # `layers.2` only through two blocks being optimised for a 2,048-way
        # prototype task. Linear CKA against the ImageNet initialisation was 0.103
        # at `layers.3` and 0.390 at `layers.2` on the shipped run: the objective
        # consumed the last stage. Supervising `layers.2` directly means the layer
        # that ships is the layer that was trained.
        #
        # Off by default (`aux_stage: null`). It adds parameters, prototypes to
        # keep alive, and a weight to tune, so it is an isolated arm rather than a
        # recipe change, and it confounds the `feature_stage` decision if run
        # simultaneously with it.
        self.aux_stage = None if aux_stage is None else int(aux_stage)
        self.aux_weight = float(aux_weight)
        self.student_aux_head: DINOHead | None = None
        self.teacher_aux_head: DINOHead | None = None
        self._aux_capture: dict[str, list[torch.Tensor]] = {"student": [], "teacher": []}
        if self.aux_stage is not None:
            aux_dim = self._stage_channels(self.aux_stage)
            aux_kwargs = {
                **head_kwargs,
                "in_dim": aux_dim,
                "out_dim": int(aux_out_dim or out_dim),
            }
            self.student_aux_head = DINOHead(**aux_kwargs)
            self.teacher_aux_head = DINOHead(**aux_kwargs)
            self.teacher_aux_head.load_state_dict(self.student_aux_head.state_dict())
            deactivate_requires_grad(self.teacher_aux_head)

        # Stochastic depth is a property of the learner, not of the target. The
        # deepcopy inherited it; this takes it back out. Recorded because "did
        # the teacher get drop path" is exactly the kind of question a loss curve
        # cannot answer later.
        self.teacher_drop_paths_disabled = disable_drop_path(self.teacher_backbone)

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
        # The DDP wrapper lives off the module tree for the same reason the
        # compiled callables do: assigning it to an attribute would register it
        # as a submodule and prefix every state-dict key with `module.`.
        self._ddp: dict[str, Any] = {}
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
        sdpa_attention: bool = False,
        logger=None,
    ) -> dict[str, Any]:
        """Apply the execution-level options and report what took effect.

        None of these change the function the model computes; they change how it
        is executed. Each is reported back (and logged into the run's event
        stream) because three of them can silently *fail* to apply:
        ``torch.compile`` falls back to eager on a backend error,
        ``fused_attention`` finds nothing to switch on a stock SwinV2 trunk, and
        ``sdpa_attention`` refuses any module that fails its conversion-time
        parity check.

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
            fused_attention: Ask timm for SDPA where it offers it. On SwinV2
                this finds nothing (see ``sdpa_attention``); it stays on for the
                baseline trunks that do expose the flag.
            sdpa_attention: Rebind SwinV2's cosine window attention to an
                algebraically identical ``F.scaled_dot_product_attention`` form
                -- see ``src/models/backbones/sdpa_attention.py``. Applied to
                both trunks **before** compilation so the compiled graphs trace
                the fused path. This is the change that stops each block saving
                two full ``[B·nW, heads, N, N]`` attention matrices for
                backward (several GB of bf16 at 192 student views), which is what
                frees the memory to raise the physical batch.
        """
        from src.models.backbones.sdpa_attention import convert_swinv2_attention_to_sdpa
        from src.utils.training.device import enable_fused_attention, maybe_compile

        applied: dict[str, Any] = {}

        if sdpa_attention:
            converted = convert_swinv2_attention_to_sdpa(self.student_backbone, logger=logger)
            converted += convert_swinv2_attention_to_sdpa(self.teacher_backbone, logger=logger)
            applied["sdpa_attention_modules"] = converted
            if logger is not None:
                if converted:
                    logger.info(
                        "Rebound %s SwinV2 window-attention modules to SDPA "
                        "(parity-checked per module).",
                        converted,
                    )
                else:
                    logger.warning(
                        "sdpa_attention was requested but no module converted; the run "
                        "stays on eager cosine attention. Check the warnings above."
                    )

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
                    "No timm fused-attention modules on %s: SwinV2 window attention is "
                    "cosine attention, for which timm offers no fused path. The fused "
                    "route on this trunk is the sdpa_attention conversion "
                    "(src/models/backbones/sdpa_attention.py), which refactors the same "
                    "function into F.scaled_dot_product_attention.",
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
            if self.student_aux_head is not None:
                self._compiled["student_aux_head"] = maybe_compile(
                    self.student_aux_head, True, compile_mode, logger
                )
                self._compiled["teacher_aux_head"] = maybe_compile(
                    self.teacher_aux_head, True, compile_mode, logger
                )
            applied["compiled"] = sorted(self._compiled)
            applied["compile_mode"] = compile_mode

        return applied

    def configure_distributed(
        self,
        context: "DistributedContext",
        *,
        gradient_as_bucket_view: bool = True,
        static_graph: bool = False,
        find_unused_parameters: bool = False,
        bucket_cap_mb: int | None = None,
        logger=None,
    ) -> dict[str, Any]:
        """Wrap the student in DDP and force the teacher to agree across ranks.

        Returns what was applied, for the run's event stream. A no-op on a
        single process, so the trainer needs no branch.

        Three decisions here are load-bearing:

        **Only the student is wrapped.** The teacher takes no gradient -- it is
        advanced by the EMA -- so it has nothing for DDP to reduce, and handing
        DDP a module whose parameters never receive gradients would force
        ``find_unused_parameters=True`` and a full graph traversal every step to
        discover that fact again.

        **The teacher is broadcast instead.** DDP broadcasts the wrapped module's
        state at construction, which makes the students identical; nothing does
        that for the teacher. It starts as a deep copy of each rank's own
        student, so identical seeding already makes them agree -- but "already
        agrees" is not a property to leave implicit for the network that produces
        the training *targets*, and after a resume at a different world size it
        would not hold at all.

        **Buffers are not broadcast per step.** SwinV2's buffers
        (``relative_coords_table``, ``relative_position_index``) are constants,
        and the projection head is LayerNorm by default precisely so no running
        statistic needs carrying. ``broadcast_buffers=True`` would add a
        collective over several megabytes to every forward to re-send constants.
        Switching the head to ``use_batch_norm: "batch"`` reintroduces buffers
        that this does not propagate -- the same coupling ``conf/model/head/\
dino.yaml`` already documents for the EMA, now with a second reason.

        Args:
            gradient_as_bucket_view: Let gradients alias DDP's reduction buckets
                instead of being copied into them. Saves a full gradient's worth
                of memory, which on a 16 GB T4 is the difference between fitting
                and not.
            static_graph: Promise the autograd graph is identical every
                iteration, which lets DDP reorder reductions and re-use its
                bucket plan. True here in practice, but left off by default:
                a wrong promise fails as a hang at the end of an epoch rather
                than as an error.
            find_unused_parameters: Leave ``False``. Every student parameter
                receives a gradient on every step; ``True`` costs a graph
                traversal per iteration to establish that.
            bucket_cap_mb: Reduction bucket size. ``None`` keeps DDP's default
                (25 MB), which is a reasonable fit for this parameter count.
        """
        from torch.nn.parallel import DistributedDataParallel
        from src.utils.training.distributed import broadcast_module_state, buffer_sync_kwarg

        if not context.enabled:
            return {"distributed": False}

        # Must happen before the wrapper is built: DDP broadcasts the wrapped
        # module's state from rank 0, and the teacher is not in that module.
        broadcast_module_state(self.teacher_backbone, context)
        broadcast_module_state(self.teacher_head, context)
        if self.teacher_aux_head is not None:
            broadcast_module_state(self.teacher_aux_head, context)

        student = _StudentPass(
            self.student_backbone,
            self.student_head,
            self._module,
            self._pool,
            owner=self,
            aux_head=self.student_aux_head,
        )
        kwargs: dict[str, Any] = {
            **buffer_sync_kwarg(False),
            "gradient_as_bucket_view": bool(gradient_as_bucket_view),
            "find_unused_parameters": bool(find_unused_parameters),
            "static_graph": bool(static_graph),
        }
        if context.device.type == "cuda":
            kwargs["device_ids"] = [context.local_rank]
            kwargs["output_device"] = context.local_rank
        if bucket_cap_mb:
            kwargs["bucket_cap_mb"] = int(bucket_cap_mb)

        self._ddp["student"] = DistributedDataParallel(student, **kwargs)

        applied = {
            "distributed": True,
            "world_size": context.world_size,
            "backend": context.backend,
            "gradient_as_bucket_view": bool(gradient_as_bucket_view),
            "static_graph": bool(static_graph),
            "find_unused_parameters": bool(find_unused_parameters),
        }
        if logger is not None:
            logger.info(
                "DDP | student wrapped across %s ranks (%s), teacher broadcast from rank 0, "
                "buffers not re-broadcast per step.",
                context.world_size, context.backend,
            )
        return applied

    def no_sync(self):
        """Suppress DDP's gradient all-reduce for one micro-batch.

        Wraps every micro-batch that is **not** an accumulation boundary. Without
        it, DDP reduces the full gradient once per micro-batch instead of once
        per optimizer step -- ``gradient_accumulation_steps`` times the traffic
        for an identical result, which on a two-T4 box without NVLink is paid
        over PCIe and is the single easiest way to make a distributed run slower
        than the single-GPU one it replaced.

        A no-op context when the run is not distributed.
        """
        student = self._ddp.get("student")
        return student.no_sync() if student is not None else nullcontext()

    @property
    def is_distributed(self) -> bool:
        return "student" in self._ddp

    def _module(self, name: str) -> nn.Module:
        """The compiled callable for ``name`` when there is one, else the module."""
        return self._compiled.get(name, getattr(self, name))

    def student_parameters(self) -> list[nn.Parameter]:
        """Parameters the optimizer should own (the teacher is EMA-only)."""
        parameters = list(self.student_backbone.parameters()) + list(self.student_head.parameters())
        if self.student_aux_head is not None:
            parameters += list(self.student_aux_head.parameters())
        return parameters

    # ---------------------------------------------- auxiliary stage capture

    def _stage_channels(self, index: int) -> int:
        """Channel count of ``feature_info[index]`` on the student trunk."""
        info = getattr(self.student_backbone, "feature_info", None)
        if info is None:
            raise AttributeError(
                f"{self.backbone_name!r} exposes no `feature_info`, so aux_stage={index} "
                "cannot resolve its input width."
            )
        channels = (
            info.channels() if hasattr(info, "channels") else [entry["num_chs"] for entry in info]
        )
        if index >= len(channels):
            raise ValueError(
                f"{self.backbone_name!r} has {len(channels)} stages; aux_stage={index} is out of range."
            )
        return int(channels[index])

    @contextmanager
    def capture_aux(self, side: str):
        """Collect ``layers[aux_stage]``'s pooled output during one forward.

        A forward hook rather than a second forward: the auxiliary objective
        supervises the *same* activations the primary one produces, so
        recomputing them would double the trunk cost and -- under stochastic
        depth -- would not even produce the same tensor.

        A no-op when no auxiliary head is configured, which is the default, so
        nothing is registered and nothing is captured on the shipped path.
        """
        if self.aux_stage is None:
            yield
            return
        backbone = getattr(self, f"{side}_backbone")
        layers = getattr(backbone, "layers", None)
        if layers is None or self.aux_stage >= len(layers):
            raise AttributeError(
                f"{self.backbone_name!r} exposes no layers[{self.aux_stage}] to hook."
            )
        bucket = self._aux_capture[side]
        bucket.clear()
        handle = layers[self.aux_stage].register_forward_hook(
            lambda _module, _inputs, output: bucket.append(self._pool(output))
        )
        try:
            yield
        finally:
            handle.remove()

    def apply_aux_head(self, side: str) -> torch.Tensor:
        """Prototype logits from the auxiliary head over the captured stage.

        Concatenates in capture order, which is the chunk order the caller used,
        so the result stays view-major exactly as the primary head's output does.
        """
        head = getattr(self, f"{side}_aux_head")
        if head is None:
            raise RuntimeError("No auxiliary head is configured; set model.head.aux_stage.")
        captured = self._aux_capture[side]
        if not captured:
            raise RuntimeError(
                "The auxiliary stage was not captured. The forward must run inside "
                "`capture_aux(side)`."
            )
        features = captured[0] if len(captured) == 1 else torch.cat(captured, dim=0)
        return self._module(f"{side}_aux_head")(features)

    def parameter_summary(self) -> dict[str, int]:
        """Parameter counts, split the way the cost table reports them.

        ``student_trainable`` is the one to read when checking that the trunk is
        actually being fine-tuned: it equals ``student_total`` in the intended
        stage-1 regime and collapses to the head's count if anything froze the
        backbone. ``teacher_total`` is reported as its own row because the
        teacher doubles the *resident* parameter count while contributing
        nothing to the gradient -- a distinction a single "model size" hides.
        """

        def count(module: nn.Module, trainable_only: bool = False) -> int:
            return sum(
                parameter.numel()
                for parameter in module.parameters()
                if not trainable_only or parameter.requires_grad
            )

        backbone = count(self.student_backbone)
        head = count(self.student_head)
        aux = count(self.student_aux_head) if self.student_aux_head is not None else 0
        teacher_aux = count(self.teacher_aux_head) if self.teacher_aux_head is not None else 0
        return {
            "backbone": backbone,
            "dino_head": head,
            "dino_aux_head": aux,
            "student_total": backbone + head + aux,
            "student_trainable": (
                count(self.student_backbone, True)
                + count(self.student_head, True)
                + (count(self.student_aux_head, True) if self.student_aux_head is not None else 0)
            ),
            "student_backbone_trainable": count(self.student_backbone, True),
            "teacher_total": count(self.teacher_backbone) + count(self.teacher_head) + teacher_aux,
            "prototype_layer": count(self.student_head.last_layer),
        }

    @torch.no_grad()
    def shape_report(self, image_size: int = 256, device: torch.device | None = None) -> dict[str, Any]:
        """Trace one image through both paths and report every intermediate shape.

        Run once at startup. The three shapes worth checking by eye are the token
        grid (``8x8`` for every ``swinv2_*_window16_256``, and what stage 2's grid
        routing consumes), the pooled width feeding the head, and the prototype
        width -- which must agree between student and teacher, because the loss
        contracts them against each other and a mismatch would surface as a shape
        error deep inside an einsum rather than here.
        """
        target = device if device is not None else next(self.student_backbone.parameters()).device
        was_training = self.training
        self.eval()
        try:
            probe = torch.zeros(1, 3, int(image_size), int(image_size), device=target)
            tokens = self.student_backbone.forward_features(probe)
            pooled = self._pool(self.student_backbone(probe))
            student_logits, bottleneck = self.student_head(pooled, return_bottleneck=True)
            teacher_logits = self.teacher_head(
                self._pool(self.teacher_backbone(probe))
            )
        finally:
            self.train(was_training)

        grid = tuple(tokens.shape[1:3]) if tokens.ndim == 4 else (1, int(tokens.shape[1]))
        return {
            "input": tuple(probe.shape),
            "backbone_tokens": tuple(tokens.shape),
            "token_grid": grid,
            "tokens_per_image": int(grid[0] * grid[1]),
            "pooled_features": tuple(pooled.shape),
            "head_input_dim": int(pooled.shape[-1]),
            "head_bottleneck": tuple(bottleneck.shape),
            "student_prototypes": tuple(student_logits.shape),
            "teacher_prototypes": tuple(teacher_logits.shape),
        }

    def ema_pairs(self) -> list[tuple[nn.Module, nn.Module]]:
        """``(student, teacher)`` module pairs the EMA advances, in order."""
        pairs = [
            (self.student_backbone, self.teacher_backbone),
            (self.student_head, self.teacher_head),
        ]
        if self.student_aux_head is not None and self.teacher_aux_head is not None:
            pairs.append((self.student_aux_head, self.teacher_aux_head))
        return pairs

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
        return_features: bool = False,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Prototype logits for a stacked ``[V * B, C, H, W]`` view tensor.

        ``return_bottleneck``, ``return_features`` and ``return_aux`` append, in
        that order, the head's L2-normalised bottleneck, the **pooled trunk
        feature** and the auxiliary stage head's prototype logits. The trunk
        feature is what ``model.loss.koleo_space=backbone`` regularises: KoLeo is
        currently applied to a 256-D bottleneck that stage 2 never sees, and
        DINOv2 applies it to ``x_norm_clstoken`` -- the backbone output -- for
        exactly that reason.

        The blocks must be **view-major** -- all of view 0, then all of view 1 --
        because :class:`~src.losses.dino.CustomDINOLoss` chunks the output back
        into per-view groups. ``MultiCropCollate`` produces exactly that layout.

        Under DDP this goes through the wrapper rather than the modules directly:
        DDP arms its reducer inside ``forward``, so calling the inner modules
        would leave the gradients unsynchronised and every rank would train its
        own model while the loss curves looked identical.

        Note what is *not* distributed here. All ``V`` views of one image stay on
        the rank that owns that image -- ``DistributedSampler`` shards the
        dataset, not the view axis -- because Eq. 1 pairs a student view against
        the teacher's output for the *same image*. Sharding views instead would
        make the cross-view pairing a collective.
        """
        student = self._ddp.get("student")
        if student is not None:
            return student(
                views,
                return_bottleneck=return_bottleneck,
                chunk_size=chunk_size,
                return_features=return_features,
                return_aux=return_aux,
            )
        with self.capture_aux("student") if return_aux else nullcontext():
            features = self._backbone_features("student_backbone", views, chunk_size)
        outputs: list[torch.Tensor] = []
        logits = self._module("student_head")(features, return_bottleneck=return_bottleneck)
        if return_bottleneck:
            outputs.extend(logits)
        else:
            outputs.append(logits)
        if return_features:
            outputs.append(features)
        if return_aux:
            outputs.append(self.apply_aux_head("student"))
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    @torch.no_grad()
    def forward_teacher_views(
        self,
        views: torch.Tensor,
        chunk_size: int | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Prototype logits for the teacher's stacked global views, under ``no_grad``.

        With ``return_aux`` the auxiliary stage head's targets come back too, so
        the C4 arm scores its second objective against an EMA teacher exactly as
        the primary one does rather than against the student's own output.
        """
        with self.capture_aux("teacher") if return_aux else nullcontext():
            features = self._backbone_features("teacher_backbone", views, chunk_size)
        logits = self._module("teacher_head")(features)
        if not return_aux:
            return logits
        return logits, self.apply_aux_head("teacher")


def build_dino(backbone_cfg: Any, head_cfg: Any, freeze_last_layer_epochs: int = 1) -> DINO:
    """Instantiate :class:`DINO` from the ``model.backbone`` and ``model.head`` nodes.

    Refuses ``model.backbone.freeze=true``. That flag is a *stage-2* setting --
    it means "the trunk is a fixed feature extractor and only the head trains" --
    and honouring it here would turn self-distillation into a projection head
    fitted to frozen ImageNet features, which adapts nothing to the domain and is
    the exact failure this stage exists to avoid. There is no error PyTorch would
    raise for it: the loss would fall, the checkpoint would be written, and the
    published encoder would be the ImageNet one. The comparison it looks like it
    is asking for is a real experiment, and it lives at
    ``experiment=control_imagenet_frozen`` in stage 2.
    """
    if bool(getattr(backbone_cfg, "freeze", False)):
        raise ValueError(
            "model.backbone.freeze=true is not a stage-1 configuration. DINO fine-tunes the "
            "trunk; freezing it would train the projection head against fixed ImageNet "
            "features and publish an unadapted encoder, with no error and a normal-looking "
            "loss curve.\n"
            "  - to pretrain (the intended regime): model.backbone.freeze=false\n"
            "  - for the frozen-ImageNet comparison, run the stage-2 control instead:\n"
            "      python -m src.trainers.moe_finetune experiment=control_imagenet_frozen"
        )
    return DINO(
        backbone_name=str(backbone_cfg.name),
        input_dim=int(head_cfg.in_dim),
        hidden_dim=int(head_cfg.hidden_dim),
        bottleneck_dim=int(head_cfg.bottleneck_dim),
        out_dim=int(head_cfg.out_dim),
        pretrained=bool(getattr(backbone_cfg, "pretrained", False)),
        dynamic_img_size=bool(getattr(backbone_cfg, "dynamic_img_size", True)),
        drop_path_rate=float(getattr(backbone_cfg, "drop_path_rate", 0.0) or 0.0),
        projection_layers=int(getattr(head_cfg, "num_layers", 3)),
        projection_use_batch_norm=getattr(head_cfg, "use_batch_norm", "layer"),
        projection_norm_last_layer=bool(getattr(head_cfg, "norm_last_layer", True)),
        freeze_last_layer_epochs=int(freeze_last_layer_epochs),
        aux_stage=(
            None
            if getattr(head_cfg, "aux_stage", None) is None
            else int(head_cfg.aux_stage)
        ),
        aux_out_dim=(
            None
            if getattr(head_cfg, "aux_out_dim", None) is None
            else int(head_cfg.aux_out_dim)
        ),
        aux_weight=float(getattr(head_cfg, "aux_weight", 1.0) or 1.0),
    )
