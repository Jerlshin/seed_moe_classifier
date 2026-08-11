"""The stage-1 throughput refactor must not change what stage 1 computes.

Six things were moved for speed: the multi-view forward was fused into one
backbone call, view normalisation and the local-crop upsample moved to the GPU,
views cross the dataloader boundary as ``uint8``, the cross-view loss became a
single contraction, the EMA became a ``foreach`` update, and the compiled
callables were kept off the module tree. Every one of them is a place where the
run would keep training, keep producing a plausible loss curve, and quietly
optimise a different objective.

Each test below pits the fast path against a literal re-implementation of the
slow one. They are equality tests, not tolerance tests, wherever equality is
achievable -- because "close enough" is precisely the failure mode that cost this
refactor a debugging session (see :func:`test_deferred_upsample_needs_antialias`).
"""

from __future__ import annotations

import copy
import random

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from src.datasets.dataset import MultiCropCollate
from src.datasets.transforms import DataAugmentationDINO
from src.losses.dino import CustomDINOLoss
from src.models.backbones.sdpa_attention import (
    convert_swinv2_attention_to_sdpa,
    sdpa_window_attention_forward,
)
from src.models.backbones.swinv2_dino import DINO
from src.utils.training.ema import TeacherEmaUpdater

NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)


def _random_image(height: int = 53, width: int = 61) -> Image.Image:
    return Image.fromarray(
        (np.random.rand(height, width, 3) * 255).astype(np.uint8), mode="RGB"
    )


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _reference_cross_view_loss(loss_fn, student_output, teacher_output, epoch,
                               student_ids, teacher_ids):
    """The pre-refactor double loop, kept verbatim as the reference."""
    student_chunks = (student_output / loss_fn.student_temp).chunk(loss_fn.num_crops)
    teacher_chunks = loss_fn.teacher_targets(teacher_output, epoch).detach().chunk(
        loss_fn.num_global_crops
    )
    total = student_output.new_zeros(())
    terms = 0
    for teacher_id, teacher_probs in zip(teacher_ids, teacher_chunks):
        for student_id, student_logits in zip(student_ids, student_chunks):
            if student_id == teacher_id:
                continue
            total = total + torch.sum(
                -teacher_probs * F.log_softmax(student_logits, dim=-1), dim=-1
            ).mean()
            terms += 1
    return total / terms


# ------------------------------------------------------------------- the loss


@pytest.mark.parametrize("centering", ["sinkhorn", "ema"])
def test_vectorised_cross_view_loss_matches_the_double_loop(centering):
    """One einsum over all 12 pairs == 12 separate log-softmax-and-dot terms.

    The tolerance is *relative*, and deliberately so. The two forms sum the same
    terms in a different order -- the einsum reduces over the prototype axis in
    one pass where the loop accumulated pair by pair -- so at a loss magnitude of
    ~30 they disagree in the last fp32 digit (~4e-6 absolute, ~1e-7 relative). An
    absolute bound tight enough to be meaningful at loss ~1 would be measuring
    float32 epsilon rather than the refactor.
    """
    loss_fn = CustomDINOLoss(
        out_dim=512, num_crops=6, warmup_teacher_temp=0.04, teacher_temp=0.07,
        warmup_teacher_temp_epochs=3, num_epochs=10, centering=centering, lambda_koleo=0.0,
    )
    batch = 5
    student = torch.randn(6 * batch, 512, requires_grad=True)
    teacher = torch.randn(2 * batch, 512)
    student_ids, teacher_ids = list(range(6)), [0, 1]

    actual = loss_fn.compute_dino_loss(student, teacher, 2, student_ids, teacher_ids)
    expected = _reference_cross_view_loss(loss_fn, student, teacher, 2, student_ids, teacher_ids)
    assert actual.item() == pytest.approx(expected.item(), rel=1e-6)

    # In float64 the reassociation has ~9 more digits of headroom, so agreement
    # there is evidence about the algebra rather than about the accumulator.
    wide_loss_fn = copy.deepcopy(loss_fn).double()
    wide_student, wide_teacher = student.double().detach(), teacher.double()
    assert wide_loss_fn.compute_dino_loss(
        wide_student, wide_teacher, 2, student_ids, teacher_ids
    ).item() == pytest.approx(
        _reference_cross_view_loss(
            wide_loss_fn, wide_student, wide_teacher, 2, student_ids, teacher_ids
        ).item(),
        rel=1e-12,
    )

    # The gradient is what actually trains the student; matching values with a
    # mismatched gradient is the harder failure to notice.
    actual_grad = torch.autograd.grad(
        loss_fn.compute_dino_loss(student, teacher, 2, student_ids, teacher_ids),
        student, retain_graph=True,
    )[0]
    expected_grad = torch.autograd.grad(
        _reference_cross_view_loss(loss_fn, student, teacher, 2, student_ids, teacher_ids),
        student,
    )[0]
    assert torch.allclose(actual_grad, expected_grad, atol=1e-7)


def test_cross_view_pairing_follows_the_ids_not_the_positions():
    """The mask is built from the view ids, so a non-canonical order still pairs.

    The globals being the student's first two views is a convention, not an
    invariant the loss may assume -- violating it would silently skip a
    global-local pair and score a view against itself.
    """
    loss_fn = CustomDINOLoss(
        out_dim=64, num_crops=4, warmup_teacher_temp=0.04, teacher_temp=0.04,
        warmup_teacher_temp_epochs=0, num_epochs=1, centering="ema", lambda_koleo=0.0,
    )
    batch = 3
    student = torch.randn(4 * batch, 64)
    teacher = torch.randn(2 * batch, 64)
    student_ids, teacher_ids = [7, 3, 1, 9], [3, 9]  # globals are views 1 and 3

    actual = loss_fn.compute_dino_loss(student, teacher, 0, student_ids, teacher_ids)
    expected = _reference_cross_view_loss(loss_fn, student, teacher, 0, student_ids, teacher_ids)
    assert actual.item() == pytest.approx(expected.item(), rel=1e-6)


def test_loss_rejects_view_blocks_that_do_not_divide():
    """A batch-major stack would land here rather than train on the wrong pairs."""
    loss_fn = CustomDINOLoss(
        out_dim=8, num_crops=6, warmup_teacher_temp=0.04, teacher_temp=0.04,
        warmup_teacher_temp_epochs=0, num_epochs=1, centering="ema", lambda_koleo=0.0,
    )
    with pytest.raises(ValueError, match="not divisible by num_crops"):
        loss_fn.compute_dino_loss(torch.randn(14, 8), torch.randn(4, 8), 0)


def test_diagnostics_stay_on_device_until_asked_for():
    """Every ``float(tensor)`` inside the step is a pipeline stall.

    The collapse diagnostics are logged once in ten steps, so they are kept as
    device tensors and only converted through ``last_metrics``.
    """
    loss_fn = CustomDINOLoss(
        out_dim=16, num_crops=6, warmup_teacher_temp=0.04, teacher_temp=0.04,
        warmup_teacher_temp_epochs=0, num_epochs=1, centering="ema", lambda_koleo=0.5,
    )
    student = [torch.randn(3, 16) for _ in range(6)]
    teacher = [torch.randn(3, 16) for _ in range(2)]

    loss_fn(student, teacher, epoch=0, student_embeddings=torch.randn(6, 16))
    assert all(torch.is_tensor(value) for value in loss_fn._metric_tensors.values())
    assert {"teacher_entropy", "prototype_kl_to_uniform", "koleo"} <= set(loss_fn._metric_tensors)
    assert all(isinstance(value, float) for value in loss_fn.last_metrics.values())

    loss_fn._metric_tensors.clear()
    loss_fn.metrics_enabled = False
    loss_fn(student, teacher, epoch=0, student_embeddings=torch.randn(6, 16))
    assert loss_fn._metric_tensors == {}, "diagnostics were computed on a non-logging step"


# ------------------------------------------------------- transforms and views


def test_deferred_upsample_reproduces_the_cpu_pipeline_exactly():
    """uint8 + GPU normalise + GPU upsample == the float CPU pipeline, bitwise.

    The two paths must agree on the *globals* (which only move the normalise) and
    on the *locals* (which also move the resize).
    """
    image = _random_image()

    _seed_all(11)
    cpu_only = DataAugmentationDINO(image_size=256, local_crop_size=101)
    _, reference_crops = cpu_only(image)

    _seed_all(11)
    deferred = DataAugmentationDINO(
        image_size=256, local_crop_size=101, output_uint8=True, return_original=False
    )
    original, crops = deferred(image)

    assert original is None
    assert all(crop.dtype == torch.uint8 for crop in crops)
    assert [tuple(crop.shape[-2:]) for crop in crops] == [(256, 256)] * 2 + [(101, 101)] * 4

    mean = torch.tensor(NORMALIZE_MEAN).view(1, -1, 1, 1)
    std = torch.tensor(NORMALIZE_STD).view(1, -1, 1, 1)
    for index, (crop, reference) in enumerate(zip(crops, reference_crops)):
        finished = crop.unsqueeze(0).float().div_(255.0).sub_(mean).div_(std)
        if index >= 2:
            finished = F.interpolate(
                finished, size=(256, 256), mode="bicubic", align_corners=False, antialias=True
            )
        assert torch.equal(finished[0], reference), f"view {index} diverged"


def test_deferred_upsample_needs_antialias():
    """``antialias=True`` is not optional, and this is why it has its own test.

    ``torchvision.transforms.Resize`` defaults to ``antialias=True``, and torch's
    antialiased bicubic kernel differs from the plain one *even when
    upsampling* -- contrary to the usual intuition that antialiasing only matters
    on the way down. Without the flag the GPU path tracks the CPU path to within
    a few hundredths of a normalised unit: far too small to look like a bug, and
    easily large enough to change what a 101 px crop teaches the model.
    """
    crop = torch.randn(1, 3, 101, 101)
    reference = T.Resize((256, 256), interpolation=T.InterpolationMode.BICUBIC)(crop)

    with_antialias = F.interpolate(
        crop, size=(256, 256), mode="bicubic", align_corners=False, antialias=True
    )
    without_antialias = F.interpolate(
        crop, size=(256, 256), mode="bicubic", align_corners=False
    )
    assert torch.equal(with_antialias, reference)
    assert not torch.allclose(without_antialias, reference, atol=1e-3)


def test_view_geometry_is_reported_honestly():
    deferred = DataAugmentationDINO(image_size=256, local_crop_size=101, output_uint8=True)
    assert deferred.view_sizes == [256, 256, 101, 101, 101, 101]
    assert deferred.upsample_locals_on_device

    cpu_only = DataAugmentationDINO(image_size=256, local_crop_size=101)
    assert cpu_only.view_sizes == [256] * 6
    assert not cpu_only.upsample_locals_on_device


def test_multicrop_collate_is_view_major():
    """Views on the outside, batch on the inside.

    ``flatten(0, 1)`` on this layout gives the ``[all of view 0, all of view 1,
    ...]`` block order the loss chunks on. A batch-major stack would interleave
    the views and score every pair against the wrong partner.
    """
    batch = [
        (None, [torch.full((3, 4, 4), float(view + 10 * sample)) for view in range(6)], sample, f"p{sample}")
        for sample in range(3)
    ]
    collated = MultiCropCollate(num_global_crops=2)(batch)

    assert collated.global_views.shape == (2, 3, 3, 4, 4)
    assert collated.local_views.shape == (4, 3, 3, 4, 4)
    assert collated.originals is None
    assert collated.paths == ("p0", "p1", "p2")

    flattened = collated.global_views.flatten(0, 1)
    # Row v * B + b must be sample b's view v.
    for view in range(2):
        for sample in range(3):
            assert flattened[view * 3 + sample][0, 0, 0].item() == view + 10 * sample


# ---------------------------------------------------------- model and the EMA


@pytest.mark.slow
def test_fused_multiview_forward_matches_the_per_view_loop():
    model = DINO(
        backbone_name="swinv2_tiny_window16_256", input_dim=768,
        hidden_dim=128, bottleneck_dim=32, out_dim=64, pretrained=False,
    ).eval()
    views = [torch.randn(2, 3, 256, 256) for _ in range(3)]

    with torch.no_grad():
        per_view = torch.cat(
            [model.student_head(model._pool(model.student_backbone(view))) for view in views]
        )
        fused = model.forward_student_views(torch.cat(views, dim=0))
        chunked = model.forward_student_views(torch.cat(views, dim=0), chunk_size=2)
        teacher_per_view = torch.cat(
            [model.teacher_head(model._pool(model.teacher_backbone(view))) for view in views[:2]]
        )
        teacher_fused = model.forward_teacher_views(torch.cat(views[:2], dim=0))

    assert torch.allclose(per_view, fused, atol=1e-5)
    assert torch.allclose(fused, chunked, atol=1e-5)
    assert torch.allclose(teacher_per_view, teacher_fused, atol=1e-5)


def test_fused_ema_matches_the_reference_update():
    from lightly.models.utils import update_momentum

    student = torch.nn.Sequential(
        torch.nn.Linear(8, 8), torch.nn.LayerNorm(8), torch.nn.Linear(8, 4)
    )
    reference_teacher = copy.deepcopy(student)
    fused_teacher = copy.deepcopy(student)
    for parameter in student.parameters():
        parameter.data.normal_()

    update_momentum(student, reference_teacher, m=0.996)
    TeacherEmaUpdater([(student, fused_teacher)]).update(0.996)

    for expected, actual in zip(reference_teacher.parameters(), fused_teacher.parameters()):
        assert torch.allclose(expected, actual, atol=1e-7)


def test_ema_refuses_a_mismatched_pair():
    """A silent ``zip`` truncation would leave the tail of the teacher frozen."""
    student = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    teacher = torch.nn.Sequential(torch.nn.Linear(4, 4))
    with pytest.raises(ValueError, match="EMA pair mismatch"):
        TeacherEmaUpdater([(student, teacher)])


# ------------------------------------------------- SDPA window attention

def _stock_and_converted_window_attention(dtype=torch.float32, attn_drop: float = 0.0):
    """One stock timm WindowAttention and a converted deep copy of it."""
    from timm.models.swin_transformer_v2 import WindowAttention

    torch.manual_seed(3)
    stock = WindowAttention(
        dim=32, window_size=(4, 4), num_heads=4, attn_drop=attn_drop
    ).to(dtype)
    converted = copy.deepcopy(stock)
    count = convert_swinv2_attention_to_sdpa(converted)
    return stock, converted, count


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("with_mask", [False, True])
def test_sdpa_window_attention_matches_stock_values_and_gradients(dtype, with_mask):
    """SDPA(q̂s, k̂, v, bias, scale=1) == softmax((q̂k̂ᵀ)s + bias)v -- values and grads.

    fp64 agreement at 1e-12 relative is evidence about the *algebra* (folding
    the per-head logit scale into q, merging the window axis into the head axis
    for the shift mask); fp32 agreement at 1e-5 is evidence the reassociation
    stays inside float epsilon. The gradients are the part that actually trains
    -- logit_scale's gradient in the stock path flows through the saved q̂k̂ᵀ
    matrix, and in the SDPA path through q itself, so matching them is exactly
    the claim that the ~12 GB of saved attention matrices were redundant.
    """
    stock, converted, count = _stock_and_converted_window_attention(dtype)
    assert count == 1

    torch.manual_seed(5)
    n = 16  # (4, 4) window
    if with_mask:
        x = torch.randn(6, n, 32, dtype=dtype)  # batch 3 x 2 windows
        mask = torch.randn(2, n, n, dtype=dtype)
    else:
        x = torch.randn(3, n, 32, dtype=dtype)
        mask = None

    stock_x = x.clone().requires_grad_(True)
    converted_x = x.clone().requires_grad_(True)
    stock_out = stock(stock_x, mask=mask)
    converted_out = converted(converted_x, mask=mask)
    tolerance = {"rtol": 1e-12, "atol": 1e-12} if dtype is torch.float64 else {"atol": 1e-5}
    assert torch.allclose(stock_out, converted_out, **tolerance)

    stock_out.square().sum().backward()
    converted_out.square().sum().backward()
    # The gradient into the *input* is what the surrounding SwinV2 block trains
    # on; the parameter gradients are what the module itself trains on. Both
    # must match for the rewrite to be a drop-in.
    assert stock_x.grad is not None
    assert torch.allclose(stock_x.grad, converted_x.grad, **tolerance), "input gradient diverged"
    for (name, stock_param), (_, converted_param) in zip(
        stock.named_parameters(), converted.named_parameters()
    ):
        assert stock_param.grad is not None, name
        assert torch.allclose(
            stock_param.grad, converted_param.grad, **tolerance
        ), f"gradient diverged on {name}"


SWINV2_BASE_STAGE_GEOMETRIES = [
    # (dim, heads, window): the four attention shapes of swinv2_base_window16_256
    # at 256 px. The 12 GB-card incident refused every heads >= 16 module while
    # converting heads 4/8, so the parity contract is pinned per real stage
    # shape, not only on a toy geometry.
    (128, 4, (16, 16)),
    (256, 8, (16, 16)),
    (512, 16, (16, 16)),
    (1024, 32, (8, 8)),
]


@pytest.mark.parametrize("dim,heads,window", SWINV2_BASE_STAGE_GEOMETRIES)
def test_sdpa_parity_holds_at_every_base_stage_geometry(dim, heads, window):
    """fp64 parity — values, input grads, parameter grads — at production shapes.

    fp64 is where the parity guard itself now operates, so this test is the
    guard's contract restated per stage: at 1e-10 the only thing that can fail
    it is semantic drift, never kernel rounding.
    """
    from timm.models.swin_transformer_v2 import WindowAttention

    torch.manual_seed(7)
    stock = WindowAttention(dim=dim, window_size=window, num_heads=heads).double()
    converted = copy.deepcopy(stock)
    assert convert_swinv2_attention_to_sdpa(converted) == 1

    n = window[0] * window[1]
    for num_windows in (1, 2):  # non-shifted probe, then a shifted-style mask
        x = torch.randn(2 * num_windows, n, dim, dtype=torch.float64)
        mask = (
            torch.randn(num_windows, n, n, dtype=torch.float64) if num_windows > 1 else None
        )
        stock_x = x.clone().requires_grad_(True)
        converted_x = x.clone().requires_grad_(True)
        stock_out = stock(stock_x, mask=mask)
        converted_out = converted(converted_x, mask=mask)
        assert torch.allclose(stock_out, converted_out, atol=1e-10, rtol=1e-10)

        stock.zero_grad(set_to_none=True)
        converted.zero_grad(set_to_none=True)
        stock_out.square().sum().backward()
        converted_out.square().sum().backward()
        assert torch.allclose(stock_x.grad, converted_x.grad, atol=1e-10, rtol=1e-10)
        for (name, stock_param), (_, converted_param) in zip(
            stock.named_parameters(), converted.named_parameters()
        ):
            assert torch.allclose(
                stock_param.grad, converted_param.grad, atol=1e-10, rtol=1e-10
            ), f"gradient diverged on {name} (num_windows={num_windows})"


def test_sdpa_conversion_is_immune_to_module_precision():
    """The parity probe runs in fp64 on a copy, whatever the module runs in.

    Regression for the 12 GB-card incident: probing in the module's own compute
    precision measures kernel rounding (TF32 selection on CUDA, bf16
    reassociation here) instead of algebra, and refuses correct modules. A bf16
    module reassociates at ~1e-2 — far past any honest gate — so it converts
    if and only if the probe is precision-independent. The live module must
    come back untouched: same dtype, same training flag, parameters unmoved.
    """
    from timm.models.swin_transformer_v2 import WindowAttention

    torch.manual_seed(11)
    module = WindowAttention(dim=64, window_size=(8, 8), num_heads=4).to(torch.bfloat16)
    module.train()
    before = {name: param.clone() for name, param in module.named_parameters()}

    assert convert_swinv2_attention_to_sdpa(module) == 1
    assert module.training, "the probe flipped the live module's mode"
    for name, param in module.named_parameters():
        assert param.dtype is torch.bfloat16, name
        assert torch.equal(param, before[name]), f"the probe moved parameter {name}"


@pytest.mark.parametrize("with_mask", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_sdpa_half_precision_stays_inside_the_stock_paths_own_error_envelope(dtype, with_mask):
    """In half precision the rewrite must track the exact function as well as eager does.

    Neither half-precision path can match an equality gate — bf16
    reassociation alone is ~1e-2 — so the honest contract is relative: measured
    against an fp64 reference of the same module, the SDPA path's error may not
    exceed twice the stock path's (measured ratio is ~0.6-0.8: the fused softmax
    accumulates in higher precision, so the rewrite is typically *closer* to the
    exact function than eager half precision is).

    fp16 is covered as well as bf16 because fp16 is not a fallback on the
    hardware this has to run on — it is the automatic choice on a T4, which has
    no hardware bfloat16 at all. The cosine-attention rewrite folds a clamped
    per-head logit scale (up to 100x) into the normalised q, and fp16's narrower
    exponent is exactly where that would be expected to go wrong if anywhere.
    """
    from timm.models.swin_transformer_v2 import WindowAttention

    torch.manual_seed(13)
    module = WindowAttention(dim=128, window_size=(8, 8), num_heads=4).eval()
    n = 64
    x = torch.randn(4 if with_mask else 2, n, 128)
    mask = torch.randn(2, n, n) if with_mask else None

    reference_module = copy.deepcopy(module).double()
    stock_half = copy.deepcopy(module).to(dtype)
    sdpa_half = copy.deepcopy(stock_half)
    assert convert_swinv2_attention_to_sdpa(sdpa_half) == 1

    with torch.no_grad():
        mask64 = mask.double() if mask is not None else None
        mask_half = mask.to(dtype) if mask is not None else None
        reference = reference_module(x.double(), mask=mask64)
        stock_error = (stock_half(x.to(dtype), mask=mask_half).double() - reference).abs().max()
        sdpa_error = (sdpa_half(x.to(dtype), mask=mask_half).double() - reference).abs().max()

    assert float(stock_error) < 0.1, "eager half precision itself is broken; the comparison is meaningless"
    assert float(sdpa_error) <= 2.0 * float(stock_error) + 1e-3


def test_sdpa_conversion_leaves_state_dict_and_module_tree_alone():
    """Conversion rebinds ``forward`` on the instance and touches nothing else.

    The bare ``student_backbone.state_dict()`` is the only handoff to stage 2,
    so like ``torch.compile`` the conversion must never rename a key or add a
    submodule. It must also be idempotent: a second call finds every forward
    already bound and converts nothing twice.
    """
    stock, converted, count = _stock_and_converted_window_attention()
    assert count == 1
    assert list(converted.state_dict()) == list(stock.state_dict())
    assert dict(converted.named_modules()).keys() == dict(stock.named_modules()).keys()
    assert convert_swinv2_attention_to_sdpa(converted) == 0  # idempotent


def test_sdpa_conversion_refuses_what_it_cannot_prove():
    """The parity guard is the licence to run this at all.

    The rewrite reimplements the forward from module attributes, so a timm
    whose semantics have drifted would make it silently wrong. A module whose
    stock forward disagrees with the rewrite -- simulated here by binding a
    different forward -- must stay unconverted, as must a module using
    attention dropout (SDPA matches it in distribution, not in RNG draws).
    """
    import types

    from timm.models.swin_transformer_v2 import WindowAttention

    drifted = WindowAttention(dim=32, window_size=(4, 4), num_heads=4)
    drifted.forward = types.MethodType(
        lambda self, x, mask=None: self.proj(x), drifted
    )
    assert convert_swinv2_attention_to_sdpa(drifted) == 0

    dropout = WindowAttention(dim=32, window_size=(4, 4), num_heads=4, attn_drop=0.1)
    assert convert_swinv2_attention_to_sdpa(dropout) == 0


@pytest.mark.slow
def test_sdpa_conversion_preserves_the_full_dino_forward():
    """End to end: a converted DINO computes the same student and teacher outputs."""
    model = DINO(
        backbone_name="swinv2_tiny_window16_256", input_dim=768,
        hidden_dim=64, bottleneck_dim=16, out_dim=32, pretrained=False,
    ).eval()
    reference = copy.deepcopy(model)

    state_before = list(model.student_backbone.state_dict())
    applied = model.configure_runtime(compile_enabled=False, sdpa_attention=True)
    assert applied["sdpa_attention_modules"] > 0
    assert list(model.student_backbone.state_dict()) == state_before

    views = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        assert torch.allclose(
            model.forward_student_views(views),
            reference.forward_student_views(views),
            atol=1e-5,
        )
        assert torch.allclose(
            model.forward_teacher_views(views),
            reference.forward_teacher_views(views),
            atol=1e-5,
        )


@pytest.mark.slow
def test_compiled_callables_never_reach_the_state_dict():
    """``_orig_mod.`` in these keys would hand stage 2 a randomly initialised trunk.

    ``student_backbone.state_dict()`` is the only handoff between the two stages,
    and stage 2 loads it with ``checkpoint_strict: false`` -- so a prefixed
    checkpoint matches zero keys, logs one line, and trains to completion against
    a random encoder.
    """
    model = DINO(
        backbone_name="swinv2_tiny_window16_256", input_dim=768,
        hidden_dim=64, bottleneck_dim=16, out_dim=32, pretrained=False,
    )
    model.configure_runtime(compile_enabled=False, fused_attention=True)

    assert not any(key.startswith("_orig_mod") for key in model.student_backbone.state_dict())
    assert not any("_compiled" in key for key in model.state_dict())
    assert "_compiled" not in dict(model.named_modules())
