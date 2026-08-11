"""Precision, capability detection and reproducibility.

These pin the decisions that let one configuration file run unchanged on an
A100, on Kaggle's T4s, on Windows and on CPU -- and pin the parts of the
objective that must **not** follow the autocast dtype when it does.

The T4 is the case worth naming. ``sm_75`` has no hardware bfloat16, no TF32 and
no FlashAttention SDPA backend, so ``amp: auto`` resolves to fp16 with a
``GradScaler`` where an Ampere card gets bf16. fp16 has ~5 exponent bits, and
three parts of this objective genuinely cannot live in it: the Sinkhorn
normaliser (unit-scale logits at ``tau = 0.04`` reach ``exp(25)``), the
log-softmax over 8,192 prototypes, and the KoLeo pairwise distances (two
near-duplicate crops underflow to *exactly* zero distance, and ``-log(0)``
clamped is a large gradient handed to the optimiser for a pair that should have
contributed nothing). All three are pinned to fp32 inside the autocast region,
and that is what these tests check -- not that the numbers are close, but that
the promotion happens at all.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.losses.dino import CustomDINOLoss, koleo_regularizer, sinkhorn_knopp
from src.utils.training import device as device_module
from src.utils.training.device import (
    autocast_context,
    build_grad_scaler,
    compile_available,
    describe_accelerator,
    resolve_amp,
    resolve_compile,
)

CUDA = torch.device("cuda")
CPU = torch.device("cpu")


# ------------------------------------------------------------ dtype selection


@pytest.mark.parametrize("requested", ["off", "false", "fp32", None])
def test_amp_off_means_no_autocast_region_at_all(requested):
    amp = resolve_amp(CUDA, requested)
    assert not amp.enabled and amp.dtype is None and not amp.needs_scaler
    assert amp.label == "off"


def test_cpu_and_mps_never_autocast():
    """Autocast exists on both; neither path has been validated for this objective."""
    for device in (CPU, torch.device("mps")):
        assert not resolve_amp(device, "auto").enabled
        assert not resolve_amp(device, "bf16").enabled


def test_auto_picks_bf16_on_ampere_and_fp16_below_it(monkeypatch):
    """One flag, two hardware answers -- and only one of them needs a scaler.

    bf16 carries fp32's exponent range, so gradients do not underflow and loss
    scaling would be pure overhead. fp16 has ~5 exponent bits and silently
    flushes small gradients to zero without one.
    """
    monkeypatch.setattr(device_module, "supports_bf16", lambda: True)
    ampere = resolve_amp(CUDA, "auto")
    assert ampere.dtype is torch.bfloat16 and not ampere.needs_scaler
    assert ampere.label == "bf16"
    assert build_grad_scaler(ampere) is None

    monkeypatch.setattr(device_module, "supports_bf16", lambda: False)
    turing = resolve_amp(CUDA, "auto")
    assert turing.dtype is torch.float16 and turing.needs_scaler
    assert turing.label == "fp16"


def test_bf16_requested_on_a_t4_is_downgraded_rather_than_refused(monkeypatch):
    """A config that runs on the dev box must still run on Kaggle.

    Emulated bf16 on ``sm_75`` is neither fast nor accurate, and refusing to
    start would mean maintaining a second config for the T4s. The downgrade is
    logged, and it is safe here only because the three fp16-hostile terms are
    pinned to fp32 -- which the tests below check.
    """
    monkeypatch.setattr(device_module, "supports_bf16", lambda: False)
    amp = resolve_amp(CUDA, "bf16")
    assert amp.dtype is torch.float16 and amp.needs_scaler


def test_an_unknown_amp_mode_is_an_error_not_a_silent_default():
    with pytest.raises(ValueError, match="Unsupported amp mode"):
        resolve_amp(CUDA, "float8")


def test_the_legacy_boolean_flag_still_means_auto(monkeypatch):
    """``experiment.training.amp: true`` predates the named modes."""
    monkeypatch.setattr(device_module, "supports_bf16", lambda: True)
    assert resolve_amp(CUDA, True).dtype is torch.bfloat16
    assert not resolve_amp(CUDA, False).enabled


def test_autocast_context_is_a_real_no_op_when_disabled():
    """Callers write one ``with`` and get an unautocast region when off."""
    amp = resolve_amp(CPU, "off")
    with autocast_context(amp):
        assert not torch.is_autocast_enabled("cpu")


# --------------------------------------------- what must stay fp32 regardless


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sinkhorn_promotes_half_precision_inputs(dtype):
    """The normaliser divides by 0.04 and exponentiates. Not in 5 exponent bits.

    Returned in fp32 whatever came in, because the result is a training
    *target*: a doubly-stochastic assignment computed at ~3 decimal digits of
    mantissa is not something to hand the student, and the cast costs nothing
    next to the backbone forward that produced the logits.
    """
    logits = torch.randn(8, 32, dtype=dtype)
    assignment = sinkhorn_knopp(logits, temperature=0.04, iterations=3)
    assert assignment.dtype is torch.float32
    assert torch.isfinite(assignment).all()
    torch.testing.assert_close(
        assignment.sum(dim=1), torch.ones(8), rtol=1e-4, atol=1e-5
    )


def test_sinkhorn_does_not_downcast_a_double_precision_input():
    """``_at_least_float32`` promotes, it does not normalise to fp32.

    A numerical test written in double precision would be silently invalidated
    by a bare ``.float()`` here.
    """
    assignment = sinkhorn_knopp(torch.randn(6, 16, dtype=torch.float64), temperature=0.07)
    assert assignment.dtype is torch.float64


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_koleo_promotes_and_stays_finite_on_near_duplicates(dtype):
    """Near-identical crops are this dataset's normal case, not its edge case.

    27 sub-varieties of four crops are near-duplicates by construction, so the
    nearest-neighbour distance KoLeo takes the log of is routinely tiny. In half
    precision it underflows to exactly zero and the clamp hands the optimiser
    ``-log(eps)`` for a pair that should have contributed almost nothing.
    """
    base = torch.randn(4, 16)
    features = torch.cat([base, base + 1e-3]).to(dtype)
    value = koleo_regularizer(features)
    assert value.dtype is torch.float32
    assert torch.isfinite(value)


def test_the_dino_loss_returns_fp32_under_a_half_precision_input():
    """The whole objective, not just its pieces."""
    criterion = CustomDINOLoss(
        out_dim=32, num_crops=4, warmup_teacher_temp=0.04, teacher_temp=0.07,
        warmup_teacher_temp_epochs=1, num_epochs=2, lambda_koleo=0.1,
    )
    student = torch.randn(4 * 3, 32, dtype=torch.bfloat16, requires_grad=True)
    teacher = torch.randn(2 * 3, 32, dtype=torch.bfloat16)
    loss = criterion(
        student, teacher, epoch=0,
        student_view_ids=[0, 1, 2, 3], teacher_view_ids=[0, 1],
        student_embeddings=torch.randn(2 * 3, 8, dtype=torch.bfloat16, requires_grad=True),
    )
    assert loss.dtype is torch.float32
    assert torch.isfinite(loss)


def test_half_precision_stays_inside_a_stated_envelope():
    """bf16 inputs give the same loss to within bf16's own resolution.

    The point is not that the numbers match -- they cannot, bf16 has ~3 decimal
    digits -- but that the *disagreement is bounded by the input precision* and
    not by something in the loss overflowing or underflowing. A term that had
    silently saturated would show up here as a difference far outside this
    envelope, or as a non-finite value.
    """
    torch.manual_seed(0)
    criterion = CustomDINOLoss(
        out_dim=64, num_crops=4, warmup_teacher_temp=0.04, teacher_temp=0.07,
        warmup_teacher_temp_epochs=1, num_epochs=2, lambda_koleo=0.1,
    )
    student = torch.randn(4 * 5, 64)
    teacher = torch.randn(2 * 5, 64)
    embeddings = torch.randn(2 * 5, 16)
    kwargs = dict(epoch=0, student_view_ids=[0, 1, 2, 3], teacher_view_ids=[0, 1])

    exact = criterion(student, teacher, student_embeddings=embeddings, **kwargs)
    half = criterion(
        student.bfloat16(), teacher.bfloat16(),
        student_embeddings=embeddings.bfloat16(), **kwargs,
    )
    # bf16 has 8 mantissa bits: ~4e-3 relative on the inputs alone.
    assert abs(float(half) - float(exact)) / abs(float(exact)) < 0.05


# ----------------------------------------------------- capability and compile


def test_the_accelerator_report_is_complete_on_cpu():
    """Runs everywhere, including a machine with no GPU at all."""
    report = describe_accelerator(CPU)
    assert report.device_type == "cpu"
    assert report.torch_version == torch.__version__
    payload = report.as_dict()
    for key in ("device_type", "supports_bf16", "compile_available", "torch_version"):
        assert key in payload
    assert isinstance(report.summary_line(), str)


def test_compile_availability_reports_a_reason_when_it_says_no():
    """"Why is Windows slower" has to be answerable from the run's own artifacts."""
    available, reason = compile_available()
    assert isinstance(available, bool)
    if not available:
        assert reason, "an unavailable compiler must explain itself"


def test_compile_auto_stays_eager_off_cuda():
    """One config file, four platforms.

    ``auto`` compiles only where inductor can actually emit a kernel; ``true`` on
    a machine that cannot is downgraded rather than left to fail inside the first
    step, where the traceback names inductor rather than the setting.
    """
    assert resolve_compile("auto", CPU) is False
    assert resolve_compile(False, CUDA) is False
    assert resolve_compile("false", CUDA) is False
    # `true` is honoured only where a compiler exists; this machine may or may
    # not have one, so assert the invariant rather than the value.
    assert resolve_compile(True, CUDA) == compile_available()[0]


# ------------------------------------------------------------ reproducibility


def _short_run(seed: int) -> list[float]:
    """A few steps of a tiny supervised loop, driven entirely by ``seed``."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    losses = []
    for _ in range(5):
        inputs = torch.randn(8, 8)
        targets = torch.randint(0, 4, (8,))
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(inputs), targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def test_the_same_seed_gives_the_same_loss_sequence():
    """The property the ablation table rests on.

    Eighteen variants are compared against each other at gaps of 0.5-2 pp. If two
    runs of the *same* configuration do not agree, none of those gaps means
    anything, so this is checked as an invariant rather than assumed from having
    called ``manual_seed``.
    """
    assert _short_run(1234) == _short_run(1234)


def test_a_different_seed_gives_a_different_sequence():
    """The control: the test above is not passing because nothing is random."""
    assert _short_run(1234) != _short_run(4321)
