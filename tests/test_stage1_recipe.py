"""The stage-1 training recipe: initialisation, schedules, budget, handoff.

Everything here pins a decision that is **silent when wrong**. A frozen trunk, a
teacher with stochastic depth, a weight-decay ramp that reached the LayerNorm
gains, a learning rate that ignored the batch, a warmup that never peaked -- none
of these raises, none changes the shape of the loss curve, and each one produces
a published encoder that is worse than the log says it is.

The backbone facts (parameter count, feature width, token grid, GFLOPs) are
*re-measured* here rather than asserted from memory, so a timm upgrade that
changed any of them fails loudly instead of quietly invalidating the cost table.
"""

from __future__ import annotations

import logging
import math

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf

from src.models.backbones.swinv2_dino import DINO, DINOHead, build_dino, disable_drop_path
from src.trainers.contrastive_pretrain import (
    WEIGHT_DECAY_FLAG,
    apply_weight_decay,
    build_optimizer,
    build_param_groups,
    build_scheduler,
    cosine_value,
    resolve_learning_rate,
    resolve_warmup_epochs,
)
from src.utils.training import BACKWARD_MULTIPLIER, StageOneBudget, measure_gflops_per_view
from tests.conftest import (
    LR_BASE,
    LR_REFERENCE_BATCH,
    PAPER_EMBED_DIM,
    PAPER_TOKEN_GRID,
    REVISED_BACKBONE,
    REVISED_BACKBONE_FEATURE_DIM,
    REVISED_BACKBONE_GFLOPS_PER_VIEW,
    REVISED_BACKBONE_PARAMS_M,
    SMALL_BACKBONE,
    SMALL_BACKBONE_FEATURE_DIM,
    SMALL_BACKBONE_GFLOPS_PER_VIEW,
    SMALL_BACKBONE_PARAMS_M,
    REVISED_DINO_BOTTLENECK_DIM,
    REVISED_DINO_HEAD_LAYERS,
    REVISED_DINO_HIDDEN_DIM,
    REVISED_DINO_OUT_DIM,
    REVISED_DROP_PATH_RATE,
    STAGE1_EFFECTIVE_BATCH,
    STAGE1_EPOCHS,
    STAGE1_LEARNING_RATE,
    STAGE1_WARMUP_EPOCHS,
)

LOGGER = logging.getLogger("tests.stage1")


def training_cfg(**overrides):
    """A minimal ``experiment.training`` node, matching the shipped defaults."""
    training = {
        "epochs": STAGE1_EPOCHS,
        "warmup_epochs": STAGE1_WARMUP_EPOCHS,
        "learning_rate": None,
        "lr_base": LR_BASE,
        "lr_reference_batch_size": LR_REFERENCE_BATCH,
        "lr_scaling": "linear",
        "weight_decay": 0.04,
        "weight_decay_final": 0.4,
        "fused_optimizer": False,
        "optimizer": {"name": "AdamW"},
        "scheduler": {"name": "cosine", "t_max": None, "eta_min": 0.0},
    }
    training.update(overrides)
    return OmegaConf.create({"experiment": {"training": training}})


def tiny_backbone():
    """``swinv2_tiny_window16_256`` without touching the network."""
    import timm

    return timm.create_model(REVISED_BACKBONE, pretrained=False, num_classes=0)


# --------------------------------------------------------------- the backbone


def test_tiny_backbone_dimensions_are_what_the_cost_table_claims():
    """Measured, not quoted. A timm change that moved any of these must fail here."""
    backbone = tiny_backbone()
    parameters = sum(p.numel() for p in backbone.parameters()) / 1e6

    assert backbone.num_features == REVISED_BACKBONE_FEATURE_DIM
    assert parameters == pytest.approx(REVISED_BACKBONE_PARAMS_M, abs=0.05)

    with torch.no_grad():
        tokens = backbone.forward_features(torch.zeros(1, 3, 256, 256))
    # 8x8 is the invariant stage 2's grid routing depends on. It is the same for
    # Tiny and Base at 256 px, which is what makes the capacity control a
    # drop-in.
    assert tokens.shape[1:3] == (8, 8)
    assert tokens.shape[1] * tokens.shape[2] == PAPER_TOKEN_GRID
    assert tokens.shape[-1] == REVISED_BACKBONE_FEATURE_DIM


@pytest.mark.slow
def test_measured_gflops_per_view_matches_the_reported_budget():
    """The number printed in the cost table, re-derived from dispatch."""
    measured = measure_gflops_per_view(tiny_backbone(), 256, torch.device("cpu"))
    if measured is None:
        pytest.skip("FlopCounterMode is unavailable on this torch build")
    # Backbone only; the trainer measures backbone + head, which adds ~0.01.
    assert measured == pytest.approx(REVISED_BACKBONE_GFLOPS_PER_VIEW, rel=0.02)


def test_shipped_small_trunk_costs_what_the_stage1_report_claims():
    """The trunk `conf/` actually selects, measured rather than quoted.

    The stage-1 evaluation report quotes 48.96 M
    parameters and ~25.6 GFLOPs/view, and the published 100-epoch checkpoint is a
    Small. Pinning them here is what stops those documents from drifting into a
    claim about the Tiny trunk the configs used to select.
    """
    import timm

    backbone = timm.create_model(SMALL_BACKBONE, pretrained=False, num_classes=0)
    parameters = sum(p.numel() for p in backbone.parameters()) / 1e6
    assert backbone.num_features == SMALL_BACKBONE_FEATURE_DIM
    assert parameters == pytest.approx(SMALL_BACKBONE_PARAMS_M, abs=0.05)

    with torch.no_grad():
        tokens = backbone.forward_features(torch.zeros(1, 3, 256, 256))
    # Same 8x8 grid as Tiny and Base: the reason a trunk swap is invisible to
    # stage 2's grid routing, and why only the channel width had to change.
    assert tokens.shape[1] * tokens.shape[2] == PAPER_TOKEN_GRID
    assert tokens.shape[-1] == SMALL_BACKBONE_FEATURE_DIM


@pytest.mark.slow
def test_shipped_small_trunk_gflops_match_the_evaluation_report():
    measured = measure_gflops_per_view(
        __import__("timm").create_model(SMALL_BACKBONE, pretrained=False, num_classes=0),
        256,
        torch.device("cpu"),
    )
    if measured is None:
        pytest.skip("FlopCounterMode is unavailable on this torch build")
    assert measured == pytest.approx(SMALL_BACKBONE_GFLOPS_PER_VIEW, rel=0.02)


# ----------------------------------------------------- initialisation regime


def test_build_dino_refuses_a_frozen_trunk():
    """The failure this guard exists for is silent: a normal loss, a stale encoder."""
    backbone_cfg = OmegaConf.create(
        {"name": REVISED_BACKBONE, "pretrained": False, "freeze": True, "drop_path_rate": 0.0}
    )
    head_cfg = OmegaConf.create(
        {
            "in_dim": REVISED_BACKBONE_FEATURE_DIM,
            "hidden_dim": 32,
            "bottleneck_dim": 16,
            "out_dim": 64,
            "num_layers": 3,
        }
    )
    with pytest.raises(ValueError, match="not a stage-1 configuration"):
        build_dino(backbone_cfg=backbone_cfg, head_cfg=head_cfg)

    # ... and builds happily once the trunk is trainable.
    backbone_cfg.freeze = False
    model = build_dino(backbone_cfg=backbone_cfg, head_cfg=head_cfg)
    summary = model.parameter_summary()
    assert summary["student_backbone_trainable"] == summary["backbone"] > 0


def test_stochastic_depth_reaches_the_student_and_not_the_teacher():
    """The teacher's outputs are the targets; drop path there is noise in the label."""
    model = DINO(
        backbone_name=REVISED_BACKBONE,
        input_dim=REVISED_BACKBONE_FEATURE_DIM,
        hidden_dim=32,
        bottleneck_dim=16,
        out_dim=64,
        drop_path_rate=REVISED_DROP_PATH_RATE,
    )

    def drop_probs(module: nn.Module) -> list[float]:
        return [
            child.drop_prob
            for child in module.modules()
            if type(child).__name__ == "DropPath"
        ]

    student = drop_probs(model.student_backbone)
    teacher = drop_probs(model.teacher_backbone)

    assert student, "timm should have built DropPath modules at drop_path_rate > 0"
    assert max(student) == pytest.approx(REVISED_DROP_PATH_RATE)
    assert len(teacher) == len(student)
    assert set(teacher) == {0.0}
    assert model.teacher_drop_paths_disabled > 0

    # The weights themselves must still start identical -- disabling drop path
    # touches a float attribute, never a parameter.
    for left, right in zip(
        model.student_backbone.state_dict().values(),
        model.teacher_backbone.state_dict().values(),
    ):
        assert torch.equal(left, right)


def test_teacher_is_deterministic_in_train_mode():
    """The observable consequence of the previous test, measured end to end."""
    model = DINO(
        backbone_name=REVISED_BACKBONE,
        input_dim=REVISED_BACKBONE_FEATURE_DIM,
        hidden_dim=32,
        bottleneck_dim=16,
        out_dim=64,
        drop_path_rate=0.5,  # exaggerated so a leak cannot hide in the noise
    )
    model.train()
    images = torch.randn(2, 3, 256, 256)
    torch.manual_seed(0)
    first = model.forward_teacher_views(images)
    torch.manual_seed(1)
    second = model.forward_teacher_views(images)
    assert torch.allclose(first, second), "the teacher must not sample dropped paths"


def test_disable_drop_path_is_idempotent_and_counts_only_what_it_changed():
    model = DINO(
        backbone_name=REVISED_BACKBONE,
        input_dim=REVISED_BACKBONE_FEATURE_DIM,
        hidden_dim=32,
        bottleneck_dim=16,
        out_dim=64,
        drop_path_rate=0.1,
    )
    assert disable_drop_path(model.teacher_backbone) == 0  # already silenced
    assert disable_drop_path(model.student_backbone) > 0
    assert disable_drop_path(model.student_backbone) == 0


# ------------------------------------------------------------- the DINO head


def test_dino_head_realises_the_configured_width_chain():
    """768 -> 1024 -> 1024 -> 256 -> 2048, with the bottleneck L2-normalised."""
    head = DINOHead(
        in_dim=REVISED_BACKBONE_FEATURE_DIM,
        out_dim=REVISED_DINO_OUT_DIM,
        hidden_dim=REVISED_DINO_HIDDEN_DIM,
        bottleneck_dim=REVISED_DINO_BOTTLENECK_DIM,
        num_layers=REVISED_DINO_HEAD_LAYERS,
    )
    widths = [
        (layer.in_features, layer.out_features)
        for layer in head.mlp
        if isinstance(layer, nn.Linear)
    ]
    assert widths == [
        (REVISED_BACKBONE_FEATURE_DIM, REVISED_DINO_HIDDEN_DIM),
        (REVISED_DINO_HIDDEN_DIM, REVISED_DINO_HIDDEN_DIM),
        (REVISED_DINO_HIDDEN_DIM, REVISED_DINO_BOTTLENECK_DIM),
    ]
    assert head.last_layer.weight.shape == (REVISED_DINO_OUT_DIM, REVISED_DINO_BOTTLENECK_DIM)

    logits, bottleneck = head(
        torch.randn(4, REVISED_BACKBONE_FEATURE_DIM), return_bottleneck=True
    )
    assert logits.shape == (4, REVISED_DINO_OUT_DIM)
    assert bottleneck.shape == (4, REVISED_DINO_BOTTLENECK_DIM)
    assert torch.allclose(bottleneck.norm(dim=-1), torch.ones(4), atol=1e-5)


def test_shape_report_traces_the_whole_path():
    model = DINO(
        backbone_name=REVISED_BACKBONE,
        input_dim=REVISED_BACKBONE_FEATURE_DIM,
        hidden_dim=REVISED_DINO_HIDDEN_DIM,
        bottleneck_dim=REVISED_DINO_BOTTLENECK_DIM,
        out_dim=REVISED_DINO_OUT_DIM,
    )
    report = model.shape_report(image_size=256)

    assert report["input"] == (1, 3, 256, 256)
    assert report["token_grid"] == (8, 8)
    assert report["tokens_per_image"] == PAPER_TOKEN_GRID
    assert report["head_input_dim"] == REVISED_BACKBONE_FEATURE_DIM
    assert report["head_bottleneck"] == (1, REVISED_DINO_BOTTLENECK_DIM)
    # Student and teacher must agree, or the loss contracts mismatched widths.
    assert report["student_prototypes"] == report["teacher_prototypes"]
    assert report["student_prototypes"] == (1, REVISED_DINO_OUT_DIM)


# ---------------------------------------------------------- learning rate


def test_learning_rate_scales_linearly_with_the_effective_batch():
    cfg = training_cfg()
    resolved, provenance = resolve_learning_rate(cfg, STAGE1_EFFECTIVE_BATCH, LOGGER)
    assert resolved == pytest.approx(STAGE1_LEARNING_RATE)
    assert provenance == {
        "learning_rate": pytest.approx(STAGE1_LEARNING_RATE),
        "rule": "linear",
        "lr_base": LR_BASE,
        "lr_reference_batch_size": LR_REFERENCE_BATCH,
        "effective_batch_size": STAGE1_EFFECTIVE_BATCH,
    }
    # At the reference batch the rule is the identity, which is the sanity check
    # that the scaling is anchored where the paper's number belongs.
    at_reference, _ = resolve_learning_rate(cfg, LR_REFERENCE_BATCH, LOGGER)
    assert at_reference == pytest.approx(LR_BASE)


def test_an_explicit_learning_rate_overrides_the_derivation():
    cfg = training_cfg(learning_rate=3e-5)
    resolved, provenance = resolve_learning_rate(cfg, 999, LOGGER)
    assert resolved == pytest.approx(3e-5)
    assert provenance["rule"] == "configured"


def test_lr_scaling_none_uses_the_base_rate_at_any_batch():
    cfg = training_cfg(lr_scaling="none")
    for batch in (16, 32, 256):
        resolved, provenance = resolve_learning_rate(cfg, batch, LOGGER)
        assert resolved == pytest.approx(LR_BASE)
        assert provenance["rule"] == "lr_base"


def test_unknown_lr_scaling_is_refused():
    with pytest.raises(ValueError, match="lr_scaling"):
        resolve_learning_rate(training_cfg(lr_scaling="sqrt"), 32, LOGGER)


# ------------------------------------------------------------ the schedule


def lr_curve(cfg, peak: float = 1.0) -> list[float]:
    """The LR the optimizer holds at the *start* of each epoch."""
    parameter = nn.Parameter(torch.zeros(2, 2))
    optimizer = optim.SGD([parameter], lr=peak)
    scheduler = build_scheduler(optimizer, cfg)
    curve = []
    for _ in range(int(cfg.experiment.training.epochs)):
        curve.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    return curve


def test_warmup_ramps_linearly_and_peaks_exactly_at_the_warmup_boundary():
    """A linear ramp, the target rate on the first post-warmup epoch, then cosine.

    ``LinearLR`` interpolates the *factor* from ``start_factor`` to 1 over
    ``total_iters`` steps, so epoch ``i`` (0-based) sits at
    ``start + (1 - start) * i / warmup`` of the target and the peak lands on the
    first post-warmup epoch. That is the intended reading of "N epochs of
    warmup": N epochs are spent warming up, and the (N+1)-th is the first at
    full rate.
    """
    cfg = training_cfg()
    curve = lr_curve(cfg, peak=STAGE1_LEARNING_RATE)
    warmup = STAGE1_WARMUP_EPOCHS
    start_factor = 1.0 / warmup

    assert curve[0] == pytest.approx(STAGE1_LEARNING_RATE * start_factor)
    for epoch in range(warmup):
        factor = start_factor + (1.0 - start_factor) * epoch / warmup
        assert curve[epoch] == pytest.approx(STAGE1_LEARNING_RATE * factor, rel=1e-6)
    # Strictly increasing through the warmup, then exactly the target.
    ramp = curve[:warmup]
    assert all(later > earlier for earlier, later in zip(ramp, ramp[1:]))
    assert curve[warmup] == pytest.approx(STAGE1_LEARNING_RATE)
    assert max(curve) == pytest.approx(STAGE1_LEARNING_RATE)
    # Monotone decay afterwards, landing on eta_min one step after the last
    # epoch. The tolerance is the cosine's own resolution, not a magic constant:
    # the last epoch sits at step N-1 of an N-step cosine, i.e. at
    # `peak * (1 - cos(pi/N)) / 2`, which is ~1.2e-3 of the peak at N = 45 and
    # would be ~4x smaller at N = 90. Hard-coding a fraction of the peak makes
    # the test a function of the epoch budget rather than of the schedule.
    tail = curve[warmup:]
    assert all(later <= earlier for earlier, later in zip(tail, tail[1:]))
    cosine_steps = len(tail)
    floor = STAGE1_LEARNING_RATE * (1.0 - math.cos(math.pi / cosine_steps)) / 2.0
    assert tail[-1] == pytest.approx(floor, rel=1e-6)


def test_the_cosine_spans_the_post_warmup_run_not_the_whole_run():
    """Otherwise the warmup silently truncates the decay and it never reaches 0."""
    cfg = training_cfg()
    scheduler = build_scheduler(optim.SGD([nn.Parameter(torch.zeros(1))], lr=1.0), cfg)
    cosine = scheduler._schedulers[1]
    assert cosine.T_max == STAGE1_EPOCHS - STAGE1_WARMUP_EPOCHS


def test_one_scheduler_object_owns_both_phases():
    """A second scheduler on the same optimizer would compose the two factors."""
    scheduler = build_scheduler(optim.SGD([nn.Parameter(torch.zeros(1))], lr=1.0), training_cfg())
    assert isinstance(scheduler, optim.lr_scheduler.SequentialLR)
    assert len(scheduler._schedulers) == 2


def test_the_schedule_round_trips_through_state_dict_mid_warmup():
    """A resume inside warmup must resume inside warmup, milestone included.

    Both halves are restored, because both are needed and the trainer restores
    both (``resume_components`` lists the optimizer and the scheduler side by
    side). ``LinearLR.get_lr`` is *recursive* -- it scales the rate the optimizer
    currently holds -- so a scheduler restored next to a fresh optimizer resumes
    the schedule's position while multiplying the wrong base, and produces a
    plausible ramp at a tenth of the intended rate for the rest of the warmup.
    Restoring the scheduler alone is the failure mode this pins against.
    """
    cfg = training_cfg()

    def fresh():
        module = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        optimizer = optim.AdamW(
            build_param_groups(module, weight_decay=0.04), lr=STAGE1_LEARNING_RATE
        )
        return optimizer, build_scheduler(optimizer, cfg)

    reference_optimizer, reference_scheduler = fresh()
    reference = []
    for _ in range(STAGE1_EPOCHS):
        reference.append(reference_optimizer.param_groups[0]["lr"])
        reference_scheduler.step()

    optimizer, scheduler = fresh()
    for _ in range(5):  # stop mid-warmup, the awkward case
        scheduler.step()
    # A scheduled weight decay is in flight too; it lives on the optimizer.
    apply_weight_decay(optimizer, 0.123)
    optimizer_state, scheduler_state = optimizer.state_dict(), scheduler.state_dict()

    restored_optimizer, restored_scheduler = fresh()
    restored_optimizer.load_state_dict(optimizer_state)
    restored_scheduler.load_state_dict(scheduler_state)

    resumed = []
    for _ in range(5, STAGE1_EPOCHS):
        resumed.append(restored_optimizer.param_groups[0]["lr"])
        restored_scheduler.step()
    assert resumed == pytest.approx(reference[5:])

    # The group split and the in-flight decay survive the optimizer round trip;
    # otherwise a resume would silently start decaying the LayerNorm gains.
    assert restored_optimizer.param_groups[0][WEIGHT_DECAY_FLAG] is True
    assert restored_optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.123)
    assert restored_optimizer.param_groups[1][WEIGHT_DECAY_FLAG] is False
    assert restored_optimizer.param_groups[1]["weight_decay"] == 0.0


def test_warmup_is_clamped_so_a_one_epoch_smoke_run_still_trains():
    """The smoke run composes the same config; an unclamped warmup would own it."""
    assert resolve_warmup_epochs(training_cfg(epochs=1)) == 0
    assert resolve_warmup_epochs(training_cfg(epochs=3)) == 2
    assert resolve_warmup_epochs(training_cfg(epochs=100)) == STAGE1_WARMUP_EPOCHS

    curve = lr_curve(training_cfg(epochs=1), peak=STAGE1_LEARNING_RATE)
    assert curve[0] == pytest.approx(STAGE1_LEARNING_RATE)


# -------------------------------------------------------- parameter groups


def test_biases_and_one_dimensional_parameters_are_excluded_from_weight_decay():
    module = nn.Sequential(
        nn.Linear(8, 8),                       # weight (2-D) + bias (1-D)
        nn.LayerNorm(8),                       # weight + bias, both 1-D
        nn.Conv2d(3, 4, 3, bias=False),        # weight only, 4-D
    )
    decayed, plain = build_param_groups(module, weight_decay=0.04)

    assert decayed[WEIGHT_DECAY_FLAG] is True
    assert plain[WEIGHT_DECAY_FLAG] is False
    assert decayed["weight_decay"] == pytest.approx(0.04)
    assert plain["weight_decay"] == 0.0
    assert all(parameter.ndim > 1 for parameter in decayed["params"])
    assert all(parameter.ndim <= 1 for parameter in plain["params"])
    assert len(decayed["params"]) == 2      # Linear.weight, Conv2d.weight
    assert len(plain["params"]) == 3        # Linear.bias, LayerNorm.weight/bias

    # Every trainable parameter lands in exactly one group.
    total = len(decayed["params"]) + len(plain["params"])
    assert total == sum(1 for p in module.parameters() if p.requires_grad)


def test_swinv2_scalar_tables_land_in_the_no_decay_group():
    """logit_scale and the relative-position tables are 1-D and must not decay."""
    model = DINO(
        backbone_name=REVISED_BACKBONE,
        input_dim=REVISED_BACKBONE_FEATURE_DIM,
        hidden_dim=32,
        bottleneck_dim=16,
        out_dim=64,
    )
    decayed, plain = build_param_groups(
        model.student_backbone, model.student_head, weight_decay=0.04
    )
    plain_ids = {id(parameter) for parameter in plain["params"]}
    scales = [
        parameter
        for name, parameter in model.student_backbone.named_parameters()
        if name.endswith("logit_scale")
    ]
    assert scales, "SwinV2 should expose per-head logit_scale parameters"
    # [heads, 1, 1]: 3-D, so DINO's literal `len(shape) == 1` rule would decay
    # the learned attention temperature toward exp(0) = 1.
    assert all(parameter.ndim == 3 for parameter in scales)
    assert all(id(parameter) in plain_ids for parameter in scales)

    # timm names the continuous position-bias MLPs in `no_weight_decay()`, and
    # those are ordinary 2-D Linear weights that no shape rule would catch.
    cpb = [
        parameter
        for name, parameter in model.student_backbone.named_parameters()
        if "cpb_mlp" in name and name.endswith(".weight")
    ]
    assert cpb, "SwinV2 should expose cpb_mlp weights"
    assert all(id(parameter) in plain_ids for parameter in cpb)

    assert decayed["params"], "the trunk's weight matrices must still decay"
    # ... and the ordinary attention/MLP projections are still in it.
    qkv = [
        parameter
        for name, parameter in model.student_backbone.named_parameters()
        if name.endswith("attn.qkv.weight")
    ]
    decayed_ids = {id(parameter) for parameter in decayed["params"]}
    assert qkv and all(id(parameter) in decayed_ids for parameter in qkv)


def test_frozen_and_duplicated_parameters_are_handled_once():
    module = nn.Linear(4, 4)
    module.bias.requires_grad = False
    decayed, plain = build_param_groups(module, module, weight_decay=0.1)
    assert len(decayed["params"]) == 1     # not two, despite being passed twice
    assert plain["params"] == []           # the frozen bias is skipped


def test_the_weight_decay_ramp_moves_only_the_decayed_group():
    """The bug this replaces: the ramp walked every group and undid the split."""
    module = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    groups = build_param_groups(module, weight_decay=0.04)
    optimizer = optim.AdamW(groups, lr=1e-4)

    for epoch in range(1, 101):
        apply_weight_decay(optimizer, cosine_value(0.04, 0.4, epoch, 100))

    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.4)
    assert optimizer.param_groups[1]["weight_decay"] == 0.0


def test_build_optimizer_preserves_the_group_split():
    module = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    optimizer = build_optimizer(
        build_param_groups(module, weight_decay=0.04),
        training_cfg(),
        torch.device("cpu"),
        LOGGER,
        learning_rate=STAGE1_LEARNING_RATE,
    )
    assert isinstance(optimizer, optim.AdamW)
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["lr"] == pytest.approx(STAGE1_LEARNING_RATE)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.04)
    assert optimizer.param_groups[1]["weight_decay"] == 0.0


# ---------------------------------------------------------------- budget


def test_budget_separates_measured_parameters_from_estimated_flops():
    budget = StageOneBudget.from_model(
        {
            "backbone": 27_580_000,
            "dino_head": 2_000_000,
            "student_total": 29_580_000,
            "student_trainable": 29_580_000,
            "teacher_total": 29_580_000,
            "prototype_layer": 524_288,
        },
        gflops_per_view=REVISED_BACKBONE_GFLOPS_PER_VIEW,
        views_per_image=6,
        global_views_per_image=2,
        epochs=STAGE1_EPOCHS,
        effective_batch_size=STAGE1_EFFECTIVE_BATCH,
        steps_per_epoch=292,
        prototypes=REVISED_DINO_OUT_DIM,
    )

    # B images x (6 student + 2 teacher) views.
    assert budget.views_per_iteration == STAGE1_EFFECTIVE_BATCH * 8

    # Student forward+backward, teacher forward only.
    expected = (
        REVISED_BACKBONE_GFLOPS_PER_VIEW
        * STAGE1_EFFECTIVE_BATCH
        * (6 * (1 + BACKWARD_MULTIPLIER) + 2)
    )
    assert budget.estimated_gflops_per_iteration == pytest.approx(expected)
    assert budget.estimated_total_flops == pytest.approx(expected * 1e9 * 292 * STAGE1_EPOCHS)

    table = budget.format_table()
    assert "[measured, fwd @ 256 px]" in table
    assert "[ESTIMATED]" in table
    # Runtime is absent until it is measured, and says so rather than showing 0.
    assert "in progress" in table
    assert budget.peak_allocated_gb is None


def test_budget_degrades_without_a_flop_count():
    budget = StageOneBudget.from_model({"backbone": 1}, gflops_per_view=None, steps_per_epoch=10)
    assert budget.estimated_gflops_per_iteration is None
    assert budget.estimated_total_flops is None
    assert "not available" in budget.format_table()
    assert "budget/gflops_per_view" not in budget.as_metrics()


def test_budget_metrics_name_estimates_as_estimates():
    budget = StageOneBudget.from_model(
        {"backbone": 1}, gflops_per_view=1.0, steps_per_epoch=10, epochs=2
    )
    metrics = budget.as_metrics()
    assert "budget/gflops_per_view" in metrics
    assert "budget/estimated_gflops_per_iteration" in metrics
    assert "budget/estimated_total_exaflops" in metrics
    assert all(isinstance(value, float) for value in metrics.values())


def test_budget_records_runtime_peaks_it_is_given():
    """The trainer tracks run-wide peaks because the loop resets CUDA's counters."""
    budget = StageOneBudget.from_model({"backbone": 1}, views_per_image=6)
    budget.record_runtime(
        torch.device("cpu"),
        training_seconds=3600.0,
        images_processed=36_000,
        peak_allocated_gb=7.5,
        peak_reserved_gb=9.0,
    )
    assert budget.images_per_second == pytest.approx(10.0)
    assert budget.views_per_second == pytest.approx(80.0)
    # CPU: no VRAM to report, and the report says so instead of inventing a zero.
    assert budget.peak_allocated_gb is None
    assert any("not tracked" in note for note in budget.notes)


# ------------------------------------------------- stage-2 handoff (contract)


def test_the_published_backbone_loads_into_the_stage_two_encoder():
    """The whole point of stage 1: `student_backbone` must be readable by stage 2.

    Checked against ``DinoV2SwinV2Encoder`` rather than against a bare timm model,
    because that is what stage 2 actually constructs -- and it loads with
    ``strict=False``, so a key-set mismatch would otherwise surface as one log
    line and a randomly-initialised trunk.
    """
    from src.models.builder import DinoV2SwinV2Encoder

    model = DINO(
        backbone_name=REVISED_BACKBONE,
        input_dim=REVISED_BACKBONE_FEATURE_DIM,
        hidden_dim=32,
        bottleneck_dim=16,
        out_dim=64,
        drop_path_rate=REVISED_DROP_PATH_RATE,
    )
    published = model.student_backbone.state_dict()

    encoder = DinoV2SwinV2Encoder(
        model_name=REVISED_BACKBONE,
        embed_dim=PAPER_EMBED_DIM,
        pretrained=False,
        freeze_backbone=True,
        token_mode="grid",
    )
    report = encoder.encoder.backbone.load_state_dict(published, strict=False)

    assert not report.missing_keys, f"stage 2 would silently randomise {report.missing_keys[:3]}"
    assert not report.unexpected_keys
    assert encoder.backbone_dim == REVISED_BACKBONE_FEATURE_DIM

    with torch.no_grad():
        z = encoder(torch.zeros(2, 3, 256, 256))
    # z in R^384 on an 8x8 grid, whatever the trunk's native width.
    assert z.shape == (2, PAPER_TOKEN_GRID, PAPER_EMBED_DIM)


def test_drop_path_does_not_change_the_published_key_set():
    """Stochastic depth must not leak into the checkpoint's structure."""
    common = {
        "backbone_name": REVISED_BACKBONE,
        "input_dim": REVISED_BACKBONE_FEATURE_DIM,
        "hidden_dim": 32,
        "bottleneck_dim": 16,
        "out_dim": 64,
    }
    plain = DINO(**common, drop_path_rate=0.0).student_backbone.state_dict()
    stochastic = DINO(**common, drop_path_rate=0.3).student_backbone.state_dict()
    assert set(plain) == set(stochastic)
